import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple

from models.utils_conv import get_gconv_layer
from models.utils_activation import get_activation

EPS = 1e-15
MAX_LOGSTD = 10


class GNNEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_layers: int,
            latent_dim: int,
            conv_layer_type: str,
            activation_function: str,
            variational: bool = False
    ):
        super().__init__()
        self.variational = variational
        self.act = get_activation(activation_name=activation_function)
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        hidden_dim = input_dim
        for l in range(num_layers):
            if l != num_layers - 1:
                # Intermediate layers
                conv = get_gconv_layer(
                    in_channels=hidden_dim,
                    out_channels=latent_dim,
                    conv_layer_type=conv_layer_type
                )
                self.conv_layers.append(conv)
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
                hidden_dim = latent_dim
            else:
                # Final layer
                if variational:
                    # Output both mean and logstd: 2 * latent_dim
                    conv = get_gconv_layer(
                        in_channels=hidden_dim,
                        out_channels=2 * latent_dim,
                        conv_layer_type=conv_layer_type
                    )
                else:
                    # Output only Z: latent_dim
                    conv = get_gconv_layer(
                        in_channels=hidden_dim,
                        out_channels=latent_dim,
                        conv_layer_type=conv_layer_type
                    )
                self.conv_layers.append(conv)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = x
        for i, conv in enumerate(self.conv_layers[:-1]):
            # Pre-LN: normalize → activate → transform → residual
            h_norm = self.layer_norms[i](h)
            h = h + self.act(conv(h_norm, edge_index))
        # Final layer (no residual, no activation)
        out = self.conv_layers[-1](h, edge_index)
        if self.variational:
            # Split output into mu and logstd
            mu, logstd = out.chunk(2, dim=-1)
            logstd = logstd.clamp(max=MAX_LOGSTD)
            return mu, logstd
        else:
            return out, None


class GNNDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            num_layers: int,
            output_dim: int,
            conv_layer_type: str,
            activation_function: str,
    ):
        super().__init__()
        self.act = get_activation(activation_name=activation_function)
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        hidden_dim = latent_dim
        for l in range(num_layers):
            if l != num_layers - 1:
                # Intermediate layers
                conv = get_gconv_layer(
                    in_channels=hidden_dim,
                    out_channels=latent_dim,
                    conv_layer_type=conv_layer_type
                )
                self.conv_layers.append(conv)
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            else:
                # Final layer
                conv = get_gconv_layer(
                    in_channels=hidden_dim,
                    out_channels=output_dim,
                    conv_layer_type=conv_layer_type
                )
                self.conv_layers.append(conv)

    def forward(
            self,
            z: torch.Tensor,
            edge_index: torch.Tensor
    ) -> torch.Tensor:
        h = z
        for i, conv in enumerate(self.conv_layers[:-1]):
            h_norm = self.layer_norms[i](h)
            h = h + self.act(conv(h_norm, edge_index))

        # Final layer
        x_hat = self.conv_layers[-1](h, edge_index)
        return x_hat


class SparsityMask(nn.Module):
    """
    Applies exact top-k sparsity to latent representations.

    Three modes:
    1. 'ste_fixed': Straight-Through Estimator with fixed temperature
    2. 'ste_annealing': STE with temperature annealing
    3. 'gumbel': Gumbel Top-K selection (differentiable relaxation)
    """

    def __init__(
            self,
            sparsity_ratio: float,
            mode: str = 'ste_fixed',
            temperature_init: float = 0.1,
            min_temperature: float = 0.01,
            annealing_steps: int = 1000
    ):
        """
        Args:
            sparsity_ratio: Fraction of entries to keep (e.g., 0.05 for 5%)
            mode: 'ste_fixed', 'ste_annealing', or 'gumbel'
            temperature_init: Initial temperature
            min_temperature: Minimum temperature for annealing
            annealing_steps: Number of steps to anneal temperature
        """
        super().__init__()
        assert 0 < sparsity_ratio < 1, "sparsity_ratio must be in (0, 1)"
        assert mode in ['ste_fixed', 'ste_annealing', 'gumbel'], \
            f"Invalid mode: {mode}. Must be 'ste_fixed', 'ste_annealing', or 'gumbel'"

        self.sparsity_ratio = sparsity_ratio
        self.mode = mode
        self.min_temperature = min_temperature
        self.annealing_steps = annealing_steps

        # Register temperature as buffer (not a parameter, but saved with model)
        self.register_buffer('temperature', torch.tensor(temperature_init))
        self.register_buffer('step_count', torch.tensor(0))

    def update_temperature(self):
        """Update temperature for annealing modes"""
        if self.mode in ['ste_annealing', 'gumbel'] and self.training:
            with torch.no_grad():
                # Exponential decay: T(t) = max(T_min, T_init * exp(-t / tau))
                tau = self.annealing_steps / 5.0  # decay constant
                initial_temp = 1.0 if self.mode == 'gumbel' else 0.1
                new_temp = max(
                    self.min_temperature,
                    initial_temp * math.exp(-self.step_count.item() / tau)
                )
                self.temperature.fill_(new_temp)
                self.step_count += 1

    def forward(
            self,
            z: Tensor,
            batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply exact top-k sparsity mask to latent z.

        Args:
            z: Latent representation [num_nodes, latent_dim]
            batch: Batch assignment for each node [num_nodes] (optional)

        Returns:
            z_sparse: Sparsified latent [num_nodes, latent_dim]
            mask_hard: Hard binary mask [num_nodes, latent_dim]
        """
        if batch is None:
            # Single graph case
            return self._apply_mask_single_graph(z)
        else:
            # Batch of graphs case
            return self._apply_mask_batched(z, batch)

    def _apply_mask_single_graph(
            self,
            z: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply mask to a single graph"""
        if self.mode in ['ste_fixed', 'ste_annealing']:
            return self._ste_mask(z)
        elif self.mode == 'gumbel':
            return self._gumbel_topk_mask(z)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _ste_mask(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Straight-Through Estimator masking"""
        # Flatten z to compute top-k across all entries
        z_flat = z.reshape(-1)  # [n_i * D]
        total_entries = z_flat.numel()

        # Compute k_i = floor(s * n_i * D), ensure at least 1
        k = max(1, int(self.sparsity_ratio * total_entries))

        # Compute magnitude and find top-k indices
        z_abs = torch.abs(z_flat)
        topk_values, topk_indices = torch.topk(z_abs, k, largest=True, sorted=False)

        # Hard mask: 1 for top-k, 0 elsewhere
        mask_hard = torch.zeros_like(z_flat)
        mask_hard[topk_indices] = 1.0
        mask_hard = mask_hard.reshape_as(z)

        # Soft mask for gradients: sigmoid((|z| - tau) / T)
        # tau is the k-th largest magnitude
        if k < total_entries:
            tau = topk_values[-1]  # k-th largest value
        else:
            tau = 0.0

        mask_soft = torch.sigmoid((z_abs - tau) / self.temperature)
        mask_soft = mask_soft.reshape_as(z)

        # Straight-through estimator:
        # Forward: use hard mask
        # Backward: use soft mask gradient
        mask_ste = mask_hard.detach() - mask_soft.detach() + mask_soft

        # Apply mask
        z_sparse = z * mask_ste

        return z_sparse, mask_hard

    def _gumbel_topk_mask(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Gumbel Top-K selection for differentiable masking.

        Note: During training, sparsity is APPROXIMATE (soft relaxation).
        During evaluation, sparsity is EXACT (hard selection).
        """
        original_shape = z.shape
        z_flat = z.reshape(-1)
        total_entries = z_flat.numel()
        k = max(1, int(self.sparsity_ratio * total_entries))

        logits = torch.abs(z_flat)

        if self.training:
            # === TRAINING: Soft differentiable relaxation ===

            # Add Gumbel noise for exploration
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + EPS) + EPS)
            perturbed_logits = logits + gumbel_noise

            # Find threshold (k-th largest perturbed value)
            kth_value, _ = torch.kthvalue(perturbed_logits, total_entries - k + 1)

            # Soft mask via sigmoid (differentiable)
            # As temperature → 0, this approaches hard selection
            mask_soft = torch.sigmoid((perturbed_logits - kth_value) / self.temperature)

            # For statistics: report effective sparsity of soft mask
            # Threshold at 0.5 to count "active" entries
            mask_hard = (mask_soft > 0.5).float()

            # Use soft mask (approximate sparsity, but fully differentiable)
            mask = mask_soft

        else:
            # === EVALUATION: Hard exact top-k ===
            kth_value, _ = torch.kthvalue(logits, total_entries - k + 1)
            mask_hard = (logits >= kth_value).float()
            mask = mask_hard

        mask = mask.reshape(original_shape)
        mask_hard = mask_hard.reshape(original_shape)
        z_sparse = z * mask

        return z_sparse, mask_hard

    def _apply_mask_batched(
            self,
            z: Tensor,
            batch: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply mask to a batch of graphs (per-graph sparsity)"""
        num_graphs = batch.max().item() + 1
        z_sparse_list = []
        mask_hard_list = []

        for graph_id in range(num_graphs):
            # Extract nodes belonging to this graph
            node_mask = (batch == graph_id)
            z_graph = z[node_mask]  # [n_i, D]

            # Apply sparsity to this graph
            z_graph_sparse, mask_graph_hard = self._apply_mask_single_graph(z_graph)

            z_sparse_list.append(z_graph_sparse)
            mask_hard_list.append(mask_graph_hard)

        # Concatenate back
        z_sparse = torch.cat(z_sparse_list, dim=0)
        mask_hard = torch.cat(mask_hard_list, dim=0)

        return z_sparse, mask_hard


class GAE(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            output_dim: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            conv_layer_type: str,
            activation_function: str,
            sparsity_ratio: float,
            variational: bool = False,
            sparsity_mode: str = 'ste_fixed',
            temperature_init: float = 0.1,
            min_temperature: float = 0.01,
            annealing_steps: int = 1000
    ):
        super().__init__()
        self.variational = variational

        # Initialize encoder
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            num_layers=num_encoder_layers,
            latent_dim=latent_dim,
            conv_layer_type=conv_layer_type,
            activation_function=activation_function,
            variational=variational
        )

        # Initialize decoder
        self.decoder = GNNDecoder(
            latent_dim=latent_dim,
            num_layers=num_decoder_layers,
            output_dim=output_dim,
            conv_layer_type=conv_layer_type,
            activation_function=activation_function
        )

        # Initialize sparsity mask
        self.sparsity_mask = SparsityMask(
            sparsity_ratio=sparsity_ratio,
            mode=sparsity_mode,
            temperature_init=temperature_init,
            min_temperature=min_temperature,
            annealing_steps=annealing_steps
        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Optional[Tensor] = None
    ) -> Tensor:
        return self.encode(x, edge_index, batch)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Optional[Tensor] = None
    ) -> Tensor:
        if self.variational:
            # Variational mode: encoder outputs mu and logstd
            self.__mu__, self.__logstd__ = self.encoder(x, edge_index)
            self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)

            # Sample z from q(z | x, G)
            z = self.reparametrize(self.__mu__, self.__logstd__)
        else:
            # Standard mode: encoder outputs z directly
            z, _ = self.encoder(x, edge_index)  # [num_nodes, latent_dim]

        # Apply exact sparsity
        z_sparse, self.__mask_hard__ = self.sparsity_mask(z, batch)

        return z_sparse

    def decode(
            self,
            z: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        return self.decoder(z, edge_index)

    @staticmethod
    def recon_loss(
            x: Tensor,
            x_hat: Tensor,
            batch: Optional[Tensor] = None
    ) -> Tensor:
        # Mean squared error per node
        node_mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)  # [num_nodes]

        if batch is None:
            # Single graph: average over nodes
            return node_mse.mean()
        else:
            # Multiple graphs: average per graph, then over batch
            num_graphs = batch.max().item() + 1
            graph_losses = []
            for graph_id in range(num_graphs):
                node_mask = (batch == graph_id)
                graph_loss = node_mse[node_mask].mean()
                graph_losses.append(graph_loss)
            return torch.stack(graph_losses).mean()

    def kl_loss(
            self,
            mu: Optional[Tensor] = None,
            logstd: Optional[Tensor] = None,
            batch: Optional[Tensor] = None
    ) -> Tensor:
        if not self.variational:
            raise ValueError("kl_loss is only available in variational mode")

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)

        # KL(q(z|x,G) || p(z)) where p(z) = N(0, I)
        # KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        kl_per_element = -0.5 * (1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2))

        if batch is None:
            # Single graph: sum over all elements, average over batch
            return kl_per_element.sum(dim=-1).mean()
        else:
            # Multiple graphs: sum per graph, then average over batch
            num_graphs = batch.max().item() + 1
            graph_kl = []
            for graph_id in range(num_graphs):
                node_mask = (batch == graph_id)
                graph_kl.append(kl_per_element[node_mask].sum())
            return torch.stack(graph_kl).mean()

    def elbo_loss(
            self,
            x: Tensor,
            x_hat: Tensor,
            batch: Optional[Tensor] = None,
            beta: float = 1.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not self.variational:
            raise ValueError("elbo_loss is only available in variational mode")

        recon_loss = self.recon_loss(x, x_hat, batch)
        kl_loss = self.kl_loss(batch=batch)

        # Negative ELBO
        loss = recon_loss + beta * kl_loss

        return loss, recon_loss, kl_loss

    def get_sparsity_stats(self) -> dict:
        if not hasattr(self, '__mask_hard__'):
            return {
                'sparsity_ratio_actual': 0.0,
                'num_nonzero': 0,
                'num_total': 0,
                'temperature': self.sparsity_mask.temperature.item(),
                'mode': self.sparsity_mask.mode,
                'variational': self.variational
            }

        mask = self.__mask_hard__
        total = mask.numel()
        nonzero = mask.sum().item()

        return {
            'sparsity_ratio_actual': nonzero / total if total > 0 else 0.0,
            'num_nonzero': int(nonzero),
            'num_total': total,
            'temperature': self.sparsity_mask.temperature.item(),
            'mode': self.sparsity_mask.mode,
            'variational': self.variational
        }


def create_sample_batch(num_graphs=8, num_nodes_per_graph=100, feature_dim=1):
    """Create a sample batch of graphs"""
    data_list = []
    for _ in range(num_graphs):
        # Random graph signal
        x = torch.randn(num_nodes_per_graph, feature_dim)
        # Random edges (small-world-like)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_nodes_per_graph * 3))
        data_list.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(data_list)


def main():
    # ============================================================
    # Strategy 1: STE with Fixed Temperature (Standard GAE)
    # ============================================================
    print("=" * 60)
    print("Strategy 1: Straight-Through Estimator (Fixed Temperature)")
    print("=" * 60)

    model_ste_fixed = GAE(
        input_dim=1,
        latent_dim=64,
        output_dim=1,
        num_encoder_layers=3,
        num_decoder_layers=3,
        conv_layer_type='SAGEConv',
        activation_function='relu',
        sparsity_ratio=0.8,
        variational=False,
        sparsity_mode='ste_fixed',
        temperature_init=0.1
    )

    # Training
    optimizer_1 = torch.optim.Adam(model_ste_fixed.parameters(), lr=0.001)
    batch = create_sample_batch()

    # for epoch in range(10_000):
    #     optimizer_1.zero_grad()
    #     z_sparse = model_ste_fixed.encode(batch.x, batch.edge_index, batch.batch)
    #     x_hat = model_ste_fixed.decode(z_sparse, batch.edge_index)
    #     loss = model_ste_fixed.recon_loss(batch.x, x_hat, batch.batch)
    #     loss.backward()
    #     optimizer_1.step()
    #
    #     stats = model_ste_fixed.get_sparsity_stats()
    #     print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
    #           f"Sparsity={stats['sparsity_ratio_actual']:.4f}, "
    #           f"Temp={stats['temperature']:.4f}")

    # ============================================================
    # Strategy 2: STE with Temperature Annealing (Standard GAE)
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 2: STE with Temperature Annealing")
    print("=" * 60)

    model_ste_annealing = GAE(
        input_dim=1,
        latent_dim=64,
        output_dim=1,
        num_encoder_layers=3,
        num_decoder_layers=3,
        conv_layer_type='SAGEConv',
        activation_function='relu',
        sparsity_ratio=0.05,
        variational=False,
        sparsity_mode='ste_annealing',
        temperature_init=1.0,
        min_temperature=0.01,
        annealing_steps=1000
    )

    optimizer_2 = torch.optim.Adam(model_ste_annealing.parameters(), lr=0.001)

    # for epoch in range(10_000):
    #     optimizer_2.zero_grad()
    #     z_sparse = model_ste_annealing.encode(batch.x, batch.edge_index, batch.batch)
    #     x_hat = model_ste_annealing.decode(z_sparse, batch.edge_index)
    #     loss = model_ste_annealing.recon_loss(batch.x, x_hat, batch.batch)
    #     loss.backward()
    #     optimizer_2.step()
    #
    #     if model_ste_annealing.training:
    #         model_ste_annealing.sparsity_mask.update_temperature()
    #
    #     stats = model_ste_annealing.get_sparsity_stats()
    #     print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
    #           f"Sparsity={stats['sparsity_ratio_actual']:.4f}, "
    #           f"Temp={stats['temperature']:.4f} (annealing)")

    # ============================================================
    # Strategy 3: Gumbel Top-K Selection (Standard GAE)
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 3: Gumbel Top-K Selection")
    print("=" * 60)

    model_gumbel = GAE(
        input_dim=1,
        latent_dim=64,
        output_dim=1,
        num_encoder_layers=3,
        num_decoder_layers=3,
        conv_layer_type='SAGEConv',
        activation_function='relu',
        sparsity_ratio=0.05,
        variational=False,
        sparsity_mode='gumbel',
        temperature_init=1.0,
        min_temperature=0.05,
        annealing_steps=1000
    )

    optimizer_3 = torch.optim.Adam(model_gumbel.parameters(), lr=0.001)

    # for epoch in range(10_000):
    #     optimizer_3.zero_grad()
    #     z_sparse = model_gumbel.encode(batch.x, batch.edge_index, batch.batch)
    #     x_hat = model_gumbel.decode(z_sparse, batch.edge_index)
    #     loss = model_gumbel.recon_loss(batch.x, x_hat, batch.batch)
    #     loss.backward()
    #     optimizer_3.step()
    #
    #     if model_gumbel.training:
    #         model_gumbel.sparsity_mask.update_temperature()
    #
    #     stats = model_gumbel.get_sparsity_stats()
    #     print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
    #           f"Sparsity={stats['sparsity_ratio_actual']:.4f}, "
    #           f"Temp={stats['temperature']:.4f} (Gumbel)")

    # ============================================================
    # VGAE Example with Gumbel
    # ============================================================
    print("\n" + "=" * 60)
    print("VGAE with Gumbel Top-K")
    print("=" * 60)

    model_vgae = GAE(
        input_dim=1,
        latent_dim=64,
        output_dim=1,
        num_encoder_layers=3,
        num_decoder_layers=2,
        conv_layer_type='GCNConv',
        activation_function='relu',
        sparsity_ratio=0.05,
        variational=True,  # Enable variational mode
        sparsity_mode='gumbel',
        temperature_init=1.0,
        min_temperature=0.05,
        annealing_steps=100
    )

    optimizer_vae = torch.optim.Adam(model_vgae.parameters(), lr=0.001)

    for epoch in range(10_000):
        optimizer_vae.zero_grad()
        z_sparse = model_vgae.encode(batch.x, batch.edge_index, batch.batch)
        x_hat = model_vgae.decode(z_sparse, batch.edge_index)

        # VGAE uses ELBO loss
        loss, recon, kl = model_vgae.elbo_loss(batch.x, x_hat, batch.batch, beta=1.0)

        loss.backward()
        optimizer_vae.step()
        if model_vgae.training:
            model_vgae.sparsity_mask.update_temperature()

        stats = model_vgae.get_sparsity_stats()
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
              f"Recon={recon.item():.4f}, KL={kl.item():.4f}, "
              f"Sparsity={stats['sparsity_ratio_actual']:.4f}")

    # ============================================================
    # Comparison Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("""
    1. STE Fixed Temperature:
       - Simplest approach
       - Constant temperature throughout training
       - Good baseline, works well in practice
       - May have sharper gradients initially

    2. STE with Annealing:
       - Starts with softer mask (higher temp)
       - Gradually hardens (lower temp)
       - Smoother training, better convergence
       - Recommended for complex tasks

    3. Gumbel Top-K:
       - Most principled differentiable approach
       - Adds exploration via Gumbel noise
       - Can escape local minima better
       - Slightly more expensive computationally
       - Best for tasks requiring diverse solutions

    Recommendations:
    - Start with STE fixed (fastest to tune)
    - Try annealing if training is unstable
    - Use Gumbel for best results (at cost of compute)
    """)


if __name__ == '__main__':
    main()
