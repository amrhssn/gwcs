import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv


class NeuralSampling(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            max_hops: int = 2
    ):
        """
        Graph Neural Sampling Module based on Mutual Information Estimation.

        Args:
            in_channels (int): Dimension of input graph signals.
            hidden_channels (int): Dimension of hidden embeddings (d).
            max_hops (int): Number of hops (R) for neighborhood encoding.
        """

        super(NeuralSampling, self).__init__()
        self.max_hops = max_hops
        self.hidden_channels = hidden_channels

        # 1. Vertex Encoder (E_w): MLP mapping signal s_v to embedding
        self.vertex_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # 2. Neighbor Encoder (P_w)
        # TAGConv implements exactly: Sum_{r=0}^K (A^r * X * W_r)
        # We apply this on the output of the vertex encoder.
        self.neighbor_encoder = TAGConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=max_hops,
            normalize=False
        )
        # Normalization factor (1/R from paper, where R = max_hops)
        # We use 1/(K+1) since TAGConv sums from r=0 to K
        self.neighbor_norm = 1.0 / (max_hops + 1)

        # 3. Affinity Network (S_w): MLP mapping (v, n) to scalar score
        # Input is concatenation of vertex embedding and neighbor embedding (2 * d)
        self.affinity_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # 1. Encode Vertex Features: H_v
        h_v = self.vertex_encoder(x)

        # 2. Encode Neighborhoods: H_neigh
        # The TAGConv takes H_v and aggregates it over K hops with per-hop weights
        h_neigh = self.neighbor_encoder(h_v, edge_index)

        # Apply normalization by (R+1) as in paper
        h_neigh = h_neigh * self.neighbor_norm

        # 3. Affinity Score: Tw(sv, qNv) = Sw(Ew(sv), Pw(qNv))
        combined = torch.cat([h_v, h_neigh], dim=1)
        logits = self.affinity_net(combined)
        attention = torch.sigmoid(logits)
        return attention, h_v, h_neigh

    def compute_loss(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            num_neg_samples: int = 1
    ) -> torch.Tensor:
        """
        Computes the Mutual Information objective (Eq 7):
        C(M) = (1/|M|) Σ_{v∈M} log(σ(Tw(sv, qNv))) +
               (1/|M|²) Σ_{(v,u)∈M×M} log(1 - σ(Tw(sv, qNu)))

        We train by maximizing C(M), which means minimizing -C(M).

        Args:
            num_neg_samples: Number of negative samples per vertex for approximation
        """

        num_nodes = x.size(0)

        # Compute embeddings
        h_v = self.vertex_encoder(x)
        h_neigh = self.neighbor_encoder(h_v, edge_index) * self.neighbor_norm

        # 1. Positive term: (1/N) Σ log(σ(T(sv, qNv)))
        # Each vertex paired with its own neighborhood
        pos_combined = torch.cat([h_v, h_neigh], dim=1)
        pos_logits = self.affinity_net(pos_combined).squeeze()
        pos_term = torch.mean(F.logsigmoid(pos_logits))  # Average over N vertices

        # 2. Negative term: (1/N²) Σ_{(v,u)} log(1 - σ(T(sv, qNu)))
        # Approximated by negative sampling
        # log(1 - σ(x)) = log(σ(-x))
        neg_sum = 0.0
        for _ in range(num_neg_samples):
            # Sample random neighborhoods for each vertex
            idx = torch.randperm(num_nodes, device=h_neigh.device)
            h_neigh_shuffled = h_neigh[idx]

            neg_combined = torch.cat([h_v, h_neigh_shuffled], dim=1)
            neg_logits = self.affinity_net(neg_combined).squeeze()
            # Divide by N (from outer sum), approximating the inner sum over N with 1 sample
            neg_sum += torch.mean(F.logsigmoid(-neg_logits))

        # Average over number of negative samples, which approximates (1/N²) Σ Σ
        neg_term = neg_sum / num_neg_samples

        # Total: C(M) = pos_term + neg_term
        # Maximize C(M) = Minimize -C(M)
        loss = -(pos_term + neg_term)
        return loss

    def sample(self, x: torch.Tensor, edge_index: torch.Tensor, M: int, use_greedy: bool = False):
        """
        Performs the sampling operation (Inference).

        Args:
            x (Tensor): Graph signal (N, F).
            edge_index (Tensor): Graph connectivity.
            M (int): Number of vertices to sample.
            use_greedy (bool): If True, use greedy submodular optimization (Section III.B).
                              If False, use fast approximation via attention scores.

        Returns:
            sampled_x (Tensor): The signal vector masked by attention (Eq. 10).
            indices (Tensor): Indices of selected nodes.
            attention (Tensor): The computed attention weights.
        """
        # 1. Compute Attention Scores (Equation 9)
        # av = σ(Tw(sv, qNv))
        attention, h_v, h_neigh = self.forward(x, edge_index)

        # 2. Select Top-M Vertices
        num_nodes = x.size(0)
        M_select = min(M, num_nodes)

        if use_greedy:
            # Greedy algorithm based on submodularity (Section III.B, Equation 8)
            top_indices = self._greedy_selection(M_select, h_v, h_neigh)
        else:
            # Fast approximation: Select top-M by attention score
            # This is computationally efficient but not theoretically optimal
            scores = attention.squeeze()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            _, top_indices = torch.topk(scores, k=M_select)

        # 3. Construct Sampled Signal (Equation 10)
        # y = Ψ * diag(a) * x
        sampled_x = torch.zeros_like(x)
        selected_att = attention[top_indices]
        sampled_x[top_indices] = x[top_indices] * selected_att

        return sampled_x, top_indices, attention

    def _greedy_selection(self, M: int, h_v: torch.Tensor, h_neigh: torch.Tensor):
        """
        Greedy algorithm to solve max C(M) subject to |M| = M (Equation 8).
        Based on submodularity of mutual information.

        Returns:
            selected_indices: Tensor of M selected vertex indices
        """
        num_nodes = h_v.size(0)
        selected = []
        remaining = set(range(num_nodes))

        for _ in range(M):
            best_idx = None
            best_score = float('-inf')

            # Evaluate each candidate vertex
            for idx in remaining:
                candidate_set = selected + [idx]
                score = self._evaluate_criterion(candidate_set, h_v, h_neigh)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return torch.tensor(selected, device=h_v.device, dtype=torch.long)

    def _evaluate_criterion(self, vertex_set, h_v, h_neigh):
        """
        Evaluate criterion function C(M) for a given vertex set (Equation 7).

        C(M) = (1/|M|) Σ_{v∈M} log σ(Tw(sv, qNv)) +
               (1/|M|²) Σ_{v,u∈M} log(1 - σ(Tw(sv, qNu)))
        """
        if len(vertex_set) == 0:
            return float('-inf')

        vertex_tensor = torch.tensor(vertex_set, device=h_v.device)
        M_size = len(vertex_set)

        # Positive term: (1/|M|) Σ log σ(Tw(sv, qNv))
        h_v_subset = h_v[vertex_tensor]
        h_neigh_subset = h_neigh[vertex_tensor]
        pos_combined = torch.cat([h_v_subset, h_neigh_subset], dim=1)
        pos_logits = self.affinity_net(pos_combined).squeeze()
        pos_term = torch.sum(F.logsigmoid(pos_logits)) / M_size

        # Negative term: (1/|M|²) Σ_{v,u} log(1 - σ(Tw(sv, qNu)))
        neg_term = 0.0
        for i in vertex_tensor:
            for j in vertex_tensor:
                neg_combined = torch.cat([h_v[i:i + 1], h_neigh[j:j + 1]], dim=1)
                neg_logits = self.affinity_net(neg_combined).squeeze()
                neg_term += F.logsigmoid(-neg_logits).item()
        neg_term = neg_term / (M_size * M_size)

        return pos_term.item() + neg_term


class NeuralRecovery(nn.Module):
    def __init__(self, in_channels: int = 1, K: int = 5, L: int = 3):
        """
        Graph Neural Recovery Module based on Algorithm Unrolling using TAGConv.

        Args:
            in_channels (int): Dimension of signal (usually 1).
            K (int): Number of unrolled layers (iterations).
            L (int): Order of the graph filter (polynomial degree).
        """
        super(NeuralRecovery, self).__init__()
        self.K = K
        self.L = L

        # We create a list of graph filters, one for each unrolled iteration.
        # Each filter learns its own coefficients for powers A^0, A^1, ..., A^L
        self.layers = nn.ModuleList([
            TAGConv(in_channels, in_channels, K=L, bias=False, normalize=True)
            for _ in range(K)
        ])

    def forward(self, y: torch.Tensor, edge_index: torch.Tensor,
                attention: torch.Tensor, mask: torch.Tensor):
        """
        Unrolled iterative recovery algorithm.

        Args:
            y (Tensor): Sampled signal y = Ψ * diag(a) * x (N x F).
            edge_index (Tensor): Graph connectivity.
            attention (Tensor): Attention weights a (N x 1).
            mask (Tensor): Binary mask Ψ indicating sampled nodes (N x F).

        Returns:
            x_hat (Tensor): Recovered graph signal (N x F).
        """
        epsilon = 1e-8

        # Initialization (Algorithm 1):
        # x_M^(0) = y / a_M  (for measured nodes)
        # x_U^(0) = 0        (for unmeasured nodes)
        # FIX: Don't divide by attention if it's used as a weight, not a scaling factor
        # The paper's Eq. 10 shows y = Ψ * diag(a) * x, so to get x back: x = y / a (only at sampled locations)
        x_measured = torch.zeros_like(y)
        # Only compute for masked (sampled) locations
        x_measured[mask > 0] = y[mask > 0] / (attention[mask > 0] + epsilon)

        x_curr = x_measured  # Start with measurements at sampled nodes, 0 elsewhere

        # Unrolled iterations (Equation 16)
        for k, conv in enumerate(self.layers):
            # Step 1: Apply trainable graph filter (Eq 16, line 1)
            # x^(k+1) ← Σ(ℓ=1 to L) h_ℓ^(k) * A^ℓ * x^(k)
            x_next = conv(x_curr, edge_index)

            # Step 2: Data consistency (Eq 16, line 2)
            # x_M^(k) ← y/a_M
            # Force measured nodes back to their observed values
            x_curr = (1 - mask) * x_next + mask * x_measured

        return x_curr


def generate_data():
    import numpy as np
    import networkx as nx
    from scipy.spatial.distance import cdist
    # 1. Create a Random Geometric Graph (Sensor Graph)
    # Common in GSP: N nodes in unit square, connected if dist < radius
    N = 200
    radius = 0.15
    seed = 42

    np.random.seed(seed)
    coords = np.random.rand(N, 2)
    dist = cdist(coords, coords)
    adj = (dist < radius).astype(float)
    np.fill_diagonal(adj, 0)

    # Ensure connectivity (keep largest component)
    G = nx.from_numpy_array(adj)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    # Re-index and update adj
    N = len(G.nodes)
    adj = nx.to_numpy_array(G)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)

    print(f"Graph generated: {N} nodes, {G.number_of_edges()} edges.")

    # 2. Create Bandlimited Graph Signals
    # Compute Laplacian eigenvectors
    L = np.diag(adj.sum(1)) - adj
    evals, evecs = np.linalg.eigh(L)

    # Bandlimited filter: Keep only first K frequencies
    bandwidth = 30
    U_k = evecs[:, :bandwidth]  # N x K

    # Generate L random smooth signals
    # x = U_k * c, where c is random coefficients
    num_signals = 1000
    c = np.random.randn(bandwidth, num_signals)  # K x L
    X = U_k @ c  # N x L

    # Normalize signals
    X = X / np.max(np.abs(X))

    # Convert to PyTorch tensors (Features usually need to be N x F)
    # Here we treat the signals as samples, so we might feed them in batches
    # or treat one signal as one feature channel.
    # For this module, let's assume we train on 1 signal channel at a time or F channels.
    # Let's reshape to (N, num_signals) for easy access.
    x_tensor = torch.tensor(X, dtype=torch.float32)
    return x_tensor, edge_index, coords, G


def visualize_results(loss_history, G, coords, original_signal, sampled_indices, sampled_signal, recovered_signal=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    num_plots = 4 if recovered_signal is not None else 3
    fig = plt.figure(figsize=(6 * num_plots, 6))

    # 1. Training Loss
    ax1 = fig.add_subplot(1, num_plots, 1)
    ax1.plot(loss_history)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # 2. Original Graph Signal
    ax2 = fig.add_subplot(1, num_plots, 2)
    if torch.is_tensor(original_signal):
        val = original_signal.squeeze().cpu().numpy()
    else:
        val = original_signal

    # Keep color scale consistent
    vmin, vmax = val.min(), val.max()

    nx.draw(G, pos=coords, node_color=val, cmap='viridis',
            node_size=50, ax=ax2, edge_color='gray', alpha=0.8, width=0.5, vmin=vmin, vmax=vmax)
    ax2.set_title("Original Graph Signal")

    # 3. Sampled Nodes
    ax3 = fig.add_subplot(1, num_plots, 3)
    nx.draw(G, pos=coords, node_color=val, cmap='viridis',
            node_size=30, ax=ax3, edge_color='lightgray', alpha=0.2, width=0.5, vmin=vmin, vmax=vmax)

    sampled_indices_np = sampled_indices.cpu().numpy()
    nx.draw_networkx_nodes(G, pos=coords, nodelist=sampled_indices_np,
                           node_color='red', node_size=100, ax=ax3, label='Sampled Nodes')
    ax3.set_title(f"Sampled Nodes (M={len(sampled_indices_np)})")
    ax3.legend()

    # 4. Recovered Signal
    if recovered_signal is not None:
        ax4 = fig.add_subplot(1, num_plots, 4)
        if torch.is_tensor(recovered_signal):
            val_rec = recovered_signal.squeeze().cpu().numpy()
        else:
            val_rec = recovered_signal

        nx.draw(G, pos=coords, node_color=val_rec, cmap='viridis',
                node_size=50, ax=ax4, edge_color='gray', alpha=0.8, width=0.5, vmin=vmin, vmax=vmax)
        ax4.set_title("Recovered Graph Signal")

    plt.tight_layout()
    plt.show()


def main():
    import time

    # Hyperparameters
    IN_CHANNELS = 1
    HIDDEN_DIM = 64
    MAX_HOPS = 3

    # Recovery Params
    RECOVERY_LAYERS_K = 5
    RECOVERY_ORDER_L = 3

    SAMPLES_M = 20  # Number of samples
    EPOCHS = 1_000
    LR = 1e-3
    ALPHA = 0.5  # Weight for sampling loss vs recovery loss
    USE_GREEDY_EVAL = False

    PRINT_EVERY = 1

    # Data
    signals, edge_index, coords, G = generate_data()
    N, num_signals = signals.shape
    print(f"Dataset: {N} nodes, {num_signals} signals")

    # Models
    sampling_module = NeuralSampling(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_DIM,
        max_hops=MAX_HOPS
    )

    recovery_module = NeuralRecovery(
        in_channels=IN_CHANNELS,
        K=RECOVERY_LAYERS_K,
        L=RECOVERY_ORDER_L
    )

    # Joint Optimizer
    params = list(sampling_module.parameters()) + list(recovery_module.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    print("\n" + "=" * 60)
    print("Starting Joint Training (Sampling + Recovery)")
    print(f"Loss = Recovery_Loss + {ALPHA} * Sampling_Loss")
    print("=" * 60)

    sampling_module.train()
    recovery_module.train()
    loss_history = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        optimizer.zero_grad()

        # 1. Get Batch
        idx = torch.randint(0, num_signals, (1,)).item()
        x_batch = signals[:, idx].unsqueeze(1)  # (N, 1)

        # 2. Forward Pass: Sampling
        attention, h_v, h_neigh = sampling_module(x_batch, edge_index)

        # Select top-M nodes (non-differentiable but gradients flow through 'a')
        scores = attention.squeeze()
        _, top_indices = torch.topk(scores, k=SAMPLES_M)

        # Create binary mask Psi (N x 1)
        mask_psi = torch.zeros_like(x_batch)
        mask_psi[top_indices] = 1.0

        # Sampled signal: y = Ψ * diag(a) * x (Equation 10)
        y = mask_psi * attention * x_batch

        # 3. Forward Pass: Recovery (with mask)
        x_hat = recovery_module(y, edge_index, attention, mask_psi)

        # 4. Losses (Equation 17)
        # L_recovery = ||x - x_hat||²
        loss_rec = F.mse_loss(x_hat, x_batch)

        # L_sample = -I(V) (Equation 18)
        loss_sample = sampling_module.compute_loss(x_batch, edge_index)

        # Total Loss: L = L_recovery + α * L_sample
        total_loss = loss_rec + ALPHA * loss_sample

        total_loss.backward()

        # FIX: Add gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        loss_history.append(total_loss.item())
        end_time = time.time()

        if (epoch + 1) % PRINT_EVERY == 0:
            # Monitor attention diversity
            att_std = attention.std().item()
            att_mean = attention.mean().item()
            print(f"Epoch {epoch + 1:04d}/{EPOCHS} | Total Loss: {total_loss.item():.4f} "
                  f"(Rec: {loss_rec.item():.4f}, Sample: {loss_sample.item():.4f}) | "
                  f"Att: μ={att_mean:.3f} σ={att_std:.3f} | "
                  f"Time: {end_time - start_time:.3f}s")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Evaluation
    print(f"\nEvaluating on test signal...")
    sampling_module.eval()
    recovery_module.eval()

    test_idx = 0
    x_test = signals[:, test_idx].unsqueeze(1)

    with torch.no_grad():
        # 1. Sample
        attention, h_v, h_neigh = sampling_module(x_test, edge_index)

        if USE_GREEDY_EVAL:
            indices = sampling_module._greedy_selection(SAMPLES_M, h_v, h_neigh)
        else:
            scores = attention.squeeze()
            _, indices = torch.topk(scores, k=SAMPLES_M)

        # Form measurement y and mask
        mask_psi = torch.zeros_like(x_test)
        mask_psi[indices] = 1.0
        y_test = mask_psi * attention * x_test

        # 2. Recover (pass mask)
        x_rec = recovery_module(y_test, edge_index, attention, mask_psi)

        # 3. Metrics
        mse = F.mse_loss(x_rec, x_test).item()
        nmse = (torch.norm(x_rec - x_test) ** 2 / torch.norm(x_test) ** 2).item()

    print(f"\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"MSE: {mse:.6f}")
    print(f"NMSE: {nmse:.6f}")
    print(f"Sampled {len(indices)} nodes out of {N}")

    # Visualization
    print("\nGenerating visualization...")
    visualize_results(loss_history, G, coords, x_test, indices, y_test, x_rec)


if __name__ == "__main__":
    main()

