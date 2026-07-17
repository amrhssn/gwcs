from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from graph_wavelet_transform.energy import energy_distribution_across_scales
from models.loss_functions import compute_loss


class SingleSampleDirectOptimization(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_filters: int,
            sparsity_ratio: float,
            model: nn.Module,
            coeffs: Optional[torch.Tensor] = None,
            energy_init_scale: float = 1.0,
            min_selected: int = 1,
            epochs: int = 1000,
            temperature_init: float = 5.0,                # default per suggested setup
            temperature_min: float = 0.1,                 # anneal to ~0.3–0.5
            temperature_anneal_factor: float = 0.95,
            temperature_anneal_freq: int = 10,
            use_gumbel: bool = True,                      # enable Gumbel-sigmoid by default
            hard_selection: bool = True,                  # STE + top-K during training
            optimizer_lr: float = 1e-3,
            optimizer_weight_decay: float = 0.0,
            energy_aware_weight_initialization: bool = True,
            device: Optional[torch.device] = None,
            seed: Optional[int] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_filters = num_filters
        self.sparsity_ratio = sparsity_ratio
        self.epochs = epochs

        # configure and freeze external model
        self.model = model
        self.model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.num_coeffs = num_nodes * num_filters
        self.min_selected = int(min_selected)
        self.num_selection = self.compute_num_selected(ratio=sparsity_ratio)

        # Temperature schedule (internal)
        self.temperature_init = float(temperature_init)
        self.temperature_min = float(temperature_min)
        self.temperature_anneal_factor = float(temperature_anneal_factor)
        self.temperature_anneal_freq = int(temperature_anneal_freq)
        self.temperature = float(temperature_init)

        # Selection/training options
        self.use_gumbel = use_gumbel
        self.hard_selection = hard_selection

        # Reproducibility
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            torch.manual_seed(int(seed))

        # Device
        self.device = device if device is not None else next(self.model.parameters(), torch.tensor(0.)).device

        # Optimization parameter: only mask logits are learnable
        # init_logits = torch.randn(self.num_coeffs, device=self.device) * 0.1
        # self.mask_logits = nn.Parameter(init_logits)
        # Optimization parameter: energy-aware initialization
        if energy_aware_weight_initialization:
            if coeffs is None:
                raise ValueError("coeffs must be provided for energy-aware initialization")
            init_logits = self.energy_aware_initialization(coeffs=coeffs, energy_init_scale=energy_init_scale)
        else:
            init_logits = torch.randn(self.num_coeffs, device=self.device) * 0.1
        self.mask_logits = nn.Parameter(init_logits)

        # Optimizer on mask logits only
        self.optimizer = torch.optim.AdamW([self.mask_logits],
                                           lr=optimizer_lr,
                                           weight_decay=optimizer_weight_decay)

        # Learning rate scheduler - ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            threshold=1e-4,
            min_lr=1e-6
        )

        self.train()

    def energy_aware_initialization(
            self,
            coeffs: Optional[torch.Tensor] = None,
            energy_init_scale: float = 1.0,
    ):
        # Ensure coeffs is on correct device
        coeffs = coeffs.to(self.device)

        # Compute energy distribution across scales: shape [num_filters]
        energy_dist = energy_distribution_across_scales(coeffs=coeffs)

        # Convert energy distribution to logit-space bias
        # Add epsilon to avoid log(0), then scale by energy_init_scale
        energy_bias = torch.log(energy_dist + 1e-8) * energy_init_scale

        # Expand to per-coefficient: repeat each scale's bias for all nodes
        # Shape: [num_nodes, num_filters] -> flatten to [num_coeffs]
        energy_bias_per_coeff = energy_bias.unsqueeze(0).repeat(self.num_nodes, 1).flatten()

        # Add small random noise for breaking symmetry within each scale
        random_noise = torch.randn(self.num_coeffs, device=self.device) * 0.1

        # Combine energy-based bias with random noise
        init_logits = energy_bias_per_coeff + random_noise

        return init_logits

    def compute_num_selected(self, ratio: float):
        # clamp ratio and compute k, ensuring 0 <= k <= num_coeffs and k >= min_selected if min_selected>0
        r = max(0.0, min(1.0, float(ratio)))
        k = int(round(r * self.num_coeffs))
        if self.min_selected is not None and self.min_selected > 0:
            k = max(self.min_selected, k)
        k = min(self.num_coeffs, k)
        return int(k)

    def forward(
            self,
            epoch: Optional[int] = None
    ):
        if epoch is not None and self.training:
            self.update_temperature(epoch)

        if self.training:
            # Compute soft scores (optionally with Gumbel noise)
            scores_soft = self.compute_scores(self.mask_logits, self.temperature, self.use_gumbel)

            if self.hard_selection:
                # STE with top-K (discrete forward, soft backward)
                mask_hard = self.topk_mask(scores_soft)
                mask = mask_hard + (scores_soft - scores_soft.detach())
            else:
                # Pure continuous relaxation
                mask = scores_soft
        else:
            # Inference: deterministic top-K on clean probabilities (no Gumbel)
            probs = torch.sigmoid(self.mask_logits)
            mask = self.topk_mask(probs)
        mask = mask.reshape(self.num_nodes, self.num_filters)
        return mask

    @staticmethod
    def generate_gumbel_noise_like(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        u = torch.rand_like(t)
        return -torch.log(-torch.log(u + eps) + eps)

    def compute_scores(
            self,
            logits: torch.Tensor,
            temperature: float,
            use_gumbel: bool
    ) -> torch.Tensor:
        if use_gumbel:
            g = self.generate_gumbel_noise_like(logits)
            return torch.sigmoid((logits + g) / (temperature + 1e-12))
        else:
            return torch.sigmoid(logits / (temperature + 1e-12))

    def update_temperature(self, epoch: int):
        if (epoch + 1) % self.temperature_anneal_freq == 0:
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_anneal_factor)

    def reset_temperature(self):
        self.temperature = float(self.temperature_init)

    def topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        if self.num_selection <= 0:
            return torch.zeros_like(scores)
        k = min(self.num_selection, scores.numel())
        # torch.topk expects 1D; scores is 1D here
        topk_indices = torch.topk(scores, k).indices
        mask = torch.zeros_like(scores)
        mask[topk_indices] = 1.0
        return mask

    def fit(
            self,
            data: Data,
            gamma_coeff_consistency: float = 0.0,
            gamma_regularization: float = 0.0,
            sparsity_penalty_coeff: float = 0.1,
            verbose: bool = True
    ):
        device = self.device
        data = data.to(device)
        for epoch in range(self.epochs):

            self.optimizer.zero_grad()
            mask = self.forward(epoch=epoch)
            x_hat = self.model(coeffs=data.coeffs,
                               sparsification_mask=mask,
                               edge_index=data.edge_index,
                               gwt_matrices=[data.G])
            loss = self.sparsification_loss(x_hat=x_hat,
                                            x=data.x,
                                            coeffs=data.coeffs,
                                            sparsification_mask=mask,
                                            gwt_matrices=[data.G],
                                            L_sym_matrices=[data.L_sym],
                                            gamma_coeff_consistency=gamma_coeff_consistency,
                                            gamma_regularization=gamma_regularization,
                                            sparsity_penalty_coeff=sparsity_penalty_coeff)
            # Backward pass
            loss.backward()
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # Update mask parameters
            self.optimizer.step()

            # Update learning rate based on loss
            self.scheduler.step(loss)

            if verbose:
                with torch.no_grad():
                    probs = torch.sigmoid(self.mask_logits)
                    # sel = self.topk_mask(probs).sum().item()
                    exp_sel = probs.sum().item()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"[Epoch {epoch:4d}] loss={loss.item():.6f} temp={self.temperature:.4f} "
                          f"lr={current_lr:.6f} E[probs]/num_coeffs={int(exp_sel)}/{self.num_coeffs}")

    def sparsification_loss(
            self,
            x_hat: torch.Tensor,
            x: torch.Tensor,
            coeffs: torch.Tensor,
            sparsification_mask: torch.Tensor,
            gwt_matrices: List[torch.Tensor],
            L_sym_matrices: List[torch.Tensor],
            gamma_coeff_consistency: float,
            gamma_regularization: float,
            sparsity_penalty_coeff: float
    ):
        model_loss, recon_loss, coeffs_consistency_loss, regularization_loss = compute_loss(
            x_hat=x_hat,
            x=x,
            coeffs=coeffs,
            sparsification_mask=sparsification_mask,
            gwt_matrices=gwt_matrices,
            L_sym=L_sym_matrices,
            gamma_coeff_consistency=gamma_coeff_consistency,
            gamma_regularization=gamma_regularization,
            batch_ptr=None
        )
        # Sparsity penalty:
        # - If hard_selection=True we already enforce exact K via top-K in forward → skip penalty.
        # - If hard_selection=False we encourage the expected count to approach K (use soft mask sums).
        sparsity_loss = torch.tensor(0.0, device=sparsification_mask.device)
        if not self.hard_selection and sparsity_penalty_coeff > 0.0:
            if self.use_gumbel:
                mask_soft_for_penalty = self.compute_scores(logits=self.mask_logits,
                                                            temperature=self.temperature,
                                                            use_gumbel=False)
            else:
                # Forward already produced a soft mask without Gumbel; reuse it
                mask_soft_for_penalty = sparsification_mask
            mask_sum = mask_soft_for_penalty.sum()
            target_k = torch.tensor(float(self.num_selection), device=sparsification_mask.device)
            sparsity_loss = sparsity_penalty_coeff * torch.abs(mask_sum - target_k)
        total_loss = model_loss + sparsity_loss
        return total_loss

    @torch.no_grad()
    def get_optimal_mask(self):
        self.eval()
        probs = torch.sigmoid(self.mask_logits)
        mask_flat = self.topk_mask(probs)
        mask = mask_flat.reshape(self.num_nodes, self.num_filters)
        return mask

    @torch.no_grad()
    def sample_mask_topk_gumbel(self) -> torch.Tensor:
        # Gumbel-TopK: exact-K, stochastic
        logits = self.mask_logits
        g = self.generate_gumbel_noise_like(logits)
        scores = logits + g
        mask_flat = self.topk_mask(scores.sigmoid())
        return mask_flat.reshape(self.num_nodes, self.num_filters)
