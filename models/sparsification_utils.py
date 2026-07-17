import numpy as np
import torch


def multilevel_importance_sampling(
        coeffs: torch.Tensor,
        keep_ratio: float
) -> torch.Tensor:
    """
    Multilevel Importance Sampling (MLIS) for graph wavelet coefficients.
    GPU-optimized vectorized version.
    """
    num_nodes, num_filters = coeffs.shape
    total_coeffs = num_nodes * num_filters
    total_to_keep = max(1, int(total_coeffs * keep_ratio))

    # Fast path: if keeping most coefficients, use simpler approach
    if keep_ratio > 0.5:
        # Flatten and sample based on squared magnitudes
        coeffs_flat = coeffs.flatten()
        squared_mags = torch.abs(coeffs_flat) ** 2
        total_energy = squared_mags.sum()

        if total_energy == 0:
            indices = torch.randperm(total_coeffs, device=coeffs.device)[:total_to_keep]
        else:
            probs = squared_mags / total_energy
            indices = torch.multinomial(probs, total_to_keep, replacement=False)

        mask = torch.zeros(total_coeffs, device=coeffs.device, dtype=coeffs.dtype)
        mask[indices] = 1.0
        return mask.view(num_nodes, num_filters)

    # Calculate per-scale L2 energy: E_k = Σ |w_{i,k}|²
    scale_energies = torch.sum(torch.abs(coeffs) ** 2, dim=0)  # Shape: (num_filters,)
    total_energy = scale_energies.sum()

    if total_energy == 0:
        # Edge case: all coefficients are zero, fall back to uniform sampling
        scale_ratios = torch.ones(num_filters, device=coeffs.device) / num_filters
    else:
        scale_ratios = scale_energies / total_energy

    # Allocate samples per scale: m_k = r_k * m
    samples_per_scale_float = scale_ratios * total_to_keep
    samples_per_scale = samples_per_scale_float.long()

    # Distribute remainder based on fractional parts
    remainder = total_to_keep - samples_per_scale.sum().item()
    if remainder > 0:
        fractional_parts = samples_per_scale_float - samples_per_scale.float()
        _, sorted_indices = torch.topk(fractional_parts, min(remainder, num_filters))
        samples_per_scale[sorted_indices] += 1

    # Ensure we don't exceed num_nodes per scale
    samples_per_scale = torch.clamp(samples_per_scale, max=num_nodes)

    # Vectorized mask generation
    mask = torch.zeros_like(coeffs)

    # Pre-compute squared coefficients and probabilities for all scales
    squared_coeffs = torch.abs(coeffs) ** 2  # (num_nodes, num_filters)
    scale_sums = squared_coeffs.sum(dim=0, keepdim=True)  # (1, num_filters)

    # Avoid division by zero
    scale_sums = torch.where(scale_sums == 0, torch.ones_like(scale_sums), scale_sums)
    probs = squared_coeffs / scale_sums  # (num_nodes, num_filters)

    # Sample within each scale
    for s in range(num_filters):
        num_samples = samples_per_scale[s].item()
        if num_samples == 0:
            continue

        p = probs[:, s]
        if p.sum() == 0:
            samples = torch.randperm(num_nodes, device=coeffs.device)[:num_samples]
        else:
            samples = torch.multinomial(p, num_samples=num_samples, replacement=False)

        mask[samples, s] = 1.0

    return mask


def multilevel_importance_sampling_with_exploration(
        coeffs: torch.Tensor,
        keep_ratio: float,
        exploration_rate: float = 0.0
) -> torch.Tensor:
    """
    Multilevel Importance Sampling (MLIS) with exploration control.
    GPU-optimized vectorized version.

    Args:
        coeffs: Graph wavelet coefficients of shape (num_nodes, num_filters)
        keep_ratio: Fraction of coefficients to keep (0 < keep_ratio <= 1)
        exploration_rate: Controls exploration vs exploitation tradeoff
            - 0.0: Pure exploitation (standard MLIS, energy-based sampling)
            - 1.0: Pure exploration (uniform random sampling)
            - 0.3: 70% MLIS + 30% uniform sampling

    Returns:
        Binary mask of shape (num_nodes, num_filters) indicating selected coefficients
    """
    num_nodes, num_filters = coeffs.shape
    total_coeffs = num_nodes * num_filters
    total_to_keep = max(1, int(total_coeffs * keep_ratio))

    # Clamp exploration_rate to valid range
    exploration_rate = max(0.0, min(1.0, exploration_rate))

    # Split samples between exploitation (MLIS) and exploration (uniform)
    num_exploit_samples = int(total_to_keep * (1.0 - exploration_rate))
    num_explore_samples = total_to_keep - num_exploit_samples

    # Initialize mask
    mask = torch.zeros_like(coeffs)

    # ========================================
    # PART 1: EXPLOITATION (MLIS sampling)
    # ========================================
    if num_exploit_samples > 0:
        # Calculate per-scale L2 energy: E_k = Σ |w_{i,k}|²
        scale_energies = torch.sum(torch.abs(coeffs) ** 2, dim=0)  # Shape: (num_filters,)
        total_energy = scale_energies.sum()

        if total_energy == 0:
            # Edge case: all coefficients are zero, fall back to uniform sampling
            scale_ratios = torch.ones(num_filters, device=coeffs.device) / num_filters
        else:
            scale_ratios = scale_energies / total_energy

        # Allocate samples per scale: m_k = r_k * m
        samples_per_scale_float = scale_ratios * num_exploit_samples
        samples_per_scale = samples_per_scale_float.long()

        # Distribute remainder based on fractional parts
        remainder = num_exploit_samples - samples_per_scale.sum().item()
        if remainder > 0:
            fractional_parts = samples_per_scale_float - samples_per_scale.float()
            _, sorted_indices = torch.topk(fractional_parts, min(remainder, num_filters))
            samples_per_scale[sorted_indices] += 1

        # Ensure we don't exceed num_nodes per scale
        samples_per_scale = torch.clamp(samples_per_scale, max=num_nodes)

        # Pre-compute squared coefficients and probabilities for all scales
        squared_coeffs = torch.abs(coeffs) ** 2  # (num_nodes, num_filters)
        scale_sums = squared_coeffs.sum(dim=0, keepdim=True)  # (1, num_filters)

        # Avoid division by zero
        scale_sums = torch.where(scale_sums == 0, torch.ones_like(scale_sums), scale_sums)
        probs = squared_coeffs / scale_sums  # (num_nodes, num_filters)

        # Sample within each scale (energy-based)
        for s in range(num_filters):
            num_samples = samples_per_scale[s].item()
            if num_samples == 0:
                continue

            p = probs[:, s]
            if p.sum() == 0:
                samples = torch.randperm(num_nodes, device=coeffs.device)[:num_samples]
            else:
                samples = torch.multinomial(p, num_samples=num_samples, replacement=False)

            mask[samples, s] = 1.0

    # ========================================
    # PART 2: EXPLORATION (Uniform sampling)
    # ========================================
    if num_explore_samples > 0:
        # Divide exploration samples equally among scales
        base_samples_per_scale = num_explore_samples // num_filters
        residual = num_explore_samples % num_filters

        # Randomly assign residual samples to scales
        residual_scales = torch.randperm(num_filters, device=coeffs.device)[:residual]

        for s in range(num_filters):
            num_samples = base_samples_per_scale
            if s in residual_scales:
                num_samples += 1

            if num_samples == 0:
                continue

            # Find indices where mask is still 0 (not yet sampled)
            available_indices = torch.where(mask[:, s] == 0)[0]

            if len(available_indices) == 0:
                continue

            # Uniformly sample from available (unsampled) nodes
            num_samples = min(num_samples, len(available_indices))
            selected_positions = torch.randperm(len(available_indices), device=coeffs.device)[:num_samples]
            selected_indices = available_indices[selected_positions]

            mask[selected_indices, s] = 1.0

    return mask



def weighted_random_per_scale(
        coeffs: torch.Tensor,
        keep_ratio: float
) -> torch.Tensor:
    """
    Generate mask using energy-based weighted sampling per scale.
    GPU-optimized version.

    Args:
        coeffs: Graph wavelet coefficients of shape (num_nodes, num_filters)
        keep_ratio: Fraction of total coefficients to keep

    Returns:
        Binary sampling mask of shape (num_nodes, num_filters)
    """
    num_nodes, num_filters = coeffs.shape
    total_coeffs = num_nodes * num_filters
    total_to_keep = int(total_coeffs * keep_ratio)

    mask = torch.zeros_like(coeffs)

    # Calculate base samples per scale
    base_samples_per_scale = int(num_nodes * keep_ratio)
    base_samples_per_scale = max(base_samples_per_scale, 1)

    # Calculate total from base allocation
    total_from_base = base_samples_per_scale * num_filters
    remainder = total_to_keep - total_from_base

    # Calculate energy per scale for ordering
    scale_energies = torch.sum(torch.abs(coeffs), dim=0)  # Shape: (num_filters,)
    # Sort scales by energy (descending)
    sorted_scale_indices = torch.argsort(scale_energies, descending=True)

    # Determine how many samples each scale should get
    samples_per_scale = torch.full((num_filters,), base_samples_per_scale,
                                   dtype=torch.long, device=coeffs.device)

    # Distribute remainder in round-robin manner based on energy ordering
    for i in range(remainder):
        scale_idx = sorted_scale_indices[i % num_filters]
        samples_per_scale[scale_idx] += 1

    # Ensure we don't exceed num_nodes per scale
    samples_per_scale = torch.clamp(samples_per_scale, max=num_nodes)

    for s in range(num_filters):
        num_samples = samples_per_scale[s].item()

        if num_samples == 0:
            continue

        coeffs_s = coeffs[:, s]
        abs_coeffs_s = torch.abs(coeffs_s)
        sum_abs = abs_coeffs_s.sum()

        if sum_abs == 0:
            # Edge case: all zero - FIX: Use device-aware randperm
            samples = torch.randperm(num_nodes, device=coeffs.device)[:num_samples]
        else:
            p = abs_coeffs_s / sum_abs
            samples = torch.multinomial(p, num_samples=num_samples, replacement=False)

        mask[samples, s] = 1

    return mask


def random_mask(
        coeffs: torch.Tensor,
        keep_ratio: float
) -> torch.Tensor:
    """
    Generate random mask (already GPU-compatible).

    Args:
        coeffs: Graph wavelet coefficients of shape (num_nodes, num_filters)
        keep_ratio: Fraction of total coefficients to keep

    Returns:
        Binary sampling mask of shape (num_nodes, num_filters)
    """
    num_nodes, num_filters = coeffs.shape
    total_coeffs = num_nodes * num_filters
    total_to_keep = int(total_coeffs * keep_ratio)

    # Create a flat mask on same device as coeffs
    flat_mask = torch.zeros(total_coeffs, dtype=coeffs.dtype, device=coeffs.device)

    # Randomly select indices to keep - already uses device
    random_indices = torch.randperm(total_coeffs, device=coeffs.device)[:total_to_keep]
    flat_mask[random_indices] = 1

    # Reshape to original dimensions
    mask = flat_mask.view(num_nodes, num_filters)
    return mask


class SparsityAnnealer:
    """Anneal keep_ratio from initial_value to final_value over num_epochs."""

    def __init__(self, initial_value=1.0, final_value=0.01, num_epochs=1000, strategy='cosine'):
        """
        Args:
            initial_value: Starting keep_ratio (default: 1.0)
            final_value: Ending keep_ratio (default: 0.01)
            num_epochs: Number of epochs to anneal over
            strategy: 'cosine', 'linear', 'exponential', or 'step'
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_epochs = num_epochs
        self.strategy = strategy
        self.current_epoch = 0

    def get_keep_ratio(self, epoch=None):
        """Get the keep_ratio for a given epoch."""
        if epoch is None:
            epoch = self.current_epoch

        if epoch >= self.num_epochs:
            return self.final_value

        progress = epoch / self.num_epochs  # 0 to 1

        if self.strategy == 'cosine':
            # Cosine annealing: smooth decay
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            keep_ratio = self.final_value + (self.initial_value - self.final_value) * cosine_factor

        elif self.strategy == 'linear':
            # Linear decay
            keep_ratio = self.initial_value + (self.final_value - self.initial_value) * progress

        elif self.strategy == 'exponential':
            # Exponential decay
            decay_rate = np.log(self.final_value / self.initial_value)
            keep_ratio = self.initial_value * np.exp(decay_rate * progress)

        elif self.strategy == 'step':
            # Step decay (4 steps)
            num_steps = 4
            step_size = self.num_epochs // num_steps
            step = min(epoch // step_size, num_steps - 1)
            step_progress = step / (num_steps - 1)
            keep_ratio = self.initial_value + (self.final_value - self.initial_value) * step_progress

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return keep_ratio

    def step(self):
        """Increment the epoch counter."""
        self.current_epoch += 1
        return self.get_keep_ratio()

    def reset(self):
        """Reset the epoch counter."""
        self.current_epoch = 0
    def is_complete(self, epoch=None):
        """Check if the annealing process has finished."""
        if epoch is None:
            epoch = self.current_epoch
        return epoch >= self.num_epochs