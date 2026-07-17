import torch


def energy_distribution_across_scales(
        coeffs: torch.Tensor
) -> torch.Tensor:
    per_scale_energy = (coeffs.abs() ** 2).sum(dim=0)
    total_energy = per_scale_energy.sum()
    if total_energy.item() == 0:
        return torch.zeros_like(per_scale_energy)
    return per_scale_energy / total_energy

