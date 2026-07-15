from typing import Callable

import torch


def scaling_kernel(
        eigenvalues: torch.Tensor,
        minimum_eigenvalue: float,
        minimum_eigenvalue_coeff: float = 50.0,
        exp_coeff: float = 0.4
) -> torch.Tensor:
  tau = minimum_eigenvalue_coeff * minimum_eigenvalue
  return exp_coeff * torch.exp(-tau * eigenvalues)


def mexican_hat_kernel(
        eigenvalues: torch.Tensor,
        scale: float
) -> torch.Tensor:
  return scale * eigenvalues * torch.exp(-scale * eigenvalues)


def default_kernels(
        minimum_eigenvalue: float,
        scales: list[float]
) -> list[Callable[[torch.Tensor], torch.Tensor]]:
  kernels = [lambda lam: scaling_kernel(eigenvalues=lam, minimum_eigenvalue=minimum_eigenvalue)]
  kernels += [lambda lam, s=s: mexican_hat_kernel(eigenvalues=lam, scale=s) for s in scales]
  return kernels


def compute_spectral_kernel_samples(
        spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
        lambdas: torch.Tensor,
) -> torch.Tensor:
  return torch.stack([kernel(lambdas) for kernel in spectral_kernels], dim=0)


def compute_dual_kernel_samples(
        spectral_kernel_samples: torch.Tensor,
        eps: float = 1e-12,
) -> torch.Tensor:
  squared_kernel_sum = torch.clamp((spectral_kernel_samples ** 2).sum(dim=0), min=eps)
  return spectral_kernel_samples / squared_kernel_sum.unsqueeze(0)
