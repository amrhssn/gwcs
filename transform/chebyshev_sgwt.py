import math
from typing import (
  Callable,
  Iterator
)

import torch


def _rescale_laplacian_spectrum_to_chebyshev_domain(
        symmetric_normalized_laplacian: torch.Tensor,
        spectrum_lower_bound: float,
        spectrum_upper_bound: float,
) -> torch.Tensor:
  device, dtype = symmetric_normalized_laplacian.device, symmetric_normalized_laplacian.dtype
  num_nodes = symmetric_normalized_laplacian.shape[0]
  a = (spectrum_upper_bound - spectrum_lower_bound) / 2.0
  b = (spectrum_upper_bound + spectrum_lower_bound) / 2.0
  idx = torch.arange(num_nodes, device=device, dtype=torch.long)
  identity = torch.sparse_coo_tensor(indices=torch.stack([idx, idx]),
                                     values=torch.ones(num_nodes, device=device, dtype=dtype),
                                     size=(num_nodes, num_nodes))
  rescaled_symmetric_normalized_laplacian = (symmetric_normalized_laplacian - b * identity) * (1.0 / a)
  return rescaled_symmetric_normalized_laplacian


def _chebyshev_nodes(
        chebyshev_degree: int,
        device: torch.device,
        dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
  M = chebyshev_degree + 1
  l = torch.arange(M, device=device, dtype=dtype)
  theta = math.pi * (l + 0.5) / M
  return theta, torch.cos(theta)


def _compute_chebyshev_coeffs(
        spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
        spectrum_lower_bound: float,
        spectrum_upper_bound: float,
        chebyshev_degree: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
  theta, x = _chebyshev_nodes(chebyshev_degree=chebyshev_degree,
                              device=device,
                              dtype=dtype)
  a = (spectrum_upper_bound - spectrum_lower_bound) / 2.0
  b = (spectrum_upper_bound + spectrum_lower_bound) / 2.0
  lam = a * x + b
  samples = torch.stack([kernel(lam).to(device=device, dtype=dtype) for kernel in spectral_kernels], dim=0)
  k = torch.arange(chebyshev_degree + 1, device=device, dtype=dtype).unsqueeze(1)
  cos_k_theta = torch.cos(k * theta.unsqueeze(0))
  return (2.0 / (chebyshev_degree + 1)) * (samples @ cos_k_theta.T)


def _chebyshev_terms(
        rescaled_symmetric_normalized_laplacian: torch.Tensor,
        vector: torch.Tensor,
        chebyshev_degree: int
) -> Iterator[torch.Tensor]:
  term_prev2 = vector
  yield term_prev2

  if chebyshev_degree == 0:
    return

  term_prev1 = torch.sparse.mm(rescaled_symmetric_normalized_laplacian, vector)
  yield term_prev1

  for _ in range(2, chebyshev_degree + 1):
    term_curr = 2.0 * torch.sparse.mm(rescaled_symmetric_normalized_laplacian, term_prev1) - term_prev2
    yield term_curr
    term_prev2, term_prev1 = term_prev1, term_curr


def _compute_dual_chebyshev_coeffs(
        chebyshev_coeffs: torch.Tensor,
        chebyshev_degree: int,
        eps: float = 1e-12
) -> torch.Tensor:
  device, dtype = chebyshev_coeffs.device, chebyshev_coeffs.dtype
  theta, _ = _chebyshev_nodes(chebyshev_degree=chebyshev_degree,
                              device=device,
                              dtype=dtype)
  k = torch.arange(chebyshev_degree + 1, device=device, dtype=dtype).unsqueeze(1)
  cos_k_theta = torch.cos(k * theta.unsqueeze(0))
  coeffs = chebyshev_coeffs.clone()
  coeffs[:, 0] *= 0.5
  forward_kernel_samples = coeffs @ cos_k_theta
  squared_kernel_sum = torch.clamp((forward_kernel_samples ** 2).sum(dim=0), min=eps)
  dual_kernel_samples = forward_kernel_samples / squared_kernel_sum.unsqueeze(0)
  dual_chebyshev_coeffs = (2.0 / (chebyshev_degree + 1)) * (dual_kernel_samples @ cos_k_theta.T)
  return dual_chebyshev_coeffs


def _resolve_rescaled_laplacian_and_chebyshev_coeffs(
        reference_tensor: torch.Tensor,
        symmetric_normalized_laplacian: torch.Tensor,
        spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
        spectrum_lower_bound: float,
        spectrum_upper_bound: float,
        chebyshev_degree: int,
        rescaled_symmetric_normalized_laplacian: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  device, dtype = reference_tensor.device, reference_tensor.dtype
  if rescaled_symmetric_normalized_laplacian is None:
    rescaled_symmetric_normalized_laplacian = _rescale_laplacian_spectrum_to_chebyshev_domain(
      symmetric_normalized_laplacian=symmetric_normalized_laplacian.to(device=device, dtype=dtype),
      spectrum_lower_bound=spectrum_lower_bound,
      spectrum_upper_bound=spectrum_upper_bound
    )
  chebyshev_coeffs = _compute_chebyshev_coeffs(spectral_kernels=spectral_kernels,
                                               spectrum_lower_bound=spectrum_lower_bound,
                                               spectrum_upper_bound=spectrum_upper_bound,
                                               chebyshev_degree=chebyshev_degree,
                                               device=device,
                                               dtype=dtype)
  return rescaled_symmetric_normalized_laplacian, chebyshev_coeffs


def analysis(
        signal: torch.Tensor,
        symmetric_normalized_laplacian: torch.Tensor,
        spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
        spectrum_lower_bound: float = 0.0,
        spectrum_upper_bound: float = 2.0,
        chebyshev_degree: int = 30,
        rescaled_symmetric_normalized_laplacian: torch.Tensor | None = None
) -> torch.Tensor:
  rescaled_symmetric_normalized_laplacian, chebyshev_coeffs = _resolve_rescaled_laplacian_and_chebyshev_coeffs(
    reference_tensor=signal,
    symmetric_normalized_laplacian=symmetric_normalized_laplacian,
    spectral_kernels=spectral_kernels,
    spectrum_lower_bound=spectrum_lower_bound,
    spectrum_upper_bound=spectrum_upper_bound,
    chebyshev_degree=chebyshev_degree,
    rescaled_symmetric_normalized_laplacian=rescaled_symmetric_normalized_laplacian
  )
  chebyshev_coeffs[:, 0] *= 0.5
  signal = signal if signal.ndim == 2 else signal.unsqueeze(1)

  chebyshev_term_generator = _chebyshev_terms(
    rescaled_symmetric_normalized_laplacian=rescaled_symmetric_normalized_laplacian,
    vector=signal,
    chebyshev_degree=chebyshev_degree
  )
  wavelet_coeffs = next(chebyshev_term_generator) @ chebyshev_coeffs[:, 0].unsqueeze(0)
  for k, term_k in enumerate(chebyshev_term_generator, start=1):
    wavelet_coeffs = wavelet_coeffs + term_k @ chebyshev_coeffs[:, k].unsqueeze(0)
  return wavelet_coeffs


def synthesis(
        wavelet_coeffs: torch.Tensor,
        symmetric_normalized_laplacian: torch.Tensor,
        spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
        spectrum_lower_bound: float = 0.0,
        spectrum_upper_bound: float = 2.0,
        chebyshev_degree: int = 30,
        rescaled_symmetric_normalized_laplacian: torch.Tensor | None = None
) -> torch.Tensor:
  rescaled_symmetric_normalized_laplacian, chebyshev_coeffs = _resolve_rescaled_laplacian_and_chebyshev_coeffs(
    reference_tensor=wavelet_coeffs,
    symmetric_normalized_laplacian=symmetric_normalized_laplacian,
    spectral_kernels=spectral_kernels,
    spectrum_lower_bound=spectrum_lower_bound,
    spectrum_upper_bound=spectrum_upper_bound,
    chebyshev_degree=chebyshev_degree,
    rescaled_symmetric_normalized_laplacian=rescaled_symmetric_normalized_laplacian
  )
  dual_chebyshev_coeffs = _compute_dual_chebyshev_coeffs(chebyshev_coeffs=chebyshev_coeffs,
                                                         chebyshev_degree=chebyshev_degree)
  dual_chebyshev_coeffs[:, 0] *= 0.5
  chebyshev_term_generator = _chebyshev_terms(
    rescaled_symmetric_normalized_laplacian=rescaled_symmetric_normalized_laplacian,
    vector=wavelet_coeffs,
    chebyshev_degree=chebyshev_degree
  )
  signal = (next(chebyshev_term_generator) * dual_chebyshev_coeffs[:, 0].unsqueeze(0)).sum(dim=1)
  for k, term_k in enumerate(chebyshev_term_generator, start=1):
    signal = signal + (term_k * dual_chebyshev_coeffs[:, k].unsqueeze(0)).sum(dim=1)
  return signal
