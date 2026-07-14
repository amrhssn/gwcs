from typing import Callable

import torch

from transform.kernels import (
  compute_dual_kernel_samples,
  compute_spectral_kernel_samples
)
from transform.utils import (
  flatten_trailing_dims,
  unflatten_trailing_dims
)


def compute_eigendecomposition(
        symmetric_normalized_laplacian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  dense_symmetric_normalized_laplacian = (
    symmetric_normalized_laplacian.to_dense()
    if symmetric_normalized_laplacian.is_sparse
    else symmetric_normalized_laplacian
  )
  eigenvalues, eigenvectors = torch.linalg.eigh(dense_symmetric_normalized_laplacian)
  return eigenvalues, eigenvectors


def resolve_eigendecomposition(
        reference_tensor: torch.Tensor,
        symmetric_normalized_laplacian: torch.Tensor,
        eigenvalues: torch.Tensor | None,
        eigenvectors: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if eigenvalues is None or eigenvectors is None:
    eigenvalues, eigenvectors = compute_eigendecomposition(symmetric_normalized_laplacian)
  device, dtype = reference_tensor.device, reference_tensor.dtype
  return eigenvalues.to(device=device, dtype=dtype), eigenvectors.to(device=device, dtype=dtype)


def analysis(
        signal: torch.Tensor,
        symmetric_normalized_laplacian: torch.Tensor,
        spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
        eigenvalues: torch.Tensor | None = None,
        eigenvectors: torch.Tensor | None = None,
) -> torch.Tensor:
  eigenvalues, eigenvectors = resolve_eigendecomposition(
    reference_tensor=signal,
    symmetric_normalized_laplacian=symmetric_normalized_laplacian,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
  )
  signal_flat, trailing_shape = flatten_trailing_dims(signal)  # (N, S)

  graph_fourier_coeffs = eigenvectors.T @ signal_flat  # (N, S), N indexes eigencomponents here
  spectral_kernel_samples = compute_spectral_kernel_samples(spectral_kernels=spectral_kernels,
                                                            lambdas=eigenvalues)  # (F, N)
  filtered = torch.einsum('fe,es->fes', spectral_kernel_samples, graph_fourier_coeffs)  # (F, N, S)
  wavelet_coeffs_flat = torch.einsum('ne,fes->nsf', eigenvectors, filtered)  # (N, S, F)
  return unflatten_trailing_dims(wavelet_coeffs_flat, trailing_shape)  # (N, *S, F)


def synthesis(
        wavelet_coeffs: torch.Tensor,
        symmetric_normalized_laplacian: torch.Tensor,
        spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
        eigenvalues: torch.Tensor | None = None,
        eigenvectors: torch.Tensor | None = None,
        eps: float = 1e-12,
) -> torch.Tensor:
  eigenvalues, eigenvectors = resolve_eigendecomposition(
    reference_tensor=wavelet_coeffs,
    symmetric_normalized_laplacian=symmetric_normalized_laplacian,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
  )
  num_nodes = wavelet_coeffs.shape[0]
  trailing_shape = wavelet_coeffs.shape[1:-1]
  num_filters = wavelet_coeffs.shape[-1]
  num_columns = 1
  for dim in trailing_shape:
    num_columns *= dim
  wavelet_coeffs_flat = wavelet_coeffs.reshape(num_nodes, num_columns, num_filters)  # (N, S, F)

  graph_fourier_coeffs_per_filter = torch.einsum('ne,nsf->esf', eigenvectors, wavelet_coeffs_flat)  # (N, S, F), eigen-index
  spectral_kernel_samples = compute_spectral_kernel_samples(spectral_kernels=spectral_kernels,
                                                            lambdas=eigenvalues)  # (F, N)
  dual_kernel_samples = compute_dual_kernel_samples(spectral_kernel_samples=spectral_kernel_samples, eps=eps)  # (F, N)
  dual_scaled_graph_fourier_coeffs = torch.einsum('esf,fe->es', graph_fourier_coeffs_per_filter, dual_kernel_samples)  # (N, S)
  signal_flat = eigenvectors @ dual_scaled_graph_fourier_coeffs  # (N, S)
  return unflatten_trailing_dims(signal_flat, trailing_shape)  # (N, *S)
