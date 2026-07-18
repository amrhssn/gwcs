"""
End-to-end smoke script for the visualization module: builds a ~100-node
grid graph, a 2-channel ABL signal, runs exact and Chebyshev SGWT
analysis/synthesis, and plots everything the visualization module supports.

Run with `uv run python scripts/visualize_sample_abl.py`.
"""

import matplotlib.pyplot as plt
import torch

from data.generation.graph import generate_graph
from data.generation.signal import approximately_bandlimited
from transform.kernels import default_kernels
from transform.exact_sgwt import (
  analysis as exact_analysis,
  synthesis as exact_synthesis,
)
from transform.chebyshev_sgwt import (
  analysis as chebyshev_analysis,
  synthesis as chebyshev_synthesis,
)
from sampler.mlis import multilevel_importance_sampling
from visualization import (
  visualize_signal,
  visualize_grid_gwt_coefficients,
  visualize_grid_reconstruction,
  visualize_grid_sparsification_mask,
)

NUM_NODES = 100
SCALES = [0.3, 1.0, 3.0]
SPECTRUM_LOWER_BOUND = 0.0
SPECTRUM_UPPER_BOUND = 2.0
SCALING_KERNEL_CUTOFF_EIGENVALUE = SPECTRUM_UPPER_BOUND / 50
CHEBYSHEV_DEGREE = 30
KEEP_RATIO = 0.3


def _make_channel(eigenvectors: torch.Tensor, seed: int) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  num_nodes = eigenvectors.shape[0]
  num_frequency_modes = max(3, num_nodes // 10)
  coeffs = torch.randn(num_frequency_modes, generator=generator)
  signal = approximately_bandlimited(coeffs=coeffs,
                                     num_frequency_modes=num_frequency_modes,
                                     signal_type='lowpass',
                                     eigenvector=eigenvectors,
                                     noise_level=0.1,
                                     random_generator=generator)
  return signal.x  # (N, 1)


def main() -> None:
  # 1. grid graph
  graph = generate_graph(graph_type='grid', num_nodes=NUM_NODES)
  rows, cols = graph.metadata.height, graph.metadata.width

  # 2. two-channel signal from two independent ABL signals
  signal = torch.cat([_make_channel(graph.eigenvectors, seed=0),
                      _make_channel(graph.eigenvectors, seed=1)], dim=1)  # (N, 2)

  kernels = default_kernels(minimum_eigenvalue=SCALING_KERNEL_CUTOFF_EIGENVALUE, scales=SCALES)

  # 3. exact and Chebyshev SGWT analysis
  exact_coeffs = exact_analysis(signal=signal,
                                symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                spectral_kernels=kernels,
                                eigenvalues=graph.eigenvalues,
                                eigenvectors=graph.eigenvectors)
  chebyshev_coeffs = chebyshev_analysis(signal=signal,
                                        symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                        spectral_kernels=kernels,
                                        spectrum_lower_bound=SPECTRUM_LOWER_BOUND,
                                        spectrum_upper_bound=SPECTRUM_UPPER_BOUND,
                                        chebyshev_degree=CHEBYSHEV_DEGREE)

  # 4. exact and Chebyshev reconstruction
  exact_reconstructed = exact_synthesis(wavelet_coeffs=exact_coeffs,
                                        symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                        spectral_kernels=kernels,
                                        eigenvalues=graph.eigenvalues,
                                        eigenvectors=graph.eigenvectors)
  chebyshev_reconstructed = chebyshev_synthesis(wavelet_coeffs=chebyshev_coeffs,
                                                symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                                spectral_kernels=kernels,
                                                spectrum_lower_bound=SPECTRUM_LOWER_BOUND,
                                                spectrum_upper_bound=SPECTRUM_UPPER_BOUND,
                                                chebyshev_degree=CHEBYSHEV_DEGREE)

  # sparsification mask, so the mask visualization also gets exercised
  mask = multilevel_importance_sampling(coeffs=exact_coeffs, keep_ratio=KEEP_RATIO,
                                        random_generator=torch.Generator().manual_seed(0))

  # 5. plot everything the visualization module supports
  visualize_signal(signal=signal, edge_index=graph.edge_index, graph_type='grid', pos=graph.metadata.pos)
  visualize_grid_gwt_coefficients(coeffs=exact_coeffs, signal=signal, scales=SCALES, rows=rows, cols=cols,
                                  gwt_mode='exact')
  visualize_grid_gwt_coefficients(coeffs=chebyshev_coeffs, signal=signal, scales=SCALES, rows=rows, cols=cols,
                                  gwt_mode='chebyshev')
  visualize_grid_reconstruction(reconstructed=exact_reconstructed, signal=signal, rows=rows, cols=cols,
                                gwt_mode='exact')
  visualize_grid_reconstruction(reconstructed=chebyshev_reconstructed, signal=signal, rows=rows, cols=cols,
                                gwt_mode='chebyshev')
  visualize_grid_sparsification_mask(mask=mask, rows=rows, cols=cols)

  plt.show()


if __name__ == '__main__':
  main()
