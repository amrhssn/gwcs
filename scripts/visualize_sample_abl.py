from typing import Literal

import torch
import networkx as nx
import matplotlib.pyplot as plt

from data.types import GraphGenerationConfig
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
  visualize_graph_gwt_coefficients,
  visualize_graph_reconstruction,
  visualize_graph_sparsification_mask,
)

GRAPH_TYPES: list[Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm']] = ['grid', 'erdos_renyi']
NUM_NODES = 100
SCALES = [0.3, 1.0, 3.0]
SPECTRUM_LOWER_BOUND = 0.0
SPECTRUM_UPPER_BOUND = 2.0
SCALING_KERNEL_CUTOFF_EIGENVALUE = SPECTRUM_UPPER_BOUND / 50
CHEBYSHEV_DEGREE = 30
KEEP_RATIO = 0.05
ERDOS_RENYI_EDGE_PROB_RANGE = (0.04, 0.06)

GRAPH_CONFIG = GraphGenerationConfig(
  graph_types=GRAPH_TYPES,
  num_nodes_range=(NUM_NODES, NUM_NODES),
  num_graphs_per_type=1,
  erdos_renyi_edge_prob_range=ERDOS_RENYI_EDGE_PROB_RANGE,
  return_largest_connected_component=True,
)


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


def _visualize_graph_type(graph_type: str) -> None:
  graph = generate_graph(graph_type=graph_type, num_nodes=NUM_NODES, graph_config=GRAPH_CONFIG,
                         random_generator=torch.Generator().manual_seed(0))

  # two-channel signal from two independent ABL signals
  signal = torch.cat([_make_channel(graph.eigenvectors, seed=0),
                      _make_channel(graph.eigenvectors, seed=1)], dim=1)  # (N, 2)

  kernels = default_kernels(minimum_eigenvalue=SCALING_KERNEL_CUTOFF_EIGENVALUE, scales=SCALES)
  num_filters = len(kernels)

  # exact and Chebyshev SGWT analysis
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

  # exact and Chebyshev reconstruction
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

  if graph_type == 'grid':
    rows, cols = graph.metadata.height, graph.metadata.width
    visualize_signal(signal=signal, edge_index=graph.edge_index, graph_type=graph_type, pos=graph.metadata.pos)
    visualize_grid_gwt_coefficients(coeffs=exact_coeffs, signal=signal, scales=SCALES, rows=rows, cols=cols,
                                    gwt_mode='exact')
    visualize_grid_gwt_coefficients(coeffs=chebyshev_coeffs, signal=signal, scales=SCALES, rows=rows, cols=cols,
                                    gwt_mode='chebyshev')
    visualize_grid_reconstruction(reconstructed=exact_reconstructed, signal=signal, rows=rows, cols=cols,
                                  gwt_mode='exact', num_filters=num_filters, keep_ratio=KEEP_RATIO)
    visualize_grid_reconstruction(reconstructed=chebyshev_reconstructed, signal=signal, rows=rows, cols=cols,
                                  gwt_mode='chebyshev', num_filters=num_filters, keep_ratio=KEEP_RATIO)
    visualize_grid_sparsification_mask(mask=mask, rows=rows, cols=cols, keep_ratio=KEEP_RATIO)
  else:
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.metadata.num_nodes))
    nx_graph.add_edges_from(graph.edge_index.t().numpy())
    layout = nx.kamada_kawai_layout(nx_graph)
    visualize_signal(signal=signal, edge_index=graph.edge_index, graph_type=graph_type, pos=None)
    visualize_graph_gwt_coefficients(coeffs=exact_coeffs, signal=signal, scales=SCALES, edge_index=graph.edge_index,
                                     layout=layout, gwt_mode='exact')
    visualize_graph_gwt_coefficients(coeffs=chebyshev_coeffs, signal=signal, scales=SCALES,
                                     edge_index=graph.edge_index, layout=layout, gwt_mode='chebyshev')
    visualize_graph_reconstruction(reconstructed=exact_reconstructed, signal=signal, edge_index=graph.edge_index,
                                   layout=layout, gwt_mode='exact', num_filters=num_filters, keep_ratio=KEEP_RATIO)
    visualize_graph_reconstruction(reconstructed=chebyshev_reconstructed, signal=signal,
                                   edge_index=graph.edge_index, layout=layout, gwt_mode='chebyshev',
                                   num_filters=num_filters, keep_ratio=KEEP_RATIO)
    visualize_graph_sparsification_mask(mask=mask, edge_index=graph.edge_index, layout=layout, keep_ratio=KEEP_RATIO)


def main() -> None:
  for graph_type in GRAPH_TYPES:
    _visualize_graph_type(graph_type)

  plt.show()


if __name__ == '__main__':
  main()
