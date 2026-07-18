from dataclasses import dataclass
from typing import (
  Literal, Union
)

import torch


@dataclass(frozen=True)
class GraphGenerationConfig:
  graph_types: list[Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm']]
  num_nodes_range: tuple[int, int]
  num_graphs_per_type: int
  barabasi_albert_min_num_edges_per_new_node_range: tuple[int, int] | None = None
  erdos_renyi_edge_prob_range: tuple[float, float] | None = None
  sbm_num_blocks_range: tuple[int, int] | None = None
  sbm_intra_cluster_prob_range: tuple[float, float] | None = None
  sbm_inter_cluster_prob_range: tuple[float, float] | None = None
  return_largest_connected_component: bool = False


@dataclass(frozen=True)
class SignalGenerationConfig:
  signal_types: list[Literal['lowpass', 'highpass', 'bandpass', 'R-sparse']]
  max_num_frequency_mode_ratio: float
  coeffs_mean: float
  coeffs_std_range: tuple[float, float]
  noise_level_range: tuple[float, float]


@dataclass(frozen=True)
class GraphWaveletTransformConfig:
  gwt_mode: Literal["chebyshev", "exact"]
  spectrum_lower_bound: float
  spectrum_upper_bound: float
  scaling_kernel_bandwidth_ratio: float
  scaling_kernel_cutoff_eigenvalue: float
  chebyshev_degree: int
  num_scales: int
  num_filters: int
  scales: list[float]


@dataclass(frozen=True)
class GenerationConfig:
  graph_config: GraphGenerationConfig
  signal_config: SignalGenerationConfig
  gwt_config: GraphWaveletTransformConfig


@dataclass(frozen=True)
class GraphMetadata:
  graph_type: str
  num_nodes: int
  num_edges: int


@dataclass(frozen=True)
class GridGraphMetadata(GraphMetadata):
  height: int
  width: int
  pos: list[list[float]]


@dataclass(frozen=True)
class ErdosRenyiGraphMetadata(GraphMetadata):
  is_directed: bool
  edge_prob_range: tuple[float, float]
  edge_prob: float


@dataclass(frozen=True)
class BarabasiAlbertGraphMetadata(GraphMetadata):
  num_edges_per_new_node: int
  num_edges_per_new_node_range: tuple[int, int]


@dataclass(frozen=True)
class StochasticBlockModelGraphMetadata(GraphMetadata):
  num_blocks: int
  block_sizes: list[int]
  edge_probs: list[list[float]]
  intra_block_prob: float
  inter_block_prob: float


@dataclass(frozen=True)
class Graph:
  edge_index: torch.Tensor
  symmetric_normalized_laplacian: torch.Tensor
  eigenvalues: torch.Tensor
  eigenvectors: torch.Tensor
  metadata: Union[
    GridGraphMetadata,
    ErdosRenyiGraphMetadata,
    BarabasiAlbertGraphMetadata,
    StochasticBlockModelGraphMetadata
  ]


@dataclass(frozen=True)
class ABLSignal:
  x: torch.Tensor
  x_clean: torch.Tensor
  signal_type: Literal['lowpass', 'highpass', 'bandpass', 'R-sparse']
  num_nodes: int
  num_frequency_modes: int
  selected_indices: torch.Tensor
  coeffs: torch.Tensor
  U_R: torch.Tensor
  noise_level: float | None
  noise_std: float | None


@dataclass(frozen=True)
class ABLGraphSignalSample:
  graph: Graph
  signal: ABLSignal
  gwt_coeffs: torch.Tensor
