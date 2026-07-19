import math
from dataclasses import dataclass, field
from typing import (
  Literal,
  Union
)

import torch


_SPECTRUM_UPPER_BOUND = 2.0
_SCALING_KERNEL_BANDWIDTH_RATIO = 50
_SCALING_KERNEL_CUTOFF_EIGENVALUE = _SPECTRUM_UPPER_BOUND / _SCALING_KERNEL_BANDWIDTH_RATIO
_NUM_SCALES = 4


def _default_scales() -> list[float]:
  return torch.logspace(math.log10(1 / _SPECTRUM_UPPER_BOUND),
                        math.log10(1 / _SCALING_KERNEL_CUTOFF_EIGENVALUE),
                        _NUM_SCALES).tolist()


@dataclass(frozen=True)
class GraphGenerationConfig:
  graph_types: list[Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm']] = field(
    default_factory=lambda: ['erdos_renyi'])
  num_nodes_range: tuple[int, int] = (200, 400)
  num_graphs_per_type: int = 200
  barabasi_albert_min_num_edges_per_new_node_range: tuple[int, int] | None = (1, 2)
  erdos_renyi_edge_prob_range: tuple[float, float] | None = (0.005, 0.01)
  sbm_num_blocks_range: tuple[int, int] | None = (2, 5)
  sbm_intra_cluster_prob_range: tuple[float, float] | None = (0.05, 0.15)
  sbm_inter_cluster_prob_range: tuple[float, float] | None = (0.01, 0.03)
  return_largest_connected_component: bool = True


@dataclass(frozen=True)
class SignalGenerationConfig:
  signal_types: list[Literal['lowpass', 'highpass', 'bandpass', 'R-sparse']] = field(
    default_factory=lambda: ['lowpass'])
  max_num_frequency_mode_ratio: float = 0.1
  coeffs_mean: float = 0.0
  coeffs_std_range: tuple[float, float] = (1.0, 2.0)
  noise_level_range: tuple[float, float] = (0.05, 0.2)


@dataclass(frozen=True)
class GraphWaveletTransformConfig:
  gwt_mode: Literal["chebyshev", "exact"] = 'chebyshev'
  spectrum_lower_bound: float = 0.0
  spectrum_upper_bound: float = _SPECTRUM_UPPER_BOUND
  scaling_kernel_bandwidth_ratio: float = _SCALING_KERNEL_BANDWIDTH_RATIO
  scaling_kernel_cutoff_eigenvalue: float = _SCALING_KERNEL_CUTOFF_EIGENVALUE
  chebyshev_degree: int = 100
  num_scales: int = _NUM_SCALES
  num_filters: int = _NUM_SCALES + 1
  scales: list[float] = field(default_factory=_default_scales)


@dataclass(frozen=True)
class GenerationConfig:
  graph_config: GraphGenerationConfig = field(default_factory=GraphGenerationConfig)
  signal_config: SignalGenerationConfig = field(default_factory=SignalGenerationConfig)
  gwt_config: GraphWaveletTransformConfig = field(default_factory=GraphWaveletTransformConfig)


@dataclass(frozen=True)
class GWCSConfig:
  model_name: str = 'gwcs'
  hidden_dim: int = 128
  activation_fn: str = 'relu'
  conv_layer_type: str = 'SAGEConv'
  processor_num_layers: int = 5


@dataclass(frozen=True)
class SGAEConfig:
  model_name: str = 'sgae'
  latent_dim: int = 128
  activation_fn: str = 'relu'
  conv_layer_type: str = 'SAGEConv'
  num_layers: int = 5
  is_variational: bool = True
  kl_beta: float = 1.0
  # STE temperature schedule
  ste_temperature_mode: Literal['fixed', 'annealing'] = 'fixed'
  ste_temperature_init: float = 0.5
  ste_temperature_final: float = 0.05
  ste_temperature_anneal_epochs: int = 100


@dataclass(frozen=True)
class TrainingConfig:
  lr: float = 1e-3
  weight_decay: float = 5e-4
  train_ratio: float = 0.6
  val_ratio: float = 0.2
  batch_size: int = 16
  num_epochs: int = 500
  lr_scheduler_factor: float = 0.5
  lr_scheduler_patience: int = 50
  lr_scheduler_min_lr: float = 1e-6
  early_stopping_patience: int = 150
  early_stopping_min_delta: float = 1e-4


@dataclass(frozen=True)
class ABLExperimentConfig:
  model_config: Union[SGAEConfig, GWCSConfig]
  target_compression_ratio: float = 0.05
  seed: int = 42
  generation_config: GenerationConfig = field(default_factory=GenerationConfig)
  training_config: TrainingConfig = field(default_factory=TrainingConfig)


def default_abl_configs_gwcs() -> ABLExperimentConfig:
  return ABLExperimentConfig(model_config=GWCSConfig())


def default_abl_configs_sgae() -> ABLExperimentConfig:
  return ABLExperimentConfig(model_config=SGAEConfig())
