from dataclasses import dataclass, field
from typing import (
  Literal,
  Union
)

from configs.common import GraphWaveletTransformConfig, GWCSConfig, SGAEConfig, TrainingConfig


@dataclass(frozen=True)
class GraphGenerationConfig:
  graph_types: list[Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm']] = field(
    default_factory=lambda: ['grid'])
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
class GenerationConfig:
  graph_config: GraphGenerationConfig = field(default_factory=GraphGenerationConfig)
  signal_config: SignalGenerationConfig = field(default_factory=SignalGenerationConfig)
  gwt_config: GraphWaveletTransformConfig = field(default_factory=GraphWaveletTransformConfig)


@dataclass(frozen=True)
class ABLExperimentConfig:
  model_config: Union[SGAEConfig, GWCSConfig]
  target_compression_ratio: float = 0.05
  seed: int = 42
  generation_config: GenerationConfig = field(default_factory=GenerationConfig)
  training_config: TrainingConfig = field(default_factory=TrainingConfig)


def default_abl_configs_gwcs() -> ABLExperimentConfig:
  return ABLExperimentConfig(model_config=GWCSConfig(),
                             training_config=TrainingConfig(early_stopping_patience=100,
                                                            num_epochs=1_000))


def default_abl_configs_sgae() -> ABLExperimentConfig:
  return ABLExperimentConfig(model_config=SGAEConfig(),
                             training_config=TrainingConfig(early_stopping_patience=100,
                                                            num_epochs=1_000))
