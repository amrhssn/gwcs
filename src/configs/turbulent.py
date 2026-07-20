from dataclasses import dataclass, field

from configs.common import GraphWaveletTransformConfig, GWCSConfig, TrainingConfig


@dataclass(frozen=True)
class TurbulentExperimentConfig:
  model_config: GWCSConfig
  target_compression_ratio: float = 0.05
  seed: int = 42
  debug: bool = False
  gwt_config: GraphWaveletTransformConfig = field(default_factory=GraphWaveletTransformConfig)
  training_config: TrainingConfig = field(default_factory=TrainingConfig)


def default_turbulent_configs_gwcs() -> TurbulentExperimentConfig:
  return TurbulentExperimentConfig(model_config=GWCSConfig())
