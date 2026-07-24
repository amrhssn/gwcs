import math
from dataclasses import dataclass, field
from typing import Literal

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
  num_layers: int = 7
  is_variational: bool = False
  kl_beta: float = 1e-4
  # STE temperature schedule
  ste_temperature_mode: Literal['fixed', 'annealing'] = 'annealing'
  ste_temperature_init: float = 0.5
  ste_temperature_final: float = 0.05
  ste_temperature_anneal_epochs: int = 100


@dataclass(frozen=True)
class TrainingConfig:
  lr: float = 1e-3
  weight_decay: float = 5e-4
  train_ratio: float = 0.6
  val_ratio: float = 0.2
  batch_size: int = 25
  num_epochs: int = 100
  lr_scheduler_factor: float = 0.5
  lr_scheduler_patience: int = 8
  lr_scheduler_min_lr: float = 1e-6
  early_stopping_patience: int = 24
  early_stopping_min_delta: float = 0.01
