import math
from dataclasses import dataclass

import torch

from data.types import (
  GraphGenerationConfig,
  SignalGenerationConfig,
  GraphWaveletTransformConfig
)


@dataclass(frozen=True)
class ABLExperimentConfig:
  seed: int
  graph_config: GraphGenerationConfig
  signal_config: SignalGenerationConfig
  gwt_config: GraphWaveletTransformConfig
  # model
  hidden_dim: int
  activation_fn: str
  conv_layer_type: str
  processor_num_layers: int
  keep_ratio: float
  lr: float
  weight_decay: float
  # training
  train_ratio: float
  val_ratio: float
  batch_size: int
  num_epochs: int
  lr_scheduler_factor: float
  lr_scheduler_patience: int
  lr_scheduler_min_lr: float
  early_stopping_patience: int
  early_stopping_min_delta: float


def default_abl_configs() -> ABLExperimentConfig:
  spectrum_upper_bound = 2.0
  scaling_kernel_bandwidth_ratio = 50
  scaling_kernel_cutoff_eigenvalue = spectrum_upper_bound / scaling_kernel_bandwidth_ratio
  num_scales = 4
  scales = torch.logspace(math.log10(1 / spectrum_upper_bound),
                          math.log10(1 / scaling_kernel_cutoff_eigenvalue),
                          num_scales).tolist()

  return ABLExperimentConfig(
    seed=42,
    graph_config=GraphGenerationConfig(
      graph_types=['erdos_renyi'],
      num_nodes_range=(200, 400),
      num_graphs_per_type=100,
      erdos_renyi_edge_prob_range=(0.005, 0.01),
      barabasi_albert_min_num_edges_per_new_node_range=(1, 2),
      sbm_num_blocks_range=(2, 5),
      sbm_intra_cluster_prob_range=(0.05, 0.15),
      sbm_inter_cluster_prob_range=(0.01, 0.03),
      return_largest_connected_component=True,
    ),
    signal_config=SignalGenerationConfig(
      signal_types=['lowpass'],
      max_num_frequency_mode_ratio=0.1,
      coeffs_mean=0.0,
      coeffs_std_range=(1.0, 2.0),
      noise_level_range=(0.05, 0.2)
    ),
    gwt_config=GraphWaveletTransformConfig(
      gwt_mode='chebyshev',
      spectrum_lower_bound=0.0,  # true minimum eigenvalue of L_sym, L_rw
      spectrum_upper_bound=spectrum_upper_bound,  # true maximum eigenvalue of L_sym, L_rw
      scaling_kernel_bandwidth_ratio=scaling_kernel_bandwidth_ratio,
      scaling_kernel_cutoff_eigenvalue=scaling_kernel_cutoff_eigenvalue,
      chebyshev_degree=100,
      num_scales=num_scales,
      num_filters=num_scales + 1,
      scales=scales
    ),
    hidden_dim=128,
    activation_fn='relu',
    conv_layer_type='SAGEConv',
    processor_num_layers=5,
    keep_ratio=0.01,
    lr=1e-3,
    weight_decay=5e-4,
    train_ratio=0.6,
    val_ratio=0.2,
    batch_size=16,
    num_epochs=1000,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=50,
    lr_scheduler_min_lr=1e-6,
    early_stopping_patience=150,
    early_stopping_min_delta=1e-4,
  )
