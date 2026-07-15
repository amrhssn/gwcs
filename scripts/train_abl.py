import math
from typing import Literal

import torch
from torch_geometric.loader import DataLoader

from data.dataset import split_dataset
from data.types import (
  GraphGenerationConfig,
  SignalGenerationConfig,
  GraphWaveletTransformConfig
)
from steps.data_generation import generate_abl_dataset


def main():
  torch.sparse.check_sparse_tensor_invariants.disable()

  # run configs
  seed = 42
  random_generator: torch.Generator | None = torch.Generator().manual_seed(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # graph configs
  graph_types: list[Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm']] = ['grid']
  num_nodes_range = (100, 1_000)
  num_graphs_per_type = 100
  # signal configs
  signal_types: list[Literal['lowpass', 'highpass', 'bandpass', 'R-sparse']] = ['lowpass']
  max_num_frequency_mode_ratio = 0.1
  coeffs_mean = 0.0
  coeffs_std_range = (1.0, 2.0)
  noise_level_range = (0.05, 0.2)
  # gwt configs
  spectrum_lower_bound = 0.0  # true minimum eigenvalue of L_sym, L_rw
  spectrum_upper_bound = 2.0  # true maximum eigenvalue of L_sym, L_rw
  scaling_kernel_bandwidth_ratio = 50
  scaling_kernel_cutoff_eigenvalue = spectrum_upper_bound / scaling_kernel_bandwidth_ratio
  chebyshev_degree = 30
  num_scales = 4
  num_filters = num_scales + 1
  scales = torch.logspace(math.log10(1 / spectrum_upper_bound),
                          math.log10(1 / scaling_kernel_cutoff_eigenvalue),
                          num_scales).tolist()
  # training
  train_ratio = 0.6
  val_ratio = 0.2
  batch_size = 16
  num_epochs = 100

  # initializing configs
  graph_config = GraphGenerationConfig(graph_types=graph_types,
                                       num_nodes_range=num_nodes_range,
                                       num_graphs_per_type=num_graphs_per_type)
  signal_config = SignalGenerationConfig(signal_types=signal_types,
                                         max_num_frequency_mode_ratio=max_num_frequency_mode_ratio,
                                         coeffs_mean=coeffs_mean,
                                         coeffs_std_range=coeffs_std_range,
                                         noise_level_range=noise_level_range)
  gwt_config = GraphWaveletTransformConfig(spectrum_lower_bound=spectrum_lower_bound,
                                           spectrum_upper_bound=spectrum_upper_bound,
                                           scaling_kernel_bandwidth_ratio=scaling_kernel_bandwidth_ratio,
                                           scaling_kernel_cutoff_eigenvalue=scaling_kernel_cutoff_eigenvalue,
                                           chebyshev_degree=chebyshev_degree,
                                           num_scales=num_scales,
                                           num_filters=num_filters,
                                           scales=scales)
  # generating a random dataset
  dataset = generate_abl_dataset(graph_config=graph_config,
                                 signal_config=signal_config,
                                 gwt_config=gwt_config,
                                 device=device,
                                 random_generator=random_generator)
  train_dataset, val_dataset, test_dataset = split_dataset(dataset=dataset,
                                                           train_ratio=train_ratio,
                                                           val_ratio=val_ratio,
                                                           random_generator=random_generator)
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            generator=random_generator)
  val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  # training loop
  for epoch in range(num_epochs):
    for batch in train_loader:
      batch = batch.to(device)


if __name__ == "__main__":
  main()