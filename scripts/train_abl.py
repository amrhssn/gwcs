import time
import math
from pathlib import Path
from typing import Literal

import torch
from torch_geometric.loader import DataLoader

from data.dataset import split_dataset
from data.types import (
  GraphGenerationConfig,
  SignalGenerationConfig,
  GraphWaveletTransformConfig
)
from steps.data_generation import load_or_generate_abl_dataset
from steps.training import run_epoch, forward_batch
from model.nigwt import NeuralInverseGraphWaveletTransform
from model.loss import reconstruction_loss
from eval.utils import relative_error

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
  torch.sparse.check_sparse_tensor_invariants.disable()

  # run configs
  seed = 42
  torch.manual_seed(seed)  # nn.Module weight init has no generator hook, so this is the only way to seed it
  random_generator: torch.Generator | None = torch.Generator().manual_seed(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # graph configs
  graph_types: list[Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm']] = ['grid']
  num_nodes_range = (100, 1_000)
  num_graphs_per_type = 50
  # signal configs
  signal_types: list[Literal['lowpass', 'highpass', 'bandpass', 'R-sparse']] = ['lowpass']
  max_num_frequency_mode_ratio = 0.1
  coeffs_mean = 0.0
  coeffs_std_range = (1.0, 2.0)
  noise_level_range = (0.05, 0.2)
  # gwt configs
  gwt_mode: Literal['chebyshev', 'exact'] = 'chebyshev'
  spectrum_lower_bound = 0.0  # true minimum eigenvalue of L_sym, L_rw
  spectrum_upper_bound = 2.0  # true maximum eigenvalue of L_sym, L_rw
  scaling_kernel_bandwidth_ratio = 50
  scaling_kernel_cutoff_eigenvalue = spectrum_upper_bound / scaling_kernel_bandwidth_ratio
  chebyshev_degree = 100
  num_scales = 4
  num_filters = num_scales + 1
  scales = torch.logspace(math.log10(1 / spectrum_upper_bound),
                          math.log10(1 / scaling_kernel_cutoff_eigenvalue),
                          num_scales).tolist()
  # model
  hidden_dim = 128
  activation_fn = 'relu'
  conv_layer_type = 'SAGEConv'
  processor_num_layers = 5
  keep_ratio = 1.0
  lr = 1e-3
  weight_decay = 5e-4

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
  gwt_config = GraphWaveletTransformConfig(gwt_mode=gwt_mode,
                                           spectrum_lower_bound=spectrum_lower_bound,
                                           spectrum_upper_bound=spectrum_upper_bound,
                                           scaling_kernel_bandwidth_ratio=scaling_kernel_bandwidth_ratio,
                                           scaling_kernel_cutoff_eigenvalue=scaling_kernel_cutoff_eigenvalue,
                                           chebyshev_degree=chebyshev_degree,
                                           num_scales=num_scales,
                                           num_filters=num_filters,
                                           scales=scales)
  dataset = load_or_generate_abl_dataset(cache_root=REPO_ROOT / "datasets" / "ABL",
                                         graph_config=graph_config,
                                         signal_config=signal_config,
                                         gwt_config=gwt_config,
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

  # initializing the model, optimizer, and schedulers
  model = NeuralInverseGraphWaveletTransform(
    num_channels=dataset.num_channels,
    hidden_dim=hidden_dim,
    activation_fn=activation_fn,
    conv_layer_type=conv_layer_type,
    processor_num_layers=processor_num_layers,
    gwt_mode=dataset.configs.gwt_config.gwt_mode,
    spectral_kernels=dataset.spectral_kernels,
    spectrum_lower_bound=dataset.configs.gwt_config.spectrum_lower_bound,
    spectrum_upper_bound=dataset.configs.gwt_config.spectrum_upper_bound,
    chebyshev_degree=dataset.configs.gwt_config.chebyshev_degree
  ).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

  # training loop
  for epoch in range(num_epochs):
    epoch_start = time.perf_counter()
    train_loss = run_epoch(loader=train_loader, model=model, device=device, metric_fn=reconstruction_loss,
                           keep_ratio=keep_ratio, random_generator=random_generator, optimizer=optimizer)
    val_loss = run_epoch(loader=val_loader, model=model, device=device, metric_fn=reconstruction_loss,
                         keep_ratio=keep_ratio, random_generator=random_generator)
    epoch_time = time.perf_counter() - epoch_start
    print(f"[*] Epoch: {epoch:5d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Time: {epoch_time:.2f}s")

  # test phase
  model.eval()
  test_losses, test_errors = [], []
  with torch.no_grad():
    for batch in test_loader:
      batch = batch.to(device)
      x_hat = forward_batch(model=model, batch=batch, keep_ratio=keep_ratio, random_generator=random_generator)
      test_losses.append(reconstruction_loss(x_hat=x_hat, x=batch.x).item())
      test_errors.append(relative_error(reconstructed=x_hat, signal=batch.x))
  test_loss = sum(test_losses) / len(test_losses)
  test_error = sum(test_errors) / len(test_errors)
  print(f"[*] Test Loss: {test_loss:.4f} | Test Relative Error: {test_error:.4f}")


if __name__ == "__main__":
  main()
