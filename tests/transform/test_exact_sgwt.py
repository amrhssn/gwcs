import torch

from transform import exact_sgwt
from eval.utils import relative_error
from tests.transform.conftest import GridGraph, HeterogeneousGridGraphs, KernelBank, MakeSignal


def test_single_graph_single_channel(
        grid_graph: GridGraph,
        kernel_bank: KernelBank,
        make_signal: MakeSignal
) -> None:
  L, eigenvectors, num_nodes = grid_graph
  kernels, *_ = kernel_bank
  x = make_signal(eigenvectors)

  W = exact_sgwt.analysis(x, L, kernels)
  x_hat = exact_sgwt.synthesis(W, L, kernels)
  assert W.shape == (num_nodes, len(kernels))
  assert x_hat.shape == x.shape

  error = relative_error(x, x_hat)
  print(f"[exact_sgwt] single_graph_single_channel relative reconstruction error = {error:.6e}")
  assert error < 1e-6


def test_single_graph_multiple_channels(
        grid_graph: GridGraph,
        kernel_bank: KernelBank,
        make_signal: MakeSignal
) -> None:
  L, eigenvectors, num_nodes = grid_graph
  kernels, *_ = kernel_bank
  num_channels = 3
  x = make_signal(eigenvectors, num_columns=num_channels)

  W = exact_sgwt.analysis(x, L, kernels)
  x_hat = exact_sgwt.synthesis(W, L, kernels)
  assert W.shape == (num_nodes, num_channels, len(kernels))
  assert x_hat.shape == x.shape

  error = relative_error(x, x_hat)
  print(f"[exact_sgwt] single_graph_multiple_channels relative reconstruction error = {error:.6e}")
  assert error < 1e-6


def test_heterogeneous_graphs_single_channel(
        heterogeneous_grid_graphs: HeterogeneousGridGraphs,
        kernel_bank: KernelBank,
        make_signal: MakeSignal
) -> None:
  L, eigenvectors_a, eigenvectors_b, num_nodes_a, num_nodes_b = heterogeneous_grid_graphs
  kernels, *_ = kernel_bank
  x_a = make_signal(eigenvectors_a, seed=1)
  x_b = make_signal(eigenvectors_b, seed=2)
  x = torch.cat([x_a, x_b], dim=0)

  W = exact_sgwt.analysis(x, L, kernels)
  x_hat = exact_sgwt.synthesis(W, L, kernels)
  assert W.shape == (num_nodes_a + num_nodes_b, len(kernels))
  assert x_hat.shape == x.shape

  error = relative_error(x, x_hat)
  print(f"[exact_sgwt] heterogeneous_graphs_single_channel relative reconstruction error = {error:.6e}")
  assert error < 1e-6


def test_heterogeneous_graphs_multiple_channels(
        heterogeneous_grid_graphs: HeterogeneousGridGraphs,
        kernel_bank: KernelBank,
        make_signal: MakeSignal
) -> None:
  L, eigenvectors_a, eigenvectors_b, num_nodes_a, num_nodes_b = heterogeneous_grid_graphs
  kernels, *_ = kernel_bank
  num_channels = 2
  x_a = make_signal(eigenvectors_a, num_columns=num_channels, seed=1)
  x_b = make_signal(eigenvectors_b, num_columns=num_channels, seed=2)
  x = torch.cat([x_a, x_b], dim=0)

  W = exact_sgwt.analysis(x, L, kernels)
  x_hat = exact_sgwt.synthesis(W, L, kernels)
  assert W.shape == (num_nodes_a + num_nodes_b, num_channels, len(kernels))
  assert x_hat.shape == x.shape

  error = relative_error(x, x_hat)
  print(f"[exact_sgwt] heterogeneous_graphs_multiple_channels relative reconstruction error = {error:.6e}")
  assert error < 1e-6


def test_shared_graph_batch_of_signals(
        grid_graph: GridGraph,
        kernel_bank: KernelBank,
        make_signal: MakeSignal
) -> None:
  L, eigenvectors, num_nodes = grid_graph
  kernels, *_ = kernel_bank
  batch_size = 4
  x_batch_first = make_signal(eigenvectors, num_columns=batch_size).permute(1, 0)  # (B, N)
  signal = x_batch_first.permute(1, 0)  # (N, B)

  W = exact_sgwt.analysis(signal, L, kernels)
  x_hat = exact_sgwt.synthesis(W, L, kernels)
  x_hat_batch_first = x_hat.permute(1, 0)
  assert W.shape == (num_nodes, batch_size, len(kernels))
  assert x_hat_batch_first.shape == x_batch_first.shape

  error = relative_error(x_batch_first, x_hat_batch_first)
  print(f"[exact_sgwt] shared_graph_batch_of_signals relative reconstruction error = {error:.6e}")
  assert error < 1e-6


def test_shared_graph_batch_of_multichannel_signals(
        grid_graph: GridGraph,
        kernel_bank: KernelBank,
        make_signal: MakeSignal
) -> None:
  L, eigenvectors, num_nodes = grid_graph
  kernels, *_ = kernel_bank
  batch_size, num_channels = 4, 2
  x_batch_first = make_signal(eigenvectors, num_columns=batch_size * num_channels) \
      .reshape(num_nodes, batch_size, num_channels).permute(1, 0, 2)  # (B, N, C)
  signal = x_batch_first.permute(1, 0, 2)  # (N, B, C)

  W = exact_sgwt.analysis(signal, L, kernels)
  x_hat = exact_sgwt.synthesis(W, L, kernels)
  x_hat_batch_first = x_hat.permute(1, 0, 2)
  assert W.shape == (num_nodes, batch_size, num_channels, len(kernels))
  assert x_hat_batch_first.shape == x_batch_first.shape

  error = relative_error(x_batch_first, x_hat_batch_first)
  print(f"[exact_sgwt] shared_graph_batch_of_multichannel_signals relative reconstruction error = {error:.6e}")
  assert error < 1e-6
