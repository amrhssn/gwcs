from typing import Callable

import torch
import pytest
from torch_geometric.utils import get_laplacian, to_dense_adj

from data.generation.graph import generate_graph
from data.generation.signal import approximately_bandlimited
from transform.kernels import default_kernels


GWT_SCALES = [0.5, 1.8420157493201936, 6.786044041487268, 25.000000000000007]
LAMBDA_MIN = 0.0
LAMBDA_MAX = 2.0  # spectrum of L_sym always lies in [0, 2]
CHEBYSHEV_DEGREE = 30
NUM_NODES = 50

GridGraph = tuple[torch.Tensor, torch.Tensor, int]
HeterogeneousGridGraphs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]
KernelBank = tuple[list[Callable[[torch.Tensor], torch.Tensor]], float, float, int]
MakeSignal = Callable[..., torch.Tensor]


def _grid_graph_laplacian(
        requested_num_nodes: int
) -> GridGraph:
  data = generate_graph(graph_type="grid", num_nodes=requested_num_nodes)
  num_nodes = data.metadata.num_nodes
  laplacian_index, laplacian_weight = get_laplacian(edge_index=data.edge_index,
                                                    normalization="sym",
                                                    num_nodes=num_nodes)
  laplacian = to_dense_adj(edge_index=laplacian_index,
                           edge_attr=laplacian_weight,
                           max_num_nodes=num_nodes).squeeze(0).to(torch.float64)
  eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
  return laplacian, eigenvectors, num_nodes


@pytest.fixture
def grid_graph() -> GridGraph:
  return _grid_graph_laplacian(NUM_NODES)


@pytest.fixture
def heterogeneous_grid_graphs() -> HeterogeneousGridGraphs:
  laplacian_a, eigenvectors_a, num_nodes_a = _grid_graph_laplacian(NUM_NODES)
  laplacian_b, eigenvectors_b, num_nodes_b = _grid_graph_laplacian(30)
  laplacian = torch.block_diag(laplacian_a, laplacian_b)
  return laplacian, eigenvectors_a, eigenvectors_b, num_nodes_a, num_nodes_b


@pytest.fixture
def kernel_bank() -> KernelBank:
  kernels = default_kernels(LAMBDA_MIN, GWT_SCALES)
  return kernels, LAMBDA_MIN, LAMBDA_MAX, CHEBYSHEV_DEGREE


@pytest.fixture
def make_signal() -> MakeSignal:
  def _make(
          eigenvectors: torch.Tensor,
          num_columns: int = 1,
          seed: int = 0
  ) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    num_nodes = eigenvectors.shape[0]
    num_frequency_modes = max(3, num_nodes // 5)
    columns = []
    for _ in range(num_columns):
      coeffs = torch.randn(num_frequency_modes, generator=generator, dtype=eigenvectors.dtype)
      x, _ = approximately_bandlimited(coeffs=coeffs,
                                       num_frequency_modes=num_frequency_modes,
                                       signal_type="lowpass",
                                       eigenvector=eigenvectors,
                                       noise_level=0.05,
                                       random_generator=generator)
      columns.append(x.squeeze(-1))  # (N, 1) -> (N,)
    signal = torch.stack(columns, dim=-1)  # (N, num_columns)
    return signal.squeeze(-1) if num_columns == 1 else signal
  return _make
