from typing import Callable

import torch
import pytest
from torch_geometric.utils import get_laplacian, to_dense_adj

from data.generation.graph import generate_graph
from data.types import GraphGenerationConfig
from transform.kernels import default_kernels


GWT_SCALES = [0.5, 1.8420157493201936, 6.786044041487268]
LAMBDA_MIN = 0.0
NUM_NODES = 20

GridGraph = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
KernelBank = tuple[list[Callable[[torch.Tensor], torch.Tensor]], int]

GRAPH_CONFIG = GraphGenerationConfig(graph_types=['grid'],
                                     num_nodes_range=(NUM_NODES, NUM_NODES),
                                     num_graphs_per_type=1)


@pytest.fixture
def grid_graph() -> GridGraph:
  data = generate_graph(graph_type="grid", num_nodes=NUM_NODES, graph_config=GRAPH_CONFIG)
  num_nodes = data.metadata.num_nodes
  laplacian_index, laplacian_weight = get_laplacian(edge_index=data.edge_index,
                                                    normalization="sym",
                                                    num_nodes=num_nodes)
  laplacian = to_dense_adj(edge_index=laplacian_index,
                           edge_attr=laplacian_weight,
                           max_num_nodes=num_nodes).squeeze(0).to(torch.float64)
  eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
  return data.edge_index, laplacian, eigenvalues, eigenvectors, num_nodes


@pytest.fixture
def kernel_bank() -> KernelBank:
  kernels = default_kernels(LAMBDA_MIN, GWT_SCALES)
  return kernels, len(kernels)
