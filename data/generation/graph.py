from typing import Literal

import torch
from torch_geometric.utils import (
  grid,
  remove_self_loops,
  is_undirected,
  get_laplacian,
  to_dense_adj,
  to_torch_coo_tensor
)

from data.types import (
  GridGraphMetadata,
  Graph
)
from data.utils_graph import is_connected


def grid_graph(
        graph_type: Literal['grid'],
        requested_num_nodes: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None
) -> tuple[torch.Tensor, GridGraphMetadata]:
  rows = int(requested_num_nodes ** 0.5)
  cols = max(1, requested_num_nodes // rows)
  num_nodes = rows * cols
  edge_index, pos = grid(height=rows,
                         width=cols,
                         dtype=dtype,
                         device=device)
  edge_index, _ = remove_self_loops(edge_index)
  metadata = GridGraphMetadata(graph_type=graph_type,
                               num_nodes=num_nodes,
                               num_edges=edge_index.size(1),
                               height=rows,
                               width=cols,
                               pos=pos.tolist())
  return edge_index, metadata


def erdos_renyi_graph():
  is_directed = False

  metadata = {
    "graph_type": graph_type,
    "num_nodes": num_nodes,
    "num_edges": edge_index.size(1),
    "edge_prob_range": erdos_renyi_edge_prob_range,
    "edge_prob": edge_prob,
  }
  pass


def barabasi_albert_graph():
  metadata = {
    "graph_type": graph_type,
    "num_nodes": num_nodes,
    "num_edges": edge_index.size(1),
    "num_edges_per_new_node": num_edges_per_new_node,
    "num_edges_per_new_node_range": barabasi_albert_min_num_edges_per_new_node_range,
  }
  pass


def stochastic_block_model_graph():
  metadata = {
    "graph_type": graph_type,
    "num_nodes": num_nodes,
    "num_edges": edge_index.size(1),
    "num_blocks": num_blocks,
    "block_sizes": block_sizes,
    "edge_probs": edge_probs,
    "intra_block_prob": intra_block_prob,
    "inter_block_prob": inter_block_prob,
  }
  pass


def generate_graph(
        graph_type: Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm'],
        num_nodes: int,
        device: torch.device | None = None
) -> Graph:
  if graph_type == 'grid':
    edge_index, metadata = grid_graph(graph_type=graph_type,
                                      requested_num_nodes=num_nodes,
                                      device=device)
  elif graph_type == 'erdos_renyi':
    edge_index = None
    metadata = None
  elif graph_type == 'barabasi_albert':
    edge_index = None
    metadata = None
  elif graph_type == 'sbm':
    edge_index = None
    metadata = None
  else:
    raise ValueError(f"Invalid graph type: {graph_type}")

  assert is_undirected(edge_index), f"Graph is not undirected: {edge_index}"
  assert is_connected(edge_index=edge_index, num_nodes=metadata.num_nodes), f"Graph is not connected: {edge_index}"

  # symmetric normalized laplacian
  laplacian_index, laplacian_weight = get_laplacian(
    edge_index=edge_index,
    normalization="sym",
    num_nodes=metadata.num_nodes
  )
  symmetric_normalized_laplacian = to_torch_coo_tensor(
    edge_index=laplacian_index,
    edge_attr=laplacian_weight,
    size=metadata.num_nodes,
    is_coalesced=True,
  )

  # spectral decomposition of symmetric normalized laplacian
  laplacian_sym = to_dense_adj(
    edge_index=laplacian_index,
    edge_attr=laplacian_weight,
    max_num_nodes=metadata.num_nodes
  ).squeeze()
  eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_sym)

  # data construction
  graph = Graph(edge_index=edge_index,
                symmetric_normalized_laplacian=symmetric_normalized_laplacian,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                metadata=metadata)
  return graph

