import dataclasses
from typing import Literal

import torch
from torch_geometric.utils import (
  grid,
  remove_self_loops,
  is_undirected,
  get_laplacian,
  to_torch_coo_tensor,
  erdos_renyi_graph as pyg_erdos_renyi_graph,
  barabasi_albert_graph as pyg_barabasi_albert_graph,
  stochastic_blockmodel_graph as pyg_stochastic_blockmodel_graph,
)

from data.types import (
  GraphGenerationConfig,
  GraphMetadata,
  GridGraphMetadata,
  ErdosRenyiGraphMetadata,
  BarabasiAlbertGraphMetadata,
  StochasticBlockModelGraphMetadata,
  Graph
)
from data.utils_graph import (
  is_connected,
  largest_connected_component_nodes,
  compute_eigendecomposition
)


def _restrict_to_largest_connected_component(
        edge_index: torch.Tensor,
        metadata: GraphMetadata,
) -> tuple[torch.Tensor, GraphMetadata]:
  kept_nodes = largest_connected_component_nodes(
    edge_index=edge_index, num_nodes=metadata.num_nodes
  ).to(edge_index.device)

  node_mapping = torch.full((metadata.num_nodes,), -1, dtype=torch.long, device=edge_index.device)
  node_mapping[kept_nodes] = torch.arange(kept_nodes.numel(), device=edge_index.device)

  edge_mask = (node_mapping[edge_index[0]] >= 0) & (node_mapping[edge_index[1]] >= 0)
  restricted_edge_index = node_mapping[edge_index[:, edge_mask]]

  metadata_updates = dict(num_nodes=kept_nodes.numel(), num_edges=restricted_edge_index.size(1))
  if isinstance(metadata, GridGraphMetadata):
    metadata_updates["pos"] = [metadata.pos[i] for i in kept_nodes.tolist()]
  metadata = dataclasses.replace(metadata, **metadata_updates)

  return restricted_edge_index, metadata


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


def erdos_renyi_graph(
        graph_type: Literal['erdos_renyi'],
        requested_num_nodes: int,
        edge_prob_range: tuple[float, float],
        device: torch.device | None = None,
        random_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ErdosRenyiGraphMetadata]:
  edge_prob = float(torch.empty(1).uniform_(edge_prob_range[0], edge_prob_range[1],
                                            generator=random_generator).item())
  edge_index = pyg_erdos_renyi_graph(num_nodes=requested_num_nodes, edge_prob=edge_prob, directed=False)
  edge_index, _ = remove_self_loops(edge_index)
  if device is not None:
    edge_index = edge_index.to(device)
  metadata = ErdosRenyiGraphMetadata(graph_type=graph_type,
                                     num_nodes=requested_num_nodes,
                                     num_edges=edge_index.size(1),
                                     is_directed=False,
                                     edge_prob_range=edge_prob_range,
                                     edge_prob=edge_prob)
  return edge_index, metadata


def barabasi_albert_graph(
        graph_type: Literal['barabasi_albert'],
        requested_num_nodes: int,
        num_edges_per_new_node_range: tuple[int, int],
        device: torch.device | None = None,
        random_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, BarabasiAlbertGraphMetadata]:
  # a new node can connect to at most requested_num_nodes - 1 existing nodes
  max_num_edges_per_new_node = min(num_edges_per_new_node_range[1], requested_num_nodes - 1)
  min_num_edges_per_new_node = min(num_edges_per_new_node_range[0], max_num_edges_per_new_node)
  num_edges_per_new_node = int(torch.randint(low=min_num_edges_per_new_node, high=max_num_edges_per_new_node + 1,
                                             size=(1,), generator=random_generator).item())
  edge_index = pyg_barabasi_albert_graph(num_nodes=requested_num_nodes, num_edges=num_edges_per_new_node)
  edge_index, _ = remove_self_loops(edge_index)
  if device is not None:
    edge_index = edge_index.to(device)
  metadata = BarabasiAlbertGraphMetadata(graph_type=graph_type,
                                         num_nodes=requested_num_nodes,
                                         num_edges=edge_index.size(1),
                                         num_edges_per_new_node=num_edges_per_new_node,
                                         num_edges_per_new_node_range=num_edges_per_new_node_range)
  return edge_index, metadata


def stochastic_block_model_graph(
        graph_type: Literal['sbm'],
        requested_num_nodes: int,
        num_blocks_range: tuple[int, int],
        intra_cluster_prob_range: tuple[float, float],
        inter_cluster_prob_range: tuple[float, float],
        device: torch.device | None = None,
        random_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, StochasticBlockModelGraphMetadata]:
  max_num_blocks = min(num_blocks_range[1], requested_num_nodes)
  min_num_blocks = min(num_blocks_range[0], max_num_blocks)
  num_blocks = int(torch.randint(low=min_num_blocks, high=max_num_blocks + 1,
                                 size=(1,), generator=random_generator).item())

  block_sizes = []
  remaining = requested_num_nodes
  for i in range(num_blocks - 1):
    max_size = remaining - (num_blocks - i - 1)
    size = int(torch.randint(low=1, high=max_size + 1, size=(1,), generator=random_generator).item())
    block_sizes.append(size)
    remaining -= size
  block_sizes.append(remaining)

  intra_block_prob = float(torch.empty(1).uniform_(intra_cluster_prob_range[0], intra_cluster_prob_range[1],
                                                   generator=random_generator).item())
  inter_block_prob = float(torch.empty(1).uniform_(inter_cluster_prob_range[0], inter_cluster_prob_range[1],
                                                   generator=random_generator).item())

  edge_probs = [[0.0] * num_blocks for _ in range(num_blocks)]
  for i in range(num_blocks):
    for j in range(i, num_blocks):
      if i == j:
        prob = intra_block_prob
      else:
        prob = float(torch.empty(1).uniform_(inter_block_prob * 0.5, inter_block_prob * 1.5,
                                             generator=random_generator).item())
      edge_probs[i][j] = prob
      edge_probs[j][i] = prob

  edge_index = pyg_stochastic_blockmodel_graph(block_sizes=block_sizes, edge_probs=edge_probs, directed=False)
  edge_index, _ = remove_self_loops(edge_index)
  if device is not None:
    edge_index = edge_index.to(device)
  metadata = StochasticBlockModelGraphMetadata(graph_type=graph_type,
                                               num_nodes=requested_num_nodes,
                                               num_edges=edge_index.size(1),
                                               num_blocks=num_blocks,
                                               block_sizes=block_sizes,
                                               edge_probs=edge_probs,
                                               intra_block_prob=intra_block_prob,
                                               inter_block_prob=inter_block_prob)
  return edge_index, metadata


def generate_graph(
        graph_type: Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm'],
        num_nodes: int,
        graph_config: GraphGenerationConfig,
        device: torch.device | None = None,
        random_generator: torch.Generator | None = None,
) -> Graph:
  if graph_type == 'grid':
    edge_index, metadata = grid_graph(graph_type=graph_type,
                                      requested_num_nodes=num_nodes,
                                      device=device)
  elif graph_type == 'erdos_renyi':
    assert graph_config.erdos_renyi_edge_prob_range is not None
    edge_index, metadata = erdos_renyi_graph(graph_type=graph_type,
                                             requested_num_nodes=num_nodes,
                                             edge_prob_range=graph_config.erdos_renyi_edge_prob_range,
                                             device=device,
                                             random_generator=random_generator)
  elif graph_type == 'barabasi_albert':
    assert graph_config.barabasi_albert_min_num_edges_per_new_node_range is not None
    edge_index, metadata = barabasi_albert_graph(
      graph_type=graph_type,
      requested_num_nodes=num_nodes,
      num_edges_per_new_node_range=graph_config.barabasi_albert_min_num_edges_per_new_node_range,
      device=device,
      random_generator=random_generator)
  elif graph_type == 'sbm':
    assert graph_config.sbm_num_blocks_range is not None
    assert graph_config.sbm_intra_cluster_prob_range is not None
    assert graph_config.sbm_inter_cluster_prob_range is not None
    edge_index, metadata = stochastic_block_model_graph(
      graph_type=graph_type,
      requested_num_nodes=num_nodes,
      num_blocks_range=graph_config.sbm_num_blocks_range,
      intra_cluster_prob_range=graph_config.sbm_intra_cluster_prob_range,
      inter_cluster_prob_range=graph_config.sbm_inter_cluster_prob_range,
      device=device,
      random_generator=random_generator)
  else:
    raise ValueError(f"Invalid graph type: {graph_type}")

  assert is_undirected(edge_index), f"Graph is not undirected: {edge_index}"

  if graph_config.return_largest_connected_component:
    edge_index, metadata = _restrict_to_largest_connected_component(edge_index=edge_index, metadata=metadata)
    assert is_connected(edge_index=edge_index, num_nodes=metadata.num_nodes), \
      f"Largest connected component of {graph_type} graph is not connected: {edge_index}"

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
    is_coalesced=False,
  )
  eigenvalues, eigenvectors = compute_eigendecomposition(
    symmetric_normalized_laplacian=symmetric_normalized_laplacian
  )

  # data construction
  graph = Graph(edge_index=edge_index,
                symmetric_normalized_laplacian=symmetric_normalized_laplacian,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                metadata=metadata)
  return graph

