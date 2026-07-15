import torch
import networkx as nx

def is_connected(
        edge_index: torch.Tensor,
        num_nodes: int,
) -> bool:
  G = nx.Graph()
  G.add_nodes_from(range(num_nodes))
  G.add_edges_from(edge_index.t().tolist())
  return nx.is_connected(G)
