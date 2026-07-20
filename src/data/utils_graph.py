import torch
import networkx as nx
from torch_geometric.utils import get_laplacian, to_torch_coo_tensor


def is_connected(
        edge_index: torch.Tensor,
        num_nodes: int,
) -> bool:
  G = nx.Graph()
  G.add_nodes_from(range(num_nodes))
  G.add_edges_from(edge_index.t().tolist())
  return nx.is_connected(G)


def largest_connected_component_nodes(
        edge_index: torch.Tensor,
        num_nodes: int,
) -> torch.Tensor:
  G = nx.Graph()
  G.add_nodes_from(range(num_nodes))
  G.add_edges_from(edge_index.t().tolist())
  largest = max(nx.connected_components(G), key=len)
  return torch.tensor(sorted(largest), dtype=torch.long)


def compute_symmetric_normalized_laplacian(
        edge_index: torch.Tensor,
        num_nodes: int,
) -> torch.Tensor:
  laplacian_index, laplacian_weight = get_laplacian(
    edge_index=edge_index,
    normalization="sym",
    num_nodes=num_nodes
  )
  return to_torch_coo_tensor(
    edge_index=laplacian_index,
    edge_attr=laplacian_weight,
    size=num_nodes,
    is_coalesced=False,
  )


def compute_eigendecomposition(
        symmetric_normalized_laplacian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  dense_symmetric_normalized_laplacian = (
    symmetric_normalized_laplacian.to_dense()
    if symmetric_normalized_laplacian.is_sparse
    else symmetric_normalized_laplacian
  )
  eigenvalues, eigenvectors = torch.linalg.eigh(dense_symmetric_normalized_laplacian)
  return eigenvalues, eigenvectors
