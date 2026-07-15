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
