from typing import Optional

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_laplacian,
    to_dense_adj
)


def compute_dense_L_sym(
        data: Data,
        normalization: Optional[str] = 'sym'
) -> torch.Tensor:
    # Get normalized Laplacian
    laplacian_index, laplacian_weight = get_laplacian(
        edge_index=data.edge_index,
        normalization=normalization,
        num_nodes=data.num_nodes
    )
    L = to_dense_adj(
        edge_index=laplacian_index,
        edge_attr=laplacian_weight,
        max_num_nodes=data.num_nodes
    ).squeeze()
    return L


def compute_sparse_L_sym(
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        normalization: Optional[str] = 'sym'
) -> torch.sparse.Tensor:
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    laplacian_index, laplacian_weight = get_laplacian(
        edge_index=edge_index,
        edge_weight=edge_weight,
        normalization=normalization,
        num_nodes=num_nodes
    )
    L_sparse = torch.sparse_coo_tensor(
        indices=laplacian_index,
        values=laplacian_weight,
        size=[num_nodes, num_nodes]
    )
    return L_sparse


def is_connected(
        edge_index: torch.Tensor,
        num_nodes: int,
) -> bool:
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())
    return nx.is_connected(G)