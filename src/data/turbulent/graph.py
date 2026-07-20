import torch
from torch_geometric.utils import grid, remove_self_loops

from data.types import GridGraphMetadata, TurbulentGraph
from data.utils_graph import compute_symmetric_normalized_laplacian


def build_turbulent_graph(
        height: int,
        width: int,
        device: torch.device | None = None,
) -> TurbulentGraph:
  edge_index, pos = grid(height=height, width=width, device=device)
  edge_index, _ = remove_self_loops(edge_index)
  num_nodes = height * width

  metadata = GridGraphMetadata(graph_type='grid',
                               num_nodes=num_nodes,
                               num_edges=edge_index.size(1),
                               height=height,
                               width=width,
                               pos=pos.tolist())
  symmetric_normalized_laplacian = compute_symmetric_normalized_laplacian(
    edge_index=edge_index,
    num_nodes=num_nodes,
  )
  return TurbulentGraph(edge_index=edge_index,
                        symmetric_normalized_laplacian=symmetric_normalized_laplacian,
                        metadata=metadata)
