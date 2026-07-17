from typing import List, Optional

import networkx as nx
import torch
import matplotlib.pyplot as plt


def visualize_graph_signal(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_type: str,
        pos: Optional[List[List[int]]] = None,
) -> None:
    num_nodes = x.size(0)
    G = nx.Graph()

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges
    edge_list = edge_index.t().numpy()
    G.add_edges_from(edge_list)

    # Determine node positions based on graph type
    if graph_type == 'grid':
        if pos is None:
            raise ValueError("For grid graphs, pos must not be None")
        # Use position data for grid graphs
        pos_dict = {}
        for i in range(num_nodes):
            pos_dict[i] = (pos[i][0], pos[i][1])
    else:
        # Use NetworkX layout for non-grid graphs
        # Kamada-Kawai layout produces aesthetically pleasing results
        pos_dict = nx.kamada_kawai_layout(G)

    # Create visualization
    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G,
                                   pos=pos_dict,
                                   node_size=80,
                                   node_color=x.numpy(),
                                   cmap='RdBu_r')
    nx.draw_networkx_edges(G,
                           pos=pos_dict,
                           edge_color='lightgray',
                           width=0.3)
    plt.colorbar(nodes, label='Signal Value')
    plt.title(f'{graph_type.replace("_", " ").title()} Graph')
    plt.axis('equal')
    plt.show()


def visualize_matrix(
        M: torch.Tensor,
        title: str,
        x_label: str,
        y_label: str
) -> None:
    plt.imshow(M, aspect='auto', cmap='RdBu_r')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()
    plt.show()
