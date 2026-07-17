import random
from typing import List, Dict, Any

import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    erdos_renyi_graph,
    barabasi_albert_graph,
    grid,
    stochastic_blockmodel_graph,
    is_undirected,
    remove_self_loops,
)

from data.data_types import GraphGenerationConfig
from data.utils_graph import (
    compute_dense_L_sym,
    is_connected
)


def generate_single_random_graph(
        graph_type: str,
        num_nodes_range,
        erdos_renyi_edge_prob_range,
        barabasi_albert_min_num_edges_per_new_node_range,
        sbm_num_blocks_range,
        sbm_intra_cluster_prob_range,
        sbm_inter_cluster_prob_range
):
    graph_data = None
    max_attempts = 100
    for attempt in range(max_attempts):
        num_nodes = random.randint(a=num_nodes_range[0], b=num_nodes_range[1])
        if graph_type == "grid":
            r = int(num_nodes ** 0.5)
            c = max(1, num_nodes // r)
            num_nodes = r * c
            edge_index, pos = grid(height=r,
                                   width=c)
            metadata = {
                "graph_type": graph_type,
                "num_nodes": num_nodes,
                "num_edges": edge_index.size(1),
                "height": r,
                "width": c,
                "pos": pos.tolist(),
            }
        elif graph_type == "erdos_renyi":
            is_directed = False
            edge_prob = random.uniform(a=erdos_renyi_edge_prob_range[0],
                                       b=erdos_renyi_edge_prob_range[1])
            edge_index = erdos_renyi_graph(num_nodes=num_nodes,
                                           edge_prob=edge_prob,
                                           directed=is_directed)
            metadata = {
                "graph_type": graph_type,
                "num_nodes": num_nodes,
                "num_edges": edge_index.size(1),
                "is_directed": is_directed,
                "edge_prob_range": erdos_renyi_edge_prob_range,
                "edge_prob": edge_prob,
            }
        elif graph_type == "barabasi_albert":
            # In a Barabasi-Albert graph, when a new node is added,
            # it can connect to at most existing nodes (it can't connect to itself)
            num_edges_per_new_node = random.randint(
                a=barabasi_albert_min_num_edges_per_new_node_range[0],
                b=barabasi_albert_min_num_edges_per_new_node_range[1])
            # Ensure we don't exceed num_nodes - 1
            num_edges_per_new_node = min(num_edges_per_new_node, num_nodes - 1)
            edge_index = barabasi_albert_graph(num_nodes=num_nodes,
                                               num_edges=num_edges_per_new_node)
            metadata = {
                "graph_type": graph_type,
                "num_nodes": num_nodes,
                "num_edges": edge_index.size(1),
                "num_edges_per_new_node": num_edges_per_new_node,
                "num_edges_per_new_node_range": barabasi_albert_min_num_edges_per_new_node_range,
            }

        elif graph_type == "sbm":
            num_blocks = random.randint(sbm_num_blocks_range[0],
                                        sbm_num_blocks_range[1] + 1)
            block_sizes = []
            remaining = num_nodes
            for i in range(num_blocks - 1):
                size = random.randint(1, max(1, remaining - (num_blocks - i - 1)))
                block_sizes.append(size)
                remaining -= size
            block_sizes.append(remaining)
            intra_block_prob = random.uniform(sbm_intra_cluster_prob_range[0],
                                              sbm_intra_cluster_prob_range[1])
            inter_block_prob = random.uniform(sbm_inter_cluster_prob_range[0],
                                              sbm_inter_cluster_prob_range[1])
            edge_probs = []
            for i in range(num_blocks):
                row = []
                for j in range(num_blocks):
                    if i == j:
                        # Diagonal: intra-block probability
                        row.append(intra_block_prob)
                    elif i < j:
                        # Upper triangle: random inter-block probability
                        prob = random.uniform(inter_block_prob * 0.5, inter_block_prob * 1.5)
                        row.append(prob)
                    else:
                        # Lower triangle: mirror upper triangle for symmetry
                        row.append(edge_probs[j][i])
                edge_probs.append(row)
            edge_index = stochastic_blockmodel_graph(
                block_sizes=block_sizes,
                edge_probs=edge_probs,
                directed=False
            )
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
        else:
            raise ValueError(f"Invalid graph type: {graph_type}")

        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)

        assert is_undirected(edge_index)

        # Assert if graph is connected
        if is_connected(edge_index=edge_index, num_nodes=num_nodes):
            # spectral decomposition
            data = Data(edge_index=edge_index, num_nodes=num_nodes)
            L_sym = compute_dense_L_sym(data=data)
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

            graph_data = {
                "edge_index": edge_index.tolist(),
                "L_sym": L_sym.tolist(),
                "eigenvalues": eigenvalues.tolist(),
                "eigenvectors": eigenvectors.tolist(),
                "metadata": metadata,
            }
            break
        elif attempt == max_attempts - 1:
            raise RuntimeError(f"Failed to generate connected graph after {max_attempts} attempts")
    return graph_data


def random_graph_generation_step(
        graph_config: GraphGenerationConfig,
) -> List[Dict[str, Any]]:
    """
    Generate random graphs of different types based on the provided configuration.

    This function creates graphs of specified types (grid, Erdos-Renyi, or Barabasi-Albert)
    with random parameters sampled from the given ranges. Each graph is represented as
    an edge index with associated metadata containing graph properties.

    Parameters
    ----------
    graph_config : GraphGenerationConfig
        Configuration object containing graph generation parameters including:
        - graph_types: types of graphs to generate
        - num_nodes_range: range for number of nodes
        - num_graphs_per_type: number of graphs to generate per type
        - barabasi_albert_min_num_edges_per_new_node_range: edges per new node for BA graphs
        - erdos_renyi_edge_prob_range: edge probability range for ER graphs

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries, each containing:
        - 'edge_index': list representation of graph edges
        - 'metadata': dictionary with graph properties (type, nodes, edges, and type-specific parameters)
    """
    all_graphs = []
    for graph_type in graph_config.graph_types:
        for graph_idx in range(graph_config.num_graphs_per_type):
            graph_data = generate_single_random_graph(
                graph_type=graph_type,
                num_nodes_range=graph_config.num_nodes_range,
                erdos_renyi_edge_prob_range=graph_config.erdos_renyi_edge_prob_range,
                barabasi_albert_min_num_edges_per_new_node_range=graph_config.barabasi_albert_min_num_edges_per_new_node_range,
                sbm_num_blocks_range=graph_config.sbm_num_blocks_range,
                sbm_intra_cluster_prob_range=graph_config.sbm_intra_cluster_prob_range,
                sbm_inter_cluster_prob_range=graph_config.sbm_inter_cluster_prob_range
            )
            all_graphs.append(graph_data)
    return all_graphs
