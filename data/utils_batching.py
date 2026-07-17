import random
from typing import List, Dict, Any, Union

import torch
from torch_geometric.data import Data, Batch

from models.sparsification_utils import weighted_random_per_scale


def get_batch(
        graph_signal_list: List[Dict[str, Any]],
        batch_size: int,
        generate_mask: bool = False,
        keep_ratio: float = 0.1,
) -> Union[Data, Batch]:
    data_list = []
    gwt_matrices = []
    L_sym_matrices = []
    num_samples = len(graph_signal_list)
    assert num_samples > batch_size
    batch_index = random.sample(range(num_samples), batch_size)
    for b in batch_index:
        graph_signal = graph_signal_list[b]
        edge_index = torch.tensor(graph_signal['graph']['edge_index'], dtype=torch.long)
        L_sym = torch.tensor(graph_signal['graph']['L_sym'], dtype=torch.float)
        x = torch.tensor(graph_signal['signal']['x'], dtype=torch.float)
        coeffs = torch.tensor(graph_signal['gwt']['coeffs'], dtype=torch.float)
        G = torch.tensor(graph_signal['gwt']['G'], dtype=torch.float)
        gwt_matrices.append(G)
        L_sym_matrices.append(L_sym)

        if generate_mask:
            # generate the random sparsification mask for this sample
            sparse_coeffs, sparse_mask = weighted_random_per_scale(coeffs=coeffs,
                                                                   keep_ratio=keep_ratio)
        else:
            sparse_mask = None

        # construct Data object
        data = Data(
            x=x,                                # (num_nodes, 1)
            coeffs=coeffs,                      # (num_nodes, num_filters)
            sparsification_mask=sparse_mask,    # (num_nodes, num_filters)
            edge_index=edge_index,
        )
        data_list.append(data)

    # Batch them together
    batch = Batch.from_data_list(data_list)

    # adding the batch GWT and L_sym matrices
    batch.gwt_matrices = gwt_matrices
    batch.L_sym = L_sym_matrices
    return batch
