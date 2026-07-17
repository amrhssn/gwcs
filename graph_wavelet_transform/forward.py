from typing import Dict, Any

import torch
from torch_geometric.data import Data


def gwt(
        x: torch.Tensor,
        G: torch.Tensor,
) -> torch.Tensor:
    w = G @ x
    return w


def batch_gwt(
        batch: Data,
        is_denoising_task: bool,
        metadata: Dict[int, Any],
        num_filters: int,
        family: str = 'mexican_hat',
) -> Data:
    x = batch.x
    if is_denoising_task:
        x_noisy = batch.x_noisy
        coeffs_noisy_list = []
    else:
        x_noisy = None
        coeffs_noisy_list = None
    ptr = batch.ptr
    num_graphs = batch.batch_size
    coeffs_list = []
    Gs = []
    for g in range(num_graphs):
        start_ptr_g = ptr[g].item()
        end_ptr_g = ptr[g + 1].item()
        num_nodes_g = end_ptr_g - start_ptr_g
        graph_index = batch.graph_index[g].item() - start_ptr_g

        # extract eigenvectors
        U = metadata[graph_index]['U']
        G = metadata[graph_index]['filter_banks'][family]
        Gs.append(G)
        # extract graph signal and edge index
        x_g = x[start_ptr_g:end_ptr_g]
        if is_denoising_task:
            x_noisy_g = x_noisy[start_ptr_g:end_ptr_g]
        else:
            x_noisy_g = None

        coeffs_g = gwt(x=x_g, G=G)
        coeffs_g = coeffs_g.reshape(num_nodes_g, num_filters)
        coeffs_list.append(coeffs_g)

        if is_denoising_task:
            coeff_noisy_g = gwt(x=x_noisy_g, G=G)
            coeff_noisy_g = coeff_noisy_g.reshape(num_nodes_g, num_filters)
            coeffs_noisy_list.append(coeff_noisy_g)
    coeffs = torch.cat(coeffs_list, dim=0)
    batch.coeffs = coeffs
    batch.Gs = Gs
    if is_denoising_task:
        coeffs_noisy = torch.cat(coeffs_noisy_list, dim=0)
        batch.coeffs_noisy = coeffs_noisy
    return batch
