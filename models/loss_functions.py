from typing import List, Optional

import torch
import torch.nn.functional as F

from graph_wavelet_transform.forward import gwt
from graph_wavelet_transform.utils import reshape_coeffs_to_2d


# ================ objective functions ===============
def compute_loss(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        coeffs: torch.Tensor,
        sparsification_mask: torch.Tensor,
        gwt_matrices: List[torch.Tensor],
        L_sym: List[torch.Tensor],
        gamma_coeff_consistency: float,
        gamma_regularization: float,
        batch_ptr: Optional[torch.Tensor] = None,
):
    # ================= Reconstruction loss =================
    recon_loss = F.mse_loss(input=x_hat, target=x, reduction='mean')

    # ================= Coefficients consistency loss =================
    coeffs_consistency_loss = torch.tensor(0.0, device=x_hat.device, requires_grad=False)
    # if batch_ptr is not None:
    #     # Batched processing
    #     num_graphs = len(batch_ptr) - 1
    #     coeffs_consistency_losses = []
    #
    #     for g in range(num_graphs):
    #         start_idx = batch_ptr[g].item()
    #         end_idx = batch_ptr[g + 1].item()
    #
    #         # Extract per-graph data
    #         x_hat_g = x_hat[start_idx:end_idx]
    #         coeffs_g = coeffs[start_idx:end_idx]
    #         mask_g = sparsification_mask[start_idx:end_idx]
    #         G_g = gwt_matrices[g]
    #
    #         # Compute GWT of reconstructed signal
    #         coeffs_hat_g = gwt(x=x_hat_g, G=G_g)
    #         num_nodes_g = end_idx - start_idx
    #         num_filters = coeffs_g.shape[1]
    #         # TODO: ensure the correctness of reshaping
    #         # coeffs_hat_g = coeffs_hat_g.reshape(num_nodes_g, num_filters)
    #         coeffs_hat_g = reshape_coeffs_to_2d(coeffs=coeffs_hat_g,
    #                                             num_nodes=num_nodes_g,
    #                                             num_filters=num_filters)
    #
    #         # Apply mask and compute loss
    #         mask_indices_g = mask_g.bool()
    #         if mask_indices_g.any():
    #             loss_g = F.mse_loss(
    #                 input=coeffs_hat_g[mask_indices_g],
    #                 target=coeffs_g[mask_indices_g],
    #                 reduction='mean'
    #             )
    #             coeffs_consistency_losses.append(loss_g)
    #
    #     # Average across graphs
    #     if len(coeffs_consistency_losses) > 0:
    #         coeffs_consistency_loss = torch.stack(coeffs_consistency_losses).mean()
    #     else:
    #         coeffs_consistency_loss = torch.tensor(0.0, device=x_hat.device)
    # else:
    #     # Single graph processing (original implementation)
    #     num_nodes, num_filters = coeffs.shape
    #     coeffs_hat = gwt(x_hat, gwt_matrices[0])
    #     coeffs_hat = coeffs_hat.reshape(num_nodes, num_filters)
    #
    #     mask_indices = sparsification_mask.bool()
    #     coeffs_consistency_loss = F.mse_loss(
    #         input=coeffs_hat[mask_indices],
    #         target=coeffs[mask_indices],
    #         reduction='mean'
    #     )

    # ================= regularization loss =================
    regularization_loss = torch.tensor(0.0, device=x_hat.device, requires_grad=False)
    # if batch_ptr is not None:
    #     num_graphs = len(batch_ptr) - 1
    #     # Batched regularization - process per graph if L_sym is batched
    #     if isinstance(L_sym, list):
    #         reg_losses = []
    #         for g in range(num_graphs):
    #             start_idx = batch_ptr[g].item()
    #             end_idx = batch_ptr[g + 1].item()
    #             x_hat_g = x_hat[start_idx:end_idx]
    #             L_sym_g = L_sym[g]
    #             reg_loss_g = torch.dot(x_hat_g.flatten(), (L_sym_g @ x_hat_g).flatten())
    #             reg_losses.append(reg_loss_g)
    #         regularization_loss = torch.stack(reg_losses).mean()
    #     else:
    #         # Single Laplacian for entire batch
    #         regularization_loss = torch.dot(x_hat.flatten(), (L_sym @ x_hat).flatten())
    # else:
    #     # Single graph
    #     regularization_loss = torch.dot(x_hat.flatten(), (L_sym @ x_hat).flatten())

    # adding the losses
    loss = recon_loss + gamma_coeff_consistency * coeffs_consistency_loss + gamma_regularization * regularization_loss
    return loss, recon_loss, coeffs_consistency_loss, regularization_loss
