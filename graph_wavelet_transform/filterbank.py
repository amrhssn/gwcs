from typing import List

import torch


def sgwt_transpose(
        G: torch.Tensor,
        num_nodes: int
) -> torch.Tensor:
    num_blocks = G.shape[0] // num_nodes
    # Reshape to separate the blocks: [(J+1), N, N]
    G_blocks = G.view(num_blocks, num_nodes, num_nodes)
    # Transpose each block: [(J+1), N, N] -> [(J+1), N, N]
    G_blocks_transposed = G_blocks.transpose(-2, -1)
    # Reshape back to the original format: [(J+1)*N, N]
    G_transposed = G_blocks_transposed.reshape(-1, num_nodes)
    return G_transposed


def scaling_kernel(
        eigenvalues: torch.Tensor,
        minimum_eigenvalue: float,
        minimum_eigenvalue_coeff: float = 50.0,
        exp_coeff: float = 0.4
) -> torch.Tensor:
    tau = minimum_eigenvalue_coeff * minimum_eigenvalue
    return exp_coeff * torch.exp(-tau * eigenvalues)


def mexican_hat_kernel(
        eigenvalues: torch.Tensor,
        scale: float
) -> torch.Tensor:
    return scale * eigenvalues * torch.exp(-scale * eigenvalues)


def scaling_filter(
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        minimum_eigenvalue: float,
) -> torch.Tensor:
    H_diag = scaling_kernel(eigenvalues=eigenvalues, minimum_eigenvalue=minimum_eigenvalue)
    H_scaling = eigenvectors @ torch.diag(H_diag) @ eigenvectors.T
    return H_scaling


def compute_mexican_hat_filters(
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        scales: List[float]
) -> torch.Tensor:
    # for each scale s_j:
    # g_s(lambda) = s * lambda * exp(-s * lambda) for each eigenvalue
    # G_s = U @ diag(g_s(Lambda)) @ U^T  (N x N matrix)
    device = eigenvalues.device
    dtype = eigenvalues.dtype
    num_nodes = eigenvalues.size(0)
    num_scales = len(scales)
    Gs_mexican_hat = torch.zeros(num_scales * num_nodes, num_nodes, device=device, dtype=dtype)
    for j, scale in enumerate(scales):
        G_responses = mexican_hat_kernel(eigenvalues=eigenvalues,
                                         scale=scale)
        G_diag = torch.diag(G_responses)
        G_s = eigenvectors @ G_diag @ eigenvectors.T  # [N, N]
        start_idx = j * num_nodes
        end_idx = (j + 1) * num_nodes
        Gs_mexican_hat[start_idx:end_idx, :] = G_s
    return Gs_mexican_hat


def construct_sgwt_filterbank(
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        scaling_function_minimum_eigenvalue: float,
        mexican_hat_filter_scales: List[float]
) -> torch.Tensor:
    num_nodes = eigenvalues.size(0)
    Gs_mexican_hat = compute_mexican_hat_filters(eigenvalues=eigenvalues,
                                                 eigenvectors=eigenvectors,
                                                 scales=mexican_hat_filter_scales)
    H_scaling = scaling_filter(eigenvalues=eigenvalues,
                               eigenvectors=eigenvectors,
                               minimum_eigenvalue=scaling_function_minimum_eigenvalue)
    G = torch.cat([H_scaling, Gs_mexican_hat], dim=0)
    G = sgwt_transpose(G=G, num_nodes=num_nodes)
    return G
