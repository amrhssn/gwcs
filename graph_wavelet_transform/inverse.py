from typing import List, Dict

import torch

from graph_wavelet_transform.utils import reshape_coeffs_to_1d
from graph_wavelet_transform.chebyshev import chebyshev_gwt_synthesis, chebyshev_gwt_synthesis_fast, precompute_synthesis_data


def igwt(
        coeffs: torch.Tensor,
        G: torch.Tensor,
) -> torch.Tensor:
    if coeffs.ndim > 1:
        coeffs = reshape_coeffs_to_1d(coeffs=coeffs)
    igwt_operator = torch.linalg.inv(G.T @ G) @ G.T
    x = igwt_operator @ coeffs
    return x


def batch_igwt(
        coeffs: torch.Tensor,
        Gs: List[torch.Tensor],
) -> torch.Tensor:
    c = 0
    x_hats = []
    for G in Gs:
        num_nodes = G.shape[1]
        coeffs_g = coeffs[c:c+num_nodes, :]
        c += num_nodes
        x_hat_g = igwt(coeffs=coeffs_g, G=G)
        x_hats.append(x_hat_g.unsqueeze(-1))
    return torch.cat(x_hats, dim=0)


def batch_chebyshev_igwt(
        coeffs: torch.Tensor,
        chebyshev_data: Dict,
        batch_ptr: torch.Tensor,
) -> torch.Tensor:
    """
    Batched Chebyshev IGWT synthesis for turbulent data.

    Uses precomputed L_sym_tilde and dual Chebyshev coefficients to avoid
    redundant computation across graphs that share the same structure.

    Args:
        coeffs: (batch_total_nodes, num_filters) GWT coefficients
        chebyshev_data: Dictionary containing:
            - chebyshev_coeffs: (num_filters, K+1) Chebyshev polynomial coefficients
            - chebyshev_eigenvalue_samples: (K+1,) eigenvalue sample points
            - L_sym_sparse: (N, N) sparse symmetric Laplacian
            - lambda_min: minimum eigenvalue
            - lambda_max: maximum eigenvalue
            - chebyshev_max_degree: degree K
            - _precomputed: (optional) precomputed synthesis data from precompute_synthesis_data()
        batch_ptr: PyG batch boundaries tensor, shape (num_graphs + 1,)

    Returns:
        x_hat: (batch_total_nodes, 1) reconstructed signal
    """
    num_graphs = len(batch_ptr) - 1

    # Precompute synthesis data ONCE for the entire batch (all graphs share the same structure)
    if '_precomputed' not in chebyshev_data:
        chebyshev_data['_precomputed'] = precompute_synthesis_data(
            chebyshev_coeffs=chebyshev_data['chebyshev_coeffs'],
            chebyshev_eigenvalue_samples=chebyshev_data['chebyshev_eigenvalue_samples'],
            L_sym_sparse=chebyshev_data['L_sym_sparse'],
            lambda_min=chebyshev_data['lambda_min'],
            lambda_max=chebyshev_data['lambda_max'],
            chebyshev_max_degree=chebyshev_data['chebyshev_max_degree'],
        )

    precomputed = chebyshev_data['_precomputed']
    x_hats = []

    for i in range(num_graphs):
        start_idx = batch_ptr[i].item()
        end_idx = batch_ptr[i + 1].item()

        coeffs_i = coeffs[start_idx:end_idx, :]

        x_hat_i = chebyshev_gwt_synthesis_fast(
            approximate_gwt_coeffs=coeffs_i,
            chebyshev_coeffs=chebyshev_data['chebyshev_coeffs'],
            precomputed=precomputed,
            chebyshev_max_degree=chebyshev_data['chebyshev_max_degree'],
        )
        # NaN CHECK 3: After Chebyshev synthesis
        if not torch.isfinite(x_hat_i).all():
            print(f"\n🔥 NaN in Chebyshev IGWT for graph {i}")
            print(f"   coeffs_i finite: {torch.isfinite(coeffs_i).all()}")
            print(f"   coeffs_i stats: min={coeffs_i.min().item():.6f}, max={coeffs_i.max().item():.6f}")
            print(f"   chebyshev_coeffs finite: {torch.isfinite(chebyshev_data['chebyshev_coeffs']).all()}")

        x_hats.append(x_hat_i.unsqueeze(-1))
        del coeffs_i, x_hat_i

    result = torch.cat(x_hats, dim=0)
    del x_hats
    return result
