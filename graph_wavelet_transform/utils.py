import torch


def reshape_coeffs_to_2d(
        coeffs: torch.Tensor,
        num_nodes: int,
        num_filters: int,
) -> torch.Tensor:
    # Reshape coefficients to (1+J, num_nodes) and transposing to get (num_nodes, 1+J)
    coeffs = coeffs.reshape(num_filters, num_nodes).transpose(1, 0)
    return coeffs


def reshape_coeffs_to_1d(coeffs: torch.Tensor) -> torch.Tensor:
    # reshaping (num_nodes, num_filters) to (num_nodes * num_filters,)
    coeffs = coeffs.transpose(1, 0).reshape(-1)
    return coeffs
