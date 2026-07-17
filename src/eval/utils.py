import torch


def relative_error(
        reconstructed: torch.Tensor,
        signal: torch.Tensor
) -> float:
  return (torch.linalg.norm(reconstructed - signal) / torch.linalg.norm(signal)).item()
