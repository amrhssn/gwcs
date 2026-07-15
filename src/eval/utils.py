import torch


def relative_error(
        signal: torch.Tensor,
        reconstructed: torch.Tensor
) -> float:
  return (torch.linalg.norm(reconstructed - signal) / torch.linalg.norm(signal)).item()
