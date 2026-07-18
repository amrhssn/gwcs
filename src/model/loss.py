import torch
import torch.nn.functional as F


def reconstruction_loss(
        reconstructed: torch.Tensor,
        signal: torch.Tensor,
) -> torch.Tensor:
  loss = F.mse_loss(input=reconstructed, target=signal, reduction='mean')
  return loss