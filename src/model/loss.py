import torch
import torch.nn.functional as F


def reconstruction_loss(
        x_hat: torch.Tensor,
        x: torch.Tensor,
) -> torch.Tensor:
  loss = F.mse_loss(input=x_hat, target=x, reduction='mean')
  return loss