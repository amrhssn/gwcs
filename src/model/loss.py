import torch
import torch.nn.functional as F


def reconstruction_loss(
        reconstructed: torch.Tensor,
        signal: torch.Tensor,
) -> torch.Tensor:
  loss = F.mse_loss(input=reconstructed, target=signal, reduction='mean')
  return loss


def kl_loss(
        mu: torch.Tensor,
        logstd: torch.Tensor,
        batch: torch.Tensor,
) -> torch.Tensor:
  kl_per_node = -0.5 * (1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2)).sum(dim=-1)  # (N,)
  num_graphs = int(batch.max()) + 1
  graph_kl = torch.zeros(num_graphs, device=mu.device, dtype=kl_per_node.dtype)
  graph_kl.index_add_(0, batch, kl_per_node)
  return graph_kl.mean()


def elbo_loss(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logstd: torch.Tensor,
        batch: torch.Tensor,
        beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  recon = reconstruction_loss(x_hat, x)
  kl = kl_loss(mu, logstd, batch)
  return recon + beta * kl, recon, kl