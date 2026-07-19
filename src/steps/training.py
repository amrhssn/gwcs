from typing import Callable

import torch
from torch_geometric.loader import DataLoader

from model.loss import elbo_loss, reconstruction_loss
from sampler.mlis_vectorized import multilevel_importance_sampling


def forward_batch_gwcs(
        model: torch.nn.Module,
        batch,
        keep_ratio: float,
        random_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  sparsification_mask = multilevel_importance_sampling(coeffs=batch.gwt_coeffs,
                                                        keep_ratio=keep_ratio,
                                                        batch=batch.batch,
                                                        random_generator=random_generator)
  sparse_gwt_coeffs = batch.gwt_coeffs * sparsification_mask
  x_hat = model(
    sparse_gwt_coeffs=sparse_gwt_coeffs,
    sparsification_mask=sparsification_mask,
    edge_index=batch.edge_index,
    symmetric_normalized_laplacian=batch.symmetric_normalized_laplacian(),
    rescaled_symmetric_normalized_laplacian=batch.rescaled_symmetric_normalized_laplacian(),
  )
  return x_hat, sparsification_mask


def run_epoch_gwcs(
        loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
        keep_ratio: float,
        random_generator: torch.Generator | None = None,
        optimizer: torch.optim.Optimizer | None = None,
) -> float:
  model.train(optimizer is not None)
  losses = []
  with torch.set_grad_enabled(optimizer is not None):
    for batch in loader:
      batch = batch.to(device)
      x_hat, _ = forward_batch_gwcs(model=model, batch=batch, keep_ratio=keep_ratio, random_generator=random_generator)
      loss = loss_fn(x_hat, batch.x)
      if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
      losses.append(loss.item() if torch.is_tensor(loss) else loss)
  return sum(losses) / len(losses)


def forward_batch_sgae(
        model: torch.nn.Module,
        batch,
        sparsity_ratio: float,
        temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
  return model(
    x=batch.x,
    edge_index=batch.edge_index,
    batch=batch.batch,
    sparsity_ratio=sparsity_ratio,
    temperature=temperature,
  )


def run_epoch_sgae(
        loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        sparsity_ratio: float,
        temperature: float,
        kl_beta: float,
        optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
  model.train(optimizer is not None)
  losses, recons, kls = [], [], []
  with torch.set_grad_enabled(optimizer is not None):
    for batch in loader:
      batch = batch.to(device)
      x_hat, _, mu, logstd = forward_batch_sgae(model=model, batch=batch, sparsity_ratio=sparsity_ratio,
                                                temperature=temperature)
      if model.is_variational:
        loss, recon, kl = elbo_loss(x_hat, batch.x, mu, logstd, batch.batch, beta=kl_beta)
      else:
        recon = reconstruction_loss(x_hat, batch.x)
        loss, kl = recon, torch.zeros((), device=device)
      if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
      losses.append(loss.item())
      recons.append(recon.item())
      kls.append(kl.item())
  return sum(losses) / len(losses), sum(recons) / len(recons), sum(kls) / len(kls)
