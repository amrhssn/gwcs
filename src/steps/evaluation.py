import logging
from typing import Callable

import torch
from torch_geometric.loader import DataLoader

from steps.training import forward_batch
from tracker import TestSample


def run_test_epoch(
        loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        loss_fn: Callable[..., torch.Tensor],
        error_fn: Callable[..., float],
        keep_ratio: float,
        random_generator: torch.Generator | None = None,
) -> list[TestSample]:
  model.eval()
  results = []
  total_batches = len(loader)
  log_every = max(1, total_batches // 5)
  logging.info(f"Running test: {total_batches} batches")
  with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
      batch = batch.to(device)
      x_hat, mask = forward_batch(model=model, batch=batch, keep_ratio=keep_ratio, random_generator=random_generator)
      for i in range(batch.num_graphs):
        start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()
        signal = batch.x[start:end].cpu()
        reconstructed = x_hat[start:end].cpu()
        results.append(TestSample(
          sample_idx=int(batch.sample_idx[i]),
          signal=signal,
          reconstructed=reconstructed,
          coeffs=batch.gwt_coeffs[start:end].cpu(),
          mask=mask[start:end].cpu(),
          loss=loss_fn(reconstructed=reconstructed, signal=signal).item(),
          error=error_fn(reconstructed=reconstructed, signal=signal),
        ))
      if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == total_batches:
        logging.info(f"Test progress: {batch_idx + 1}/{total_batches} batches")
  return results
