import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.loader import DataLoader

import tracker
from data.dataset import ABLGraphSignalDataset
from data.types import GraphWaveletTransformConfig
from steps.training import forward_batch
from visualization import (
  visualize_grid_reconstruction,
  visualize_grid_gwt_coefficients,
  visualize_grid_sparsification_mask,
)


@dataclass(frozen=True)
class TestSample:
  sample_idx: int
  signal: torch.Tensor
  reconstructed: torch.Tensor
  coeffs: torch.Tensor
  mask: torch.Tensor
  loss: float
  error: float


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
  with torch.no_grad():
    for batch in loader:
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
  return results


def save_test_figures(
        results: list[TestSample],
        test_dataset: ABLGraphSignalDataset,
        run_dir: Path,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  for result in results:
    graph_metadata = test_dataset.get_sample(result.sample_idx).graph.metadata
    if graph_metadata.graph_type != 'grid':
      logging.warning(f"[*] Skipping figures for sample {result.sample_idx}: "
                      f"visualization not implemented for graph_type={graph_metadata.graph_type!r}")
      continue

    rows, cols = graph_metadata.height, graph_metadata.width
    filename = f"sample_{result.sample_idx:04d}.png"

    tracker.save_figure(run_dir=run_dir, category="reconstruction", filename=filename,
                        fig=visualize_grid_reconstruction(reconstructed=result.reconstructed, signal=result.signal,
                                                           rows=rows, cols=cols, gwt_mode=gwt_config.gwt_mode))
    tracker.save_figure(run_dir=run_dir, category="coefficients", filename=filename,
                        fig=visualize_grid_gwt_coefficients(coeffs=result.coeffs, signal=result.signal,
                                                             scales=gwt_config.scales, rows=rows, cols=cols,
                                                             gwt_mode=gwt_config.gwt_mode))
    tracker.save_figure(run_dir=run_dir, category="mask", filename=filename,
                        fig=visualize_grid_sparsification_mask(mask=result.mask, rows=rows, cols=cols))
