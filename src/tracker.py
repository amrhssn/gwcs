import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import torch

from compression import ResolvedCompression
from configs.abl import ABLExperimentConfig, GraphWaveletTransformConfig
from data.dataset import ABLGraphSignalDataset
from visualization import (
  visualize_grid_reconstruction,
  visualize_grid_gwt_coefficients,
  visualize_grid_sparsification_mask,
  visualize_grid_sgae_mask,
  visualize_graph_reconstruction,
  visualize_graph_gwt_coefficients,
  visualize_graph_sparsification_mask,
  visualize_graph_sgae_mask,
)


@dataclass(frozen=True)
class RunMetadata:
  seed: int
  device: str
  dataset_dir: str
  config: Any
  compression: ResolvedCompression


@dataclass(frozen=True)
class GWCSEpochRecord:
  epoch: int
  train_loss: float
  val_loss: float


@dataclass(frozen=True)
class SGAEEpochRecord:
  epoch: int
  train_loss: float
  train_recon: float
  train_kl: float
  val_loss: float
  val_recon: float
  val_kl: float


@dataclass(frozen=True)
class RunSummary:
  best_epoch: int
  best_val_loss: float
  test_loss_mean: float
  test_loss_std: float
  test_error_mean: float
  test_error_std: float
  achieved_compression_ratio_mean: float
  achieved_compression_ratio_std: float


@dataclass(frozen=True)
class GWCSTestSample:
  sample_idx: int
  signal: torch.Tensor
  reconstructed: torch.Tensor
  coeffs: torch.Tensor
  mask: torch.Tensor
  loss: float
  error: float
  compression_ratio: float


@dataclass(frozen=True)
class SGAETestSample:
  sample_idx: int
  signal: torch.Tensor
  reconstructed: torch.Tensor
  mask: torch.Tensor
  loss: float
  error: float
  compression_ratio: float


def create_run_dir(root: Path) -> Path:
  run_dir = root / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
  run_dir.mkdir(parents=True)
  return run_dir


def setup_logging(run_dir: Path) -> None:
  logging.basicConfig(level=logging.INFO,
                      format="[*] %(asctime)s | %(message)s",
                      datefmt="%Y-%m-%d %H:%M:%S",
                      handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")])


def log_run_info(run_dir: Path, dataset_dir: Path, device: torch.device, config: Any) -> None:
  logging.info(f"Run dir: {run_dir}")
  logging.info(f"Dataset dir: {dataset_dir}")
  logging.info(f"Device: {device}")
  logging.info(f"Config:\n{json.dumps(asdict(config), indent=2)}")


def _log_model_info(model: torch.nn.Module, hyperparams: dict[str, Any], device: torch.device) -> None:
  num_params = sum(p.numel() for p in model.parameters())
  hyperparams_str = " | ".join(f"{k}={v}" for k, v in hyperparams.items())
  logging.info(f"Model: {model.__class__.__name__} | {hyperparams_str} | "
               f"params={num_params:,} | device={device}")
  logging.info(f"Architecture:\n{model}")


def log_model_info_gwcs(model: torch.nn.Module, configs: ABLExperimentConfig, keep_ratio: float,
                        device: torch.device) -> None:
  _log_model_info(model=model, hyperparams={
    "hidden_dim": configs.model_config.hidden_dim,
    "conv_layer": configs.model_config.conv_layer_type,
    "processor_layers": configs.model_config.processor_num_layers,
    "keep_ratio": keep_ratio,
  }, device=device)


def log_sgae_model_info(model: torch.nn.Module, configs: ABLExperimentConfig, sparsity_ratio: float,
                        device: torch.device) -> None:
  _log_model_info(model=model, hyperparams={
    "latent_dim": configs.model_config.latent_dim,
    "conv_layer": configs.model_config.conv_layer_type,
    "num_layers": configs.model_config.num_layers,
    "sparsity_ratio": sparsity_ratio,
    "is_variational": configs.model_config.is_variational,
  }, device=device)


def log_batch_progress(epoch: int, total_epochs: int, batch_idx: int, total_batches: int, loss: float,
                       batch_time: float, prefix: str = "Batch") -> None:
  log_every = 1
  if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == total_batches:
    logging.info(f"    -> [epoch {epoch + 1}/{total_epochs}] {prefix.lower()} {batch_idx + 1}/{total_batches} "
                 f"- loss: {loss:.4f} - batch_time: {batch_time:.2f}s")


def save_run_metadata(run_dir: Path, metadata: RunMetadata) -> None:
  (run_dir / "config.json").write_text(json.dumps(asdict(metadata), indent=2))


def save_history(run_dir: Path, history: list[Any]) -> None:
  (run_dir / "history.json").write_text(json.dumps([asdict(record) for record in history], indent=2))


def save_summary(run_dir: Path, dataset_path: str, summary: RunSummary) -> None:
  (run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
  logging.info(
    "Run finished:\n"
    f"  run_id: {run_dir.name}\n"
    f"  dataset: {dataset_path}\n"
    f"  best_epoch: {summary.best_epoch}\n"
    f"  best_val_loss: {summary.best_val_loss:.4f}\n"
    f"  test_loss: {summary.test_loss_mean:.4f}±{summary.test_loss_std:.4f}\n"
    f"  test_error: {summary.test_error_mean:.4f}±{summary.test_error_std:.4f}\n"
    f"  compression_ratio: {summary.achieved_compression_ratio_mean:.4f}±"
    f"{summary.achieved_compression_ratio_std:.4f}"
  )


def summarize(values: list[float]) -> tuple[float, float]:
  return statistics.mean(values), statistics.pstdev(values)


def save_figure(run_dir: Path, category: str, filename: str, fig: plt.Figure, dpi: int = 150) -> None:
  fig_dir = run_dir / "figs" / category
  fig_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(fig_dir / filename, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def save_gwcs_test_figures(
        results: list[GWCSTestSample],
        test_dataset: ABLGraphSignalDataset,
        run_dir: Path,
        gwt_config: GraphWaveletTransformConfig,
        keep_ratio: float,
) -> None:
  total = len(results)
  log_every = max(1, total // 5)
  logging.info(f"Saving test figures: {total} samples")
  for i, result in enumerate(results):
    graph = test_dataset.get_sample(result.sample_idx).graph
    metadata = graph.metadata
    filename = f"sample_{result.sample_idx:04d}.png"

    if metadata.graph_type == 'grid':
      rows, cols = metadata.height, metadata.width
      reconstruction_fig = visualize_grid_reconstruction(reconstructed=result.reconstructed, signal=result.signal,
                                                         rows=rows, cols=cols, gwt_mode=gwt_config.gwt_mode,
                                                         num_filters=gwt_config.num_filters, keep_ratio=keep_ratio)
      coefficients_fig = visualize_grid_gwt_coefficients(coeffs=result.coeffs, signal=result.signal,
                                                         scales=gwt_config.scales, rows=rows, cols=cols,
                                                         gwt_mode=gwt_config.gwt_mode)
      mask_fig = visualize_grid_sparsification_mask(mask=result.mask, rows=rows, cols=cols, keep_ratio=keep_ratio)
    else:
      nx_graph = nx.Graph()
      nx_graph.add_nodes_from(range(metadata.num_nodes))
      nx_graph.add_edges_from(graph.edge_index.t().numpy())
      layout = nx.kamada_kawai_layout(nx_graph)
      reconstruction_fig = visualize_graph_reconstruction(reconstructed=result.reconstructed, signal=result.signal,
                                                          edge_index=graph.edge_index, layout=layout,
                                                          gwt_mode=gwt_config.gwt_mode,
                                                          num_filters=gwt_config.num_filters, keep_ratio=keep_ratio)
      coefficients_fig = visualize_graph_gwt_coefficients(coeffs=result.coeffs, signal=result.signal,
                                                          scales=gwt_config.scales, edge_index=graph.edge_index,
                                                          layout=layout, gwt_mode=gwt_config.gwt_mode)
      mask_fig = visualize_graph_sparsification_mask(mask=result.mask, edge_index=graph.edge_index, layout=layout,
                                                     keep_ratio=keep_ratio)

    save_figure(run_dir=run_dir, category="reconstruction", filename=filename, fig=reconstruction_fig)
    save_figure(run_dir=run_dir, category="coefficients", filename=filename, fig=coefficients_fig)
    save_figure(run_dir=run_dir, category="mask", filename=filename, fig=mask_fig)

    if (i + 1) % log_every == 0 or (i + 1) == total:
      logging.info(f"Figure progress: {i + 1}/{total} samples")


def save_sgae_test_figures(
        results: list[SGAETestSample],
        test_dataset: ABLGraphSignalDataset,
        run_dir: Path,
) -> None:
  total = len(results)
  log_every = max(1, total // 5)
  logging.info(f"Saving test figures: {total} samples")
  for i, result in enumerate(results):
    graph = test_dataset.get_sample(result.sample_idx).graph
    metadata = graph.metadata
    filename = f"sample_{result.sample_idx:04d}.png"

    if metadata.graph_type == 'grid':
      rows, cols = metadata.height, metadata.width
      reconstruction_fig = visualize_grid_reconstruction(reconstructed=result.reconstructed, signal=result.signal,
                                                         rows=rows, cols=cols,
                                                         compression_ratio=result.compression_ratio)
      mask_fig = visualize_grid_sgae_mask(mask=result.mask, rows=rows, cols=cols,
                                          compression_ratio=result.compression_ratio)
    else:
      nx_graph = nx.Graph()
      nx_graph.add_nodes_from(range(metadata.num_nodes))
      nx_graph.add_edges_from(graph.edge_index.t().numpy())
      layout = nx.kamada_kawai_layout(nx_graph)
      reconstruction_fig = visualize_graph_reconstruction(reconstructed=result.reconstructed, signal=result.signal,
                                                          edge_index=graph.edge_index, layout=layout,
                                                          compression_ratio=result.compression_ratio)
      mask_fig = visualize_graph_sgae_mask(mask=result.mask, edge_index=graph.edge_index, layout=layout,
                                           compression_ratio=result.compression_ratio)

    save_figure(run_dir=run_dir, category="reconstruction", filename=filename, fig=reconstruction_fig)
    save_figure(run_dir=run_dir, category="mask", filename=filename, fig=mask_fig)

    if (i + 1) % log_every == 0 or (i + 1) == total:
      logging.info(f"Figure progress: {i + 1}/{total} samples")
