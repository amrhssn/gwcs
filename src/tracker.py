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

from data.dataset import ABLGraphSignalDataset
from data.types import GraphWaveletTransformConfig
from visualization import (
  visualize_grid_reconstruction,
  visualize_grid_gwt_coefficients,
  visualize_grid_sparsification_mask,
  visualize_graph_reconstruction,
  visualize_graph_gwt_coefficients,
  visualize_graph_sparsification_mask,
)


@dataclass(frozen=True)
class RunMetadata:
  seed: int
  device: str
  dataset_dir: str
  config: Any


@dataclass(frozen=True)
class EpochRecord:
  epoch: int
  train_loss: float
  val_loss: float


@dataclass(frozen=True)
class RunSummary:
  best_epoch: int
  best_val_loss: float
  test_loss_mean: float
  test_loss_std: float
  test_error_mean: float
  test_error_std: float


@dataclass(frozen=True)
class TestSample:
  sample_idx: int
  signal: torch.Tensor
  reconstructed: torch.Tensor
  coeffs: torch.Tensor
  mask: torch.Tensor
  loss: float
  error: float


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


def log_model_info(model: torch.nn.Module, configs: Any, device: torch.device) -> None:
  num_params = sum(p.numel() for p in model.parameters())
  logging.info(f"Model: {model.__class__.__name__} | hidden_dim={configs.hidden_dim} | "
               f"conv_layer={configs.conv_layer_type} | processor_layers={configs.processor_num_layers} | "
               f"params={num_params:,} | device={device}")
  logging.info(f"Architecture:\n{model}")


def save_run_metadata(run_dir: Path, metadata: RunMetadata) -> None:
  (run_dir / "config.json").write_text(json.dumps(asdict(metadata), indent=2))


def save_history(run_dir: Path, history: list[EpochRecord]) -> None:
  (run_dir / "history.json").write_text(json.dumps([asdict(record) for record in history], indent=2))


def save_summary(run_dir: Path, summary: RunSummary) -> None:
  (run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))


def summarize(values: list[float]) -> tuple[float, float]:
  return statistics.mean(values), statistics.pstdev(values)


def save_figure(run_dir: Path, category: str, filename: str, fig: plt.Figure, dpi: int = 150) -> None:
  fig_dir = run_dir / "figs" / category
  fig_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(fig_dir / filename, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def save_test_figures(
        results: list[TestSample],
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
