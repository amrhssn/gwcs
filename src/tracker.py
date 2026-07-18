import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


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


def create_run_dir(root: Path) -> Path:
  run_dir = root / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
  run_dir.mkdir(parents=True)
  return run_dir


def setup_logging(run_dir: Path) -> None:
  logging.basicConfig(level=logging.INFO,
                      format="[*] %(asctime)s | %(message)s",
                      datefmt="%Y-%m-%d %H:%M:%S",
                      handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")])


def log_run_info(run_dir: Path, dataset_dir: Path, config: Any) -> None:
  logging.info(f"Run dir: {run_dir}")
  logging.info(f"Dataset dir: {dataset_dir}")
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
