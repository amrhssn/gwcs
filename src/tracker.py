import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


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
                      format="%(message)s",
                      handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")])


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
