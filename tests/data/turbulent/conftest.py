import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from configs.common import GraphWaveletTransformConfig

HEIGHT = 4
WIDTH = 6
SPECTRUM_LOWER_BOUND = 0.0
SPECTRUM_UPPER_BOUND = 2.0
CHEBYSHEV_DEGREE = 30
NUM_SCALES = 2


def density_value(trajectory_id: int, t: int, h, w) -> float:
  return 1000.0 * trajectory_id + 10.0 * t + h + w


def pressure_value(trajectory_id: int, t: int, h, w) -> float:
  return 2000.0 * trajectory_id + 20.0 * t + h + w


def velocity_x_value(trajectory_id: int, t: int, h, w) -> float:
  return 3000.0 * trajectory_id + 30.0 * t + h + w


def velocity_y_value(trajectory_id: int, t: int, h, w) -> float:
  return 4000.0 * trajectory_id + 40.0 * t + h + w


def write_synthetic_raw_file(
        path: Path,
        tcool: float,
        n_trajectories: int,
        n_timesteps: int,
        height: int = HEIGHT,
        width: int = WIDTH,
) -> None:
  hh, ww = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
  density = np.zeros((n_trajectories, n_timesteps, height, width), dtype=np.float64)
  pressure = np.zeros_like(density)
  velocity = np.zeros((n_trajectories, n_timesteps, height, width, 2), dtype=np.float64)
  for trajectory_id in range(n_trajectories):
    for t in range(n_timesteps):
      density[trajectory_id, t] = density_value(trajectory_id, t, hh, ww)
      pressure[trajectory_id, t] = pressure_value(trajectory_id, t, hh, ww)
      velocity[trajectory_id, t, :, :, 0] = velocity_x_value(trajectory_id, t, hh, ww)
      velocity[trajectory_id, t, :, :, 1] = velocity_y_value(trajectory_id, t, hh, ww)

  with h5py.File(path, "w") as f:
    f.attrs["tcool"] = tcool
    f.attrs["n_trajectories"] = n_trajectories
    f.attrs["n_spatial_dims"] = 2
    f.create_dataset("t0_fields/density", data=density)
    f.create_dataset("t0_fields/pressure", data=pressure)
    f.create_dataset("t1_fields/velocity", data=velocity)


@pytest.fixture
def gwt_config() -> GraphWaveletTransformConfig:
  scaling_kernel_bandwidth_ratio = 50
  scaling_kernel_cutoff_eigenvalue = SPECTRUM_UPPER_BOUND / scaling_kernel_bandwidth_ratio
  scales = torch.logspace(math.log10(1 / SPECTRUM_UPPER_BOUND),
                          math.log10(1 / scaling_kernel_cutoff_eigenvalue),
                          NUM_SCALES).tolist()
  return GraphWaveletTransformConfig(gwt_mode="chebyshev",
                                     spectrum_lower_bound=SPECTRUM_LOWER_BOUND,
                                     spectrum_upper_bound=SPECTRUM_UPPER_BOUND,
                                     scaling_kernel_bandwidth_ratio=scaling_kernel_bandwidth_ratio,
                                     scaling_kernel_cutoff_eigenvalue=scaling_kernel_cutoff_eigenvalue,
                                     chebyshev_degree=CHEBYSHEV_DEGREE,
                                     num_scales=NUM_SCALES,
                                     num_filters=NUM_SCALES + 1,
                                     scales=scales)


@pytest.fixture
def tiny_raw_split_dir(tmp_path: Path) -> Path:
  raw_split_dir = tmp_path / "raw" / "train"
  raw_split_dir.mkdir(parents=True)
  write_synthetic_raw_file(raw_split_dir / "turbulent_radiative_layer_tcool_0.1.hdf5",
                           tcool=0.1, n_trajectories=2, n_timesteps=3)
  write_synthetic_raw_file(raw_split_dir / "turbulent_radiative_layer_tcool_0.3.hdf5",
                           tcool=0.3, n_trajectories=2, n_timesteps=3)
  return raw_split_dir
