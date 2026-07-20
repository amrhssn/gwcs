from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


FIELD_NAMES: tuple[str, ...] = ('density', 'pressure', 'velocity_x', 'velocity_y')


@dataclass(frozen=True)
class RawFileInfo:
  path: Path
  tcool: float
  n_trajectories: int
  n_timesteps: int
  height: int
  width: int


def list_raw_files(raw_split_dir: Path) -> list[Path]:
  return sorted(raw_split_dir.glob("*.hdf5"))


def read_file_info(path: Path) -> RawFileInfo:
  with h5py.File(path, "r") as f:
    tcool = float(f.attrs["tcool"])
    n_trajectories = int(f.attrs["n_trajectories"])
    _, n_timesteps, height, width = f["t0_fields/density"].shape
  return RawFileInfo(path=path,
                     tcool=tcool,
                     n_trajectories=n_trajectories,
                     n_timesteps=n_timesteps,
                     height=height,
                     width=width)


def read_trajectory_signal(
        path: Path,
        trajectory_id: int
) -> torch.Tensor:
  with h5py.File(path, "r") as f:
    density = np.asarray(f["t0_fields/density"][trajectory_id])    # (T, H, W)
    pressure = np.asarray(f["t0_fields/pressure"][trajectory_id])  # (T, H, W)
    velocity = np.asarray(f["t1_fields/velocity"][trajectory_id])  # (T, H, W, 2)

  n_timesteps, height, width = density.shape

  def flatten_field(field: np.ndarray) -> torch.Tensor:
    # row-major (T, H, W) -> (T, N) flatten matches PyG grid()'s node ordering i = h*width + w
    return torch.from_numpy(field.reshape(n_timesteps, height * width)).T  # (N, T)

  channels = [
    flatten_field(density),
    flatten_field(pressure),
    flatten_field(velocity[..., 0]),
    flatten_field(velocity[..., 1]),
  ]
  return torch.stack(channels, dim=-1).to(torch.float32)  # (N, T, C), channel order == FIELD_NAMES
