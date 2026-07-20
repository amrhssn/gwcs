from pathlib import Path

from data.turbulent.raw import FIELD_NAMES, read_file_info, read_trajectory_signal

from tests.data.turbulent.conftest import (
  HEIGHT,
  WIDTH,
  density_value,
  pressure_value,
  velocity_x_value,
  velocity_y_value,
  write_synthetic_raw_file,
)


def test_read_file_info_matches_written_attrs(tmp_path: Path) -> None:
  path = tmp_path / "sample.hdf5"
  write_synthetic_raw_file(path, tcool=0.42, n_trajectories=3, n_timesteps=5)

  info = read_file_info(path)
  assert info.tcool == 0.42
  assert info.n_trajectories == 3
  assert info.n_timesteps == 5
  assert info.height == HEIGHT
  assert info.width == WIDTH


def test_read_trajectory_signal_channel_order_and_values(tmp_path: Path) -> None:
  path = tmp_path / "sample.hdf5"
  n_trajectories, n_timesteps = 2, 4
  write_synthetic_raw_file(path, tcool=0.1, n_trajectories=n_trajectories, n_timesteps=n_timesteps)

  trajectory_id = 1
  signal = read_trajectory_signal(path, trajectory_id)
  num_nodes = HEIGHT * WIDTH
  assert signal.shape == (num_nodes, n_timesteps, len(FIELD_NAMES))
  assert FIELD_NAMES == ('density', 'pressure', 'velocity_x', 'velocity_y')

  # spot-check several (h, w, t) coordinates -- a transpose/reshape bug would still produce a
  # correctly-shaped tensor, so only checking values at known node/time coordinates catches it.
  for h, w, t in [(0, 0, 0), (2, 3, 1), (HEIGHT - 1, WIDTH - 1, n_timesteps - 1), (1, 0, 2)]:
    node_id = h * WIDTH + w
    assert signal[node_id, t, 0].item() == density_value(trajectory_id, t, h, w)
    assert signal[node_id, t, 1].item() == pressure_value(trajectory_id, t, h, w)
    assert signal[node_id, t, 2].item() == velocity_x_value(trajectory_id, t, h, w)
    assert signal[node_id, t, 3].item() == velocity_y_value(trajectory_id, t, h, w)
