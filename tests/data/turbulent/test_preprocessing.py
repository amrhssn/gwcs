from pathlib import Path
from unittest import mock

import pytest
import torch

from configs.abl import GraphWaveletTransformConfig
from data.turbulent.graph import build_turbulent_graph
from eval.utils import relative_error
from steps.turbulent_preprocessing import load_or_preprocess_turbulent_split, preprocess_turbulent_split
from transform.chebyshev_sgwt import analysis as chebyshev_analysis
from transform.chebyshev_sgwt import synthesis as chebyshev_synthesis
from transform.kernels import default_kernels

from tests.data.turbulent.conftest import HEIGHT, WIDTH

RELATIVE_ERROR_TOLERANCE = 5e-2


def test_batched_trajectory_gwt_coeffs_match_per_timestep_loop(gwt_config: GraphWaveletTransformConfig) -> None:
  graph = build_turbulent_graph(height=HEIGHT, width=WIDTH)
  spectral_kernels = default_kernels(minimum_eigenvalue=gwt_config.scaling_kernel_cutoff_eigenvalue,
                                     scales=gwt_config.scales)
  num_nodes = HEIGHT * WIDTH
  num_timesteps, num_channels = 5, 4
  random_generator = torch.Generator().manual_seed(0)
  x_traj = torch.randn(num_nodes, num_timesteps, num_channels, generator=random_generator)

  batched_coeffs = chebyshev_analysis(signal=x_traj,
                                      symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                      spectral_kernels=spectral_kernels,
                                      spectrum_lower_bound=gwt_config.spectrum_lower_bound,
                                      spectrum_upper_bound=gwt_config.spectrum_upper_bound,
                                      chebyshev_degree=gwt_config.chebyshev_degree)  # (N, T, C, F)

  looped_coeffs = torch.stack([
    chebyshev_analysis(signal=x_traj[:, t, :],
                       symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                       spectral_kernels=spectral_kernels,
                       spectrum_lower_bound=gwt_config.spectrum_lower_bound,
                       spectrum_upper_bound=gwt_config.spectrum_upper_bound,
                       chebyshev_degree=gwt_config.chebyshev_degree)  # (N, C, F)
    for t in range(num_timesteps)
  ], dim=1)  # (N, T, C, F)

  assert batched_coeffs.shape == looped_coeffs.shape
  assert torch.allclose(batched_coeffs, looped_coeffs, atol=1e-5)


def test_reconstruction_error_is_small(
        tiny_raw_split_dir: Path,
        tmp_path: Path,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  dataset = preprocess_turbulent_split(raw_split_dir=tiny_raw_split_dir, gwt_config=gwt_config,
                                       samples_dir=tmp_path / "samples", height=HEIGHT, width=WIDTH)
  for sample_idx in [0, len(dataset) // 2, len(dataset) - 1]:
    sample = dataset.get_sample(sample_idx)
    reconstructed = chebyshev_synthesis(wavelet_coeffs=sample.gwt_coeffs,
                                        symmetric_normalized_laplacian=sample.graph.symmetric_normalized_laplacian,
                                        spectral_kernels=dataset.spectral_kernels,
                                        spectrum_lower_bound=gwt_config.spectrum_lower_bound,
                                        spectrum_upper_bound=gwt_config.spectrum_upper_bound,
                                        chebyshev_degree=gwt_config.chebyshev_degree)
    error = relative_error(reconstructed, sample.x)
    assert error < RELATIVE_ERROR_TOLERANCE, f"sample {sample_idx}: relative reconstruction error {error:.4f}"


def test_saved_sample_tensors_are_not_views_into_full_trajectory_storage(
        tiny_raw_split_dir: Path,
        tmp_path: Path,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  dataset = preprocess_turbulent_split(raw_split_dir=tiny_raw_split_dir, gwt_config=gwt_config,
                                       samples_dir=tmp_path / "samples", height=HEIGHT, width=WIDTH)
  for sample_file in dataset.sample_files:
    signal = torch.load(sample_file, weights_only=False)
    for tensor in (signal.x, signal.gwt_coeffs):
      # a bare slice of the full-trajectory tensor is a view sharing the trajectory's whole storage;
      # torch.save on such a view persists that entire storage, not just the sliced-out sample -- this
      # regressed disk usage by ~num_timesteps x on the real dataset before .clone() was added.
      assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(), (
        f"{sample_file.name}: tensor storage is larger than the tensor itself -- an unsliced view "
        f"into the full trajectory's storage was persisted instead of an owned copy"
      )


def test_cache_hit_skips_recomputation_and_key_changes_with_hparams(
        tiny_raw_split_dir: Path,
        tmp_path: Path,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  cache_dir = tmp_path / "processed"

  first = load_or_preprocess_turbulent_split(dataset_dir=cache_dir, raw_split_dir=tiny_raw_split_dir,
                                             gwt_config=gwt_config, height=HEIGHT, width=WIDTH)
  assert len(list(cache_dir.iterdir())) == 1

  with mock.patch("steps.turbulent_preprocessing.chebyshev_analysis") as mocked_analysis:
    second = load_or_preprocess_turbulent_split(dataset_dir=cache_dir, raw_split_dir=tiny_raw_split_dir,
                                                gwt_config=gwt_config, height=HEIGHT, width=WIDTH)
    mocked_analysis.assert_not_called()

  assert len(second) == len(first)
  for sample_idx in range(len(first)):
    a, b = first.get_sample(sample_idx), second.get_sample(sample_idx)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.gwt_coeffs, b.gwt_coeffs)
    assert a.tcool == b.tcool
    assert a.trajectory_id == b.trajectory_id
    assert a.timestep == b.timestep

  changed_gwt_config = GraphWaveletTransformConfig(**{**gwt_config.__dict__, "chebyshev_degree": gwt_config.chebyshev_degree + 1})
  load_or_preprocess_turbulent_split(dataset_dir=cache_dir, raw_split_dir=tiny_raw_split_dir,
                                     gwt_config=changed_gwt_config, height=HEIGHT, width=WIDTH)
  assert len(list(cache_dir.iterdir())) == 2, "changing chebyshev_degree must produce a distinct cache entry"


def test_preprocessing_writes_one_semantically_named_file_per_sample(
        tiny_raw_split_dir: Path,
        tmp_path: Path,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  cache_dir = tmp_path / "processed"

  dataset = load_or_preprocess_turbulent_split(dataset_dir=cache_dir, raw_split_dir=tiny_raw_split_dir,
                                               gwt_config=gwt_config, height=HEIGHT, width=WIDTH)

  # tiny_raw_split_dir has 2 raw files x 2 trajectories x 3 timesteps -- storage must be split
  # per sample (not collapsed into one file per trajectory or one monolithic file), so a large real
  # split's memory usage stays a function of batch size rather than dataset size.
  sample_files = sorted((dataset.storage_path / "samples").glob("*.pt"))
  assert len(sample_files) == 12
  assert not (dataset.storage_path / "samples.pt").exists()

  expected_names = {
    f"turbulent_radiative_layer_tcool_{tcool}_traj{traj:03d}_t{t:03d}.pt"
    for tcool in ("0.1", "0.3") for traj in (0, 1) for t in (0, 1, 2)
  }
  assert {f.name for f in sample_files} == expected_names

  for sample_file in sample_files:
    signal = torch.load(sample_file, weights_only=False)
    assert signal.x.shape == (HEIGHT * WIDTH, 4)

  assert len(dataset) == len(sample_files)


def test_resumes_after_partial_failure_without_recomputing_completed_trajectories(
        tiny_raw_split_dir: Path,
        tmp_path: Path,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  cache_dir = tmp_path / "processed"
  call_count = 0

  def flaky_analysis(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    if call_count == 3:
      raise RuntimeError("simulated crash")
    return chebyshev_analysis(*args, **kwargs)

  with mock.patch("steps.turbulent_preprocessing.chebyshev_analysis", side_effect=flaky_analysis):
    with pytest.raises(RuntimeError, match="simulated crash"):
      load_or_preprocess_turbulent_split(dataset_dir=cache_dir, raw_split_dir=tiny_raw_split_dir,
                                         gwt_config=gwt_config, height=HEIGHT, width=WIDTH)

  cache_dirs = list(cache_dir.iterdir())
  assert len(cache_dirs) == 1, "a crash must not leave behind a duplicate cache entry on retry"
  incomplete_dir = cache_dirs[0]
  assert not (incomplete_dir / "config.json").exists()
  saved_before_crash = list((incomplete_dir / "samples").glob("*.pt"))
  assert len(saved_before_crash) == 2 * 3, (
    "the 2 trajectories computed before the 3rd crashed must already have all their per-timestep samples persisted"
  )

  with mock.patch("steps.turbulent_preprocessing.chebyshev_analysis", side_effect=flaky_analysis) as mocked:
    dataset = load_or_preprocess_turbulent_split(dataset_dir=cache_dir, raw_split_dir=tiny_raw_split_dir,
                                                 gwt_config=gwt_config, height=HEIGHT, width=WIDTH)
    assert mocked.call_count == 2, "resuming must recompute only the trajectories that hadn't been saved yet"

  assert dataset.storage_path == incomplete_dir, "resuming must reuse the incomplete cache dir, not create a new one"
  assert (incomplete_dir / "config.json").exists()
  assert len(list(cache_dir.iterdir())) == 1
  assert len(dataset) == 4 * 3
