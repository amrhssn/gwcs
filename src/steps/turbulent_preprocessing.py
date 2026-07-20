import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch

from configs.common import GraphWaveletTransformConfig
from data.turbulent.graph import build_turbulent_graph
from data.turbulent.raw import FIELD_NAMES, list_raw_files, read_file_info, read_trajectory_signal
from data.turbulent.dataset import TurbulentGraphSignalDataset
from data.types import TurbulentFieldMetadata, TurbulentSampleSignal
from transform.kernels import default_kernels
from transform.chebyshev_sgwt import analysis as chebyshev_analysis


def _sample_file_name(
        path: Path,
        trajectory_id: int,
        timestep: int,
) -> str:
  return f"{path.stem}_traj{trajectory_id:03d}_t{timestep:03d}.pt"


def _cache_key(gwt_config: GraphWaveletTransformConfig, height: int, width: int) -> str:
  payload = {"gwt_config": asdict(gwt_config), "height": height, "width": width}
  config_json = json.dumps(payload, sort_keys=True)
  return hashlib.md5(config_json.encode()).hexdigest()[:8]


def _short_config_str(gwt_config: GraphWaveletTransformConfig, height: int, width: int) -> str:
  return f"h={height}_w={width}_gwt={gwt_config.gwt_mode}_deg={gwt_config.chebyshev_degree}"


def _load_gwt_config(path: Path) -> GraphWaveletTransformConfig:
  data = json.loads(path.read_text())
  return GraphWaveletTransformConfig(**data["gwt_config"])


def preprocess_turbulent_split(
        raw_split_dir: Path,
        gwt_config: GraphWaveletTransformConfig,
        samples_dir: Path,
        height: int = 128,
        width: int = 384,
) -> TurbulentGraphSignalDataset:
  graph = build_turbulent_graph(height=height, width=width)
  spectral_kernels = default_kernels(minimum_eigenvalue=gwt_config.scaling_kernel_cutoff_eigenvalue,
                                     scales=gwt_config.scales)
  samples_dir.mkdir(parents=True, exist_ok=True)

  raw_files = list_raw_files(raw_split_dir)
  file_infos = [read_file_info(path) for path in raw_files]
  total_trajectories = sum(info.n_trajectories for info in file_infos)
  logging.info(f"Preprocessing turbulent split: {raw_split_dir} "
              f"({len(raw_files)} raw files, {total_trajectories} trajectories total)")

  sample_files: list[Path] = []
  trajectory_num = 0
  for path, info in zip(raw_files, file_infos):
    for trajectory_id in range(info.n_trajectories):
      trajectory_num += 1
      trajectory_sample_files = [samples_dir / _sample_file_name(path, trajectory_id, t)
                                 for t in range(info.n_timesteps)]

      if all(f.exists() for f in trajectory_sample_files):
        sample_files.extend(trajectory_sample_files)
        logging.info(f"[{trajectory_num}/{total_trajectories}] skipping already-cached "
                    f"trajectory {trajectory_id} of {path.name}")
        continue

      x_traj = read_trajectory_signal(path, trajectory_id)  # (N, T, C)
      gwt_coeffs_traj = chebyshev_analysis(signal=x_traj,
                                           symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                           spectral_kernels=spectral_kernels,
                                           spectrum_lower_bound=gwt_config.spectrum_lower_bound,
                                           spectrum_upper_bound=gwt_config.spectrum_upper_bound,
                                           chebyshev_degree=gwt_config.chebyshev_degree)  # (N, T, C, F)

      for t, sample_file in enumerate(trajectory_sample_files):
        signal = TurbulentSampleSignal(x=x_traj[:, t, :].clone(),
                                       gwt_coeffs=gwt_coeffs_traj[:, t, :, :].clone(),
                                       tcool=info.tcool,
                                       trajectory_id=trajectory_id,
                                       timestep=t)
        tmp_sample_file = sample_file.with_suffix(".pt.tmp")
        torch.save(signal, tmp_sample_file)
        tmp_sample_file.rename(sample_file)  # atomic, so a killed process never leaves a corrupt cache file
      sample_files.extend(trajectory_sample_files)
      logging.info(f"[{trajectory_num}/{total_trajectories}] saved {info.n_timesteps} samples for "
                  f"trajectory {trajectory_id} of {path.name}")

  return TurbulentGraphSignalDataset(sample_files=sample_files,
                                     graph=graph,
                                     field_metadata=TurbulentFieldMetadata(FIELD_NAMES),
                                     gwt_config=gwt_config)


def load_or_preprocess_turbulent_split(
        dataset_dir: Path,
        raw_split_dir: Path,
        gwt_config: GraphWaveletTransformConfig,
        height: int = 128,
        width: int = 384,
        debug: bool = False,
) -> TurbulentGraphSignalDataset:
  dataset_dir.mkdir(parents=True, exist_ok=True)
  cache_key = _cache_key(gwt_config, height, width)

  matches = sorted(dataset_dir.glob(f"*_{cache_key}"))
  complete_matches = [m for m in matches if (m / "config.json").exists()]
  if complete_matches:
    matched_cache_dir = complete_matches[-1]
    logging.info(f"Using cached turbulent split: {matched_cache_dir.name}")
    graph = build_turbulent_graph(height=height, width=width)
    sample_files = sorted((matched_cache_dir / "samples").glob("*.pt"))
    if debug:
      sample_files = sample_files[:10]
    cached_gwt_config = _load_gwt_config(matched_cache_dir / "config.json")
    dataset = TurbulentGraphSignalDataset(sample_files=sample_files,
                                          graph=graph,
                                          field_metadata=TurbulentFieldMetadata(FIELD_NAMES),
                                          gwt_config=cached_gwt_config)
    dataset.storage_path = matched_cache_dir
    return dataset

  incomplete_matches = [m for m in matches if m not in complete_matches]
  if incomplete_matches:
    new_cache_dir = incomplete_matches[-1]
    logging.info(f"Resuming incomplete turbulent split cache: {new_cache_dir.name}")
  else:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    short_config = _short_config_str(gwt_config=gwt_config,
                                     height=height,
                                     width=width)
    new_cache_dir = dataset_dir / f"{timestamp}_{short_config}_{cache_key}"
    logging.info(f"No cached turbulent split found (key={cache_key}); preprocessing {raw_split_dir}")
  samples_dir = new_cache_dir / "samples"

  dataset = preprocess_turbulent_split(raw_split_dir=raw_split_dir,
                                       gwt_config=gwt_config,
                                       samples_dir=samples_dir,
                                       height=height,
                                       width=width)
  (new_cache_dir / "config.json").write_text(
    json.dumps({"gwt_config": asdict(gwt_config), "height": height, "width": width}, indent=2, sort_keys=True))
  logging.info(f"Saved preprocessed turbulent split: {new_cache_dir.name}")
  dataset.storage_path = new_cache_dir
  if debug:
    dataset.sample_files = dataset.sample_files[:10]
  return dataset
