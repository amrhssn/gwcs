import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch

from configs.abl import (
  GraphGenerationConfig,
  SignalGenerationConfig,
  GenerationConfig,
)
from configs.common import GraphWaveletTransformConfig
from data.generation.graph import generate_graph
from data.generation.signal import approximately_bandlimited
from data.types import ABLGraphSignalSample
from data.dataset import ABLGraphSignalDataset
from transform.kernels import default_kernels
from transform.exact_sgwt import analysis as exact_analysis
from transform.chebyshev_sgwt import analysis as chebyshev_analysis



def _generate_num_nodes(
        graph_config: GraphGenerationConfig,
        random_generator: torch.Generator | None = None
) -> int:
  num_nodes = int(torch.randint(low=graph_config.num_nodes_range[0],
                                high=graph_config.num_nodes_range[1] + 1,
                                size=(1,),
                                generator=random_generator).item())
  return num_nodes


def _generate_abl_signal_type(
        signal_config: SignalGenerationConfig,
        random_generator: torch.Generator | None = None
) -> Literal['lowpass', 'highpass', 'bandpass', 'R-sparse']:
  signal_type_idx = int(torch.randint(low=0, high=len(signal_config.signal_types),
                                      size=(1,),
                                      generator=random_generator).item())
  signal_type = signal_config.signal_types[signal_type_idx]
  return signal_type


def _generate_abl_signal_coeffs(
        signal_config: SignalGenerationConfig,
        num_nodes: int,
        device: torch.device | None = None,
        random_generator: torch.Generator | None = None
) -> tuple[torch.Tensor, int]:
  coeffs_mean = signal_config.coeffs_mean
  coeffs_std = float(torch.empty(1).uniform_(signal_config.coeffs_std_range[0],
                                             signal_config.coeffs_std_range[1],
                                             generator=random_generator).item())
  low = min(3, num_nodes)
  max_modes = min(num_nodes, max(low, int(num_nodes * signal_config.max_num_frequency_mode_ratio)))
  num_frequency_modes = int(torch.randint(low=low,
                                          high=max_modes + 1,
                                          size=(1,),
                                          generator=random_generator).item())
  coeffs = torch.normal(mean=coeffs_mean,
                        std=coeffs_std,
                        size=(num_frequency_modes,),
                        generator=random_generator).to(device)
  return coeffs, num_frequency_modes


def _generate_abl_signal_noise_level(
        signal_config: SignalGenerationConfig,
        random_generator: torch.Generator | None = None
) -> float:
  noise_level = float(torch.empty(1).uniform_(signal_config.noise_level_range[0],
                                              signal_config.noise_level_range[1],
                                              generator=random_generator).item())
  return noise_level



def generate_abl_dataset(
        graph_config: GraphGenerationConfig,
        signal_config: SignalGenerationConfig,
        gwt_config: GraphWaveletTransformConfig,
        device: torch.device | None = None,
        random_generator: torch.Generator | None = None,
) -> ABLGraphSignalDataset:
  samples: list[ABLGraphSignalSample] = []
  spectral_kernels = default_kernels(minimum_eigenvalue=gwt_config.scaling_kernel_cutoff_eigenvalue,
                                     scales=gwt_config.scales)
  total_samples = len(graph_config.graph_types) * graph_config.num_graphs_per_type
  log_every = max(1, total_samples // 100)
  logging.info(f"Generating dataset: {total_samples} samples, graph_types={graph_config.graph_types}")
  for graph_type in graph_config.graph_types:
    for graph_idx in range(graph_config.num_graphs_per_type):
      # graph parameters
      num_nodes = _generate_num_nodes(graph_config=graph_config,
                                      random_generator=random_generator)
      # signal parameters
      signal_type = _generate_abl_signal_type(signal_config=signal_config,
                                              random_generator=random_generator)
      noise_level = _generate_abl_signal_noise_level(signal_config=signal_config,
                                                     random_generator=random_generator)

      # generation
      graph = generate_graph(graph_type=graph_type,
                             num_nodes=num_nodes,
                             graph_config=graph_config,
                             device=device,
                             random_generator=random_generator)
      coeffs, num_frequency_modes = _generate_abl_signal_coeffs(signal_config=signal_config,
                                                                num_nodes=graph.eigenvectors.shape[0],
                                                                device=device,
                                                                random_generator=random_generator)
      abl_signal = approximately_bandlimited(coeffs=coeffs,
                                             num_frequency_modes=num_frequency_modes,
                                             signal_type=signal_type,
                                             eigenvector=graph.eigenvectors,
                                             noise_level=noise_level,
                                             random_generator=random_generator)
      if gwt_config.gwt_mode == "exact":
        gwt_coeffs = exact_analysis(signal=abl_signal.x,
                                    symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                    spectral_kernels=spectral_kernels,
                                    eigenvalues=graph.eigenvalues,
                                    eigenvectors=graph.eigenvectors)
      else:
        gwt_coeffs = chebyshev_analysis(signal=abl_signal.x,
                                        symmetric_normalized_laplacian=graph.symmetric_normalized_laplacian,
                                        spectral_kernels=spectral_kernels,
                                        spectrum_lower_bound=gwt_config.spectrum_lower_bound,
                                        spectrum_upper_bound=gwt_config.spectrum_upper_bound,
                                        chebyshev_degree=gwt_config.chebyshev_degree)
      sample = ABLGraphSignalSample(graph=graph,
                                    signal=abl_signal,
                                    gwt_coeffs=gwt_coeffs)
      samples.append(sample)
      if len(samples) % log_every == 0 or len(samples) == total_samples:
        logging.info(f"Generated {len(samples)}/{total_samples} samples")
  generation_config = GenerationConfig(graph_config=graph_config,
                                       signal_config=signal_config,
                                       gwt_config=gwt_config)
  dataset = ABLGraphSignalDataset(samples=samples,
                                  configs=generation_config)
  return dataset


def _cache_key(configs: GenerationConfig) -> str:
  config_json = json.dumps(asdict(configs), sort_keys=True)
  return hashlib.md5(config_json.encode()).hexdigest()[:8]


def _short_config_str(
        graph_config: GraphGenerationConfig,
        signal_config: SignalGenerationConfig,
        gwt_config: GraphWaveletTransformConfig,
) -> str:
  return (f"g={'-'.join(graph_config.graph_types)}"
         f"_n={graph_config.num_graphs_per_type}"
         f"_s={'-'.join(signal_config.signal_types)}"
         f"_gwt={gwt_config.gwt_mode}")


def _load_configs(path: Path) -> GenerationConfig:
  data = json.loads(path.read_text())
  return GenerationConfig(
    graph_config=GraphGenerationConfig(**data["graph_config"]),
    signal_config=SignalGenerationConfig(**data["signal_config"]),
    gwt_config=GraphWaveletTransformConfig(**data["gwt_config"]),
  )


def load_or_generate_abl_dataset(
        dataset_dir: Path,
        graph_config: GraphGenerationConfig,
        signal_config: SignalGenerationConfig,
        gwt_config: GraphWaveletTransformConfig,
        device: torch.device | None = None,
        random_generator: torch.Generator | None = None,
) -> ABLGraphSignalDataset:
  dataset_dir.mkdir(parents=True, exist_ok=True)
  configs = GenerationConfig(graph_config=graph_config, signal_config=signal_config, gwt_config=gwt_config)
  cache_key = _cache_key(configs)

  matches = sorted(dataset_dir.glob(f"*_{cache_key}"))
  if matches:
    cache_dir = matches[-1]
    logging.info(f"Using cached dataset: {cache_dir.name}")
    samples = torch.load(cache_dir / "samples.pt", weights_only=False)
    cached_configs = _load_configs(cache_dir / "config.json")
    dataset = ABLGraphSignalDataset(samples=samples, configs=cached_configs)
    dataset.storage_path = cache_dir
    return dataset

  logging.info(f"No cached dataset found (key={cache_key}); generating new dataset")
  dataset = generate_abl_dataset(graph_config=graph_config, signal_config=signal_config,
                                 gwt_config=gwt_config, device=device, random_generator=random_generator)
  timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
  short_config = _short_config_str(graph_config, signal_config, gwt_config)
  cache_dir = dataset_dir / f"{timestamp}_{short_config}_{cache_key}"
  cache_dir.mkdir(parents=True)
  torch.save(dataset.samples, cache_dir / "samples.pt")
  (cache_dir / "config.json").write_text(json.dumps(asdict(configs), indent=2, sort_keys=True))
  logging.info(f"Saved dataset: {cache_dir.name}")
  dataset.storage_path = cache_dir
  return dataset
