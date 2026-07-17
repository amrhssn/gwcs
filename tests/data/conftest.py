import math
from typing import Literal

import pytest
import torch

from data.dataset import ABLGraphSignalDataset
from data.types import (
  GraphGenerationConfig,
  SignalGenerationConfig,
  GraphWaveletTransformConfig,
)
from steps.data_generation import generate_abl_dataset


NUM_NODES_RANGE = (30, 80)
NUM_GRAPHS_PER_TYPE = 6
SPECTRUM_LOWER_BOUND = 0.0
SPECTRUM_UPPER_BOUND = 2.0
CHEBYSHEV_DEGREE = 4
NUM_SCALES = 2
BATCH_SIZE = 3
SEED = 0


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
def small_abl_dataset(gwt_config: GraphWaveletTransformConfig) -> ABLGraphSignalDataset:
  graph_types: list[Literal['grid']] = ['grid']
  random_generator = torch.Generator().manual_seed(SEED)
  graph_config = GraphGenerationConfig(graph_types=graph_types,
                                       num_nodes_range=NUM_NODES_RANGE,
                                       num_graphs_per_type=NUM_GRAPHS_PER_TYPE)
  signal_config = SignalGenerationConfig(signal_types=['lowpass'],
                                         max_num_frequency_mode_ratio=0.1,
                                         coeffs_mean=0.0,
                                         coeffs_std_range=(1.0, 2.0),
                                         noise_level_range=(0.05, 0.2))
  return generate_abl_dataset(graph_config=graph_config,
                              signal_config=signal_config,
                              gwt_config=gwt_config,
                              random_generator=random_generator)
