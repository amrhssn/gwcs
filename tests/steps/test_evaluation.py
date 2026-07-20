import math

import pytest
import torch
from torch_geometric.loader import DataLoader

from compression import resolve_gwcs_compression, resolve_sgae_compression
from configs.common import GraphWaveletTransformConfig
from data.dataset import ABLGraphSignalDataset
from eval.utils import relative_error
from model.loss import reconstruction_loss
from model.nigwt import NeuralInverseGraphWaveletTransform
from model.sgae import SparseVariationalGraphAutoEncoder
from steps.evaluation import run_gwcs_test_epoch, run_sgae_test_epoch

BATCH_SIZE = 3
LATENT_DIM = 8


def _expected_gwcs_ratio(num_nodes: int, num_filters: int, keep_ratio: float) -> float:
  kept_per_channel = max(1, math.floor(num_nodes * num_filters * keep_ratio))
  return kept_per_channel / num_nodes


def _expected_sgae_ratio(num_nodes: int, num_channels: int, latent_dim: int, sparsity_ratio: float) -> float:
  kept = max(1, round(sparsity_ratio * num_nodes * latent_dim))
  return kept / (num_nodes * num_channels)


def test_gwcs_achieved_compression_ratio_matches_derived_keep_ratio(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  target_compression_ratio = 0.9
  compression = resolve_gwcs_compression(target_compression_ratio=target_compression_ratio,
                                         num_filters=small_abl_dataset.num_filters)
  loader = DataLoader(dataset=small_abl_dataset, batch_size=BATCH_SIZE, shuffle=False)
  model = NeuralInverseGraphWaveletTransform(
    num_channels=small_abl_dataset.num_channels,
    hidden_dim=8,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    processor_num_layers=2,
    gwt_mode=gwt_config.gwt_mode,
    spectral_kernels=small_abl_dataset.spectral_kernels,
    spectrum_lower_bound=gwt_config.spectrum_lower_bound,
    spectrum_upper_bound=gwt_config.spectrum_upper_bound,
    chebyshev_degree=gwt_config.chebyshev_degree,
  )
  random_generator = torch.Generator().manual_seed(0)

  results = run_gwcs_test_epoch(loader=loader, model=model, device=torch.device("cpu"),
                                loss_fn=reconstruction_loss, error_fn=relative_error,
                                keep_ratio=compression.keep_ratio, random_generator=random_generator)

  assert len(results) == len(small_abl_dataset)
  for result in results:
    num_nodes = result.signal.shape[0]
    expected_ratio = _expected_gwcs_ratio(num_nodes=num_nodes, num_filters=small_abl_dataset.num_filters,
                                          keep_ratio=compression.keep_ratio)
    assert result.compression_ratio == pytest.approx(expected_ratio, rel=1e-5)
    assert result.mask.shape[-1] == small_abl_dataset.num_filters


def test_sgae_achieved_compression_ratio_matches_derived_sparsity_ratio(
        small_abl_dataset: ABLGraphSignalDataset,
) -> None:
  target_compression_ratio = 2.0
  compression = resolve_sgae_compression(target_compression_ratio=target_compression_ratio,
                                         num_channels=small_abl_dataset.num_channels,
                                         latent_dim=LATENT_DIM)
  loader = DataLoader(dataset=small_abl_dataset, batch_size=BATCH_SIZE, shuffle=False)
  model = SparseVariationalGraphAutoEncoder(
    num_channels=small_abl_dataset.num_channels,
    latent_dim=LATENT_DIM,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=True,
  )

  results = run_sgae_test_epoch(loader=loader, model=model, device=torch.device("cpu"),
                                loss_fn=reconstruction_loss, error_fn=relative_error,
                                sparsity_ratio=compression.sparsity_ratio, temperature=0.5,
                                num_channels=small_abl_dataset.num_channels, latent_dim=LATENT_DIM)

  assert len(results) == len(small_abl_dataset)
  for result in results:
    num_nodes = result.signal.shape[0]
    expected_ratio = _expected_sgae_ratio(num_nodes=num_nodes, num_channels=small_abl_dataset.num_channels,
                                          latent_dim=LATENT_DIM, sparsity_ratio=compression.sparsity_ratio)
    assert result.compression_ratio == pytest.approx(expected_ratio, rel=1e-5)
