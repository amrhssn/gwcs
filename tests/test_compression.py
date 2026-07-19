import math

import pytest
import torch

from compression import resolve_gwcs_compression, resolve_sgae_compression
from model.sgae import SparseVariationalGraphAutoEncoder
from sampler.mlis_vectorized import multilevel_importance_sampling


@pytest.mark.parametrize("target_compression_ratio,num_filters", [
  (0.5, 1), (1.0, 3), (2.5, 5), (5.0, 5),
])
def test_resolve_gwcs_compression_formula(target_compression_ratio: float, num_filters: int) -> None:
  compression = resolve_gwcs_compression(target_compression_ratio=target_compression_ratio, num_filters=num_filters)
  assert compression.keep_ratio == pytest.approx(target_compression_ratio / num_filters)
  assert compression.sparsity_ratio is None
  assert compression.target_compression_ratio == target_compression_ratio


@pytest.mark.parametrize("target_compression_ratio,num_filters", [
  (0.5, 1), (1.0, 3), (2.5, 5), (5.0, 5), (0.01, 8),
])
def test_resolve_gwcs_compression_round_trip(target_compression_ratio: float, num_filters: int) -> None:
  compression = resolve_gwcs_compression(target_compression_ratio=target_compression_ratio, num_filters=num_filters)
  assert compression.keep_ratio * num_filters == pytest.approx(target_compression_ratio)


@pytest.mark.parametrize("target_compression_ratio,num_filters", [
  (0.0, 3),      # keep_ratio == 0, not in (0, 1]
  (-1.0, 3),     # negative
  (3.01, 3),     # keep_ratio > 1
  (10.0, 3),     # keep_ratio >> 1
])
def test_resolve_gwcs_compression_rejects_unachievable_ratio(target_compression_ratio: float,
                                                              num_filters: int) -> None:
  with pytest.raises(ValueError, match="target_compression_ratio"):
    resolve_gwcs_compression(target_compression_ratio=target_compression_ratio, num_filters=num_filters)


@pytest.mark.parametrize("target_compression_ratio,num_channels,latent_dim", [
  (0.05, 1, 128), (0.5, 3, 64), (1.0, 2, 8), (0.1, 5, 100),
])
def test_resolve_sgae_compression_formula(target_compression_ratio: float, num_channels: int,
                                          latent_dim: int) -> None:
  compression = resolve_sgae_compression(target_compression_ratio=target_compression_ratio,
                                         num_channels=num_channels, latent_dim=latent_dim)
  assert compression.sparsity_ratio == pytest.approx(target_compression_ratio * num_channels / latent_dim)
  assert compression.keep_ratio is None
  assert compression.target_compression_ratio == target_compression_ratio


@pytest.mark.parametrize("target_compression_ratio,num_channels,latent_dim", [
  (0.05, 1, 128), (0.5, 3, 64), (1.0, 2, 8), (0.1, 5, 100),
])
def test_resolve_sgae_compression_round_trip(target_compression_ratio: float, num_channels: int,
                                             latent_dim: int) -> None:
  compression = resolve_sgae_compression(target_compression_ratio=target_compression_ratio,
                                         num_channels=num_channels, latent_dim=latent_dim)
  assert compression.sparsity_ratio * latent_dim / num_channels == pytest.approx(target_compression_ratio)


@pytest.mark.parametrize("target_compression_ratio,num_channels,latent_dim", [
  (0.0, 1, 128),     # sparsity_ratio == 0, not in (0, 1)
  (-0.1, 1, 128),    # negative
  (128.0, 1, 128),   # sparsity_ratio == 1, not < 1
  (200.0, 1, 128),   # sparsity_ratio > 1
])
def test_resolve_sgae_compression_rejects_unachievable_ratio(target_compression_ratio: float, num_channels: int,
                                                             latent_dim: int) -> None:
  with pytest.raises(ValueError, match="target_compression_ratio"):
    resolve_sgae_compression(target_compression_ratio=target_compression_ratio, num_channels=num_channels,
                             latent_dim=latent_dim)


@pytest.mark.parametrize("num_nodes", [5, 17, 50])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_filters", [1, 3, 5])
def test_gwcs_achieved_ratio_matches_derived_keep_ratio_across_shapes(
        num_nodes: int, num_channels: int, num_filters: int
) -> None:
  target_compression_ratio = 0.6 * num_filters  # keeps keep_ratio == 0.6, always in (0, 1]
  compression = resolve_gwcs_compression(target_compression_ratio=target_compression_ratio, num_filters=num_filters)

  generator = torch.Generator().manual_seed(0)
  coeffs = torch.randn(num_nodes, num_channels, num_filters, generator=generator)
  mask = multilevel_importance_sampling(coeffs=coeffs, keep_ratio=compression.keep_ratio,
                                        random_generator=generator)

  achieved_ratio = mask.float().mean().item() * num_filters
  kept_per_channel = max(1, math.floor(num_nodes * num_filters * compression.keep_ratio))
  expected_ratio = kept_per_channel / num_nodes
  assert achieved_ratio == pytest.approx(expected_ratio, rel=1e-5)


@pytest.mark.parametrize("num_nodes", [5, 17, 50])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("latent_dim", [4, 16])
def test_sgae_achieved_ratio_matches_derived_sparsity_ratio_across_shapes(
        num_nodes: int, num_channels: int, latent_dim: int
) -> None:
  target_compression_ratio = 0.5 * latent_dim / num_channels  # keeps sparsity_ratio == 0.5, always in (0, 1)
  compression = resolve_sgae_compression(target_compression_ratio=target_compression_ratio,
                                         num_channels=num_channels, latent_dim=latent_dim)

  model = SparseVariationalGraphAutoEncoder(num_channels=num_channels, latent_dim=latent_dim,
                                            activation_fn="relu", conv_layer_type="SAGEConv",
                                            num_layers=1, is_variational=False)
  z = torch.randn(num_nodes, latent_dim)
  batch = torch.zeros(num_nodes, dtype=torch.long)
  z_sparse, _ = model._ste_sparsify(z=z, batch=batch, sparsity_ratio=compression.sparsity_ratio, temperature=0.5)

  kept = int((z_sparse != 0).sum().item())
  expected_kept = max(1, round(compression.sparsity_ratio * num_nodes * latent_dim))
  assert kept == expected_kept

  achieved_ratio = kept / (num_nodes * num_channels)
  expected_ratio = expected_kept / (num_nodes * num_channels)
  assert achieved_ratio == pytest.approx(expected_ratio, rel=1e-9)
