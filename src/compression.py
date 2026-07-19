from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedCompression:
  target_compression_ratio: float
  keep_ratio: float | None = None
  sparsity_ratio: float | None = None


def resolve_gwcs_compression(target_compression_ratio: float, num_filters: int) -> ResolvedCompression:
  keep_ratio = target_compression_ratio / num_filters
  if not 0 < keep_ratio <= 1:
    raise ValueError(
      f"target_compression_ratio={target_compression_ratio:.4g} is not achievable with "
      f"num_filters={num_filters} (implies keep_ratio={keep_ratio:.4g}, must be in (0, 1]). "
      f"Valid range for target_compression_ratio here: (0, {num_filters:.4g}]."
    )
  return ResolvedCompression(target_compression_ratio=target_compression_ratio, keep_ratio=keep_ratio)


def resolve_sgae_compression(target_compression_ratio: float, num_channels: int, latent_dim: int) -> ResolvedCompression:
  sparsity_ratio = target_compression_ratio * num_channels / latent_dim
  if not 0 < sparsity_ratio < 1:
    max_ratio = latent_dim / num_channels
    raise ValueError(
      f"target_compression_ratio={target_compression_ratio:.4g} is not achievable with "
      f"num_channels={num_channels}, latent_dim={latent_dim} (implies sparsity_ratio={sparsity_ratio:.4g}, "
      f"must be in (0, 1)). Valid range for target_compression_ratio here: (0, {max_ratio:.4g})."
    )
  return ResolvedCompression(target_compression_ratio=target_compression_ratio, sparsity_ratio=sparsity_ratio)
