import torch
import torch.nn as nn
from torch import Tensor

from model.utils import get_activation_fn, get_gconv_layer

MAX_LOGSTD = 10


class SparseVariationalGraphAutoEncoder(nn.Module):
  def __init__(
          self,
          num_channels: int,
          latent_dim: int,
          activation_fn: str,
          conv_layer_type: str,
          num_layers: int,
          is_variational: bool,
  ):
    super().__init__()
    self.is_variational = is_variational
    self.activation_fn = get_activation_fn(name=activation_fn)

    encoder_out_dim = 2 * latent_dim if is_variational else latent_dim
    self.encoder_layers, self.encoder_norms = self._init_gnn(
      input_dim=num_channels,
      latent_dim=latent_dim,
      output_dim=encoder_out_dim,
      num_layers=num_layers,
      conv_layer_type=conv_layer_type
    )
    self.decoder_layers, self.decoder_norms = self._init_gnn(
      input_dim=latent_dim,
      latent_dim=latent_dim,
      output_dim=num_channels,
      num_layers=num_layers,
      conv_layer_type=conv_layer_type
    )

  @staticmethod
  def _init_gnn(
          input_dim: int,
          latent_dim: int,
          output_dim: int,
          num_layers: int,
          conv_layer_type: str,
  ) -> tuple[nn.ModuleList, nn.ModuleList]:
    layers, norms = nn.ModuleList(), nn.ModuleList()
    dim = input_dim
    for l in range(num_layers):
      is_last = l == num_layers - 1
      out = output_dim if is_last else latent_dim
      layers.append(
        get_gconv_layer(in_channels=dim,
                        out_channels=out,
                        conv_layer_type=conv_layer_type)
      )
      if not is_last:
        norms.append(nn.LayerNorm(dim) if dim > 1 else nn.Identity())
      dim = out
    return layers, norms

  def _run_gnn(
          self,
          h: Tensor,
          edge_index: Tensor,
          layers: nn.ModuleList,
          norms: nn.ModuleList
  ) -> Tensor:
    for conv, norm in zip(layers[:-1], norms):
      transformed = self.activation_fn(conv(norm(h), edge_index))
      h = h + transformed if h.shape[-1] == transformed.shape[-1] else transformed
    return layers[-1](h, edge_index)

  def _ste_sparsify(
          self,
          z: Tensor,
          batch: Tensor,
          sparsity_ratio: float,
          temperature: float
  ) -> tuple[Tensor, Tensor]:
    # Per graph: keep round(sparsity_ratio * n_i * D) entries (by |z|) out of
    # that graph's whole (n_i, D) block.
    # Forward = hard top-k; backward = gradient of a sigmoid soft mask around the graph's own threshold (STE).
    assert 0 < sparsity_ratio < 1, "sparsity_ratio must be in (0, 1)"
    num_graphs = int(batch.max()) + 1
    mask = torch.zeros_like(z)
    for g in range(num_graphs):
      idx = batch == g
      flat = z[idx].reshape(-1)
      k = max(1, round(sparsity_ratio * flat.numel()))
      magnitude = flat.abs()
      topk_val, topk_idx = torch.topk(magnitude, k)
      hard_mask = torch.zeros_like(flat)
      hard_mask[topk_idx] = 1.0
      soft_mask = torch.sigmoid((magnitude - topk_val.min()) / temperature)
      mask[idx] = (hard_mask.detach() - soft_mask.detach() + soft_mask).reshape(z[idx].shape)
    return z * mask, mask

  def encode(
          self,
          x: Tensor,
          edge_index: Tensor,
          batch: Tensor,
          sparsity_ratio: float,
          temperature: float,
  ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    h = self._run_gnn(x, edge_index, self.encoder_layers, self.encoder_norms)
    if self.is_variational:
      mu, logstd = h.chunk(2, dim=-1)
      logstd = logstd.clamp(max=MAX_LOGSTD)
      z = mu + torch.randn_like(logstd) * logstd.exp() if self.training else mu
    else:
      mu, logstd, z = None, None, h
    z_sparse, mask = self._ste_sparsify(z, batch, sparsity_ratio, temperature)
    return z_sparse, mask, mu, logstd

  def decode(
          self,
          z: Tensor,
          edge_index: Tensor
  ) -> Tensor:
    return self._run_gnn(z, edge_index, self.decoder_layers, self.decoder_norms)

  def forward(
          self,
          x: Tensor,
          edge_index: Tensor,
          batch: Tensor,
          sparsity_ratio: float,
          temperature: float,
  ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    z_sparse, mask, mu, logstd = self.encode(x, edge_index, batch, sparsity_ratio, temperature)
    x_hat = self.decode(z_sparse, edge_index)
    return x_hat, mask, mu, logstd
