from typing import Callable, Literal

import torch
import torch.nn as nn

from model.utils import (
  get_activation_fn,
  get_activation_module,
  get_gconv_layer,
)
from transform import exact_sgwt, chebyshev_sgwt


class NeuralInverseGraphWaveletTransform(torch.nn.Module):
  def __init__(
          self,
          num_channels: int,
          hidden_dim: int,
          activation_fn: str,
          conv_layer_type: str,
          processor_num_layers: int,
          igwt_mode: Literal["chebyshev", "exact"],
          spectral_kernels: list[Callable[[torch.Tensor], torch.Tensor]],
          spectrum_lower_bound: float = 0.0,
          spectrum_upper_bound: float = 2.0,
          chebyshev_degree: int = 30,
  ):
    super().__init__()
    self.num_channels = num_channels
    self.num_filters = len(spectral_kernels)
    self.spectral_kernels = spectral_kernels
    self.hidden_dim = hidden_dim
    self.igwt_mode = igwt_mode
    self.spectrum_lower_bound = spectrum_lower_bound
    self.spectrum_upper_bound = spectrum_upper_bound
    self.chebyshev_degree = chebyshev_degree
    self.activation_fn = get_activation_fn(name=activation_fn)
    self.effective_hidden_dim = num_channels * hidden_dim

    # Encoder
    self.mlp1_encoder = nn.Sequential(
      nn.Linear(in_features=2 * self.num_filters, out_features=hidden_dim),
      nn.LayerNorm(hidden_dim),
      get_activation_module(name=activation_fn),
      nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
    )
    self.scale_encoder = nn.Embedding(num_embeddings=self.num_filters, embedding_dim=hidden_dim)
    self.mlp2_encoder = nn.Sequential(
      nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim),
      nn.LayerNorm(hidden_dim),
      get_activation_module(name=activation_fn),
      nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
    )

    # Processor
    self.conv_layers = nn.ModuleList()
    self.layer_norms = nn.ModuleList()
    for _ in range(processor_num_layers):
      self.conv_layers.append(
        get_gconv_layer(in_channels=self.effective_hidden_dim,
                        out_channels=self.effective_hidden_dim,
                        conv_layer_type=conv_layer_type)
      )
      self.layer_norms.append(nn.LayerNorm(self.effective_hidden_dim))

    # Decoder
    self.decoder = nn.Sequential(
      nn.Linear(in_features=self.effective_hidden_dim, out_features=self.effective_hidden_dim),
      get_activation_module(name=activation_fn),
      nn.Linear(in_features=self.effective_hidden_dim, out_features=num_channels),
    )

  def forward(
          self,
          sparse_gwt_coeffs: torch.Tensor,
          sparsification_mask: torch.Tensor,
          edge_index: torch.Tensor,
          symmetric_normalized_laplacian: torch.Tensor,
          eigenvalues: torch.Tensor | None = None,
          eigenvectors: torch.Tensor | None = None,
          rescaled_symmetric_normalized_laplacian: torch.Tensor | None = None,
  ) -> torch.Tensor:
    encoded_inputs = self.encode(sparse_gwt_coeffs, sparsification_mask)
    processed_inputs = self.process(encoded_inputs, edge_index)
    x_hat = self.decode(processed_inputs, sparse_gwt_coeffs, symmetric_normalized_laplacian,
                        eigenvalues, eigenvectors, rescaled_symmetric_normalized_laplacian)
    return x_hat

  def encode(
          self,
          sparse_gwt_coeffs: torch.Tensor,
          sparsification_mask: torch.Tensor,
  ) -> torch.Tensor:
    sparsification_mask = sparsification_mask.float()  # (N, C, F)

    inputs = torch.cat([sparse_gwt_coeffs, sparsification_mask], dim=-1)  # (N, C, 2F)
    encoded_inputs = self.mlp1_encoder(inputs)  # (N, C, H)
    encoded_inputs = self.activation_fn(encoded_inputs)

    node_scale_embeddings = sparse_gwt_coeffs @ self.scale_encoder.weight  # (N, C, H)
    augmented_encoded_inputs = torch.cat([encoded_inputs, node_scale_embeddings], dim=-1)  # (N, C, 2H)
    encoded_inputs = self.mlp2_encoder(augmented_encoded_inputs)  # (N, C, H)
    encoded_inputs = self.activation_fn(encoded_inputs)
    return encoded_inputs

  def process(
          self,
          encoded_inputs: torch.Tensor,
          edge_index: torch.Tensor,
  ) -> torch.Tensor:
    num_nodes, num_channels, hidden_dim = encoded_inputs.shape
    x = encoded_inputs.reshape(num_nodes, num_channels * hidden_dim)
    for conv, norm in zip(self.conv_layers, self.layer_norms):
      normalized = norm(x)
      x = x + conv(self.activation_fn(normalized), edge_index)
    return x.reshape(num_nodes, num_channels, hidden_dim)

  def decode(
          self,
          processed_inputs: torch.Tensor,
          sparse_gwt_coeffs: torch.Tensor,
          symmetric_normalized_laplacian: torch.Tensor,
          eigenvalues: torch.Tensor | None = None,
          eigenvectors: torch.Tensor | None = None,
          rescaled_symmetric_normalized_laplacian: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_nodes, num_channels, hidden_dim = processed_inputs.shape
    processed_flat = processed_inputs.reshape(num_nodes, num_channels * hidden_dim)

    with torch.no_grad():
      if self.igwt_mode == "exact":
        x_hat_igwt = exact_sgwt.synthesis(
          wavelet_coeffs=sparse_gwt_coeffs,
          symmetric_normalized_laplacian=symmetric_normalized_laplacian,
          spectral_kernels=self.spectral_kernels,
          eigenvalues=eigenvalues,
          eigenvectors=eigenvectors,
        )
      else:
        x_hat_igwt = chebyshev_sgwt.synthesis(
          wavelet_coeffs=sparse_gwt_coeffs,
          symmetric_normalized_laplacian=symmetric_normalized_laplacian,
          spectral_kernels=self.spectral_kernels,
          spectrum_lower_bound=self.spectrum_lower_bound,
          spectrum_upper_bound=self.spectrum_upper_bound,
          chebyshev_degree=self.chebyshev_degree,
          rescaled_symmetric_normalized_laplacian=rescaled_symmetric_normalized_laplacian,
        )
    residual = self.decoder(processed_flat)
    return x_hat_igwt + residual  # (N, C)
