import pytest
import torch

from model.nigwt import NeuralInverseGraphWaveletTransform
from transform import exact_sgwt
from tests.model.conftest import GridGraph, KernelBank


@pytest.mark.parametrize("conv_layer_type", ["GCNConv", "SAGEConv", "GATConv", "GINConv"])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("igwt_mode", ["exact", "chebyshev"])
def test_forward_shape(
        grid_graph: GridGraph,
        kernel_bank: KernelBank,
        igwt_mode: str,
        num_channels: int,
        conv_layer_type: str,
) -> None:
  edge_index, laplacian, eigenvalues, eigenvectors, num_nodes = grid_graph
  kernels, num_filters = kernel_bank
  generator = torch.Generator().manual_seed(0)

  x = torch.randn(num_nodes, num_channels, dtype=torch.float64, generator=generator)
  gwt_coeffs = exact_sgwt.analysis(x, laplacian, kernels, eigenvalues, eigenvectors)
  sparsification_mask = torch.rand(num_nodes, num_channels, num_filters, generator=generator) > 0.5
  sparse_gwt_coeffs = gwt_coeffs * sparsification_mask.to(gwt_coeffs.dtype)

  model = NeuralInverseGraphWaveletTransform(
    num_channels=num_channels,
    spectral_kernels=kernels,
    hidden_dim=8,
    activation_fn="relu",
    conv_layer_type=conv_layer_type,
    processor_num_layers=2,
    igwt_mode=igwt_mode,
  ).to(torch.float64)

  x_hat = model(
    sparse_gwt_coeffs=sparse_gwt_coeffs,
    sparsification_mask=sparsification_mask,
    edge_index=edge_index,
    symmetric_normalized_laplacian=laplacian,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
  )

  assert x_hat.shape == (num_nodes, num_channels)
