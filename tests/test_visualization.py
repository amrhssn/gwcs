import re

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure

from data.generation.graph import generate_graph
from data.generation.signal import approximately_bandlimited
from transform.exact_sgwt import analysis
from transform.kernels import default_kernels
from sampler.mlis import multilevel_importance_sampling
from visualization import (
  visualize_signal,
  visualize_grid_reconstruction,
  visualize_grid_gwt_coefficients,
  visualize_grid_sparsification_mask,
)

NUM_NODES = 12
SCALES = [0.5, 1.5]


@pytest.fixture(autouse=True)
def _close_figures():
  yield
  plt.close('all')


@pytest.fixture
def grid_graph():
  return generate_graph(graph_type='grid', num_nodes=NUM_NODES)


def _make_signal(eigenvectors: torch.Tensor, num_channels: int, seed: int = 0) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  num_nodes = eigenvectors.shape[0]
  num_frequency_modes = max(3, num_nodes // 3)
  channels = []
  for _ in range(num_channels):
    coeffs = torch.randn(num_frequency_modes, generator=generator)
    signal = approximately_bandlimited(coeffs=coeffs,
                                       num_frequency_modes=num_frequency_modes,
                                       signal_type='lowpass',
                                       eigenvector=eigenvectors,
                                       noise_level=0.1,
                                       random_generator=generator)
    channels.append(signal.x)  # (N, 1)
  return torch.cat(channels, dim=1)  # (N, C)


def _main_axes(fig: Figure) -> list:
  """Plotting axes carry a title; colorbar axes created alongside them don't."""
  return [ax for ax in fig.axes if ax.get_title()]


def test_visualize_signal_never_calls_show(monkeypatch, grid_graph):
  monkeypatch.setattr(plt, 'show', lambda *a, **k: pytest.fail("visualize_signal must not call plt.show()"))
  signal = _make_signal(grid_graph.eigenvectors, num_channels=1)
  fig = visualize_signal(signal=signal, edge_index=grid_graph.edge_index,
                         graph_type='grid', pos=grid_graph.metadata.pos)
  assert isinstance(fig, Figure)


def test_visualize_signal_requires_pos_for_grid(grid_graph):
  signal = _make_signal(grid_graph.eigenvectors, num_channels=1)
  with pytest.raises(ValueError):
    visualize_signal(signal=signal, edge_index=grid_graph.edge_index, graph_type='grid', pos=None)


def test_visualize_signal_one_subplot_per_channel(grid_graph):
  num_channels = 3
  signal = _make_signal(grid_graph.eigenvectors, num_channels=num_channels)
  fig = visualize_signal(signal=signal, edge_index=grid_graph.edge_index,
                         graph_type='grid', pos=grid_graph.metadata.pos)

  assert fig.get_suptitle()
  main_axes = _main_axes(fig)
  assert len(main_axes) == num_channels
  for c, ax in enumerate(main_axes):
    assert f'Channel {c}' in ax.get_title()
    caption = ax.texts[0].get_text()
    assert f'N={NUM_NODES} nodes' in caption


def test_visualize_grid_reconstruction_grid_shape(grid_graph):
  num_channels = 2
  rows, cols = grid_graph.metadata.height, grid_graph.metadata.width
  signal = _make_signal(grid_graph.eigenvectors, num_channels=num_channels)
  reconstructed = signal + 0.01 * torch.randn_like(signal)

  fig = visualize_grid_reconstruction(reconstructed=reconstructed, signal=signal, rows=rows, cols=cols,
                                      gwt_mode='exact')

  assert 'exact' in fig.get_suptitle().lower()
  main_axes = _main_axes(fig)
  assert len(main_axes) == num_channels * 3
  for c in range(num_channels):
    original, reconstruction, error = main_axes[3 * c:3 * c + 3]
    assert 'Original Signal' in original.get_title() and f'Channel {c}' in original.get_title()
    assert 'Reconstruction' in reconstruction.get_title() and f'Channel {c}' in reconstruction.get_title()
    assert 'Absolute Reconstruction Error' in error.get_title() and f'Channel {c}' in error.get_title()
    assert re.search(r'RMSE: [\d.]+', error.texts[0].get_text())


def test_visualize_grid_reconstruction_saves_figure(monkeypatch, tmp_path, grid_graph):
  monkeypatch.chdir(tmp_path)
  rows, cols = grid_graph.metadata.height, grid_graph.metadata.width
  signal = _make_signal(grid_graph.eigenvectors, num_channels=1)
  reconstructed = signal + 0.01 * torch.randn_like(signal)

  visualize_grid_reconstruction(reconstructed=reconstructed, signal=signal, rows=rows, cols=cols,
                                gwt_mode='exact', save_fig=True)

  saved = list((tmp_path / 'gwt_figures').glob(f'*{rows}x{cols}*.png'))
  assert len(saved) == 1


def test_visualize_grid_gwt_coefficients_grid_shape(grid_graph):
  num_channels = 2
  rows, cols = grid_graph.metadata.height, grid_graph.metadata.width
  kernels = default_kernels(minimum_eigenvalue=0.1, scales=SCALES)
  signal = _make_signal(grid_graph.eigenvectors, num_channels=num_channels)
  coeffs = analysis(signal=signal,
                    symmetric_normalized_laplacian=grid_graph.symmetric_normalized_laplacian,
                    spectral_kernels=kernels,
                    eigenvalues=grid_graph.eigenvalues,
                    eigenvectors=grid_graph.eigenvectors)
  assert coeffs.shape == (NUM_NODES, num_channels, len(kernels))

  fig = visualize_grid_gwt_coefficients(coeffs=coeffs, signal=signal, scales=SCALES, rows=rows, cols=cols,
                                        gwt_mode='exact')

  assert 'exact' in fig.get_suptitle().lower()
  main_axes = _main_axes(fig)
  block_size = len(kernels) + 1  # leading signal column + one column per filter
  assert len(main_axes) == num_channels * block_size
  for c in range(num_channels):
    channel_axes = main_axes[block_size * c:block_size * (c + 1)]
    assert f'Signal — Channel {c}' in channel_axes[0].get_title()
    assert 'Scaling Function' in channel_axes[1].get_title()
    for f, ax in enumerate(channel_axes[2:], start=1):
      assert f'Wavelet Scale {f}' in ax.get_title()
      assert f'{SCALES[f - 1]:.2f}' in ax.get_title()


def test_visualize_grid_gwt_coefficients_single_filter(grid_graph):
  rows, cols = grid_graph.metadata.height, grid_graph.metadata.width
  kernels = default_kernels(minimum_eigenvalue=0.1, scales=[])  # scaling kernel only, F == 1
  signal = _make_signal(grid_graph.eigenvectors, num_channels=1)
  coeffs = analysis(signal=signal,
                    symmetric_normalized_laplacian=grid_graph.symmetric_normalized_laplacian,
                    spectral_kernels=kernels,
                    eigenvalues=grid_graph.eigenvalues,
                    eigenvectors=grid_graph.eigenvectors)
  assert coeffs.shape == (NUM_NODES, 1, 1)

  fig = visualize_grid_gwt_coefficients(coeffs=coeffs, signal=signal, scales=[], rows=rows, cols=cols,
                                        gwt_mode='exact')

  assert len(_main_axes(fig)) == 2  # signal column + the single scaling-function column


def test_visualize_grid_sparsification_mask_grid_shape(grid_graph):
  num_channels, keep_ratio = 2, 0.5
  rows, cols = grid_graph.metadata.height, grid_graph.metadata.width
  kernels = default_kernels(minimum_eigenvalue=0.1, scales=SCALES)
  signal = _make_signal(grid_graph.eigenvectors, num_channels=num_channels)
  coeffs = analysis(signal=signal,
                    symmetric_normalized_laplacian=grid_graph.symmetric_normalized_laplacian,
                    spectral_kernels=kernels,
                    eigenvalues=grid_graph.eigenvalues,
                    eigenvectors=grid_graph.eigenvectors)
  mask = multilevel_importance_sampling(coeffs=coeffs, keep_ratio=keep_ratio,
                                        random_generator=torch.Generator().manual_seed(0))
  assert mask.shape == coeffs.shape

  fig = visualize_grid_sparsification_mask(mask=mask, rows=rows, cols=cols)

  assert fig.get_suptitle()
  main_axes = _main_axes(fig)
  assert len(main_axes) == num_channels * len(kernels)
  for c in range(num_channels):
    for f in range(len(kernels)):
      ax = main_axes[len(kernels) * c + f]
      assert f'Filter {f}' in ax.get_title() and f'Channel {c}' in ax.get_title()
      expected_ratio = mask[:, c, f].float().mean().item()
      caption = ax.texts[0].get_text()
      match = re.search(r'Kept ([\d.]+)% of nodes', caption)
      assert match is not None
      # caption rounds to one decimal place (0.1% = 0.001 fraction), so allow half that as tolerance
      assert abs(float(match.group(1)) / 100 - expected_ratio) < 5e-4
