import os
from typing import (
  Literal,
  Optional,
)

import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _add_caption(
        ax: plt.Axes,
        text: str,
) -> None:
  ax.text(0.5, -0.35, text, transform=ax.transAxes, ha='center', va='top', fontsize=8, wrap=True)


def visualize_signal(
        signal: torch.Tensor,
        edge_index: torch.Tensor,
        graph_type: Literal['grid', 'erdos_renyi', 'barabasi_albert', 'sbm'],
        pos: Optional[list[list[float]]] = None,
) -> plt.Figure:
  num_nodes, num_channels = signal.shape
  graph = nx.Graph()
  graph.add_nodes_from(range(num_nodes))
  graph.add_edges_from(edge_index.t().numpy())

  if graph_type == 'grid':
    if pos is None:
      raise ValueError("For grid graphs, pos must not be None")
    layout = {i: (pos[i][0], pos[i][1]) for i in range(num_nodes)}
  else:
    layout = nx.kamada_kawai_layout(graph)

  fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, 5), squeeze=False)
  for c in range(num_channels):
    ax = axes[0][c]
    channel_signal = signal[:, c].numpy()
    nodes = nx.draw_networkx_nodes(graph, pos=layout, ax=ax, node_size=80,
                                   node_color=channel_signal, cmap='RdBu_r')
    nx.draw_networkx_edges(graph, pos=layout, ax=ax, edge_color='lightgray', width=0.3)
    fig.colorbar(nodes, ax=ax, label='Signal Value')
    ax.set_title(f'{graph_type.replace("_", " ").title()} Graph — Channel {c}')
    ax.set_aspect('equal')
    ax.axis('off')
    _add_caption(ax, f'N={num_nodes} nodes, value range [{channel_signal.min():.3f}, {channel_signal.max():.3f}].')

  fig.subplots_adjust(bottom=0.25)
  return fig


def visualize_grid_reconstruction(
        reconstructed: torch.Tensor,
        signal: torch.Tensor,
        rows: int,
        cols: int,
        save_fig: bool = False,
) -> plt.Figure:
  num_channels = signal.shape[1]
  fig, axes = plt.subplots(num_channels, 3, figsize=(18, 5 * num_channels), squeeze=False)

  for c in range(num_channels):
    channel_signal = signal[:, c].reshape(rows, cols)
    channel_reconstructed = reconstructed[:, c].reshape(rows, cols)
    error = channel_signal - channel_reconstructed
    abs_error = torch.abs(error)
    ax0, ax1, ax2 = axes[c]

    im0 = ax0.imshow(channel_signal.numpy(), aspect='auto', cmap='RdBu_r')
    ax0.set_title(f'Original Signal (x) — Channel {c}')
    ax0.set_xlabel('cols')
    ax0.set_ylabel('rows')
    fig.colorbar(im0, ax=ax0)
    _add_caption(ax0, f'Ground-truth signal reshaped to the {rows}x{cols} grid.')

    im1 = ax1.imshow(channel_reconstructed.numpy(), aspect='auto', cmap='RdBu_r')
    ax1.set_title(f'Reconstruction (x_recon) — Channel {c}')
    ax1.set_xlabel('cols')
    ax1.set_ylabel('rows')
    fig.colorbar(im1, ax=ax1)
    _add_caption(ax1, 'Model reconstruction from the sparsified GWT coefficients.')

    im2 = ax2.imshow(abs_error.numpy(), aspect='auto', cmap='Reds')
    ax2.set_title(f'Absolute Reconstruction Error |x - x_recon| — Channel {c}')
    ax2.set_xlabel('cols')
    ax2.set_ylabel('rows')
    fig.colorbar(im2, ax=ax2)
    mse = torch.mean(error ** 2).item()
    _add_caption(ax2, f'Max abs error: {abs_error.max().item():.4f} | '
                      f'MSE: {mse:.4f} | RMSE: {mse ** 0.5:.4f}.')

  fig.tight_layout()
  fig.subplots_adjust(bottom=0.2 / num_channels)

  if save_fig:
    save_dir = 'gwt_figures'
    os.makedirs(save_dir, exist_ok=True)
    filename = f'inverse_sgwt_reconstruction_{rows}x{cols}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")

  return fig


def visualize_grid_gwt_coefficients(
        coeffs: torch.Tensor,
        scales: list[float],
        rows: int,
        cols: int,
) -> plt.Figure:
  num_channels, num_filters = coeffs.shape[1], coeffs.shape[2]
  aspect_ratio = cols / rows
  fig_height = 4
  fig_width = fig_height * aspect_ratio
  fig, axes = plt.subplots(num_channels, num_filters,
                           figsize=(2 * fig_width * num_filters, 2 * fig_height * num_channels),
                           squeeze=False)

  for c in range(num_channels):
    for f in range(num_filters):
      ax = axes[c][f]
      scale_coeffs = coeffs[:, c, f].reshape(rows, cols)
      im = ax.imshow(scale_coeffs.numpy(), aspect='auto', cmap='RdBu_r')
      if f == 0:
        ax.set_title(f'Scaling Function — Channel {c}')
        caption = 'Low-pass scaling coefficients (filter 0).'
      else:
        ax.set_title(f'Wavelet Scale {f} (s={scales[f - 1]:.2f}) — Channel {c}')
        caption = f'Wavelet-scale {f} coefficients at scale s={scales[f - 1]:.2f}.'
      ax.set_xlabel('cols')
      ax.set_ylabel('rows')
      fig.colorbar(im, ax=ax)
      _add_caption(ax, caption)

  fig.tight_layout()
  fig.subplots_adjust(bottom=0.15 / num_channels)
  return fig


def visualize_grid_sparsification_mask(
        mask: torch.Tensor,
        rows: int,
        cols: int,
) -> plt.Figure:
  num_channels, num_filters = mask.shape[1], mask.shape[2]
  aspect_ratio = cols / rows
  fig_height = 4
  fig_width = fig_height * aspect_ratio
  fig, axes = plt.subplots(num_channels, num_filters,
                           figsize=(2 * fig_width * num_filters, 2 * fig_height * num_channels),
                           squeeze=False)
  cmap = ListedColormap(['white', 'black'])

  for c in range(num_channels):
    for f in range(num_filters):
      ax = axes[c][f]
      filter_mask = mask[:, c, f].reshape(rows, cols)
      ax.imshow(filter_mask.numpy(), aspect='auto', cmap=cmap, vmin=0, vmax=1)
      keep_ratio = filter_mask.float().mean().item()
      ax.set_title(f'Filter {f} — Channel {c}')
      ax.set_xlabel('cols')
      ax.set_ylabel('rows')
      _add_caption(ax, f'Kept {keep_ratio:.1%} of nodes for this (channel, filter) pair.')

  fig.tight_layout()
  fig.subplots_adjust(bottom=0.15 / num_channels)
  return fig
