import os
from typing import (
  Literal,
  Optional,
)

import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SUPTITLE_FONTSIZE = 36
TITLE_FONTSIZE = 17
LABEL_FONTSIZE = 15
CAPTION_FONTSIZE = 14

# The constants above were tuned for a per-subplot height of this many
# inches. Grid-shaped figures size their subplots off a `fig_height` that
# scales with content (num channels/filters), so their text must scale with
# it too, or it shrinks relative to the (larger) canvas.
_FONT_SIZE_REFERENCE_HEIGHT = 4.0


def _scaled_fontsizes(fig_height: float) -> tuple[float, float, float, float]:
  scale = fig_height / _FONT_SIZE_REFERENCE_HEIGHT
  return (SUPTITLE_FONTSIZE * scale, TITLE_FONTSIZE * scale, LABEL_FONTSIZE * scale, CAPTION_FONTSIZE * scale)


def _add_caption(
        ax: plt.Axes,
        text: str,
        fontsize: float = CAPTION_FONTSIZE,
) -> None:
  ax.text(0.5, -0.35, text, transform=ax.transAxes, ha='center', va='top', fontsize=fontsize, wrap=True)


def _add_column_separator(
        fig: plt.Figure,
        left_ax: plt.Axes,
        right_ax: plt.Axes,
        top_ax: plt.Axes,
        bottom_ax: plt.Axes,
) -> None:
  fig.canvas.draw()
  left_pos = left_ax.get_position()
  right_pos = right_ax.get_position()
  sep_x = (left_pos.x1 + right_pos.x0) / 2
  y0 = bottom_ax.get_position().y0
  y1 = top_ax.get_position().y1
  fig.add_artist(Line2D([sep_x, sep_x], [y0, y1], transform=fig.transFigure,
                        color='gray', linestyle=':', linewidth=1))


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
    ax.set_title(f'{graph_type.replace("_", " ").title()} Graph — Channel {c}', fontsize=TITLE_FONTSIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    _add_caption(ax, f'N={num_nodes} nodes, value range [{channel_signal.min():.3f}, {channel_signal.max():.3f}].')

  # fig.suptitle('Graph Signal', fontsize=SUPTITLE_FONTSIZE, fontweight='bold')
  fig.subplots_adjust(bottom=0.25)
  return fig


def visualize_grid_reconstruction(
        reconstructed: torch.Tensor,
        signal: torch.Tensor,
        rows: int,
        cols: int,
        gwt_mode: Literal['exact', 'chebyshev'],
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
    ax0.set_title(f'Signal — C{c}', fontsize=TITLE_FONTSIZE)
    ax0.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax0.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im0, ax=ax0)
    # _add_caption(ax0, f'Ground-truth signal reshaped to the {rows}x{cols} grid.')

    im1 = ax1.imshow(channel_reconstructed.numpy(), aspect='auto', cmap='RdBu_r')
    ax1.set_title(f'Reconstruction — C{c}', fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im1, ax=ax1)
    # _add_caption(ax1, 'Model reconstruction from the sparsified GWT coefficients.')

    im2 = ax2.imshow(abs_error.numpy(), aspect='auto', cmap='Reds')
    ax2.set_title(f'Recon Error |x - x_recon| — C{c}', fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im2, ax=ax2)
    mse = torch.mean(error ** 2).item()
    _add_caption(ax2, f'Max abs error: {abs_error.max().item():.4f} | '
                      f'MSE: {mse:.4f} | RMSE: {mse ** 0.5:.4f}.')

  fig.suptitle(f'Signal vs. Reconstruction — {gwt_mode.title()} SGWT', fontsize=SUPTITLE_FONTSIZE, fontweight='bold')
  fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        signal: torch.Tensor,
        scales: list[float],
        rows: int,
        cols: int,
        gwt_mode: Literal['exact', 'chebyshev'],
) -> plt.Figure:
  num_channels, num_filters = coeffs.shape[1], coeffs.shape[2]
  num_columns = num_filters + 1
  aspect_ratio = cols / rows
  fig_height = 8
  fig_width = fig_height * aspect_ratio
  suptitle_fontsize, title_fontsize, label_fontsize, caption_fontsize = _scaled_fontsizes(fig_height)
  fig, axes = plt.subplots(num_channels, num_columns,
                           figsize=(2 * fig_width * num_columns, 2 * fig_height * num_channels),
                           squeeze=False)

  for c in range(num_channels):
    signal_ax = axes[c][0]
    channel_signal = signal[:, c].reshape(rows, cols)
    signal_im = signal_ax.imshow(channel_signal.numpy(), aspect='auto', cmap='RdBu_r')
    signal_ax.set_title(f'Signal — C{c}', fontsize=title_fontsize)
    signal_ax.set_xlabel('cols', fontsize=label_fontsize)
    signal_ax.set_ylabel('rows', fontsize=label_fontsize)
    fig.colorbar(signal_im, ax=signal_ax)
    # _add_caption(signal_ax, f'Value range [{channel_signal.min():.3f}, {channel_signal.max():.3f}].',
    #             fontsize=caption_fontsize)

    for f in range(num_filters):
      ax = axes[c][f + 1]
      scale_coeffs = coeffs[:, c, f].reshape(rows, cols)
      im = ax.imshow(scale_coeffs.numpy(), aspect='auto', cmap='RdBu_r')
      if f == 0:
        ax.set_title(f'Scaling Function — C{c}', fontsize=title_fontsize)
        # caption = 'Low-pass scaling coefficients (filter 0).'
      else:
        ax.set_title(f'Wavelet Scale {f} (s={scales[f - 1]:.2f}) — C{c}', fontsize=title_fontsize)
        # caption = f'Wavelet-scale {f} coefficients at scale s={scales[f - 1]:.2f}.'
      ax.set_xlabel('cols', fontsize=label_fontsize)
      ax.set_ylabel('rows', fontsize=label_fontsize)
      fig.colorbar(im, ax=ax)
      # _add_caption(ax, caption, fontsize=caption_fontsize)

  fig.suptitle(f'SGWT Coefficients — {gwt_mode.title()} SGWT', fontsize=suptitle_fontsize, fontweight='bold')
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  fig.subplots_adjust(bottom=0.15 / num_channels)
  _add_column_separator(fig, left_ax=axes[0][0], right_ax=axes[0][1], top_ax=axes[0][0], bottom_ax=axes[-1][0])
  return fig


def visualize_grid_sparsification_mask(
        mask: torch.Tensor,
        rows: int,
        cols: int,
) -> plt.Figure:
  num_channels, num_filters = mask.shape[1], mask.shape[2]
  aspect_ratio = cols / rows
  fig_height = 8
  fig_width = fig_height * aspect_ratio
  suptitle_fontsize, title_fontsize, label_fontsize, caption_fontsize = _scaled_fontsizes(fig_height)
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
      ax.set_title(f'Filter {f} — Channel {c}', fontsize=title_fontsize)
      ax.set_xlabel('cols', fontsize=label_fontsize)
      ax.set_ylabel('rows', fontsize=label_fontsize)
      _add_caption(ax, f'Kept {keep_ratio:.1%} of nodes for this (channel, filter) pair.', fontsize=caption_fontsize)

  fig.suptitle('Sparsification Mask', fontsize=suptitle_fontsize, fontweight='bold')
  legend_handles = [Patch(facecolor='black', edgecolor='gray', label='Kept'),
                   Patch(facecolor='white', edgecolor='gray', label='Discarded')]
  fig.legend(handles=legend_handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.93),
            fontsize=label_fontsize)
  fig.tight_layout(rect=(0, 0, 1, 0.90))
  fig.subplots_adjust(bottom=0.15 / num_channels)
  return fig
