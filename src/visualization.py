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
  return SUPTITLE_FONTSIZE * scale, TITLE_FONTSIZE * scale, LABEL_FONTSIZE * scale, CAPTION_FONTSIZE * scale


def _add_header(
        fig: plt.Figure,
        fig_height: float,
        title: str,
        title_fontsize: float = SUPTITLE_FONTSIZE,
        caption: Optional[str] = None,
        caption_fontsize: float = CAPTION_FONTSIZE,
        legend_handles: Optional[list[Patch]] = None,
        legend_fontsize: float = LABEL_FONTSIZE,
) -> float:
  """Stacks a suptitle, an optional caption, and an optional legend at the
  top-center of the figure. Offsets are sized in inches (from the fontsizes
  actually used) rather than figure fractions, so the header's absolute size
  tracks its own content and stays constant as fig_height grows with
  num_channels/num_filters, instead of overlapping or leaving an
  ever-larger gap above the plots. Returns the top y-fraction the plot area
  should stop at, for use as `fig.tight_layout(rect=(0, 0, 1, plot_top))`.
  """
  margin_in = 0.08
  offset_in = margin_in + title_fontsize / 72 * 1.4
  fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=1 - margin_in / fig_height)

  if caption is not None:
    num_caption_lines = caption.count('\n') + 1
    fig.text(0.5, 1 - offset_in / fig_height, caption, ha='center', va='top', fontsize=caption_fontsize, wrap=True)
    offset_in += caption_fontsize / 72 * 1.6 * num_caption_lines

  if legend_handles is not None:
    offset_in += 0.1
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(legend_handles),
              bbox_to_anchor=(0.5, 1 - offset_in / fig_height), fontsize=legend_fontsize)
    offset_in += legend_fontsize / 72 * 1.6

  return 1 - (offset_in + margin_in) / fig_height


def _sparsity_caption(
        num_nodes: int,
        num_filters: Optional[int] = None,
        keep_ratio: Optional[float] = None,
) -> str:
  parts = [f'N={num_nodes} nodes']
  if num_filters is not None:
    parts.append(f'filters={num_filters}')
  if keep_ratio is not None:
    parts.append(f'keep_ratio={keep_ratio:.3f}')
  if num_filters is not None and keep_ratio is not None:
    num_kept = keep_ratio * num_filters * num_nodes
    compression_ratio = num_kept / num_nodes
    parts.append(f'kept≈{num_kept:.0f} numbers vs. {num_nodes} original (compression_ratio={compression_ratio:.3f})')
  return ' | '.join(parts)


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

  fig_height = 5
  fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, fig_height), squeeze=False)
  for c in range(num_channels):
    ax = axes[0][c]
    channel_signal = signal[:, c].numpy()
    nodes = nx.draw_networkx_nodes(graph, pos=layout, ax=ax, node_size=80,
                                   node_color=channel_signal, cmap='RdBu_r')
    nx.draw_networkx_edges(graph, pos=layout, ax=ax, edge_color='lightgray', width=0.3)
    fig.colorbar(nodes, ax=ax, label='Signal Value')
    ax.set_title(f'Channel {c} (range [{channel_signal.min():.3f}, {channel_signal.max():.3f}])',
                fontsize=TITLE_FONTSIZE)
    ax.set_aspect('equal')
    ax.axis('off')

  plot_top = _add_header(fig, fig_height=fig_height,
                         title=f'{graph_type.replace("_", " ").title()} Graph Signal ({num_nodes} nodes)')
  fig.tight_layout(rect=(0, 0, 1, plot_top))
  return fig


def visualize_grid_reconstruction(
        reconstructed: torch.Tensor,
        signal: torch.Tensor,
        rows: int,
        cols: int,
        gwt_mode: Literal['exact', 'chebyshev'],
        num_filters: Optional[int] = None,
        keep_ratio: Optional[float] = None,
) -> plt.Figure:
  num_channels = signal.shape[1]
  num_nodes = rows * cols
  sparsity_caption = _sparsity_caption(num_nodes, num_filters, keep_ratio)
  fig_height = 5 * num_channels
  fig, axes = plt.subplots(num_channels, 3, figsize=(18, fig_height), squeeze=False)
  error_lines = []

  for c in range(num_channels):
    channel_signal = signal[:, c].reshape(rows, cols)
    channel_reconstructed = reconstructed[:, c].reshape(rows, cols)
    error = channel_signal - channel_reconstructed
    abs_error = torch.abs(error)
    ax0, ax1, ax2 = axes[c]

    im0 = ax0.imshow(channel_signal.numpy(), aspect='auto', cmap='RdBu_r')
    ax0.set_title(f'Signal — Channel {c}', fontsize=TITLE_FONTSIZE)
    ax0.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax0.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im0, ax=ax0)

    im1 = ax1.imshow(channel_reconstructed.numpy(), aspect='auto', cmap='RdBu_r')
    ax1.set_title('Reconstruction', fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(abs_error.numpy(), aspect='auto', cmap='Reds')
    mse = torch.mean(error ** 2).item()
    ax2.set_title('Recon Error |x - x_recon|', fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im2, ax=ax2)
    error_lines.append(f'Channel {c}: max {abs_error.max().item():.4f} | MSE {mse:.4f} | RMSE {mse ** 0.5:.4f}')

  caption = sparsity_caption + '\n' + ' | '.join(error_lines)
  plot_top = _add_header(fig, fig_height=fig_height, title=f'Signal vs. Reconstruction — {gwt_mode.title()} SGWT',
                         caption=caption)
  fig.tight_layout(rect=(0, 0, 1, plot_top))
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
  num_nodes = rows * cols
  sparsity_caption = _sparsity_caption(num_nodes, num_filters)
  num_columns = num_filters + 1
  aspect_ratio = cols / rows
  panel_height = 8
  panel_width = panel_height * aspect_ratio
  # Panels are rendered at 2x `panel_height`/`panel_width` below, so fonts
  # must be scaled against that actual per-row height, not the nominal one.
  row_height = 2 * panel_height
  fig_height = row_height * num_channels
  suptitle_fontsize, title_fontsize, label_fontsize, caption_fontsize = _scaled_fontsizes(row_height)
  fig, axes = plt.subplots(num_channels, num_columns,
                           figsize=(2 * panel_width * num_columns, fig_height),
                           squeeze=False)

  for c in range(num_channels):
    signal_ax = axes[c][0]
    channel_signal = signal[:, c].reshape(rows, cols)
    signal_im = signal_ax.imshow(channel_signal.numpy(), aspect='auto', cmap='RdBu_r')
    signal_ax.set_title(f'Signal — Channel {c}', fontsize=title_fontsize)
    signal_ax.set_xlabel('cols', fontsize=label_fontsize)
    signal_ax.set_ylabel('rows', fontsize=label_fontsize)
    fig.colorbar(signal_im, ax=signal_ax)

    for f in range(num_filters):
      ax = axes[c][f + 1]
      scale_coeffs = coeffs[:, c, f].reshape(rows, cols)
      im = ax.imshow(scale_coeffs.numpy(), aspect='auto', cmap='RdBu_r')
      if f == 0:
        ax.set_title('Scaling Function', fontsize=title_fontsize)
      else:
        ax.set_title(f'Wavelet Scale {f} (s={scales[f - 1]:.2f})', fontsize=title_fontsize)
      ax.set_xlabel('cols', fontsize=label_fontsize)
      ax.set_ylabel('rows', fontsize=label_fontsize)
      fig.colorbar(im, ax=ax)

  plot_top = _add_header(fig, fig_height=fig_height, title=f'SGWT Coefficients — {gwt_mode.title()} SGWT',
                         title_fontsize=suptitle_fontsize, caption=sparsity_caption,
                         caption_fontsize=caption_fontsize)
  fig.tight_layout(rect=(0, 0, 1, plot_top))
  _add_column_separator(fig, left_ax=axes[0][0], right_ax=axes[0][1], top_ax=axes[0][0], bottom_ax=axes[-1][0])
  return fig


def _draw_graph_panel(
        ax: plt.Axes,
        graph: nx.Graph,
        layout: dict,
        values,
        cmap,
        vmin: float | None = None,
        vmax: float | None = None,
        edge_color: str = 'lightgray',
        edge_width: float = 0.3,
        node_size: float = 30,
):
  nodes = nx.draw_networkx_nodes(graph, pos=layout, ax=ax, node_size=node_size,
                                 node_color=values, cmap=cmap, vmin=vmin, vmax=vmax)
  nx.draw_networkx_edges(graph, pos=layout, ax=ax, edge_color=edge_color, width=edge_width)
  ax.axis('off')
  return nodes


def _build_networkx_graph(num_nodes: int, edge_index: torch.Tensor) -> nx.Graph:
  graph = nx.Graph()
  graph.add_nodes_from(range(num_nodes))
  graph.add_edges_from(edge_index.t().numpy())
  return graph


def visualize_graph_reconstruction(
        reconstructed: torch.Tensor,
        signal: torch.Tensor,
        edge_index: torch.Tensor,
        layout: dict,
        gwt_mode: Literal['exact', 'chebyshev'],
        num_filters: Optional[int] = None,
        keep_ratio: Optional[float] = None,
) -> plt.Figure:
  num_channels = signal.shape[1]
  num_nodes = signal.shape[0]
  sparsity_caption = _sparsity_caption(num_nodes, num_filters, keep_ratio)
  graph = _build_networkx_graph(num_nodes, edge_index)
  fig_height = 5 * num_channels
  fig, axes = plt.subplots(num_channels, 3, figsize=(18, fig_height), squeeze=False)
  error_lines = []

  for c in range(num_channels):
    channel_signal = signal[:, c].numpy()
    channel_reconstructed = reconstructed[:, c].numpy()
    abs_error = abs(channel_signal - channel_reconstructed)
    ax0, ax1, ax2 = axes[c]

    nodes0 = _draw_graph_panel(ax0, graph, layout, channel_signal, 'RdBu_r')
    ax0.set_title(f'Signal — Channel {c}', fontsize=TITLE_FONTSIZE)
    fig.colorbar(nodes0, ax=ax0)

    nodes1 = _draw_graph_panel(ax1, graph, layout, channel_reconstructed, 'RdBu_r')
    ax1.set_title('Reconstruction', fontsize=TITLE_FONTSIZE)
    fig.colorbar(nodes1, ax=ax1)

    nodes2 = _draw_graph_panel(ax2, graph, layout, abs_error, 'Reds')
    mse = (abs_error ** 2).mean().item()
    ax2.set_title('Recon Error |x - x_recon|', fontsize=TITLE_FONTSIZE)
    fig.colorbar(nodes2, ax=ax2)
    error_lines.append(f'Channel {c}: max {abs_error.max().item():.4f} | MSE {mse:.4f} | RMSE {mse ** 0.5:.4f}')

  caption = sparsity_caption + '\n' + ' | '.join(error_lines)
  plot_top = _add_header(fig, fig_height=fig_height, title=f'Signal vs. Reconstruction — {gwt_mode.title()} SGWT',
                         caption=caption)
  fig.tight_layout(rect=(0, 0, 1, plot_top))
  return fig


def visualize_graph_gwt_coefficients(
        coeffs: torch.Tensor,
        signal: torch.Tensor,
        scales: list[float],
        edge_index: torch.Tensor,
        layout: dict,
        gwt_mode: Literal['exact', 'chebyshev'],
) -> plt.Figure:
  num_channels, num_filters = coeffs.shape[1], coeffs.shape[2]
  num_nodes = signal.shape[0]
  sparsity_caption = _sparsity_caption(num_nodes, num_filters)
  graph = _build_networkx_graph(num_nodes, edge_index)
  fig_height = 5 * num_channels
  fig, axes = plt.subplots(num_channels, num_filters + 1,
                           figsize=(6 * (num_filters + 1), fig_height), squeeze=False)

  for c in range(num_channels):
    signal_ax = axes[c][0]
    signal_nodes = _draw_graph_panel(signal_ax, graph, layout, signal[:, c].numpy(), 'RdBu_r')
    signal_ax.set_title(f'Signal — Channel {c}', fontsize=TITLE_FONTSIZE)
    fig.colorbar(signal_nodes, ax=signal_ax)

    for f in range(num_filters):
      ax = axes[c][f + 1]
      nodes = _draw_graph_panel(ax, graph, layout, coeffs[:, c, f].numpy(), 'RdBu_r')
      if f == 0:
        ax.set_title('Scaling Function', fontsize=TITLE_FONTSIZE)
      else:
        ax.set_title(f'Wavelet Scale {f} (s={scales[f - 1]:.2f})', fontsize=TITLE_FONTSIZE)
      fig.colorbar(nodes, ax=ax)

  plot_top = _add_header(fig, fig_height=fig_height, title=f'SGWT Coefficients — {gwt_mode.title()} SGWT',
                         caption=sparsity_caption)
  fig.tight_layout(rect=(0, 0, 1, plot_top))
  return fig


def visualize_graph_sparsification_mask(
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        layout: dict,
        keep_ratio: Optional[float] = None,
) -> plt.Figure:
  num_channels, num_filters = mask.shape[1], mask.shape[2]
  num_nodes = mask.shape[0]
  sparsity_caption = _sparsity_caption(num_nodes, num_filters, keep_ratio)
  graph = _build_networkx_graph(num_nodes, edge_index)
  cmap = ListedColormap(['white', 'black'])
  fig_height = 5 * num_channels
  fig, axes = plt.subplots(num_channels, num_filters, figsize=(6 * num_filters, fig_height), squeeze=False)

  for c in range(num_channels):
    for f in range(num_filters):
      ax = axes[c][f]
      filter_mask = mask[:, c, f].numpy()
      _draw_graph_panel(ax, graph, layout, filter_mask, cmap, vmin=0, vmax=1,
                        edge_color='#999999', edge_width=0.6, node_size=50)
      empirical_keep_ratio = filter_mask.mean()
      ax.set_title(f'Filter {f} (kept {empirical_keep_ratio:.1%})', fontsize=TITLE_FONTSIZE)

  legend_handles = [Patch(facecolor='black', edgecolor='gray', label='Kept'),
                   Patch(facecolor='white', edgecolor='gray', label='Discarded')]
  plot_top = _add_header(fig, fig_height=fig_height, title='Sparsification Mask', caption=sparsity_caption,
                         legend_handles=legend_handles)
  fig.tight_layout(rect=(0, 0, 1, plot_top))
  return fig


def visualize_grid_sparsification_mask(
        mask: torch.Tensor,
        rows: int,
        cols: int,
        keep_ratio: Optional[float] = None,
) -> plt.Figure:
  num_channels, num_filters = mask.shape[1], mask.shape[2]
  num_nodes = rows * cols
  sparsity_caption = _sparsity_caption(num_nodes, num_filters, keep_ratio)
  aspect_ratio = cols / rows
  panel_height = 8
  panel_width = panel_height * aspect_ratio
  row_height = 2 * panel_height
  fig_height = row_height * num_channels
  suptitle_fontsize, title_fontsize, label_fontsize, caption_fontsize = _scaled_fontsizes(row_height)
  fig, axes = plt.subplots(num_channels, num_filters,
                           figsize=(2 * panel_width * num_filters, fig_height),
                           squeeze=False)
  cmap = ListedColormap(['white', 'black'])

  for c in range(num_channels):
    for f in range(num_filters):
      ax = axes[c][f]
      filter_mask = mask[:, c, f].reshape(rows, cols)
      ax.imshow(filter_mask.numpy(), aspect='auto', cmap=cmap, vmin=0, vmax=1)
      empirical_keep_ratio = filter_mask.float().mean().item()
      ax.set_title(f'Filter {f} (kept {empirical_keep_ratio:.1%})', fontsize=title_fontsize)
      ax.set_xlabel('cols', fontsize=label_fontsize)
      ax.set_ylabel('rows', fontsize=label_fontsize)

  legend_handles = [Patch(facecolor='black', edgecolor='gray', label='Kept'),
                   Patch(facecolor='white', edgecolor='gray', label='Discarded')]
  plot_top = _add_header(fig, fig_height=fig_height, title='Sparsification Mask', title_fontsize=suptitle_fontsize,
                         caption=sparsity_caption, caption_fontsize=caption_fontsize,
                         legend_handles=legend_handles, legend_fontsize=label_fontsize)
  fig.tight_layout(rect=(0, 0, 1, plot_top))
  return fig
