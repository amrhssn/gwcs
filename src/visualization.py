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
from matplotlib.ticker import MaxNLocator


SUPTITLE_FONTSIZE = 36
TITLE_FONTSIZE = 17
LABEL_FONTSIZE = 15
CAPTION_FONTSIZE = 14
_FONT_SIZE_REFERENCE_HEIGHT = 4.0
JOINT_TITLE_FONTSIZE = 11
JOINT_ROW_LABEL_FONTSIZE = 12
JOINT_CAPTION_FONTSIZE = 9
JOINT_COLORBAR_TICK_FONTSIZE = 6.5
JOINT_MASK_PCT_FONTSIZE = 8.5
JOINT_ROW_GAP_IN = 0.28
JOINT_ROW_GAP_TO_RECON_IN = 0.36



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
  margin_in = 0.08
  offset_in = margin_in + title_fontsize / 72 * 1.4
  header_artists = [fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=1 - margin_in / fig_height)]

  if caption is not None:
    num_caption_lines = caption.count('\n') + 1
    header_artists.append(fig.text(0.5, 1 - offset_in / fig_height, caption, ha='center', va='top',
                                   fontsize=caption_fontsize, wrap=True))
    offset_in += caption_fontsize / 72 * 1.6 * num_caption_lines

  if legend_handles is not None:
    offset_in += 0.1
    header_artists.append(fig.legend(handles=legend_handles, loc='upper center', ncol=len(legend_handles),
                                     bbox_to_anchor=(0.5, 1 - offset_in / fig_height), fontsize=legend_fontsize))

  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  lowest_bottom_px = min(artist.get_window_extent(renderer).y0 for artist in header_artists)
  return max(0.0, (lowest_bottom_px - margin_in * fig.dpi) / fig.bbox.height)


def _sparsity_caption(
        num_nodes: int,
        num_filters: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        compression_ratio: Optional[float] = None,
) -> str:
  parts = [f'N={num_nodes} nodes']
  if num_filters is not None:
    parts.append(f'filters={num_filters}')
  if keep_ratio is not None:
    parts.append(f'keep_ratio={keep_ratio:.3f}')
  if num_filters is not None and keep_ratio is not None:
    num_kept = keep_ratio * num_filters * num_nodes
    derived_compression_ratio = num_kept / num_nodes
    parts.append(f'kept≈{num_kept:.0f} numbers vs. {num_nodes} original '
                 f'(compression_ratio={derived_compression_ratio:.3f})')
  elif compression_ratio is not None:
    parts.append(f'compression_ratio={compression_ratio:.3f}')
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
  fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, fig_height), squeeze=False,
                           layout='constrained')
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
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
  return fig


def visualize_grid_reconstruction(
        reconstructed: torch.Tensor,
        signal: torch.Tensor,
        rows: int,
        cols: int,
        gwt_mode: Optional[Literal['exact', 'chebyshev']] = None,
        num_filters: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        compression_ratio: Optional[float] = None,
) -> plt.Figure:
  num_channels = signal.shape[1]
  num_nodes = rows * cols
  sparsity_caption = _sparsity_caption(num_nodes, num_filters, keep_ratio, compression_ratio)
  aspect_ratio = cols / rows
  panel_height = 5
  panel_width = panel_height * aspect_ratio
  fig_height = panel_height * num_channels
  fig, axes = plt.subplots(num_channels, 3, figsize=(panel_width * 3, fig_height), squeeze=False,
                           layout='constrained')
  error_lines = []

  for c in range(num_channels):
    channel_signal = signal[:, c].reshape(rows, cols)
    channel_reconstructed = reconstructed[:, c].reshape(rows, cols)
    error = channel_signal - channel_reconstructed
    abs_error = torch.abs(error)
    ax0, ax1, ax2 = axes[c]

    im0 = ax0.imshow(channel_signal.numpy(), aspect='equal', cmap='RdBu_r')
    ax0.set_title(f'Signal — Channel {c}', fontsize=TITLE_FONTSIZE)
    ax0.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax0.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im0, ax=ax0)

    im1 = ax1.imshow(channel_reconstructed.numpy(), aspect='equal', cmap='RdBu_r')
    ax1.set_title('Reconstruction', fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(abs_error.numpy(), aspect='equal', cmap='Reds')
    mse = torch.mean(error ** 2).item()
    ax2.set_title('Recon Error |x - x_recon|', fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
    fig.colorbar(im2, ax=ax2)
    error_lines.append(f'Channel {c}: max {abs_error.max().item():.4f} | MSE {mse:.4f} | RMSE {mse ** 0.5:.4f}')

  caption = sparsity_caption + '\n' + ' | '.join(error_lines)
  title = f'Signal vs. Reconstruction — {gwt_mode.title()} SGWT' if gwt_mode is not None else 'Signal vs. Reconstruction'
  plot_top = _add_header(fig, fig_height=fig_height, title=title, caption=caption)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
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
  row_height = panel_height
  fig_height = row_height * num_channels
  suptitle_fontsize, title_fontsize, label_fontsize, caption_fontsize = _scaled_fontsizes(row_height)
  fig, axes = plt.subplots(num_channels, num_columns,
                           figsize=(panel_width * num_columns, fig_height),
                           squeeze=False, layout='constrained')

  for c in range(num_channels):
    signal_ax = axes[c][0]
    channel_signal = signal[:, c].reshape(rows, cols)
    signal_im = signal_ax.imshow(channel_signal.numpy(), aspect='equal', cmap='RdBu_r')
    signal_ax.set_title(f'Signal — Channel {c}', fontsize=title_fontsize)
    signal_ax.set_xlabel('cols', fontsize=label_fontsize)
    signal_ax.set_ylabel('rows', fontsize=label_fontsize)
    fig.colorbar(signal_im, ax=signal_ax)

    for f in range(num_filters):
      ax = axes[c][f + 1]
      scale_coeffs = coeffs[:, c, f].reshape(rows, cols)
      im = ax.imshow(scale_coeffs.numpy(), aspect='equal', cmap='RdBu_r')
      if f == 0:
        ax.set_title('Scaling Function', fontsize=title_fontsize)
      else:
        ax.set_title(f'Scale {f} (s={scales[f - 1]:.2f})', fontsize=title_fontsize)
      ax.set_xlabel('cols', fontsize=label_fontsize)
      ax.set_ylabel('rows', fontsize=label_fontsize)
      fig.colorbar(im, ax=ax)

  plot_top = _add_header(fig, fig_height=fig_height, title=f'SGWT Coefficients — {gwt_mode.title()} SGWT',
                         title_fontsize=suptitle_fontsize, caption=sparsity_caption,
                         caption_fontsize=caption_fontsize)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
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
  ax.set_aspect('equal')
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
        gwt_mode: Optional[Literal['exact', 'chebyshev']] = None,
        num_filters: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        compression_ratio: Optional[float] = None,
) -> plt.Figure:
  num_channels = signal.shape[1]
  num_nodes = signal.shape[0]
  sparsity_caption = _sparsity_caption(num_nodes, num_filters, keep_ratio, compression_ratio)
  graph = _build_networkx_graph(num_nodes, edge_index)
  fig_height = 5 * num_channels
  fig, axes = plt.subplots(num_channels, 3, figsize=(18, fig_height), squeeze=False, layout='constrained')
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
  title = f'Signal vs. Reconstruction — {gwt_mode.title()} SGWT' if gwt_mode is not None else 'Signal vs. Reconstruction'
  plot_top = _add_header(fig, fig_height=fig_height, title=title, caption=caption)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
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
                           figsize=(6 * (num_filters + 1), fig_height), squeeze=False, layout='constrained')

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
        ax.set_title(f'Scale {f} (s={scales[f - 1]:.2f})', fontsize=TITLE_FONTSIZE)
      fig.colorbar(nodes, ax=ax)

  plot_top = _add_header(fig, fig_height=fig_height, title=f'SGWT Coefficients — {gwt_mode.title()} SGWT',
                         caption=sparsity_caption)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
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
  fig, axes = plt.subplots(num_channels, num_filters, figsize=(6 * num_filters, fig_height), squeeze=False,
                           layout='constrained')

  for c in range(num_channels):
    for f in range(num_filters):
      ax = axes[c][f]
      filter_mask = mask[:, c, f].numpy()
      node_size = 1 + filter_mask * 49
      _draw_graph_panel(ax, graph, layout, filter_mask, cmap, vmin=0, vmax=1,
                        edge_color='#999999', edge_width=0.6, node_size=node_size)
      empirical_keep_ratio = filter_mask.mean()
      ax.set_title(f'Filter {f} (kept {empirical_keep_ratio:.1%})', fontsize=TITLE_FONTSIZE)

  legend_handles = [Patch(facecolor='black', edgecolor='gray', label='Kept'),
                   Patch(facecolor='white', edgecolor='gray', label='Discarded')]
  plot_top = _add_header(fig, fig_height=fig_height, title='Sparsification Mask', caption=sparsity_caption,
                         legend_handles=legend_handles)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
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
  row_height = panel_height
  fig_height = row_height * num_channels
  suptitle_fontsize, title_fontsize, label_fontsize, caption_fontsize = _scaled_fontsizes(row_height)
  fig, axes = plt.subplots(num_channels, num_filters,
                           figsize=(panel_width * num_filters, fig_height),
                           squeeze=False, layout='constrained')
  cmap = ListedColormap(['white', 'black'])

  for c in range(num_channels):
    for f in range(num_filters):
      ax = axes[c][f]
      filter_mask = mask[:, c, f].reshape(rows, cols)
      ax.imshow(filter_mask.numpy(), aspect='equal', cmap=cmap, vmin=0, vmax=1)
      empirical_keep_ratio = filter_mask.float().mean().item()
      ax.set_title(f'Filter {f} (kept {empirical_keep_ratio:.1%})', fontsize=title_fontsize)
      ax.set_xlabel('cols', fontsize=label_fontsize)
      ax.set_ylabel('rows', fontsize=label_fontsize)

  legend_handles = [Patch(facecolor='black', edgecolor='gray', label='Kept'),
                   Patch(facecolor='white', edgecolor='gray', label='Discarded')]
  plot_top = _add_header(fig, fig_height=fig_height, title='Sparsification Mask', title_fontsize=suptitle_fontsize,
                         caption=sparsity_caption, caption_fontsize=caption_fontsize,
                         legend_handles=legend_handles, legend_fontsize=label_fontsize)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
  return fig


def _draw_sgae_mask_matrix(ax: plt.Axes, mask: torch.Tensor, latent_dim: int) -> None:
  cmap = ListedColormap(['white', 'black'])
  ax.imshow(mask.numpy().T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none')
  ax.set_title(f'Latent Sparsity Mask ({latent_dim} dims x nodes)', fontsize=TITLE_FONTSIZE)
  ax.set_xlabel('node index', fontsize=LABEL_FONTSIZE)
  ax.set_ylabel('latent dim', fontsize=LABEL_FONTSIZE)


def visualize_graph_sgae_mask(
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        layout: dict,
        compression_ratio: Optional[float] = None,
) -> plt.Figure:
  num_nodes, latent_dim = mask.shape
  sparsity_caption = _sparsity_caption(num_nodes, compression_ratio=compression_ratio)
  graph = _build_networkx_graph(num_nodes, edge_index)
  fig_height = 6
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, fig_height), layout='constrained')

  kept_per_node = mask.float().mean(dim=1).numpy()
  nodes = _draw_graph_panel(ax0, graph, layout, kept_per_node, 'viridis', vmin=0, vmax=1)
  ax0.set_title('Fraction of Latent Dims Kept per Node', fontsize=TITLE_FONTSIZE)
  fig.colorbar(nodes, ax=ax0)

  _draw_sgae_mask_matrix(ax1, mask, latent_dim)

  legend_handles = [Patch(facecolor='black', edgecolor='gray', label='Kept'),
                   Patch(facecolor='white', edgecolor='gray', label='Discarded')]
  plot_top = _add_header(fig, fig_height=fig_height, title='Sparsification Mask (SGAE Latent Space)',
                         caption=sparsity_caption, legend_handles=legend_handles)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
  return fig


def visualize_grid_sgae_mask(
        mask: torch.Tensor,
        rows: int,
        cols: int,
        compression_ratio: Optional[float] = None,
) -> plt.Figure:
  num_nodes, latent_dim = mask.shape
  sparsity_caption = _sparsity_caption(num_nodes, compression_ratio=compression_ratio)
  fig_height = 6
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, fig_height), layout='constrained')

  kept_per_node = mask.float().mean(dim=1).reshape(rows, cols).numpy()
  im0 = ax0.imshow(kept_per_node, aspect='equal', cmap='viridis', vmin=0, vmax=1)
  ax0.set_title('Fraction of Latent Dims Kept per Node', fontsize=TITLE_FONTSIZE)
  ax0.set_xlabel('cols', fontsize=LABEL_FONTSIZE)
  ax0.set_ylabel('rows', fontsize=LABEL_FONTSIZE)
  fig.colorbar(im0, ax=ax0)

  _draw_sgae_mask_matrix(ax1, mask, latent_dim)

  legend_handles = [Patch(facecolor='black', edgecolor='gray', label='Kept'),
                   Patch(facecolor='white', edgecolor='gray', label='Discarded')]
  plot_top = _add_header(fig, fig_height=fig_height, title='Sparsification Mask (SGAE Latent Space)',
                         caption=sparsity_caption, legend_handles=legend_handles)
  fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))
  return fig


def _paper_column_titles(scales: list[float]) -> list[str]:
  return ['Scaling'] + [f'Scale (s={s:.2f})' for s in scales]


def _paper_figure_caption(
        num_nodes: int,
        channel: int,
        compression_ratio: Optional[float],
        max_error: float,
        mse: float,
        num_filters: Optional[int] = None,
        gwt_mode: Optional[str] = None,
        mask_legend: str = 'mask: black=kept, white=discarded',
) -> str:
  parts = [f'N={num_nodes} nodes', f'channel={channel}']
  if num_filters is not None:
    parts.append(f'filters={num_filters}')
  if compression_ratio is not None:
    parts.append(f'compression_ratio={compression_ratio:.3f}')
  if gwt_mode is not None:
    parts.append(f'{gwt_mode.title()} SGWT')
  parts.append(f'error: max {max_error:.4f}, MSE {mse:.4f}, RMSE {mse ** 0.5:.4f}')
  parts.append(mask_legend)
  return ' | '.join(parts)


def _paper_inset_colorbar(fig: plt.Figure, ax: plt.Axes, mappable) -> None:
  cax = ax.inset_axes((1.08, 0.06, 0.08, 0.88))
  cb = fig.colorbar(mappable, cax=cax)
  cb.locator = MaxNLocator(nbins=3)
  cb.update_ticks()
  cb.ax.tick_params(labelsize=JOINT_COLORBAR_TICK_FONTSIZE)


def _build_paper_figure_skeleton(
        num_cols: int,
        panel_width_in: float,
        panel_height_in: float,
        row_specs: list[tuple[str, list[int]]],
        row_gaps_in: list[float],
) -> tuple[plt.Figure, dict[tuple[int, int], plt.Axes], float, float, float]:
  assert len(row_gaps_in) == len(row_specs) - 1, 'need exactly one gap between each pair of rows'
  left_margin_in = 0.4
  right_margin_in = 0.95
  col_gap_in = 0.95
  title_headroom_in = 0.34
  bottom_margin_in = 0.6

  num_rows = len(row_specs)
  fig_width_in = (left_margin_in + num_cols * panel_width_in + (num_cols - 1) * col_gap_in
                 + right_margin_in)
  fig_height_in = (title_headroom_in + num_rows * panel_height_in + sum(row_gaps_in) + bottom_margin_in)

  fig = plt.figure(figsize=(fig_width_in, fig_height_in))

  row_y_top_in = []
  cursor_in = title_headroom_in
  for row in range(num_rows):
    row_y_top_in.append(cursor_in)
    cursor_in += panel_height_in
    if row < num_rows - 1:
      cursor_in += row_gaps_in[row]

  def rect(col: int, row: int) -> tuple[float, float, float, float]:
    x0_in = left_margin_in + col * (panel_width_in + col_gap_in)
    y_top_in = row_y_top_in[row]
    return (x0_in / fig_width_in,
            1 - (y_top_in + panel_height_in) / fig_height_in,
            panel_width_in / fig_width_in,
            panel_height_in / fig_height_in)

  axes: dict[tuple[int, int], plt.Axes] = {}
  for row, (_, cols_used) in enumerate(row_specs):
    for col in cols_used:
      axes[(row, col)] = fig.add_axes(rect(col, row))

  row_label_x_in = left_margin_in * 0.3
  for row, (label, _) in enumerate(row_specs):
    y_center_in = row_y_top_in[row] + panel_height_in / 2
    fig.text(row_label_x_in / fig_width_in, 1 - y_center_in / fig_height_in, label,
             rotation=90, ha='center', va='center', fontsize=JOINT_ROW_LABEL_FONTSIZE, fontweight='bold')

  return fig, axes, bottom_margin_in, fig_width_in, fig_height_in


def _paper_caption(fig: plt.Figure, caption: str, bottom_margin_in: float, fig_height_in: float) -> None:
  fig.text(0.5, bottom_margin_in * 0.45 / fig_height_in, caption, ha='center', va='center',
           fontsize=JOINT_CAPTION_FONTSIZE, wrap=True)


def visualize_grid_joint_figure(
        signal: torch.Tensor,
        reconstructed: torch.Tensor,
        coeffs: torch.Tensor,
        mask: torch.Tensor,
        rows: int,
        cols: int,
        scales: list[float],
        compression_ratio: Optional[float] = None,
        gwt_mode: Optional[Literal['exact', 'chebyshev']] = None,
        channel: int = 0,
) -> plt.Figure:
  num_filters = coeffs.shape[2]
  assert num_filters == len(scales) + 1, 'coeffs must hold the scaling function plus one entry per scale'
  num_nodes = rows * cols
  if compression_ratio is None:
    compression_ratio = float(mask.float().mean()) * num_filters
  aspect_ratio = cols / rows
  panel_height_in = 2.3
  panel_width_in = panel_height_in * aspect_ratio

  recon_col0 = max(0, (num_filters - 3) // 2)
  recon_cols = [recon_col0, recon_col0 + 1, recon_col0 + 2]
  row_specs = [('SGWT', list(range(num_filters))), ('Mask', list(range(num_filters))),
              ('Reconstruction', recon_cols)]
  fig, axes, bottom_margin_in, _, fig_height_in = _build_paper_figure_skeleton(
    num_filters, panel_width_in, panel_height_in, row_specs,
    [JOINT_ROW_GAP_IN, JOINT_ROW_GAP_TO_RECON_IN])

  signal_grid = signal[:, channel].reshape(rows, cols).numpy()
  reconstructed_grid = reconstructed[:, channel].reshape(rows, cols).numpy()
  error_grid = abs(signal_grid - reconstructed_grid)
  column_titles = _paper_column_titles(scales)
  mask_cmap = ListedColormap(['white', 'black'])

  for f in range(num_filters):
    ax = axes[(0, f)]
    im = ax.imshow(coeffs[:, channel, f].reshape(rows, cols).numpy(), aspect='equal', cmap='RdBu_r')
    ax.set_title(column_titles[f], fontsize=JOINT_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    _paper_inset_colorbar(fig, ax, im)

    mask_ax = axes[(1, f)]
    filter_mask = mask[:, channel, f].numpy()
    mask_ax.imshow(filter_mask.reshape(rows, cols), aspect='equal', cmap=mask_cmap, vmin=0, vmax=1)
    mask_ax.set_title(f'{filter_mask.mean() * 100:.1f}%', fontsize=JOINT_MASK_PCT_FONTSIZE)
    mask_ax.set_xticks([])
    mask_ax.set_yticks([])

  recon_data = [signal_grid, reconstructed_grid, error_grid]
  recon_cmaps = ['RdBu_r', 'RdBu_r', 'Reds']
  for col, data, title, cmap in zip(recon_cols, recon_data, ['Signal', 'Reconstruction', 'Error'], recon_cmaps):
    ax = axes[(2, col)]
    im = ax.imshow(data, aspect='equal', cmap=cmap)
    ax.set_title(title, fontsize=JOINT_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    _paper_inset_colorbar(fig, ax, im)

  mse = float((error_grid ** 2).mean())
  caption = _paper_figure_caption(num_nodes=num_nodes, channel=channel, compression_ratio=compression_ratio,
                                  max_error=float(error_grid.max()), mse=mse, num_filters=num_filters,
                                  gwt_mode=gwt_mode)
  _paper_caption(fig, caption, bottom_margin_in, fig_height_in)
  return fig


def visualize_graph_joint_figure(
        signal: torch.Tensor,
        reconstructed: torch.Tensor,
        coeffs: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        layout: dict,
        scales: list[float],
        compression_ratio: Optional[float] = None,
        gwt_mode: Optional[Literal['exact', 'chebyshev']] = None,
        channel: int = 0,
) -> plt.Figure:
  num_filters = coeffs.shape[2]
  assert num_filters == len(scales) + 1, 'coeffs must hold the scaling function plus one entry per scale'
  num_nodes = signal.shape[0]
  if compression_ratio is None:
    compression_ratio = float(mask.float().mean()) * num_filters
  graph = _build_networkx_graph(num_nodes, edge_index)
  panel_size_in = 2.6

  recon_col0 = max(0, (num_filters - 3) // 2)
  recon_cols = [recon_col0, recon_col0 + 1, recon_col0 + 2]
  row_specs = [('SGWT', list(range(num_filters))), ('Mask', list(range(num_filters))),
              ('Reconstruction', recon_cols)]
  fig, axes, bottom_margin_in, _, fig_height_in = _build_paper_figure_skeleton(
    num_filters, panel_size_in, panel_size_in, row_specs,
    [JOINT_ROW_GAP_IN, JOINT_ROW_GAP_TO_RECON_IN])

  signal_values = signal[:, channel].numpy()
  reconstructed_values = reconstructed[:, channel].numpy()
  error_values = abs(signal_values - reconstructed_values)
  column_titles = _paper_column_titles(scales)
  mask_cmap = ListedColormap(['white', 'black'])

  for f in range(num_filters):
    ax = axes[(0, f)]
    nodes = _draw_graph_panel(ax, graph, layout, coeffs[:, channel, f].numpy(), 'RdBu_r')
    ax.set_title(column_titles[f], fontsize=JOINT_TITLE_FONTSIZE)
    _paper_inset_colorbar(fig, ax, nodes)

    mask_ax = axes[(1, f)]
    filter_mask = mask[:, channel, f].numpy()
    node_size = 1 + filter_mask * 49
    _draw_graph_panel(mask_ax, graph, layout, filter_mask, mask_cmap, vmin=0, vmax=1,
                      edge_color='#999999', edge_width=0.5, node_size=node_size)
    mask_ax.set_title(f'{filter_mask.mean() * 100:.1f}%', fontsize=JOINT_MASK_PCT_FONTSIZE)

  recon_data = [signal_values, reconstructed_values, error_values]
  recon_cmaps = ['RdBu_r', 'RdBu_r', 'Reds']
  for col, data, title, cmap in zip(recon_cols, recon_data, ['Signal', 'Reconstruction', 'Error'], recon_cmaps):
    ax = axes[(2, col)]
    nodes = _draw_graph_panel(ax, graph, layout, data, cmap)
    ax.set_title(title, fontsize=JOINT_TITLE_FONTSIZE)
    _paper_inset_colorbar(fig, ax, nodes)

  mse = float((error_values ** 2).mean())
  caption = _paper_figure_caption(num_nodes=num_nodes, channel=channel, compression_ratio=compression_ratio,
                                  max_error=float(error_values.max()), mse=mse, num_filters=num_filters,
                                  gwt_mode=gwt_mode)
  _paper_caption(fig, caption, bottom_margin_in, fig_height_in)
  return fig


def visualize_grid_sgae_joint_figure(
        signal: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: torch.Tensor,
        rows: int,
        cols: int,
        compression_ratio: Optional[float] = None,
        channel: int = 0,
) -> plt.Figure:
  num_nodes = rows * cols
  if compression_ratio is None:
    compression_ratio = float(mask.float().mean()) * mask.shape[-1] / signal.shape[1]
  aspect_ratio = cols / rows
  panel_height_in = 2.3
  panel_width_in = panel_height_in * aspect_ratio

  row_specs = [('Mask', [1]), ('Reconstruction', [0, 1, 2])]
  fig, axes, bottom_margin_in, _, fig_height_in = _build_paper_figure_skeleton(
    3, panel_width_in, panel_height_in, row_specs, [JOINT_ROW_GAP_TO_RECON_IN])

  signal_grid = signal[:, channel].reshape(rows, cols).numpy()
  reconstructed_grid = reconstructed[:, channel].reshape(rows, cols).numpy()
  error_grid = abs(signal_grid - reconstructed_grid)

  kept_per_node = mask.float().mean(dim=1).reshape(rows, cols).numpy()
  mask_ax = axes[(0, 1)]
  im = mask_ax.imshow(kept_per_node, aspect='equal', cmap='viridis', vmin=0, vmax=1)
  mask_ax.set_title(f'{kept_per_node.mean() * 100:.1f}% kept', fontsize=JOINT_MASK_PCT_FONTSIZE)
  mask_ax.set_xticks([])
  mask_ax.set_yticks([])
  _paper_inset_colorbar(fig, mask_ax, im)

  recon_data = [signal_grid, reconstructed_grid, error_grid]
  recon_cmaps = ['RdBu_r', 'RdBu_r', 'Reds']
  for col, data, title, cmap in zip([0, 1, 2], recon_data, ['Signal', 'Reconstruction', 'Error'], recon_cmaps):
    ax = axes[(1, col)]
    im = ax.imshow(data, aspect='equal', cmap=cmap)
    ax.set_title(title, fontsize=JOINT_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    _paper_inset_colorbar(fig, ax, im)

  mse = float((error_grid ** 2).mean())
  caption = _paper_figure_caption(num_nodes=num_nodes, channel=channel, compression_ratio=compression_ratio,
                                  max_error=float(error_grid.max()), mse=mse,
                                  mask_legend='mask color = fraction of latent dims kept per node')
  _paper_caption(fig, caption, bottom_margin_in, fig_height_in)
  return fig


def visualize_graph_sgae_joint_figure(
        signal: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        layout: dict,
        compression_ratio: Optional[float] = None,
        channel: int = 0,
) -> plt.Figure:
  num_nodes = signal.shape[0]
  if compression_ratio is None:
    compression_ratio = float(mask.float().mean()) * mask.shape[-1] / signal.shape[1]
  graph = _build_networkx_graph(num_nodes, edge_index)
  panel_size_in = 2.6

  row_specs = [('Mask', [1]), ('Reconstruction', [0, 1, 2])]
  fig, axes, bottom_margin_in, _, fig_height_in = _build_paper_figure_skeleton(
    3, panel_size_in, panel_size_in, row_specs, [JOINT_ROW_GAP_TO_RECON_IN])

  signal_values = signal[:, channel].numpy()
  reconstructed_values = reconstructed[:, channel].numpy()
  error_values = abs(signal_values - reconstructed_values)

  kept_per_node = mask.float().mean(dim=1).numpy()
  mask_ax = axes[(0, 1)]
  nodes = _draw_graph_panel(mask_ax, graph, layout, kept_per_node, 'viridis', vmin=0, vmax=1)
  mask_ax.set_title(f'{kept_per_node.mean() * 100:.1f}% kept', fontsize=JOINT_MASK_PCT_FONTSIZE)
  _paper_inset_colorbar(fig, mask_ax, nodes)

  recon_data = [signal_values, reconstructed_values, error_values]
  recon_cmaps = ['RdBu_r', 'RdBu_r', 'Reds']
  for col, data, title, cmap in zip([0, 1, 2], recon_data, ['Signal', 'Reconstruction', 'Error'], recon_cmaps):
    ax = axes[(1, col)]
    nodes = _draw_graph_panel(ax, graph, layout, data, cmap)
    ax.set_title(title, fontsize=JOINT_TITLE_FONTSIZE)
    _paper_inset_colorbar(fig, ax, nodes)

  mse = float((error_values ** 2).mean())
  caption = _paper_figure_caption(num_nodes=num_nodes, channel=channel, compression_ratio=compression_ratio,
                                  max_error=float(error_values.max()), mse=mse,
                                  mask_legend='mask color = fraction of latent dims kept per node')
  _paper_caption(fig, caption, bottom_margin_in, fig_height_in)
  return fig
