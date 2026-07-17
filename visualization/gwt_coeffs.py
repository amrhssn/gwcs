import os
from typing import List, Optional

import torch
import matplotlib.pyplot as plt

from graph_wavelet_transform.energy import energy_distribution_across_scales


def visualize_grid_graph_sgwt_coefficients(
        coeffs: torch.Tensor,
        num_filters: int,
        scales: List[float],
        rows: int,
        cols: int,
) -> None:
    aspect_ratio = cols / rows
    fig_height = 4
    fig_width = fig_height * aspect_ratio
    fig, axes = plt.subplots(1, num_filters, figsize=(2 * fig_width * num_filters, 2 * fig_height))
    if num_filters == 1:
        axes = [axes]
    for i in range(num_filters):
        scale_coeffs = coeffs[:, i].reshape(rows, cols)
        im = axes[i].imshow(scale_coeffs, aspect='auto', cmap='RdBu_r')
        if i == 0:
            axes[i].set_title('Scaling Function')
        else:
            axes[i].set_title(f'Wavelet Scale {i} (s={scales[i-1]:.2f})')
        axes[i].set_xlabel('cols')
        axes[i].set_ylabel('rows')
        plt.colorbar(im, ax=axes[i])
    plt.tight_layout()
    plt.show()


def visualize_energy_distribution_across_scales(coeffs: torch.Tensor) -> None:
    energy_distribution = energy_distribution_across_scales(coeffs=coeffs)
    scales = list(range(coeffs.shape[1]))
    fig, ax = plt.subplots(figsize=(max(6, len(scales) * 2.0), 4))
    ax.bar(scales, energy_distribution.cpu().numpy(), color="orange")
    ax.set_xlabel("Scale index")
    ax.set_ylabel("Energy ratio")
    ax.set_title("Energy distribution across scales")
    ax.set_xticks(scales)
    ax.set_ylim(0, 1)
    for i, v in enumerate(energy_distribution.cpu().numpy()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.show()


def visualize_grid_graph_inverse_sgwt_reconstruction(
        x_recon: torch.Tensor,
        x: torch.Tensor,
        rows: int,
        cols: int,
        save_fig: bool = False,
        filename_prefix: Optional[str] = None,   # <-- NEW: e.g. "sparse_vgae_sample0"
        save_dir: Optional[str] = None,           # <-- NEW: override default 'gwt_figures'
):
    x = x.squeeze()
    x_recon = x_recon.squeeze()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(x.reshape(rows, cols), aspect='auto', cmap='RdBu_r')
    axes[0].set_title('Original Signal (x)')
    axes[0].set_xlabel('cols')
    axes[0].set_ylabel('rows')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(x_recon.reshape(rows, cols), aspect='auto', cmap='RdBu_r')
    axes[1].set_title('Reconstruction (x_recon)')
    axes[1].set_xlabel('cols')
    axes[1].set_ylabel('rows')
    plt.colorbar(im1, ax=axes[1])

    error = x - x_recon
    im2 = axes[2].imshow(torch.abs(error).reshape(rows, cols), aspect='auto', cmap='Reds')
    axes[2].set_title('Absolute Reconstruction Error |x - x_recon|')
    axes[2].set_xlabel('cols')
    axes[2].set_ylabel('rows')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_fig:
        out_dir = save_dir if save_dir else 'gwt_figures'
        os.makedirs(out_dir, exist_ok=True)
        prefix = filename_prefix if filename_prefix else f'reconstruction_{rows}x{cols}'
        filepath = os.path.join(out_dir, f'{prefix}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")

    plt.show()

    print(f"Max absolute error: {torch.abs(error).max().item():.6f}")
    print(f"Mean squared error: {torch.mean(error ** 2).item():.6f}")
    print(f"RMSE: {torch.sqrt(torch.mean(error ** 2)).item():.6f}")