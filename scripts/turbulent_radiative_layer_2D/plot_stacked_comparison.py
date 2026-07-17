# -*- coding: utf-8 -*-
"""
plot_stacked_comparison.py

For each of the 8 requested test samples, produces ONE figure with 3 rows:
  Row 0: GXN
  Row 1: Sparse-VGAE
  Row 2: NIGWT  (GNNNeuralIGWT)

Each row has 3 columns:  Original Signal | Reconstruction | |Error|

Layout rules
------------
* Y-axis label ("y") appears ONCE on the leftmost column only, via fig.text.
* X-axis label ("x") appears ONCE on the bottom row only, via fig.text.
* Column headers (Original Signal / Reconstruction / Reconstruction Error) at top.
* Row labels (GXN / Sparse-VGAE / NIGWT) on the left side, outside the axes.
* No suptitle wasting space.
* Large fonts throughout for slide visibility.
"""

import sys
import os
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import deque

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Loading
# =============================================================================

def load_test_signals(test_dir: str, field_index: int = 0):
    """Load test signals, coeffs, and shared graph data."""
    test_dir = Path(test_dir)
    shared = torch.load(test_dir / "shared_data.pt", map_location='cpu', weights_only=False)
    edge_index = shared['edge_index']
    num_nodes  = shared['num_nodes']
    num_rows   = shared['num_rows']
    num_cols   = shared['num_columns']

    sample_files = sorted(
        [f for f in test_dir.glob("*.pt") if f.stem.isdigit()],
        key=lambda f: int(f.stem)
    )

    signals = []
    coeffs_list = []
    for f in sample_files:
        s = torch.load(f, map_location='cpu', weights_only=False)
        # .float() handles both float16 (half-precision storage) and float32
        signals.append(s.x[:, field_index].float())
        if hasattr(s, 'coeffs') and s.coeffs is not None:
            coeffs_list.append(s.coeffs[:, field_index, :].float())

    signals = torch.stack(signals, dim=1)         # (N, num_samples)
    coeffs = torch.stack(coeffs_list, dim=2) if coeffs_list else None
    return signals, coeffs, edge_index, num_nodes, num_rows, num_cols


# =============================================================================
# GXN Model Loading and Inference
# =============================================================================

def load_gxn_model(gxn_checkpoint: str, edge_index: torch.Tensor, device):
    from baselines.gxn.neural_sampling import NeuralSampling, NeuralRecovery

    ckpt = torch.load(gxn_checkpoint, map_location=device, weights_only=False)
    hidden_dim   = ckpt.get('hidden_dim',   128)
    max_hops     = ckpt.get('max_hops',     2)
    recovery_K   = ckpt.get('recovery_K',   8)
    recovery_L   = ckpt.get('recovery_L',   5)
    norm_mean    = ckpt.get('norm_mean',    0.0)
    norm_std     = ckpt.get('norm_std',     1.0)

    sampling_model = NeuralSampling(
        in_channels=1, hidden_channels=hidden_dim, max_hops=max_hops
    ).to(device)
    recovery_model = NeuralRecovery(
        in_channels=1, K=recovery_K, L=recovery_L
    ).to(device)

    sampling_model.load_state_dict(ckpt['sampling_model'])
    recovery_model.load_state_dict(ckpt['recovery_model'])
    sampling_model.eval()
    recovery_model.eval()

    return sampling_model, recovery_model, norm_mean, norm_std


def gxn_reconstruct(sampling_model, recovery_model, norm_mean, norm_std,
                    x_orig, edge_index, keep_ratio, device):
    """Reconstruct using GXN. Returns denormalized reconstruction."""
    num_nodes = x_orig.shape[0]
    M = max(1, int(keep_ratio * num_nodes))

    mean_t = torch.tensor(norm_mean, dtype=torch.float32)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32)

    x_norm = (x_orig - mean_t) / std_t
    x_in   = x_norm.unsqueeze(1).to(device)

    with torch.no_grad():
        attention, _, _ = sampling_model(x_in, edge_index)
        scores = attention.squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        _, top_indices = torch.topk(scores, k=M)
        mask_psi = torch.zeros_like(x_in)
        mask_psi[top_indices] = 1.0
        y = mask_psi * attention * x_in
        x_hat_norm = recovery_model(y, edge_index, attention, mask_psi)

    x_hat = x_hat_norm.squeeze().cpu() * std_t + mean_t
    return x_hat


# =============================================================================
# Sparse-VGAE Model Loading and Inference
# =============================================================================

def load_vgae_model(vgae_checkpoint: str, keep_ratio: float, device):
    from baselines.sparse_VGAE.gae import GAE

    ckpt = torch.load(vgae_checkpoint, map_location=device, weights_only=False)
    latent_dim    = ckpt.get('latent_dim',    128)
    hidden_layers = ckpt.get('hidden_layers', 4)
    norm_mean     = ckpt.get('norm_mean',     0.0)
    norm_std      = ckpt.get('norm_std',      1.0)

    model = GAE(
        input_dim=1,
        latent_dim=latent_dim,
        output_dim=1,
        num_encoder_layers=hidden_layers,
        num_decoder_layers=hidden_layers,
        conv_layer_type='SAGEConv',
        activation_function='silu',
        sparsity_ratio=keep_ratio,
        variational=False,
        sparsity_mode='ste_annealing',
        temperature_init=1.0,
        min_temperature=0.01,
        annealing_steps=1000,
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, norm_mean, norm_std


def vgae_reconstruct(model, norm_mean, norm_std, x_orig, edge_index, device):
    """Reconstruct using Sparse-VGAE. Returns denormalized reconstruction."""
    mean_t = torch.tensor(norm_mean, dtype=torch.float32)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32)

    x_norm = (x_orig - mean_t) / std_t
    x_in   = x_norm.unsqueeze(1).to(device)

    with torch.no_grad():
        z_sparse  = model.encode(x_in, edge_index)
        x_hat_norm = model.decode(z_sparse, edge_index)

    x_hat = x_hat_norm.squeeze().cpu() * std_t + mean_t
    return x_hat


# =============================================================================
# NIGWT (GNNNeuralIGWT) Model Loading and Inference
# =============================================================================

def load_gwt_model(gwt_run_dir: str, gwt_checkpoint: str, device):
    run_dir = Path(gwt_run_dir)

    config_path = run_dir / 'reconstructor_pretraining_config_turbulent.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(str(config_path), 'r') as f:
        cfg = json.load(f)

    # Load normalization stats if present
    norm_stats = None
    norm_metadata_path = run_dir / 'normalization_metadata.json'
    if norm_metadata_path.exists():
        with open(str(norm_metadata_path), 'r') as f:
            norm_metadata = json.load(f)
        norm_stats_path = norm_metadata.get('normalization_stats_path')
        if norm_stats_path and Path(norm_stats_path).exists():
            norm_stats = torch.load(norm_stats_path, map_location='cpu', weights_only=False)

    # Load checkpoint
    ckpt_path = Path(gwt_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"NIGWT checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state  = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Get num_filters
    test_data_dir   = Path(cfg['test_data_dir'])
    gwt_config_path = test_data_dir / 'gwt_config.json'
    if gwt_config_path.exists():
        with open(str(gwt_config_path), 'r') as f:
            gwt_config_json = json.load(f)
        num_filters = gwt_config_json['num_filters']
    else:
        num_filters = 5  # default: 4 scales + 1 scaling function

    use_normalization = cfg.get('use_normalization', False)
    model_name        = cfg.get('model_name', 'GNNNeuralIGWT')

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT
    if model_name == 'GUNetNeuralIGWT':
        model = GUNetNeuralIGWT(
            num_filters=num_filters,
            hidden_dim=cfg['hidden_dim'],
            output_dim=cfg['output_dim'],
            activation_function=cfg['activation_function'],
            encoder_type=cfg['encoder_type'],
            fusion_type=cfg.get('fusion_type', 'mlp'),
            decoder_type=cfg['decoder_type'],
            conv_layer_type=cfg['conv_layer_type'],
            processor_num_layers=cfg['processor_num_layers'],
            pooling_type=cfg.get('pooling_type', 'top_k_pooling'),
            pooling_ratio=cfg.get('pooling_ratio', 0.5),
            predict_residual=cfg['predict_residual'],
            enable_scale_embeddings=cfg['enable_scale_embeddings'],
            igwt_mode=cfg['igwt_mode'],
            use_normalization=use_normalization,
            normalization_stats=norm_stats,
        ).to(device)
    else:
        model = GNNNeuralIGWT(
            num_filters=num_filters,
            hidden_dim=cfg['hidden_dim'],
            output_dim=cfg['output_dim'],
            activation_function=cfg['activation_function'],
            encoder_type=cfg['encoder_type'],
            decoder_type=cfg['decoder_type'],
            conv_layer_type=cfg['conv_layer_type'],
            processor_num_layers=cfg['processor_num_layers'],
            predict_residual=cfg['predict_residual'],
            enable_scale_embeddings=cfg['enable_scale_embeddings'],
            igwt_mode=cfg['igwt_mode'],
            use_normalization=use_normalization,
            normalization_stats=norm_stats,
        ).to(device)

    model.load_state_dict(new_state)
    model.eval()
    return model, cfg, norm_stats


def gwt_reconstruct(model, cfg, norm_stats, coeffs_sample, edge_index,
                    matrix_cache, device, field_index=0):
    from models.sparsification_utils import multilevel_importance_sampling

    coeffs_flat   = coeffs_sample.to(device)
    keep_ratio    = cfg.get('keep_ratio', 0.01)
    mask          = multilevel_importance_sampling(coeffs=coeffs_flat, keep_ratio=keep_ratio)
    edge_index_dev = edge_index.to(device)
    batch_vec     = torch.zeros(coeffs_flat.shape[0], dtype=torch.long, device=device)
    ptr           = torch.tensor([0, coeffs_flat.shape[0]], device=device)
    chebyshev_data = matrix_cache.get([0])

    with torch.no_grad():
        x_hat = model(
            coeffs=coeffs_flat,
            sparsification_mask=mask,
            edge_index=edge_index_dev,
            batch=batch_vec,
            chebyshev_data=chebyshev_data,
            batch_ptr=ptr,
        )

    x_hat_cpu = x_hat.squeeze().cpu()

    use_normalization = cfg.get('use_normalization', False)
    if use_normalization and norm_stats is not None:
        field_names = ['density', 'pressure', 'velocity_x', 'velocity_y']
        fname = field_names[field_index]
        gwt_m = norm_stats[fname]['mean']
        gwt_s = norm_stats[fname]['std']
        x_hat_cpu = x_hat_cpu * gwt_s + gwt_m

    return x_hat_cpu


# =============================================================================
# Stacked Figure Plotting
# =============================================================================

def plot_stacked_sample(
    sample_idx: int,
    x_orig: torch.Tensor,
    x_gxn: torch.Tensor,
    x_vgae: torch.Tensor,
    x_nigwt: torch.Tensor,
    num_rows: int,
    num_cols: int,
    output_dir: str,
):
    """
    Plot one figure with 3 rows (GXN / Sparse-VGAE / NIGWT) x 3 cols
    (Original | Reconstruction | Error).

    Layout:
    - Column headers at the very top (once).
    - Row labels on the far left (once per row), positioned via fig.text so
      they never overlap the Y-axis tick numbers.
    - Y-axis tick labels shown only on leftmost column.
    - X-axis tick labels shown only on bottom row.
    - Single shared "x" axis label at the bottom, single "y" label on the left.
    - No suptitle.
    """

    ROW_LABELS   = ['GXN', 'Sparse-VGAE', 'NIGWT']
    COL_HEADERS  = ['Original Signal', 'Reconstruction', 'Reconstruction Error']
    N_ROWS, N_COLS = 3, 3

    FONT_TITLE   = 22   # column header font size
    FONT_ROWLBL  = 20   # row label font size
    FONT_AXLBL   = 18   # axis label ("x", "y") font size
    FONT_TICK    = 14   # tick label font size

    # Data for each row: list of (original, reconstructed) pairs
    row_data = [
        (x_orig, x_gxn),
        (x_orig, x_vgae),
        (x_orig, x_nigwt),
    ]

    # ---- compute shared vmin/vmax for original+recon panels ----
    all_signal_vals = torch.cat([x_orig, x_gxn, x_vgae, x_nigwt])
    sig_vmin = all_signal_vals.min().item()
    sig_vmax = all_signal_vals.max().item()

    # ---- compute per-row error vmax (don't share across rows) ----
    err_vmaxes = []
    for (xo, xr) in row_data:
        err = torch.abs(xo - xr)
        err_vmaxes.append(err.max().item())

    # ---- figure geometry ----
    # We need extra left margin for the row labels.
    # Each subplot has aspect ratio cols/rows ~ 384/128 = 3.
    cell_h = 3.2   # inches per subplot row
    cell_w = cell_h * (num_cols / num_rows)  # keep physical aspect

    left_margin   = 1.6   # inches: room for row label + Y tick labels
    right_margin  = 0.25  # inches
    top_margin    = 0.9   # inches: room for column headers
    bottom_margin = 0.7   # inches: room for x-axis label
    hspace_in     = 0.35  # inches between rows
    wspace_in     = 0.55  # inches between cols (room for colorbars)

    total_w = left_margin + N_COLS * cell_w + (N_COLS - 1) * wspace_in + right_margin
    total_h = top_margin  + N_ROWS * cell_h + (N_ROWS - 1) * hspace_in + bottom_margin

    fig = plt.figure(figsize=(total_w, total_h))

    # Convert margins to figure fractions
    left_f   = left_margin  / total_w
    right_f  = 1.0 - right_margin / total_w
    top_f    = 1.0 - top_margin   / total_h
    bottom_f = bottom_margin / total_h

    gs = gridspec.GridSpec(
        N_ROWS, N_COLS,
        figure=fig,
        left=left_f, right=right_f,
        top=top_f,   bottom=bottom_f,
        hspace=hspace_in / cell_h,
        wspace=wspace_in / cell_w,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(N_COLS)] for r in range(N_ROWS)]

    # ---- draw images ----
    for r, (xo, xr) in enumerate(row_data):
        xo_np  = xo.cpu().numpy().reshape(num_rows, num_cols)
        xr_np  = xr.cpu().numpy().reshape(num_rows, num_cols)
        err_np = np.abs(xo_np - xr_np)

        # Original
        im0 = axes[r][0].imshow(xo_np, aspect='auto', cmap='RdBu_r',
                                 vmin=sig_vmin, vmax=sig_vmax)
        plt.colorbar(im0, ax=axes[r][0], fraction=0.046, pad=0.04)

        # Reconstruction
        im1 = axes[r][1].imshow(xr_np, aspect='auto', cmap='RdBu_r',
                                 vmin=sig_vmin, vmax=sig_vmax)
        plt.colorbar(im1, ax=axes[r][1], fraction=0.046, pad=0.04)

        # Error
        im2 = axes[r][2].imshow(err_np, aspect='auto', cmap='Reds',
                                 vmin=0, vmax=err_vmaxes[r])
        plt.colorbar(im2, ax=axes[r][2], fraction=0.046, pad=0.04)

    # ---- tick labels: only bottom row gets x-ticks, only left col gets y-ticks ----
    for r in range(N_ROWS):
        for c in range(N_COLS):
            ax = axes[r][c]
            ax.tick_params(labelsize=FONT_TICK)
            if r < N_ROWS - 1:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

    # ---- column headers (top of figure, above top row) ----
    # Place them in axes-fraction coordinates of the top-row axes
    for c, header in enumerate(COL_HEADERS):
        ax_top = axes[0][c]
        ax_top.set_title(header, fontsize=FONT_TITLE, fontweight='bold', pad=10)

    # ---- row labels: placed in figure coordinates to the left of the left column ----
    # We use the centre of each row's axes (in figure coords) for vertical placement.
    for r, label in enumerate(ROW_LABELS):
        # Get the bounding box of the left axis in figure coordinates
        ax_left = axes[r][0]
        # Use a text object anchored to the axis transform so it scales correctly
        ax_left.text(
            -0.22, 0.5,          # x=-0.22 in axes-fraction coords puts it well left of y-ticks
            label,
            transform=ax_left.transAxes,
            fontsize=FONT_ROWLBL,
            fontweight='bold',
            va='center',
            ha='right',
            rotation=90,
        )

    # ---- single Y-axis label ("y") on the leftmost column ----
    # Place it via fig.text at the vertical center of all rows
    fig.text(
        left_f * 0.28,           # just to the left of the subplot area
        (top_f + bottom_f) / 2,
        'y',
        ha='center', va='center',
        fontsize=FONT_AXLBL, fontweight='bold',
        rotation=90,
    )

    # ---- single X-axis label ("x") on the bottom ----
    fig.text(
        (left_f + right_f) / 2,
        bottom_f * 0.30,
        'x',
        ha='center', va='center',
        fontsize=FONT_AXLBL, fontweight='bold',
    )

    # ---- save ----
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'stacked_sample{sample_idx}.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stacked 3-model comparison plot (GXN / Sparse-VGAE / NIGWT)")
    parser.add_argument('--test-dir',        required=True)
    parser.add_argument('--field-index',     type=int,   default=0)
    parser.add_argument('--keep-ratio',      type=float, default=0.01)
    parser.add_argument('--gxn-checkpoint',  required=True)
    parser.add_argument('--vgae-checkpoint', required=True)
    parser.add_argument('--gwt-checkpoint',  required=True)
    parser.add_argument('--gwt-run-dir',     required=True)
    parser.add_argument('--output-dir',      default='baseline_results/stacked')
    parser.add_argument('--sample-indices',  type=int, nargs='+', default=list(range(8)))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load test data ---
    print("Loading test data...")
    signals, coeffs, edge_index, num_nodes, num_rows, num_cols = \
        load_test_signals(args.test_dir, args.field_index)
    print(f"Signals: {signals.shape}, Grid: {num_rows}×{num_cols}={num_nodes} nodes")

    edge_index_dev = edge_index.to(device)

    # --- Load GXN ---
    print("Loading GXN model...")
    sampling_model, recovery_model, gxn_mean, gxn_std = load_gxn_model(
        args.gxn_checkpoint, edge_index_dev, device)

    # --- Load Sparse-VGAE ---
    print("Loading Sparse-VGAE model...")
    vgae_model, vgae_mean, vgae_std = load_vgae_model(
        args.vgae_checkpoint, args.keep_ratio, device)

    # --- Load NIGWT + matrix cache ---
    print("Loading NIGWT model...")
    gwt_model, gwt_cfg, gwt_norm_stats = load_gwt_model(
        args.gwt_run_dir, args.gwt_checkpoint, device)

    if coeffs is None:
        raise ValueError("NIGWT requires pre-computed coeffs in the .pt files.")

    print("Building Chebyshev matrix cache...")
    from data.dataset_turbulent import (GraphSignalDatasetTurbulent,
                                        create_matrix_cache_turbulent)
    dataset_for_cache = GraphSignalDatasetTurbulent(
        base_data_dir=args.test_dir,
        field_index=args.field_index,
        preload_to_memory=False,
        use_normalization=False,
    )
    matrix_cache = create_matrix_cache_turbulent(dataset=dataset_for_cache, device=device)

    # --- Per-sample inference + plot ---
    print(f"\nGenerating stacked plots for samples: {args.sample_indices}")
    for sample_idx in args.sample_indices:
        if sample_idx >= signals.shape[1]:
            print(f"  Skipping sample {sample_idx}: out of range")
            continue
        print(f"\n--- Sample {sample_idx} ---")

        x_orig = signals[:, sample_idx].cpu()  # physical units, (N,)

        # GXN reconstruction
        x_gxn = gxn_reconstruct(
            sampling_model, recovery_model, gxn_mean, gxn_std,
            x_orig, edge_index_dev, args.keep_ratio, device
        )

        # Sparse-VGAE reconstruction
        x_vgae = vgae_reconstruct(
            vgae_model, vgae_mean, vgae_std,
            x_orig, edge_index_dev, device
        )

        # NIGWT reconstruction
        coeffs_sample = coeffs[:, :, sample_idx]  # (N, num_filters)
        x_nigwt = gwt_reconstruct(
            gwt_model, gwt_cfg, gwt_norm_stats,
            coeffs_sample, edge_index, matrix_cache, device,
            field_index=args.field_index,
        )

        # Print MSEs (physical units)
        mse_gxn   = torch.nn.functional.mse_loss(x_gxn,   x_orig).item()
        mse_vgae  = torch.nn.functional.mse_loss(x_vgae,  x_orig).item()
        mse_nigwt = torch.nn.functional.mse_loss(x_nigwt, x_orig).item()
        print(f"  MSE — GXN: {mse_gxn:.6f} | Sparse-VGAE: {mse_vgae:.6f} | NIGWT: {mse_nigwt:.6f}")

        plot_stacked_sample(
            sample_idx=sample_idx,
            x_orig=x_orig,
            x_gxn=x_gxn,
            x_vgae=x_vgae,
            x_nigwt=x_nigwt,
            num_rows=num_rows,
            num_cols=num_cols,
            output_dir=args.output_dir,
        )

    print(f"\nAll stacked plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()