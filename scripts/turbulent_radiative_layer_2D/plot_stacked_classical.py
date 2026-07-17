# -*- coding: utf-8 -*-

import sys
import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.graph_signal_sampling import sampling_and_reconstruction
from thgsp.graphs.core import SparseTensor, GraphBase


# =============================================================================
# Constants  (IDENTICAL to plot_stacked_comparison.py)
# =============================================================================

FONT_TITLE  = 22
FONT_TICK   = 14
FONT_CBAR   = 14
FONT_SUPER  = 24

# Row labels differ from plot_stacked_comparison.py (everything else is identical)
ROW_LABELS  = ['', '', '']
COL_HEADERS = ['Original Signal', 'Reconstruction', 'Reconstruction Error']


# =============================================================================
# Data Loading  (identical to run_classical_baselines.py)
# =============================================================================

def load_all_signals(processed_split_dir: str, field_index: int = 0,
                     max_samples: int = None):
    split_dir = Path(processed_split_dir)
    shared    = torch.load(split_dir / "shared_data.pt",
                           map_location='cpu', weights_only=False)
    edge_index = shared['edge_index']
    num_nodes  = shared['num_nodes']
    num_rows   = shared['num_rows']
    num_cols   = shared['num_columns']

    sample_files = sorted(
        [f for f in split_dir.glob("*.pt") if f.stem.isdigit()],
        key=lambda f: int(f.stem),
    )
    if max_samples:
        sample_files = sample_files[:max_samples]

    signals = []
    for f in sample_files:
        s = torch.load(f, map_location='cpu', weights_only=False)
        # .float() handles both float16 (half-precision storage) and float32
        signals.append(s.x[:, field_index].float())
    signals = torch.stack(signals, dim=1)   # (N, num_samples)
    return signals, edge_index, num_nodes, num_rows, num_cols


# =============================================================================
# Graph Objects  (identical to run_classical_baselines.py)
# =============================================================================

def build_thgsp_objects(edge_index, num_nodes, x):
    row, col = edge_index
    value = torch.ones(row.numel(), dtype=x.dtype)
    adj   = SparseTensor(row=row, col=col, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    G     = GraphBase(adj, dtype=x.dtype, device=x.device)

    deg = torch.zeros(num_nodes, dtype=x.dtype)
    deg.scatter_add_(0, row, torch.ones(row.numel(), dtype=x.dtype))

    L_row = row
    L_col = col
    L_val = -torch.ones(row.numel(), dtype=x.dtype)
    diag_idx = torch.arange(num_nodes)
    L_row = torch.cat([L_row, diag_idx])
    L_col = torch.cat([L_col, diag_idx])
    L_val = torch.cat([L_val, deg])
    L_comb = SparseTensor(row=L_row, col=L_col, value=L_val,
                          sparse_sizes=(num_nodes, num_nodes))

    inv_sqrt_deg = deg.pow(-0.5)
    inv_sqrt_deg[deg == 0] = 0.0
    L_sym_val = -inv_sqrt_deg[L_row] * L_val.clone() * inv_sqrt_deg[L_col]
    diag_mask             = (L_row == L_col)
    L_sym_val[diag_mask]  = 1.0
    L_sym = SparseTensor(row=L_row, col=L_col, value=L_sym_val,
                         sparse_sizes=(num_nodes, num_nodes))

    return G, adj, L_comb, L_sym


# =============================================================================
# Classical Reconstruction
# =============================================================================

def reconstruct_classical(method, x_norm, num_nodes, sample_size,
                           G, adj, L_comb, L_sym):
    """
    Run one classical method on a normalized 1-D signal (N,).
    Returns x_hat_norm (N,) tensor in normalized space, or None on failure.
    """
    x_2d = x_norm.unsqueeze(-1)   # (N, 1)

    args_dict = {
        'num_nodes':        num_nodes,
        'sample_size':      sample_size,
        'thgsp_graph':      G,
        'thgsp_adj':        adj,
        'thgsp_L_comb':     L_comb,
        'thgsp_L_sym':      L_sym,
        'signal_bandwidth': sample_size,
    }
    search_params = {
        'fastgsss_bandwidth':   'exact',
        'fastgsss_nu':          75.0,
        'fastgsss_order':       12,
        'fastgsss_recon_order': 3,
        'bsgda_mu':             0.01,
        'bsgda_epsilon':        1e-5,
        'bsgda_p_hops':         12,
        'bsgda_reg_order':      1,
        'rsbs_bandwidth_mode':  'exact',
        'rsbs_num_estimation':  1,
        'rsbs_epsilon':         1e-2,
        'rsbs_order':           30,
        'rsbs_mu':              0.01,
        'rsbs_reg_order':       1,
        'rsbs_lap_type':        'comb',
    }

    try:
        _S, x_hat_norm = sampling_and_reconstruction(
            method_name=method,
            x=x_2d,
            args=args_dict,
            search_params=search_params,
        )
        return x_hat_norm.squeeze()   # (N,)
    except Exception as e:
        print(f"    [{method}] FAILED: {e}")
        return None


# =============================================================================
# Stacked Figure  �  IDENTICAL layout/style to plot_stacked_comparison.py
# =============================================================================

def plot_stacked_sample(sample_idx, reconstructions, x_orig_np,
                        num_rows, num_cols, output_dir):
    """
    Parameters
    ----------
    sample_idx      : int
    reconstructions : list of 3 numpy arrays (fastgsss, bsgda, rsbs),
                      all in physical units, shape (N,).
                      Pass np.full(N, np.nan) for a failed method.
    x_orig_np       : numpy (N,)  physical units
    num_rows, num_cols : grid dimensions
    output_dir      : str
    """
    os.makedirs(output_dir, exist_ok=True)

    n_models = 3   # rows
    n_cols   = 3   # original | recon | error

    # -- figure size: IDENTICAL to plot_stacked_comparison.py --------------
    fig_w = n_cols * 5.0   # 15 inches
    fig_h = n_models * 4.0 # 12 inches
    fig   = plt.figure(figsize=(fig_w, fig_h))

    # -- GridSpec: thin label column + 3 image columns ---------------------
    # IDENTICAL to plot_stacked_comparison.py
    gs = gridspec.GridSpec(
        n_models, n_cols + 1,                        # +1 for row-label column
        figure=fig,
        width_ratios=[0.04] + [1.0] * n_cols,
        hspace=0.06,
        wspace=0.08,
    )

    orig_2d  = x_orig_np.reshape(num_rows, num_cols)
    vmin_sig = float(orig_2d.min())
    vmax_sig = float(orig_2d.max())

    # Global error max across all 3 models (shared colour scale for errors)
    # IDENTICAL logic to plot_stacked_comparison.py
    all_errors = [np.abs(orig_2d - r.reshape(num_rows, num_cols))
                  for r in reconstructions]
    emax = float(max(e[np.isfinite(e)].max() if np.isfinite(e).any() else 0.0
                     for e in all_errors))

    axes_img = [[None] * n_cols for _ in range(n_models)]

    for row in range(n_models):
        recon_np = reconstructions[row].reshape(num_rows, num_cols)
        error_np = all_errors[row]
        panels   = [orig_2d,  recon_np, error_np]
        cmaps    = ['RdBu_r', 'RdBu_r', 'hot']        # 'hot' matches reference
        vmins    = [vmin_sig, vmin_sig,  0.0]
        vmaxs    = [vmax_sig, vmax_sig,  emax]

        # -- row label axis (thin leftmost column) -------------------------
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.set_axis_off()
        ax_label.text(
            0.5, 0.5, ROW_LABELS[row],
            ha='center', va='center',
            rotation=90,
            fontsize=FONT_TITLE, fontweight='bold',
            transform=ax_label.transAxes,
        )

        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col + 1])
            axes_img[row][col] = ax

            im = ax.imshow(
                panels[col], cmap=cmaps[col], aspect='auto',
                vmin=vmins[col], vmax=vmaxs[col], origin='upper',
            )

            # Colourbar ONLY on rightmost column  (identical to reference)
            if col == n_cols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cbar.ax.tick_params(labelsize=FONT_CBAR)

            # -- tick labels -----------------------------------------------
            ax.tick_params(labelsize=FONT_TICK)

            # x-ticks: only bottom row
            if row < n_models - 1:
                ax.tick_params(labelbottom=False)

            # y-ticks: only leftmost image column (col == 0)
            if col > 0:
                ax.tick_params(labelleft=False)

            # -- column headers: only top row ------------------------------
            if row == 0:
                ax.set_title(COL_HEADERS[col],
                             fontsize=FONT_TITLE, fontweight='bold', pad=8)

            # -- x-axis label: bottom row, centre column only --------------
            if row == n_models - 1 and col == 1:
                ax.set_xlabel('cols', fontsize=FONT_TITLE)

            # -- y-axis label: leftmost image column every row -------------
            if col == 0:
                ax.set_ylabel('rows', fontsize=FONT_TITLE)

    plt.suptitle(
        f'Test Sample {sample_idx}  -  Original vs Reconstruction',
        fontsize=FONT_SUPER, fontweight='bold', y=1.01,
    )

    out_path = os.path.join(output_dir, f'sample{sample_idx}_stacked.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stacked classical baseline plot (FastGSSS / BSGDA / RSBS) "
                    "- identical layout to plot_stacked_comparison.py")
    parser.add_argument('--processed-split-dir', required=True,
                        help='Path to processed test split dir with shared_data.pt')
    parser.add_argument('--field-index',    type=int,   default=0)
    parser.add_argument('--keep-ratio',     type=float, default=0.01)
    parser.add_argument('--output-dir',     type=str,
                        default='baseline_results/classical_stacked')
    parser.add_argument('--sample-indices', type=int, nargs='+',
                        default=list(range(8)))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load test data ---
    print("Loading test data...")
    max_needed = max(args.sample_indices) + 1
    signals, edge_index, num_nodes, num_rows, num_cols = \
        load_all_signals(args.processed_split_dir, args.field_index,
                         max_samples=max_needed)
    print(f"Signals: {signals.shape}, Grid: {num_rows}x{num_cols}={num_nodes} nodes")

    # Z-score normalise (identical to run_classical_baselines.py)
    sig_mean = signals.mean()
    sig_std  = signals.std()
    if sig_std < 1e-8:
        sig_std = torch.tensor(1.0)
    print(f"Normalization: mean={sig_mean:.4f}, std={sig_std:.4f}")

    sample_size = max(1, int(args.keep_ratio * num_nodes))
    print(f"Nodes: {num_nodes}, sample_size M={sample_size} ({args.keep_ratio*100:.2f}%)")

    # Build graph objects once (shared across samples & methods)
    print("Building graph objects...")
    G, adj, L_comb, L_sym = build_thgsp_objects(
        edge_index, num_nodes, signals[:, 0])

    # --- Per-sample loop ---
    print(f"\nGenerating stacked classical plots for samples: {args.sample_indices}")
    all_mses = {m: [] for m in ['fastgsss', 'bsgda', 'rsbs']}

    for sample_idx in args.sample_indices:
        if sample_idx >= signals.shape[1]:
            print(f"  Skipping sample {sample_idx}: out of range")
            continue
        print(f"\n--- Sample {sample_idx} ---")

        x_orig    = signals[:, sample_idx].cpu()          # physical units (N,)
        x_norm    = (x_orig - sig_mean) / sig_std         # normalized   (N,)
        x_orig_np = x_orig.numpy()

        recons_phys = []   # physical-unit reconstructions for the 3 rows
        for method in ['fastgsss', 'bsgda', 'rsbs']:
            x_hat_norm = reconstruct_classical(
                method, x_norm, num_nodes, sample_size,
                G, adj, L_comb, L_sym,
            )

            if x_hat_norm is not None:
                mse = torch.nn.functional.mse_loss(x_hat_norm, x_norm).item()
                all_mses[method].append(mse)
                print(f"    [{method}] MSE (normalized): {mse:.6f}")
                # Denormalise to physical units for plotting
                x_phys = (x_hat_norm.cpu() * sig_std + sig_mean).numpy()
            else:
                # Failed: pass NaN array so imshow still renders a blank panel
                x_phys = np.full(num_nodes, np.nan, dtype=np.float32)

            recons_phys.append(x_phys)

        plot_stacked_sample(
            sample_idx=sample_idx,
            reconstructions=recons_phys,   # [fastgsss, bsgda, rsbs] in physical units
            x_orig_np=x_orig_np,
            num_rows=num_rows,
            num_cols=num_cols,
            output_dir=args.output_dir,
        )

    # --- Summary ---
    print("\n=== CLASSICAL BASELINE RESULTS (normalized MSE) ===")
    summary = {}
    for method, mses in all_mses.items():
        if mses:
            mean_m = float(np.mean(mses))
            std_m  = float(np.std(mses))
            print(f"  {method}: mean MSE = {mean_m:.6f} +/- {std_m:.6f}")
            summary[method] = {
                'mean_mse': mean_m,
                'std_mse':  std_m,
                'all_mses': mses,
            }

    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"All stacked plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()