# -*- coding: utf-8 -*-
"""
plot_classical_sample.py

Generates a single reconstruction frame for each classical GSS baseline
(FastGSSS, BSGDA, RSBS) for a specified test sample index.

Follows the exact same 3-panel layout as plot_50_frames.py:
  Original Signal | Reconstruction | Absolute Error
Both RMSE-annotated and clean (_clean) variants are saved.

Output files (in --output-dir):
  fastgsss_sample{idx}.png  /  fastgsss_sample{idx}_clean.png
  bsgda_sample{idx}.png     /  bsgda_sample{idx}_clean.png
  rsbs_sample{idx}.png      /  rsbs_sample{idx}_clean.png

Usage
-----
  python -u plot_classical_sample.py \
      --test-dir     <path/to/processed/test> \
      --sample-index 47 \
      --field-index  0 \
      --keep-ratio   0.01 \
      --output-dir   <path/to/output>
"""

import sys
import os
import time
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data loading  (signals only — no coeffs needed for classical methods)
# =============================================================================

def load_test_data(test_dir: str, field_index: int = 0, max_samples: int = None):
    """
    Load test signals from disk.  Only the first `max_samples` files are
    read so we never allocate the full test-set tensor for a single-sample job.
    """
    test_dir = Path(test_dir)
    shared     = torch.load(test_dir / "shared_data.pt",
                            map_location='cpu', weights_only=False)
    edge_index = shared['edge_index']
    num_nodes  = int(shared['num_nodes'])
    num_rows   = int(shared['num_rows'])
    num_cols   = int(shared['num_columns'])

    sample_files = sorted(
        [f for f in test_dir.glob("*.pt") if f.stem.isdigit()],
        key=lambda f: int(f.stem)
    )
    if max_samples is not None:
        sample_files = sample_files[:max_samples]

    signals = []
    for f in sample_files:
        s = torch.load(f, map_location='cpu', weights_only=False)
        signals.append(s.x[:, field_index].float())

    signals = torch.stack(signals, dim=1)   # (N_nodes, n_loaded)
    return signals, edge_index, num_nodes, num_rows, num_cols


# =============================================================================
# Classical GSS graph objects + inference
# (identical to plot_50_frames.py)
# =============================================================================

def build_classical_graph(edge_index, num_nodes, dtype=torch.float32):
    from thgsp.graphs.core import SparseTensor, GraphBase
    row, col = edge_index
    val  = torch.ones(row.numel(), dtype=dtype)
    adj  = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(num_nodes, num_nodes))
    G    = GraphBase(adj, dtype=dtype, device=torch.device('cpu'))
    deg  = torch.zeros(num_nodes, dtype=dtype)
    deg.scatter_add_(0, row, torch.ones(row.numel(), dtype=dtype))
    Lr   = torch.cat([row, torch.arange(num_nodes)])
    Lc   = torch.cat([col, torch.arange(num_nodes)])
    Lv   = torch.cat([-torch.ones(row.numel(), dtype=dtype), deg])
    Lco  = SparseTensor(row=Lr, col=Lc, value=Lv,
                        sparse_sizes=(num_nodes, num_nodes))
    isd  = deg.pow(-0.5); isd[deg == 0] = 0.
    Lsv  = -isd[Lr] * Lv.clone() * isd[Lc]
    Lsv[Lr == Lc] = 1.
    Ls   = SparseTensor(row=Lr, col=Lc, value=Lsv,
                        sparse_sizes=(num_nodes, num_nodes))
    return G, adj, Lco, Ls


def infer_classical(method, x_norm, G, adj, Lco, Ls, num_nodes, keep_ratio):
    from baselines.graph_signal_sampling import sampling_and_reconstruction
    ss = max(1, int(keep_ratio * num_nodes))
    sp = {
        'fastgsss_bandwidth':    'exact',
        'fastgsss_nu':           75.0,
        'fastgsss_order':        12,
        'fastgsss_recon_order':  3,
        'bsgda_mu':              0.01,
        'bsgda_epsilon':         1e-5,
        'bsgda_p_hops':          12,
        'bsgda_reg_order':       1,
        'rsbs_bandwidth_mode':   'exact',
        'rsbs_num_estimation':   1,
        'rsbs_epsilon':          1e-2,
        'rsbs_order':            30,
        'rsbs_mu':               0.01,
        'rsbs_reg_order':        1,
        'rsbs_lap_type':         'comb',
    }
    _, xh = sampling_and_reconstruction(
        method_name=method,
        x=x_norm.unsqueeze(-1),
        args={
            'num_nodes':        num_nodes,
            'sample_size':      ss,
            'thgsp_graph':      G,
            'thgsp_adj':        adj,
            'thgsp_L_comb':     Lco,
            'thgsp_L_sym':      Ls,
            'signal_bandwidth': ss,
        },
        search_params=sp)
    return xh.squeeze()


# =============================================================================
# Frame drawing + saving
# (identical to plot_50_frames.py)
# =============================================================================

def _draw_frame(x_o, x_r, err, vmin, vmax, show_rmse, rmse_val):
    FS_TITLE    = 30
    FS_LABEL    = 27
    FS_TICK     = 24
    FS_SUPTITLE = 32
    TK_LEN_MAJ  = 18
    TK_WID_MAJ  = 4
    TK_LEN_MIN  = 10
    TK_WID_MIN  = 3
    TK_PAD      = 12

    fig_h = 8 if show_rmse else 7
    fig, axes = plt.subplots(1, 3, figsize=(27, fig_h))

    im0 = axes[0].imshow(x_o, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Signal', fontsize=FS_TITLE, pad=12)
    cb0 = plt.colorbar(im0, ax=axes[0])
    cb0.ax.tick_params(which='major', labelsize=FS_TICK,
                       length=TK_LEN_MAJ, width=TK_WID_MAJ, pad=TK_PAD)
    cb0.ax.tick_params(which='minor', length=TK_LEN_MIN, width=TK_WID_MIN)

    im1 = axes[1].imshow(x_r, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('Reconstruction', fontsize=FS_TITLE, pad=12)
    cb1 = plt.colorbar(im1, ax=axes[1])
    cb1.ax.tick_params(which='major', labelsize=FS_TICK,
                       length=TK_LEN_MAJ, width=TK_WID_MAJ, pad=TK_PAD)
    cb1.ax.tick_params(which='minor', length=TK_LEN_MIN, width=TK_WID_MIN)

    im2 = axes[2].imshow(err, aspect='auto', cmap='Reds', vmin=0)
    axes[2].set_title('Absolute Error', fontsize=FS_TITLE, pad=12)
    cb2 = plt.colorbar(im2, ax=axes[2])
    cb2.ax.tick_params(which='major', labelsize=FS_TICK,
                       length=TK_LEN_MAJ, width=TK_WID_MAJ, pad=TK_PAD)
    cb2.ax.tick_params(which='minor', length=TK_LEN_MIN, width=TK_WID_MIN)

    for ax in axes:
        ax.set_xlabel('x', fontsize=FS_LABEL, labelpad=10)
        ax.set_ylabel('y', fontsize=FS_LABEL, labelpad=10)
        ax.tick_params(axis='both', which='major',
                       labelsize=FS_TICK, length=TK_LEN_MAJ,
                       width=TK_WID_MAJ, pad=TK_PAD)
        ax.tick_params(axis='both', which='minor',
                       length=TK_LEN_MIN, width=TK_WID_MIN)

    if show_rmse:
        fig.suptitle(f'RMSE = {rmse_val:.6f}', fontsize=FS_SUPTITLE, y=1.01)

    plt.tight_layout()
    return fig


def save_frame(x_orig, x_recon, num_rows, num_cols, filepath, filepath_clean=None):
    """
    Save two variants of a 3-panel reconstruction frame:
      filepath        — with "RMSE = X.XXXXXX" suptitle
      filepath_clean  — same frame but NO suptitle (cleaner for paper figures)
    Returns the scalar RMSE value.
    """
    x_o  = x_orig.cpu().numpy().reshape(num_rows, num_cols)
    x_r  = x_recon.cpu().numpy().reshape(num_rows, num_cols)
    err  = np.abs(x_o - x_r)
    vmin, vmax = float(x_o.min()), float(x_o.max())
    rmse = float(np.sqrt(np.mean((x_o - x_r) ** 2)))

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    fig = _draw_frame(x_o, x_r, err, vmin, vmax, show_rmse=True, rmse_val=rmse)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if filepath_clean is None:
        base, ext = os.path.splitext(filepath)
        filepath_clean = base + '_clean' + ext

    fig = _draw_frame(x_o, x_r, err, vmin, vmax, show_rmse=False, rmse_val=rmse)
    fig.savefig(filepath_clean, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return rmse


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot one sample reconstruction for classical GSS baselines "
                    "(FastGSSS, BSGDA, RSBS)")
    parser.add_argument('--test-dir',      required=True,
                        help='Path to processed test directory')
    parser.add_argument('--sample-index',  type=int, default=47,
                        help='0-based sample index to render (default: 47 = 48th sample)')
    parser.add_argument('--field-index',   type=int,   default=0,
                        help='Field index within each sample (default: 0)')
    parser.add_argument('--keep-ratio',    type=float, default=0.01,
                        help='Sparsity keep ratio (default: 0.01 = 1%%)')
    parser.add_argument('--output-dir',    required=True,
                        help='Directory to save output PNGs')
    args = parser.parse_args()

    idx = args.sample_index
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Target sample index : {idx}  (1-based: sample {idx + 1})")
    print(f"Keep ratio          : {args.keep_ratio}")
    print(f"Output dir          : {args.output_dir}\n")

    # ── Load up to sample_index + 1 samples (no more) ───────────────────────
    # Normalization anchors (mean/std) are computed from the loaded batch,
    # matching the behaviour of plot_50_frames.py run with --num-frames (idx+1).
    n_load = idx + 1
    print(f"Loading {n_load} test samples (up to index {idx})...")
    signals, edge_index, num_nodes, nr, nc = load_test_data(
        args.test_dir, args.field_index, max_samples=n_load)

    if signals.shape[1] <= idx:
        raise ValueError(
            f"Dataset has only {signals.shape[1]} samples; "
            f"cannot access index {idx}.")

    print(f"  Grid: {nr} × {nc} = {num_nodes} nodes")
    x = signals[:, idx].cpu()   # (N_nodes,) — the target sample

    # Normalization anchors for classical methods (over all loaded samples)
    cl_mu = signals.mean()
    cl_sd = signals.std()
    if cl_sd < 1e-8:
        cl_sd = torch.tensor(1.0)
    print(f"  Signal stats (over {n_load} loaded samples): "
          f"mean={cl_mu:.6f}, std={cl_sd:.6f}\n")

    # ── Build classical graph objects (once) ─────────────────────────────────
    print("Building classical graph objects...")
    t0 = time.time()
    G, adj, Lco, Ls = build_classical_graph(edge_index, num_nodes)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    x_norm = (x - cl_mu) / cl_sd

    # ── Run each classical method and save frame ─────────────────────────────
    classical_methods = [
        ('fastgsss', 'FastGSSS'),
        ('bsgda',    'BSGDA'),
        ('rsbs',     'RSBS'),
    ]

    print("=" * 60)
    print(f"Classical GSS reconstruction — sample {idx} (1-based: {idx+1})")
    print("=" * 60)

    for mname, label in classical_methods:
        fp = os.path.join(args.output_dir, f'{mname}_sample{idx}.png')
        print(f"\n  [{label}]  running...")
        try:
            t0 = time.time()
            xh_norm = infer_classical(mname, x_norm, G, adj, Lco, Ls,
                                      num_nodes, args.keep_ratio)
            xh_phys = xh_norm.cpu() * cl_sd + cl_mu
            elapsed = time.time() - t0
            rmse = save_frame(x, xh_phys, nr, nc, fp)
            base, ext = os.path.splitext(fp)
            print(f"  {label:<10}  RMSE={rmse:.6f}  ({elapsed:.1f}s)")
            print(f"    → {fp}")
            print(f"    → {base}_clean{ext}")
        except Exception as e:
            print(f"  {label:<10}  FAILED: {e}")

    print(f"\nAll classical frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
