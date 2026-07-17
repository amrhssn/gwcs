# -*- coding: utf-8 -*-
"""
plot_classical_sample_single.py

Runs ONE classical GSS method (FastGSSS, BSGDA, or RSBS) on a single test
sample and saves reconstruction plots in the same 3-panel format as
plot_classical_sample.py.

Designed to be called as three independent parallel SLURM jobs — one per
method — rather than sequentially.  Adds thread pinning and torch
performance tweaks to extract maximum CPU throughput.

Usage
-----
  python -u plot_classical_sample_single.py \
      --test-dir     <path/to/processed/test> \
      --method       fastgsss          # or bsgda / rsbs
      --sample-index 47 \
      --field-index  0 \
      --keep-ratio   0.01 \
      --output-dir   <path/to/output>
"""

# ── Thread / performance tuning ──────────────────────────────────────────────
# Must be set BEFORE numpy / torch are imported so MKL / OpenBLAS picks them up.
import os, sys

_NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, str(_NCPU))

import time
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Use all available threads for intra-op parallelism
torch.set_num_threads(_NCPU)
torch.set_num_interop_threads(max(1, _NCPU // 2))

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALID_METHODS = ('fastgsss', 'bsgda', 'rsbs')


# =============================================================================
# Data loading — identical to plot_classical_sample.py
# =============================================================================

def load_test_data(test_dir: str, field_index: int = 0, max_samples: int = None):
    test_dir   = Path(test_dir)
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
# Classical graph objects — built once, shared across any method
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


# =============================================================================
# Inference — one method at a time
# =============================================================================

def infer_classical(method, x_norm, G, adj, Lco, Ls, num_nodes, keep_ratio):
    from baselines.graph_signal_sampling import sampling_and_reconstruction
    ss = max(1, int(keep_ratio * num_nodes))

    # These are the validated optimal hyperparameters from the ABL grid search.
    # Do NOT change them — any reduction would cost reconstruction quality.
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
# Frame drawing + saving — identical to plot_classical_sample.py
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
    x_o   = x_orig.cpu().numpy().reshape(num_rows, num_cols)
    x_r   = x_recon.cpu().numpy().reshape(num_rows, num_cols)
    err   = np.abs(x_o - x_r)
    vmin, vmax = float(x_o.min()), float(x_o.max())
    rmse  = float(np.sqrt(np.mean((x_o - x_r) ** 2)))

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    fig = _draw_frame(x_o, x_r, err, vmin, vmax, show_rmse=True,  rmse_val=rmse)
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
        description="Run a SINGLE classical GSS method on one test sample.")
    parser.add_argument('--test-dir',     required=True,
                        help='Path to processed test directory')
    parser.add_argument('--method',       required=True,
                        choices=VALID_METHODS,
                        help='Classical method: fastgsss | bsgda | rsbs')
    parser.add_argument('--sample-index', type=int, default=47,
                        help='0-based sample index (default: 47 = 48th sample)')
    parser.add_argument('--field-index',  type=int, default=0,
                        help='Field index within each sample (default: 0)')
    parser.add_argument('--keep-ratio',   type=float, default=0.01,
                        help='Sparsity keep ratio (default: 0.01 = 1%%)')
    parser.add_argument('--output-dir',   required=True,
                        help='Directory to save output PNGs')
    args = parser.parse_args()

    method = args.method.lower()
    idx    = args.sample_index
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Method              : {method.upper()}")
    print(f"Target sample index : {idx}  (1-based: sample {idx + 1})")
    print(f"Keep ratio          : {args.keep_ratio}")
    print(f"Output dir          : {args.output_dir}")
    print(f"CPU threads         : {_NCPU}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    # Load idx+1 samples so normalization stats match plot_classical_sample.py
    n_load = idx + 1
    print(f"Loading {n_load} test samples (up to index {idx})...")
    t0 = time.time()
    signals, edge_index, num_nodes, nr, nc = load_test_data(
        args.test_dir, args.field_index, max_samples=n_load)
    print(f"  Done in {time.time()-t0:.1f}s")

    if signals.shape[1] <= idx:
        raise ValueError(
            f"Dataset has only {signals.shape[1]} samples; "
            f"cannot access index {idx}.")

    print(f"  Grid: {nr} × {nc} = {num_nodes} nodes")
    x     = signals[:, idx].cpu()   # (N_nodes,)
    cl_mu = signals.mean()
    cl_sd = signals.std()
    if cl_sd < 1e-8:
        cl_sd = torch.tensor(1.0)
    print(f"  Signal stats (over {n_load} loaded samples): "
          f"mean={cl_mu:.6f}, std={cl_sd:.6f}\n")

    # ── Build graph objects ───────────────────────────────────────────────────
    print("Building classical graph objects...")
    t0 = time.time()
    G, adj, Lco, Ls = build_classical_graph(edge_index, num_nodes)
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # ── Run the requested method ──────────────────────────────────────────────
    x_norm = (x - cl_mu) / cl_sd
    fp     = os.path.join(args.output_dir, f'{method}_sample{idx}.png')

    print("=" * 60)
    print(f"Running {method.upper()} — sample {idx} (1-based: {idx+1})")
    print("=" * 60)

    t0   = time.time()
    xh_n = infer_classical(method, x_norm, G, adj, Lco, Ls,
                            num_nodes, args.keep_ratio)
    xh   = xh_n.cpu() * cl_sd + cl_mu
    elapsed = time.time() - t0

    rmse = save_frame(x, xh, nr, nc, fp)
    base, ext = os.path.splitext(fp)
    print(f"\n  {method.upper():<10}  RMSE={rmse:.6f}  ({elapsed:.1f}s)")
    print(f"    → {fp}")
    print(f"    → {base}_clean{ext}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
