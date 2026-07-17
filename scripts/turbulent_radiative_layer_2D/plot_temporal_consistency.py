# -*- coding: utf-8 -*-
"""
plot_temporal_consistency.py

Temporal-consistency analysis of reconstruction models on PDE test sets.

The key idea (from Amit's feedback):
  Each test sample is a time snapshot from a physical simulation. A good
  reconstruction model should not only have low per-frame RMSE but should
  also faithfully track the *change* between consecutive frames (the temporal
  gradient).  A model that outputs a blurry "mean field" at every frame will
  have poor temporal tracking even if it avoids catastrophically bad frames.

For every consecutive pair of test samples (t → t+1) that belong to the SAME
simulation trajectory, we compute three metrics:

  dRMSE  — Temporal-gradient error:
            ||( x_hat[t+1]-x_hat[t] ) - ( x[t+1]-x[t] )||₂ / sqrt(N)
            Lower is better.  Directly measures dynamic-tracking accuracy.

  TC     — Temporal correlation (Pearson):
            corr( x_hat[t+1]-x_hat[t],  x[t+1]-x[t] )
            Higher is better (range [−1,1]).  Captures direction of change.

  RelErr — Normalised temporal error:
            dRMSE / (||x[t+1]-x[t]||₂/sqrt(N))
            Lower is better.  Answers: "relative to how much the signal
            changes, how wrong is the model's predicted change?"

Trajectory boundaries are SKIPPED automatically using the trajectory_index
and time_index fields stored in each sample's signal_metadata.

Outputs (per dataset, in --output-dir)
---------------------------------------
  temporal_drmse_curves.png    — dRMSE per transition (line plot)
  temporal_tc_curves.png       — TC per transition (line plot)
  temporal_summary.png         — bar chart: mean dRMSE + mean TC
  temporal_metrics.json        — all summary statistics for the paper

Usage
-----
  python -u plot_temporal_consistency.py \\
      --test-dir            <processed/test> \\
      --gwt-checkpoint      <.../*.pt> \\
      --gwt-run-dir         <.../runs_xxx/TIMESTAMP_dir> \\
      --vgae-checkpoint     <.../svgae_99k/best_checkpoint.pt> \\
      --vgae-50k-checkpoint <.../svgae_50k/best_checkpoint.pt> \\
      --gxn-checkpoint      <.../gxn_99k/best_checkpoint.pt> \\
      --field-index 0 \\
      --keep-ratio  0.01 \\
      --output-dir  <results/plots/temporal>
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Model colours / display names ────────────────────────────────────────────
MODEL_STYLES = {
    'gwcs':     {'label': 'GWCS',     'color': '#1f77b4', 'lw': 2.8, 'zorder': 4},
    'sgae_99k': {'label': 'SGAE-99K', 'color': '#ff7f0e', 'lw': 2.2, 'zorder': 3},
    'sgae_50k': {'label': 'SGAE-50K', 'color': '#d62728', 'lw': 2.2, 'zorder': 2},
    'gxn':      {'label': 'GXN',      'color': '#2ca02c', 'lw': 2.2, 'zorder': 1},
}

# ── Plotting style (matches existing RMSE-curve plots in the project) ─────────
plt.rcParams.update({
    "figure.dpi":         200,
    "savefig.dpi":        200,
    "font.size":          34,
    "axes.labelsize":     34,
    "axes.titlesize":     36,
    "legend.fontsize":    28,
    "xtick.labelsize":    28,
    "ytick.labelsize":    28,
    "xtick.major.size":   18,  "xtick.major.width": 4,
    "xtick.minor.size":   10,  "xtick.minor.width": 3,
    "ytick.major.size":   18,  "ytick.major.width": 4,
    "ytick.minor.size":   10,  "ytick.minor.width": 3,
    "xtick.major.pad":    4,   "ytick.major.pad":   4,
    "axes.linewidth":     1.5,
    "lines.linewidth":    2.8,
})


# =============================================================================
# Data helpers
# =============================================================================

def get_sample_files(test_dir: Path):
    """Return test .pt files sorted by sample index."""
    return sorted(
        [f for f in test_dir.glob("*.pt") if f.stem.isdigit()],
        key=lambda f: int(f.stem)
    )


def load_sample(filepath: Path, field_index: int):
    """
    Load one sample.  Returns:
        x_orig       (N,)      – raw signal in physical units
        coeffs       (N, F)    – SGWT coefficients (or None)
        traj_idx     int/None  – trajectory index from signal_metadata
        time_idx     int/None  – time-step within trajectory
    """
    s = torch.load(filepath, map_location='cpu', weights_only=False)
    x = s.x[:, field_index].float()
    coeffs = (s.coeffs[:, field_index, :].float()
              if (hasattr(s, 'coeffs') and s.coeffs is not None) else None)

    traj_idx = time_idx = None
    if hasattr(s, 'signal_metadata') and s.signal_metadata is not None:
        meta     = s.signal_metadata
        traj_idx = int(meta['trajectory_index']) if 'trajectory_index' in meta else None
        time_idx = int(meta['time_index'])        if 'time_index'        in meta else None

    return x, coeffs, traj_idx, time_idx


def load_shared(test_dir: Path):
    shared = torch.load(test_dir / "shared_data.pt",
                        map_location='cpu', weights_only=False)
    return (shared['edge_index'],
            int(shared['num_nodes']),
            int(shared['num_rows']),
            int(shared['num_columns']))


def is_trajectory_boundary(traj_prev, t_prev, traj_cur, t_cur):
    """Return True if (prev→cur) crosses a trajectory boundary."""
    # Different trajectory IDs → always a boundary
    if traj_prev is not None and traj_cur is not None:
        if traj_prev != traj_cur:
            return True
    # Time index didn't advance by exactly 1 → boundary (also catches resets)
    if t_prev is not None and t_cur is not None:
        if t_cur != t_prev + 1:
            return True
    # No metadata at all: assume consecutive (safe fallback)
    return False


# =============================================================================
# Model loaders + inference  (identical to plot_50_frames.py)
# =============================================================================

def load_gxn(ckpt_path, device):
    from baselines.gxn.neural_sampling import NeuralSampling, NeuralRecovery
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sm = NeuralSampling(1, ckpt.get('hidden_dim', 128),
                        ckpt.get('max_hops', 2)).to(device)
    rm = NeuralRecovery(1, ckpt.get('recovery_K', 8),
                        ckpt.get('recovery_L', 5)).to(device)
    sm.load_state_dict(ckpt['sampling_model']); sm.eval()
    rm.load_state_dict(ckpt['recovery_model']); rm.eval()
    return sm, rm, ckpt.get('norm_mean', 0.0), ckpt.get('norm_std', 1.0)


def infer_gxn(sm, rm, norm_mean, norm_std, x, edge_index, keep_ratio, device):
    N  = x.shape[0]
    M  = max(1, int(keep_ratio * N))
    mu = torch.tensor(norm_mean, device=device)
    sd = torch.tensor(norm_std,  device=device)
    xi = ((x.to(device) - mu) / sd).unsqueeze(1)
    with torch.no_grad():
        att, _, _ = sm(xi, edge_index)
        sc = att.squeeze()
        if sc.dim() == 0:
            sc = sc.unsqueeze(0)
        _, ti = torch.topk(sc, k=M)
        mp = torch.zeros_like(xi); mp[ti] = 1.0
        xh = rm(mp * att * xi, edge_index, att, mp)
    return (xh.squeeze() * sd + mu).cpu()


def load_vgae(ckpt_path, keep_ratio, device):
    from baselines.sparse_VGAE.gae import GAE
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GAE(1, ckpt.get('latent_dim', 128), 1,
                ckpt.get('hidden_layers', 4), ckpt.get('hidden_layers', 4),
                'SAGEConv', 'silu', keep_ratio, False,
                'ste_annealing', 1.0, 0.01, 1000).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt.get('norm_mean', 0.0), ckpt.get('norm_std', 1.0)


def infer_vgae(model, norm_mean, norm_std, x, edge_index, device):
    mu = torch.tensor(norm_mean, device=device)
    sd = torch.tensor(norm_std,  device=device)
    xi = ((x.to(device) - mu) / sd).unsqueeze(1)
    with torch.no_grad():
        xh = model.decode(model.encode(xi, edge_index), edge_index)
    return (xh.squeeze() * sd + mu).cpu()


def load_gwcs(run_dir: Path, ckpt_path: str, device):
    # detect config JSON (handles both turbulent and other dataset configs)
    cfg_files = list(run_dir.glob('reconstructor_pretraining_config*.json'))
    if not cfg_files:
        raise FileNotFoundError(f"No config JSON in {run_dir}")
    with open(str(cfg_files[0])) as f:
        cfg = json.load(f)

    norm_stats = None
    nmp = run_dir / 'normalization_metadata.json'
    if nmp.exists():
        with open(str(nmp)) as f:
            nm = json.load(f)
        nsp = nm.get('normalization_stats_path')
        if nsp and Path(nsp).exists():
            norm_stats = torch.load(nsp, map_location='cpu', weights_only=False)

    raw_sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace('module.', ''): v
          for k, v in raw_sd.get('model_state_dict', raw_sd).items()}

    nf  = 5
    gcp = Path(cfg['test_data_dir']) / 'gwt_config.json'
    if gcp.exists():
        with open(str(gcp)) as f:
            nf = json.load(f)['num_filters']

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT
    un  = cfg.get('use_normalization', False)
    kw  = dict(num_filters=nf,
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
               use_normalization=un,
               normalization_stats=norm_stats)
    mn = cfg.get('model_name', 'GNNNeuralIGWT')
    if mn == 'GUNetNeuralIGWT':
        kw.update(fusion_type=cfg.get('fusion_type', 'mlp'),
                  pooling_type=cfg.get('pooling_type', 'top_k_pooling'),
                  pooling_ratio=cfg.get('pooling_ratio', 0.5))
        model = GUNetNeuralIGWT(**kw).to(device)
    else:
        model = GNNNeuralIGWT(**kw).to(device)
    model.load_state_dict(sd); model.eval()
    return model, cfg, norm_stats


def infer_gwcs(model, cfg, norm_stats, coeffs, edge_index,
               matrix_cache, device, field_index=0):
    from models.sparsification_utils import multilevel_importance_sampling
    c    = coeffs.to(device)
    mask = multilevel_importance_sampling(coeffs=c,
                                          keep_ratio=cfg.get('keep_ratio', 0.01))
    bv   = torch.zeros(c.shape[0], dtype=torch.long, device=device)
    ptr  = torch.tensor([0, c.shape[0]], device=device)
    cd   = matrix_cache.get([0])
    with torch.no_grad():
        xh = model(coeffs=c, sparsification_mask=mask,
                   edge_index=edge_index.to(device),
                   batch=bv, chebyshev_data=cd, batch_ptr=ptr)
    xh = xh.squeeze().cpu()
    if cfg.get('use_normalization', False) and norm_stats is not None:
        fname = list(norm_stats.keys())[field_index]
        xh = xh * norm_stats[fname]['std'] + norm_stats[fname]['mean']
    return xh


# =============================================================================
# Temporal metric computation
# =============================================================================

def rms(v: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(v.float() ** 2)))


def compute_temporal_metrics(delta_hat: torch.Tensor, delta_true: torch.Tensor):
    """
    delta_hat  = x_hat[t+1] - x_hat[t]   (reconstructed temporal change)
    delta_true = x[t+1]    - x[t]         (ground-truth temporal change)

    Returns:
        dRMSE   float  — RMSE of the error in the temporal change (physical units)
        TC      float  — Pearson correlation in [−1, 1]
        mag     float  — RMS magnitude of the true temporal change (reference)
        rel_err float  — dRMSE / mag  (relative temporal error; 0=perfect, 1=no tracking)
    """
    mag   = rms(delta_true)
    drmse = rms(delta_hat - delta_true)
    rel   = drmse / mag if mag > 1e-12 else float('nan')

    dh = delta_hat.numpy().ravel().astype(np.float64)
    dt = delta_true.numpy().ravel().astype(np.float64)
    if np.std(dh) < 1e-12 or np.std(dt) < 1e-12:
        tc = 0.0
    else:
        tc = float(np.corrcoef(dh, dt)[0, 1])

    return drmse, tc, mag, rel


# =============================================================================
# Plotting
# =============================================================================

def _apply_legend(ax, ncol=4):
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=ncol, frameon=True, framealpha=0.95, borderaxespad=0)


def plot_drmse_curves(transitions, drmse_dict, mag_list, output_path: str,
                      active_keys):
    """Line plot: dRMSE per transition + ground-truth change magnitude."""
    fig, ax = plt.subplots(figsize=(22, 10))
    xs = np.arange(1, len(transitions) + 1)

    for key in active_keys:
        s = MODEL_STYLES[key]
        ax.plot(xs, drmse_dict[key], color=s['color'], lw=s['lw'],
                label=s['label'], zorder=s['zorder'])

    # Reference: magnitude of ground-truth temporal change
    ax.plot(xs, mag_list, color='black', lw=2.0, ls='--',
            label=r'$\|\Delta x_{\rm orig}\|_{\rm rms}$',
            alpha=0.55, zorder=0)

    ax.set_xlabel('Transition Index  (t → t+1)')
    ax.set_ylabel('Temporal $\\Delta$RMSE')
    _apply_legend(ax, ncol=min(len(active_keys) + 1, 5))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tc_curves(transitions, tc_dict, output_path: str, active_keys):
    """Line plot: temporal Pearson correlation per transition."""
    fig, ax = plt.subplots(figsize=(22, 10))
    xs = np.arange(1, len(transitions) + 1)

    for key in active_keys:
        s = MODEL_STYLES[key]
        ax.plot(xs, tc_dict[key], color=s['color'], lw=s['lw'],
                label=s['label'], zorder=s['zorder'])

    ax.set_xlabel('Transition Index  (t → t+1)')
    ax.set_ylabel('Temporal Correlation')
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, color='k', ls='--', lw=1.5, alpha=0.3)
    ax.axhline(y=0.0, color='k', ls=':',  lw=1.5, alpha=0.3)
    _apply_legend(ax, ncol=min(len(active_keys), 4))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary(summary: dict, output_path: str, active_keys):
    """
    Two-panel bar chart:
      Left  — mean dRMSE (lower = better temporal-gradient accuracy)
      Right — mean TC    (higher = better temporal correlation)
    """
    labels = [MODEL_STYLES[k]['label'] for k in active_keys]
    colors = [MODEL_STYLES[k]['color'] for k in active_keys]
    x      = np.arange(len(active_keys))
    w      = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 10))

    # ── Left: mean dRMSE ─────────────────────────────────────────────────
    vals  = [summary[k]['mean_drmse']  for k in active_keys]
    errs  = [summary[k]['std_drmse']   for k in active_keys]
    bars  = ax1.bar(x, vals, width=w, color=colors, edgecolor='k',
                    linewidth=1.5, yerr=errs, capsize=6, error_kw={'lw': 2})
    ax1.set_xticks(x);  ax1.set_xticklabels(labels, fontsize=28)
    ax1.set_ylabel('Mean Temporal $\\Delta$RMSE', fontsize=28)
    ax1.grid(axis='y', alpha=0.4)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2., v * 1.03,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

    # ── Right: mean TC ────────────────────────────────────────────────────
    vals  = [summary[k]['mean_tc']  for k in active_keys]
    errs  = [summary[k]['std_tc']   for k in active_keys]
    bars  = ax2.bar(x, vals, width=w, color=colors, edgecolor='k',
                    linewidth=1.5, yerr=errs, capsize=6, error_kw={'lw': 2})
    ax2.set_xticks(x);  ax2.set_xticklabels(labels, fontsize=28)
    ax2.set_ylabel('Mean Temporal Correlation', fontsize=28)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(y=1.0, color='k', ls='--', lw=1.5, alpha=0.4)
    ax2.grid(axis='y', alpha=0.4)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2., v + 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Temporal-consistency analysis for PDE reconstruction models')
    ap.add_argument('--test-dir',            required=True)
    ap.add_argument('--gwt-checkpoint',      required=True)
    ap.add_argument('--gwt-run-dir',         required=True)
    ap.add_argument('--vgae-checkpoint',     default=None,
                    help='SGAE-99K checkpoint')
    ap.add_argument('--vgae-50k-checkpoint', default=None,
                    help='SGAE-50K checkpoint')
    ap.add_argument('--gxn-checkpoint',      default=None,
                    help='GXN checkpoint')
    ap.add_argument('--field-index',  type=int,   default=0)
    ap.add_argument('--keep-ratio',   type=float, default=0.01)
    ap.add_argument('--num-samples',  type=int,   default=None,
                    help='Limit to first N test samples (default: all)')
    ap.add_argument('--signal-std',   type=float, default=None,
                    help='Signal std for normalisation (computed from test data if omitted)')
    ap.add_argument('--output-dir',   required=True)
    args = ap.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dir = Path(args.test_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device     : {device}")
    print(f"Test dir   : {test_dir}")
    print(f"Output dir : {args.output_dir}\n")

    # ── Shared graph data ────────────────────────────────────────────────────
    edge_index, num_nodes, nr, nc = load_shared(test_dir)
    ei_dev = edge_index.to(device)
    print(f"Graph: {num_nodes} nodes  ({nr} × {nc})")

    # ── Sample files ─────────────────────────────────────────────────────────
    sample_files = get_sample_files(test_dir)
    if args.num_samples:
        sample_files = sample_files[:args.num_samples]
    N = len(sample_files)
    print(f"Evaluating {N} test samples\n")

    # ── Load models ──────────────────────────────────────────────────────────
    gxn_sm = gxn_rm = gxn_mu = gxn_sd = None
    if args.gxn_checkpoint and Path(args.gxn_checkpoint).exists():
        print("Loading GXN …")
        gxn_sm, gxn_rm, gxn_mu, gxn_sd = load_gxn(args.gxn_checkpoint, device)

    vgae_m = vgae_mu = vgae_sd = None
    if args.vgae_checkpoint and Path(args.vgae_checkpoint).exists():
        print("Loading SGAE-99K …")
        vgae_m, vgae_mu, vgae_sd = load_vgae(
            args.vgae_checkpoint, args.keep_ratio, device)

    vgae50_m = vgae50_mu = vgae50_sd = None
    if args.vgae_50k_checkpoint and Path(args.vgae_50k_checkpoint).exists():
        print("Loading SGAE-50K …")
        vgae50_m, vgae50_mu, vgae50_sd = load_vgae(
            args.vgae_50k_checkpoint, args.keep_ratio, device)

    print("Loading GWCS …")
    gwcs_m, gwcs_cfg, gwcs_ns = load_gwcs(
        Path(args.gwt_run_dir), args.gwt_checkpoint, device)

    print("Building Chebyshev cache …")
    from data.dataset_turbulent import (GraphSignalDatasetTurbulent,
                                        create_matrix_cache_turbulent)
    ds_tmp = GraphSignalDatasetTurbulent(
        base_data_dir=args.test_dir,
        field_index=args.field_index,
        preload_to_memory=False,
        use_normalization=False)
    mc = create_matrix_cache_turbulent(dataset=ds_tmp, device=device)
    print()

    # Determine which model keys are active (checkpoint provided + found)
    active_keys = ['gwcs']   # GWCS always present
    if vgae_m    is not None: active_keys.append('sgae_99k')
    if vgae50_m  is not None: active_keys.append('sgae_50k')
    if gxn_sm    is not None: active_keys.append('gxn')

    # ── Streaming temporal-metrics pass ──────────────────────────────────────
    # We keep only TWO consecutive samples in memory at any time.
    # This is crucial for large graphs (N ≥ 49K) where stacking everything
    # would exhaust RAM.

    # Welford's online algorithm for computing signal_std across all test samples.
    # Used for signal-normalised dRMSE:  norm = dRMSE / signal_std
    _sig_n    = 0
    _sig_mean = 0.0
    _sig_M2   = 0.0   # running sum of squared deviations

    drmse_dict = {k: [] for k in active_keys}
    tc_dict    = {k: [] for k in active_keys}
    rel_dict   = {k: [] for k in active_keys}
    mag_list   = []
    transitions = []   # (i, i+1) index pairs that were not boundaries
    skipped     = 0

    # These will hold the PREVIOUS frame's data
    prev_orig  = None
    prev_xhat  = {}
    prev_traj  = None
    prev_t_idx = None

    print("="*60)
    print("Streaming inference + temporal metric computation …")
    print("="*60)

    for i, fpath in enumerate(sample_files):
        x, coeffs, traj_idx, t_idx = load_sample(fpath, args.field_index)

        # ── Update running signal statistics (Welford, vectorised over nodes) ─
        _xnp     = x.numpy()
        _n_new   = int(_xnp.size)
        _mn_new  = float(_xnp.mean())
        _var_new = float(_xnp.var())    # population var of this sample
        if _sig_n == 0:
            _sig_mean = _mn_new
            _sig_M2   = _var_new * _n_new
        else:
            _delta    = _mn_new - _sig_mean
            _sig_mean = (_sig_n * _sig_mean + _n_new * _mn_new) / (_sig_n + _n_new)
            _sig_M2  += _var_new * _n_new + (_delta ** 2) * _sig_n * _n_new / (_sig_n + _n_new)
        _sig_n += _n_new

        # ── Current frame reconstructions ─────────────────────────────────
        cur_xhat = {}

        # GWCS
        cur_xhat['gwcs'] = infer_gwcs(
            gwcs_m, gwcs_cfg, gwcs_ns,
            coeffs, edge_index, mc, device, args.field_index)

        # SGAE-99K
        if vgae_m is not None:
            cur_xhat['sgae_99k'] = infer_vgae(
                vgae_m, vgae_mu, vgae_sd, x, ei_dev, device)

        # SGAE-50K
        if vgae50_m is not None:
            cur_xhat['sgae_50k'] = infer_vgae(
                vgae50_m, vgae50_mu, vgae50_sd, x, ei_dev, device)

        # GXN
        if gxn_sm is not None:
            cur_xhat['gxn'] = infer_gxn(
                gxn_sm, gxn_rm, gxn_mu, gxn_sd,
                x, ei_dev, args.keep_ratio, device)

        # ── Temporal metrics (only if prev frame exists and same trajectory) ──
        if prev_orig is not None:
            boundary = is_trajectory_boundary(
                prev_traj, prev_t_idx, traj_idx, t_idx)

            if boundary:
                skipped += 1
                print(f"  [{i+1}/{N}]  SKIPPED (trajectory boundary: "
                      f"traj {prev_traj}→{traj_idx}, t {prev_t_idx}→{t_idx})")
            else:
                delta_true = x - prev_orig  # true temporal change
                for key in active_keys:
                    delta_hat = cur_xhat[key] - prev_xhat[key]
                    drmse, tc, mag, rel = compute_temporal_metrics(
                        delta_hat, delta_true)
                    drmse_dict[key].append(drmse)
                    tc_dict[key].append(tc)
                    rel_dict[key].append(rel)

                mag_list.append(rms(delta_true))
                transitions.append((i - 1, i))

                if (i + 1) % 50 == 0 or i == N - 1:
                    print(f"  [{i+1}/{N}]  "
                          f"GWCS dRMSE={drmse_dict['gwcs'][-1]:.5f}  "
                          f"TC={tc_dict['gwcs'][-1]:.4f}")

        # Advance: store current as previous
        prev_orig  = x
        prev_xhat  = cur_xhat
        prev_traj  = traj_idx
        prev_t_idx = t_idx

    n_valid = len(transitions)
    print(f"\n  Valid transitions : {n_valid}")
    print(f"  Skipped (boundary): {skipped}")

    if n_valid == 0:
        print("ERROR: No valid transitions found. "
              "Check trajectory metadata in .pt files.")
        return

    # ── Compute signal_std from Welford accumulators (or use CLI override) ────
    if getattr(args, 'signal_std', None) is not None:
        signal_std = float(args.signal_std)
        print(f"\nSignal std (CLI override): {signal_std:.6f}")
    elif _sig_n > 0:
        import math as _math
        signal_std = _math.sqrt(_sig_M2 / _sig_n)
        print(f"\nSignal std (computed from {_sig_n:,} values across {N} samples): "
              f"{signal_std:.6f}")
    else:
        signal_std = 1.0
        print("\nWARNING: could not compute signal_std — defaulting to 1.0")

    # ── Summary statistics ────────────────────────────────────────────────────
    summary = {}
    for key in active_keys:
        d = [v for v in drmse_dict[key] if not np.isnan(v)]
        t = [v for v in tc_dict[key]    if not np.isnan(v)]
        r = [v for v in rel_dict[key]   if not np.isnan(v)]
        summary[key] = {
            'label':        MODEL_STYLES[key]['label'],
            'mean_drmse':   float(np.mean(d))  if d else None,
            'std_drmse':    float(np.std(d))   if d else None,
            'mean_tc':      float(np.mean(t))  if t else None,
            'std_tc':       float(np.std(t))   if t else None,
            'mean_rel_err': float(np.mean(r))  if r else None,
            'n_transitions': len(d),
        }
    summary['_reference'] = {
        'mean_orig_change_magnitude': float(np.mean(mag_list)),
        'std_orig_change_magnitude':  float(np.std(mag_list)),
        'n_valid_transitions': n_valid,
        'n_skipped_boundary_transitions': skipped,
    }

    # Save JSON
    json_path = os.path.join(args.output_dir, 'temporal_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Print table ──────────────────────────────────────────────────────────
    ref_mag = summary['_reference']['mean_orig_change_magnitude']
    print("\n" + "="*72)
    print("TEMPORAL CONSISTENCY RESULTS  (Physical Units)")
    print(f"  Reference: mean ||Δx_orig||_rms = {ref_mag:.6f}  (ground-truth change)")
    print("="*72)
    print(f"  {'Model':<12}  {'dRMSE (mean±std)':>24}  "
          f"{'TC (mean±std)':>20}  {'RelErr':>10}")
    print("  " + "-"*68)
    for key in active_keys:
        s = summary[key]
        print(f"  {s['label']:<12}  "
              f"{s['mean_drmse']:>10.5f} ± {s['std_drmse']:<10.5f}  "
              f"{s['mean_tc']:>8.4f} ± {s['std_tc']:<8.4f}  "
              f"{s['mean_rel_err']:>8.4f}")
    print("="*72)

    # ── Signal-normalised dRMSE table ────────────────────────────────────────
    print("\n" + "="*72)
    print(f"TEMPORAL CONSISTENCY RESULTS  (Signal-Normalised,  σ = {signal_std:.6f})")
    print("  norm_dRMSE = dRMSE ÷ σ_signal  |  0 → perfect;  1 → error = signal σ")
    print("="*72)
    print(f"  {'Model':<12}  {'norm_dRMSE (mean±std)':>26}  {'TC (mean±std)':>20}")
    print("  " + "-"*60)
    for key in active_keys:
        s = summary[key]
        nd_mean = s['mean_drmse'] / signal_std
        nd_std  = s['std_drmse']  / signal_std
        print(f"  {s['label']:<12}  "
              f"{nd_mean:>12.6f} ± {nd_std:<12.6f}  "
              f"{s['mean_tc']:>8.4f} ± {s['std_tc']:<8.4f}")
    print("="*72)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_drmse_curves(
        transitions, drmse_dict, mag_list,
        os.path.join(args.output_dir, 'temporal_drmse_curves.png'),
        active_keys)

    plot_tc_curves(
        transitions, tc_dict,
        os.path.join(args.output_dir, 'temporal_tc_curves.png'),
        active_keys)

    plot_summary(
        summary,
        os.path.join(args.output_dir, 'temporal_summary.png'),
        active_keys)

    # ── Signal-normalised dRMSE plots ────────────────────────────────────────
    # norm_i = drmse_i / signal_std
    # signal_std = population std of ALL test-sample signal values (physical units)
    #
    # norm = 0   → perfect temporal tracking (zero delta error)
    # norm = 1   → temporal error equals signal standard deviation
    # This is an ABSOLUTE metric — independent of which models are in the comparison.
    # TC is already in [-1,1] so it is not rescaled.
    print("\n" + "="*72)
    print("Generating Signal-Normalised dRMSE plots  (dRMSE ÷ σ_signal)")
    print(f"  σ_signal = {signal_std:.6f}")
    print("="*72)

    if signal_std < 1e-12:
        print("  WARNING: signal_std ≈ 0 — skipping normalised plots.")
    else:
        drmse_means = {k: summary[k]['mean_drmse'] for k in active_keys
                       if summary[k]['mean_drmse'] is not None}
        drmse_stds  = {k: summary[k]['std_drmse']  for k in active_keys
                       if summary[k]['std_drmse']  is not None}

        for k in active_keys:
            if k in drmse_means:
                print(f"  {MODEL_STYLES[k]['label']:<12}  "
                      f"mean_dRMSE={drmse_means[k]:.5f}  →  "
                      f"norm={drmse_means[k]/signal_std:.6f}")

        def _sig_norm_d(arr):
            """Divide each dRMSE value by signal_std."""
            return [v / signal_std for v in arr]

        # ── Signal-norm dRMSE curve ───────────────────────────────────────
        xs_n = np.arange(1, len(transitions) + 1)
        fig_nd, ax_nd = plt.subplots(figsize=(22, 10))
        for key in active_keys:
            s = MODEL_STYLES[key]
            ax_nd.plot(xs_n, _sig_norm_d(drmse_dict[key]),
                       color=s['color'], lw=s['lw'],
                       label=s['label'], zorder=s['zorder'])
        ax_nd.axhline(1.0, color='gray', lw=1.2, ls='--', alpha=0.6,
                      label='dError = signal σ')
        ax_nd.set_xlabel('Transition Index  (t → t+1)')
        ax_nd.set_ylabel(r'Temporal $\Delta$RMSE  ($\div\,\sigma_{\rm signal}$)')
        _apply_legend(ax_nd, ncol=min(len(active_keys) + 1, 5))
        ax_nd.grid(True, alpha=0.3)
        fig_nd.tight_layout()
        out_nd = os.path.join(args.output_dir, 'temporal_drmse_normalised.png')
        fig_nd.savefig(out_nd, dpi=200, bbox_inches='tight')
        plt.close(fig_nd)
        print(f"  Saved: {out_nd}")

        # ── Signal-norm summary bar chart (norm dRMSE + TC) ───────────────
        labels_n = [MODEL_STYLES[k]['label'] for k in active_keys]
        colors_n = [MODEL_STYLES[k]['color'] for k in active_keys]
        x_n = np.arange(len(active_keys))
        w_n = 0.55

        fig_ns, (ax_ns1, ax_ns2) = plt.subplots(1, 2, figsize=(26, 10))

        # Left: signal-normalised mean dRMSE
        vals_n  = [drmse_means.get(k, 0.0) / signal_std for k in active_keys]
        errs_n  = [drmse_stds.get(k, 0.0)  / signal_std for k in active_keys]
        bars_n = ax_ns1.bar(x_n, vals_n, width=w_n, color=colors_n,
                            edgecolor='k', linewidth=1.5,
                            yerr=errs_n, capsize=6, error_kw={'lw': 2})
        ax_ns1.set_xticks(x_n);  ax_ns1.set_xticklabels(labels_n, fontsize=28)
        ax_ns1.set_ylabel(r'Mean Temporal $\Delta$RMSE  ($\div\,\sigma_{\rm signal}$)',
                          fontsize=26)
        ax_ns1.axhline(1.0, color='gray', lw=1.2, ls='--', alpha=0.6)
        ax_ns1.grid(axis='y', alpha=0.4)
        for bar, v in zip(bars_n, vals_n):
            ax_ns1.text(bar.get_x() + bar.get_width() / 2., v * 1.03,
                        f'{v:.4f}', ha='center', va='bottom',
                        fontsize=22, fontweight='bold')

        # Right: TC (natural scale, unchanged)
        vals_tc = [summary[k]['mean_tc'] for k in active_keys]
        errs_tc = [summary[k]['std_tc']  for k in active_keys]
        bars_tc = ax_ns2.bar(x_n, vals_tc, width=w_n, color=colors_n,
                             edgecolor='k', linewidth=1.5,
                             yerr=errs_tc, capsize=6, error_kw={'lw': 2})
        ax_ns2.set_xticks(x_n);  ax_ns2.set_xticklabels(labels_n, fontsize=28)
        ax_ns2.set_ylabel('Mean Temporal Correlation', fontsize=28)
        ax_ns2.set_ylim(0, 1.15)
        ax_ns2.axhline(y=1.0, color='k', ls='--', lw=1.5, alpha=0.4)
        ax_ns2.grid(axis='y', alpha=0.4)
        for bar, v in zip(bars_tc, vals_tc):
            ax_ns2.text(bar.get_x() + bar.get_width() / 2., v + 0.01,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=22, fontweight='bold')

        fig_ns.tight_layout()
        out_ns = os.path.join(args.output_dir, 'temporal_summary_normalised.png')
        fig_ns.savefig(out_ns, dpi=200, bbox_inches='tight')
        plt.close(fig_ns)
        print(f"  Saved: {out_ns}")

        # ── Augment JSON with signal-normalised stats ─────────────────────
        summary['_signal_norm_drmse'] = {
            'signal_std':    float(signal_std),
            'normalization': 'drmse_div_signal_std',
        }
        for k in active_keys:
            if summary[k]['mean_drmse'] is not None:
                summary[k]['norm_mean_drmse'] = float(
                    summary[k]['mean_drmse'] / signal_std)
                summary[k]['norm_std_drmse']  = float(
                    summary[k]['std_drmse']  / signal_std)

        # Re-save JSON with signal-norm stats appended
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Updated: {json_path}  (added signal-normalised dRMSE fields)")

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == '__main__':
    main()
