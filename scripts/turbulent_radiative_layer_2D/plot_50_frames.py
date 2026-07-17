# -*- coding: utf-8 -*-
"""
plot_50_frames.py

Generates N individual reconstruction frames per model, saving one PNG per
model per sample in {output_dir}/.  Format: Original | Reconstruction | Error
(same 3-panel layout used by the training baseline scripts).

Output files:
  gwcs_sample{i}.png
  svgae_99k_sample{i}.png
  svgae_50k_sample{i}.png
  gxn_sample{i}.png
  fastgsss_sample{i}.png  (only when --run-classical)
  bsgda_sample{i}.png
  rsbs_sample{i}.png

All PNGs are always overwritten so re-running with a newer checkpoint
produces fresh results.

Usage
-----
  python plot_50_frames.py \
      --test-dir       <path/to/processed/test> \
      --gwt-run-dir    <path/to/runs_xxx/TIMESTAMP_dir> \
      --gwt-checkpoint <path/to/runs_xxx/TIMESTAMP_dir/checkpoints/NN.pt> \
      --vgae-checkpoint     <.../svgae_99k/best_checkpoint.pt> \
      --vgae-50k-checkpoint <.../svgae_50k/best_checkpoint.pt> \
      --gxn-checkpoint      <.../gxn_99k/best_checkpoint.pt> \
      [--run-classical]   # re-runs classical algorithms inline (slow on large graphs)
      --num-frames 50     # default: 50
      --output-dir  <path/to/output>
"""

import sys
import os
import json
import time
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data loading
# =============================================================================

def load_test_data(test_dir: str, field_index: int = 0, max_samples: int = None):
    """
    Load test data from disk.  Only the first `max_samples` files are read
    to avoid allocating the full (N_nodes × N_filters × N_test) tensor when
    we only need a subset of samples.
    """
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
    if max_samples is not None:
        sample_files = sample_files[:max_samples]

    signals, coeffs_list = [], []
    for f in sample_files:
        s = torch.load(f, map_location='cpu', weights_only=False)
        signals.append(s.x[:, field_index].float())
        if hasattr(s, 'coeffs') and s.coeffs is not None:
            coeffs_list.append(s.coeffs[:, field_index, :].float())

    signals = torch.stack(signals, dim=1)           # (N_nodes, n_loaded)
    coeffs  = torch.stack(coeffs_list, dim=2) if coeffs_list else None  # (N_nodes, F, n_loaded)
    return signals, coeffs, edge_index, num_nodes, num_rows, num_cols


# =============================================================================
# Model loaders + inference  (re-use same logic as plot_central_8_samples.py)
# =============================================================================

def load_gxn(ckpt_path, device):
    from baselines.gxn.neural_sampling import NeuralSampling, NeuralRecovery
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sm = NeuralSampling(1, ckpt.get('hidden_dim', 128), ckpt.get('max_hops', 2)).to(device)
    rm = NeuralRecovery(1, ckpt.get('recovery_K', 8), ckpt.get('recovery_L', 5)).to(device)
    sm.load_state_dict(ckpt['sampling_model']); sm.eval()
    rm.load_state_dict(ckpt['recovery_model']); rm.eval()
    return sm, rm, ckpt.get('norm_mean', 0.0), ckpt.get('norm_std', 1.0)


def infer_gxn(sm, rm, norm_mean, norm_std, x_orig, edge_index, keep_ratio, device):
    N = x_orig.shape[0]
    M = max(1, int(keep_ratio * N))
    mu = torch.tensor(norm_mean, device=device)
    sd = torch.tensor(norm_std,  device=device)
    xi = ((x_orig.to(device) - mu) / sd).unsqueeze(1)
    with torch.no_grad():
        att, _, _ = sm(xi, edge_index)
        sc = att.squeeze()
        if sc.dim() == 0: sc = sc.unsqueeze(0)
        _, ti = torch.topk(sc, k=M)
        mp = torch.zeros_like(xi); mp[ti] = 1.0
        xh = rm(mp * att * xi, edge_index, att, mp)
    return (xh.squeeze() * sd + mu).cpu()


def load_vgae(ckpt_path, keep_ratio, device):
    from baselines.sparse_VGAE.gae import GAE
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GAE(1, ckpt.get('latent_dim', 128), 1,
                ckpt.get('hidden_layers', 4), ckpt.get('hidden_layers', 4),
                'SAGEConv', 'silu', keep_ratio, False, 'ste_annealing',
                1.0, 0.01, 1000).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt.get('norm_mean', 0.0), ckpt.get('norm_std', 1.0)


def infer_vgae(model, norm_mean, norm_std, x_orig, edge_index, device):
    mu = torch.tensor(norm_mean, device=device)
    sd = torch.tensor(norm_std,  device=device)
    xi = ((x_orig.to(device) - mu) / sd).unsqueeze(1)
    with torch.no_grad():
        xh = model.decode(model.encode(xi, edge_index), edge_index)
    return (xh.squeeze() * sd + mu).cpu()


def load_nigwt(run_dir, ckpt_path, device):
    run_dir = Path(run_dir)
    with open(str(run_dir / 'reconstructor_pretraining_config_turbulent.json')) as f:
        cfg = json.load(f)

    norm_stats = None
    nmp = run_dir / 'normalization_metadata.json'
    if nmp.exists():
        with open(str(nmp)) as f:
            nm = json.load(f)
        nsp = nm.get('normalization_stats_path')
        if nsp and Path(nsp).exists():
            norm_stats = torch.load(nsp, map_location='cpu', weights_only=False)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace('module.', ''): v
          for k, v in ckpt.get('model_state_dict', ckpt).items()}

    td  = Path(cfg['test_data_dir'])
    nf  = 5
    gcp = td / 'gwt_config.json'
    if gcp.exists():
        with open(str(gcp)) as f:
            nf = json.load(f)['num_filters']

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT
    mn  = cfg.get('model_name', 'GNNNeuralIGWT')
    un  = cfg.get('use_normalization', False)
    kw  = dict(num_filters=nf, hidden_dim=cfg['hidden_dim'],
               output_dim=cfg['output_dim'],
               activation_function=cfg['activation_function'],
               encoder_type=cfg['encoder_type'],
               decoder_type=cfg['decoder_type'],
               conv_layer_type=cfg['conv_layer_type'],
               processor_num_layers=cfg['processor_num_layers'],
               predict_residual=cfg['predict_residual'],
               enable_scale_embeddings=cfg['enable_scale_embeddings'],
               igwt_mode=cfg['igwt_mode'],
               use_normalization=un, normalization_stats=norm_stats)
    if mn == 'GUNetNeuralIGWT':
        kw.update(fusion_type=cfg.get('fusion_type', 'mlp'),
                  pooling_type=cfg.get('pooling_type', 'top_k_pooling'),
                  pooling_ratio=cfg.get('pooling_ratio', 0.5))
        model = GUNetNeuralIGWT(**kw).to(device)
    else:
        model = GNNNeuralIGWT(**kw).to(device)
    model.load_state_dict(sd); model.eval()
    return model, cfg, norm_stats


def infer_nigwt(model, cfg, norm_stats, coeffs_sample, edge_index,
                matrix_cache, device, field_index=0):
    from models.sparsification_utils import multilevel_importance_sampling
    c  = coeffs_sample.to(device)
    kr = cfg.get('keep_ratio', 0.01)
    mask = multilevel_importance_sampling(coeffs=c, keep_ratio=kr)
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
    sp = {'fastgsss_bandwidth': 'exact', 'fastgsss_nu': 75.0,
          'fastgsss_order': 12, 'fastgsss_recon_order': 3,
          'bsgda_mu': 0.01, 'bsgda_epsilon': 1e-5, 'bsgda_p_hops': 12,
          'bsgda_reg_order': 1,
          'rsbs_bandwidth_mode': 'exact', 'rsbs_num_estimation': 1,
          'rsbs_epsilon': 1e-2, 'rsbs_order': 30,
          'rsbs_mu': 0.01, 'rsbs_reg_order': 1, 'rsbs_lap_type': 'comb'}
    _, xh = sampling_and_reconstruction(
        method_name=method, x=x_norm.unsqueeze(-1),
        args={'num_nodes': num_nodes, 'sample_size': ss,
              'thgsp_graph': G, 'thgsp_adj': adj,
              'thgsp_L_comb': Lco, 'thgsp_L_sym': Ls,
              'signal_bandwidth': ss},
        search_params=sp)
    return xh.squeeze()


# =============================================================================
# Frame saving helper
# =============================================================================

def _draw_frame(x_o, x_r, err, vmin, vmax, show_rmse, rmse_val):
    """
    Internal helper: build a figure for one model reconstruction.
    Returns (fig, rmse_val).  Caller is responsible for saving & closing.

    show_rmse=True  → includes "RMSE = X.XXXXXX" as a suptitle
    show_rmse=False → no suptitle; figure is slightly shorter (no wasted top space)
    """
    # ── Font sizes (3× larger than defaults for clear readability) ──────────
    FS_TITLE    = 30   # panel titles ("Original Signal" etc.)
    FS_LABEL    = 27   # x / y axis labels
    FS_TICK     = 24   # tick-label numbers on axes and colorbars
    FS_SUPTITLE = 32   # RMSE suptitle

    # ── Tick mark geometry: 5× the matplotlib defaults ──────────────────────
    #   Default major: length=3.5pt, width=0.8pt  →  ×5 = 17.5pt / 4pt
    #   Default minor: length=2.0pt, width=0.6pt  →  ×5 = 10pt   / 3pt
    TK_LEN_MAJ  = 18   # major tick length  (pt)
    TK_WID_MAJ  = 4    # major tick width   (pt)
    TK_LEN_MIN  = 10   # minor tick length  (pt)
    TK_WID_MIN  = 3    # minor tick width   (pt)
    TK_PAD      = 12   # gap between tick end and label (pt)

    fig_h = 8 if show_rmse else 7
    fig, axes = plt.subplots(1, 3, figsize=(27, fig_h))

    # ── Panel 0: Original Signal ─────────────────────────────────────────────
    im0 = axes[0].imshow(x_o, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Signal', fontsize=FS_TITLE, pad=12)
    cb0 = plt.colorbar(im0, ax=axes[0])
    cb0.ax.tick_params(which='major', labelsize=FS_TICK,
                       length=TK_LEN_MAJ, width=TK_WID_MAJ, pad=TK_PAD)
    cb0.ax.tick_params(which='minor', length=TK_LEN_MIN, width=TK_WID_MIN)

    # ── Panel 1: Reconstruction ───────────────────────────────────────────────
    im1 = axes[1].imshow(x_r, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('Reconstruction', fontsize=FS_TITLE, pad=12)
    cb1 = plt.colorbar(im1, ax=axes[1])
    cb1.ax.tick_params(which='major', labelsize=FS_TICK,
                       length=TK_LEN_MAJ, width=TK_WID_MAJ, pad=TK_PAD)
    cb1.ax.tick_params(which='minor', length=TK_LEN_MIN, width=TK_WID_MIN)

    # ── Panel 2: Absolute Error ───────────────────────────────────────────────
    im2 = axes[2].imshow(err, aspect='auto', cmap='Reds', vmin=0)
    axes[2].set_title('Absolute Error', fontsize=FS_TITLE, pad=12)
    cb2 = plt.colorbar(im2, ax=axes[2])
    cb2.ax.tick_params(which='major', labelsize=FS_TICK,
                       length=TK_LEN_MAJ, width=TK_WID_MAJ, pad=TK_PAD)
    cb2.ax.tick_params(which='minor', length=TK_LEN_MIN, width=TK_WID_MIN)

    # ── Axis tick marks ───────────────────────────────────────────────────────
    for ax in axes:
        ax.set_xlabel('x', fontsize=FS_LABEL, labelpad=10)
        ax.set_ylabel('y', fontsize=FS_LABEL, labelpad=10)
        ax.tick_params(axis='both', which='major',
                       labelsize=FS_TICK, length=TK_LEN_MAJ,
                       width=TK_WID_MAJ, pad=TK_PAD)
        ax.tick_params(axis='both', which='minor',
                       length=TK_LEN_MIN, width=TK_WID_MIN)

    # ── Optional RMSE suptitle ────────────────────────────────────────────────
    if show_rmse:
        fig.suptitle(f'RMSE = {rmse_val:.6f}', fontsize=FS_SUPTITLE, y=1.01)

    plt.tight_layout()
    return fig


def save_frame(x_orig, x_recon, num_rows, num_cols, filepath, filepath_clean=None):
    """
    Save two variants of a 3-panel reconstruction frame:
      filepath        — with "RMSE = X.XXXXXX" suptitle
      filepath_clean  — same frame but NO suptitle (cleaner layout, saves vertical space)

    Pass filepath_clean=None to skip generating the clean variant.
    Returns the scalar RMSE value.
    """
    x_o = x_orig.cpu().numpy().reshape(num_rows, num_cols)
    x_r = x_recon.cpu().numpy().reshape(num_rows, num_cols)
    err = np.abs(x_o - x_r)
    # %%%CHANGED: anchor vmin/vmax to the ORIGINAL signal only, not the reconstruction.
    # Previously: min/max of (orig, recon) combined — caused GXN's near-uniform reconstruction
    # to push the colormap range wider (e.g. 0–125 vs 0–100 for other models), making the
    # "Original Signal" panel look lighter in GXN's row even though it's the same ground truth.
    # Fix: all rows now use the same colormap scale (the ground-truth range), so the Original
    # Signal panels are visually identical across all model rows.
    vmin, vmax = float(x_o.min()), float(x_o.max())
    rmse = float(np.sqrt(np.mean((x_o - x_r) ** 2)))

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Variant 1: with RMSE header
    fig = _draw_frame(x_o, x_r, err, vmin, vmax, show_rmse=True,  rmse_val=rmse)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Variant 2: clean (no header) — saved alongside with "_clean" suffix
    if filepath_clean is None:
        # Auto-generate clean path: insert "_clean" before the extension
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
        description="Save N individual reconstruction frames per model")
    parser.add_argument('--test-dir',            required=True)
    parser.add_argument('--gwt-checkpoint',      required=True)
    parser.add_argument('--gwt-run-dir',         required=True)
    parser.add_argument('--vgae-checkpoint',     default=None)
    parser.add_argument('--vgae-50k-checkpoint', default=None)
    parser.add_argument('--gxn-checkpoint',      default=None)
    parser.add_argument('--run-classical',       action='store_true',
                        help='Re-run classical algorithms inline (slow on large graphs)')
    parser.add_argument('--field-index',  type=int,   default=0)
    parser.add_argument('--keep-ratio',   type=float, default=0.01)
    parser.add_argument('--num-frames',   type=int,   default=50,
                        help='Number of samples to render (default 50, first N samples)')
    parser.add_argument('--output-dir',   default='baseline_results/frames_50')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load only the first num_frames test samples — do NOT load the whole test set.
    # Stacking all samples × filters into one tensor can easily exhaust RAM on large graphs.
    print("Loading test data (first {} samples only)...".format(args.num_frames))
    signals, coeffs, edge_index, num_nodes, nr, nc = \
        load_test_data(args.test_dir, args.field_index, max_samples=args.num_frames)
    N = signals.shape[1]
    print(f"  Loaded {N} samples, grid {nr}×{nc}={num_nodes} nodes")
    print(f"  Will render frames for samples 0..{N-1}")

    ei_dev = edge_index.to(device)

    # ── Load models ────────────────────────────────────────────────────────────
    gxn_sm = gxn_rm = gxn_mu = gxn_sd = None
    if args.gxn_checkpoint and Path(args.gxn_checkpoint).exists():
        print("Loading GXN...")
        gxn_sm, gxn_rm, gxn_mu, gxn_sd = load_gxn(args.gxn_checkpoint, device)

    vgae_m = vgae_mu = vgae_sd = None
    if args.vgae_checkpoint and Path(args.vgae_checkpoint).exists():
        print("Loading SGAE-99K...")
        vgae_m, vgae_mu, vgae_sd = load_vgae(args.vgae_checkpoint, args.keep_ratio, device)

    vgae50_m = vgae50_mu = vgae50_sd = None
    if args.vgae_50k_checkpoint and Path(args.vgae_50k_checkpoint).exists():
        print("Loading SGAE-50K...")
        vgae50_m, vgae50_mu, vgae50_sd = load_vgae(
            args.vgae_50k_checkpoint, args.keep_ratio, device)

    print("Loading GWCS...")
    nigwt_m, nigwt_cfg, nigwt_ns = load_nigwt(
        args.gwt_run_dir, args.gwt_checkpoint, device)
    if coeffs is None:
        raise ValueError("GWCS requires pre-computed coeffs in .pt files. "
                         "Check that preprocessing saved coeffs correctly.")
    print("Building Chebyshev cache...")
    from data.dataset_turbulent import GraphSignalDatasetTurbulent, create_matrix_cache_turbulent
    ds_cache = GraphSignalDatasetTurbulent(
        base_data_dir=args.test_dir, field_index=args.field_index,
        preload_to_memory=False, use_normalization=False)
    mc = create_matrix_cache_turbulent(dataset=ds_cache, device=device)

    # Classical graph objects (built once, only if needed)
    cl_graph = cl_mu = cl_sd = None
    if args.run_classical:
        print("Building classical graph objects (once)...")
        t0 = time.time()
        cl_graph = build_classical_graph(edge_index, num_nodes)
        cl_mu = signals.mean()
        cl_sd = signals.std()
        if cl_sd < 1e-8:
            cl_sd = torch.tensor(1.0)
        print(f"  Built in {time.time()-t0:.1f}s")

    # ── Per-sample rendering loop ──────────────────────────────────────────────
    print(f"\nRendering {N} frames...")
    for i in range(N):
        x = signals[:, i].cpu()
        print(f"\n[{i+1}/{N}] sample {i}")

        # GWCS  (saves gwcs_sample{i}.png  +  gwcs_sample{i}_clean.png)
        fp = os.path.join(args.output_dir, f'gwcs_sample{i}.png')
        xh = infer_nigwt(nigwt_m, nigwt_cfg, nigwt_ns,
                         coeffs[:, :, i], edge_index, mc, device, args.field_index)
        rmse = save_frame(x, xh, nr, nc, fp)
        print(f"  GWCS       RMSE={rmse:.6f}  → {fp}  +  gwcs_sample{i}_clean.png")

        # GXN  (saves gxn_sample{i}.png  +  gxn_sample{i}_clean.png)
        if gxn_sm is not None:
            fp = os.path.join(args.output_dir, f'gxn_sample{i}.png')
            xh = infer_gxn(gxn_sm, gxn_rm, gxn_mu, gxn_sd,
                           x, ei_dev, args.keep_ratio, device)
            rmse = save_frame(x, xh, nr, nc, fp)
            print(f"  GXN        RMSE={rmse:.6f}  → {fp}  +  gxn_sample{i}_clean.png")

        # SGAE-99K  (saves svgae_99k_sample{i}.png  +  svgae_99k_sample{i}_clean.png)
        if vgae_m is not None:
            fp = os.path.join(args.output_dir, f'svgae_99k_sample{i}.png')
            xh = infer_vgae(vgae_m, vgae_mu, vgae_sd, x, ei_dev, device)
            rmse = save_frame(x, xh, nr, nc, fp)
            print(f"  SGAE-99K  RMSE={rmse:.6f}  → {fp}  +  svgae_99k_sample{i}_clean.png")

        # SGAE-50K  (saves svgae_50k_sample{i}.png  +  svgae_50k_sample{i}_clean.png)
        if vgae50_m is not None:
            fp = os.path.join(args.output_dir, f'svgae_50k_sample{i}.png')
            xh = infer_vgae(vgae50_m, vgae50_mu, vgae50_sd, x, ei_dev, device)
            rmse = save_frame(x, xh, nr, nc, fp)
            print(f"  SGAE-50K  RMSE={rmse:.6f}  → {fp}  +  svgae_50k_sample{i}_clean.png")

        # Classical (inline re-run)
        if cl_graph is not None:
            G, adj, Lco, Ls = cl_graph
            x_norm = (x - cl_mu) / cl_sd
            for mname, label in [('fastgsss', 'FastGSSS'),
                                  ('bsgda',    'BSGDA'),
                                  ('rsbs',     'RSBS')]:
                fp = os.path.join(args.output_dir, f'{mname}_sample{i}.png')
                try:
                    t0 = time.time()
                    xh_n = infer_classical(mname, x_norm, G, adj, Lco, Ls,
                                          num_nodes, args.keep_ratio)
                    xh_p = xh_n.cpu() * cl_sd + cl_mu
                    rmse = save_frame(x, xh_p, nr, nc, fp)
                    print(f"  {label:10s}RMSE={rmse:.6f}  ({time.time()-t0:.1f}s)  → {fp}  +  _clean")
                except Exception as e:
                    print(f"  {label:10s}FAILED: {e}")

    print(f"\nAll frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
