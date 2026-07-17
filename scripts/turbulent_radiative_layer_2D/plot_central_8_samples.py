# -*- coding: utf-8 -*-
"""
plot_central_8_samples.py

For each of 8 centrally-chosen test samples, produces ONE figure with
up to 6 rows (NIGWT / Sparse-VGAE / GXN / FastGSSS / BSGDA / RSBS)
x 3 columns (Original | Reconstruction | Error).

"Central" = samples near the middle of the test set, avoiding edge
samples that may be atypical.

All reconstructions are denormalized to PHYSICAL UNITS.
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
import matplotlib.gridspec as gridspec
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Loading
# =============================================================================

def load_test_signals(test_dir: str, field_index: int = 0):
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

    signals = torch.stack(signals, dim=1)
    coeffs = torch.stack(coeffs_list, dim=2) if coeffs_list else None
    return signals, coeffs, edge_index, num_nodes, num_rows, num_cols


def pick_central_indices(num_samples: int, count: int = 8):
    """Pick `count` evenly spaced indices centered around the middle."""
    if num_samples <= count:
        return list(range(num_samples))
    mid = num_samples // 2
    half = count // 2
    start = mid - half
    return list(range(start, start + count))


# =============================================================================
# GXN
# =============================================================================

def load_gxn_model(ckpt_path: str, device):
    from baselines.gxn.neural_sampling import NeuralSampling, NeuralRecovery
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sm = NeuralSampling(in_channels=1,
                        hidden_channels=ckpt.get('hidden_dim', 128),
                        max_hops=ckpt.get('max_hops', 2)).to(device)
    rm = NeuralRecovery(in_channels=1,
                        K=ckpt.get('recovery_K', 8),
                        L=ckpt.get('recovery_L', 5)).to(device)
    sm.load_state_dict(ckpt['sampling_model']); sm.eval()
    rm.load_state_dict(ckpt['recovery_model']); rm.eval()
    return sm, rm, ckpt.get('norm_mean', 0.0), ckpt.get('norm_std', 1.0)


def gxn_reconstruct(sm, rm, norm_mean, norm_std, x_orig, edge_index,
                    keep_ratio, device):
    N = x_orig.shape[0]
    M = max(1, int(keep_ratio * N))
    mean_t = torch.tensor(norm_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32, device=device)
    x_in = ((x_orig.to(device) - mean_t) / std_t).unsqueeze(1)
    with torch.no_grad():
        att, _, _ = sm(x_in, edge_index)
        sc = att.squeeze()
        if sc.dim() == 0: sc = sc.unsqueeze(0)
        _, ti = torch.topk(sc, k=M)
        mp = torch.zeros_like(x_in); mp[ti] = 1.0
        y = mp * att * x_in
        xh = rm(y, edge_index, att, mp)
    return (xh.squeeze() * std_t + mean_t).cpu()


# =============================================================================
# Sparse-VGAE
# =============================================================================

def load_vgae_model(ckpt_path: str, keep_ratio: float, device):
    from baselines.sparse_VGAE.gae import GAE
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GAE(input_dim=1, latent_dim=ckpt.get('latent_dim', 128),
                output_dim=1,
                num_encoder_layers=ckpt.get('hidden_layers', 4),
                num_decoder_layers=ckpt.get('hidden_layers', 4),
                conv_layer_type='SAGEConv', activation_function='silu',
                sparsity_ratio=keep_ratio, variational=False,
                sparsity_mode='ste_annealing',
                temperature_init=1.0, min_temperature=0.01,
                annealing_steps=1000).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt.get('norm_mean', 0.0), ckpt.get('norm_std', 1.0)


def vgae_reconstruct(model, norm_mean, norm_std, x_orig, edge_index, device):
    mean_t = torch.tensor(norm_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32, device=device)
    x_in = ((x_orig.to(device) - mean_t) / std_t).unsqueeze(1)
    with torch.no_grad():
        z = model.encode(x_in, edge_index)
        xh = model.decode(z, edge_index)
    return (xh.squeeze() * std_t + mean_t).cpu()


# =============================================================================
# NIGWT
# =============================================================================

def load_gwt_model(run_dir: str, ckpt_path: str, device):
    run_dir = Path(run_dir)
    with open(str(run_dir / 'reconstructor_pretraining_config_turbulent.json'), 'r') as f:
        cfg = json.load(f)

    norm_stats = None
    nmp = run_dir / 'normalization_metadata.json'
    if nmp.exists():
        with open(str(nmp), 'r') as f:
            nm = json.load(f)
        nsp = nm.get('normalization_stats_path')
        if nsp and Path(nsp).exists():
            norm_stats = torch.load(nsp, map_location='cpu', weights_only=False)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    td = Path(cfg['test_data_dir'])
    gcp = td / 'gwt_config.json'
    nf = 5
    if gcp.exists():
        with open(str(gcp), 'r') as f:
            nf = json.load(f)['num_filters']

    un = cfg.get('use_normalization', False)
    mn = cfg.get('model_name', 'GNNNeuralIGWT')

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT
    if mn == 'GUNetNeuralIGWT':
        model = GUNetNeuralIGWT(
            num_filters=nf, hidden_dim=cfg['hidden_dim'],
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
            use_normalization=un, normalization_stats=norm_stats,
        ).to(device)
    else:
        model = GNNNeuralIGWT(
            num_filters=nf, hidden_dim=cfg['hidden_dim'],
            output_dim=cfg['output_dim'],
            activation_function=cfg['activation_function'],
            encoder_type=cfg['encoder_type'],
            decoder_type=cfg['decoder_type'],
            conv_layer_type=cfg['conv_layer_type'],
            processor_num_layers=cfg['processor_num_layers'],
            predict_residual=cfg['predict_residual'],
            enable_scale_embeddings=cfg['enable_scale_embeddings'],
            igwt_mode=cfg['igwt_mode'],
            use_normalization=un, normalization_stats=norm_stats,
        ).to(device)

    model.load_state_dict(sd); model.eval()
    return model, cfg, norm_stats


def gwt_reconstruct(model, cfg, norm_stats, coeffs_sample, edge_index,
                    matrix_cache, device, field_index=0):
    from models.sparsification_utils import multilevel_importance_sampling
    c = coeffs_sample.to(device)
    kr = cfg.get('keep_ratio', 0.01)
    mask = multilevel_importance_sampling(coeffs=c, keep_ratio=kr)
    ei = edge_index.to(device)
    bv = torch.zeros(c.shape[0], dtype=torch.long, device=device)
    ptr = torch.tensor([0, c.shape[0]], device=device)
    cd = matrix_cache.get([0])

    with torch.no_grad():
        xh = model(coeffs=c, sparsification_mask=mask, edge_index=ei,
                    batch=bv, chebyshev_data=cd, batch_ptr=ptr)

    xh_cpu = xh.squeeze().cpu()

    # Denormalize: model outputs normalized space when use_normalization=True
    un = cfg.get('use_normalization', False)
    if un and norm_stats is not None:
        fn = list(norm_stats.keys())
        fname = fn[field_index]
        xh_cpu = xh_cpu * norm_stats[fname]['std'] + norm_stats[fname]['mean']

    return xh_cpu


# =============================================================================
# Classical (inline, graph built once)
# =============================================================================

def build_classical_graph(edge_index, num_nodes, dtype=torch.float32):
    from thgsp.graphs.core import SparseTensor, GraphBase
    row, col = edge_index
    val = torch.ones(row.numel(), dtype=dtype)
    adj = SparseTensor(row=row, col=col, value=val,
                       sparse_sizes=(num_nodes, num_nodes))
    G = GraphBase(adj, dtype=dtype, device=torch.device('cpu'))

    deg = torch.zeros(num_nodes, dtype=dtype)
    deg.scatter_add_(0, row, torch.ones(row.numel(), dtype=dtype))
    Lr = torch.cat([row, torch.arange(num_nodes)])
    Lc = torch.cat([col, torch.arange(num_nodes)])
    Lv = torch.cat([-torch.ones(row.numel(), dtype=dtype), deg])
    Lcomb = SparseTensor(row=Lr, col=Lc, value=Lv,
                         sparse_sizes=(num_nodes, num_nodes))

    isd = deg.pow(-0.5); isd[deg == 0] = 0.0
    Lsv = -isd[Lr] * Lv.clone() * isd[Lc]
    dm = Lr == Lc; Lsv[dm] = 1.0
    Lsym = SparseTensor(row=Lr, col=Lc, value=Lsv,
                        sparse_sizes=(num_nodes, num_nodes))
    return G, adj, Lcomb, Lsym


def classical_reconstruct(method, x_norm, G, adj, Lc, Ls, num_nodes,
                          keep_ratio):
    from baselines.graph_signal_sampling import sampling_and_reconstruction
    ss = max(1, int(keep_ratio * num_nodes))
    args_dict = {'num_nodes': num_nodes, 'sample_size': ss,
                 'thgsp_graph': G, 'thgsp_adj': adj,
                 'thgsp_L_comb': Lc, 'thgsp_L_sym': Ls,
                 'signal_bandwidth': ss}
    # Use fast parameters to keep runtime reasonable
    sp = {'fastgsss_bandwidth': 'exact', 'fastgsss_nu': 75.0,
          'fastgsss_order': 6, 'fastgsss_recon_order': 2,
          'bsgda_mu': 0.01, 'bsgda_epsilon': 1e-4,
          'bsgda_p_hops': 6, 'bsgda_reg_order': 1,
          'rsbs_bandwidth_mode': 'exact', 'rsbs_num_estimation': 1,
          'rsbs_epsilon': 1e-2, 'rsbs_order': 12,
          'rsbs_mu': 0.01, 'rsbs_reg_order': 1, 'rsbs_lap_type': 'comb'}
    _, xh = sampling_and_reconstruction(method_name=method, x=x_norm.unsqueeze(-1),
                                        args=args_dict, search_params=sp)
    return xh.squeeze()


# =============================================================================
# Plotting
# =============================================================================

def plot_stacked(sample_idx, x_orig, recons, num_rows, num_cols, output_dir):
    """
    Stacked figure: N rows x 3 cols.
    recons: dict {label: tensor} in display order.
    """
    labels = list(recons.keys())
    NR = len(labels)
    NC = 3
    COL_HEADERS = ['Original Signal', 'Reconstruction', 'Reconstruction Error']

    FONT_T = 22; FONT_R = 20; FONT_A = 18; FONT_K = 14

    row_data = [(x_orig, xr) for xr in recons.values()]

    all_v = torch.cat([x_orig] + list(recons.values()))
    vmin, vmax = all_v.min().item(), all_v.max().item()
    err_vm = [torch.abs(xo - xr).max().item() for xo, xr in row_data]

    ch = 3.2; cw = ch * (num_cols / num_rows)
    lm, rm, tm, bm = 1.6, 0.25, 0.9, 0.7
    hs, ws = 0.35, 0.55
    tw = lm + NC * cw + (NC - 1) * ws + rm
    th = tm + NR * ch + (NR - 1) * hs + bm

    fig = plt.figure(figsize=(tw, th))
    lf = lm / tw; rf = 1 - rm / tw; tf = 1 - tm / th; bf = bm / th

    gs = gridspec.GridSpec(NR, NC, figure=fig,
                           left=lf, right=rf, top=tf, bottom=bf,
                           hspace=hs / ch, wspace=ws / cw)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(NC)] for r in range(NR)]

    for r, (xo, xr) in enumerate(row_data):
        xo_np = xo.cpu().numpy().reshape(num_rows, num_cols)
        xr_np = xr.cpu().numpy().reshape(num_rows, num_cols)
        err_np = np.abs(xo_np - xr_np)

        im0 = axes[r][0].imshow(xo_np, aspect='auto', cmap='RdBu_r',
                                 vmin=vmin, vmax=vmax)
        plt.colorbar(im0, ax=axes[r][0], fraction=0.046, pad=0.04)
        im1 = axes[r][1].imshow(xr_np, aspect='auto', cmap='RdBu_r',
                                 vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=axes[r][1], fraction=0.046, pad=0.04)
        im2 = axes[r][2].imshow(err_np, aspect='auto', cmap='Reds',
                                 vmin=0, vmax=err_vm[r])
        plt.colorbar(im2, ax=axes[r][2], fraction=0.046, pad=0.04)

    for r in range(NR):
        for c in range(NC):
            axes[r][c].tick_params(labelsize=FONT_K)
            if r < NR - 1: axes[r][c].set_xticklabels([])
            if c > 0:      axes[r][c].set_yticklabels([])

    for c, h in enumerate(COL_HEADERS):
        axes[0][c].set_title(h, fontsize=FONT_T, fontweight='bold', pad=10)

    for r, lb in enumerate(labels):
        axes[r][0].text(-0.22, 0.5, lb, transform=axes[r][0].transAxes,
                        fontsize=FONT_R, fontweight='bold',
                        va='center', ha='right', rotation=90)

    fig.text(lf * 0.28, (tf + bf) / 2, 'y', ha='center', va='center',
             fontsize=FONT_A, fontweight='bold', rotation=90)
    fig.text((lf + rf) / 2, bf * 0.30, 'x', ha='center', va='center',
             fontsize=FONT_A, fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f'central_sample{sample_idx}.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot 8 central test samples: Original | Recon | Error for all models")

    parser.add_argument('--test-dir',        required=True)
    parser.add_argument('--gwt-checkpoint',  required=True)
    parser.add_argument('--gwt-run-dir',     required=True)
    parser.add_argument('--vgae-checkpoint',     default=None,
                        help='Sparse-VGAE ~99K checkpoint (latent_dim=90)')
    parser.add_argument('--vgae-50k-checkpoint', default=None,
                        help='Sparse-VGAE ~50K checkpoint (latent_dim=64)')
    parser.add_argument('--gxn-checkpoint',      default=None)
    parser.add_argument('--run-classical',        action='store_true')
    parser.add_argument('--field-index',     type=int,   default=0)
    parser.add_argument('--keep-ratio',      type=float, default=0.01)
    parser.add_argument('--num-frames',      type=int,   default=8,
                        help="Number of central samples to plot (default 8)")
    parser.add_argument('--output-dir',      default='baseline_results/central_frames')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading test data...")
    signals, coeffs, edge_index, num_nodes, nr, nc = \
        load_test_signals(args.test_dir, args.field_index)
    ns = signals.shape[1]
    print(f"  {ns} test samples, grid {nr}x{nc}={num_nodes} nodes")

    # Pick central indices
    indices = pick_central_indices(ns, args.num_frames)
    print(f"  Central {args.num_frames} indices: {indices}")

    ei_dev = edge_index.to(device)

    # Load NIGWT
    print("Loading NIGWT...")
    gwt_m, gwt_cfg, gwt_ns = load_gwt_model(args.gwt_run_dir,
                                              args.gwt_checkpoint, device)
    if coeffs is None:
        raise ValueError("NIGWT needs pre-computed coeffs in .pt files")
    print("Building Chebyshev cache...")
    from data.dataset_turbulent import GraphSignalDatasetTurbulent, create_matrix_cache_turbulent
    ds_cache = GraphSignalDatasetTurbulent(base_data_dir=args.test_dir,
                                           field_index=args.field_index,
                                           preload_to_memory=False,
                                           use_normalization=False)
    mc = create_matrix_cache_turbulent(dataset=ds_cache, device=device)

    # Load VGAE-99K
    vgae_m, vgae_mean, vgae_std = None, None, None
    if args.vgae_checkpoint and Path(args.vgae_checkpoint).exists():
        print("Loading VGAE-99K...")
        vgae_m, vgae_mean, vgae_std = load_vgae_model(
            args.vgae_checkpoint, args.keep_ratio, device)

    # Load VGAE-50K (optional second SVGAE variant)
    vgae50_m, vgae50_mean, vgae50_std = None, None, None
    if args.vgae_50k_checkpoint and Path(args.vgae_50k_checkpoint).exists():
        print("Loading VGAE-50K...")
        vgae50_m, vgae50_mean, vgae50_std = load_vgae_model(
            args.vgae_50k_checkpoint, args.keep_ratio, device)

    # Load GXN
    gxn_sm, gxn_rm, gxn_mean, gxn_std = None, None, None, None
    if args.gxn_checkpoint and Path(args.gxn_checkpoint).exists():
        print("Loading GXN...")
        gxn_sm, gxn_rm, gxn_mean, gxn_std = load_gxn_model(
            args.gxn_checkpoint, device)

    # Build classical graph once
    cl_obj = None
    cl_mean = cl_std = None
    if args.run_classical:
        print("Building classical graph objects (once)...")
        t0 = time.time()
        cl_obj = build_classical_graph(edge_index, num_nodes)
        cl_mean = signals.mean()
        cl_std  = signals.std()
        if cl_std < 1e-8: cl_std = torch.tensor(1.0)
        print(f"  Built in {time.time()-t0:.1f}s")

    # Process each central sample
    print(f"\nGenerating frames for {len(indices)} central samples...")
    for si in indices:
        if si >= ns:
            print(f"  Skipping {si}: out of range"); continue
        print(f"\n--- Sample {si} (central) ---")
        x_orig = signals[:, si].cpu()
        recons = {}

        # NIGWT
        cs = coeffs[:, :, si]
        xn = gwt_reconstruct(gwt_m, gwt_cfg, gwt_ns, cs, edge_index,
                             mc, device, args.field_index)
        recons['NIGWT'] = xn
        rmse = torch.sqrt(torch.mean((xn - x_orig)**2)).item()
        print(f"  NIGWT    RMSE: {rmse:.6f}")

        # VGAE-99K
        if vgae_m is not None:
            xv = vgae_reconstruct(vgae_m, vgae_mean, vgae_std,
                                  x_orig, ei_dev, device)
            recons['SVGAE-99K'] = xv
            print(f"  SVGAE-99K RMSE: {torch.sqrt(torch.mean((xv-x_orig)**2)).item():.6f}")

        # VGAE-50K
        if vgae50_m is not None:
            xv50 = vgae_reconstruct(vgae50_m, vgae50_mean, vgae50_std,
                                    x_orig, ei_dev, device)
            recons['SVGAE-50K'] = xv50
            print(f"  SVGAE-50K RMSE: {torch.sqrt(torch.mean((xv50-x_orig)**2)).item():.6f}")

        # GXN
        if gxn_sm is not None:
            xg = gxn_reconstruct(gxn_sm, gxn_rm, gxn_mean, gxn_std,
                                 x_orig, ei_dev, args.keep_ratio, device)
            recons['GXN'] = xg
            print(f"  GXN      RMSE: {torch.sqrt(torch.mean((xg-x_orig)**2)).item():.6f}")

        # Classical
        if cl_obj is not None:
            G, adj, Lc, Ls = cl_obj
            xn_cl = (x_orig - cl_mean) / cl_std
            for mname, lbl in [('fastgsss','FastGSSS'),
                               ('bsgda','BSGDA'),
                               ('rsbs','RSBS')]:
                try:
                    t0 = time.time()
                    xh_n = classical_reconstruct(mname, xn_cl, G, adj, Lc, Ls,
                                                 num_nodes, args.keep_ratio)
                    xh_p = xh_n.cpu() * cl_std + cl_mean
                    recons[lbl] = xh_p
                    el = time.time() - t0
                    print(f"  {lbl:8s}  RMSE: {torch.sqrt(torch.mean((xh_p-x_orig)**2)).item():.6f}  ({el:.1f}s)")
                except Exception as e:
                    print(f"  {lbl:8s}  FAILED: {e}")

        if recons:
            plot_stacked(si, x_orig, recons, nr, nc, args.output_dir)

    print(f"\nAll frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
