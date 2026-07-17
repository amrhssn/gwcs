# -*- coding: utf-8 -*-
"""
plot_stacked_all_models.py

For each requested test sample, produces ONE figure with up to 6 rows:
  Row 0: NIGWT  (GNNNeuralIGWT)
  Row 1: Sparse-VGAE
  Row 2: GXN
  Row 3: FastGSSS   (classical)
  Row 4: BSGDA      (classical)
  Row 5: RSBS       (classical)

Each row has 3 columns:  Original Signal | Reconstruction | |Error|

All reconstructions are in PHYSICAL UNITS for fair visual comparison.
Neural models (GXN, VGAE) and NIGWT: optional via checkpoint args.
Classical baselines: optional, computed inline (graph objects built once).

Speed notes for classical methods:
- Graph Laplacian objects are built ONCE outside the sample loop.
- Use --fast flag to lower approximation orders (faster, slightly less accurate).
- Only computed for the requested sample indices (typically 8).
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

def load_gxn_model(gxn_checkpoint: str, device):
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
    """Reconstruct using GXN. Returns physical-unit reconstruction."""
    num_nodes = x_orig.shape[0]
    M = max(1, int(keep_ratio * num_nodes))

    mean_t = torch.tensor(norm_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32, device=device)

    x_norm = (x_orig.to(device) - mean_t) / std_t
    x_in   = x_norm.unsqueeze(1)

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

    x_hat = (x_hat_norm.squeeze() * std_t + mean_t).cpu()
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
    """Reconstruct using Sparse-VGAE. Returns physical-unit reconstruction."""
    mean_t = torch.tensor(norm_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32, device=device)

    x_norm = (x_orig.to(device) - mean_t) / std_t
    x_in   = x_norm.unsqueeze(1)

    with torch.no_grad():
        z_sparse  = model.encode(x_in, edge_index)
        x_hat_norm = model.decode(z_sparse, edge_index)

    x_hat = (x_hat_norm.squeeze() * std_t + mean_t).cpu()
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
        num_filters = 5

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
    """Reconstruct using NIGWT. Returns physical-unit reconstruction."""
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

    # Denormalize: model outputs normalized space when use_normalization=True
    use_normalization = cfg.get('use_normalization', False)
    if use_normalization and norm_stats is not None:
        field_names = list(norm_stats.keys())
        fname = field_names[field_index]
        gwt_m = norm_stats[fname]['mean']
        gwt_s = norm_stats[fname]['std']
        x_hat_cpu = x_hat_cpu * gwt_s + gwt_m

    return x_hat_cpu


# =============================================================================
# Classical Baselines (inline, graph objects built once)
# =============================================================================

def build_classical_graph_objects(edge_index, num_nodes, dtype=torch.float32):
    """Build thgsp graph objects ONCE. Reused for all samples and methods."""
    from thgsp.graphs.core import SparseTensor, GraphBase

    row, col = edge_index
    value = torch.ones(row.numel(), dtype=dtype)
    adj = SparseTensor(row=row, col=col, value=value,
                       sparse_sizes=(num_nodes, num_nodes))
    G = GraphBase(adj, dtype=dtype, device=torch.device('cpu'))

    deg = torch.zeros(num_nodes, dtype=dtype)
    deg.scatter_add_(0, row, torch.ones(row.numel(), dtype=dtype))
    L_row = torch.cat([row, torch.arange(num_nodes)])
    L_col = torch.cat([col, torch.arange(num_nodes)])
    L_val = torch.cat([-torch.ones(row.numel(), dtype=dtype), deg])
    L_comb = SparseTensor(row=L_row, col=L_col, value=L_val,
                          sparse_sizes=(num_nodes, num_nodes))

    inv_sqrt_deg = deg.pow(-0.5)
    inv_sqrt_deg[deg == 0] = 0.0
    L_sym_val = -inv_sqrt_deg[L_row] * L_val.clone() * inv_sqrt_deg[L_col]
    diag_mask = L_row == L_col
    L_sym_val[diag_mask] = 1.0
    L_sym = SparseTensor(row=L_row, col=L_col, value=L_sym_val,
                         sparse_sizes=(num_nodes, num_nodes))

    return G, adj, L_comb, L_sym


def classical_reconstruct(method, x_norm, G, adj, L_comb, L_sym,
                          num_nodes, keep_ratio, fast=False):
    """
    Run one classical method on one normalized signal.
    Returns normalized reconstruction (caller denormalizes).
    fast=True reduces approximation orders for speed.
    """
    from baselines.graph_signal_sampling import sampling_and_reconstruction

    sample_size = max(1, int(keep_ratio * num_nodes))
    x_2d = x_norm.unsqueeze(-1)  # (N, 1)

    args_dict = {
        'num_nodes': num_nodes,
        'sample_size': sample_size,
        'thgsp_graph': G,
        'thgsp_adj': adj,
        'thgsp_L_comb': L_comb,
        'thgsp_L_sym': L_sym,
        'signal_bandwidth': sample_size,
    }

    if fast:
        search_params = {
            'fastgsss_bandwidth': 'exact',
            'fastgsss_nu': 75.0,
            'fastgsss_order': 6,
            'fastgsss_recon_order': 2,
            'bsgda_mu': 0.01,
            'bsgda_epsilon': 1e-4,
            'bsgda_p_hops': 6,
            'bsgda_reg_order': 1,
            'rsbs_bandwidth_mode': 'exact',
            'rsbs_num_estimation': 1,
            'rsbs_epsilon': 1e-2,
            'rsbs_order': 12,
            'rsbs_mu': 0.01,
            'rsbs_reg_order': 1,
            'rsbs_lap_type': 'comb',
        }
    else:
        search_params = {
            'fastgsss_bandwidth': 'exact',
            'fastgsss_nu': 75.0,
            'fastgsss_order': 12,
            'fastgsss_recon_order': 3,
            'bsgda_mu': 0.01,
            'bsgda_epsilon': 1e-5,
            'bsgda_p_hops': 12,
            'bsgda_reg_order': 1,
            'rsbs_bandwidth_mode': 'exact',
            'rsbs_num_estimation': 1,
            'rsbs_epsilon': 1e-2,
            'rsbs_order': 30,
            'rsbs_mu': 0.01,
            'rsbs_reg_order': 1,
            'rsbs_lap_type': 'comb',
        }

    S, x_hat_norm = sampling_and_reconstruction(
        method_name=method,
        x=x_2d,
        args=args_dict,
        search_params=search_params,
    )
    return x_hat_norm.squeeze()


# =============================================================================
# Stacked Figure Plotting (variable number of rows)
# =============================================================================

def plot_stacked_sample(
    sample_idx: int,
    x_orig: torch.Tensor,
    row_recons: dict,
    num_rows: int,
    num_cols: int,
    output_dir: str,
):
    """
    Plot one figure with N_MODEL_ROWS rows x 3 cols.
    row_recons: OrderedDict of {label: x_recon_tensor} in display order.
    """
    ROW_LABELS  = list(row_recons.keys())
    COL_HEADERS = ['Original Signal', 'Reconstruction', 'Reconstruction Error']
    N_ROWS = len(ROW_LABELS)
    N_COLS = 3

    FONT_TITLE   = 22
    FONT_ROWLBL  = 20
    FONT_AXLBL   = 18
    FONT_TICK    = 14

    row_data = [(x_orig, xr) for xr in row_recons.values()]

    # Shared signal vmin/vmax across all panels
    all_vals = [x_orig] + list(row_recons.values())
    all_signal_vals = torch.cat(all_vals)
    sig_vmin = all_signal_vals.min().item()
    sig_vmax = all_signal_vals.max().item()

    # Per-row error vmax
    err_vmaxes = []
    for (xo, xr) in row_data:
        err_vmaxes.append(torch.abs(xo - xr).max().item())

    # Figure geometry
    cell_h = 3.2
    cell_w = cell_h * (num_cols / num_rows)

    left_margin   = 1.6
    right_margin  = 0.25
    top_margin    = 0.9
    bottom_margin = 0.7
    hspace_in     = 0.35
    wspace_in     = 0.55

    total_w = left_margin + N_COLS * cell_w + (N_COLS - 1) * wspace_in + right_margin
    total_h = top_margin  + N_ROWS * cell_h + (N_ROWS - 1) * hspace_in + bottom_margin

    fig = plt.figure(figsize=(total_w, total_h))

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

    # Draw images
    for r, (xo, xr) in enumerate(row_data):
        xo_np  = xo.cpu().numpy().reshape(num_rows, num_cols)
        xr_np  = xr.cpu().numpy().reshape(num_rows, num_cols)
        err_np = np.abs(xo_np - xr_np)

        im0 = axes[r][0].imshow(xo_np, aspect='auto', cmap='RdBu_r',
                                 vmin=sig_vmin, vmax=sig_vmax)
        plt.colorbar(im0, ax=axes[r][0], fraction=0.046, pad=0.04)

        im1 = axes[r][1].imshow(xr_np, aspect='auto', cmap='RdBu_r',
                                 vmin=sig_vmin, vmax=sig_vmax)
        plt.colorbar(im1, ax=axes[r][1], fraction=0.046, pad=0.04)

        im2 = axes[r][2].imshow(err_np, aspect='auto', cmap='Reds',
                                 vmin=0, vmax=err_vmaxes[r])
        plt.colorbar(im2, ax=axes[r][2], fraction=0.046, pad=0.04)

    # Tick labels: only bottom row gets x-ticks, only left col gets y-ticks
    for r in range(N_ROWS):
        for c in range(N_COLS):
            ax = axes[r][c]
            ax.tick_params(labelsize=FONT_TICK)
            if r < N_ROWS - 1:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

    # Column headers
    for c, header in enumerate(COL_HEADERS):
        axes[0][c].set_title(header, fontsize=FONT_TITLE, fontweight='bold', pad=10)

    # Row labels
    for r, label in enumerate(ROW_LABELS):
        axes[r][0].text(
            -0.22, 0.5, label,
            transform=axes[r][0].transAxes,
            fontsize=FONT_ROWLBL, fontweight='bold',
            va='center', ha='right', rotation=90,
        )

    # Shared axis labels
    fig.text(left_f * 0.28, (top_f + bottom_f) / 2, 'y',
             ha='center', va='center', fontsize=FONT_AXLBL, fontweight='bold', rotation=90)
    fig.text((left_f + right_f) / 2, bottom_f * 0.30, 'x',
             ha='center', va='center', fontsize=FONT_AXLBL, fontweight='bold')

    # Save
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
        description="Stacked comparison plot: NIGWT + SVGAE + GXN + classical baselines (all physical units)")

    # Required
    parser.add_argument('--test-dir',        required=True)
    parser.add_argument('--gwt-checkpoint',  required=True)
    parser.add_argument('--gwt-run-dir',     required=True)

    # Optional neural models
    parser.add_argument('--vgae-checkpoint', default=None)
    parser.add_argument('--gxn-checkpoint',  default=None)

    # Classical baselines
    parser.add_argument('--run-classical', action='store_true',
                        help="Include classical baselines (FastGSSS, BSGDA, RSBS)")
    parser.add_argument('--fast', action='store_true',
                        help="Use lower approximation orders for classical methods (faster)")

    # Config
    parser.add_argument('--field-index',     type=int,   default=0)
    parser.add_argument('--keep-ratio',      type=float, default=0.01)
    parser.add_argument('--output-dir',      default='baseline_results/stacked_all')
    parser.add_argument('--sample-indices',  type=int, nargs='+', default=list(range(8)))

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load test data ---
    print("Loading test data...")
    signals, coeffs, edge_index, num_nodes, num_rows, num_cols = \
        load_test_signals(args.test_dir, args.field_index)
    print(f"Signals: {signals.shape}, Grid: {num_rows}x{num_cols}={num_nodes} nodes")

    edge_index_dev = edge_index.to(device)

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

    # --- Load VGAE (optional) ---
    vgae_model, vgae_mean, vgae_std = None, None, None
    if args.vgae_checkpoint and Path(args.vgae_checkpoint).exists():
        print("Loading Sparse-VGAE model...")
        vgae_model, vgae_mean, vgae_std = load_vgae_model(
            args.vgae_checkpoint, args.keep_ratio, device)

    # --- Load GXN (optional) ---
    gxn_sampling, gxn_recovery, gxn_mean, gxn_std = None, None, None, None
    if args.gxn_checkpoint and Path(args.gxn_checkpoint).exists():
        print("Loading GXN model...")
        gxn_sampling, gxn_recovery, gxn_mean, gxn_std = load_gxn_model(
            args.gxn_checkpoint, device)

    # --- Build classical graph objects ONCE (optional) ---
    classical_objects = None
    classical_norm_mean = None
    classical_norm_std = None
    if args.run_classical:
        print("Building classical graph objects (done once)...")
        t0 = time.time()
        G, adj, L_comb, L_sym = build_classical_graph_objects(
            edge_index, num_nodes)
        classical_objects = (G, adj, L_comb, L_sym)
        # Classical methods normalize with global mean/std of test signals
        classical_norm_mean = signals.mean()
        classical_norm_std  = signals.std()
        if classical_norm_std < 1e-8:
            classical_norm_std = torch.tensor(1.0)
        print(f"  Graph objects built in {time.time() - t0:.1f}s")
        print(f"  Classical normalization: mean={classical_norm_mean:.4f}, std={classical_norm_std:.4f}")
        if args.fast:
            print("  Using FAST mode (reduced approximation orders)")

    # --- Per-sample inference + plot ---
    print(f"\nGenerating stacked plots for samples: {args.sample_indices}")
    for sample_idx in args.sample_indices:
        if sample_idx >= signals.shape[1]:
            print(f"  Skipping sample {sample_idx}: out of range")
            continue
        print(f"\n--- Sample {sample_idx} ---")

        x_orig = signals[:, sample_idx].cpu()

        # Ordered dict of reconstructions (display order = insertion order)
        recons = {}

        # NIGWT
        coeffs_sample = coeffs[:, :, sample_idx]
        x_nigwt = gwt_reconstruct(
            gwt_model, gwt_cfg, gwt_norm_stats,
            coeffs_sample, edge_index, matrix_cache, device,
            field_index=args.field_index,
        )
        recons['NIGWT'] = x_nigwt
        print(f"  NIGWT  RMSE: {torch.sqrt(torch.mean((x_nigwt - x_orig)**2)).item():.6f}")

        # Sparse-VGAE
        if vgae_model is not None:
            x_vgae = vgae_reconstruct(
                vgae_model, vgae_mean, vgae_std,
                x_orig, edge_index_dev, device)
            recons['Sparse-VGAE'] = x_vgae
            print(f"  VGAE   RMSE: {torch.sqrt(torch.mean((x_vgae - x_orig)**2)).item():.6f}")

        # GXN
        if gxn_sampling is not None:
            x_gxn = gxn_reconstruct(
                gxn_sampling, gxn_recovery, gxn_mean, gxn_std,
                x_orig, edge_index_dev, args.keep_ratio, device)
            recons['GXN'] = x_gxn
            print(f"  GXN    RMSE: {torch.sqrt(torch.mean((x_gxn - x_orig)**2)).item():.6f}")

        # Classical baselines
        if classical_objects is not None:
            G, adj, L_comb, L_sym = classical_objects
            x_norm = (x_orig - classical_norm_mean) / classical_norm_std

            for method_name, label in [('fastgsss', 'FastGSSS'),
                                        ('bsgda', 'BSGDA'),
                                        ('rsbs', 'RSBS')]:
                try:
                    t0 = time.time()
                    x_hat_norm = classical_reconstruct(
                        method_name, x_norm, G, adj, L_comb, L_sym,
                        num_nodes, args.keep_ratio, fast=args.fast)
                    # Denormalize to physical units
                    x_hat_phys = x_hat_norm.cpu() * classical_norm_std + classical_norm_mean
                    recons[label] = x_hat_phys
                    elapsed = time.time() - t0
                    rmse = torch.sqrt(torch.mean((x_hat_phys - x_orig)**2)).item()
                    print(f"  {label:8s} RMSE: {rmse:.6f}  ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  {label:8s} FAILED: {e}")

        # Plot
        if recons:
            plot_stacked_sample(
                sample_idx=sample_idx,
                x_orig=x_orig,
                row_recons=recons,
                num_rows=num_rows,
                num_cols=num_cols,
                output_dir=args.output_dir,
            )

    print(f"\nAll stacked plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
