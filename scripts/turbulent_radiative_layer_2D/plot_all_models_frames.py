#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_all_models_frames.py
=========================
Saves per-sample comparison frames for NIGWT | Sparse-VGAE | GXN side-by-side.

Layout (one PNG per test sample):
  Row 1:  Original signal | NIGWT recon | SVGAE recon | GXN recon
  Row 2:  (blank)         | NIGWT |err| | SVGAE |err| | GXN |err|

Colorscale fix
  The signal colormap (RdBu_r) uses GLOBAL vmin/vmax computed from the 1st/99th
  percentile of ALL test signals — so the scale is identical across every frame.

RMSE comparability
  All three models output physical-unit values (after their own denormalization):
    - NIGWT:  denormalise with norm_stats from normalization_metadata.json
    - SVGAE:  denormalise with norm_mean / norm_std stored in best_checkpoint.pt
    - GXN:    same as SVGAE
  RMSE = sqrt(mean((x_orig - x_hat)^2)) in the original signal units.

Usage:
    python plot_all_models_frames.py \\
        --test-dir          /path/to/processed/test \\
        --field-index       0 \\
        --keep-ratio        0.01 \\
        --gwt-checkpoint    /path/to/runs/.../checkpoints/42.pt \\
        --gwt-run-dir       /path/to/runs/... \\
        --svgae-checkpoint  /path/to/baseline_results/svgae/best_checkpoint.pt \\
        --svgae-latent-dim  64 --svgae-hidden-layers 4 \\
        --gxn-checkpoint    /path/to/baseline_results/gxn/best_checkpoint.pt \\
        --gxn-hidden-dim    128 --gxn-max-hops 2 \\
        --output-dir        /path/to/output_frames \\
        --num-samples       50

The script also saves:
  - rmse_comparison.csv   : per-sample RMSE for all three models
  - rmse_summary.json     : mean RMSE per model
"""

import sys
import os
import json
import csv
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Colormap / style constants
# =============================================================================

CMAP_SIG = "RdBu_r"      # physical signal
CMAP_ERR = "Reds"         # absolute error
FONT     = dict(fontsize=12, fontweight="bold")
TICK     = dict(labelsize=9)


# =============================================================================
# Data loading
# =============================================================================

def load_test_data(test_dir: str, field_index: int = 0):
    """
    Load ALL test signals + GWT coefficients from processed .pt files.

    Returns
    -------
    signals      : (N, S) tensor  — original physical-unit signals
    coeffs       : (N, F, S) tensor  — GWT coefficients (or None)
    edge_index   : (2, E) tensor
    num_rows, num_cols : int
    field_names  : list[str]
    """
    test_dir = Path(test_dir)
    shared   = torch.load(test_dir / "shared_data.pt",
                          map_location="cpu", weights_only=False)
    edge_index  = shared["edge_index"]
    num_rows    = shared["num_rows"]
    num_cols    = shared["num_columns"]
    field_names = shared.get("field_names", [f"field_{field_index}"])

    sample_files = sorted(
        [f for f in test_dir.glob("*.pt") if f.stem.isdigit()],
        key=lambda f: int(f.stem),
    )

    signals, coeffs_list = [], []
    for f in sample_files:
        s = torch.load(f, map_location="cpu", weights_only=False)
        # .float() handles both float16 (half-precision storage) and float32
        signals.append(s.x[:, field_index].float())
        if hasattr(s, "coeffs") and s.coeffs is not None:
            coeffs_list.append(s.coeffs[:, field_index, :].float())

    signals = torch.stack(signals, dim=1)                         # (N, S)
    coeffs  = (torch.stack(coeffs_list, dim=2)                   # (N, F, S)
               if coeffs_list else None)
    return signals, coeffs, edge_index, num_rows, num_cols, field_names


# =============================================================================
# Global color range (FIXED across all frames)
# =============================================================================

def compute_global_color_range(signals: torch.Tensor,
                                pct_low: float = 1.0,
                                pct_high: float = 99.0):
    """
    Compute vmin/vmax from percentiles of ALL test signals so the colormap
    scale never changes between frames.
    """
    flat = signals.flatten().numpy()
    vmin = float(np.percentile(flat, pct_low))
    vmax = float(np.percentile(flat, pct_high))
    # Symmetric around 0 for diverging colormaps (nice for velocity/density diff)
    absmax = max(abs(vmin), abs(vmax))
    # If data is clearly non-symmetric (e.g. density always positive), don't force symmetry
    if vmin < 0 and vmax > 0:
        vmin, vmax = -absmax, absmax
    return vmin, vmax


# =============================================================================
# NIGWT model loading and inference
# =============================================================================

def load_nigwt(gwt_run_dir: str, gwt_checkpoint: str, device):
    run_dir = Path(gwt_run_dir)

    config_path = run_dir / "reconstructor_pretraining_config_turbulent.json"
    if not config_path.exists():
        raise FileNotFoundError(f"NIGWT config not found: {config_path}")
    with open(str(config_path)) as f:
        cfg = json.load(f)

    # Normalization stats
    norm_stats = None
    norm_meta_path = run_dir / "normalization_metadata.json"
    if norm_meta_path.exists():
        with open(str(norm_meta_path)) as f:
            norm_meta = json.load(f)
        nsp = norm_meta.get("normalization_stats_path")
        if nsp and Path(nsp).exists():
            norm_stats = torch.load(nsp, map_location="cpu", weights_only=False)

    ckpt = torch.load(gwt_checkpoint, map_location=device, weights_only=False)
    state = {k.replace("module.", ""): v
             for k, v in ckpt.get("model_state_dict", ckpt).items()}

    # num_filters
    test_data_dir = Path(cfg.get("test_data_dir", gwt_run_dir))
    gwt_cfg_path  = test_data_dir / "gwt_config.json"
    num_filters   = 5
    if gwt_cfg_path.exists():
        with open(str(gwt_cfg_path)) as f:
            num_filters = json.load(f)["num_filters"]

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT
    use_norm   = cfg.get("use_normalization", False)
    model_name = cfg.get("model_name", "GNNNeuralIGWT")

    if model_name == "GUNetNeuralIGWT":
        model = GUNetNeuralIGWT(
            num_filters=num_filters,
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["output_dim"],
            activation_function=cfg["activation_function"],
            encoder_type=cfg["encoder_type"],
            fusion_type=cfg.get("fusion_type", "mlp"),
            decoder_type=cfg["decoder_type"],
            conv_layer_type=cfg["conv_layer_type"],
            processor_num_layers=cfg["processor_num_layers"],
            pooling_type=cfg.get("pooling_type", "top_k_pooling"),
            pooling_ratio=cfg.get("pooling_ratio", 0.5),
            predict_residual=cfg["predict_residual"],
            enable_scale_embeddings=cfg["enable_scale_embeddings"],
            igwt_mode=cfg["igwt_mode"],
            use_normalization=use_norm,
            normalization_stats=norm_stats,
        ).to(device)
    else:
        model = GNNNeuralIGWT(
            num_filters=num_filters,
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["output_dim"],
            activation_function=cfg["activation_function"],
            encoder_type=cfg["encoder_type"],
            decoder_type=cfg["decoder_type"],
            conv_layer_type=cfg["conv_layer_type"],
            processor_num_layers=cfg["processor_num_layers"],
            predict_residual=cfg["predict_residual"],
            enable_scale_embeddings=cfg["enable_scale_embeddings"],
            igwt_mode=cfg["igwt_mode"],
            use_normalization=use_norm,
            normalization_stats=norm_stats,
        ).to(device)

    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  NIGWT loaded: {model_name}  {n:,} params  norm={use_norm}")
    return model, cfg, norm_stats


@torch.no_grad()
def nigwt_reconstruct(model, cfg, norm_stats,
                      coeffs_sample, edge_index, matrix_cache, device,
                      field_index: int = 0):
    """Run NIGWT forward pass; return denormalised (physical-unit) prediction."""
    from models.sparsification_utils import multilevel_importance_sampling
    keep_ratio  = cfg.get("keep_ratio", 0.01)
    coeffs_dev  = coeffs_sample.to(device)
    mask        = multilevel_importance_sampling(coeffs=coeffs_dev, keep_ratio=keep_ratio)
    ei          = edge_index.to(device)
    batch       = torch.zeros(coeffs_dev.shape[0], dtype=torch.long, device=device)
    ptr         = torch.tensor([0, coeffs_dev.shape[0]], device=device)
    cheby_data  = matrix_cache.get([0])

    x_hat = model(
        coeffs=coeffs_dev, sparsification_mask=mask,
        edge_index=ei, batch=batch,
        chebyshev_data=cheby_data, batch_ptr=ptr,
    ).squeeze().cpu()

    use_norm = cfg.get("use_normalization", False)
    if use_norm and norm_stats is not None:
        field_names = list(norm_stats.keys())
        fname       = field_names[field_index]          # dynamic — no hardcoding
        x_hat       = x_hat * norm_stats[fname]["std"] + norm_stats[fname]["mean"]
    return x_hat


# =============================================================================
# Sparse-VGAE loading and inference
# =============================================================================

def load_svgae(checkpoint_path: str, latent_dim: int, hidden_layers: int, device):
    from baselines.sparse_VGAE.gae import GAE

    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_mean  = ckpt.get("norm_mean", 0.0)
    norm_std   = ckpt.get("norm_std",  1.0)
    ld         = ckpt.get("latent_dim",    latent_dim)
    hl         = ckpt.get("hidden_layers", hidden_layers)

    # We don't know num_nodes here — the model is lazy-init on first forward pass
    # so we load with the stored architecture args and let it build from state_dict
    model = GAE(
        input_dim=1, latent_dim=ld, output_dim=1,
        num_encoder_layers=hl, num_decoder_layers=hl,
        conv_layer_type='SAGEConv', activation_function='silu',
        sparsity_ratio=0.01,
        variational=False,
        sparsity_mode='ste_annealing',
        temperature_init=1.0, min_temperature=0.01,
        annealing_steps=250,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SVGAE loaded: latent_dim={ld}, hidden_layers={hl}  {n:,} params")
    return model, float(norm_mean), float(norm_std)


@torch.no_grad()
def svgae_reconstruct(model, norm_mean, norm_std, signal_raw, edge_index, device):
    """
    Normalize → encode → decode → denormalize.
    signal_raw: (N,) tensor in physical units.
    Returns:    (N,) tensor in physical units.
    """
    x_norm = ((signal_raw - norm_mean) / max(norm_std, 1e-8)).unsqueeze(1).to(device)
    ei     = edge_index.to(device)
    z      = model.encode(x_norm, ei)
    x_hat  = model.decode(z, ei).squeeze().cpu()
    return x_hat * norm_std + norm_mean


# =============================================================================
# GXN loading and inference
# =============================================================================

def load_gxn(checkpoint_path: str, hidden_dim: int, max_hops: int, device):
    from baselines.gxn.neural_sampling import NeuralSampling, NeuralRecovery

    ckpt      = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_mean = ckpt.get("norm_mean", 0.0)
    norm_std  = ckpt.get("norm_std",  1.0)
    hd        = ckpt.get("hidden_dim",  hidden_dim)
    mh        = ckpt.get("max_hops",    max_hops)
    rK        = ckpt.get("recovery_K",  8)
    rL        = ckpt.get("recovery_L",  5)

    # We need num_nodes to build the model; infer from sampling state dict
    # NeuralSampling first layer width == 1 input feature, so we just init
    sampling_model = NeuralSampling(hidden_dim=hd, max_hops=mh).to(device)
    recovery_model = NeuralRecovery(in_channels=1, hidden_dim=hd,
                                    K=rK, L=rL).to(device)
    sampling_model.load_state_dict(ckpt["sampling_model"])
    recovery_model.load_state_dict(ckpt["recovery_model"])
    sampling_model.eval()
    recovery_model.eval()

    ns = sum(p.numel() for p in sampling_model.parameters() if p.requires_grad)
    nr = sum(p.numel() for p in recovery_model.parameters() if p.requires_grad)
    print(f"  GXN loaded: hidden_dim={hd}, max_hops={mh}  "
          f"Sampling={ns:,} + Recovery={nr:,} = {ns+nr:,} params")
    # Find M from keep_ratio if stored, else from checkpoint; default 1%
    keep_ratio = ckpt.get("keep_ratio", 0.01)
    return sampling_model, recovery_model, float(norm_mean), float(norm_std), keep_ratio


@torch.no_grad()
def gxn_reconstruct(sampling_model, recovery_model,
                    norm_mean, norm_std, keep_ratio,
                    signal_raw, edge_index, num_nodes, device):
    """
    Normalize → NeuralSampling → top-K mask → NeuralRecovery → denormalize.
    signal_raw: (N,) tensor.
    Returns:    (N,) tensor in physical units.
    """
    x_norm = ((signal_raw - norm_mean) / max(norm_std, 1e-8)).unsqueeze(1).to(device)
    ei     = edge_index.to(device)

    M = max(1, int(keep_ratio * num_nodes))
    attention, _, _ = sampling_model(x_norm, ei)
    _, top_idx = torch.topk(attention.squeeze(), k=M)
    mask = torch.zeros_like(x_norm)
    mask[top_idx] = 1.0
    y     = mask * attention * x_norm
    x_hat = recovery_model(y, ei, attention, mask).squeeze().cpu()
    return x_hat * norm_std + norm_mean


# =============================================================================
# Frame rendering
# =============================================================================

def render_frame(
        x_orig, x_nigwt, x_svgae, x_gxn,
        rows, cols,
        global_vmin, global_vmax,
        sample_index: int,
        field_name: str,
        out_dir: Path,
        frame_index: int,
):
    """
    Save one PNG frame with 2 rows × 4 columns:
      Row 1: Original | NIGWT | SVGAE | GXN       (fixed signal colorscale)
      Row 2:   —      | NIGWT err | SVGAE err | GXN err  (per-frame max error scale)

    Errors are plotted with a SHARED error scale across models (max of all three).
    """
    x_orig  = x_orig.squeeze().cpu().float()
    x_nigwt = x_nigwt.squeeze().cpu().float() if x_nigwt is not None else None
    x_svgae = x_svgae.squeeze().cpu().float() if x_svgae is not None else None
    x_gxn   = x_gxn.squeeze().cpu().float()   if x_gxn   is not None else None

    def rmse(a, b):
        if a is None or b is None:
            return float('nan')
        return float(((a - b) ** 2).mean() ** 0.5)

    rmse_nigwt = rmse(x_orig, x_nigwt)
    rmse_svgae = rmse(x_orig, x_svgae)
    rmse_gxn   = rmse(x_orig, x_gxn)

    # error images
    err_nigwt = (x_orig - x_nigwt).abs() if x_nigwt is not None else torch.zeros_like(x_orig)
    err_svgae = (x_orig - x_svgae).abs() if x_svgae is not None else torch.zeros_like(x_orig)
    err_gxn   = (x_orig - x_gxn).abs()   if x_gxn   is not None else torch.zeros_like(x_orig)

    # Shared error scale (max across all three models for this sample)
    emax = max(
        float(err_nigwt.reshape(rows, cols).max()),
        float(err_svgae.reshape(rows, cols).max()),
        float(err_gxn.reshape(rows, cols).max()),
        1e-10,
    )

    fig = plt.figure(figsize=(22, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.35, wspace=0.25,
                            left=0.04, right=0.96,
                            top=0.88,  bottom=0.06)

    title = (f"Sample {sample_index}  |  field={field_name}  |  "
             f"NIGWT RMSE={rmse_nigwt:.4f}   "
             f"SVGAE RMSE={rmse_svgae:.4f}   "
             f"GXN RMSE={rmse_gxn:.4f}")
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def _plot(ax, data, title_str, cmap, vmin, vmax, show_cbar=True):
        im = ax.imshow(data.reshape(rows, cols),
                       aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title_str, fontsize=11, fontweight="bold")
        ax.set_xlabel("x (cols)", fontsize=9)
        ax.set_ylabel("y (rows)", fontsize=9)
        ax.tick_params(labelsize=8)
        if show_cbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    # Row 0: signals  — GLOBAL fixed colorscale
    _plot(fig.add_subplot(gs[0, 0]), x_orig,  "Original Signal",
          CMAP_SIG, global_vmin, global_vmax)
    if x_nigwt is not None:
        _plot(fig.add_subplot(gs[0, 1]), x_nigwt, "NIGWT Reconstruction",
              CMAP_SIG, global_vmin, global_vmax)
    if x_svgae is not None:
        _plot(fig.add_subplot(gs[0, 2]), x_svgae, "SVGAE Reconstruction",
              CMAP_SIG, global_vmin, global_vmax)
    if x_gxn is not None:
        _plot(fig.add_subplot(gs[0, 3]), x_gxn, "GXN Reconstruction",
              CMAP_SIG, global_vmin, global_vmax)

    # Row 1: error maps  — shared error scale for fair comparison
    fig.add_subplot(gs[1, 0]).axis('off')   # no error for original
    _plot(fig.add_subplot(gs[1, 1]), err_nigwt,
          f"|Error| NIGWT  (RMSE={rmse_nigwt:.4f})",
          CMAP_ERR, 0.0, emax)
    _plot(fig.add_subplot(gs[1, 2]), err_svgae,
          f"|Error| SVGAE  (RMSE={rmse_svgae:.4f})",
          CMAP_ERR, 0.0, emax)
    _plot(fig.add_subplot(gs[1, 3]), err_gxn,
          f"|Error| GXN  (RMSE={rmse_gxn:.4f})",
          CMAP_ERR, 0.0, emax)

    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"frame_{frame_index:04d}.png"
    plt.savefig(str(fpath), dpi=130, bbox_inches="tight")
    plt.close(fig)

    return fpath, rmse_nigwt, rmse_svgae, rmse_gxn


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare NIGWT | SVGAE | GXN per-sample with fixed colorscale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--test-dir",        required=True,
                        help="Processed test split directory")
    parser.add_argument("--field-index",     type=int, default=0)
    parser.add_argument("--keep-ratio",      type=float, default=0.01)
    parser.add_argument("--num-samples",     type=int, default=None,
                        help="How many test samples to visualise (default: all)")
    parser.add_argument("--color-pct-low",   type=float, default=1.0,
                        help="Lower percentile for fixed colorscale")
    parser.add_argument("--color-pct-high",  type=float, default=99.0,
                        help="Upper percentile for fixed colorscale")

    # NIGWT
    parser.add_argument("--gwt-checkpoint",  default=None,
                        help="Path to NIGWT best checkpoint .pt")
    parser.add_argument("--gwt-run-dir",     default=None,
                        help="NIGWT run directory (contains config JSON)")

    # SVGAE
    parser.add_argument("--svgae-checkpoint",   default=None,
                        help="Path to Sparse-VGAE best_checkpoint.pt")
    parser.add_argument("--svgae-latent-dim",   type=int, default=64)
    parser.add_argument("--svgae-hidden-layers", type=int, default=4)

    # GXN
    parser.add_argument("--gxn-checkpoint",   default=None,
                        help="Path to GXN best_checkpoint.pt")
    parser.add_argument("--gxn-hidden-dim",   type=int, default=128)
    parser.add_argument("--gxn-max-hops",     type=int, default=2)

    parser.add_argument("--output-dir",   default="all_models_frames")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load test data ----
    print("\nLoading test data...")
    signals, coeffs, edge_index, num_rows, num_cols, field_names = \
        load_test_data(args.test_dir, args.field_index)
    num_nodes   = num_rows * num_cols
    num_samples = signals.shape[1]
    if args.num_samples is not None:
        num_samples = min(num_samples, args.num_samples)
        signals = signals[:, :num_samples]
        if coeffs is not None:
            coeffs = coeffs[:, :, :num_samples]
    field_name  = field_names[args.field_index] if args.field_index < len(field_names) \
                  else f"field_{args.field_index}"
    print(f"  {num_samples} samples  |  grid {num_rows}×{num_cols}  |  field={field_name}")

    # ---- fixed colorscale (computed ONCE from all test signals) ----
    print("\nComputing global color range...")
    global_vmin, global_vmax = compute_global_color_range(
        signals, args.color_pct_low, args.color_pct_high)
    print(f"  vmin={global_vmin:.4g}  vmax={global_vmax:.4g}  "
          f"(percentiles {args.color_pct_low}–{args.color_pct_high})")

    # ---- load NIGWT ----
    nigwt_model = nigwt_cfg = nigwt_norm_stats = matrix_cache = None
    if args.gwt_checkpoint and args.gwt_run_dir:
        print("\nLoading NIGWT...")
        nigwt_model, nigwt_cfg, nigwt_norm_stats = load_nigwt(
            args.gwt_run_dir, args.gwt_checkpoint, device)
        if coeffs is None:
            print("  ⚠ No GWT coefficients in test data — NIGWT inference skipped")
            nigwt_model = None
        else:
            from data.dataset_turbulent import (
                GraphSignalDatasetTurbulent,
                create_matrix_cache_turbulent,
            )
            ds = GraphSignalDatasetTurbulent(
                base_data_dir=args.test_dir,
                field_index=args.field_index,
                preload_to_memory=False,
                use_normalization=False,
            )
            matrix_cache = create_matrix_cache_turbulent(dataset=ds, device=device)
            print("  NIGWT matrix cache ready.")
    else:
        print("\n[NIGWT] skipped — no checkpoint / run-dir provided")

    # ---- load SVGAE ----
    svgae_model = svgae_mean = svgae_std = None
    if args.svgae_checkpoint:
        print("\nLoading Sparse-VGAE...")
        svgae_model, svgae_mean, svgae_std = load_svgae(
            args.svgae_checkpoint,
            args.svgae_latent_dim, args.svgae_hidden_layers, device)
    else:
        print("\n[SVGAE] skipped — no checkpoint provided")

    # ---- load GXN ----
    gxn_samp = gxn_rec = gxn_mean = gxn_std = gxn_keep = None
    if args.gxn_checkpoint:
        print("\nLoading GXN...")
        gxn_samp, gxn_rec, gxn_mean, gxn_std, gxn_keep = load_gxn(
            args.gxn_checkpoint,
            args.gxn_hidden_dim, args.gxn_max_hops, device)
    else:
        print("\n[GXN] skipped — no checkpoint provided")

    # ---- verify normalization comparability ----
    print("\n[Normalization comparability check]")
    print(f"  SVGAE  norm_mean={svgae_mean:.4f}  norm_std={svgae_std:.4f}"
          if svgae_model else "  SVGAE  not loaded")
    print(f"  GXN    norm_mean={gxn_mean:.4f}    norm_std={gxn_std:.4f}"
          if gxn_samp else "  GXN    not loaded")
    if nigwt_norm_stats:
        fn = list(nigwt_norm_stats.keys())[args.field_index] \
             if args.field_index < len(nigwt_norm_stats) else list(nigwt_norm_stats.keys())[0]
        nm = nigwt_norm_stats[fn]["mean"]
        ns = nigwt_norm_stats[fn]["std"]
        print(f"  NIGWT  norm_mean={nm:.4f}  norm_std={ns:.4f}  (field '{fn}')")
    print("  → RMSE values are all in physical units — comparable across models.")

    # ---- generate frames ----
    out_root = Path(args.output_dir)
    frames_dir = out_root / "comparison_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = out_root / "rmse_comparison.csv"
    rmse_rows = []

    print(f"\nGenerating {num_samples} comparison frames → {frames_dir}")

    for i in tqdm(range(num_samples), desc="Frames"):
        x_orig = signals[:, i]   # (N,)

        # NIGWT
        x_nigwt = None
        if nigwt_model is not None and coeffs is not None:
            c = coeffs[:, :, i]   # (N, F)
            x_nigwt = nigwt_reconstruct(
                nigwt_model, nigwt_cfg, nigwt_norm_stats,
                c, edge_index, matrix_cache, device,
                field_index=args.field_index,
            )

        # SVGAE
        x_svgae = None
        if svgae_model is not None:
            x_svgae = svgae_reconstruct(
                svgae_model, svgae_mean, svgae_std,
                x_orig, edge_index, device)

        # GXN
        x_gxn = None
        if gxn_samp is not None:
            x_gxn = gxn_reconstruct(
                gxn_samp, gxn_rec, gxn_mean, gxn_std, gxn_keep,
                x_orig, edge_index, num_nodes, device)

        fpath, rmse_n, rmse_s, rmse_g = render_frame(
            x_orig, x_nigwt, x_svgae, x_gxn,
            num_rows, num_cols,
            global_vmin, global_vmax,
            sample_index=i,
            field_name=field_name,
            out_dir=frames_dir,
            frame_index=i,
        )
        rmse_rows.append({'sample': i,
                          'rmse_nigwt': rmse_n,
                          'rmse_svgae': rmse_s,
                          'rmse_gxn':   rmse_g})

    # ---- save RMSE CSV ----
    with open(str(csv_path), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample', 'rmse_nigwt', 'rmse_svgae', 'rmse_gxn'])
        w.writeheader()
        w.writerows(rmse_rows)
    print(f"\nRMSE per sample → {csv_path}")

    # ---- save RMSE summary ----
    def _mean(rows, key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float('nan')

    summary = {
        'num_samples':   num_samples,
        'field':         field_name,
        'global_vmin':   global_vmin,
        'global_vmax':   global_vmax,
        'mean_rmse_nigwt': _mean(rmse_rows, 'rmse_nigwt'),
        'mean_rmse_svgae': _mean(rmse_rows, 'rmse_svgae'),
        'mean_rmse_gxn':   _mean(rmse_rows, 'rmse_gxn'),
        'note': (
            'All RMSE values are in physical (denormalised) units. '
            'Colorscale is fixed across ALL frames using '
            f'[{args.color_pct_low}th, {args.color_pct_high}th] percentiles '
            'of the test signals.'
        ),
    }
    summary_path = out_root / "rmse_summary.json"
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mean RMSE NIGWT : {summary['mean_rmse_nigwt']:.6f}")
    print(f"  Mean RMSE SVGAE : {summary['mean_rmse_svgae']:.6f}")
    print(f"  Mean RMSE GXN   : {summary['mean_rmse_gxn']:.6f}")
    print(f"  Colorscale      : [{global_vmin:.4g}, {global_vmax:.4g}]  (FIXED)")
    print(f"\nFrames    → {frames_dir}/")
    print(f"RMSE CSV  → {csv_path}")
    print(f"Summary   → {summary_path}")
    print("\nAssemble GIF with:")
    print(f"  cd {frames_dir}")
    print("  ffmpeg -framerate 8 -pattern_type glob -i 'frame_*.png' "
          "-vf scale=1920:-1 comparison.gif")


if __name__ == "__main__":
    main()
