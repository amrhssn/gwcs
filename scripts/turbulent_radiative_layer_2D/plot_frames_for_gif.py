#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_frames_for_gif.py

Saves per-sample visualisation frames (Original | Reconstruction | Error)
for NIGWT and Sparse-VGAE so that they can be assembled into GIFs.

One output folder per model:
  <output-dir>/nigwt_frames/   → nigwt_frame_<idx:04d>.png
  <output-dir>/svgae_frames/   → svgae_frame_<idx:04d>.png

Each frame has a model-name header and 3 adjacent panels:
  Original Signal | Reconstruction | |Error|

Usage:
    python plot_frames_for_gif.py \
        --test-dir          "$PROCESSED_DIR/test" \
        --field-index       0 \
        --keep-ratio        0.01 \
        --gwt-checkpoint    "$GWT_CHECKPOINT" \
        --gwt-run-dir       "$GWT_RUN_DIR" \
        --vgae-checkpoint   "$RESULTS_DIR/sparse_vgae/best_checkpoint.pt" \
        --output-dir        "$RESULTS_DIR/gif_frames"
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt

# ── project root ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data helpers
# =============================================================================

def load_test_signals(test_dir: str, field_index: int = 0):
    """Load ALL test signals, coeffs, and shared graph data."""
    test_dir = Path(test_dir)
    shared = torch.load(test_dir / "shared_data.pt", map_location="cpu", weights_only=False)
    edge_index = shared["edge_index"]
    num_nodes  = shared["num_nodes"]
    num_rows   = shared["num_rows"]
    num_cols   = shared["num_columns"]

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

    signals = torch.stack(signals, dim=1)  # (N, num_samples)
    coeffs  = torch.stack(coeffs_list, dim=2) if coeffs_list else None
    return signals, coeffs, edge_index, num_nodes, num_rows, num_cols


# =============================================================================
# NIGWT (GNNNeuralIGWT) helpers  - mirrors compare_all_visualizations.py
# =============================================================================

def load_gwt_model(gwt_run_dir: str, gwt_checkpoint: str, device):
    run_dir = Path(gwt_run_dir)

    config_path = run_dir / "reconstructor_pretraining_config_turbulent.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(str(config_path)) as f:
        cfg = json.load(f)

    # normalization stats
    norm_stats = None
    norm_meta_path = run_dir / "normalization_metadata.json"
    if norm_meta_path.exists():
        with open(str(norm_meta_path)) as f:
            norm_meta = json.load(f)
        nsp = norm_meta.get("normalization_stats_path")
        if nsp and Path(nsp).exists():
            norm_stats = torch.load(nsp, map_location="cpu", weights_only=False)

    # checkpoint
    ckpt_path = Path(gwt_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"GWT checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # num_filters from gwt_config.json
    test_data_dir = Path(cfg["test_data_dir"])
    gwt_config_path = test_data_dir / "gwt_config.json"
    if gwt_config_path.exists():
        with open(str(gwt_config_path)) as f:
            gwt_config = json.load(f)
        num_filters = gwt_config["num_filters"]
    else:
        num_filters = 5

    use_normalization = cfg.get("use_normalization", False)
    model_name = cfg.get("model_name", "GNNNeuralIGWT")

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT

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
            use_normalization=use_normalization,
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
            use_normalization=use_normalization,
            normalization_stats=norm_stats,
        ).to(device)

    model.load_state_dict(new_state)
    model.eval()
    return model, cfg, norm_stats


def gwt_reconstruct(model, cfg, norm_stats, coeffs_sample, edge_index,
                    matrix_cache, device, field_index=0):
    from models.sparsification_utils import multilevel_importance_sampling

    coeffs_flat = coeffs_sample.to(device)
    keep_ratio  = cfg.get("keep_ratio", 0.01)
    mask        = multilevel_importance_sampling(coeffs=coeffs_flat, keep_ratio=keep_ratio)

    edge_index_dev = edge_index.to(device)
    batch_vec = torch.zeros(coeffs_flat.shape[0], dtype=torch.long, device=device)
    ptr       = torch.tensor([0, coeffs_flat.shape[0]], device=device)
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

    use_normalization = cfg.get("use_normalization", False)
    if use_normalization and norm_stats is not None:
        field_names = ["density", "pressure", "velocity_x", "velocity_y"]
        fname = field_names[field_index]
        x_hat_cpu = x_hat_cpu * norm_stats[fname]["std"] + norm_stats[fname]["mean"]

    return x_hat_cpu


# =============================================================================
# Sparse-VGAE helpers  - mirrors run_sparse_vgae_baseline.py
# =============================================================================

def load_vgae_model(vgae_checkpoint: str, edge_index: torch.Tensor, device):
    from baselines.sparse_VGAE.gae import GAE

    ckpt = torch.load(vgae_checkpoint, map_location=device, weights_only=False)
    latent_dim    = ckpt.get("latent_dim",    128)
    hidden_layers = ckpt.get("hidden_layers", 4)
    norm_mean     = torch.tensor(ckpt.get("norm_mean", 0.0))
    norm_std      = torch.tensor(ckpt.get("norm_std",  1.0))

    model = GAE(
        input_dim=1,
        latent_dim=latent_dim,
        output_dim=1,
        num_encoder_layers=hidden_layers,
        num_decoder_layers=hidden_layers,
        conv_layer_type="SAGEConv",
        activation_function="silu",
        sparsity_ratio=0.01,
        variational=False,
        sparsity_mode="ste_fixed",
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, norm_mean, norm_std


def vgae_reconstruct(model, norm_mean, norm_std, x_raw, edge_index, device):
    """Reconstruct a single sample with VGAE; returns physical-units tensor (N,)."""
    x_norm = ((x_raw - norm_mean) / norm_std).unsqueeze(1).to(device)
    edge_index_dev = edge_index.to(device)
    with torch.no_grad():
        z  = model.encode(x_norm, edge_index_dev)
        xh = model.decode(z, edge_index_dev)
    return xh.squeeze().cpu() * norm_std + norm_mean


# =============================================================================
# Frame-saving helper
# =============================================================================

FONT = dict(fontsize=14, fontweight="bold")
TICK = dict(labelsize=11)
CMAP_SIG = "RdBu_r"
CMAP_ERR = "Reds"


def save_frame(x_orig, x_recon, rows, cols, model_label, frame_index,
               out_dir: Path, sample_index: int):
    """
    Save one PNG frame:
        Model Name header
        3 panels side by side:  Original | Reconstruction | |Error|
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    x_orig  = x_orig.squeeze().cpu()
    x_recon = x_recon.squeeze().cpu()
    error   = (x_orig - x_recon).abs()

    # shared colour range for original & recon
    vmin = float(x_orig.reshape(rows, cols).min())
    vmax = float(x_orig.reshape(rows, cols).max())
    emax = float(error.reshape(rows, cols).max())

    mse  = float(((x_orig - x_recon) ** 2).mean())
    rmse = mse ** 0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"{model_label}   -   sample {sample_index}   (RMSE = {rmse:.4f})",
                 fontsize=16, fontweight="bold", y=1.01)

    col_titles = ["Original Signal", "Reconstruction", "|Error|"]
    cmaps      = [CMAP_SIG, CMAP_SIG, CMAP_ERR]
    imgs       = [x_orig, x_recon, error]
    vmins      = [vmin,   vmin,    0.0]
    vmaxs      = [vmax,   vmax,    emax]

    for ax, img, title, cmap, vmi, vma in zip(
            axes, imgs, col_titles, cmaps, vmins, vmaxs):
        im = ax.imshow(img.reshape(rows, cols),
                       aspect="auto", cmap=cmap, vmin=vmi, vmax=vma)
        ax.set_title(title, **FONT)
        ax.set_xlabel("x (cols)", fontsize=12)
        ax.set_ylabel("y (rows)", fontsize=12)
        ax.tick_params(**TICK)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fpath = out_dir / f"frame_{frame_index:04d}.png"
    plt.savefig(str(fpath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Save per-sample frames for NIGWT and SVGAE GIFs"
    )
    parser.add_argument("--test-dir",        required=True)
    parser.add_argument("--field-index",     type=int,   default=0)
    parser.add_argument("--keep-ratio",      type=float, default=0.01)
    parser.add_argument("--gwt-checkpoint",  required=True)
    parser.add_argument("--gwt-run-dir",     required=True)
    parser.add_argument("--vgae-checkpoint", required=True)
    parser.add_argument("--output-dir",      default="gif_frames")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_root   = Path(args.output_dir)
    nigwt_dir  = out_root / "nigwt_frames"
    svgae_dir  = out_root / "svgae_frames"

    # ── load data ────────────────────────────────────────────────────────────
    print("Loading test data …")
    signals, coeffs, edge_index, num_nodes, num_rows, num_cols = \
        load_test_signals(args.test_dir, args.field_index)
    num_samples = signals.shape[1]
    print(f"  {num_samples} test samples  |  grid {num_rows}×{num_cols}  ({num_nodes} nodes)")

    # ── load NIGWT ───────────────────────────────────────────────────────────
    print("Loading NIGWT model …")
    gwt_model, gwt_cfg, gwt_norm_stats = load_gwt_model(
        args.gwt_run_dir, args.gwt_checkpoint, device)

    from data.dataset_turbulent import (
        GraphSignalDatasetTurbulent,
        create_matrix_cache_turbulent,
    )
    dataset_for_cache = GraphSignalDatasetTurbulent(
        base_data_dir=args.test_dir,
        field_index=args.field_index,
        preload_to_memory=False,
        use_normalization=False,
    )
    matrix_cache = create_matrix_cache_turbulent(dataset=dataset_for_cache, device=device)
    print("  NIGWT ready.")

    # ── load SVGAE ───────────────────────────────────────────────────────────
    print("Loading Sparse-VGAE model …")
    vgae_model, vgae_mean, vgae_std = load_vgae_model(
        args.vgae_checkpoint, edge_index, device)
    print("  SVGAE ready.")

    # ── generate frames ──────────────────────────────────────────────────────
    print(f"\nGenerating frames for all {num_samples} samples …")

    for frame_idx in range(num_samples):
        x_orig = signals[:, frame_idx]  # (N,)  physical units

        # NIGWT
        if coeffs is not None:
            c = coeffs[:, :, frame_idx]  # (N, F)
            x_nigwt = gwt_reconstruct(
                gwt_model, gwt_cfg, gwt_norm_stats,
                c, edge_index, matrix_cache, device,
                field_index=args.field_index,
            )
        else:
            # fallback: just use zeros (coeffs must be present for NIGWT)
            x_nigwt = torch.zeros_like(x_orig)
            print(f"  [WARN] No coeffs found for sample {frame_idx}; NIGWT skipped.")

        save_frame(x_orig, x_nigwt, num_rows, num_cols,
                   model_label="NIGWT (GNNNeuralIGWT)",
                   frame_index=frame_idx,
                   out_dir=nigwt_dir,
                   sample_index=frame_idx)

        # SVGAE
        x_svgae = vgae_reconstruct(
            vgae_model, vgae_mean, vgae_std,
            x_orig, edge_index, device,
        )
        save_frame(x_orig, x_svgae, num_rows, num_cols,
                   model_label="Sparse-VGAE",
                   frame_index=frame_idx,
                   out_dir=svgae_dir,
                   sample_index=frame_idx)

        if (frame_idx + 1) % 10 == 0 or frame_idx == num_samples - 1:
            print(f"  [{frame_idx + 1}/{num_samples}] frames saved")

    print(f"\nAll frames saved:")
    print(f"  NIGWT  → {nigwt_dir}/")
    print(f"  SVGAE  → {svgae_dir}/")
    print("\nNext: run  make_gifs.sh  to assemble the GIFs.")


if __name__ == "__main__":
    main()