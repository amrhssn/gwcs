# -*- coding: utf-8 -*-
"""
compare_all_visualizations.py
Loads a specific GWT checkpoint and plots the same 8 test samples
using visualize_grid_graph_inverse_sgwt_reconstruction, identical
to how GXN and VGAE baseline scripts produce their per-sample figures.

Usage:
    python compare_all_visualizations.py \
        --test-dir .../processed/20260204_193616/test \
        --field-index 0 \
        --keep-ratio 0.01 \
        --gwt-checkpoint .../checkpoints/397.pt \
        --gwt-run-dir .../runs_turbulent/2026-03-06_.../ \
        --output-dir .../baseline_results/gwt_vis \
        --sample-indices 0 1 2 3 4 5 6 7
"""

import sys
import os
import json
import argparse
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.gwt_coeffs import visualize_grid_graph_inverse_sgwt_reconstruction


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
    coeffs = torch.stack(coeffs_list, dim=2) if coeffs_list else None  # (N, F, num_samples)
    return signals, coeffs, edge_index, num_nodes, num_rows, num_cols


# =============================================================================
# GWT Model Loading and Inference
# =============================================================================
def load_gwt_model(gwt_run_dir: str, gwt_checkpoint: str, device):
    import json
    run_dir = Path(gwt_run_dir)

    # Load config JSON
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
        raise FileNotFoundError(f"GWT checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    # Strip DDP 'module.' prefix if present
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Get num_filters from gwt_config.json in the test_data_dir
    # From preprocessing: num_filters = num_scales + 1 = 5
    test_data_dir = Path(cfg['test_data_dir'])
    gwt_config_path = test_data_dir / 'gwt_config.json'
    if gwt_config_path.exists():
        with open(str(gwt_config_path), 'r') as f:
            gwt_config = json.load(f)
        num_filters = gwt_config['num_filters']
    else:
        # Fallback: infer from checkpoint weights
        # The encoder input layer weight tells us num_filters
        # Look for the first linear/conv layer that takes num_filters as input
        num_filters = 5  # default: 4 scales + 1 scaling function

    use_normalization = cfg.get('use_normalization', False)

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT
    model_name = cfg.get('model_name', 'GNNNeuralIGWT')

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

    coeffs_flat = coeffs_sample.to(device)
    keep_ratio = cfg.get('keep_ratio', 0.01)
    mask = multilevel_importance_sampling(coeffs=coeffs_flat, keep_ratio=keep_ratio)

    edge_index_dev = edge_index.to(device)
    batch_vec = torch.zeros(coeffs_flat.shape[0], dtype=torch.long, device=device)
    ptr = torch.tensor([0, coeffs_flat.shape[0]], device=device)
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

    # Denormalize if normalization was used during training
    # norm_stats is {'density': {'mean': float, 'std': float}, ...}
    use_normalization = cfg.get('use_normalization', False)
    if use_normalization and norm_stats is not None:
        field_names = ['density', 'pressure', 'velocity_x', 'velocity_y']
        fname = field_names[field_index]
        gwt_m = norm_stats[fname]['mean']
        gwt_s = norm_stats[fname]['std']
        x_hat_cpu = x_hat_cpu * gwt_s + gwt_m

    return x_hat_cpu


# =============================================================================
# Main: plot same 8 samples as GXN/VGAE using visualize_grid_graph_inverse_sgwt_reconstruction
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot GWT reconstruction for fixed test samples (same format as GXN/VGAE)")
    parser.add_argument('--test-dir',       required=True,
                        help='Path to processed test split dir')
    parser.add_argument('--field-index',    type=int, default=0)
    parser.add_argument('--keep-ratio',     type=float, default=0.01)
    parser.add_argument('--gwt-checkpoint', required=True,
                        help='Full path to specific GWT .pt checkpoint file')
    parser.add_argument('--gwt-run-dir',    required=True,
                        help='Path to GWT run directory (contains config JSON)')
    parser.add_argument('--output-dir',     default='baseline_results/gwt_vis')
    parser.add_argument('--sample-indices', type=int, nargs='+',
                        default=list(range(8)))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load test data ---
    print("Loading test data...")
    signals, coeffs, edge_index, num_nodes, num_rows, num_cols = \
        load_test_signals(args.test_dir, args.field_index)
    print(f"Test signals: {signals.shape}, Graph: {num_rows}x{num_cols}={num_nodes} nodes")

    if coeffs is None:
        raise ValueError("GWT requires pre-computed coeffs in the processed .pt files. "
                         "Ensure your preprocessing saved 'coeffs' in each sample.")

    # --- Build matrix cache (required for IGWT) ---
    print("Building Chebyshev matrix cache...")
    from data.dataset_turbulent import GraphSignalDatasetTurbulent, ChebyshevMatrixCacheTurbulent, create_matrix_cache_turbulent
    dataset_for_cache = GraphSignalDatasetTurbulent(
        base_data_dir=args.test_dir,
        field_index=args.field_index,
        preload_to_memory=False,
        use_normalization=False,
    )
    matrix_cache = create_matrix_cache_turbulent(dataset=dataset_for_cache, device=device)

    # --- Load GWT model ---
    print("Loading GWT main model...")
    gwt_model, gwt_cfg, gwt_norm_stats = load_gwt_model(
        args.gwt_run_dir, args.gwt_checkpoint, device)
    print("GWT model loaded.")

    # --- Reconstruct and visualize each sample ---
    print(f"\nGenerating GWT visualizations for samples: {args.sample_indices}")
    test_mses = []

    for sample_idx in args.sample_indices:
        print(f"\n--- Sample {sample_idx} ---")
        x_orig = signals[:, sample_idx]              # raw physical units (N,)
        coeffs_sample = coeffs[:, :, sample_idx]    # (N, num_filters)

        x_gwt = gwt_reconstruct(
            gwt_model, gwt_cfg, gwt_norm_stats,
            coeffs_sample, edge_index, matrix_cache, device,
            field_index=args.field_index,
        )

        # MSE in physical units
        mse = torch.nn.functional.mse_loss(x_gwt, x_orig).item()
        test_mses.append(mse)
        print(f"  MSE (physical): {mse:.6f}")

        # Save visualization using the same function as GXN/VGAE
        visualize_grid_graph_inverse_sgwt_reconstruction(
            x_recon=x_gwt,
            x=x_orig.cpu(),
            rows=num_rows,
            cols=num_cols,
            save_fig=True,
            filename_prefix=f'gwt_sample{sample_idx}',
            save_dir=args.output_dir,
        )

    mean_mse = sum(test_mses) / len(test_mses) if test_mses else 0.0
    print(f"\nGWT mean MSE (physical): {mean_mse:.6f}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()