#!/usr/bin/env python3
"""
Compare model parameters across different baselines.
Reads checkpoints and prints detailed parameter counts.
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from collections import OrderedDict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def count_parameters(state_dict: dict) -> dict:
    """Count parameters per layer and total."""
    layer_counts = OrderedDict()
    total = 0
    
    for name, param in state_dict.items():
        numel = param.numel()
        layer_counts[name] = numel
        total += numel
    
    return {'total': total, 'layers': layer_counts}


def load_sparse_vgae_params(checkpoint_path: str) -> dict:
    """Load Sparse-VGAE checkpoint and count parameters."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model state dict
    state_dict = ckpt.get('model', ckpt.get('model_state_dict', None))
    if state_dict is None:
        raise ValueError("Could not find model weights in checkpoint")
    
    counts = count_parameters(state_dict)
    
    # Extract architecture info
    info = {
        'model_name': 'Sparse-VGAE',
        'latent_dim': ckpt.get('latent_dim', 'N/A'),
        'hidden_layers': ckpt.get('hidden_layers', 'N/A'),
        'sparsity_ratio': ckpt.get('sparsity_ratio', 'N/A'),
        **counts
    }
    
    return info


def load_gxn_params(checkpoint_path: str) -> dict:
    """Load GXN checkpoint and count parameters."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # GXN has two models
    sampling_dict = ckpt.get('sampling_model', {})
    recovery_dict = ckpt.get('recovery_model', {})
    
    sampling_counts = count_parameters(sampling_dict)
    recovery_counts = count_parameters(recovery_dict)
    
    info = {
        'model_name': 'GXN',
        'hidden_dim': ckpt.get('hidden_dim', 'N/A'),
        'max_hops': ckpt.get('max_hops', 'N/A'),
        'recovery_K': ckpt.get('recovery_K', 'N/A'),
        'recovery_L': ckpt.get('recovery_L', 'N/A'),
        'sampling_params': sampling_counts['total'],
        'recovery_params': recovery_counts['total'],
        'total': sampling_counts['total'] + recovery_counts['total'],
    }
    
    return info


def load_nigwt_params(checkpoint_path: str, run_dir: str = None) -> dict:
    """Load NIGWT checkpoint and count parameters."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model state dict (unwrap DDP if needed)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if isinstance(state_dict, dict):
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    counts = count_parameters(state_dict)
    
    # Try to load config for architecture details
    config_info = {}
    if run_dir:
        config_path = Path(run_dir) / 'reconstructor_pretraining_config_turbulent.json'
        if config_path.exists():
            with open(str(config_path), 'r') as f:
                cfg = json.load(f)
            config_info = {
                'model_name': cfg.get('model_name', 'GNNNeuralIGWT'),
                'hidden_dim': cfg.get('hidden_dim', 'N/A'),
                'processor_num_layers': cfg.get('processor_num_layers', 'N/A'),
                'conv_layer_type': cfg.get('conv_layer_type', 'N/A'),
                'encoder_type': cfg.get('encoder_type', 'N/A'),
                'decoder_type': cfg.get('decoder_type', 'N/A'),
            }
    
    info = {
        'model_name': 'NIGWT',
        **config_info,
        **counts
    }
    
    return info


def print_model_summary(info: dict, show_layers: bool = False):
    """Print formatted model summary."""
    print(f"\n{'='*80}")
    print(f"MODEL: {info['model_name']}")
    print(f"{'='*80}")
    
    # Print architecture info
    print("\nArchitecture:")
    for key, val in info.items():
        if key in ['model_name', 'total', 'layers', 'sampling_params', 'recovery_params']:
            continue
        print(f"  {key}: {val}")
    
    # Print parameter counts
    print(f"\nParameter Count:")
    if 'sampling_params' in info:  # GXN special case
        print(f"  Sampling Model:  {info['sampling_params']:>12,}")
        print(f"  Recovery Model:  {info['recovery_params']:>12,}")
        print(f"  {'-'*40}")
    print(f"  Total:           {info['total']:>12,}")
    
    # Print layer details if requested
    if show_layers and 'layers' in info:
        print(f"\nLayer-wise Breakdown:")
        for name, count in info['layers'].items():
            print(f"  {name:<50} {count:>12,}")


def compare_models(checkpoints: dict, show_layers: bool = False):
    """Compare multiple models side by side."""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Extract totals
    models = []
    for name, info in checkpoints.items():
        models.append({
            'name': info['model_name'],
            'total': info['total'],
            'info': info
        })
    
    # Sort by parameter count
    models.sort(key=lambda x: x['total'])
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Parameters':>15} {'Relative Size':>15}")
    print(f"{'-'*50}")
    
    min_params = models[0]['total']
    for m in models:
        ratio = m['total'] / min_params
        print(f"{m['name']:<20} {m['total']:>15,} {ratio:>14.2f}x")
    
    # Print detailed summaries if requested
    if show_layers:
        for name, info in checkpoints.items():
            print_model_summary(info, show_layers=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare model parameters from checkpoints")
    parser.add_argument('--gxn-checkpoint', type=str, help='Path to GXN checkpoint')
    parser.add_argument('--vgae-checkpoint', type=str, help='Path to Sparse-VGAE checkpoint')
    parser.add_argument('--nigwt-checkpoint', type=str, help='Path to NIGWT checkpoint')
    parser.add_argument('--nigwt-run-dir', type=str, help='Path to NIGWT run directory (for config)')
    parser.add_argument('--show-layers', action='store_true', help='Show layer-wise breakdown')
    parser.add_argument('--output-json', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    checkpoints = {}
    
    # Load GXN
    if args.gxn_checkpoint:
        try:
            info = load_gxn_params(args.gxn_checkpoint)
            checkpoints['gxn'] = info
            print_success(f"Loaded GXN: {info['total']:,} parameters")
        except Exception as e:
            print_error(f"Failed to load GXN checkpoint: {e}")
    
    # Load Sparse-VGAE
    if args.vgae_checkpoint:
        try:
            info = load_sparse_vgae_params(args.vgae_checkpoint)
            checkpoints['vgae'] = info
            print_success(f"Loaded Sparse-VGAE: {info['total']:,} parameters")
        except Exception as e:
            print_error(f"Failed to load Sparse-VGAE checkpoint: {e}")
    
    # Load NIGWT
    if args.nigwt_checkpoint:
        try:
            info = load_nigwt_params(args.nigwt_checkpoint, args.nigwt_run_dir)
            checkpoints['nigwt'] = info
            print_success(f"Loaded NIGWT: {info['total']:,} parameters")
        except Exception as e:
            print_error(f"Failed to load NIGWT checkpoint: {e}")
    
    if not checkpoints:
        print_error("No checkpoints loaded. Provide at least one checkpoint path.")
        return
    
    # Compare models
    compare_models(checkpoints, show_layers=args.show_layers)
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(checkpoints, f, indent=2, default=str)
        print_success(f"\nSaved results to: {args.output_json}")


def print_success(msg):
    print(f"\033[92m✓ {msg}\033[0m")


def print_error(msg):
    print(f"\033[91m✗ {msg}\033[0m")


if __name__ == "__main__":
    main()