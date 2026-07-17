"""
plot_rmse_curves.py

Computes and plots per-sample RMSE across all test samples for:
- GWCS (GNNNeuralIGWT)
- Sparse-VGAE (99K and 50K)
- GXN  (optional)
- Classical baselines: FastGSSS, BSGDA, RSBS (optional, from test_metrics.json)

ALL reconstructions are denormalized to PHYSICAL UNITS before computing RMSE
so that every model is on the same scale and directly comparable.

Output plots (saved to --output-dir):
  rmse_ml_gwcs_all_samples.png   — ML/GWCS only, ALL test samples    (always)
  rmse_50_samples.png            — ML/GWCS only, first 50 samples     (always)
  rmse_with_classical_50.png     — ML/GWCS + classical, first N       (if --classical-results given)
"""

import sys
import os
import json
import math
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Loading
# =============================================================================

def load_test_metadata(test_dir: str):
    """Load shared graph metadata and return sorted list of per-sample files."""
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
    return sample_files, edge_index, num_nodes, num_rows, num_cols


def load_single_sample(filepath, field_index: int = 0):
    """
    Load one sample .pt file and return (signal, coeffs_1d) in float32.
    signal:      (N_nodes,)
    coeffs_1d:   (N_nodes, N_filters)  or None if coeffs not stored
    """
    s = torch.load(filepath, map_location='cpu', weights_only=False)
    signal = s.x[:, field_index].float()                        # (N_nodes,)
    coeffs = None
    if hasattr(s, 'coeffs') and s.coeffs is not None:
        coeffs = s.coeffs[:, field_index, :].float()            # (N_nodes, N_filters)
    return signal, coeffs


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
    """
    Reconstruct using Sparse-VGAE in PHYSICAL UNITS.
    1. Normalize input (model expects normalized data)
    2. Model outputs normalized reconstruction
    3. Denormalize back to physical units
    """
    mean_t = torch.tensor(norm_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32, device=device)

    x_norm = (x_orig.to(device) - mean_t) / std_t
    x_in   = x_norm.unsqueeze(1)  # (N, 1)

    with torch.no_grad():
        z_sparse   = model.encode(x_in, edge_index)
        x_hat_norm = model.decode(z_sparse, edge_index)  # (N, 1) normalized

    # Denormalize to physical units
    x_hat_physical = (x_hat_norm.squeeze() * std_t + mean_t).cpu()
    return x_hat_physical


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
    """
    Reconstruct using GXN in PHYSICAL UNITS.
    Same normalize-run-denormalize pattern as VGAE.
    """
    num_nodes = x_orig.shape[0]
    M = max(1, int(keep_ratio * num_nodes))

    mean_t = torch.tensor(norm_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32, device=device)

    x_norm = (x_orig.to(device) - mean_t) / std_t
    x_in   = x_norm.unsqueeze(1)  # (N, 1)

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

    # Denormalize to physical units
    x_hat_physical = (x_hat_norm.squeeze() * std_t + mean_t).cpu()
    return x_hat_physical


# =============================================================================
# NIGWT (GNNNeuralIGWT) Model Loading and Inference
# =============================================================================

def load_gwt_model(gwt_run_dir: str, gwt_checkpoint: str, device):
    import json
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
    """
    Reconstruct using NIGWT and return result in PHYSICAL UNITS.

    When use_normalization=True the model decode() does:
        x_hat_igwt = (igwt_physical - mean) / std   (normalize IGWT output)
        x_hat = x_hat_igwt + residual                (add normalized residual)
    So the model output is in NORMALIZED space.
    We must denormalize: x_physical = x_hat * std + mean
    """
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

    # Denormalize to physical units.
    # The model outputs in NORMALIZED space when use_normalization=True
    # because decode() normalizes the IGWT output and adds a normalized residual.
    # We reverse that here using the same field stats the model used internally.
    use_normalization = cfg.get('use_normalization', False)
    if use_normalization and norm_stats is not None:
        # Use dynamic field names from the stats dict (works for any dataset)
        field_names = list(norm_stats.keys())
        fname = field_names[field_index]
        gwt_m = norm_stats[fname]['mean']
        gwt_s = norm_stats[fname]['std']
        x_hat_cpu = x_hat_cpu * gwt_s + gwt_m

    return x_hat_cpu


# =============================================================================
# RMSE Computation and Plotting
# =============================================================================

def plot_rmse_curves(args):
    """Main function to compute and plot RMSE curves for all models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # --- Load test metadata (graph structure + file list only — no bulk tensor load) ---
    print("Loading test metadata...")
    sample_files, edge_index, num_nodes, num_rows, num_cols = \
        load_test_metadata(args.test_dir)
    num_samples = len(sample_files)
    print(f"  Found {num_samples} test samples")
    print(f"  Grid: {num_rows} x {num_cols} = {num_nodes} nodes\n")

    edge_index_dev = edge_index.to(device)

    # --- Load Sparse-VGAE-99K ---
    vgae_model, vgae_mean, vgae_std = None, None, None
    if args.vgae_checkpoint and Path(args.vgae_checkpoint).exists():
        print("Loading Sparse-VGAE-99K model...")
        vgae_model, vgae_mean, vgae_std = load_vgae_model(
            args.vgae_checkpoint, args.keep_ratio, device)
        print(f"  VGAE-99K normalization: mean={vgae_mean:.6f}, std={vgae_std:.6f}")
    else:
        print("Skipping Sparse-VGAE-99K (no checkpoint provided)")

    # --- Load Sparse-VGAE-50K ---
    vgae50_model, vgae50_mean, vgae50_std = None, None, None
    if args.vgae_50k_checkpoint and Path(args.vgae_50k_checkpoint).exists():
        print("Loading Sparse-VGAE-50K model...")
        vgae50_model, vgae50_mean, vgae50_std = load_vgae_model(
            args.vgae_50k_checkpoint, args.keep_ratio, device)
        print(f"  VGAE-50K normalization: mean={vgae50_mean:.6f}, std={vgae50_std:.6f}")
    else:
        print("Skipping Sparse-VGAE-50K (no checkpoint provided)")

    # --- Load GXN ---
    gxn_sampling, gxn_recovery, gxn_mean, gxn_std = None, None, None, None
    if args.gxn_checkpoint and Path(args.gxn_checkpoint).exists():
        print("\nLoading GXN model...")
        gxn_sampling, gxn_recovery, gxn_mean, gxn_std = load_gxn_model(
            args.gxn_checkpoint, device)
        print(f"  GXN normalization: mean={gxn_mean:.6f}, std={gxn_std:.6f}")
    else:
        print("Skipping GXN (no checkpoint provided)")

    # --- Load NIGWT + matrix cache ---
    print("\nLoading NIGWT model...")
    gwt_model, gwt_cfg, gwt_norm_stats = load_gwt_model(
        args.gwt_run_dir, args.gwt_checkpoint, device)

    if gwt_norm_stats is not None:
        print(f"  NIGWT uses per-field normalization:")
        for field_name, stats in gwt_norm_stats.items():
            print(f"    {field_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    else:
        print(f"  NIGWT: no normalization")

    # Quick check: does the first sample file have coeffs?
    _sig0, _c0 = load_single_sample(sample_files[0], args.field_index)
    if _c0 is None:
        raise ValueError("NIGWT requires pre-computed coeffs in the .pt files. "
                         "Check that preprocessing saved 'coeffs' in each sample.")
    del _sig0, _c0

    print("\nBuilding Chebyshev matrix cache...")
    from data.dataset_turbulent import GraphSignalDatasetTurbulent, create_matrix_cache_turbulent

    dataset_for_cache = GraphSignalDatasetTurbulent(
        base_data_dir=args.test_dir,
        field_index=args.field_index,
        preload_to_memory=False,
        use_normalization=False,
    )

    matrix_cache = create_matrix_cache_turbulent(
        dataset=dataset_for_cache,
        device=device
    )

    # --- Load classical baseline results (optional) ---
    classical_data = None
    if args.classical_results and Path(args.classical_results).exists():
        print(f"\nLoading classical baseline results from {args.classical_results}")
        with open(args.classical_results, 'r') as f:
            classical_data = json.load(f)
        for method_name, method_data in classical_data.items():
            n_phys = len(method_data.get('all_rmses_physical', []))
            n_mse  = len(method_data.get('all_mses', []))
            src = "all_rmses_physical" if n_phys > 0 else "all_mses (converted via signal_std)"
            print(f"  {method_name}: {max(n_phys, n_mse)} samples  [{src}]")
    else:
        print("Skipping classical baselines (no results file provided)")

    # --- Compute RMSE for all samples (lazy loading — one .pt file at a time) ---
    # We never stack all samples into a single tensor.  Each sample file is loaded,
    # all models run on it, then the tensors are discarded before the next sample.
    # Peak RAM ∝ ONE sample (N_nodes × N_filters × 4 bytes) instead of ALL samples.
    print(f"\nComputing RMSE in PHYSICAL UNITS for fair comparison")
    print(f"Processing {num_samples} samples (lazy load, one at a time)...\n")

    rmse_nigwt  = []
    rmse_vgae   = []   # 99K
    rmse_vgae50 = []   # 50K
    rmse_gxn    = []
    rmse_classical = {}
    if classical_data:
        for method_name in classical_data:
            rmse_classical[method_name] = []

    # Track last-sample reconstructions for the diagnostic printout
    x_orig_last = x_vgae_last = x_vgae50_last = x_gxn_last = x_nigwt_last = None

    # Welford's online algorithm for computing signal_std across all test samples.
    # Used for signal-normalised RMSE:  norm = RMSE / signal_std
    #   norm = 0  → perfect reconstruction (zero error)
    #   norm = 1  → error equals signal standard deviation
    _sig_n    = 0
    _sig_mean = 0.0
    _sig_M2   = 0.0   # running sum of squared deviations

    for sample_idx in tqdm(range(num_samples), desc="Processing", unit="sample"):
        # ── Load one sample file (physical units, float32) ────────────────────
        # x_orig:        (N_nodes,)           — raw physical-unit signal
        # coeffs_sample: (N_nodes, N_filters) — wavelet coefficients for this sample
        x_orig, coeffs_sample = load_single_sample(
            sample_files[sample_idx], args.field_index)
        x_orig_last = x_orig

        # ── Update running signal statistics (Welford, vectorised over nodes) ─
        _xnp     = x_orig.numpy()
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

        # ── NIGWT: coeffs → model → denormalize → physical units ─────────────
        x_nigwt = gwt_reconstruct(
            gwt_model, gwt_cfg, gwt_norm_stats,
            coeffs_sample, edge_index, matrix_cache, device,
            field_index=args.field_index,
        )
        rmse_nigwt.append(torch.sqrt(torch.mean((x_nigwt - x_orig) ** 2)).item())
        x_nigwt_last = x_nigwt

        # ── Sparse-VGAE-99K: normalize → infer → denormalize ─────────────────
        if vgae_model is not None:
            x_vgae = vgae_reconstruct(
                vgae_model, vgae_mean, vgae_std,
                x_orig, edge_index_dev, device
            )
            rmse_vgae.append(torch.sqrt(torch.mean((x_vgae - x_orig) ** 2)).item())
            x_vgae_last = x_vgae

        # ── Sparse-VGAE-50K ───────────────────────────────────────────────────
        if vgae50_model is not None:
            x_vgae50 = vgae_reconstruct(
                vgae50_model, vgae50_mean, vgae50_std,
                x_orig, edge_index_dev, device
            )
            rmse_vgae50.append(torch.sqrt(torch.mean((x_vgae50 - x_orig) ** 2)).item())
            x_vgae50_last = x_vgae50

        # ── GXN: normalize → infer → denormalize ──────────────────────────────
        if gxn_sampling is not None:
            x_gxn = gxn_reconstruct(
                gxn_sampling, gxn_recovery, gxn_mean, gxn_std,
                x_orig, edge_index_dev, args.keep_ratio, device
            )
            rmse_gxn.append(torch.sqrt(torch.mean((x_gxn - x_orig) ** 2)).item())
            x_gxn_last = x_gxn

        # ── Classical baselines ───────────────────────────────────────────────
        # Prefer all_rmses_physical (pre-computed in physical units by the
        # classical baseline script — exact same scale as neural model RMSE).
        # Fall back to sqrt(MSE_norm) * signal_std using the std stored in JSON.
        if classical_data:
            for method_name, method_data in classical_data.items():
                all_rmses_phys = method_data.get('all_rmses_physical', [])
                if all_rmses_phys and sample_idx < len(all_rmses_phys):
                    rmse_classical[method_name].append(float(all_rmses_phys[sample_idx]))
                else:
                    all_mses = method_data.get('all_mses', [])
                    sig_std  = float(method_data.get('signal_std', 1.0))
                    if sample_idx < len(all_mses):
                        rmse_classical[method_name].append(
                            math.sqrt(all_mses[sample_idx]) * sig_std)

    # --- Compute signal_std from Welford accumulators (or use CLI override) ---
    if getattr(args, 'signal_std', None) is not None:
        signal_std = float(args.signal_std)
        print(f"\nSignal std (CLI override): {signal_std:.6f}")
    elif _sig_n > 0:
        signal_std = math.sqrt(_sig_M2 / _sig_n)
        print(f"\nSignal std (computed from {_sig_n:,} values across {num_samples} samples): "
              f"{signal_std:.6f}")
    else:
        signal_std = 1.0
        print("\nWARNING: could not compute signal_std — defaulting to 1.0")

    # --- Diagnostic: value ranges from last sample (all in physical units) ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Value Ranges on Last Sample (Physical Units)")
    print("=" * 70)
    if x_orig_last is not None:
        print(f"Original signal:      min={x_orig_last.min():.6f}, max={x_orig_last.max():.6f}")
    if x_vgae_last is not None:
        print(f"VGAE last recon:      min={x_vgae_last.min():.6f}, max={x_vgae_last.max():.6f}")
    if x_vgae50_last is not None:
        print(f"VGAE-50K last recon:  min={x_vgae50_last.min():.6f}, max={x_vgae50_last.max():.6f}")
    if x_gxn_last is not None:
        print(f"GXN last recon:       min={x_gxn_last.min():.6f}, max={x_gxn_last.max():.6f}")
    if x_nigwt_last is not None:
        print(f"GWCS last recon:      min={x_nigwt_last.min():.6f}, max={x_nigwt_last.max():.6f}")
    print("=" * 70)

    # --- Plot RMSE curves ---
    print("\nPlotting RMSE curves...")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Font / tick sizes ────────────────────────────────────────────────────
    # Text sizes rolled back to 3× baseline values.
    # Tick MARK geometry kept at 5× defaults so the dashes are clearly visible.
    plt.rcParams.update({
        "figure.dpi":           200,
        "savefig.dpi":          200,
        "font.size":            34,    # base font
        "axes.labelsize":       34,    # x / y axis labels
        "axes.titlesize":       36,    # (unused — titles removed)
        "legend.fontsize":      28,    # legend entries
        "xtick.labelsize":      28,    # tick number labels
        "ytick.labelsize":      28,
        "xtick.major.size":     18,    # major tick length  (5× default 3.5)
        "xtick.major.width":    4,     # major tick width   (5× default 0.8)
        "xtick.minor.size":     10,    # minor tick length  (5× default 2)
        "xtick.minor.width":    3,     # minor tick width   (5× default 0.6)
        "ytick.major.size":     18,
        "ytick.major.width":    4,
        "ytick.minor.size":     10,
        "ytick.minor.width":    3,
        "xtick.major.pad":      4,     # gap between tick end and label
        "ytick.major.pad":      4,
        "axes.linewidth":       1.5,
        "lines.linewidth":      2.8,
    })

    classical_colors = {'fastgsss': '#17becf', 'bsgda': '#9467bd', 'rsbs': '#8c564b'}
    classical_labels = {'fastgsss': 'FastGSSS', 'bsgda': 'BSGDA', 'rsbs': 'RSBS'}

    sample_indices = np.arange(1, num_samples + 1)  # 1-indexed for x-axis

    # ---------------------------------------------------------------
    # PLOT 1: ML baselines + GWCS — ALL test samples (no classical)
    # ---------------------------------------------------------------
    print("\n  Generating Plot 1: ML + GWCS, all samples...")
    fig1, ax1 = plt.subplots(figsize=(22, 10))

    ax1.plot(sample_indices, rmse_nigwt, linewidth=2.8, label="GWCS", color='#1f77b4')

    if rmse_vgae:
        ax1.plot(sample_indices[:len(rmse_vgae)], rmse_vgae, linewidth=2.8,
                 label="SGAE-99K", color='#ff7f0e')

    if rmse_vgae50:
        ax1.plot(sample_indices[:len(rmse_vgae50)], rmse_vgae50, linewidth=2.8,
                 label="SGAE-50K", color='#e31a1c')

    if rmse_gxn:
        ax1.plot(sample_indices[:len(rmse_gxn)], rmse_gxn, linewidth=2.8,
                 label="GXN", color='#2ca02c')

    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("RMSE")
    ax1.grid(True, alpha=0.3)
    # Legend centred above the axes (replaces the removed title)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=True, framealpha=0.95, borderaxespad=0)
    fig1.tight_layout()

    out_ml = os.path.join(args.output_dir, "rmse_ml_gwcs_all_samples.png")
    fig1.savefig(out_ml, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {out_ml}")

    # ---------------------------------------------------------------
    # PLOT 1b: ML baselines + GWCS — first 50 samples (no classical)
    #          Always generated, regardless of classical data.
    # ---------------------------------------------------------------
    n_50 = min(50, num_samples)
    print(f"\n  Generating Plot 1b: ML + GWCS, first {n_50} samples...")
    idx_50 = np.arange(1, n_50 + 1)
    fig1b, ax1b = plt.subplots(figsize=(22, 10))

    ax1b.plot(idx_50, rmse_nigwt[:n_50], linewidth=2.8, label="GWCS", color='#1f77b4')

    if rmse_vgae:
        ax1b.plot(idx_50[:min(n_50, len(rmse_vgae))],
                  rmse_vgae[:n_50], linewidth=2.8,
                  label="SGAE-99K", color='#ff7f0e')

    if rmse_vgae50:
        ax1b.plot(idx_50[:min(n_50, len(rmse_vgae50))],
                  rmse_vgae50[:n_50], linewidth=2.8,
                  label="SGAE-50K", color='#e31a1c')

    if rmse_gxn:
        ax1b.plot(idx_50[:min(n_50, len(rmse_gxn))],
                  rmse_gxn[:n_50], linewidth=2.8,
                  label="GXN", color='#2ca02c')

    ax1b.set_xlabel("Frame Index")
    ax1b.set_ylabel("RMSE")
    ax1b.grid(True, alpha=0.3)
    # Legend centred above the axes (replaces the removed title)
    ax1b.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=4, frameon=True, framealpha=0.95, borderaxespad=0)
    fig1b.tight_layout()

    out_50 = os.path.join(args.output_dir, "rmse_50_samples.png")
    fig1b.savefig(out_50, dpi=200, bbox_inches="tight")
    plt.close(fig1b)
    print(f"  Saved: {out_50}")

    # ---------------------------------------------------------------
    # PLOT 2: ML baselines + NIGWT + Classical — first N samples
    #         (only when --classical-results is provided)
    # ---------------------------------------------------------------
    if rmse_classical:
        n_classical = args.num_classical_samples
        # Cap to the minimum of: requested N, available classical samples, total samples
        max_classical_avail = max(
            (len(v) for v in rmse_classical.values() if v), default=0
        )
        n_plot = min(n_classical, max_classical_avail, num_samples)
        print(f"\n  Generating Plot 2: ML + GWCS + Classical, first {n_plot} samples...")

        idx_cls = np.arange(1, n_plot + 1)
        fig2, ax2 = plt.subplots(figsize=(22, 10))

        ax2.plot(idx_cls, rmse_nigwt[:n_plot], linewidth=2.8, label="GWCS", color='#1f77b4')

        if rmse_vgae:
            n_v = min(n_plot, len(rmse_vgae))
            ax2.plot(idx_cls[:n_v], rmse_vgae[:n_v], linewidth=2.8,
                     label="SGAE-99K", color='#ff7f0e')

        if rmse_vgae50:
            n_v50 = min(n_plot, len(rmse_vgae50))
            ax2.plot(idx_cls[:n_v50], rmse_vgae50[:n_v50], linewidth=2.8,
                     label="SGAE-50K", color='#e31a1c')

        if rmse_gxn:
            n_g = min(n_plot, len(rmse_gxn))
            ax2.plot(idx_cls[:n_g], rmse_gxn[:n_g], linewidth=2.8,
                     label="GXN", color='#2ca02c')

        for method_name, method_rmse in rmse_classical.items():
            if method_rmse:
                n_c = min(n_plot, len(method_rmse))
                ax2.plot(idx_cls[:n_c], method_rmse[:n_c], linewidth=2.2, linestyle='--',
                         label=classical_labels.get(method_name, method_name),
                         color=classical_colors.get(method_name, None))

        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("RMSE")
        ax2.grid(True, alpha=0.3)
        # Count active series so we can fit them in as few rows as possible
        n_series = (1                              # GWCS always
                    + (1 if rmse_vgae   else 0)
                    + (1 if rmse_vgae50 else 0)
                    + (1 if rmse_gxn    else 0)
                    + sum(1 for v in rmse_classical.values() if v))
        leg_ncol = min(n_series, 4)   # ≤4 per row keeps labels readable
        # Legend centred above the axes (replaces the removed title)
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                   ncol=leg_ncol, frameon=True, framealpha=0.95, borderaxespad=0)
        fig2.tight_layout()

        out_cls = os.path.join(args.output_dir, "rmse_with_classical_50.png")
        fig2.savefig(out_cls, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {out_cls}")
    else:
        print("\n  Skipping Plot 2 (no classical results provided)")

    # --- Print summary statistics ---
    print("\n" + "=" * 80)
    print("RMSE Summary Statistics (Physical Units)")
    print("=" * 80)
    print(f"{'Model':<20} {'Mean RMSE':<14} {'Std RMSE':<14} {'Min RMSE':<14} {'Max RMSE':<14}")
    print("-" * 80)
    print(f"{'GWCS':<20} {np.mean(rmse_nigwt):<14.6f} {np.std(rmse_nigwt):<14.6f} {np.min(rmse_nigwt):<14.6f} {np.max(rmse_nigwt):<14.6f}")
    if rmse_vgae:
        print(f"{'SGAE-99K':<20} {np.mean(rmse_vgae):<14.6f} {np.std(rmse_vgae):<14.6f} {np.min(rmse_vgae):<14.6f} {np.max(rmse_vgae):<14.6f}")
    if rmse_vgae50:
        print(f"{'SGAE-50K':<20} {np.mean(rmse_vgae50):<14.6f} {np.std(rmse_vgae50):<14.6f} {np.min(rmse_vgae50):<14.6f} {np.max(rmse_vgae50):<14.6f}")
    if rmse_gxn:
        print(f"{'GXN':<20} {np.mean(rmse_gxn):<14.6f} {np.std(rmse_gxn):<14.6f} {np.min(rmse_gxn):<14.6f} {np.max(rmse_gxn):<14.6f}")
    for method_name, method_rmse in rmse_classical.items():
        if method_rmse:
            label = classical_labels.get(method_name, method_name)
            print(f"{label:<20} {np.mean(method_rmse):<14.6f} {np.std(method_rmse):<14.6f} {np.min(method_rmse):<14.6f} {np.max(method_rmse):<14.6f}")
    print("=" * 80)

    # --- Signal-normalised summary (RMSE ÷ signal_std) ---
    print("\n" + "=" * 80)
    print(f"Signal-Normalised RMSE Summary  (RMSE ÷ σ_signal,  σ = {signal_std:.6f})")
    print("  norm = 0 → perfect reconstruction (zero error)")
    print("  norm = 1 → error equals signal standard deviation")
    print("=" * 80)
    print(f"{'Model':<20} {'Mean norm':<14} {'Std norm':<14} {'Min norm':<14} {'Max norm':<14}")
    print("-" * 80)
    print(f"{'GWCS':<20} {np.mean(rmse_nigwt)/signal_std:<14.6f} {np.std(rmse_nigwt)/signal_std:<14.6f} {np.min(rmse_nigwt)/signal_std:<14.6f} {np.max(rmse_nigwt)/signal_std:<14.6f}")
    if rmse_vgae:
        print(f"{'SGAE-99K':<20} {np.mean(rmse_vgae)/signal_std:<14.6f} {np.std(rmse_vgae)/signal_std:<14.6f} {np.min(rmse_vgae)/signal_std:<14.6f} {np.max(rmse_vgae)/signal_std:<14.6f}")
    if rmse_vgae50:
        print(f"{'SGAE-50K':<20} {np.mean(rmse_vgae50)/signal_std:<14.6f} {np.std(rmse_vgae50)/signal_std:<14.6f} {np.min(rmse_vgae50)/signal_std:<14.6f} {np.max(rmse_vgae50)/signal_std:<14.6f}")
    if rmse_gxn:
        print(f"{'GXN':<20} {np.mean(rmse_gxn)/signal_std:<14.6f} {np.std(rmse_gxn)/signal_std:<14.6f} {np.min(rmse_gxn)/signal_std:<14.6f} {np.max(rmse_gxn)/signal_std:<14.6f}")
    for method_name, method_rmse in rmse_classical.items():
        if method_rmse:
            label = classical_labels.get(method_name, method_name)
            print(f"{label:<20} {np.mean(method_rmse)/signal_std:<14.6f} {np.std(method_rmse)/signal_std:<14.6f} {np.min(method_rmse)/signal_std:<14.6f} {np.max(method_rmse)/signal_std:<14.6f}")
    print("=" * 80)

    # --- Save numerical results to JSON ---
    results_dict = {
        'gwcs': rmse_nigwt,
    }
    if rmse_vgae:
        results_dict['svgae_99k'] = rmse_vgae
    if rmse_vgae50:
        results_dict['svgae_50k'] = rmse_vgae50
    if rmse_gxn:
        results_dict['gxn'] = rmse_gxn
    for method_name, method_rmse in rmse_classical.items():
        if method_rmse:
            results_dict[method_name] = method_rmse

    json_path = os.path.join(args.output_dir, "rmse_per_sample.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved per-sample RMSE to: {json_path}")

    # ---------------------------------------------------------------
    # SIGNAL-NORMALISED RMSE PLOTS
    # norm_i = rmse_i / signal_std
    # signal_std = population std of ALL test-sample signal values (physical units)
    #
    # norm = 0   → perfect reconstruction (zero error)
    # norm = 1   → error equals signal standard deviation (noise floor)
    # norm > 1   → model error larger than typical signal variation
    #
    # This is an ABSOLUTE metric: independent of which models are in the comparison,
    # and 0 genuinely means zero error (unlike min-max or Z-score).
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating Signal-Normalised RMSE plots  (RMSE ÷ σ_signal)")
    print(f"  σ_signal = {signal_std:.6f}  (population std over all {num_samples} test samples)")
    print("=" * 80)

    # Gather per-model (label, color, rmse_array) for active models only
    active_models = [("GWCS", '#1f77b4', rmse_nigwt)]
    if rmse_vgae:
        active_models.append(("SGAE-99K", '#ff7f0e', rmse_vgae))
    if rmse_vgae50:
        active_models.append(("SGAE-50K", '#e31a1c', rmse_vgae50))
    if rmse_gxn:
        active_models.append(("GXN", '#2ca02c', rmse_gxn))

    if signal_std < 1e-12:
        print("  WARNING: signal_std ≈ 0 — skipping normalised plots.")
    else:
        for lbl, _, r in active_models:
            print(f"    {lbl}: mean RMSE={np.mean(r):.6f}  →  norm={np.mean(r)/signal_std:.4f}")
        for mname, r in rmse_classical.items():
            if r:
                print(f"    {classical_labels.get(mname, mname)}: mean RMSE={np.mean(r):.6f}  →  norm={np.mean(r)/signal_std:.4f}")

        def _sig_norm(arr):
            """Divide each RMSE value by signal_std."""
            return [v / signal_std for v in arr]

        # ── Signal-norm Plot 1: ML + GWCS, ALL samples ──────────────────
        print(f"\n  Generating Signal-Norm Plot 1: ML + GWCS, all {num_samples} samples...")
        figN1, axN1 = plt.subplots(figsize=(22, 10))
        for lbl, col, r in active_models:
            axN1.plot(sample_indices[:len(r)], _sig_norm(r),
                      linewidth=2.8, label=lbl, color=col)
        axN1.axhline(1.0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6,
                     label='Error = signal σ')
        axN1.set_xlabel("Frame Index")
        axN1.set_ylabel(r"Normalised RMSE  ($\div\,\sigma_{\rm signal}$)")
        axN1.grid(True, alpha=0.3)
        axN1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=True, framealpha=0.95, borderaxespad=0)
        figN1.tight_layout()
        out_norm_all = os.path.join(args.output_dir, "rmse_normalised_all_samples.png")
        figN1.savefig(out_norm_all, dpi=200, bbox_inches="tight")
        plt.close(figN1)
        print(f"  Saved: {out_norm_all}")

        # ── Signal-norm Plot 2: ML + GWCS, first 50 samples ─────────────
        n_50_norm = min(50, num_samples)
        idx_50_norm = np.arange(1, n_50_norm + 1)
        print(f"\n  Generating Signal-Norm Plot 2: ML + GWCS, first {n_50_norm} samples...")
        figN2, axN2 = plt.subplots(figsize=(22, 10))
        for lbl, col, r in active_models:
            n_avail = min(n_50_norm, len(r))
            axN2.plot(idx_50_norm[:n_avail], _sig_norm(r[:n_50_norm])[:n_avail],
                      linewidth=2.8, label=lbl, color=col)
        axN2.axhline(1.0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6,
                     label='Error = signal σ')
        axN2.set_xlabel("Frame Index")
        axN2.set_ylabel(r"Normalised RMSE  ($\div\,\sigma_{\rm signal}$)")
        axN2.grid(True, alpha=0.3)
        axN2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                    ncol=4, frameon=True, framealpha=0.95, borderaxespad=0)
        figN2.tight_layout()
        out_norm_50 = os.path.join(args.output_dir, "rmse_normalised_50_samples.png")
        figN2.savefig(out_norm_50, dpi=200, bbox_inches="tight")
        plt.close(figN2)
        print(f"  Saved: {out_norm_50}")

        # ── Signal-norm Plot 3: ML + GWCS + Classical, first N samples ──
        if rmse_classical:
            n_classical_norm = args.num_classical_samples
            max_cls_norm = max((len(v) for v in rmse_classical.values() if v), default=0)
            n_plot_norm = min(n_classical_norm, max_cls_norm, num_samples)
            idx_cls_norm = np.arange(1, n_plot_norm + 1)
            print(f"\n  Generating Signal-Norm Plot 3: ML + GWCS + Classical, "
                  f"first {n_plot_norm} samples...")
            figN3, axN3 = plt.subplots(figsize=(22, 10))
            for lbl, col, r in active_models:
                n_av = min(n_plot_norm, len(r))
                axN3.plot(idx_cls_norm[:n_av], _sig_norm(r[:n_plot_norm])[:n_av],
                          linewidth=2.8, label=lbl, color=col)
            for mname, mrmse in rmse_classical.items():
                if mrmse:
                    n_c = min(n_plot_norm, len(mrmse))
                    axN3.plot(idx_cls_norm[:n_c], _sig_norm(mrmse[:n_plot_norm])[:n_c],
                              linewidth=2.2, linestyle='--',
                              label=classical_labels.get(mname, mname),
                              color=classical_colors.get(mname, None))
            axN3.axhline(1.0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6,
                         label='Error = signal σ')
            axN3.set_xlabel("Frame Index")
            axN3.set_ylabel(r"Normalised RMSE  ($\div\,\sigma_{\rm signal}$)")
            axN3.grid(True, alpha=0.3)
            n_series_norm = len(active_models) + sum(1 for v in rmse_classical.values() if v)
            axN3.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                        ncol=min(n_series_norm, 4), frameon=True, framealpha=0.95,
                        borderaxespad=0)
            figN3.tight_layout()
            out_norm_cls = os.path.join(args.output_dir,
                                        "rmse_normalised_with_classical_50.png")
            figN3.savefig(out_norm_cls, dpi=200, bbox_inches="tight")
            plt.close(figN3)
            print(f"  Saved: {out_norm_cls}")

        # ── Save signal-normalised values to JSON ────────────────────────
        norm_results = {
            'signal_std':      signal_std,
            'normalization':   'rmse_div_signal_std',
        }
        for lbl, _, r in active_models:
            key = lbl.lower().replace('-', '_')
            norm_results[key] = _sig_norm(r)
        for mname, mrmse in rmse_classical.items():
            if mrmse:
                norm_results[mname] = _sig_norm(mrmse)
        norm_json_path = os.path.join(args.output_dir, "rmse_normalised_per_sample.json")
        with open(norm_json_path, 'w') as f:
            json.dump(norm_results, f, indent=2)
        print(f"\nSaved signal-normalised per-sample RMSE to: {norm_json_path}")

        # ── Signal-normalised summary table (stdout) ─────────────────────
        print("\n" + "=" * 80)
        print(f"Signal-Normalised RMSE Summary  (RMSE ÷ σ_signal,  σ = {signal_std:.6f})")
        print("  norm = 0 → perfect;  norm = 1 → error = signal std")
        print("=" * 80)
        print(f"{'Model':<20} {'Mean norm':>12} {'Std norm':>12}")
        print("-" * 48)
        for lbl, _, r in active_models:
            nr = np.array(r)
            print(f"{lbl:<20} {np.mean(nr)/signal_std:>12.6f} {np.std(nr)/signal_std:>12.6f}")
        for mname, mrmse in rmse_classical.items():
            if mrmse:
                nr = np.array(mrmse)
                label = classical_labels.get(mname, mname)
                print(f"{label:<20} {np.mean(nr)/signal_std:>12.6f} {np.std(nr)/signal_std:>12.6f}")
        print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE for GWCS, Sparse-VGAE, GXN, and classical baselines (all on same physical scale)")

    # Required
    parser.add_argument('--test-dir',        required=True, help="Path to test data directory")
    parser.add_argument('--gwt-checkpoint',  required=True, help="Path to GWCS checkpoint")
    parser.add_argument('--gwt-run-dir',     required=True, help="Path to NIGWT run directory")

    # Optional model checkpoints (skipped if not provided)
    parser.add_argument('--vgae-checkpoint',     default=None,
                        help="Path to Sparse-VGAE ~99K checkpoint (latent_dim=90, optional)")
    parser.add_argument('--vgae-50k-checkpoint', default=None,
                        help="Path to Sparse-VGAE ~50K checkpoint (latent_dim=64, optional)")
    parser.add_argument('--gxn-checkpoint',      default=None,
                        help="Path to GXN checkpoint (optional)")
    parser.add_argument('--classical-results', default=None,
                        help="Path to classical test_metrics.json (optional)")

    # Config
    parser.add_argument('--field-index',           type=int,   default=0,    help="Field index to evaluate")
    parser.add_argument('--keep-ratio',             type=float, default=0.01, help="Sparsity keep ratio")
    parser.add_argument('--num-classical-samples',  type=int,   default=50,
                        help="Max samples to include in the ML+classical comparison plot (default: 50)")
    parser.add_argument('--signal-std',             type=float, default=None,
                        help="Signal std for normalisation (computed from test data if omitted)")
    parser.add_argument('--output-dir',             default='baseline_results/rmse_plots', help="Output directory")

    args = parser.parse_args()
    plot_rmse_curves(args)


if __name__ == "__main__":
    main()
