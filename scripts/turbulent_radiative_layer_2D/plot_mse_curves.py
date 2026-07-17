"""
plot_mse_curves.py

Variant of plot_rmse_curves.py that reports MSE (not RMSE) in:
  1. Physical units  (MSE = mean squared error, same scale as the signal)
  2. Signal-variance-normalised units  (MSE / σ²_signal)
        normalised MSE = 0  → perfect reconstruction (zero error)
        normalised MSE = 1  → error variance equals signal variance

Models evaluated:
  - GWCS  (GNNNeuralIGWT)
  - SGAE-99K / SGAE-50K  (Sparse-VGAE)
  - GXN  (optional)
  - Classical baselines: FastGSSS, BSGDA, RSBS (optional, from test_metrics.json)

ALL reconstructions are denormalised to PHYSICAL UNITS before computing MSE
so that every model is on the same scale and directly comparable.

Output files (saved to --output-dir):
  mse_ml_gwcs_all_samples.png        — ML/GWCS only, ALL test samples
  mse_50_samples.png                 — ML/GWCS only, first 50 samples
  mse_with_classical_50.png          — ML/GWCS + classical, first N  (if --classical-results given)
  mse_normalised_all_samples.png     — signal-variance-normalised, all samples
  mse_normalised_50_samples.png      — signal-variance-normalised, first 50
  mse_normalised_with_classical_50.png — signal-variance-normalised + classical
  mse_per_sample.json                — raw per-sample MSE values (physical)
  mse_normalised_per_sample.json     — normalised per-sample values + signal_std
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
    s = torch.load(filepath, map_location='cpu', weights_only=False)
    signal = s.x[:, field_index].float()
    coeffs = None
    if hasattr(s, 'coeffs') and s.coeffs is not None:
        coeffs = s.coeffs[:, field_index, :].float()
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
        input_dim=1, latent_dim=latent_dim, output_dim=1,
        num_encoder_layers=hidden_layers, num_decoder_layers=hidden_layers,
        conv_layer_type='SAGEConv', activation_function='silu',
        sparsity_ratio=keep_ratio, variational=False,
        sparsity_mode='ste_annealing', temperature_init=1.0,
        min_temperature=0.01, annealing_steps=1000,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, norm_mean, norm_std


def vgae_reconstruct(model, norm_mean, norm_std, x_orig, edge_index, device):
    mean_t = torch.tensor(norm_mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(norm_std,  dtype=torch.float32, device=device)
    x_norm = (x_orig.to(device) - mean_t) / std_t
    x_in   = x_norm.unsqueeze(1)
    with torch.no_grad():
        z_sparse   = model.encode(x_in, edge_index)
        x_hat_norm = model.decode(z_sparse, edge_index)
    x_hat_physical = (x_hat_norm.squeeze() * std_t + mean_t).cpu()
    return x_hat_physical


# =============================================================================
# GXN Model Loading and Inference
# =============================================================================

def load_gxn_model(gxn_checkpoint: str, device):
    from baselines.gxn.neural_sampling import NeuralSampling, NeuralRecovery
    ckpt = torch.load(gxn_checkpoint, map_location=device, weights_only=False)
    hidden_dim = ckpt.get('hidden_dim', 128)
    max_hops   = ckpt.get('max_hops',   2)
    recovery_K = ckpt.get('recovery_K', 8)
    recovery_L = ckpt.get('recovery_L', 5)
    norm_mean  = ckpt.get('norm_mean',  0.0)
    norm_std   = ckpt.get('norm_std',   1.0)
    sampling_model = NeuralSampling(in_channels=1, hidden_channels=hidden_dim, max_hops=max_hops).to(device)
    recovery_model = NeuralRecovery(in_channels=1, K=recovery_K, L=recovery_L).to(device)
    sampling_model.load_state_dict(ckpt['sampling_model'])
    recovery_model.load_state_dict(ckpt['recovery_model'])
    sampling_model.eval(); recovery_model.eval()
    return sampling_model, recovery_model, norm_mean, norm_std


def gxn_reconstruct(sampling_model, recovery_model, norm_mean, norm_std,
                    x_orig, edge_index, keep_ratio, device):
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
    x_hat_physical = (x_hat_norm.squeeze() * std_t + mean_t).cpu()
    return x_hat_physical


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

    norm_stats = None
    norm_metadata_path = run_dir / 'normalization_metadata.json'
    if norm_metadata_path.exists():
        with open(str(norm_metadata_path), 'r') as f:
            norm_metadata = json.load(f)
        norm_stats_path = norm_metadata.get('normalization_stats_path')
        if norm_stats_path and Path(norm_stats_path).exists():
            norm_stats = torch.load(norm_stats_path, map_location='cpu', weights_only=False)

    ckpt_path = Path(gwt_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"NIGWT checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state  = {k.replace('module.', ''): v for k, v in state_dict.items()}

    test_data_dir   = Path(cfg['test_data_dir'])
    gwt_config_path = test_data_dir / 'gwt_config.json'
    num_filters = 5
    if gwt_config_path.exists():
        with open(str(gwt_config_path), 'r') as f:
            num_filters = json.load(f)['num_filters']

    use_normalization = cfg.get('use_normalization', False)
    model_name        = cfg.get('model_name', 'GNNNeuralIGWT')

    from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT
    if model_name == 'GUNetNeuralIGWT':
        model = GUNetNeuralIGWT(
            num_filters=num_filters, hidden_dim=cfg['hidden_dim'],
            output_dim=cfg['output_dim'], activation_function=cfg['activation_function'],
            encoder_type=cfg['encoder_type'], fusion_type=cfg.get('fusion_type', 'mlp'),
            decoder_type=cfg['decoder_type'], conv_layer_type=cfg['conv_layer_type'],
            processor_num_layers=cfg['processor_num_layers'],
            pooling_type=cfg.get('pooling_type', 'top_k_pooling'),
            pooling_ratio=cfg.get('pooling_ratio', 0.5),
            predict_residual=cfg['predict_residual'],
            enable_scale_embeddings=cfg['enable_scale_embeddings'],
            igwt_mode=cfg['igwt_mode'], use_normalization=use_normalization,
            normalization_stats=norm_stats,
        ).to(device)
    else:
        model = GNNNeuralIGWT(
            num_filters=num_filters, hidden_dim=cfg['hidden_dim'],
            output_dim=cfg['output_dim'], activation_function=cfg['activation_function'],
            encoder_type=cfg['encoder_type'], decoder_type=cfg['decoder_type'],
            conv_layer_type=cfg['conv_layer_type'],
            processor_num_layers=cfg['processor_num_layers'],
            predict_residual=cfg['predict_residual'],
            enable_scale_embeddings=cfg['enable_scale_embeddings'],
            igwt_mode=cfg['igwt_mode'], use_normalization=use_normalization,
            normalization_stats=norm_stats,
        ).to(device)

    model.load_state_dict(new_state)
    model.eval()
    return model, cfg, norm_stats


def gwt_reconstruct(model, cfg, norm_stats, coeffs_sample, edge_index,
                    matrix_cache, device, field_index=0):
    from models.sparsification_utils import multilevel_importance_sampling
    coeffs_flat    = coeffs_sample.to(device)
    keep_ratio     = cfg.get('keep_ratio', 0.01)
    mask           = multilevel_importance_sampling(coeffs=coeffs_flat, keep_ratio=keep_ratio)
    edge_index_dev = edge_index.to(device)
    batch_vec      = torch.zeros(coeffs_flat.shape[0], dtype=torch.long, device=device)
    ptr            = torch.tensor([0, coeffs_flat.shape[0]], device=device)
    chebyshev_data = matrix_cache.get([0])

    with torch.no_grad():
        x_hat = model(
            coeffs=coeffs_flat, sparsification_mask=mask,
            edge_index=edge_index_dev, batch=batch_vec,
            chebyshev_data=chebyshev_data, batch_ptr=ptr,
        )

    x_hat_cpu = x_hat.squeeze().cpu()
    use_normalization = cfg.get('use_normalization', False)
    if use_normalization and norm_stats is not None:
        field_names = list(norm_stats.keys())
        fname = field_names[field_index]
        gwt_m = norm_stats[fname]['mean']
        gwt_s = norm_stats[fname]['std']
        x_hat_cpu = x_hat_cpu * gwt_s + gwt_m
    return x_hat_cpu


# =============================================================================
# MSE Computation and Plotting
# =============================================================================

def plot_mse_curves(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("Loading test metadata...")
    sample_files, edge_index, num_nodes, num_rows, num_cols = \
        load_test_metadata(args.test_dir)
    num_samples = len(sample_files)
    print(f"  Found {num_samples} test samples")
    print(f"  Grid: {num_rows} x {num_cols} = {num_nodes} nodes\n")

    edge_index_dev = edge_index.to(device)

    # --- Load models ---
    vgae_model, vgae_mean, vgae_std = None, None, None
    if args.vgae_checkpoint and Path(args.vgae_checkpoint).exists():
        print("Loading Sparse-VGAE-99K model...")
        vgae_model, vgae_mean, vgae_std = load_vgae_model(
            args.vgae_checkpoint, args.keep_ratio, device)
        print(f"  VGAE-99K: mean={vgae_mean:.6f}, std={vgae_std:.6f}")
    else:
        print("Skipping Sparse-VGAE-99K (no checkpoint provided)")

    vgae50_model, vgae50_mean, vgae50_std = None, None, None
    if args.vgae_50k_checkpoint and Path(args.vgae_50k_checkpoint).exists():
        print("Loading Sparse-VGAE-50K model...")
        vgae50_model, vgae50_mean, vgae50_std = load_vgae_model(
            args.vgae_50k_checkpoint, args.keep_ratio, device)
        print(f"  VGAE-50K: mean={vgae50_mean:.6f}, std={vgae50_std:.6f}")
    else:
        print("Skipping Sparse-VGAE-50K (no checkpoint provided)")

    gxn_sampling, gxn_recovery, gxn_mean, gxn_std = None, None, None, None
    if args.gxn_checkpoint and Path(args.gxn_checkpoint).exists():
        print("\nLoading GXN model...")
        gxn_sampling, gxn_recovery, gxn_mean, gxn_std = load_gxn_model(
            args.gxn_checkpoint, device)
        print(f"  GXN: mean={gxn_mean:.6f}, std={gxn_std:.6f}")
    else:
        print("Skipping GXN (no checkpoint provided)")

    print("\nLoading NIGWT model...")
    gwt_model, gwt_cfg, gwt_norm_stats = load_gwt_model(
        args.gwt_run_dir, args.gwt_checkpoint, device)
    if gwt_norm_stats is not None:
        for field_name, stats in gwt_norm_stats.items():
            print(f"    {field_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    _sig0, _c0 = load_single_sample(sample_files[0], args.field_index)
    if _c0 is None:
        raise ValueError("NIGWT requires pre-computed coeffs in the .pt files.")
    del _sig0, _c0

    print("\nBuilding Chebyshev matrix cache...")
    from data.dataset_turbulent import GraphSignalDatasetTurbulent, create_matrix_cache_turbulent
    dataset_for_cache = GraphSignalDatasetTurbulent(
        base_data_dir=args.test_dir, field_index=args.field_index,
        preload_to_memory=False, use_normalization=False,
    )
    matrix_cache = create_matrix_cache_turbulent(dataset=dataset_for_cache, device=device)

    # --- Load classical baseline results ---
    classical_data = None
    if args.classical_results and Path(args.classical_results).exists():
        print(f"\nLoading classical baseline results from {args.classical_results}")
        with open(args.classical_results, 'r') as f:
            classical_data = json.load(f)
    else:
        print("Skipping classical baselines (no results file provided)")

    # --- Per-sample MSE accumulators ---
    mse_nigwt  = []
    mse_vgae   = []
    mse_vgae50 = []
    mse_gxn    = []
    mse_classical = {}
    if classical_data:
        for method_name in classical_data:
            mse_classical[method_name] = []

    # Welford online algorithm for signal_std (used to compute signal_var = signal_std²)
    _sig_n    = 0
    _sig_mean = 0.0
    _sig_M2   = 0.0

    print(f"\nComputing MSE in PHYSICAL UNITS for {num_samples} samples (lazy load)...\n")

    for sample_idx in tqdm(range(num_samples), desc="Processing", unit="sample"):
        x_orig, coeffs_sample = load_single_sample(
            sample_files[sample_idx], args.field_index)

        # --- Welford update ---
        _xnp     = x_orig.numpy()
        _n_new   = int(_xnp.size)
        _mn_new  = float(_xnp.mean())
        _var_new = float(_xnp.var())
        if _sig_n == 0:
            _sig_mean = _mn_new
            _sig_M2   = _var_new * _n_new
        else:
            _delta    = _mn_new - _sig_mean
            _sig_mean = (_sig_n * _sig_mean + _n_new * _mn_new) / (_sig_n + _n_new)
            _sig_M2  += _var_new * _n_new + (_delta ** 2) * _sig_n * _n_new / (_sig_n + _n_new)
        _sig_n += _n_new

        # --- NIGWT ---
        x_nigwt = gwt_reconstruct(
            gwt_model, gwt_cfg, gwt_norm_stats,
            coeffs_sample, edge_index, matrix_cache, device,
            field_index=args.field_index,
        )
        mse_nigwt.append(torch.mean((x_nigwt - x_orig) ** 2).item())

        # --- VGAE-99K ---
        if vgae_model is not None:
            x_vgae = vgae_reconstruct(vgae_model, vgae_mean, vgae_std,
                                       x_orig, edge_index_dev, device)
            mse_vgae.append(torch.mean((x_vgae - x_orig) ** 2).item())

        # --- VGAE-50K ---
        if vgae50_model is not None:
            x_vgae50 = vgae_reconstruct(vgae50_model, vgae50_mean, vgae50_std,
                                          x_orig, edge_index_dev, device)
            mse_vgae50.append(torch.mean((x_vgae50 - x_orig) ** 2).item())

        # --- GXN ---
        if gxn_sampling is not None:
            x_gxn = gxn_reconstruct(
                gxn_sampling, gxn_recovery, gxn_mean, gxn_std,
                x_orig, edge_index_dev, args.keep_ratio, device)
            mse_gxn.append(torch.mean((x_gxn - x_orig) ** 2).item())

        # --- Classical baselines ---
        # Convert physical RMSE → physical MSE  (rmse² = mse in physical units)
        # Or convert normalised MSE → physical MSE  (mse_norm * signal_std² = mse_physical)
        if classical_data:
            for method_name, method_data in classical_data.items():
                all_rmses_phys = method_data.get('all_rmses_physical', [])
                if all_rmses_phys and sample_idx < len(all_rmses_phys):
                    mse_classical[method_name].append(float(all_rmses_phys[sample_idx]) ** 2)
                else:
                    all_mses = method_data.get('all_mses', [])
                    sig_std  = float(method_data.get('signal_std', 1.0))
                    if sample_idx < len(all_mses):
                        # all_mses is normalised MSE → multiply by signal_var to get physical MSE
                        mse_classical[method_name].append(
                            float(all_mses[sample_idx]) * sig_std ** 2)

    # --- Signal std / variance ---
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

    signal_var = signal_std ** 2
    print(f"Signal variance (σ²): {signal_var:.6f}")

    # --- Plot settings ---
    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "font.size": 34, "axes.labelsize": 34, "axes.titlesize": 36,
        "legend.fontsize": 28, "xtick.labelsize": 28, "ytick.labelsize": 28,
        "xtick.major.size": 18, "xtick.major.width": 4,
        "xtick.minor.size": 10, "xtick.minor.width": 3,
        "ytick.major.size": 18, "ytick.major.width": 4,
        "ytick.minor.size": 10, "ytick.minor.width": 3,
        "xtick.major.pad": 4, "ytick.major.pad": 4,
        "axes.linewidth": 1.5, "lines.linewidth": 2.8,
    })

    classical_colors = {'fastgsss': '#17becf', 'bsgda': '#9467bd', 'rsbs': '#8c564b'}
    classical_labels = {'fastgsss': 'FastGSSS', 'bsgda': 'BSGDA', 'rsbs': 'RSBS'}
    sample_indices   = np.arange(1, num_samples + 1)

    active_models = [("GWCS", '#1f77b4', mse_nigwt)]
    if mse_vgae:   active_models.append(("SGAE-99K", '#ff7f0e', mse_vgae))
    if mse_vgae50: active_models.append(("SGAE-50K", '#e31a1c', mse_vgae50))
    if mse_gxn:    active_models.append(("GXN",      '#2ca02c', mse_gxn))

    def _make_plot(ax, x_idx, models_subset, classical_subset=None,
                   ylabel="MSE", add_ref_line=False):
        for lbl, col, arr in models_subset:
            n = min(len(x_idx), len(arr))
            ax.plot(x_idx[:n], arr[:n], linewidth=2.8, label=lbl, color=col)
        if classical_subset:
            for mname, marr in classical_subset.items():
                if marr:
                    n = min(len(x_idx), len(marr))
                    ax.plot(x_idx[:n], marr[:n], linewidth=2.2, linestyle='--',
                            label=classical_labels.get(mname, mname),
                            color=classical_colors.get(mname, None))
        if add_ref_line:
            ax.axhline(1.0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6,
                       label='Error = signal variance')
        ax.set_xlabel("Frame Index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        n_series = len(models_subset) + (sum(1 for v in classical_subset.values() if v)
                                         if classical_subset else 0) + (1 if add_ref_line else 0)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=min(n_series, 4), frameon=True, framealpha=0.95, borderaxespad=0)

    print("\nPlotting MSE curves (Physical Units)...")

    # ── Plot 1: all samples, neural only ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 10))
    _make_plot(ax, sample_indices, active_models, ylabel="MSE")
    fig.tight_layout()
    out = os.path.join(args.output_dir, "mse_ml_gwcs_all_samples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Plot 2: first 50, neural only ──────────────────────────────────────
    n_50  = min(50, num_samples)
    idx50 = np.arange(1, n_50 + 1)
    fig, ax = plt.subplots(figsize=(22, 10))
    _make_plot(ax, idx50, active_models, ylabel="MSE")
    fig.tight_layout()
    out = os.path.join(args.output_dir, "mse_50_samples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Plot 3: first N, neural + classical ────────────────────────────────
    if mse_classical:
        max_cls = max((len(v) for v in mse_classical.values() if v), default=0)
        n_cls   = min(args.num_classical_samples, max_cls, num_samples)
        idx_cls = np.arange(1, n_cls + 1)
        fig, ax = plt.subplots(figsize=(22, 10))
        _make_plot(ax, idx_cls, active_models, classical_subset=mse_classical, ylabel="MSE")
        fig.tight_layout()
        out = os.path.join(args.output_dir, "mse_with_classical_50.png")
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {out}")

    # --- Physical-unit summary ---
    print("\n" + "=" * 80)
    print("MSE Summary Statistics (Physical Units)")
    print("=" * 80)
    print(f"{'Model':<20} {'Mean MSE':<16} {'Std MSE':<16} {'Min MSE':<16} {'Max MSE':<16}")
    print("-" * 80)
    for lbl, _, arr in active_models:
        a = np.array(arr)
        print(f"{lbl:<20} {a.mean():<16.6f} {a.std():<16.6f} {a.min():<16.6f} {a.max():<16.6f}")
    for mname, marr in mse_classical.items():
        if marr:
            a = np.array(marr)
            print(f"{classical_labels.get(mname,mname):<20} {a.mean():<16.6f} {a.std():<16.6f} {a.min():<16.6f} {a.max():<16.6f}")
    print("=" * 80)

    # --- Save physical MSE to JSON ---
    results_dict = {}
    for lbl, _, arr in active_models:
        results_dict[lbl.lower().replace('-', '_')] = arr
    for mname, marr in mse_classical.items():
        if marr:
            results_dict[mname] = marr
    json_path = os.path.join(args.output_dir, "mse_per_sample.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved per-sample MSE (physical) to: {json_path}")

    # ==========================================================================
    # SIGNAL-VARIANCE-NORMALISED MSE PLOTS
    #
    #   norm_MSE = MSE / σ²_signal
    #
    #   norm_MSE = 0  →  perfect reconstruction (zero error)
    #   norm_MSE = 1  →  error variance equals signal variance
    #   norm_MSE > 1  →  model error larger than typical signal variation
    #
    #   Relationship to normalised RMSE:  norm_MSE = (norm_RMSE)²
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Generating Signal-Variance-Normalised MSE plots  (MSE ÷ σ²_signal)")
    print(f"  σ_signal  = {signal_std:.6f}")
    print(f"  σ²_signal = {signal_var:.6f}  (population variance over all {num_samples} test samples)")
    print("  norm_MSE = 0 → perfect;  norm_MSE = 1 → error variance = signal variance")
    print("=" * 80)

    if signal_var < 1e-24:
        print("  WARNING: signal_var ≈ 0 — skipping normalised plots.")
        return

    def _sig_norm_mse(arr):
        return [v / signal_var for v in arr]

    for lbl, _, r in active_models:
        print(f"    {lbl}: mean MSE={np.mean(r):.6f}  →  norm={np.mean(r)/signal_var:.4f}")
    for mname, marr in mse_classical.items():
        if marr:
            print(f"    {classical_labels.get(mname,mname)}: mean MSE={np.mean(marr):.6f}"
                  f"  →  norm={np.mean(marr)/signal_var:.4f}")

    ylabel_norm = r"Normalised MSE  ($\div\,\sigma^2_{\rm signal}$)"

    # ── Norm Plot 1: all samples, neural only ─────────────────────────────
    print(f"\n  Generating Norm Plot 1: ML + GWCS, all {num_samples} samples...")
    fig, ax = plt.subplots(figsize=(22, 10))
    _make_plot(ax, sample_indices,
               [(lbl, col, _sig_norm_mse(r)) for lbl, col, r in active_models],
               ylabel=ylabel_norm, add_ref_line=True)
    fig.tight_layout()
    out = os.path.join(args.output_dir, "mse_normalised_all_samples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Norm Plot 2: first 50, neural only ────────────────────────────────
    n_50_norm = min(50, num_samples)
    idx_50_norm = np.arange(1, n_50_norm + 1)
    print(f"\n  Generating Norm Plot 2: ML + GWCS, first {n_50_norm} samples...")
    fig, ax = plt.subplots(figsize=(22, 10))
    _make_plot(ax, idx_50_norm,
               [(lbl, col, _sig_norm_mse(r[:n_50_norm])) for lbl, col, r in active_models],
               ylabel=ylabel_norm, add_ref_line=True)
    fig.tight_layout()
    out = os.path.join(args.output_dir, "mse_normalised_50_samples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")

    # ── Norm Plot 3: first N, neural + classical ──────────────────────────
    if mse_classical:
        max_cls_n = max((len(v) for v in mse_classical.values() if v), default=0)
        n_cls_n   = min(args.num_classical_samples, max_cls_n, num_samples)
        idx_cls_n = np.arange(1, n_cls_n + 1)
        print(f"\n  Generating Norm Plot 3: ML + GWCS + Classical, first {n_cls_n} samples...")
        norm_classical = {k: _sig_norm_mse(v[:n_cls_n]) for k, v in mse_classical.items() if v}
        fig, ax = plt.subplots(figsize=(22, 10))
        _make_plot(ax, idx_cls_n,
                   [(lbl, col, _sig_norm_mse(r[:n_cls_n])) for lbl, col, r in active_models],
                   classical_subset=norm_classical,
                   ylabel=ylabel_norm, add_ref_line=True)
        fig.tight_layout()
        out = os.path.join(args.output_dir, "mse_normalised_with_classical_50.png")
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {out}")

    # --- Normalised summary table ---
    print("\n" + "=" * 80)
    print(f"Signal-Variance-Normalised MSE Summary  (MSE ÷ σ²,  σ² = {signal_var:.6f})")
    print("  norm = 0 → perfect;  norm = 1 → error variance = signal variance")
    print("=" * 80)
    print(f"{'Model':<20} {'Mean norm':>14} {'Std norm':>14}")
    print("-" * 50)
    for lbl, _, arr in active_models:
        a = np.array(arr)
        print(f"{lbl:<20} {a.mean()/signal_var:>14.6f} {a.std()/signal_var:>14.6f}")
    for mname, marr in mse_classical.items():
        if marr:
            a = np.array(marr)
            print(f"{classical_labels.get(mname,mname):<20} {a.mean()/signal_var:>14.6f} {a.std()/signal_var:>14.6f}")
    print("=" * 80)

    # --- Save normalised MSE to JSON ---
    norm_dict = {'signal_std': signal_std, 'signal_var': signal_var,
                 'normalization': 'mse_div_signal_var'}
    for lbl, _, arr in active_models:
        norm_dict[lbl.lower().replace('-', '_')] = _sig_norm_mse(arr)
    for mname, marr in mse_classical.items():
        if marr:
            norm_dict[mname] = _sig_norm_mse(marr)
    norm_json_path = os.path.join(args.output_dir, "mse_normalised_per_sample.json")
    with open(norm_json_path, 'w') as f:
        json.dump(norm_dict, f, indent=2)
    print(f"\nSaved signal-variance-normalised per-sample MSE to: {norm_json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot MSE (physical + signal-variance-normalised) for all models")

    parser.add_argument('--test-dir',        required=True)
    parser.add_argument('--gwt-checkpoint',  required=True)
    parser.add_argument('--gwt-run-dir',     required=True)
    parser.add_argument('--vgae-checkpoint',     default=None)
    parser.add_argument('--vgae-50k-checkpoint', default=None)
    parser.add_argument('--gxn-checkpoint',      default=None)
    parser.add_argument('--classical-results',   default=None)
    parser.add_argument('--field-index',          type=int,   default=0)
    parser.add_argument('--keep-ratio',           type=float, default=0.01)
    parser.add_argument('--num-classical-samples',type=int,   default=50)
    parser.add_argument('--signal-std',           type=float, default=None,
                        help="Signal std override (computed from test data if omitted)")
    parser.add_argument('--output-dir', default='baseline_results/mse_plots')

    args = parser.parse_args()
    plot_mse_curves(args)


if __name__ == "__main__":
    main()
