# -*- coding: utf-8 -*-
"""
Sparse VGAE Baseline Training on turbulent data.
Encodes raw signal to latent, applies sparsity, decodes back.
Single GPU - the graph is shared, only signal varies per sample.

CSV columns (vgae_losses.csv):
  epoch, train_total, train_recon, val_total, val_recon
  + physical_train_rmse, physical_val_rmse
  (physical_* = sqrt(normalized_mse) * signal_std, same units as NIGWT)
  val_* and physical_val_* only written every val_freq epochs.

Saves visualizations for 8 fixed test samples.
"""
import sys, os, json, argparse, csv
import torch
import torch.nn.functional as F
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.sparse_VGAE.gae import GAE


def load_turbulent_data(split_dir: str, field_index: int = 0, max_samples=None):
    split_dir = Path(split_dir)
    shared = torch.load(split_dir / "shared_data.pt", map_location='cpu', weights_only=False)
    edge_index = shared['edge_index']
    num_nodes  = shared['num_nodes']
    num_rows   = shared['num_rows']
    num_cols   = shared['num_columns']

    sample_files = sorted(
        [f for f in split_dir.glob("*.pt") if f.stem.isdigit()],
        key=lambda f: int(f.stem)
    )
    if max_samples:
        sample_files = sample_files[:max_samples]

    signals = []
    for f in sample_files:
        s = torch.load(f, map_location='cpu', weights_only=False)
        # .float() handles both float16 (half-precision storage) and float32
        signals.append(s.x[:, field_index].float())

    signals = torch.stack(signals, dim=1)  # (N, num_samples)
    return signals, edge_index, num_nodes, num_rows, num_cols


def compute_normalization_stats(signals: torch.Tensor):
    """Compute mean and std from training signals for z-score normalization."""
    mean = signals.mean()
    std  = signals.std()
    if std < 1e-8:
        std = torch.tensor(1.0)
    return mean, std


def compute_val_loss(model, val_signals_norm, edge_index, device,
                     variational=False, num_val=50):
    """Compute validation loss on a subset of val samples."""
    model.eval()
    n = min(num_val, val_signals_norm.shape[1])
    indices = torch.randperm(val_signals_norm.shape[1])[:n]
    losses = []
    with torch.no_grad():
        for i in indices:
            x_batch = val_signals_norm[:, i].unsqueeze(1).to(device)
            z_sparse = model.encode(x_batch, edge_index)
            x_hat = model.decode(z_sparse, edge_index)
            if variational:
                loss, _, _ = model.elbo_loss(x_batch, x_hat, beta=1.0)
            else:
                loss = model.recon_loss(x_batch, x_hat)
            losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    # For VGAE, total loss == recon loss (no separate sample loss term)
    return avg, avg


def train_sparse_vgae(
    train_dir: str,
    test_dir: str,
    field_index: int = 0,
    keep_ratio: float = 0.01,
    epochs: int = 500,
    lr: float = 1e-4,
    latent_dim: int = 128,
    hidden_layers: int = 4,
    sparsity_mode: str = 'ste_annealing',
    variational: bool = False,
    max_samples: int = None,
    output_dir: str = "baseline_results/sparse_vgae",
    val_freq: int = 10,
    vis_sample_indices: list = None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if vis_sample_indices is None:
        vis_sample_indices = list(range(8))

    # Load data
    print("Loading training data...")
    train_signals, edge_index, num_nodes, num_rows, num_cols = \
        load_turbulent_data(train_dir, field_index, max_samples)

    # Z-score normalize from train stats
    mean, std = compute_normalization_stats(train_signals)
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")
    train_signals_norm = (train_signals - mean) / std
    train_signals_norm = train_signals_norm.to(device)
    edge_index = edge_index.to(device)

    # Load test signals for validation proxy
    print("Loading test data for validation...")
    test_signals_raw, _, _, _, _ = load_turbulent_data(test_dir, field_index, max_samples)
    test_signals_norm = (test_signals_raw - mean) / std
    test_signals_norm_dev = test_signals_norm.to(device)

    print(f"Nodes: {num_nodes}, keep_ratio={keep_ratio}")

    model = GAE(
        input_dim=1,
        latent_dim=latent_dim,
        output_dim=1,
        num_encoder_layers=hidden_layers,
        num_decoder_layers=hidden_layers,
        conv_layer_type='SAGEConv',
        activation_function='silu',
        sparsity_ratio=keep_ratio,
        variational=variational,
        sparsity_mode=sparsity_mode,
        temperature_init=1.0,
        min_temperature=0.01,
        annealing_steps=max(1, epochs // 2),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Sparse-VGAE: latent_dim={latent_dim}, hidden_layers={hidden_layers} → {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    num_train = train_signals_norm.shape[1]
    best_loss = float('inf')

    # CSV for loss tracking - one row per epoch
    # val_total and val_recon computed every val_freq epochs, empty string otherwise
    csv_path = os.path.join(output_dir, 'vgae_losses.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # physical_*_rmse = sqrt(normalized_mse) * signal_std  → same physical units as NIGWT
    csv_writer.writerow(['epoch', 'train_total', 'train_recon', 'val_total', 'val_recon',
                         'physical_train_rmse', 'physical_val_rmse'])

    print(f"\nTraining Sparse-VGAE for {epochs} epochs on {num_train} samples...")

    for epoch in range(epochs):
        model.train()
        idx = torch.randint(0, num_train, (1,)).item()
        x_batch = train_signals_norm[:, idx].unsqueeze(1)

        optimizer.zero_grad()
        z_sparse = model.encode(x_batch, edge_index)
        x_hat = model.decode(z_sparse, edge_index)

        if variational:
            loss, recon_loss, _ = model.elbo_loss(x_batch, x_hat, beta=1.0)
        else:
            loss = model.recon_loss(x_batch, x_hat)
            recon_loss = loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Temperature annealing step
        if hasattr(model.sparsity_mask, 'update_temperature'):
            model.sparsity_mask.update_temperature()

        scheduler.step(loss.item())

        train_total = loss.item()
        train_recon = recon_loss.item()

        # physical RMSE for train batch: sqrt(normalized_mse) * std
        physical_train_rmse = (train_recon ** 0.5) * std.item()

        # Compute validation loss every val_freq epochs; write empty otherwise
        if (epoch + 1) % val_freq == 0:
            val_total, val_recon = compute_val_loss(
                model, test_signals_norm_dev, edge_index, device, variational
            )
            physical_val_rmse = (val_recon ** 0.5) * std.item()
            csv_writer.writerow([epoch + 1, train_total, train_recon, val_total, val_recon,
                                  physical_train_rmse, physical_val_rmse])
        else:
            csv_writer.writerow([epoch + 1, train_total, train_recon, '', '',
                                  physical_train_rmse, ''])
        csv_file.flush()

        if (epoch + 1) % 30 == 0:
            stats = model.get_sparsity_stats()
            print(f"Epoch {epoch+1}/{epochs} | Loss={train_total:.6f} | "
                  f"Sparsity={stats['sparsity_ratio_actual']:.4f} | "
                  f"Temp={stats['temperature']:.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'loss': best_loss,
                'norm_mean': mean.item(),
                'norm_std': std.item(),
                'latent_dim': latent_dim,
                'hidden_layers': hidden_layers,
            }, os.path.join(output_dir, 'best_checkpoint.pt'))

    csv_file.close()
    print(f"Loss CSV saved to: {csv_path}")

    # Test evaluation
    print("\nRunning test evaluation...")
    ckpt = torch.load(os.path.join(output_dir, 'best_checkpoint.pt'), map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    test_mses = []
    with torch.no_grad():
        for i in range(test_signals_norm.shape[1]):
            x_test = test_signals_norm[:, i].unsqueeze(1).to(device)
            z_sparse = model.encode(x_test, edge_index)
            x_hat = model.decode(z_sparse, edge_index)
            mse = F.mse_loss(x_hat.squeeze(), x_test.squeeze()).item()
            test_mses.append(mse)

    mean_mse = sum(test_mses) / len(test_mses)
    # Convert normalized MSE → physical RMSE (same units as NIGWT test metrics)
    test_rmses_physical = [(m ** 0.5) * std.item() for m in test_mses]
    mean_rmse_physical  = sum(test_rmses_physical) / len(test_rmses_physical)
    print(f"\nSparse-VGAE Test MSE (normalized): {mean_mse:.6f}")
    print(f"Sparse-VGAE Test RMSE (physical):  {mean_rmse_physical:.6f}")
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump({'mean_mse': mean_mse, 'all_mses': test_mses,
                   'mean_rmse_physical': mean_rmse_physical,
                   'all_rmses_physical': test_rmses_physical,
                   'norm_mean': mean.item(), 'norm_std': std.item()}, f)

    # Save visualizations for fixed 8 sample indices (physical units)
    print(f"\nSaving visualizations for samples: {vis_sample_indices}")
    from visualization.gwt_coeffs import visualize_grid_graph_inverse_sgwt_reconstruction
    with torch.no_grad():
        for sample_idx in vis_sample_indices:
            if sample_idx >= test_signals_raw.shape[1]:
                print(f"  Skipping sample {sample_idx}: out of range")
                continue
            x_vis_norm = test_signals_norm[:, sample_idx].unsqueeze(1).to(device)
            z_sparse = model.encode(x_vis_norm, edge_index)
            x_hat_vis_norm = model.decode(z_sparse, edge_index)

            # Denormalize for visualization - physical units
            x_vis_orig = test_signals_raw[:, sample_idx].cpu()
            x_hat_vis_orig = x_hat_vis_norm.squeeze().cpu() * std + mean

            visualize_grid_graph_inverse_sgwt_reconstruction(
                x_recon=x_hat_vis_orig,
                x=x_vis_orig,
                rows=num_rows,
                cols=num_cols,
                save_fig=True,
                filename_prefix=f'sparse_vgae_sample{sample_idx}',
                save_dir=output_dir,
            )

    return mean_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--field-index', type=int, default=0)
    parser.add_argument('--keep-ratio', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--hidden-layers', type=int, default=4)
    parser.add_argument('--sparsity-mode', type=str, default='ste_annealing',
                        choices=['ste_fixed', 'ste_annealing', 'gumbel'])
    parser.add_argument('--variational', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='baseline_results/sparse_vgae')
    parser.add_argument('--val-freq', type=int, default=10,
                        help='Validate every N epochs and write val loss to CSV row')
    parser.add_argument('--vis-sample-indices', type=int, nargs='+',
                        default=list(range(8)),
                        help='Test sample indices to visualize after training')
    args = parser.parse_args()
    train_sparse_vgae(**vars(args))