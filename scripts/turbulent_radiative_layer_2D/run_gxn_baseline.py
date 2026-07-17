# -*- coding: utf-8 -*-
"""
GXN Baseline Training: NeuralSampling + NeuralRecovery on turbulent data.
Trains on processed .pt files (single field, single GPU only).

CSV columns (gxn_losses.csv):
  epoch, train_total, train_recon, val_total, val_recon
  + physical_train_rmse, physical_val_rmse
  (physical_* = sqrt(normalized_recon_mse) * signal_std, same units as NIGWT)
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

from baselines.gxn.neural_sampling import NeuralSampling, NeuralRecovery


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
    mean = signals.mean()
    std  = signals.std()
    if std < 1e-8:
        std = torch.tensor(1.0)
    return mean, std


def compute_val_loss(sampling_model, recovery_model, val_signals_norm,
                     edge_index, M, alpha, device, num_val=50):
    """Compute validation loss on a subset of val samples."""
    sampling_model.eval()
    recovery_model.eval()
    n = min(num_val, val_signals_norm.shape[1])
    indices = torch.randperm(val_signals_norm.shape[1])[:n]
    total_losses, recon_losses = [], []
    with torch.no_grad():
        for i in indices:
            x_batch = val_signals_norm[:, i].unsqueeze(1).to(device)
            attention, _, _ = sampling_model(x_batch, edge_index)
            scores = attention.squeeze()
            _, top_indices = torch.topk(scores, k=M)
            mask_psi = torch.zeros_like(x_batch)
            mask_psi[top_indices] = 1.0
            y = mask_psi * attention * x_batch
            x_hat = recovery_model(y, edge_index, attention, mask_psi)
            loss_rec = F.mse_loss(x_hat, x_batch)
            loss_sample = sampling_model.compute_loss(x_batch, edge_index)
            total_loss = loss_rec + alpha * loss_sample
            total_losses.append(total_loss.item())
            recon_losses.append(loss_rec.item())
    sampling_model.train()
    recovery_model.train()
    return (sum(total_losses) / len(total_losses),
            sum(recon_losses) / len(recon_losses))


def train_gxn(
    train_dir: str,
    test_dir: str,
    field_index: int = 0,
    keep_ratio: float = 0.01,
    epochs: int = 400,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    max_hops: int = 2,
    recovery_K: int = 8,
    recovery_L: int = 5,
    alpha: float = 0.05,
    max_samples: int = None,
    output_dir: str = "baseline_results/gxn",
    val_freq: int = 10,
    vis_sample_indices: list = None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if vis_sample_indices is None:
        vis_sample_indices = list(range(8))

    # Load training data
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

    M = max(1, int(keep_ratio * num_nodes))
    print(f"Nodes: {num_nodes}, Sample size M={M} ({keep_ratio*100:.1f}%)")

    # Build models
    sampling_model = NeuralSampling(
        in_channels=1,
        hidden_channels=hidden_dim,
        max_hops=max_hops,
    ).to(device)

    recovery_model = NeuralRecovery(
        in_channels=1,
        K=recovery_K,
        L=recovery_L,
    ).to(device)

    n_params_sampling = sum(p.numel() for p in sampling_model.parameters() if p.requires_grad)
    n_params_recovery = sum(p.numel() for p in recovery_model.parameters() if p.requires_grad)
    n_params_total    = n_params_sampling + n_params_recovery
    print(f"✓ GXN: hidden_dim={hidden_dim}, max_hops={max_hops}, K={recovery_K}, L={recovery_L}")
    print(f"       NeuralSampling={n_params_sampling:,}  NeuralRecovery={n_params_recovery:,}  Total={n_params_total:,} parameters")

    params = list(sampling_model.parameters()) + list(recovery_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    num_train = train_signals_norm.shape[1]
    best_loss = float('inf')

    # CSV for loss tracking - one row per epoch
    # val_total and val_recon are computed every val_freq epochs, NaN otherwise
    csv_path = os.path.join(output_dir, 'gxn_losses.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # physical_*_rmse = sqrt(normalized_recon_mse) * signal_std → same physical units as NIGWT
    csv_writer.writerow(['epoch', 'train_total', 'train_recon', 'val_total', 'val_recon',
                         'physical_train_rmse', 'physical_val_rmse'])

    print(f"\nTraining GXN for {epochs} epochs on {num_train} samples...")

    # Cache last computed val losses so we can carry them forward if needed
    # but we write NaN for non-val epochs to keep it honest
    for epoch in range(epochs):
        sampling_model.train()
        recovery_model.train()

        idx = torch.randint(0, num_train, (1,)).item()
        x_batch = train_signals_norm[:, idx].unsqueeze(1)

        optimizer.zero_grad()

        attention, _, _ = sampling_model(x_batch, edge_index)
        scores = attention.squeeze()
        _, top_indices = torch.topk(scores, k=M)
        mask_psi = torch.zeros_like(x_batch)
        mask_psi[top_indices] = 1.0
        y = mask_psi * attention * x_batch
        x_hat = recovery_model(y, edge_index, attention, mask_psi)

        loss_rec = F.mse_loss(x_hat, x_batch)
        loss_sample = sampling_model.compute_loss(x_batch, edge_index)
        total_loss = loss_rec + alpha * loss_sample
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss.item())

        train_total = total_loss.item()
        train_recon = loss_rec.item()

        # physical RMSE for train: sqrt(normalized recon mse) * std
        physical_train_rmse = (train_recon ** 0.5) * std.item()

        # Compute validation loss every val_freq epochs; write empty otherwise
        if (epoch + 1) % val_freq == 0:
            val_total, val_recon = compute_val_loss(
                sampling_model, recovery_model,
                test_signals_norm_dev, edge_index, M, alpha, device
            )
            physical_val_rmse = (val_recon ** 0.5) * std.item()
            csv_writer.writerow([epoch + 1, train_total, train_recon, val_total, val_recon,
                                  physical_train_rmse, physical_val_rmse])
        else:
            csv_writer.writerow([epoch + 1, train_total, train_recon, '', '',
                                  physical_train_rmse, ''])
        csv_file.flush()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss={train_total:.6f} "
                  f"(Rec={train_recon:.6f}, Sample={loss_sample.item():.6f})")

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'sampling_model': sampling_model.state_dict(),
                'recovery_model': recovery_model.state_dict(),
                'loss': best_loss,
                'norm_mean': mean.item(),
                'norm_std': std.item(),
                'hidden_dim': hidden_dim,
                'max_hops': max_hops,
                'recovery_K': recovery_K,
                'recovery_L': recovery_L,
            }, os.path.join(output_dir, 'best_checkpoint.pt'))

    csv_file.close()
    print(f"Loss CSV saved to: {csv_path}")

    # Test evaluation
    print("\nRunning test evaluation...")
    ckpt = torch.load(os.path.join(output_dir, 'best_checkpoint.pt'), map_location=device)
    sampling_model.load_state_dict(ckpt['sampling_model'])
    recovery_model.load_state_dict(ckpt['recovery_model'])
    sampling_model.eval()
    recovery_model.eval()

    test_mses = []
    with torch.no_grad():
        for i in range(test_signals_norm.shape[1]):
            x_test = test_signals_norm[:, i].unsqueeze(1).to(device)
            attention, _, _ = sampling_model(x_test, edge_index)
            _, top_indices = torch.topk(attention.squeeze(), k=M)
            mask_psi = torch.zeros_like(x_test)
            mask_psi[top_indices] = 1.0
            y = mask_psi * attention * x_test
            x_hat = recovery_model(y, edge_index, attention, mask_psi)
            mse = F.mse_loss(x_hat.squeeze(), x_test.squeeze()).item()
            test_mses.append(mse)

    mean_mse = sum(test_mses) / len(test_mses)
    # Convert normalized MSE → physical RMSE (same units as NIGWT test metrics)
    test_rmses_physical = [(m ** 0.5) * std.item() for m in test_mses]
    mean_rmse_physical  = sum(test_rmses_physical) / len(test_rmses_physical)
    print(f"\nGXN Test MSE (normalized): {mean_mse:.6f}")
    print(f"GXN Test RMSE (physical):  {mean_rmse_physical:.6f}")
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
            attention, _, _ = sampling_model(x_vis_norm, edge_index)
            _, top_indices = torch.topk(attention.squeeze(), k=M)
            mask_psi = torch.zeros_like(x_vis_norm)
            mask_psi[top_indices] = 1.0
            y = mask_psi * attention * x_vis_norm
            x_hat_vis_norm = recovery_model(y, edge_index, attention, mask_psi)

            # Denormalize for visualization - physical units
            x_vis_orig = test_signals_raw[:, sample_idx].cpu()
            x_hat_vis_orig = x_hat_vis_norm.squeeze().cpu() * std + mean

            visualize_grid_graph_inverse_sgwt_reconstruction(
                x_recon=x_hat_vis_orig,
                x=x_vis_orig,
                rows=num_rows,
                cols=num_cols,
                save_fig=True,
                filename_prefix=f'gxn_sample{sample_idx}',
                save_dir=output_dir,
            )

    return mean_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--field-index', type=int, default=0)
    parser.add_argument('--keep-ratio', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--max-hops', type=int, default=2)
    parser.add_argument('--recovery-K', type=int, default=8)
    parser.add_argument('--recovery-L', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='baseline_results/gxn')
    parser.add_argument('--val-freq', type=int, default=10,
                        help='Validate every N epochs and write val loss to CSV row')
    parser.add_argument('--vis-sample-indices', type=int, nargs='+',
                        default=list(range(8)),
                        help='Test sample indices to visualize after training')
    args = parser.parse_args()
    train_gxn(**vars(args))