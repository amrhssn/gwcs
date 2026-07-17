import os
import time
from datetime import datetime
from typing import Optional

import torch

from paths import ROOT_PROJECT_PATH
from utils.random import set_random_seed
from data.data_types import SparseGAEConfig
from data.dataset import (
    GraphSignalDataset,
    create_data_splits,
    create_dataloaders
)
from data.utils_io import (
    save_config,
    save_json
)
from models.sparsification_utils import SparsityAnnealer
from baselines.sparse_VGAE.gae import GAE


random_seed = 13
set_random_seed(random_seed=random_seed)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(
            self,
            patience=50,
            min_delta=0.0,
            verbose=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.is_best_epoch = False

    def __call__(self, score):
        """Call this method with validation loss after each epoch."""
        self.is_best_epoch = False
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.is_best_epoch = True
            if self.verbose:
                print(f"  → Initial best score: {score:.6f}")
            return
        # Check if score improved
        improved = score < (self.best_score - self.min_delta)
        if improved:
            if self.verbose:
                print(f"  → Validation improved: {self.best_score:.6f} → {score:.6f}")
            self.best_score = score
            self.counter = 0
            self.is_best_epoch = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def get_pretraining_configs(base_data_dir: str) -> SparseGAEConfig:
    # data config
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    batch_size = 16
    keep_ratio = 0.01

    mask_generation_mode = "weighted_random_per_scale"
    # scheduler config
    sparsity_initial_value = 1.0
    sparsity_final_value = keep_ratio
    sparsity_scheduler_final_epoch = 0
    sparsity_scheduler_strategy = 'cosine'
    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 50
    lr_scheduler_min_lr = 1e-6
    early_stopping_patience = 300
    early_stopping_patience_min_delta = 1e-4
    # model config
    model_name = 'VGAE'  # 'GAE', 'VGAE'
    if model_name == 'VGAE':
        variational = True
    else:
        variational = False
    sparsity_mask_mode = 'gumbel'  # 'ste_fixed', 'ste_annealing', 'gumbel'
    processor_num_layers = 5
    output_dim = 1
    latent_dim = 256
    activation_function = 'relu'
    conv_layer_type = 'SAGEConv'
    beta_kl = 0.0001
    # optimizer config
    lr = 1e-3
    weight_decay = 5e-4
    # training config
    epochs = 5_000
    print(f"keep ratio (5 GW filters, {keep_ratio} of wavelet coeffs): ")
    keep_ratio_new = (5 / latent_dim) * keep_ratio
    print(f"keep_ratio = (num_filters=5 / latent_dim={latent_dim}) * keep_ratio={keep_ratio} = "
          f"new keep_ratio={keep_ratio_new}")
    sparse_gae_config = SparseGAEConfig(
        base_data_dir=base_data_dir,
        random_seed=random_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        keep_ratio=keep_ratio_new,
        sparsity_initial_value=sparsity_initial_value,
        sparsity_final_value=sparsity_final_value,
        sparsity_scheduler_final_epoch=sparsity_scheduler_final_epoch,
        sparsity_scheduler_strategy=sparsity_scheduler_strategy,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        early_stopping_patience=early_stopping_patience,
        early_stopping_patience_min_delta=early_stopping_patience_min_delta,
        model_name=model_name,
        processor_num_layers=processor_num_layers,
        output_dim=output_dim,
        latent_dim=latent_dim,
        activation_function=activation_function,
        conv_layer_type=conv_layer_type,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        variational=variational,
        sparsity_mask_mode=sparsity_mask_mode,
        beta_kl=beta_kl,
    )
    return sparse_gae_config


def forward_step(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        mode: str,
        configs: SparseGAEConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
):
    if mode == 'train':
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training mode")
        model.train()
    elif mode == 'eval':
        model.eval()
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")
    # Context manager for gradient computation
    context = torch.no_grad() if mode == 'eval' else torch.enable_grad()

    batch_total_losses = []
    batch_recon_losses = []
    batch_kl_losses = []
    with context:
        for batch_index, batch in enumerate(data_loader):
            batch = batch.to(device)

            # Forward pass
            z_sparse = model.encode(batch.x, batch.edge_index, batch.batch)
            x_hat = model.decode(z_sparse, batch.edge_index)

            # Compute loss
            if configs.variational:
                loss, recon, kl = model.elbo_loss(batch.x, x_hat, batch.batch, beta=configs.beta_kl)
            else:
                loss = model.recon_loss(batch.x, x_hat, batch.batch)
                recon = loss
                kl = torch.tensor(0.0).to(device)

            # Store losses
            batch_total_losses.append(loss.item())
            batch_recon_losses.append(recon.item())
            batch_kl_losses.append(kl.item())

            # Backward pass and optimization (only in training mode)
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                # Add gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        model.sparsity_mask.update_temperature()
        stats = model.get_sparsity_stats()

    # Compute epoch averages
    epoch_total_loss = sum(batch_total_losses) / len(batch_total_losses)
    epoch_recon_loss = sum(batch_recon_losses) / len(batch_recon_losses)
    epoch_kl_loss = sum(batch_kl_losses) / len(batch_kl_losses)

    forward_metrics = {
        'total_loss': epoch_total_loss,
        'recon_loss': epoch_recon_loss,
        'kl_loss': epoch_kl_loss,
        'sparsity_stats': stats,
    }
    return forward_metrics


def main():
    # Data directories
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    base_data_dir = os.path.join(local_storage_dir, 'data', '2025-10-25_11-45-48')
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(local_storage_dir, 'runs', run_timestamp)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Pretraining configs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    configs = get_pretraining_configs(base_data_dir=base_data_dir)

    save_config(config=configs,
                config_name='sparse_gae_pretraining_config',
                save_dir=run_dir)

    # Create dataset
    dataset = GraphSignalDataset(base_data_dir=base_data_dir)

    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset=dataset,
        train_ratio=configs.train_ratio,
        val_ratio=configs.val_ratio,
        test_ratio=configs.test_ratio,
        seed=random_seed
    )
    save_json(
        data={
            "train": train_dataset.indices,
            "val": val_dataset.indices,
            "test": test_dataset.indices,
        },
        filename='split_indices.json',
        save_dir=run_dir
    )

    # Create dataloaders
    sparsity_scheduler = SparsityAnnealer(
        initial_value=configs.sparsity_initial_value,
        final_value=configs.sparsity_final_value,
        num_epochs=configs.sparsity_scheduler_final_epoch,
        strategy=configs.sparsity_scheduler_strategy
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=configs.batch_size,
        keep_ratio=configs.keep_ratio,
        sparsity_scheduler=sparsity_scheduler
    )

    # Initialize model
    model = GAE(
        input_dim=dataset[0].x.size(1),
        latent_dim=configs.latent_dim,
        output_dim=configs.output_dim,
        num_encoder_layers=configs.processor_num_layers,
        num_decoder_layers=configs.processor_num_layers,
        conv_layer_type=configs.conv_layer_type,
        activation_function=configs.activation_function,
        sparsity_ratio=configs.keep_ratio,
        variational=configs.variational,
        sparsity_mode=configs.sparsity_mask_mode
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=configs.lr_scheduler_factor,
        patience=configs.lr_scheduler_patience,
        min_lr=configs.lr_scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=configs.early_stopping_patience,
        min_delta=configs.early_stopping_patience_min_delta,
        verbose=True
    )

    # Training
    best_epoch = -1
    best_loss = float('inf')
    best_model_state = None
    best_checkpoint_path = None

    # Loss history tracking for visualization
    train_losses_history = {
        'total': [],
        'recon': [],
        'kl': []
    }
    val_losses_history = {
        'total': [],
        'recon': [],
        'kl': []
    }
    # Training loop
    for epoch in range(configs.epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # ============ Training ============
        train_metrics = forward_step(
            model=model,
            data_loader=train_loader,
            mode='train',
            configs=configs,
            optimizer=optimizer,
            device=device
        )

        # ============ Validation ============
        val_metrics = forward_step(
            model=model,
            data_loader=val_loader,
            mode='eval',
            configs=configs,
            optimizer=None,
            device=device
        )

        # ============ Track Loss History ============
        train_losses_history['total'].append(train_metrics['total_loss'])
        train_losses_history['recon'].append(train_metrics['recon_loss'])
        train_losses_history['kl'].append(train_metrics['kl_loss'])

        val_losses_history['total'].append(val_metrics['total_loss'])
        val_losses_history['recon'].append(val_metrics['recon_loss'])
        val_losses_history['kl'].append(val_metrics['kl_loss'])

        # calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # ============ Print Epoch Summary ============
        print(f"[*] Epoch {epoch:5d} | "
              f"Train Loss: {train_metrics['total_loss']:.6f} "
              f"(R: {train_metrics['recon_loss']:.6f}, "                            
              f"Val Loss: {val_metrics['total_loss']:.6f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {current_lr:.2e}")

        # ============ Update Schedulers ============
        lr_scheduler.step(val_metrics['total_loss'])
        early_stopping(val_metrics['total_loss'])

        # ============ Save Best Model ============
        if early_stopping.is_best_epoch:
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
            }, checkpoint_path)
            print(f"  → Model checkpoint saved to {checkpoint_path}")
            best_checkpoint_path = checkpoint_path
            best_epoch = epoch

        # ============ Check Early Stopping ============
        if early_stopping.should_stop:
            print("=" * 80)
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {early_stopping.best_score:.6f}")
            break

    # saving losses history
    save_json(data=train_losses_history, filename='train_loss.json', save_dir=run_dir)
    save_json(data=val_losses_history, filename='val_loss.json', save_dir=run_dir)

    if best_checkpoint_path is not None:
        print("=" * 80)
        print(f"Loading best model from epoch {best_epoch}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded (Val Loss: {checkpoint['val_loss']:.6f})")
        print("=" * 80)
    else:
        print("\n[!] Warning: No checkpoint saved. Using final model state for testing.")

    # ============ Test the Model ============
    print("\n[*] Running test evaluation...")
    test_metrics = forward_step(
        model=model,
        data_loader=test_loader,
        mode='eval',
        configs=configs,
        optimizer=None,
        device=device
    )

    print(f"\n{'=' * 80}")
    print("TEST RESULTS:")
    print(f"  Test Loss (Total):              {test_metrics['total_loss']:.6f}")
    print(f"  Test Loss (Reconstruction):     {test_metrics['recon_loss']:.6f}")
    print(f"  Test Loss (KL): {test_metrics['kl_loss']:.6f}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
