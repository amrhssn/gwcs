"""
Pretraining script for reconstruction model on turbulent radiative layer 2D data.

Adaptations for turbulent data:
- Uses GraphSignalDatasetTurbulent with field selection
- Uses ChebyshevMatrixCacheTurbulent for shared graph structure
- Uses batch_chebyshev_igwt instead of batch_igwt
- Smaller batch sizes (4 instead of 32) due to larger graphs (49k nodes)
- Adjusted hyperparameters for larger graphs
- Saves to local_storage/runs_turbulent/

Features:
- Chebyshev-based IGWT (memory-efficient for large graphs)
- Field-specific or multi-field training
- Lightweight collate function
- Mask generation on GPU
- Early stopping activated after sparsity annealing completes
- Loss history tracking and artifact saving
- Test evaluation with best model loading
"""

import os
import time
from datetime import datetime
from typing import Optional, List

import torch

from paths import ROOT_PROJECT_PATH
from utils.random import set_random_seed
from data.data_types_turbulent import ReconstructorPretrainingConfigTurbulent
from data.dataset_turbulent import (
    GraphSignalDatasetTurbulent,
    create_data_splits_turbulent,
    create_optimized_dataloaders_turbulent,
    create_matrix_cache_turbulent,
    ChebyshevMatrixCacheTurbulent,
)
from data.utils_io import save_config, save_json
from models.sparsification_utils import SparsityAnnealer
from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT

print("=" * 80)
print("TURBULENT RECONSTRUCTION PRETRAINING - Imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("=" * 80)

random_seed = 13
set_random_seed(random_seed=random_seed)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
            self,
            patience: int = 50,
            min_delta: float = 0.0,
            verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.is_best_epoch = False

    def __call__(self, score: float):
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


def generate_mask_on_gpu_turbulent(
        coeffs: torch.Tensor,
        keep_ratio: float,
        mode: str = "mlis"
) -> torch.Tensor:
    """
    Generate sparsification mask on GPU for turbulent data.

    Args:
        coeffs: (batch_total_nodes, num_filters) GWT coefficients
        keep_ratio: Fraction of coefficients to keep
        mode: "mlis", "weighted_random_per_scale", or "random"

    Returns:
        mask: (batch_total_nodes, num_filters) boolean mask
    """
    from models.sparsification_utils import (
        multilevel_importance_sampling,
        weighted_random_per_scale,
        random_mask,
    )

    if mode == "mlis":
        mask = multilevel_importance_sampling(coeffs=coeffs, keep_ratio=keep_ratio)
    elif mode == "weighted_random_per_scale":
        mask = weighted_random_per_scale(coeffs=coeffs, keep_ratio=keep_ratio)
    elif mode == "random":
        mask = random_mask(coeffs=coeffs, keep_ratio=keep_ratio)
    else:
        raise ValueError(f"Unknown mask generation mode: {mode}")

    return mask


def forward_step_turbulent(
        model: torch.nn.Module,
        all_batches: List,
        matrix_cache: ChebyshevMatrixCacheTurbulent,
        mode: str,
        configs: ReconstructorPretrainingConfigTurbulent,
        sparsity_scheduler: SparsityAnnealer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu'),
        verbose: bool = False
) -> dict:
    """
    Forward step for training or evaluation on turbulent data.

    Args:
        model: The model to train/evaluate
        all_batches: Pre-collected list of batches for this epoch
        matrix_cache: Chebyshev matrix cache
        mode: 'train' or 'eval'
        configs: Configuration object
        sparsity_scheduler: Sparsity annealing scheduler
        optimizer: Optimizer (required for training)
        device: Target device
        verbose: Print verbose output

    Returns:
        Dictionary of metrics
    """
    if mode == 'train':
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training mode")
        model.train()
    elif mode == 'eval':
        model.eval()
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")

    context = torch.no_grad() if mode == 'eval' else torch.enable_grad()

    batch_total_losses = []
    batch_recon_losses = []
    batch_regularization_losses = []

    # Profiling
    time_data_loading = 0
    time_transfer = 0
    time_cache_lookup = 0
    time_mask_gen = 0
    time_forward = 0
    time_backward = 0

    t0 = time.time()

    with context:
        for batch_idx, batch in enumerate(all_batches):
            batch_start = time.time()

            # PROFILE: Data loading (already done by list())
            data_load_time = batch_start - t0
            time_data_loading += data_load_time

            # PROFILE: Transfer to GPU
            transfer_start = time.time()
            x = batch['x'].to(device, non_blocking=True)
            coeffs = batch['coeffs'].to(device, non_blocking=True)
            edge_index = batch['edge_index'].to(device, non_blocking=True)
            batch_vec = batch['batch'].to(device, non_blocking=True)
            ptr = batch['ptr'].to(device, non_blocking=True)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start
            time_transfer += transfer_time

            # PROFILE: Get Chebyshev data from cache (same for all samples)
            cache_start = time.time()
            chebyshev_data = matrix_cache.get(batch['indices'])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            cache_time = time.time() - cache_start
            time_cache_lookup += cache_time

            # PROFILE: Generate mask on GPU
            mask_start = time.time()
            if configs.generate_mask:
                current_keep_ratio = sparsity_scheduler.get_keep_ratio()
                sparsification_mask = generate_mask_on_gpu_turbulent(
                    coeffs=coeffs,
                    keep_ratio=current_keep_ratio,
                    mode=configs.mask_generation_mode
                )
            else:
                sparsification_mask = torch.ones_like(coeffs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            mask_time = time.time() - mask_start
            time_mask_gen += mask_time

            # PROFILE: Forward pass
            forward_start = time.time()
            x_hat = model(
                coeffs=coeffs,
                sparsification_mask=sparsification_mask,
                edge_index=edge_index,
                batch=batch_vec,
                chebyshev_data=chebyshev_data,
                batch_ptr=ptr,
            )

            # Compute loss
            recon_loss = torch.nn.functional.mse_loss(x_hat, x.unsqueeze(-1))
            regularization_loss = configs.gamma_regularization * sum(
                p.pow(2).sum() for p in model.parameters()
            )
            total_loss = recon_loss + regularization_loss

            if device.type == 'cuda':
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            time_forward += forward_time

            # PROFILE: Backward pass (only in training)
            backward_time = 0
            if mode == 'train':
                backward_start = time.time()
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                backward_time = time.time() - backward_start
                time_backward += backward_time

            # Record losses
            batch_total_losses.append(total_loss.item())
            batch_recon_losses.append(recon_loss.item())
            batch_regularization_losses.append(
                regularization_loss.item() if isinstance(regularization_loss, torch.Tensor) else regularization_loss
            )

            # Print per-batch timing (first 3 batches only)
            if verbose and batch_idx < 3:
                print(f"    Batch {batch_idx}: Load={data_load_time:.3f}s, "
                      f"Transfer={transfer_time:.3f}s, Cache={cache_time:.3f}s, "
                      f"MaskGen={mask_time:.3f}s, Forward={forward_time:.3f}s, "
                      f"Backward={backward_time:.3f}s")

            t0 = time.time()

    # Print epoch timing summary
    if verbose:
        print(f"  [TIMING] Data: {time_data_loading:.2f}s, Transfer: {time_transfer:.2f}s, "
              f"Cache: {time_cache_lookup:.2f}s, MaskGen: {time_mask_gen:.2f}s, "
              f"Forward: {time_forward:.2f}s, Backward: {time_backward:.2f}s")

    # Compute average metrics
    avg_total_loss = sum(batch_total_losses) / len(batch_total_losses)
    avg_recon_loss = sum(batch_recon_losses) / len(batch_recon_losses)
    avg_regularization_loss = sum(batch_regularization_losses) / len(batch_regularization_losses)

    metrics = {
        'total_loss': avg_total_loss,
        'recon_loss': avg_recon_loss,
        'regularization_loss': avg_regularization_loss,
    }

    return metrics


def get_pretraining_configs_turbulent(
        base_data_dir: str,
        field_index: Optional[int] = None,
) -> ReconstructorPretrainingConfigTurbulent:
    """
    Get configuration for turbulent reconstruction pretraining.

    Args:
        base_data_dir: Path to processed turbulent data directory
        field_index: Which field to train on (0-3 for specific field, None for all)

    Returns:
        Configuration object
    """
    # Determine field name for logging
    field_names = ['density', 'pressure', 'velocity_x', 'velocity_y']
    if field_index is None:
        field_name = 'all_fields'
        output_dim = 4
    else:
        field_name = field_names[field_index]
        output_dim = 1
    keep_ratio = 0.01
    configs = ReconstructorPretrainingConfigTurbulent(
        base_data_dir=base_data_dir,
        random_seed=random_seed,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=4,  # Smaller due to large graphs (49k nodes)
        generate_mask=True,
        keep_ratio=0.1,
        mask_generation_mode='mlis',
        sparsity_initial_value=1.0,
        sparsity_final_value=keep_ratio,
        sparsity_scheduler_final_epoch=0,
        sparsity_scheduler_strategy='linear',
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=10,
        lr_scheduler_min_lr=1e-6,
        early_stopping_patience=30,
        early_stopping_patience_min_delta=1e-4,
        model_name='GUNetNeuralIGWT',
        processor_num_layers=2,  # Reduced for larger graphs
        output_dim=output_dim,
        hidden_dim=64,  # Reduced from 128
        activation_function='relu',
        encoder_type='mlp',
        fusion_type='mlp',
        decoder_type='mlp',
        conv_layer_type='SAGEConv',
        pooling_type='sag_pooling',
        pooling_ratio=0.5,
        predict_residual=True,
        enable_scale_embeddings=True,
        lr=1e-3,
        weight_decay=1e-5,
        gamma_coeff_consistency=0.0,
        gamma_regularization=0.001,
        epochs=200,
        cache_mode='full',  # Use full cache since graph is shared
        lru_cache_size=0,  # Not used with full cache
        num_workers=0,  # Recommended when data is preloaded
        preload_to_memory=True,
        # Turbulent-specific
        field_index=field_index,
        field_name=field_name,
        igwt_mode='chebyshev',
    )

    return configs


def main_turbulent():
    """Main training function for turbulent reconstruction."""
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels
    torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on Ampere+ GPUs
    verbose = False  # Set to True for detailed timing logs

    # Setup
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    base_data_dir = os.path.join(local_storage_dir, 'processed_data', 'turbulent_radiative_layer_2D', 'test')

    # Choose field: None for all fields, or 0-3 for specific field
    # 0: density, 1: pressure, 2: velocity_x, 3: velocity_y
    field_index = 0  # Training on density only

    configs = get_pretraining_configs_turbulent(
        base_data_dir=base_data_dir,
        field_index=field_index,
    )

    print("\n" + "=" * 80)
    print(f"CONFIGURATION - Field: {configs.field_name}")
    print("=" * 80)
    print(f"Data directory: {configs.base_data_dir}")
    print(f"Field index: {configs.field_index}")
    print(f"Output dim: {configs.output_dim}")
    print(f"IGWT mode: {configs.igwt_mode}")
    print(f"Batch size: {configs.batch_size}")
    print(f"Hidden dim: {configs.hidden_dim}")
    print("=" * 80)

    # Create run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_turbulent_{configs.field_name}"
    run_dir = os.path.join(local_storage_dir, 'runs_turbulent', run_name)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    # Save config
    save_config(config=configs,
                config_name='reconstructor_pretraining_config_turbulent',
                save_dir=run_dir)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    dataset = GraphSignalDatasetTurbulent(
        base_data_dir=configs.base_data_dir,
        field_index=configs.field_index,
        preload_to_memory=configs.preload_to_memory,
    )
    print(f"✓ Loaded {len(dataset)} samples")

    # Split dataset
    train_dataset, val_dataset, test_dataset = create_data_splits_turbulent(
        dataset,
        train_ratio=configs.train_ratio,
        val_ratio=configs.val_ratio,
        test_ratio=configs.test_ratio,
        seed=configs.random_seed,
    )
    print(f"✓ Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Save split indices
    save_json(
        data={
            "train": list(train_dataset.indices),
            "val": list(val_dataset.indices),
            "test": list(test_dataset.indices),
        },
        filename='split_indices.json',
        save_dir=run_dir
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_optimized_dataloaders_turbulent(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
    )
    print(f"✓ Created dataloaders")

    # Create matrix cache
    print("\n" + "=" * 80)
    print("CREATING MATRIX CACHE")
    print("=" * 80)
    matrix_cache = create_matrix_cache_turbulent(dataset=dataset, device=device)

    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    if configs.model_name == 'GNNNeuralIGWT':
        model = GNNNeuralIGWT(
            num_filters=dataset.gwt_config.num_filters,
            hidden_dim=configs.hidden_dim,
            output_dim=configs.output_dim,
            activation_function=configs.activation_function,
            encoder_type=configs.encoder_type,
            decoder_type=configs.decoder_type,
            conv_layer_type=configs.conv_layer_type,
            processor_num_layers=configs.processor_num_layers,
            predict_residual=configs.predict_residual,
            enable_scale_embeddings=configs.enable_scale_embeddings,
            igwt_mode=configs.igwt_mode,
        )
    elif configs.model_name == 'GUNetNeuralIGWT':
        model = GUNetNeuralIGWT(
            num_filters=dataset.gwt_config.num_filters,
            hidden_dim=configs.hidden_dim,
            output_dim=configs.output_dim,
            activation_function=configs.activation_function,
            encoder_type=configs.encoder_type,
            fusion_type=configs.fusion_type,
            decoder_type=configs.decoder_type,
            conv_layer_type=configs.conv_layer_type,
            processor_num_layers=configs.processor_num_layers,
            pooling_type=configs.pooling_type,
            pooling_ratio=configs.pooling_ratio,
            predict_residual=configs.predict_residual,
            enable_scale_embeddings=configs.enable_scale_embeddings,
            igwt_mode=configs.igwt_mode,
        )
    else:
        raise ValueError(f"Unknown model: {configs.model_name}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Created {configs.model_name} with {num_params:,} parameters")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=configs.lr_scheduler_factor,
        patience=configs.lr_scheduler_patience,
        min_lr=configs.lr_scheduler_min_lr,
    )

    # Create schedulers
    sparsity_scheduler = SparsityAnnealer(
        initial_value=configs.sparsity_initial_value,
        final_value=configs.sparsity_final_value,
        num_epochs=configs.sparsity_scheduler_final_epoch,
        strategy=configs.sparsity_scheduler_strategy
    )

    early_stopping = EarlyStopping(
        patience=configs.early_stopping_patience,
        min_delta=configs.early_stopping_patience_min_delta,
        verbose=True
    )

    # Warmup
    print("\nWarming up...")
    train_batches_warmup = list(train_loader)
    if train_batches_warmup:
        batch = train_batches_warmup[0]
        _ = matrix_cache.get(batch['indices'])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("✓ Warmup complete")

    # Training state
    sparsity_annealing_complete = False
    best_epoch = -1
    best_checkpoint_path = None

    # Loss history tracking
    train_losses_history = {
        'total': [],
        'recon': [],
        'regularization': []
    }
    val_losses_history = {
        'total': [],
        'recon': [],
        'regularization': []
    }

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    for epoch in range(configs.epochs):
        epoch_start_time = time.time()
        current_sparsity = sparsity_scheduler.get_keep_ratio()
        current_lr = optimizer.param_groups[0]['lr']

        # Pre-collect all batches for this epoch
        train_batches = list(train_loader)
        val_batches = list(val_loader)

        # ============ Training ============
        train_metrics = forward_step_turbulent(
            model=model,
            all_batches=train_batches,
            matrix_cache=matrix_cache,
            mode='train',
            configs=configs,
            sparsity_scheduler=sparsity_scheduler,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
        )

        # ============ Validation ============
        val_metrics = forward_step_turbulent(
            model=model,
            all_batches=val_batches,
            matrix_cache=matrix_cache,
            mode='eval',
            configs=configs,
            sparsity_scheduler=sparsity_scheduler,
            device=device,
            verbose=verbose,
        )

        # ============ Track Loss History ============
        train_losses_history['total'].append(train_metrics['total_loss'])
        train_losses_history['recon'].append(train_metrics['recon_loss'])
        train_losses_history['regularization'].append(train_metrics['regularization_loss'])

        val_losses_history['total'].append(val_metrics['total_loss'])
        val_losses_history['recon'].append(val_metrics['recon_loss'])
        val_losses_history['regularization'].append(val_metrics['regularization_loss'])

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # ============ Print Epoch Summary ============
        print(f"[*] Epoch {epoch:5d} | "
              f"Train Loss: {train_metrics['total_loss']:.6f} "
              f"(R: {train_metrics['recon_loss']:.6f}, "
              f"Reg: {train_metrics['regularization_loss']:.6f}) | "
              f"Val Loss: {val_metrics['total_loss']:.6f} | "
              f"Time: {epoch_time:.2f}s | "
              f"Sparsity: {current_sparsity:.4f} | "
              f"LR: {current_lr:.2e}")

        # ============ Update Schedulers ============
        sparsity_scheduler.step()

        # Check if sparsity annealing just completed
        if not sparsity_annealing_complete and epoch + 1 >= configs.sparsity_scheduler_final_epoch:
            sparsity_annealing_complete = True
            print("=" * 80)
            print(f"Sparsity annealing complete at epoch {epoch}")
            print(f"Final keep_ratio: {current_sparsity:.6f}")
            print("Starting LR scheduling and early stopping...")
            print("=" * 80)

        if sparsity_annealing_complete:
            lr_scheduler.step(train_metrics['total_loss'])
            early_stopping(val_metrics['total_loss'])

        # ============ Save Best Model ============
        if sparsity_annealing_complete and early_stopping.is_best_epoch:
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
        if sparsity_annealing_complete and early_stopping.should_stop:
            print("=" * 80)
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {early_stopping.best_score:.6f}")
            break

    # Save losses history
    save_json(data=train_losses_history, filename='train_loss.json', save_dir=run_dir)
    save_json(data=val_losses_history, filename='val_loss.json', save_dir=run_dir)

    # Print cache statistics
    print("\n" + "-" * 80)
    print("CACHE STATISTICS")
    print("-" * 80)
    stats = matrix_cache.get_stats()
    print(f"  Cache type: {stats['cache_type']}")
    print(f"  Memory usage: ~{stats['memory_mb']:.1f} MB")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")

    # Load best model for testing
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
    test_batches = list(test_loader)
    test_metrics = forward_step_turbulent(
        model=model,
        all_batches=test_batches,
        matrix_cache=matrix_cache,
        mode='eval',
        configs=configs,
        sparsity_scheduler=sparsity_scheduler,
        device=device,
        verbose=verbose,
    )

    # Save test metrics
    save_json(data=test_metrics, filename='test_metrics.json', save_dir=run_dir)
    print(f"Test metrics saved to {os.path.join(run_dir, 'test_metrics.json')}")

    print(f"\n{'=' * 80}")
    print("TEST RESULTS:")
    print(f"  Test Loss (Total):          {test_metrics['total_loss']:.6f}")
    print(f"  Test Loss (Reconstruction): {test_metrics['recon_loss']:.6f}")
    print(f"  Test Loss (Regularization): {test_metrics['regularization_loss']:.6f}")
    print(f"{'=' * 80}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    if early_stopping.best_score is not None:
        print(f"Best validation loss: {early_stopping.best_score:.6f}")

    # Cleanup
    matrix_cache.clear()


if __name__ == '__main__':
    main_turbulent()