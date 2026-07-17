
"""
Pretraining script for reconstruction model with optimized data loading.

Features:
- GPU matrix caching (Full or LRU)
- Lightweight collate function (only batches small tensors)
- Mask generation on GPU (MLIS default)
- Background prefetching for LRU cache
- num_workers=0 when data is preloaded to RAM
"""

import os
import time
from datetime import datetime
from typing import Optional, Union, List

import torch

from paths import ROOT_PROJECT_PATH
from utils.random import set_random_seed
from data.data_types import ReconstructorPretrainingConfig
from data.dataset import (
    GraphSignalDataset,
    create_data_splits,
    create_optimized_dataloaders,
    create_matrix_cache,
    generate_mask_on_gpu,
    FullMatrixCache,
    LRUMatrixCache,
    MatrixCache,
)
from data.utils_io import (
    save_config,
    save_json
)
from models.sparsification_utils import SparsityAnnealer
from models.reconstruction import (
    GNNNeuralIGWT,
    GUNetNeuralIGWT
)
from models.loss_functions import compute_loss

print("=" * 80)
print("SCRIPT STARTED - Imports successful")
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


def forward_step(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        all_batches: List,
        matrix_cache: MatrixCache,
        mode: str,
        configs: ReconstructorPretrainingConfig,
        sparsity_scheduler: SparsityAnnealer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu'),
        verbose: bool = False
) -> dict:
    """
    Forward step for training or evaluation.

    Args:
        model: The model to train/evaluate
        data_loader: DataLoader (not used directly, kept for API compatibility)
        all_batches: Pre-collected list of batches for this epoch
        matrix_cache: GPU matrix cache (FullMatrixCache or LRUMatrixCache)
        mode: 'train' or 'eval'
        configs: Configuration object
        sparsity_scheduler: Sparsity annealing scheduler
        optimizer: Optimizer (required for training)
        device: Target device

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
    batch_coeffs_consistency_losses = []
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
            batch = batch.to(device, non_blocking=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start
            time_transfer += transfer_time

            # PROFILE: Get matrices from cache
            cache_start = time.time()
            gwt_matrices, L_sym_matrices = matrix_cache.get(batch._indices)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            cache_time = time.time() - cache_start
            time_cache_lookup += cache_time

            # Prefetch NEXT batch (for LRU cache)
            if isinstance(matrix_cache, LRUMatrixCache) and batch_idx + 1 < len(all_batches):
                next_indices = all_batches[batch_idx + 1]._indices
                matrix_cache.prefetch(next_indices)

            # PROFILE: Generate mask on GPU
            mask_start = time.time()
            if configs.generate_mask:
                current_keep_ratio = sparsity_scheduler.get_keep_ratio()
                sparsification_mask = generate_mask_on_gpu(
                    coeffs=batch.coeffs,
                    keep_ratio=current_keep_ratio,
                    mode=configs.mask_generation_mode
                )
            else:
                sparsification_mask = torch.ones_like(batch.coeffs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            mask_time = time.time() - mask_start
            time_mask_gen += mask_time

            # PROFILE: Forward pass
            forward_start = time.time()
            x_hat = model(
                coeffs=batch.coeffs,
                sparsification_mask=sparsification_mask,
                edge_index=batch.edge_index,
                gwt_matrices=gwt_matrices,
                batch=batch.batch
            )

            loss, recon_loss, coeffs_consistency_loss, regularization_loss = compute_loss(
                x_hat=x_hat,
                x=batch.x,
                coeffs=batch.coeffs,
                sparsification_mask=sparsification_mask,
                gwt_matrices=gwt_matrices,
                L_sym=L_sym_matrices,
                gamma_coeff_consistency=configs.gamma_coeff_consistency,
                gamma_regularization=configs.gamma_regularization,
                batch_ptr=batch.ptr
            )
            forward_time = time.time() - forward_start
            time_forward += forward_time

            batch_total_losses.append(loss.item())
            batch_recon_losses.append(recon_loss.item())
            batch_coeffs_consistency_losses.append(coeffs_consistency_loss.item())
            batch_regularization_losses.append(regularization_loss.item())

            # PROFILE: Backward pass
            backward_time = 0
            if mode == 'train':
                backward_start = time.time()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                backward_time = time.time() - backward_start
                time_backward += backward_time

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

    # Compute epoch averages
    epoch_total_loss = sum(batch_total_losses) / len(batch_total_losses)
    epoch_recon_loss = sum(batch_recon_losses) / len(batch_recon_losses)
    epoch_coeffs_consistency_loss = sum(batch_coeffs_consistency_losses) / len(batch_coeffs_consistency_losses)
    epoch_regularization_loss = sum(batch_regularization_losses) / len(batch_regularization_losses)

    forward_metrics = {
        'total_loss': epoch_total_loss,
        'recon_loss': epoch_recon_loss,
        'coeffs_consistency_loss': epoch_coeffs_consistency_loss,
        'regularization_loss': epoch_regularization_loss
    }
    return forward_metrics


def get_pretraining_configs(base_data_dir: str) -> ReconstructorPretrainingConfig:
    """Create configuration for pretraining."""
    # Data config
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    batch_size = 32
    generate_mask = True
    keep_ratio = 0.01
    mask_generation_mode = "weighted_random_per_scale"  # "mlis", "weighted_random_per_scale", "random"

    # Scheduler config
    sparsity_initial_value = 1.0
    sparsity_final_value = keep_ratio
    sparsity_scheduler_final_epoch = 0
    sparsity_scheduler_strategy = 'cosine'
    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 50
    lr_scheduler_min_lr = 1e-6
    early_stopping_patience = 300
    early_stopping_patience_min_delta = 1e-4

    # Model config
    model_name = 'GUNetNeuralIGWT'  # "GUNetNeuralIGWT", "GNNNeuralIGWT"
    processor_num_layers = 5
    output_dim = 1
    hidden_dim = 128
    activation_function = 'relu'
    encoder_type = 'mlp'
    fusion_type = 'mlp'
    decoder_type = 'mlp'
    conv_layer_type = 'SAGEConv'
    pooling_type = "sag_pooling"
    pooling_ratio = 0.5
    predict_residual = True

    # Optimizer config
    lr = 1e-3
    weight_decay = 5e-4

    # Training config
    gamma_coeff_consistency = 0.0
    gamma_regularization = 0.0
    epochs = 5_000

    # Optimized data loading config
    cache_mode = 'lru'  # 'full' or 'lru'
    lru_cache_size = 400  # Max matrices in LRU cache
    num_workers = 0  # 0 when data is preloaded to RAM
    preload_to_memory = True

    reconstructor_pretraining_config = ReconstructorPretrainingConfig(
        base_data_dir=base_data_dir,
        random_seed=random_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        generate_mask=generate_mask,
        keep_ratio=keep_ratio,
        mask_generation_mode=mask_generation_mode,
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
        hidden_dim=hidden_dim,
        activation_function=activation_function,
        encoder_type=encoder_type,
        fusion_type=fusion_type,
        decoder_type=decoder_type,
        conv_layer_type=conv_layer_type,
        pooling_type=pooling_type,
        pooling_ratio=pooling_ratio,
        predict_residual=predict_residual,
        lr=lr,
        weight_decay=weight_decay,
        gamma_coeff_consistency=gamma_coeff_consistency,
        gamma_regularization=gamma_regularization,
        epochs=epochs,
        cache_mode=cache_mode,
        lru_cache_size=lru_cache_size,
        num_workers=num_workers,
        preload_to_memory=preload_to_memory,
    )
    return reconstructor_pretraining_config


def main():
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels
    torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on Ampere+ GPUs
    # Verbose mode for timing logs
    verbose = False  # Set to True for detailed timing logs

    # Data directories
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    base_data_dir = os.path.join(local_storage_dir, 'data', '2026-01-06_15-41-27')
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(local_storage_dir, 'runs', run_timestamp)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Pretraining configs
    configs = get_pretraining_configs(base_data_dir=base_data_dir)
    save_config(config=configs,
                config_name='reconstructor_pretraining_config',
                save_dir=run_dir)

    # Create dataset with RAM preloading
    dataset = GraphSignalDataset(
        base_data_dir=base_data_dir,
        preload_to_memory=configs.preload_to_memory
    )

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
            "train": list(train_dataset.indices),
            "val": list(val_dataset.indices),
            "test": list(test_dataset.indices),
        },
        filename='split_indices.json',
        save_dir=run_dir
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create matrix cache
    matrix_cache = create_matrix_cache(
        dataset=dataset,
        device=device,
        cache_mode=configs.cache_mode,
        lru_cache_size=configs.lru_cache_size
    )

    # Create optimized dataloaders
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        shuffle=True
    )

    # Create sparsity scheduler
    sparsity_scheduler = SparsityAnnealer(
        initial_value=configs.sparsity_initial_value,
        final_value=configs.sparsity_final_value,
        num_epochs=configs.sparsity_scheduler_final_epoch,
        strategy=configs.sparsity_scheduler_strategy
    )

    # Initialize model
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
            predict_residual=configs.predict_residual
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
            processor_num_layers=dataset.gwt_config.num_scales,
            pooling_type=configs.pooling_type,
            pooling_ratio=configs.pooling_ratio,
            predict_residual=configs.predict_residual,
            debug_mode=False
        )
    else:
        raise NotImplementedError(f"Model {configs.model_name} not implemented")

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

    # Warmup
    print("\nWarming up...")
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        _ = matrix_cache.get(batch._indices)
        break
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
        'coeffs_consistency': [],
        'regularization': []
    }
    val_losses_history = {
        'total': [],
        'recon': [],
        'coeffs_consistency': [],
        'regularization': []
    }

    # Training loop
    print(f"\n{'=' * 80}")
    print("STARTING TRAINING")
    print(f"{'=' * 80}\n")

    for epoch in range(configs.epochs):
        epoch_start_time = time.time()
        current_sparsity = sparsity_scheduler.get_keep_ratio()
        current_lr = optimizer.param_groups[0]['lr']

        # Pre-collect all batches for this epoch (enables proper prefetching)
        train_batches = list(train_loader)
        val_batches = list(val_loader)

        # ============ Training ============
        train_metrics = forward_step(
            model=model,
            data_loader=train_loader,
            all_batches=train_batches,
            matrix_cache=matrix_cache,
            mode='train',
            configs=configs,
            sparsity_scheduler=sparsity_scheduler,
            optimizer=optimizer,
            device=device,
            verbose=verbose
        )

        # ============ Validation ============
        val_metrics = forward_step(
            model=model,
            data_loader=val_loader,
            all_batches=val_batches,
            matrix_cache=matrix_cache,
            mode='eval',
            configs=configs,
            sparsity_scheduler=sparsity_scheduler,
            optimizer=None,
            device=device,
            verbose=verbose
        )

        # ============ Track Loss History ============
        train_losses_history['total'].append(train_metrics['total_loss'])
        train_losses_history['recon'].append(train_metrics['recon_loss'])
        train_losses_history['coeffs_consistency'].append(train_metrics['coeffs_consistency_loss'])
        train_losses_history['regularization'].append(train_metrics['regularization_loss'])

        val_losses_history['total'].append(val_metrics['total_loss'])
        val_losses_history['recon'].append(val_metrics['recon_loss'])
        val_losses_history['coeffs_consistency'].append(val_metrics['coeffs_consistency_loss'])
        val_losses_history['regularization'].append(val_metrics['regularization_loss'])

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # ============ Print Epoch Summary ============
        print(f"[*] Epoch {epoch:5d} | "
              f"Train Loss: {train_metrics['total_loss']:.6f} "
              f"(R: {train_metrics['recon_loss']:.6f}, "
              f"C: {train_metrics['coeffs_consistency_loss']:.6f}, "
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
    print(f"  Cache usage: {stats['cache_size']}/{stats['max_size']} ({stats['usage_pct']:.1f}%)")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
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
    test_metrics = forward_step(
        model=model,
        data_loader=test_loader,
        all_batches=test_batches,
        matrix_cache=matrix_cache,
        mode='eval',
        configs=configs,
        sparsity_scheduler=sparsity_scheduler,
        optimizer=None,
        device=device
    )

    # Save test metrics
    save_json(data=test_metrics, filename='test_metrics.json', save_dir=run_dir)
    print(f"Test metrics saved to {os.path.join(run_dir, 'test_metrics.json')}")

    print(f"\n{'=' * 80}")
    print("TEST RESULTS:")
    print(f"  Test Loss (Total):              {test_metrics['total_loss']:.6f}")
    print(f"  Test Loss (Reconstruction):     {test_metrics['recon_loss']:.6f}")
    print(f"  Test Loss (Coeffs Consistency): {test_metrics['coeffs_consistency_loss']:.6f}")
    print(f"  Test Loss (Regularization):     {test_metrics['regularization_loss']:.6f}")
    print(f"{'=' * 80}\n")

    # Cleanup
    if isinstance(matrix_cache, LRUMatrixCache):
        matrix_cache.stop_prefetching()
    matrix_cache.clear()


if __name__ == "__main__":
    main()