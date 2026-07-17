"""
train.py — Unified NIGWT Training Script
==========================================
Supports: turbulent_radiative_layer_2D | viscoelastic_instability | pbfm

Key fixes vs model_training.py / model_trainingV.py:
  1. Field names loaded DYNAMICALLY from shared_data.pt (not hardcoded turbulent list).
  2. denormalize_predictions uses dynamic field names — no KeyError for viscoelastic/pbfm.
  3. Unified dataset routing for all 3 datasets via --dataset argument.
  4. run directory: runs_turbulent | runs_viscoelastic | runs_pbfm.

Multi-GPU: launched by torchrun (see pipeline.py or shell scripts).
  torchrun --nproc_per_node=4 train.py --dataset turbulent_radiative_layer_2D --use-normalization

Usage:
    torchrun --nproc_per_node=4 train.py --dataset turbulent_radiative_layer_2D --use-normalization
    torchrun --nproc_per_node=4 train.py --dataset viscoelastic_instability --use-normalization
    torchrun --nproc_per_node=4 train.py --dataset pbfm --use-normalization
"""

import sys
import os
import time
import argparse
import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from paths import ROOT_PROJECT_PATH
from utils.random import set_random_seed
from data.data_types_turbulent import ReconstructorPretrainingConfigTurbulent
from data.dataset_turbulent import (
    GraphSignalDatasetTurbulent,
    create_optimized_dataloaders_turbulent,
    create_matrix_cache_turbulent,
    ChebyshevMatrixCacheTurbulent,
)
from data.utils_io import save_config, save_json, load_best_checkpoint, load_config, load_json
from models.sparsification_utils import SparsityAnnealer
from models.sparsification_utils import (
    multilevel_importance_sampling,
    multilevel_importance_sampling_with_exploration,
    weighted_random_per_scale,
    random_mask,
)
from models.reconstruction import GNNNeuralIGWT, GUNetNeuralIGWT

print("=" * 80)
print("NIGWT TRAINING - Imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:     {torch.cuda.get_device_name(0)}")
print("=" * 80)

random_seed = 13
set_random_seed(random_seed=random_seed)


# =============================================================================
# DATASET → PATHS  (short run-dir suffix)
# =============================================================================

# All paths are relative to ROOT_PROJECT_PATH (resolved in paths.py, i.e. the
# directory containing this project's own local_storage/ directory)
DATASET_CONFIGS = {
    'turbulent_radiative_layer_2D': {
        'processed_subpath': 'local_storage/datasets/turbulent_radiative_layer_2D/processed',
        'runs_subpath':      'local_storage/runs_turbulent',
        'run_label':         'turbulent',
    },
    'viscoelastic_instability': {
        'processed_subpath': 'local_storage/datasets/viscoelastic_instability/processed',
        'runs_subpath':      'local_storage/runs_viscoelastic',
        'run_label':         'viscoelastic',
    },
    # 'pbfm' renamed to 'kolmogorov' — reflects actual Kolmogorov flow physics
    'kolmogorov': {
        'processed_subpath': 'local_storage/datasets/kolmogorov/processed',
        'runs_subpath':      'local_storage/runs_kolmogorov',
        'run_label':         'kolmogorov',
    },
    'dynamic_stall': {
        'processed_subpath': 'local_storage/datasets/dynamic_stall/processed',
        'runs_subpath':      'local_storage/runs_dynamic_stall',
        'run_label':         'dynamic_stall',
    },
}


def get_dataset_paths(dataset: str) -> Dict[str, Path]:
    """Return absolute paths for a dataset."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. "
            f"Choose from: {list(DATASET_CONFIGS.keys())}"
        )
    cfg  = DATASET_CONFIGS[dataset]
    root = Path(ROOT_PROJECT_PATH)
    return {
        'processed':  root / cfg['processed_subpath'],
        'runs':       root / cfg['runs_subpath'],
        'run_label':  cfg['run_label'],
    }


# =============================================================================
# DYNAMIC FIELD NAME HELPERS
# =============================================================================

def load_field_names_from_shared_data(train_data_dir: str) -> List[str]:
    """
    Load field names dynamically from shared_data.pt in the train split.
    This is the correct way — never hardcode field names.
    """
    shared_path = Path(train_data_dir) / 'shared_data.pt'
    if not shared_path.exists():
        raise FileNotFoundError(f"shared_data.pt not found at {shared_path}")
    shared = torch.load(str(shared_path), map_location='cpu', weights_only=False)
    field_names = shared['field_names']
    print(f"  Loaded field names from shared_data.pt: {field_names}")
    return field_names


# =============================================================================
# DENORMALIZE  (dynamic field names — no hardcoding)
# =============================================================================

def denormalize_predictions(
        x: torch.Tensor,
        stats: Dict,
        field_index: Optional[int],
) -> torch.Tensor:
    """
    Denormalize predictions to original physical scale.

    Args:
        x:           Predictions in normalised space, shape (N, C)
        stats:       {field_name: {'mean': float, 'std': float}}
        field_index: Which field (int) or None for multi-field

    Returns:
        Denormalised predictions.
    """
    # FIX: use dynamic field names loaded from stats dict — not hardcoded turbulent list
    field_names = list(stats.keys())

    if field_index is None:
        x_denorm = torch.zeros_like(x)
        for idx, name in enumerate(field_names):
            m = stats[name]['mean']
            s = stats[name]['std']
            x_denorm[:, idx:idx+1] = x[:, idx:idx+1] * s + m
        return x_denorm
    else:
        name = field_names[field_index]
        m = stats[name]['mean']
        s = stats[name]['std']
        return x * s + m


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 0.0, verbose: bool = True):
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose
        self.counter   = 0
        self.best_score       = None
        self.should_stop      = False
        self.is_best_epoch    = False

    def __call__(self, score: float):
        self.is_best_epoch = False
        if self.best_score is None:
            self.best_score    = score
            self.is_best_epoch = True
            if self.verbose:
                print(f"  → Initial best score: {score:.6f}")
            return
        improved = score < (self.best_score - self.min_delta)
        if improved:
            if self.verbose:
                print(f"  → Validation improved: {self.best_score:.6f} → {score:.6f}")
            self.best_score    = score
            self.counter       = 0
            self.is_best_epoch = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# =============================================================================
# MASK GENERATION
# =============================================================================

def generate_mask_on_gpu(
        coeffs: torch.Tensor,
        keep_ratio: float,
        mode: str = "mlis",
        exploration_rate: float = 0.0,
) -> torch.Tensor:
    if mode == "mlis":
        return multilevel_importance_sampling(coeffs=coeffs, keep_ratio=keep_ratio)
    elif mode == "mlis_with_exploration":
        return multilevel_importance_sampling_with_exploration(
            coeffs=coeffs, keep_ratio=keep_ratio, exploration_rate=exploration_rate)
    elif mode == "weighted_random_per_scale":
        return weighted_random_per_scale(coeffs=coeffs, keep_ratio=keep_ratio)
    elif mode == "random":
        return random_mask(coeffs=coeffs, keep_ratio=keep_ratio)
    else:
        raise ValueError(f"Unknown mask mode: {mode}")


# =============================================================================
# FORWARD STEP (train / eval)
# =============================================================================

def forward_step(
        model: torch.nn.Module,
        dataloader,
        matrix_cache: ChebyshevMatrixCacheTurbulent,
        mode: str,
        configs: ReconstructorPretrainingConfigTurbulent,
        sparsity_scheduler: SparsityAnnealer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu'),
        verbose: bool = False,
        normalization_stats: Optional[Dict] = None,
        compute_physical_metrics: bool = False,
        rank: int = 0,
) -> dict:
    if mode == 'train':
        if optimizer is None:
            raise ValueError("Optimizer required for training")
        model.train()
    elif mode == 'eval':
        model.eval()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    context = torch.no_grad() if mode == 'eval' else torch.enable_grad()

    batch_total_losses = []
    batch_recon_losses = []
    batch_reg_losses   = []

    t0 = time.time()

    with context:
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"  {mode.capitalize()}",
            leave=True,
            ncols=120,
            disable=(rank != 0),
        )
        for batch_idx, batch in pbar:
            x        = batch['x'].to(device, non_blocking=True)
            coeffs   = batch['coeffs'].to(device, non_blocking=True)
            batch_vec = batch['batch'].to(device, non_blocking=True)
            ptr       = batch['ptr'].to(device, non_blocking=True)

            edge_index     = matrix_cache.get_batched_edge_index(batch['batch_size'])
            chebyshev_data = matrix_cache.get(batch['indices'])

            if configs.generate_mask:
                current_keep_ratio = sparsity_scheduler.get_keep_ratio()
                sparsification_mask = generate_mask_on_gpu(
                    coeffs=coeffs,
                    keep_ratio=current_keep_ratio,
                    mode=configs.mask_generation_mode,
                    exploration_rate=configs.exploration_rate,
                )
            else:
                sparsification_mask = torch.ones_like(coeffs)

            x_hat = model(
                coeffs=coeffs,
                sparsification_mask=sparsification_mask,
                edge_index=edge_index,
                batch=batch_vec,
                chebyshev_data=chebyshev_data,
                batch_ptr=ptr,
            )

            if compute_physical_metrics and normalization_stats is not None:
                x_hat_d = denormalize_predictions(x_hat, normalization_stats, configs.field_index)
                x_d     = denormalize_predictions(x.unsqueeze(-1), normalization_stats, configs.field_index)
                recon_loss = torch.nn.functional.mse_loss(x_hat_d, x_d)
            else:
                recon_loss = torch.nn.functional.mse_loss(x_hat, x.unsqueeze(-1))

            reg_loss   = configs.gamma_regularization * sum(
                p.pow(2).sum() for p in model.module.parameters()
            )
            total_loss = recon_loss + reg_loss

            if mode == 'train':
                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(total_loss):
                    del x, coeffs, batch_vec, ptr, sparsification_mask, x_hat
                    del recon_loss, reg_loss, total_loss, chebyshev_data
                    t0 = time.time()
                    continue
                total_loss.backward()

                has_nan_grad = any(
                    param.grad is not None and not torch.isfinite(param.grad).all()
                    for _, param in model.named_parameters()
                )
                if has_nan_grad:
                    del x, coeffs, batch_vec, ptr, sparsification_mask, x_hat
                    del recon_loss, reg_loss, total_loss, chebyshev_data
                    t0 = time.time()
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_total_losses.append(total_loss.item())
            batch_recon_losses.append(recon_loss.item())
            batch_reg_losses.append(
                reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss)

            avg_loss  = sum(batch_total_losses) / len(batch_total_losses)
            avg_recon = sum(batch_recon_losses) / len(batch_recon_losses)
            pbar.set_postfix({'loss': f'{avg_loss:.6f}', 'recon': f'{avg_recon:.6f}'})

            del x, coeffs, batch_vec, ptr, sparsification_mask, x_hat
            del recon_loss, reg_loss, total_loss, chebyshev_data
            t0 = time.time()

    avg_total = sum(batch_total_losses) / max(len(batch_total_losses), 1)
    avg_recon = sum(batch_recon_losses) / max(len(batch_recon_losses), 1)
    avg_reg   = sum(batch_reg_losses)   / max(len(batch_reg_losses), 1)

    metrics_tensor = torch.tensor([avg_total, avg_recon, avg_reg], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)

    return {
        'total_loss':          metrics_tensor[0].item(),
        'recon_loss':          metrics_tensor[1].item(),
        'regularization_loss': metrics_tensor[2].item(),
    }


# =============================================================================
# VERSIONED DATA HELPERS
# =============================================================================

def get_latest_processed_version(base_data_dir: str) -> str:
    """Return the directory name of the latest versioned processed data."""
    entries = [
        e for e in os.listdir(base_data_dir)
        if os.path.isdir(os.path.join(base_data_dir, e))
    ]
    version_dirs = []
    for entry in entries:
        parts = entry.split('_')
        if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
            try:
                int(parts[0]); int(parts[1])
                version_dirs.append(entry)
            except ValueError:
                continue
    if not version_dirs:
        raise ValueError(f"No valid version directories in: {base_data_dir}")
    version_dirs.sort(reverse=True)
    return version_dirs[0]


# =============================================================================
# CONFIG BUILDER
# =============================================================================

def get_pretraining_configs(
        base_data_dir: str,
        field_index: Optional[int] = None,
        data_version: Optional[str] = None,
        max_samples: Optional[int] = None,
        hidden_dim: int = 256,
        processor_num_layers: int = 5,
) -> ReconstructorPretrainingConfigTurbulent:
    """
    Build training configuration.

    FIX: field names loaded dynamically from shared_data.pt instead of
    hardcoding ['density', 'pressure', 'velocity_x', 'velocity_y'].

    Args:
        hidden_dim:           Hidden dimension for GNN layers (default 256 → ~993K params).
                              Use 96 for ~99K params (matching GXN baseline size).
        processor_num_layers: Number of GNN processor layers (default 5).
                              Use 3 for ~99K params.
    """
    if data_version is None:
        data_version = get_latest_processed_version(base_data_dir)
        print(f"Using latest data version: {data_version}")

    version_dir       = os.path.join(base_data_dir, data_version)
    train_data_dir    = os.path.join(version_dir, 'train')
    valid_data_dir    = os.path.join(version_dir, 'valid')
    test_data_dir     = os.path.join(version_dir, 'test')

    # Turbulent's WellDataset uses a 'val' directory (not 'valid'), so the valid split
    # may have been skipped during preprocessing.  Fall back to test as validation.
    if not os.path.isdir(valid_data_dir):
        print(f"  ⚠  No 'valid' split found at {valid_data_dir}. "
              f"Using 'test' split as validation.")
        valid_data_dir = test_data_dir

    # FIX: load field names dynamically from shared_data.pt
    field_names = load_field_names_from_shared_data(train_data_dir)

    if field_index is None:
        field_name = 'all_fields'
        output_dim = len(field_names)
    else:
        if field_index >= len(field_names):
            raise ValueError(
                f"field_index={field_index} out of range for "
                f"dataset with fields {field_names}"
            )
        field_name = field_names[field_index]
        output_dim = 1

    keep_ratio = 0.01
    configs = ReconstructorPretrainingConfigTurbulent(
        base_data_dir=base_data_dir,
        train_data_dir=train_data_dir,
        validation_data_dir=valid_data_dir,
        test_data_dir=test_data_dir,
        random_seed=random_seed,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=32,
        generate_mask=True,
        keep_ratio=keep_ratio,
        mask_generation_mode='mlis_with_exploration',
        exploration_rate=0.5,
        sparsity_initial_value=1.0,
        sparsity_final_value=keep_ratio,
        sparsity_scheduler_final_epoch=0,
        sparsity_scheduler_strategy='linear',
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=10,
        lr_scheduler_min_lr=1e-6,
        early_stopping_patience=30,
        early_stopping_patience_min_delta=1e-4,
        model_name='GNNNeuralIGWT',
        processor_num_layers=processor_num_layers,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        activation_function='silu',
        encoder_type='mlp',
        fusion_type='mlp',
        decoder_type='mlp',
        conv_layer_type='SAGEConv',
        pooling_type='top_k_pooling',
        pooling_ratio=0.5,
        predict_residual=True,
        enable_scale_embeddings=True,
        lr=1e-4,
        weight_decay=1e-5,
        gamma_coeff_consistency=0.0,
        gamma_regularization=0.001,
        epochs=400,
        cache_mode='full',
        lru_cache_size=0,
        num_workers=0,
        preload_to_memory=False,
        field_index=field_index,
        field_name=field_name,
        igwt_mode='chebyshev',
    )
    return configs


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main_train(
        dataset: str = 'turbulent_radiative_layer_2D',
        max_samples: Optional[int] = None,
        resume_run_timestamp: Optional[str] = None,
        use_normalization: bool = False,
        hidden_dim: int = 256,
        num_layers: int = 5,
        processed_dir: Optional[str] = None,
):
    """
    Main DDP training entry point.

    Args:
        dataset:               One of 'turbulent_radiative_layer_2D',
                               'viscoelastic_instability', 'pbfm'.
        max_samples:           Cap samples per split (None = all).
        resume_run_timestamp:  Timestamp string to resume training.
        use_normalization:     If True, apply Z-score normalisation.
        hidden_dim:            GNN hidden dimension (default 256 → ~993K params;
                               use 96 for ~99K params matching GXN).
        num_layers:            Number of GNN processor layers (default 5;
                               use 3 for ~99K params).
        processed_dir:         Override the default processed-data base directory.
                               Should be the directory that contains the timestamped
                               version subdirs (e.g. /path/to/.../processed/).
                               If None, falls back to DATASET_CONFIGS default path.
    """
    torch.backends.cudnn.benchmark  = True
    torch.backends.cuda.matmul.allow_tf32 = True
    verbose = False

    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    local_rank = int(os.environ['LOCAL_RANK'])
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = (rank == 0)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    paths = get_dataset_paths(dataset)
    # --processed-dir overrides the default path from DATASET_CONFIGS
    base_data_dir = processed_dir if processed_dir is not None else str(paths['processed'])
    runs_dir      = str(paths['runs'])
    run_label     = paths['run_label']

    if is_main:
        print(f"\n{'='*80}")
        print(f"NIGWT CONFIG  —  dataset={dataset}")
        print(f"  hidden_dim={hidden_dim}  num_layers={num_layers}")
        print(f"  (use compare_model_params.py after training to verify exact param count)")
        print(f"{'='*80}\n")

    # =========================================================================
    # Determine run directory
    # =========================================================================
    if resume_run_timestamp is not None:
        matching = [
            d for d in os.listdir(runs_dir)
            if d.startswith(resume_run_timestamp)
        ]
        if not matching:
            raise FileNotFoundError(
                f"No run matching '{resume_run_timestamp}' in {runs_dir}"
            )
        if len(matching) > 1:
            raise ValueError(f"Multiple runs match '{resume_run_timestamp}': {matching}")

        run_name       = matching[0]
        run_dir        = os.path.join(runs_dir, run_name)
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')

        configs = load_config(
            config_dir=run_dir,
            config_name='reconstructor_pretraining_config_turbulent.json',
        )
        use_normalization = configs.use_normalization
        if is_main:
            print(f"\n🔄 RESUMING from run: {run_dir}")

        # field_names_for_dataset must be loaded dynamically too (for normalization)
        field_names_for_dataset = load_field_names_from_shared_data(configs.train_data_dir)

    else:
        field_index = 0
        configs = get_pretraining_configs(
            base_data_dir=base_data_dir,
            field_index=field_index,
            max_samples=max_samples,
            hidden_dim=hidden_dim,
            processor_num_layers=num_layers,
        )
        configs = dataclasses.replace(configs, use_normalization=use_normalization)

        # FIX: load field names dynamically for normalization stats computation
        field_names_for_dataset = load_field_names_from_shared_data(configs.train_data_dir)

        if is_main:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name  = f"{timestamp}_{run_label}_{configs.field_name}"
            run_dir   = os.path.join(runs_dir, run_name)
            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_config(
                config=configs,
                config_name='reconstructor_pretraining_config_turbulent',
                save_dir=run_dir,
            )
        else:
            run_dir = None
            checkpoint_dir = None

        run_dir_list = [run_dir]
        dist.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')

    if is_main:
        print("\n" + "=" * 80)
        print(f"CONFIGURATION — Field: {configs.field_name} | Dataset: {dataset}")
        print("=" * 80)
        print(f"  Data dir  : {configs.base_data_dir}")
        print(f"  Field idx : {configs.field_index}  ({configs.field_name})")
        print(f"  Fields    : {field_names_for_dataset}")
        print(f"  Output dim: {configs.output_dim}")
        print(f"  Batch size: {configs.batch_size} per GPU ({configs.batch_size * world_size} total)")
        print(f"  Run dir   : {run_dir}")
        print(f"  World size: {world_size}")
        print("=" * 80)

    # =========================================================================
    # Normalization statistics
    # =========================================================================
    normalization_stats = None
    if use_normalization:
        from data.normalization_turbulent import (
            compute_normalization_stats,
            save_normalization_stats,
            load_normalization_stats,
        )

        norm_metadata_path = os.path.join(run_dir, 'normalization_metadata.json')

        if resume_run_timestamp is not None and os.path.exists(norm_metadata_path):
            if is_main:
                print("\n[NORMALIZATION] Loading saved statistics (resume)")
            norm_metadata = load_json(
                filename='normalization_metadata.json', save_dir=run_dir)
            normalization_stats = load_normalization_stats(
                norm_metadata['normalization_stats_path'])
        else:
            version_root = Path(configs.train_data_dir).parent
            stats_path   = version_root / 'normalization_stats.pt'

            if is_main:
                if not stats_path.exists():
                    print("\n[NORMALIZATION] Computing statistics...")
                    normalization_stats = compute_normalization_stats(
                        data_dir=configs.train_data_dir,
                    )
                    save_normalization_stats(
                        stats=normalization_stats, save_path=str(stats_path))
                else:
                    print("\n[NORMALIZATION] Loading existing statistics")

                save_json(
                    data={
                        'use_normalization':          True,
                        'normalization_stats_path':   str(stats_path),
                        'field_names':                field_names_for_dataset,
                    },
                    filename='normalization_metadata.json',
                    save_dir=run_dir,
                )

            dist.barrier()
            normalization_stats = load_normalization_stats(str(stats_path))

    # =========================================================================
    # Datasets
    # =========================================================================
    local_preload = (local_rank == 0) and configs.preload_to_memory

    train_dataset = GraphSignalDatasetTurbulent(
        base_data_dir=configs.train_data_dir,
        field_index=configs.field_index,
        preload_to_memory=local_preload,
        max_samples=max_samples,
        use_normalization=use_normalization,
        normalization_stats=normalization_stats,
    )
    val_dataset = GraphSignalDatasetTurbulent(
        base_data_dir=configs.validation_data_dir,
        field_index=configs.field_index,
        preload_to_memory=local_preload,
        max_samples=max_samples,
        use_normalization=use_normalization,
        normalization_stats=normalization_stats,
    )
    test_dataset = GraphSignalDatasetTurbulent(
        base_data_dir=configs.test_data_dir,
        field_index=configs.field_index,
        preload_to_memory=local_preload,
        max_samples=max_samples,
        use_normalization=use_normalization,
        normalization_stats=normalization_stats,
    )

    if is_main:
        print(f"✓ Datasets: train={len(train_dataset)}, "
              f"val={len(val_dataset)}, test={len(test_dataset)}")
        save_json(
            data={
                "train_files":      train_dataset.file_list,
                "validation_files": val_dataset.file_list,
                "test_files":       test_dataset.file_list,
                "train_dir":        configs.train_data_dir,
                "validation_dir":   configs.validation_data_dir,
                "test_dir":         configs.test_data_dir,
            },
            filename='dataset_info.json',
            save_dir=run_dir,
        )

    # =========================================================================
    # Dataloaders
    # =========================================================================
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler  = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader, val_loader, test_loader = create_optimized_dataloaders_turbulent(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        test_sampler=test_sampler,
    )

    # =========================================================================
    # Matrix cache + Model
    # =========================================================================
    matrix_cache = create_matrix_cache_turbulent(dataset=train_dataset, device=device)

    if configs.model_name == 'GNNNeuralIGWT':
        model = GNNNeuralIGWT(
            num_filters=train_dataset.gwt_config.num_filters,
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
            use_normalization=use_normalization,
            normalization_stats=normalization_stats,
        )
    elif configs.model_name == 'GUNetNeuralIGWT':
        model = GUNetNeuralIGWT(
            num_filters=train_dataset.gwt_config.num_filters,
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
            use_normalization=use_normalization,
            normalization_stats=normalization_stats,
        )
    else:
        raise ValueError(f"Unknown model: {configs.model_name}")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    if is_main:
        n_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print(f"✓ Model: {configs.model_name}  ({n_params:,} parameters)")

    # =========================================================================
    # Optimizer / Schedulers
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=configs.lr_scheduler_factor,
        patience=configs.lr_scheduler_patience,
        min_lr=configs.lr_scheduler_min_lr,
    )
    sparsity_scheduler = SparsityAnnealer(
        initial_value=configs.sparsity_initial_value,
        final_value=configs.sparsity_final_value,
        num_epochs=configs.sparsity_scheduler_final_epoch,
        strategy=configs.sparsity_scheduler_strategy,
    )
    early_stopping = EarlyStopping(
        patience=configs.early_stopping_patience,
        min_delta=configs.early_stopping_patience_min_delta,
        verbose=is_main,
    )

    # =========================================================================
    # Checkpoint resume
    # =========================================================================
    start_epoch = 0
    best_epoch  = -1
    best_checkpoint_path = None

    if resume_run_timestamp is not None:
        checkpoint = load_best_checkpoint(checkpoint_dir)

        has_nan = any(
            not torch.isfinite(t).all()
            for t in checkpoint['model_state_dict'].values()
        )
        if has_nan:
            raise ValueError("Checkpoint contains NaN weights — cannot resume")

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if 'early_stopping_best_score' in checkpoint:
            early_stopping.best_score = checkpoint['early_stopping_best_score']
            early_stopping.counter    = checkpoint.get('early_stopping_counter', 0)
        else:
            early_stopping.best_score = checkpoint['val_loss']

        if 'sparsity_scheduler_epoch' in checkpoint:
            sparsity_scheduler.current_epoch = checkpoint['sparsity_scheduler_epoch']
        else:
            sparsity_scheduler.current_epoch = start_epoch

        best_epoch = checkpoint['epoch']
        best_checkpoint_path = os.path.join(checkpoint_dir, f"{best_epoch}.pt")

        if is_main:
            print(f"✓ Resumed from epoch {checkpoint['epoch']} "
                  f"(val_loss={checkpoint['val_loss']:.6f})")
        del checkpoint
        torch.cuda.empty_cache()

    # Warmup
    warmup_iter = iter(train_loader)
    batch = next(warmup_iter)
    _ = matrix_cache.get(batch['indices'])
    del batch, warmup_iter
    torch.cuda.synchronize()
    if is_main:
        print("✓ Warmup complete")

    # =========================================================================
    # Training loop
    # =========================================================================
    sparsity_annealing_complete = False
    train_losses_history = {'total': [], 'recon': [], 'regularization': []}
    val_losses_history   = {'total': [], 'recon': [], 'regularization': []}

    if is_main:
        print(f"\n{'='*80}\nSTARTING TRAINING  ({configs.epochs - start_epoch} epochs remaining)\n{'='*80}\n")

    for epoch in range(start_epoch, configs.epochs):
        train_sampler.set_epoch(epoch)
        epoch_start   = time.time()
        current_sparsity = sparsity_scheduler.get_keep_ratio()
        current_lr    = optimizer.param_groups[0]['lr']

        train_metrics = forward_step(
            model=model, dataloader=train_loader, matrix_cache=matrix_cache,
            mode='train', configs=configs, sparsity_scheduler=sparsity_scheduler,
            optimizer=optimizer, device=device, verbose=verbose,
            normalization_stats=normalization_stats, rank=rank,
        )
        val_metrics = forward_step(
            model=model, dataloader=val_loader, matrix_cache=matrix_cache,
            mode='eval', configs=configs, sparsity_scheduler=sparsity_scheduler,
            device=device, verbose=verbose,
            normalization_stats=normalization_stats,
            compute_physical_metrics=False, rank=rank,
        )

        for key in ['total', 'recon', 'regularization']:
            train_losses_history[key].append(train_metrics[f'{key}_loss'])
            val_losses_history[key].append(val_metrics[f'{key}_loss'])

        sparsity_scheduler.step()
        if not sparsity_annealing_complete and sparsity_scheduler.is_complete():
            sparsity_annealing_complete = True
            if is_main:
                print(f"  ✓ Sparsity annealing complete at epoch {epoch}")

        lr_scheduler.step(val_metrics['total_loss'])
        if sparsity_annealing_complete:
            early_stopping(val_metrics['total_loss'])

        epoch_time = time.time() - epoch_start

        if is_main:
            print(f"\nEpoch {epoch+1}/{configs.epochs} ({epoch_time:.1f}s) | "
                  f"LR: {current_lr:.2e} | Sparsity: {current_sparsity:.3f}")
            print(f"  Train → Total: {train_metrics['total_loss']:.6f} | "
                  f"Recon: {train_metrics['recon_loss']:.6f}")
            print(f"  Val   → Total: {val_metrics['total_loss']:.6f} | "
                  f"Recon: {val_metrics['recon_loss']:.6f}")

        if is_main and early_stopping.is_best_epoch:
            best_epoch = epoch
            best_checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
            torch.save({
                'epoch':                       epoch,
                'model_state_dict':            model.module.state_dict(),
                'optimizer_state_dict':        optimizer.state_dict(),
                'lr_scheduler_state_dict':     lr_scheduler.state_dict(),
                'train_loss':                  train_metrics['total_loss'],
                'val_loss':                    val_metrics['total_loss'],
                'early_stopping_best_score':   early_stopping.best_score,
                'early_stopping_counter':      early_stopping.counter,
                'sparsity_scheduler_epoch':    sparsity_scheduler.current_epoch,
            }, best_checkpoint_path)
            print(f"  ✓ Checkpoint saved → {best_checkpoint_path}")

        if is_main and (epoch + 1) % 10 == 0:
            save_json(data=train_losses_history, filename='train_loss.json', save_dir=run_dir)
            save_json(data=val_losses_history,   filename='val_loss.json',   save_dir=run_dir)

        if early_stopping.should_stop:
            if is_main:
                print(f"\n  Early stopping at epoch {epoch+1}")
            break

    if is_main:
        save_json(data=train_losses_history, filename='train_loss.json', save_dir=run_dir)
        save_json(data=val_losses_history,   filename='val_loss.json',   save_dir=run_dir)
        print(f"\n✓ Loss history saved")

    # =========================================================================
    # Test evaluation
    # =========================================================================
    if is_main:
        print(f"\n{'='*80}\nTEST EVALUATION\n{'='*80}")

    if best_checkpoint_path is not None:
        checkpoint = load_best_checkpoint(checkpoint_dir)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        if is_main:
            print(f"✓ Loaded best checkpoint (epoch {best_epoch})")
        del checkpoint

    test_metrics = forward_step(
        model=model, dataloader=test_loader, matrix_cache=matrix_cache,
        mode='eval', configs=configs, sparsity_scheduler=sparsity_scheduler,
        device=device, verbose=verbose,
        normalization_stats=normalization_stats,
        compute_physical_metrics=True, rank=rank,
    )

    if is_main:
        print(f"\nTest Results (physical units):")
        print(f"  Total MSE  : {test_metrics['total_loss']:.6f}")
        print(f"  Recon MSE  : {test_metrics['recon_loss']:.6f}")
        save_json(data=test_metrics, filename='test_metrics.json', save_dir=run_dir)
        print(f"✓ Test metrics → {run_dir}/test_metrics.json")
        print(f"✓ Training complete. Run: {run_dir}")

    dist.destroy_process_group()
    return run_dir


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NIGWT model via torchrun.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='turbulent_radiative_layer_2D',
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset to train on',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Max samples per split (for quick tests)',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Run timestamp to resume from (e.g. "2026-03-23_13-48-59")',
    )
    parser.add_argument(
        '--use-normalization',
        action='store_true',
        help='Apply Z-score normalization',
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default=None,
        help=(
            'Override the default processed-data directory. '
            'Must be the directory that contains the timestamped version subdirs '
            '(e.g. /path/to/datasets/viscoelastic_instability/processed/). '
            'If omitted, uses the default path from DATASET_CONFIGS.'
        ),
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help=(
            'GNN hidden dimension. Default 256 → ~993K params. '
            'Use 96 for ~99K params (matching GXN baseline size).'
        ),
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=5,
        help=(
            'Number of GNN processor layers. Default 5. '
            'Use 3 together with --hidden-dim 96 for ~99K param model.'
        ),
    )
    args = parser.parse_args()

    main_train(
        dataset=args.dataset,
        max_samples=args.max_samples,
        resume_run_timestamp=args.resume,
        use_normalization=args.use_normalization,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        processed_dir=args.processed_dir,
    )
