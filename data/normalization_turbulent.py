"""
Normalization utilities for turbulent radiative layer 2D data.

Provides Z-score normalization functionality:
- Compute per-field mean/std statistics from training data
- Apply normalization: (x - mean) / std
- Apply denormalization: x_normalized * std + mean
"""

import os
from typing import Dict, List
from pathlib import Path

import torch
from tqdm import tqdm


def compute_normalization_stats(
    data_dir: str,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for each field in a single pass (Chan's parallel algorithm).

    Reads each file exactly once instead of twice, cutting I/O time in half.
    This avoids NCCL watchdog timeouts when called from rank-0 inside a DDP job.
    """
    # Load field info from shared_data.pt
    shared_data_path = Path(data_dir) / 'shared_data.pt'
    shared_data = torch.load(shared_data_path, map_location='cpu', weights_only=False)
    field_names = shared_data['field_names']
    num_fields = len(field_names)

    if verbose:
        print(f"\n{'='*80}\nCOMPUTING NORMALIZATION STATISTICS (single-pass)\n{'='*80}")
        print(f"Training data directory: {data_dir}")
        print(f"Fields ({num_fields}): {field_names}\n{'='*80}")

    # Get sample file list
    file_list = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.pt') and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0])
    )

    if len(file_list) == 0:
        raise ValueError(f"No data files found in {data_dir}")

    # Chan's parallel algorithm accumulators (float64 for numerical stability)
    count = torch.zeros(num_fields, dtype=torch.int64)
    mean  = torch.zeros(num_fields, dtype=torch.float64)
    M2    = torch.zeros(num_fields, dtype=torch.float64)   # sum of squared deviations

    for filename in tqdm(file_list, desc="Computing stats", disable=not verbose):
        filepath = os.path.join(data_dir, filename)
        data     = torch.load(filepath, map_location='cpu', weights_only=False)
        x        = data.x.double()          # (N, num_fields)
        n_new    = x.shape[0]

        # Per-batch mean and M2
        mean_new = x.mean(dim=0)            # (num_fields,)
        M2_new   = ((x - mean_new) ** 2).sum(dim=0)   # (num_fields,)

        # Merge batch stats into running totals (Chan's formula)
        n_old       = count.double()
        n_combined  = n_old + n_new
        delta       = mean_new - mean
        mean        = (mean * n_old + mean_new * n_new) / n_combined
        # When n_old == 0 the last term vanishes; no division-by-zero risk
        M2          = M2 + M2_new + delta ** 2 * n_old * n_new / n_combined
        count       = count + n_new

    stds = torch.sqrt(M2 / count.double())

    # Build stats dict
    stats = {}
    for idx, field_name in enumerate(field_names):
        stats[field_name] = {
            'mean': mean[idx].item(),
            'std':  stds[idx].item(),
        }

    if verbose:
        print(f"\n{'='*80}\nNORMALIZATION STATISTICS\n{'='*80}")
        for field_name, field_stats in stats.items():
            print(f"{field_name:20s} | mean: {field_stats['mean']:12.6f} | std: {field_stats['std']:12.6f}")
        print(f"{'='*80}\n")

    return stats


def save_normalization_stats(stats: Dict[str, Dict[str, float]], save_path: str) -> None:
    """
    Save normalization statistics to disk.

    Args:
        stats: Dictionary mapping field names to {'mean': float, 'std': float}
        save_path: Path where to save the stats file (.pt file)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(stats, save_path)
    print(f"✓ Saved normalization stats to: {save_path}")


def load_normalization_stats(load_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load normalization statistics from disk.

    Args:
        load_path: Path to the stats file (.pt file)

    Returns:
        Dictionary mapping field names to {'mean': float, 'std': float}
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Normalization stats file not found: {load_path}")

    stats = torch.load(load_path, map_location='cpu', weights_only=False)
    print(f"✓ Loaded normalization stats from: {load_path}")

    return stats


def normalize_field(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Apply Z-score normalization to a field.

    Args:
        x: Input tensor (any shape)
        mean: Mean value for normalization
        std: Standard deviation for normalization

    Returns:
        Normalized tensor: (x - mean) / std
    """
    return (x - mean) / std


def denormalize_field(x_normalized: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Reverse Z-score normalization for a field.

    Args:
        x_normalized: Normalized tensor (any shape)
        mean: Mean value used for normalization
        std: Standard deviation used for normalization

    Returns:
        Denormalized tensor: x_normalized * std + mean
    """
    return x_normalized * std + mean
