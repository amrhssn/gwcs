"""
Parallel data processing for turbulent_radiative_layer_2D dataset.

This module provides parallelized preprocessing using multiprocessing.Pool
with PRE-COMPUTED Chebyshev cache to avoid redundant polynomial computation.

Key optimizations:
1. Load ALL sample data in main process ONCE
2. PRE-COMPUTE Chebyshev polynomials in main process (the expensive part!)
3. Pass pre-computed cache to workers
4. Workers only do matrix-vector multiplications (fast)

Usage:
    from processing_steps.data_processing_parallel import preprocess_turbulent_parallel

    preprocess_turbulent_parallel(
        dataset_name="turbulent_radiative_layer_2D",
        raw_dataset_dir="/path/to/raw",
        processed_dataset_dir="/path/to/output",
        split="train",
        num_workers=8
    )
"""

import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Literal, Optional, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import grid, remove_self_loops, is_undirected, to_undirected
from the_well.data import WellDataset
from tqdm import tqdm

from data.utils_graph import compute_sparse_L_sym
from graph_wavelet_transform.chebyshev import chebyshev_gwt_analysis
# =============================================================================
# Global worker state (initialized once per worker via Pool initializer)
# Now includes PRE-COMPUTED Chebyshev data!
# =============================================================================
_worker_L_sym_sparse = None
_worker_edge_index = None
_worker_pos = None
_worker_visualization_pos = None
_worker_chebyshev_data = None  # Pre-computed Chebyshev polynomials


def _init_worker(
        L_sym_sparse_data: dict,
        edge_index_list: list,
        pos_list: list,
        visualization_pos_list: list,
        chebyshev_precomputed: dict,
):
    """
    Initialize worker process state with PRE-COMPUTED Chebyshev data.

    The expensive Chebyshev polynomial computation is done ONCE in the main
    process and passed here. Workers only do fast matrix-vector products.
    """
    global _worker_L_sym_sparse, _worker_edge_index, _worker_pos
    global _worker_visualization_pos, _worker_chebyshev_data

    # Reconstruct tensors from serialized data
    _worker_pos = torch.tensor(pos_list, dtype=torch.float32)
    _worker_edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    _worker_visualization_pos = torch.tensor(visualization_pos_list, dtype=torch.float32)

    # Reconstruct sparse L_sym
    _worker_L_sym_sparse = torch.sparse_coo_tensor(
        indices=torch.tensor(L_sym_sparse_data['indices'], dtype=torch.long),
        values=torch.tensor(L_sym_sparse_data['values'], dtype=torch.float32),
        size=L_sym_sparse_data['size']
    ).coalesce()

    # Reconstruct pre-computed Chebyshev data
    _worker_chebyshev_data = {
        'chebyshev_coeffs': torch.tensor(chebyshev_precomputed['chebyshev_coeffs'], dtype=torch.float32),
        'chebyshev_eigenvalue_samples': torch.tensor(chebyshev_precomputed['chebyshev_eigenvalue_samples'],
                                                     dtype=torch.float32),
        'L_sym_tilde': torch.sparse_coo_tensor(
            indices=torch.tensor(chebyshev_precomputed['L_sym_tilde_indices'], dtype=torch.long),
            values=torch.tensor(chebyshev_precomputed['L_sym_tilde_values'], dtype=torch.float32),
            size=chebyshev_precomputed['L_sym_tilde_size']
        ).coalesce(),
        'cache_key': chebyshev_precomputed['cache_key'],
    }


def _apply_gwt_with_precomputed(
        x: torch.Tensor,
        L_sym_tilde: torch.Tensor,
        chebyshev_coeffs: torch.Tensor,
        num_nodes: int,
        num_filters: int,
        chebyshev_max_degree: int,
) -> torch.Tensor:
    """
    Apply GWT using pre-computed Chebyshev coefficients.

    This is the FAST part - just matrix-vector multiplications.
    The expensive polynomial fitting was done in the main process.
    """
    coeffs = torch.zeros(num_nodes, num_filters, dtype=torch.float32)

    # Apply Chebyshev polynomial approximation for each filter
    for filter_idx in range(num_filters):
        # Get Chebyshev coefficients for this filter
        filter_coeffs = chebyshev_coeffs[filter_idx]  # Shape: (chebyshev_max_degree + 1,)

        # Chebyshev recurrence: T_0(x) = 1, T_1(x) = x, T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
        T_prev_prev = x.clone()  # T_0 * x = x
        result = filter_coeffs[0] * T_prev_prev

        if chebyshev_max_degree >= 1:
            # T_1(L_tilde) * x = L_tilde @ x
            T_prev = torch.sparse.mm(L_sym_tilde, x.unsqueeze(1)).squeeze(1)
            result = result + filter_coeffs[1] * T_prev

            for k in range(2, chebyshev_max_degree + 1):
                # T_k(L_tilde) * x = 2 * L_tilde @ T_{k-1} - T_{k-2}
                T_curr = 2 * torch.sparse.mm(L_sym_tilde, T_prev.unsqueeze(1)).squeeze(1) - T_prev_prev
                result = result + filter_coeffs[k] * T_curr
                T_prev_prev = T_prev
                T_prev = T_curr

        coeffs[:, filter_idx] = result

    return coeffs


def _process_single_sample_fast(
        args: Tuple[int, torch.Tensor, int, int],
        num_nodes: int,
        num_fields: int,
        num_filters: int,
        num_rows: int,
        num_columns: int,
        num_scales: int,
        K: int,
        field_names: list,
        field_to_index: dict,
        index_to_field: dict,
        gwt_scales: list,
        lambda_min: float,
        lambda_max: float,
        bandpass_lambda_min: float,
        chebyshev_max_degree: int,
        processed_dataset_dir: str,
) -> None:
    """
    Process a single sample using PRE-COMPUTED Chebyshev data.

    This is now FAST because the expensive polynomial computation
    was already done in the main process.
    """
    global _worker_L_sym_sparse, _worker_edge_index, _worker_pos
    global _worker_visualization_pos, _worker_chebyshev_data

    # Unpack pre-loaded data
    sample_index, x, trajectory_index, time_index = args

    # Initialize GWT coefficients tensor
    coeffs = torch.zeros(num_nodes, num_fields, num_filters, dtype=torch.float32)

    # Apply GWT using pre-computed Chebyshev data (FAST!)
    for field_index in range(num_fields):
        coeffs[:, field_index, :] = _apply_gwt_with_precomputed(
            x=x[:, field_index],
            L_sym_tilde=_worker_chebyshev_data['L_sym_tilde'],
            chebyshev_coeffs=_worker_chebyshev_data['chebyshev_coeffs'],
            num_nodes=num_nodes,
            num_filters=num_filters,
            chebyshev_max_degree=chebyshev_max_degree,
        )

    # Construct metadata dictionaries
    graph_metadata = {
        "graph_type": "grid",
        "num_nodes": num_nodes,
        "num_edges": _worker_edge_index.size(1),
        "num_rows": num_rows,
        "num_columns": num_columns,
        "visualization_pos": _worker_visualization_pos.tolist(),
        "L_sym_sparse": _worker_L_sym_sparse,
    }

    gwt_metadata = {
        "type": "chebyshev_approximation",
        "num_filters": num_filters,
        "scales": gwt_scales,
        "num_scales": num_scales,
        "bandpass_lambda_min": bandpass_lambda_min,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "K": K,
        "chebyshev_max_degree": chebyshev_max_degree,
        "chebyshev_cache_key": _worker_chebyshev_data['cache_key'],
        "chebyshev_coeffs": _worker_chebyshev_data['chebyshev_coeffs'],
        "chebyshev_eigenvalue_samples": _worker_chebyshev_data['chebyshev_eigenvalue_samples'],
        "chebyshev_L_sym_tilde": _worker_chebyshev_data['L_sym_tilde'],
    }

    signal_metadata = {
        "sample_index": sample_index,
        "trajectory_index": trajectory_index,
        "time_index": time_index,
        "field_names": field_names,
        "num_fields": num_fields,
        "field_to_index": field_to_index,
        "index_to_field": index_to_field,
        "x_dims": "(num_nodes, num_fields)",
        "coeffs_dims": "(num_nodes, num_fields, num_filters)",
    }

    # Create PyG Data object
    data = Data(
        x=x,
        coeffs=coeffs,
        pos=_worker_pos,
        edge_index=_worker_edge_index,
        graph_metadata=graph_metadata,
        gwt_metadata=gwt_metadata,
        signal_metadata=signal_metadata
    )

    # Save to disk
    save_path = os.path.join(processed_dataset_dir, f'{sample_index}.pt')
    torch.save(data, save_path)


def _get_trajectory_and_timestep_info(sample_index: int, metadata) -> Tuple[int, int]:
    """Compute trajectory_index and time_index for a given sample."""
    cumulative_samples = 0
    for file_idx in range(metadata.n_files):
        n_steps = metadata.n_steps_per_trajectory[file_idx]
        n_trajectories = metadata.n_trajectories_per_file[file_idx]

        for traj_index in range(n_trajectories):
            if sample_index < cumulative_samples + n_steps:
                timestep = sample_index - cumulative_samples
                trajectory_id = cumulative_samples // n_steps
                return trajectory_id, timestep
            cumulative_samples += n_steps
    raise IndexError(f"Sample index {sample_index} out of range")


def preprocess_turbulent_parallel(
        dataset_name: str,
        raw_dataset_dir: str,
        processed_dataset_dir: str,
        split: Literal["train", "valid", "test"],
        num_workers: Optional[int] = None,
        sample_indices: Optional[List[int]] = None,
        chunk_size: int = 1,
) -> None:
    """
    Parallel preprocessing with PRE-COMPUTED Chebyshev polynomials.

    The key insight: Chebyshev polynomial computation is expensive (~18s)
    but only needs to be done ONCE for a given graph structure. We do this
    in the main process and share the result with all workers.

    Args:
        dataset_name: Name of the dataset
        raw_dataset_dir: Path to raw data directory
        processed_dataset_dir: Path to save processed data
        split: Data split to process
        num_workers: Number of parallel workers
        sample_indices: Optional list of sample indices to process
        chunk_size: Samples per worker task
    """
    import time
    os.makedirs(processed_dataset_dir, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"\n{'=' * 70}")
    print(f"PARALLEL Processing (PRE-COMPUTED CHEBYSHEV): {dataset_name}")
    print(f"Split: {split}")
    print(f"Workers: {num_workers}")
    print(f"Output: {processed_dataset_dir}")
    print(f"{'=' * 70}\n")

    # =========================================================================
    # PHASE 1: Load dataset
    # =========================================================================
    print("Phase 1: Loading dataset...")
    t0 = time.perf_counter()

    well_base_path = os.path.dirname(raw_dataset_dir)
    dataset = WellDataset(
        well_base_path=well_base_path,
        well_dataset_name=dataset_name,
        well_split_name=split,
        n_steps_input=1,
        n_steps_output=0,
        use_normalization=False,
    )

    field_names = [name for group in dataset.metadata.field_names.values() for name in group]
    num_fields = len(field_names)
    index_to_field = {idx: field for idx, field in enumerate(field_names)}
    field_to_index = {field: idx for idx, field in enumerate(field_names)}

    total_samples = len(dataset)
    if sample_indices is not None:
        indices_to_process = sample_indices
    else:
        indices_to_process = list(range(total_samples))

    dataset_load_time = time.perf_counter() - t0
    print(f"  ✓ Dataset loaded in {dataset_load_time:.2f}s ({len(indices_to_process)} samples)")

    # =========================================================================
    # PHASE 2: Construct graph
    # =========================================================================
    print("\nPhase 2: Constructing graph...")
    t0 = time.perf_counter()

    pos = dataset[0]['space_grid'].reshape(-1, 2)
    num_rows, num_columns = dataset[0]['space_grid'].shape[:2]
    num_nodes = num_rows * num_columns
    edge_index, visualization_pos = grid(height=num_rows, width=num_columns)
    edge_index, _ = remove_self_loops(edge_index)
    if not is_undirected(edge_index=edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index=edge_index, num_nodes=num_nodes)

    L_sym_sparse = compute_sparse_L_sym(edge_index=edge_index, edge_weight=None, num_nodes=num_nodes)

    graph_time = time.perf_counter() - t0
    print(f"  ✓ Graph: {num_rows}x{num_columns} = {num_nodes} nodes ({graph_time:.2f}s)")

    # GWT configs
    lambda_max = 2.0
    K = 1_000
    bandpass_lambda_min = lambda_max / K
    num_scales = 4
    num_filters = num_scales + 1
    gwt_scales = np.logspace(np.log10(1 / lambda_max), np.log10(1 / bandpass_lambda_min), num_scales).tolist()
    lambda_min = 0.0
    chebyshev_max_degree = 30

    # =========================================================================
    # PHASE 3: PRE-COMPUTE CHEBYSHEV POLYNOMIALS (THE EXPENSIVE PART!)
    # =========================================================================
    print("\nPhase 3: Pre-computing Chebyshev polynomials (one-time cost)...")
    t0 = time.perf_counter()

    # Use first sample to trigger Chebyshev computation
    sample = dataset[0]
    x_dummy = sample['input_fields'].squeeze().reshape(num_nodes, num_fields)

    chebyshev_cache = {}
    _, cache_key, chebyshev_cache = chebyshev_gwt_analysis(
        x=x_dummy[:, 0],
        L_sym_sparse=L_sym_sparse,
        edge_index=edge_index,
        num_nodes=num_nodes,
        gwt_scales=gwt_scales,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        chebyshev_max_degree=chebyshev_max_degree,
        cache=chebyshev_cache,
    )

    # Extract pre-computed data for serialization
    cached_data = chebyshev_cache[cache_key]
    L_sym_tilde = cached_data['L_sym_tilde'].coalesce()

    chebyshev_precomputed = {
        'chebyshev_coeffs': cached_data['chebyshev_coeffs'].tolist(),
        'chebyshev_eigenvalue_samples': cached_data['chebyshev_eigenvalue_samples'].tolist(),
        'L_sym_tilde_indices': L_sym_tilde.indices().tolist(),
        'L_sym_tilde_values': L_sym_tilde.values().tolist(),
        'L_sym_tilde_size': list(L_sym_tilde.size()),
        'cache_key': cache_key,
    }

    chebyshev_time = time.perf_counter() - t0
    print(f"  ✓ Chebyshev polynomials computed in {chebyshev_time:.2f}s")
    print(f"  This cost is now AMORTIZED across all {len(indices_to_process)} samples!")

    # =========================================================================
    # PHASE 4: Pre-load sample data
    # =========================================================================
    print(f"\nPhase 4: Pre-loading {len(indices_to_process)} samples...")
    t0 = time.perf_counter()

    preloaded_data: List[Tuple[int, torch.Tensor, int, int]] = []
    for idx in tqdm(indices_to_process, desc="Loading", unit="sample"):
        sample = dataset[idx]
        trajectory_index, time_index = _get_trajectory_and_timestep_info(idx, dataset.metadata)
        x = sample['input_fields'].squeeze().reshape(num_nodes, num_fields)
        preloaded_data.append((idx, x, trajectory_index, time_index))

    preload_time = time.perf_counter() - t0
    print(f"  ✓ Pre-loaded in {preload_time:.2f}s")

    del dataset

    # =========================================================================
    # PHASE 5: Serialize for multiprocessing
    # =========================================================================
    print("\nPhase 5: Preparing workers...")

    L_sym_coalesced = L_sym_sparse.coalesce()
    L_sym_sparse_data = {
        'indices': L_sym_coalesced.indices().tolist(),
        'values': L_sym_coalesced.values().tolist(),
        'size': list(L_sym_coalesced.size()),
    }

    process_fn = partial(
        _process_single_sample_fast,
        num_nodes=num_nodes,
        num_fields=num_fields,
        num_filters=num_filters,
        num_rows=num_rows,
        num_columns=num_columns,
        num_scales=num_scales,
        K=K,
        field_names=field_names,
        field_to_index=field_to_index,
        index_to_field=index_to_field,
        gwt_scales=gwt_scales,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        bandpass_lambda_min=bandpass_lambda_min,
        chebyshev_max_degree=chebyshev_max_degree,
        processed_dataset_dir=processed_dataset_dir,
    )

    actual_workers = min(num_workers, len(preloaded_data))
    print(f"  Using {actual_workers} workers")

    # =========================================================================
    # PHASE 6: Parallel processing (NOW FAST!)
    # =========================================================================
    print(f"\nPhase 6: Processing in parallel...")
    t0 = time.perf_counter()

    with Pool(
            actual_workers,
            initializer=_init_worker,
            initargs=(
                    L_sym_sparse_data,
                    edge_index.tolist(),
                    pos.tolist(),
                    visualization_pos.tolist(),
                    chebyshev_precomputed,
            ),
    ) as pool:
        list(tqdm(
            pool.imap_unordered(process_fn, preloaded_data, chunksize=chunk_size),
            total=len(preloaded_data),
            desc="Processing",
            unit="sample"
        ))

    process_time = time.perf_counter() - t0
    total_time = dataset_load_time + graph_time + chebyshev_time + preload_time + process_time

    print(f"\n{'=' * 70}")
    print(f"✓ COMPLETED!")
    print(f"{'=' * 70}")
    print(f"\nTiming breakdown:")
    print(f"  Dataset load:     {dataset_load_time:>8.2f}s")
    print(f"  Graph construct:  {graph_time:>8.2f}s")
    print(f"  Chebyshev (1x):   {chebyshev_time:>8.2f}s  ← One-time cost!")
    print(f"  Data preload:     {preload_time:>8.2f}s")
    print(f"  Parallel proc:    {process_time:>8.2f}s  ({len(preloaded_data) / process_time:.2f} samples/sec)")
    print(f"  ─────────────────────────────")
    print(f"  Total:            {total_time:>8.2f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from dotenv import load_dotenv
    from paths import ROOT_PROJECT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    load_dotenv(Path(ROOT_PROJECT_PATH) / ".env")
    raw_base = os.getenv("TURBULENT_DATA_BASE_PATH")
    raw_dataset_dir = str(Path(ROOT_PROJECT_PATH) / raw_base)

    output_dir = args.output or str(Path(ROOT_PROJECT_PATH) / "local_storage" / "processed_parallel" / args.split)
    sample_indices = list(range(args.num_samples)) if args.num_samples else None

    preprocess_turbulent_parallel(
        dataset_name="turbulent_radiative_layer_2D",
        raw_dataset_dir=raw_dataset_dir,
        processed_dataset_dir=output_dir,
        split=args.split,
        num_workers=args.workers,
        sample_indices=sample_indices,
    )