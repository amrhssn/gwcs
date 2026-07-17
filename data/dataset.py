import os
import threading
import queue
from collections import OrderedDict
from typing import Optional, Callable, List, Union, Dict, Tuple

import torch
from torch.utils.data import random_split, Subset, DataLoader
from torch_geometric.data import Dataset, Data, Batch

from paths import ROOT_PROJECT_PATH
from data.utils_io import load_config
from models.sparsification_utils import (
    multilevel_importance_sampling,
    weighted_random_per_scale,
    random_mask,
    SparsityAnnealer
)


class GraphSignalDataset(Dataset):
    def __init__(
            self,
            base_data_dir: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            preload_to_memory: bool = False,
    ):
        self.root = base_data_dir
        self.raw_data_dir = os.path.join(base_data_dir, 'raw')
        self.graph_config = load_config(config_dir=base_data_dir, config_name='graph_config.json')
        self.signal_config = load_config(config_dir=base_data_dir, config_name='signal_config.json')
        self.gwt_config = load_config(config_dir=base_data_dir, config_name='gwt_config.json')

        self.file_list = sorted(
            [f for f in os.listdir(self.raw_dir) if f.endswith('.pt')],
            key=lambda x: int(x.split('.')[0])
        )

        # Preload all data to RAM
        self.preload_to_memory = preload_to_memory
        self.cached_data = []

        if self.preload_to_memory:
            import sys
            print(f"Preloading {len(self.file_list)} samples to RAM...")
            total_size = 0
            for idx in range(len(self.file_list)):
                data = self._load_from_disk(idx)
                # Add index to data for tracking (used by matrix cache)
                data['_idx'] = idx
                self.cached_data.append(data)
                total_size += sys.getsizeof(data)
                if (idx + 1) % 100 == 0:
                    print(f"  Loaded {idx + 1}/{len(self.file_list)} samples ({total_size / 1e9:.2f} GB)")
            print(f"✓ Preloaded {len(self.cached_data)} samples (~{total_size / 1e9:.2f} GB in RAM)")

        super().__init__(str(self.root), transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        """Required by PyG Dataset."""
        return self.file_list

    @property
    def processed_file_names(self) -> List[str]:
        """We don't use processed files, so return empty."""
        return []

    def len(self) -> int:
        """Return the number of samples."""
        return len(self.file_list)

    def _load_from_disk(self, idx: int) -> Dict:
        """Load a single sample from disk as a dictionary."""
        file_path = os.path.join(self.raw_dir, self.file_list[idx])
        data_dict = torch.load(file_path, map_location='cpu', weights_only=False)
        data_dict['_idx'] = idx
        return data_dict

    def get(self, idx: int) -> Dict:
        """Load and return a single graph signal sample as dictionary."""
        if self.preload_to_memory:
            return self.cached_data[idx]
        else:
            return self._load_from_disk(idx)


# ============================================================================
# Matrix Caches for GPU
# ============================================================================

class FullMatrixCache:
    """Cache ALL GWT/L_sym matrices on GPU."""

    def __init__(self, dataset: GraphSignalDataset, device: torch.device):
        self.device = device
        self.gwt_cache = {}
        self.lsym_cache = {}

        print("\n" + "=" * 80)
        print("FULL GPU CACHE MODE")
        print("=" * 80)
        print("Caching ALL matrices on GPU...")
        import time
        start = time.time()

        for idx in range(len(dataset)):
            data = dataset[idx]
            self.gwt_cache[idx] = data['G'].to(device)
            self.lsym_cache[idx] = data['L_sym'].to(device)

        elapsed = time.time() - start
        print(f"✓ Cached {len(self.gwt_cache)} matrices ({elapsed:.1f}s)")

        # Estimate memory
        sample_size = self.gwt_cache[0].numel() * 4 / 1024 / 1024  # MB
        total_mb = sample_size * 2 * len(self.gwt_cache)
        print(f"GPU memory used: ~{total_mb:.1f} MB")
        print("=" * 80)

    def get(self, indices: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get matrices for batch."""
        gwt = [self.gwt_cache[i] for i in indices]
        lsym = [self.lsym_cache[i] for i in indices]
        return gwt, lsym

    def get_stats(self) -> Dict:
        """Get cache statistics (always 100% hit rate for full cache)."""
        return {
            'cache_size': len(self.gwt_cache),
            'max_size': len(self.gwt_cache),
            'usage_pct': 100.0,
            'hits': 0,
            'misses': 0,
            'hit_rate': 100.0
        }

    def clear(self):
        """Clear cache."""
        self.gwt_cache.clear()
        self.lsym_cache.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


class LRUMatrixCache:
    """
    Fixed-size LRU cache for GPU matrices with background prefetching.

    Features:
    - CUDA streams for async transfers
    - Batched GPU transfers for cache misses
    - Proper prefetch queue management
    - Thread-safe operations
    """

    def __init__(
            self,
            dataset: GraphSignalDataset,
            device: torch.device,
            max_cache_size: int = 100,
            prefetch_size: int = 10
    ):
        self.dataset = dataset
        self.device = device
        self.max_cache_size = max_cache_size
        self.prefetch_size = prefetch_size

        # LRU cache: OrderedDict maintains insertion order
        self.gwt_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.lsym_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

        # Thread-safe lock
        self.lock = threading.Lock()

        # Prefetch queue and thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self.stop_prefetch = threading.Event()
        self.prefetch_thread = None

        # CUDA stream for async transfers
        if device.type == 'cuda':
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None

        # Statistics
        self.hits = 0
        self.misses = 0

        print("\n" + "=" * 80)
        print("LRU CACHE MODE (OPTIMIZED)")
        print("=" * 80)
        print(f"Cache size: {max_cache_size} samples")
        print(f"Prefetch queue: {prefetch_size} batches")
        print(f"CUDA streams: {'Enabled' if self.transfer_stream else 'Disabled'}")

        # Estimate memory usage
        sample_data = dataset[0]
        matrix_size = sample_data['G'].numel() * 4 / 1024 / 1024  # MB
        total_mb = matrix_size * 2 * max_cache_size
        print(f"Estimated GPU memory: ~{total_mb:.1f} MB")
        print(f"Per matrix pair: {matrix_size * 2:.1f} MB")
        print("=" * 80)

    def _evict_lru(self):
        """Evict least recently used item."""
        if len(self.gwt_cache) >= self.max_cache_size:
            old_idx = next(iter(self.gwt_cache))
            del self.gwt_cache[old_idx]
            del self.lsym_cache[old_idx]

    def get(self, indices: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get matrices for batch with batched GPU transfer for cache misses.
        Uses non_blocking transfers and single sync at the end.
        """
        gwt_list = []
        lsym_list = []

        # Separate hits from misses
        missing_indices = []
        missing_positions = []

        with self.lock:
            for pos, idx in enumerate(indices):
                if idx in self.gwt_cache:
                    # Cache hit - move to end (most recently used)
                    self.gwt_cache.move_to_end(idx)
                    self.lsym_cache.move_to_end(idx)
                    self.hits += 1
                else:
                    # Track missing for batched load
                    missing_indices.append(idx)
                    missing_positions.append(pos)
                    self.misses += 1

            # Batch load all missing matrices
            if missing_indices:
                # Collect CPU tensors first (fast - from RAM)
                gwt_cpu_list = []
                lsym_cpu_list = []
                for idx in missing_indices:
                    data = self.dataset[idx]
                    gwt_cpu_list.append(data['G'])
                    lsym_cpu_list.append(data['L_sym'])

                # Transfer all to GPU with non_blocking (overlapped)
                if self.transfer_stream is not None:
                    with torch.cuda.stream(self.transfer_stream):
                        for i, idx in enumerate(missing_indices):
                            self._evict_lru()
                            gwt = gwt_cpu_list[i].to(self.device, non_blocking=True)
                            lsym = lsym_cpu_list[i].to(self.device, non_blocking=True)
                            self.gwt_cache[idx] = gwt
                            self.lsym_cache[idx] = lsym
                    # Single sync after all transfers
                    self.transfer_stream.synchronize()
                else:
                    # CPU fallback
                    for i, idx in enumerate(missing_indices):
                        self._evict_lru()
                        gwt = gwt_cpu_list[i].to(self.device)
                        lsym = lsym_cpu_list[i].to(self.device)
                        self.gwt_cache[idx] = gwt
                        self.lsym_cache[idx] = lsym

            # Build result lists
            for idx in indices:
                gwt_list.append(self.gwt_cache[idx])
                lsym_list.append(self.lsym_cache[idx])

        return gwt_list, lsym_list

    def prefetch(self, indices: List[int]):
        """Prefetch matrices in background (non-blocking)."""
        if not self.prefetch_queue.full():
            try:
                self.prefetch_queue.put_nowait(indices)
            except queue.Full:
                pass

    def _prefetch_worker(self):
        """Background thread that prefetches matrices."""
        while not self.stop_prefetch.is_set():
            try:
                indices = self.prefetch_queue.get(timeout=0.1)

                with self.lock:
                    # Collect missing indices
                    missing = [idx for idx in indices if idx not in self.gwt_cache]

                    if missing:
                        # Batch load missing
                        gwt_cpu = [self.dataset[idx]['G'] for idx in missing]
                        lsym_cpu = [self.dataset[idx]['L_sym'] for idx in missing]

                        if self.transfer_stream is not None:
                            with torch.cuda.stream(self.transfer_stream):
                                for i, idx in enumerate(missing):
                                    if idx not in self.gwt_cache:
                                        self._evict_lru()
                                        self.gwt_cache[idx] = gwt_cpu[i].to(self.device, non_blocking=True)
                                        self.lsym_cache[idx] = lsym_cpu[i].to(self.device, non_blocking=True)
                            self.transfer_stream.synchronize()
                        else:
                            for i, idx in enumerate(missing):
                                if idx not in self.gwt_cache:
                                    self._evict_lru()
                                    self.gwt_cache[idx] = gwt_cpu[i].to(self.device)
                                    self.lsym_cache[idx] = lsym_cpu[i].to(self.device)

            except queue.Empty:
                continue

    def start_prefetching(self):
        """Start background prefetch thread."""
        if self.prefetch_thread is None:
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True
            )
            self.prefetch_thread.start()
            print("✓ Prefetch thread started")

    def stop_prefetching(self):
        """Stop background prefetch thread."""
        if self.prefetch_thread is not None:
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=1.0)
            print("✓ Prefetch thread stopped")

    def clear(self):
        """Clear all cached matrices."""
        with self.lock:
            self.gwt_cache.clear()
            self.lsym_cache.clear()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = 100 * self.hits / total if total > 0 else 0
            return {
                'cache_size': len(self.gwt_cache),
                'max_size': self.max_cache_size,
                'usage_pct': 100 * len(self.gwt_cache) / self.max_cache_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


# Type alias for cache
MatrixCache = Union[FullMatrixCache, LRUMatrixCache]


def create_matrix_cache(
        dataset: GraphSignalDataset,
        device: torch.device,
        cache_mode: str = 'lru',
        lru_cache_size: int = 100
) -> MatrixCache:
    """
    Factory function to create the appropriate matrix cache.

    Args:
        dataset: The graph signal dataset
        device: Target device (GPU/CPU)
        cache_mode: 'full' or 'lru'
        lru_cache_size: Max matrices in LRU cache (only used if cache_mode='lru')

    Returns:
        FullMatrixCache or LRUMatrixCache instance
    """
    if cache_mode == 'full':
        return FullMatrixCache(dataset, device)
    elif cache_mode == 'lru':
        cache = LRUMatrixCache(dataset, device, max_cache_size=lru_cache_size)
        cache.start_prefetching()
        return cache
    else:
        raise ValueError(f"Unknown cache mode: {cache_mode}. Use 'full' or 'lru'.")


# ============================================================================
# Data Splits
# ============================================================================

def create_data_splits(
        dataset: GraphSignalDataset,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    assert train_ratio + val_ratio + test_ratio == 1

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def create_splits_from_indices(
    dataset: GraphSignalDataset,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> tuple[Subset, Subset, Subset]:
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


# ============================================================================
# Collate Function (Optimized - lightweight)
# ============================================================================

def optimized_collate_fn(batch_list: List[Dict]) -> Batch:
    """
    Efficient collation - only batches small tensors.
    Large matrices (GWT, L_sym) are retrieved from GPU cache separately.
    """
    # Extract data and create Data objects
    data_objects = []
    indices = []
    for item in batch_list:
        data = Data(
            x=item['x'],
            edge_index=item['edge_index'],
            coeffs=item['coeffs'],
        )
        data_objects.append(data)
        indices.append(item['_idx'])

    # Batch using PyG
    batch = Batch.from_data_list(data_objects)

    # Store indices for GPU cache lookup
    batch._indices = indices

    return batch


# ============================================================================
# Mask Generation (GPU-optimized)
# ============================================================================

def generate_mask_on_gpu(
        coeffs: torch.Tensor,
        keep_ratio: float,
        mode: str = "mlis"
) -> torch.Tensor:
    """
    Generate sparsification mask on GPU.

    Args:
        coeffs: Coefficient tensor on GPU
        keep_ratio: Fraction of coefficients to keep
        mode: Mask generation mode - 'mlis', 'weighted_random_per_scale', or 'random'

    Returns:
        Binary mask tensor on GPU
    """
    if mode == "mlis":
        return multilevel_importance_sampling(coeffs=coeffs, keep_ratio=keep_ratio)
    elif mode == "weighted_random_per_scale":
        return weighted_random_per_scale(coeffs=coeffs, keep_ratio=keep_ratio)
    elif mode == "random":
        return random_mask(coeffs=coeffs, keep_ratio=keep_ratio)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mlis', 'weighted_random_per_scale', or 'random'.")


# ============================================================================
# DataLoader Creation (Optimized)
# ============================================================================

def create_optimized_dataloaders(
        train_dataset: Subset,
        val_dataset: Subset,
        test_dataset: Subset,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create optimized dataloaders using lightweight collate function.

    When data is preloaded to RAM, use num_workers=0 to avoid overhead.
    Matrices are handled separately via GPU cache.

    Args:
        train_dataset: Training subset
        val_dataset: Validation subset
        test_dataset: Test subset
        batch_size: Batch size
        num_workers: Number of DataLoader workers (0 recommended when data is preloaded)
        shuffle: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Build loader kwargs
    base_kwargs = {
        'batch_size': batch_size,
        'collate_fn': optimized_collate_fn,
        'num_workers': num_workers,
    }

    # These options only work with num_workers > 0
    if num_workers > 0:
        base_kwargs.update({
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
        })

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        **base_kwargs
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **base_kwargs
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        **base_kwargs
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Legacy functions (kept for backward compatibility)
# ============================================================================

def custom_collate_fn(
        data_list: List[Data],
        generate_mask: bool = False,
        keep_ratio: float = 0.1,
        mask_generation_mode: str = "weighted_random_per_scale",
) -> Union[Data, Batch]:
    """Legacy collate function - kept for backward compatibility."""
    assert mask_generation_mode in ["mlis", "weighted_random_per_scale", "random"]

    # Extract dense matrices before batching
    gwt_matrices = [data.G for data in data_list]
    L_sym_matrices = [data.L_sym for data in data_list]
    eigenvalues_list = [data.eigenvalues for data in data_list]
    eigenvectors_list = [data.eigenvectors for data in data_list]

    # Extract metadata dictionaries before batching
    graph_metadata_list = [data.graph_metadata for data in data_list]
    signal_metadata_list = [data.signal_metadata for data in data_list]

    if generate_mask:
        for data in data_list:
            data.sparsification_mask = torch.ones_like(data.coeffs)

    cleaned_data_list = []
    for data in data_list:
        cleaned_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            coeffs=data.coeffs,
        )
        if hasattr(data, 'sparsification_mask'):
            cleaned_data.sparsification_mask = data.sparsification_mask
        cleaned_data_list.append(cleaned_data)

    batch = Batch.from_data_list(cleaned_data_list)

    batch.gwt_matrices = gwt_matrices
    batch.L_sym_matrices = L_sym_matrices
    batch.eigenvalues = eigenvalues_list
    batch.eigenvectors = eigenvectors_list
    batch.graph_metadata = graph_metadata_list
    batch.signal_metadata = signal_metadata_list

    batch.generate_mask = generate_mask
    batch.keep_ratio = keep_ratio
    batch.mask_generation_mode = mask_generation_mode

    return batch


def create_dataloaders(
        train_dataset: Subset,
        val_dataset: Subset,
        test_dataset: Subset,
        batch_size: int = 32,
        generate_mask: bool = False,
        keep_ratio: float = 0.1,
        mask_generation_mode: str = "weighted_random_per_scale",
        sparsity_scheduler: Optional[SparsityAnnealer] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Legacy dataloader creation - kept for backward compatibility."""
    assert mask_generation_mode in ["mlis", "weighted_random_per_scale", "random"]

    def collate_fn(data_list):
        if sparsity_scheduler is not None:
            current_keep_ratio = sparsity_scheduler.get_keep_ratio()
        else:
            current_keep_ratio = keep_ratio
        return custom_collate_fn(data_list, generate_mask, current_keep_ratio, mask_generation_mode)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, test_loader


def main():
    # Create dataset
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    base_data_dir = os.path.join(local_storage_dir, 'data', '2026-01-06_15-41-27')

    print("=" * 80)
    print("CPU→GPU TRANSFER BENCHMARK")
    print("=" * 80)

    # Test with and without RAM preloading
    for preload in [False, True]:
        print(f"\n{'=' * 80}")
        print(f"Testing with preload_to_memory={preload}")
        print(f"{'=' * 80}\n")

        dataset = GraphSignalDataset(
            base_data_dir=base_data_dir,
            preload_to_memory=preload
        )

        print(f"Total samples: {len(dataset)}")

        # Split into train/val/test
        train_dataset, val_dataset, test_dataset = create_data_splits(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Test different worker configurations
        worker_configs = [
            (0, 1, "no_prefetch"),
            (4, 2, "prefetch"),
            (8, 4, "prefetch"),
        ]

        for num_workers, prefetch, desc in worker_configs:
            print(f"\n--- Workers={num_workers}, Prefetch={prefetch if num_workers > 0 else 'N/A'} ---")

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=32,
                shuffle=True,
                collate_fn=lambda data_list: custom_collate_fn(data_list, True, 0.1, "weighted_random_per_scale"),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=prefetch if num_workers > 0 else None,
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Device: {device}")

            print("Warming up...")
            for _ in enumerate(train_loader):
                break

            print("\nBenchmarking...")
            import time

            timings = {
                'cpu_load': [],
                'cpu_to_gpu': [],
                'total': []
            }

            t0 = time.time()
            for batch_index, batch in enumerate(train_loader):
                t_load = time.time() - t0

                t_transfer_start = time.time()

                batch = batch.to(device, non_blocking=True)
                batch.gwt_matrices = [mat.to(device, non_blocking=True) for mat in batch.gwt_matrices]
                batch.L_sym_matrices = [mat.to(device, non_blocking=True) for mat in batch.L_sym_matrices]

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                t_transfer = time.time() - t_transfer_start
                t_total = t_load + t_transfer

                timings['cpu_load'].append(t_load)
                timings['cpu_to_gpu'].append(t_transfer)
                timings['total'].append(t_total)

                if batch_index < 5:
                    print(f"  Batch {batch_index}: "
                          f"Load={t_load:.3f}s, "
                          f"Transfer={t_transfer:.3f}s, "
                          f"Total={t_total:.3f}s | "
                          f"Batch size={batch.num_graphs}, "
                          f"Nodes={batch.x.shape[0]}")

                t0 = time.time()

                if batch_index >= 9:
                    break

            if len(timings['cpu_load']) > 1:
                import numpy as np
                print(f"\n  Statistics (batches 1-9):")
                print(
                    f"    CPU Load:     {np.mean(timings['cpu_load'][1:]):.3f}s ± {np.std(timings['cpu_load'][1:]):.3f}s")
                print(
                    f"    CPU→GPU:      {np.mean(timings['cpu_to_gpu'][1:]):.3f}s ± {np.std(timings['cpu_to_gpu'][1:]):.3f}s")
                print(f"    Total:        {np.mean(timings['total'][1:]):.3f}s ± {np.std(timings['total'][1:]):.3f}s")

                avg_nodes = batch.x.shape[0]
                avg_edges = batch.edge_index.shape[1]
                estimated_mb = (avg_nodes * 4 + avg_edges * 2 * 8 + avg_nodes * 5 * 4) / 1024 / 1024
                estimated_mb += 32 * 2 * (1024 * 1024 * 4) / 1024 / 1024

                avg_transfer_time = np.mean(timings['cpu_to_gpu'][1:])
                bandwidth = estimated_mb / avg_transfer_time if avg_transfer_time > 0 else 0
                print(f"    Est. Bandwidth: {bandwidth:.1f} MB/s")

            del train_loader
            if device.type == 'cuda':
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()