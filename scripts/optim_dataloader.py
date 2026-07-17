"""
Optimized DataLoader Benchmark with LRU Cache
==============================================
Tests: RAM preload + workers + GPU transfer + mask generation + LRU caching

Features:
1. Full GPU cache (for small datasets that fit in GPU memory)
2. LRU cache with prefetching (for large datasets)
3. CUDA streams for async transfers
4. Batched GPU transfers for cache misses
5. Proper epoch-based iteration

Run: python scripts/optim_dataloader.py
"""

import os
import sys
import time
import threading
import queue
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# Dataset
# ============================================================================

class GraphDataset(Dataset):
    """Lightweight dataset with RAM preloading"""

    def __init__(self, data_dir: str, preload: bool = True):
        self.data_dir = os.path.join(data_dir, 'raw')
        self.files = sorted(
            [f for f in os.listdir(self.data_dir) if f.endswith('.pt')],
            key=lambda x: int(x.split('.')[0])
        )
        print(f"Found {len(self.files)} files")

        self.cache = []
        if preload:
            print(f"Preloading {len(self.files)} samples to RAM...")
            for i in range(len(self.files)):
                if i % 100 == 0:
                    print(f"  Loading {i}/{len(self.files)}")
                data = self._load(i)
                # Add index to data for tracking
                data['_idx'] = i
                self.cache.append(data)
            print("✓ Preloaded to RAM")
        self.preload = preload

    def _load(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        return torch.load(path, map_location='cpu', weights_only=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.cache[idx] if self.preload else self._load(idx)


# ============================================================================
# Mask Generation (GPU-optimized)
# ============================================================================

def generate_mask_gpu(coeffs: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    Weighted random mask per scale (on GPU)
    Selects top-k coefficients by magnitude within each scale
    """
    num_nodes, num_filters = coeffs.shape
    num_keep = max(1, int(num_nodes * num_filters * keep_ratio))

    # Flatten and get magnitudes
    coeffs_flat = coeffs.flatten()
    magnitudes = coeffs_flat.abs()

    # Weighted sampling: probability ∝ magnitude
    probs = magnitudes / (magnitudes.sum() + 1e-8)

    # Sample indices
    indices = torch.multinomial(probs, num_keep, replacement=False)

    # Create mask
    mask_flat = torch.zeros_like(coeffs_flat)
    mask_flat[indices] = 1.0

    return mask_flat.view(num_nodes, num_filters)


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch_list: List[Dict]) -> Batch:
    """Efficient collation - only small tensors"""

    # Extract data
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

    # Store matrix indices for GPU cache lookup
    batch._indices = indices

    return batch


# ============================================================================
# Full GPU Matrix Cache (for small datasets)
# ============================================================================

class FullMatrixCache:
    """Cache ALL GWT/L_sym matrices on GPU"""

    def __init__(self, dataset: GraphDataset, device: torch.device):
        self.device = device
        self.gwt_cache = {}
        self.lsym_cache = {}

        print("\n" + "=" * 80)
        print("FULL GPU CACHE MODE")
        print("=" * 80)
        print("Caching ALL matrices on GPU...")
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
        """Get matrices for batch"""
        gwt = [self.gwt_cache[i] for i in indices]
        lsym = [self.lsym_cache[i] for i in indices]
        return gwt, lsym

    def get_stats(self) -> Dict:
        """Get cache statistics (always 100% hit rate for full cache)"""
        return {
            'cache_size': len(self.gwt_cache),
            'max_size': len(self.gwt_cache),
            'usage_pct': 100.0,
            'hits': 0,
            'misses': 0,
            'hit_rate': 100.0
        }

    def clear(self):
        """Clear cache"""
        self.gwt_cache.clear()
        self.lsym_cache.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


# ============================================================================
# LRU Matrix Cache (for large datasets) - IMPROVED
# ============================================================================

class LRUMatrixCache:
    """
    Fixed-size LRU cache for GPU matrices with background prefetching.

    Improvements:
    - CUDA streams for async transfers
    - Batched GPU transfers for cache misses
    - Proper prefetch queue management
    - Thread-safe operations
    """

    def __init__(
            self,
            dataset: GraphDataset,
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
        """Evict least recently used item"""
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
        """Prefetch matrices in background (non-blocking)"""
        if not self.prefetch_queue.full():
            try:
                self.prefetch_queue.put_nowait(indices)
            except queue.Full:
                pass

    def _prefetch_worker(self):
        """Background thread that prefetches matrices"""
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
        """Start background prefetch thread"""
        if self.prefetch_thread is None:
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True
            )
            self.prefetch_thread.start()
            print("✓ Prefetch thread started")

    def stop_prefetching(self):
        """Stop background prefetch thread"""
        if self.prefetch_thread is not None:
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=1.0)
            print("✓ Prefetch thread stopped")

    def clear(self):
        """Clear all cached matrices"""
        with self.lock:
            self.gwt_cache.clear()
            self.lsym_cache.clear()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def get_stats(self) -> Dict:
        """Get cache statistics"""
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


# ============================================================================
# Benchmark Function - Full Epoch Iteration
# ============================================================================

def run_training_simulation(
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        cache_mode: str = 'lru',
        lru_cache_size: int = 800,
        num_epochs: int = 10,
        shuffle: bool = True,
        verbose: bool = True
):
    """
    Simulate training loop: iterate over ALL batches for multiple epochs.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: DataLoader workers (0 recommended when data is preloaded)
        cache_mode: 'full' or 'lru'
        lru_cache_size: Max matrices in LRU cache
        num_epochs: Number of epochs to run
        shuffle: Whether to shuffle data each epoch
        verbose: Print per-batch info
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 80)
    print("TRAINING SIMULATION - FULL EPOCH ITERATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"Cache mode: {cache_mode}")
    print(f"Epochs: {num_epochs}")
    print(f"Shuffle: {shuffle}")

    # Load dataset
    dataset = GraphDataset(data_dir, preload=True)
    num_samples = len(dataset)
    num_batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    print(f"Dataset size: {num_samples}")
    print(f"Batches per epoch: {num_batches_per_epoch}")

    # Create cache
    if cache_mode == 'full':
        cache = FullMatrixCache(dataset, device)
    elif cache_mode == 'lru':
        cache = LRUMatrixCache(
            dataset, device,
            max_cache_size=lru_cache_size,
            prefetch_size=10
        )
        cache.start_prefetching()
    else:
        raise ValueError(f"Unknown cache mode: {cache_mode}")

    # Create dataloader with conditional parameters
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
    }

    # These options only work with num_workers > 0
    if num_workers > 0:
        loader_kwargs.update({
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
        })

    loader = DataLoader(dataset, **loader_kwargs)

    # Warmup
    print("\nWarming up...")
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        _ = cache.get(batch._indices)
        break
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Training simulation
    print(f"\n{'=' * 80}")
    print("STARTING TRAINING SIMULATION")
    print(f"{'=' * 80}\n")

    epoch_stats = []

    for epoch in range(num_epochs):
        epoch_start = time.time()

        epoch_timings = {
            'data_load': [],
            'gpu_transfer': [],
            'cache_lookup': [],
            'mask_gen': [],
            'batch_total': []
        }

        # Pre-collect all batches for this epoch (enables proper prefetching)
        all_batches = list(loader)

        t0 = time.time()
        for batch_idx, batch in enumerate(all_batches):
            batch_start = time.time()

            # Time: Data loading (already done by list())
            t_load = batch_start - t0

            # Time: Transfer to GPU
            t1 = time.time()
            batch = batch.to(device, non_blocking=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_transfer = time.time() - t1

            # Time: Get matrices from cache
            t2 = time.time()
            gwt_mats, lsym_mats = cache.get(batch._indices)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_cache = time.time() - t2

            # Prefetch NEXT batch (proper fix - use pre-collected batches)
            if cache_mode == 'lru' and batch_idx + 1 < len(all_batches):
                next_indices = all_batches[batch_idx + 1]._indices
                cache.prefetch(next_indices)

            # Time: Generate mask on GPU
            t3 = time.time()
            mask = generate_mask_gpu(batch.coeffs, keep_ratio=0.1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_mask = time.time() - t3

            # Record timings
            t_batch_total = time.time() - batch_start
            epoch_timings['data_load'].append(t_load)
            epoch_timings['gpu_transfer'].append(t_transfer)
            epoch_timings['cache_lookup'].append(t_cache)
            epoch_timings['mask_gen'].append(t_mask)
            epoch_timings['batch_total'].append(t_batch_total)

            # Print progress
            if verbose:
                print(f"  Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1}/{len(all_batches)} | "
                      f"Load={t_load:.3f}s Cache={t_cache:.3f}s Mask={t_mask:.3f}s | "
                      f"Nodes={batch.x.shape[0]}")

            t0 = time.time()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_batch_time = np.mean(epoch_timings['batch_total'])
        throughput = num_samples / epoch_time

        epoch_stats.append({
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            'avg_batch_time': avg_batch_time,
            'throughput': throughput,
            'avg_cache_lookup': np.mean(epoch_timings['cache_lookup']),
            'avg_mask_gen': np.mean(epoch_timings['mask_gen']),
        })

        print(f"\n{'─' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs} Complete")
        print(f"  Time: {epoch_time:.2f}s | Throughput: {throughput:.1f} samples/sec")
        print(f"  Avg batch: {avg_batch_time:.3f}s | Cache lookup: {np.mean(epoch_timings['cache_lookup']):.3f}s")
        print(f"{'─' * 60}\n")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    avg_epoch_time = np.mean([s['epoch_time'] for s in epoch_stats])
    avg_throughput = np.mean([s['throughput'] for s in epoch_stats])

    print(f"\nAcross {num_epochs} epochs:")
    print(f"  Average epoch time: {avg_epoch_time:.2f}s ± {np.std([s['epoch_time'] for s in epoch_stats]):.2f}s")
    print(f"  Average throughput: {avg_throughput:.1f} samples/sec")
    print(f"  Total training time: {sum([s['epoch_time'] for s in epoch_stats]):.1f}s")

    # Cache statistics
    print("\n" + "-" * 80)
    print("CACHE STATISTICS")
    print("-" * 80)
    stats = cache.get_stats()
    print(f"  Cache usage: {stats['cache_size']}/{stats['max_size']} ({stats['usage_pct']:.1f}%)")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")

    print("=" * 80)

    # Cleanup
    if cache_mode == 'lru':
        cache.stop_prefetching()
    cache.clear()

    return epoch_stats


# ============================================================================
# Main
# ============================================================================

def main():
    local_storage = os.path.join(PROJECT_ROOT, "local_storage")
    data_dir = os.path.join(local_storage, 'data', '2026-01-06_15-41-27')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run training simulation for 10 epochs
    run_training_simulation(
        data_dir=data_dir,
        batch_size=64,
        num_workers=0,  # Data is preloaded to RAM - workers add overhead
        cache_mode='lru',
        lru_cache_size=800,  # 80% of dataset fits in GPU cache
        num_epochs=10,
        shuffle=True,
        verbose=True  # Set to False for less output
    )


if __name__ == "__main__":
    main()