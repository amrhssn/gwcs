import os
from typing import Optional, Callable, List, Dict, Tuple

import torch
from tqdm import tqdm
from torch.utils.data import random_split, Subset, DataLoader, Dataset as TorchDataset
###CHANGES: need Sampler type hint for the updated dataloader function
from torch.utils.data import Sampler
from torch_geometric.data import Dataset

from data.utils_io import load_config
from models.sparsification_utils import (
    multilevel_importance_sampling,
    weighted_random_per_scale,
    random_mask,
    SparsityAnnealer
)
from graph_wavelet_transform.chebyshev import precompute_synthesis_data



class GraphSignalDatasetTurbulent(Dataset):
    """
    Dataset for turbulent radiative layer 2D data.

    Expects processed data in base_data_dir/raw/ with:
    - .pt files containing PyG Data objects with:
        - x: (N, num_fields) original signal
        - coeffs: (N, num_fields, num_filters) GWT coefficients
        - edge_index: (2, E) graph connectivity
        - pos: (N, 2) node positions
        - graph_metadata: dict with L_sym_sparse
        - gwt_metadata: dict with chebyshev coefficients and parameters
    - chebyshev_synthesis_data.pt: shared Chebyshev data for IGWT
    - Config JSON files: graph_config.json, signal_config.json, gwt_config.json
    """

    def __init__(
            self,
            base_data_dir: str,
            field_index: Optional[int] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            preload_to_memory: bool = False,
            max_samples: Optional[int] = None,
            use_normalization: bool = False,
            normalization_stats: Optional[Dict] = None,
    ):
        """
        Args:
            base_data_dir: Path to processed data directory
            field_index: Which field to use (0=density, 1=pressure, 2=velocity_x, 3=velocity_y, None=all)
            transform: Optional transform to apply
            pre_transform: Optional pre-transform
            preload_to_memory: Whether to preload all samples to RAM
            max_samples: Maximum number of samples to load (for testing). None = load all.
            use_normalization: Whether to apply Z-score normalization
            normalization_stats: Dictionary of normalization stats (if None, loads from file)
        """
        self.base_data_dir = base_data_dir
        self.root = base_data_dir
        self.raw_data_dir = base_data_dir
        self.field_index = field_index
        self.use_normalization = use_normalization

        # Load config files
        self.graph_config = load_config(config_dir=base_data_dir, config_name='graph_config.json')
        self.signal_config = load_config(config_dir=base_data_dir, config_name='signal_config.json')
        self.gwt_config = load_config(config_dir=base_data_dir, config_name='gwt_config.json')

        # Load shared Chebyshev synthesis data (same for all samples)
        chebyshev_data_path = os.path.join(base_data_dir, 'chebyshev_synthesis_data.pt')
        self.chebyshev_synthesis_data = torch.load(chebyshev_data_path, map_location='cpu', weights_only=False)

        # Load shared graph data (edge_index is same for all samples)
        shared_data_path = os.path.join(base_data_dir, 'shared_data.pt')
        shared_data = torch.load(shared_data_path, map_location='cpu', weights_only=False)
        self.shared_edge_index = shared_data['edge_index']

        # Load normalization stats if needed
        self.normalization_stats = None
        if self.use_normalization:
            if normalization_stats is not None:
                self.normalization_stats = normalization_stats
            else:
                # Load from file (go up one level to version root)
                from pathlib import Path
                from data.normalization_turbulent import load_normalization_stats
                version_root = Path(base_data_dir).parent
                stats_path = version_root / 'normalization_stats.pt'
                self.normalization_stats = load_normalization_stats(str(stats_path))

        # Get list of data files
        self.file_list = sorted(
            [f for f in os.listdir(base_data_dir)
             if f.endswith('.pt') and f.split('.')[0].isdigit()],
            key=lambda x: int(x.split('.')[0])
        )

        # Apply max_samples limit if specified
        if max_samples is not None:
            self.file_list = self.file_list[:max_samples]

        # Preload all data to RAM if requested
        self.preload_to_memory = preload_to_memory
        self.cached_data = []

        if self.preload_to_memory:
            total_size = 0
            pbar = tqdm(
                range(len(self.file_list)),
                desc=f"Preloading {len(self.file_list)} samples to RAM",
                unit="sample",
                ncols=120,
            )
            for idx in pbar:
                data = self._load_from_disk(idx)
                self.cached_data.append(data)
                for v in data.values():
                    if isinstance(v, torch.Tensor):
                        total_size += v.element_size() * v.numel()
                pbar.set_postfix({'RAM': f'{total_size / 1e9:.2f} GB'})
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
        """Load a single sample from disk and extract relevant fields."""
        file_path = os.path.join(self.base_data_dir, self.file_list[idx])
        pyg_data = torch.load(file_path, map_location='cpu', weights_only=False)

        # Extract fields — always cast to float32 regardless of storage dtype
        # (data may be stored as float16 to halve disk usage)
        x = pyg_data.x.float()      # (N, num_fields)
        coeffs = pyg_data.coeffs.float()  # (N, num_fields, num_filters)

        # Slice to selected field if specified
        if self.field_index is not None:
            x = x[:, self.field_index].contiguous()  # (N,) — own storage
            coeffs = coeffs[:, self.field_index, :].contiguous()  # (N, num_filters) — own storage

        # Apply normalization if enabled
        # Apply normalization if enabled
        if self.use_normalization and self.normalization_stats is not None:
            from data.normalization_turbulent import normalize_field
            
            # ✅ FIX: Get field names dynamically from signal_config instead of hardcoding
            # OLD (WRONG): field_names = ['density', 'pressure', 'velocity_x', 'velocity_y']
            field_names = self.signal_config.field_names  # ✅ Use actual field names from config
        
            if self.field_index is None:
                # Normalize all fields independently
                x_normalized = torch.zeros_like(x)
                for field_idx, field_name in enumerate(field_names):
                    if field_name in self.normalization_stats:  # ✅ Safety check
                        stats = self.normalization_stats[field_name]
                        x_normalized[:, field_idx] = normalize_field(
                            x[:, field_idx],
                            stats['mean'],
                            stats['std']
                        )
                    else:
                        # If field not in stats, keep original (shouldn't happen in normal case)
                        x_normalized[:, field_idx] = x[:, field_idx]
                x = x_normalized
            else:
                # Normalize single field
                field_name = field_names[self.field_index]
                if field_name in self.normalization_stats:  # ✅ Safety check
                    stats = self.normalization_stats[field_name]
                    x = normalize_field(x, stats['mean'], stats['std'])

        # Return as dictionary for compatibility with existing pipeline
        data_dict = {
            'x': x,
            'coeffs': coeffs,
            '_idx': idx,
        }

        return data_dict

    def get(self, idx: int) -> Dict:
        """Load and return a single graph signal sample as dictionary."""
        if self.preload_to_memory:
            return self.cached_data[idx]
        else:
            return self._load_from_disk(idx)

    def get_normalization_stats(self) -> Optional[Dict]:
        """Return normalization statistics if available."""
        return self.normalization_stats


# ============================================================================
# Matrix Cache for Turbulent Data (Chebyshev-based)
# ============================================================================

class ChebyshevMatrixCacheTurbulent:
    """
    Cache for turbulent data using Chebyshev synthesis.

    Since all samples share the same graph structure, we only need to cache
    the Chebyshev synthesis data once (not per sample).
    """

    def __init__(
            self,
            dataset: GraphSignalDatasetTurbulent,
            device: torch.device
    ):
        self.device = device
        self.chebyshev_data = {}

        print("\n" + "=" * 80)
        print("CHEBYSHEV CACHE MODE (Turbulent Data)")
        print("=" * 80)
        print("Caching shared Chebyshev synthesis data on GPU...")

        # Get sparse tensor and coalesce it before moving to device
        L_sym_sparse = dataset.chebyshev_synthesis_data['L_sym_sparse']
        if L_sym_sparse.is_sparse and not L_sym_sparse.is_coalesced():
            L_sym_sparse = L_sym_sparse.coalesce()

        # Move shared Chebyshev data to GPU (only once!)
        self.chebyshev_data = {
            'chebyshev_coeffs': dataset.chebyshev_synthesis_data['chebyshev_coeffs'].to(device),
            'chebyshev_eigenvalue_samples': dataset.chebyshev_synthesis_data['chebyshev_eigenvalue_samples'].to(device),
            'L_sym_sparse': L_sym_sparse.to(device),
            'lambda_min': dataset.chebyshev_synthesis_data['lambda_min'],
            'lambda_max': dataset.chebyshev_synthesis_data['lambda_max'],
            'chebyshev_max_degree': dataset.chebyshev_synthesis_data['chebyshev_max_degree'],
        }

        # Precompute dual Chebyshev coefficients and L_sym_tilde ONCE
        print("Precomputing synthesis data (dual filters + L_sym_tilde)...")
        self.chebyshev_data['_precomputed'] = precompute_synthesis_data(
            chebyshev_coeffs=self.chebyshev_data['chebyshev_coeffs'],
            chebyshev_eigenvalue_samples=self.chebyshev_data['chebyshev_eigenvalue_samples'],
            L_sym_sparse=self.chebyshev_data['L_sym_sparse'],
            lambda_min=self.chebyshev_data['lambda_min'],
            lambda_max=self.chebyshev_data['lambda_max'],
            chebyshev_max_degree=self.chebyshev_data['chebyshev_max_degree'],
        )
        print("✓ Precomputed synthesis data cached on GPU")

        # Store shared edge_index on GPU (same for all samples)
        self.edge_index = dataset.shared_edge_index.to(device)
        self.num_nodes = dataset.graph_config.num_nodes

        # Cache for pre-built batched edge indices
        self._batched_edge_index_cache: Dict[int, torch.Tensor] = {}

        # Estimate memory
        chebyshev_size = self.chebyshev_data['chebyshev_coeffs'].numel() * 4 / 1024 / 1024  # MB
        eigenval_size = self.chebyshev_data['chebyshev_eigenvalue_samples'].numel() * 4 / 1024 / 1024
        # L_sym_sparse is sparse, so memory is proportional to nnz
        lsym_size = self.chebyshev_data['L_sym_sparse']._values().numel() * 4 / 1024 / 1024
        total_mb = chebyshev_size + eigenval_size + lsym_size

        print(f"✓ Cached shared Chebyshev data")
        print(f"  - chebyshev_coeffs: {self.chebyshev_data['chebyshev_coeffs'].shape}")
        print(f"  - L_sym_sparse: {self.chebyshev_data['L_sym_sparse'].shape} (sparse)")
        print(f"GPU memory used: ~{total_mb:.1f} MB")
        print("=" * 80)

    def get(self, indices: List[int]) -> Dict:
        """
        Get Chebyshev data for batch.

        Since all samples share the same graph, we return the same data for all indices.
        """
        return self.chebyshev_data

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        L_sym = self.chebyshev_data['L_sym_sparse']
        if L_sym.is_sparse:
            lsym_nnz = L_sym._nnz() if hasattr(L_sym, '_nnz') else L_sym._values().numel()
        else:
            lsym_nnz = L_sym.numel()

        return {
            'cache_type': 'chebyshev_shared',
            'memory_mb': sum([
                self.chebyshev_data['chebyshev_coeffs'].numel() * 4 / 1024 / 1024,
                self.chebyshev_data['chebyshev_eigenvalue_samples'].numel() * 4 / 1024 / 1024,
                lsym_nnz * 4 / 1024 / 1024,
            ]),
            'hit_rate': 100.0,
        }

    def clear(self):
        """Clear cache."""
        self.chebyshev_data.clear()
        self._batched_edge_index_cache.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def get_batched_edge_index(self, batch_size: int) -> torch.Tensor:
        """
        Build batched edge_index for a batch of graphs.
        Caches result for reuse.
        """
        if batch_size in self._batched_edge_index_cache:
            return self._batched_edge_index_cache[batch_size]

        if batch_size == 1:
            return self.edge_index

        # Build with offsets on CPU first, then transfer once
        edge_index_cpu = self.edge_index.cpu()
        edge_indices = []
        for i in range(batch_size):
            offset = i * self.num_nodes
            edge_indices.append(edge_index_cpu + offset)

        batched = torch.cat(edge_indices, dim=1).to(self.device)

        # Cache it (only cache common sizes to limit memory)
        if len(self._batched_edge_index_cache) < 5:
            self._batched_edge_index_cache[batch_size] = batched

        return batched

# ============================================================================
# Data Loading Utilities
# ============================================================================

def create_data_splits_turbulent(
        dataset: GraphSignalDatasetTurbulent,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_ratio, val_ratio, test_ratio],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def optimized_collate_turbulent(batch: List[Dict]) -> Dict:
    # Extract data
    x_list = [sample['x'] for sample in batch]
    coeffs_list = [sample['coeffs'] for sample in batch]
    indices = [sample['_idx'] for sample in batch]

    num_graphs = len(x_list)  # Store this BEFORE overwriting anything

    # Concatenate x and coeffs
    x = torch.cat(x_list, dim=0)
    coeffs = torch.cat(coeffs_list, dim=0)

    # Build batch vector and ptr
    num_nodes_per_graph = [xi.shape[0] for xi in x_list]
    batch_vec = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes_per_graph)])
    ptr = torch.cumsum(torch.tensor([0] + num_nodes_per_graph), dim=0)

    return {
        'x': x,
        'coeffs': coeffs,
        'batch': batch_vec,
        'ptr': ptr,
        'indices': indices,
        'batch_size': num_graphs,  # Correct: number of graphs in batch
    }


###CHANGES: added train_sampler/val_sampler/test_sampler params so model_training can pass
# DistributedSampler for DDP — when a sampler is given, shuffle must be False on the DataLoader
def create_optimized_dataloaders_turbulent(
        train_dataset: TorchDataset,
        val_dataset: TorchDataset,
        test_dataset: TorchDataset,
        batch_size: int,
        num_workers: int = 0,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        test_sampler: Optional[Sampler] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create optimized dataloaders for turbulent data.

    Args:
        train_dataset: Training subset
        val_dataset: Validation subset
        test_dataset: Test subset
        batch_size: Batch size
        num_workers: Number of DataLoader workers (0 recommended if preloaded)
        train_sampler: Optional sampler for training (used for DDP DistributedSampler)
        val_sampler: Optional sampler for validation (used for DDP DistributedSampler)
        test_sampler: Optional sampler for test (used for DDP DistributedSampler)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        ###CHANGES: shuffle must be False when using a sampler (sampler handles ordering)
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=optimized_collate_turbulent,
        pin_memory=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=optimized_collate_turbulent,
        pin_memory=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=optimized_collate_turbulent,
        pin_memory=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader


def create_matrix_cache_turbulent(
        dataset: GraphSignalDatasetTurbulent,
        device: torch.device,
) -> ChebyshevMatrixCacheTurbulent:
    """Factory function to create Chebyshev matrix cache for turbulent data."""
    return ChebyshevMatrixCacheTurbulent(dataset=dataset, device=device)