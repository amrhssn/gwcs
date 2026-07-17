"""
preprocess.py — Unified Data Preprocessing Pipeline
=====================================================
Supports: turbulent_radiative_layer_2D | viscoelastic_instability | pbfm | dynamic_stall

Key differences per dataset:
  turbulent_radiative_layer_2D : WellDataset, no spatial crop, 4 fields
  viscoelastic_instability     : WellDataset, crop rows 167-344 (178 rows × 512 cols),
                                  fields from WellDataset metadata
  pbfm                         : HDF5 (kolmogorov_train/test.h5),
                                  2 fields: velocity_x, velocity_y, 128×128 grid
  dynamic_stall                : HDF5 (dynamic_stall_train/test.h5),
                                  auto-detected fields & grid from file structure

GPU acceleration:
  When a CUDA device is available, signals are stacked into batches and the
  Chebyshev SpMM is computed on GPU (one SpMM call per field per batch instead
  of one SpMV per sample). Use --no-gpu to force CPU-only processing.

Usage:
    python preprocess.py --dataset turbulent_radiative_layer_2D
    python preprocess.py --dataset viscoelastic_instability
    python preprocess.py --dataset pbfm
    python preprocess.py --dataset dynamic_stall
    python preprocess.py --dataset dynamic_stall --no-gpu   # force CPU
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Literal, Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import grid, remove_self_loops, is_undirected, to_undirected
from tqdm import tqdm

from paths import ROOT_PROJECT_PATH
from data.utils_graph import compute_sparse_L_sym
from graph_wavelet_transform.chebyshev import apply_chebyshev_filters_with_L_tilde, chebyshev_gwt_analysis


# =============================================================================
# DATASET CONFIGURATIONS
# All paths relative to ROOT_PROJECT_PATH unless absolute.
# =============================================================================

# ---------------------------------------------------------------------------
# Raw-data paths are ABSOLUTE, resolved from environment variables (see
# .env.example at the project root). This lets each dataset's raw data live
# anywhere on disk (e.g. a shared/read-only collaborator location) while
# processed output always lands under this project's own local_storage/.
#
# EDIT ME: set RAW_DATA_ROOT (and optionally COLLAB_RAW_DATA_ROOT, if your
# turbulent raw data lives in a separate shared/read-only location) in your
# own .env file, or export them before running.
#
# Processed output paths are RELATIVE to ROOT_PROJECT_PATH, so all processed
# data lands in <ROOT_PROJECT_PATH>/local_storage/datasets/<name>/processed/
#
# Split-name aliases: map our canonical split name → actual directory / WellDataset name.
# turbulent uses 'valid' (confirmed); other datasets may differ.
# ---------------------------------------------------------------------------

RAW_DATA_ROOT = os.environ.get('RAW_DATA_ROOT', '/path/to/your/graph_wavelet_compressed_sensing')
# Optional: a separate (e.g. read-only, collaborator-owned) raw-data location for turbulent data.
# Falls back to RAW_DATA_ROOT if not set.
COLLAB_RAW_DATA_ROOT = os.environ.get('COLLAB_RAW_DATA_ROOT', RAW_DATA_ROOT)

DATASET_CONFIGS: Dict[str, Dict] = {
    'turbulent_radiative_layer_2D': {
        # Raw data may live in a separate (e.g. read-only, collaborator-owned) location;
        # has data/{train,test,valid}/
        'raw_subpath': (
            f'{COLLAB_RAW_DATA_ROOT}/'
            'local_storage/datasets/turbulent_radiative_layer_2D'
        ),
        'processed_subpath': 'local_storage/datasets/turbulent_radiative_layer_2D/processed',
        'loader':            'well',
        'crop_rows':         None,
        # turbulent directories use 'valid' (not 'val')
        'split_alias': {'train': 'train', 'valid': 'valid', 'test': 'test'},
    },
    'viscoelastic_instability': {
        'raw_subpath': (
            f'{RAW_DATA_ROOT}/'
            'local_storage/datasets/viscoelastic_instability'
        ),
        'processed_subpath': 'local_storage/datasets/viscoelastic_instability/processed',
        'loader':            'well',
        # Crop to rows 167–344 (178 rows × 512 cols).
        # Applied to BOTH graph construction AND every signal sample.
        'crop_rows':         (167, 345),
        'split_alias': {'train': 'train', 'valid': 'valid', 'test': 'test'},
    },
    'kolmogorov': {
        # Previously called 'pbfm' — renamed to reflect actual Kolmogorov flow physics
        'raw_subpath': (
            f'{RAW_DATA_ROOT}/'
            'local_storage/datasets/pbfm'
        ),
        'processed_subpath': 'local_storage/datasets/kolmogorov/processed',
        'loader':            'hdf5_kolmogorov',   # reads kolmogorov_train/test.h5
        'crop_rows':         None,
        'split_alias': {},
    },
    'dynamic_stall': {
        'raw_subpath': (
            f'{RAW_DATA_ROOT}/'
            'local_storage/datasets/dynamic'
        ),
        'processed_subpath': 'local_storage/datasets/dynamic_stall/processed',
        'loader':            'hdf5_dynamic',      # reads dynamic_stall_train/test.h5
        'crop_rows':         None,
        'split_alias': {},
        # MUST use float32: raw field values reach ~2.44e+07, which overflows
        # float16 (max ≈ 65504) and turns every large value into inf.
        'force_float32':     True,
    },
}


# =============================================================================
# GWT CONFIGURATION  (unchanged from existing scripts)
# =============================================================================

GWT_CONFIG = {
    'lambda_max':          2.0,
    'lambda_min':          0.0,
    'K':                   500,
    'num_scales':          4,
    'chebyshev_max_degree': 100,
}
GWT_CONFIG['bandpass_lambda_min'] = GWT_CONFIG['lambda_max'] / GWT_CONFIG['K']
GWT_CONFIG['num_filters']         = GWT_CONFIG['num_scales'] + 1
GWT_CONFIG['gwt_scales'] = np.logspace(
    np.log10(1 / GWT_CONFIG['lambda_max']),
    np.log10(1 / GWT_CONFIG['bandpass_lambda_min']),
    GWT_CONFIG['num_scales']
).tolist()


# =============================================================================
# SHARED STATE
# =============================================================================

class SharedState:
    """Holds all shared (graph-level) data for worker threads. Read-only after init."""

    def __init__(self):
        self.L_sym_sparse = None
        self.L_sym_tilde  = None
        self.edge_index   = None
        self.pos          = None
        self.visualization_pos    = None
        self.chebyshev_coeffs     = None
        self.chebyshev_eigenvalue_samples = None
        self.cache_key            = None

        self.num_nodes:   int = 0
        self.num_fields:  int = 0
        self.num_filters: int = 0
        self.num_rows:    int = 0
        self.num_columns: int = 0
        self.field_names:    List[str] = []
        self.field_to_index: Dict[str, int] = {}
        self.index_to_field: Dict[int, str] = {}

        self.processed_dataset_dir: str = ""
        self.save_lock = Lock()
        # Storage dtype: float16 halves disk usage; dataset loader always casts back to float32
        self.save_dtype: torch.dtype = torch.float16


_shared: Optional[SharedState] = None


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_dataset_paths(dataset_name: str) -> Tuple[Path, Path]:
    """Return (raw_base_path, processed_base_path) for the given dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Choose from: {list(DATASET_CONFIGS.keys())}"
        )
    cfg = DATASET_CONFIGS[dataset_name]

    root = Path(ROOT_PROJECT_PATH)
    raw_path  = Path(cfg['raw_subpath'])
    proc_path = Path(cfg['processed_subpath'])

    if not raw_path.is_absolute():
        raw_path = root / raw_path
    if not proc_path.is_absolute():
        proc_path = root / proc_path

    return raw_path, proc_path


# =============================================================================
# PBFM HDF5 LOADER
# =============================================================================

def load_pbfm_samples(
        split: str,
        base_path: str,
        num_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], int, int]:
    """
    Load PBFM Kolmogorov HDF5 samples.

    HDF5 shape: (num_sims, T, C=2, H=128, W=128)
    Returns list of (H*W, 2) numpy arrays plus H, W.

    Train/valid : kolmogorov_train.h5 — 80/20 simulation split.
    Test        : kolmogorov_test.h5  — all simulations.
    """
    import h5py

    if split in ['train', 'valid']:
        h5_file = os.path.join(base_path, 'raw', 'kolmogorov_train.h5')
    else:
        h5_file = os.path.join(base_path, 'raw', 'kolmogorov_test.h5')

    with h5py.File(h5_file, 'r') as f:
        fields = f['fields'][:]     # (num_sims, T, 2, 128, 128)

    num_sims = fields.shape[0]
    T  = fields.shape[1]
    C  = fields.shape[2]
    H  = fields.shape[3]
    W  = fields.shape[4]

    if split == 'train':
        sims = fields[:int(0.8 * num_sims)]
    elif split == 'valid':
        sims = fields[int(0.8 * num_sims):]
    else:
        sims = fields

    all_frames = sims.reshape(-1, C, H, W)
    if num_samples is not None:
        all_frames = all_frames[:num_samples]

    samples = []
    for frame in all_frames:
        # (C, H, W) → (H, W, C) → (H*W, C)
        x = frame.transpose(1, 2, 0).reshape(H * W, C).astype(np.float32)
        samples.append(x)

    return samples, H, W


# =============================================================================
# DYNAMIC STALL HDF5 LOADER  (auto-detects shape and field names)
# =============================================================================

def load_dynamic_stall_samples(
        split: str,
        base_path: str,
        num_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], int, int, List[str]]:
    """
    Load dynamic_stall HDF5 samples with automatic structure detection.

    Tries common HDF5 layouts:
      - (num_sims, T, C, H, W)  — Kolmogorov-style
      - (N, C, H, W)            — flat frame list
      - (N, H, W, C)            — channels-last variant

    Field names are read from file attributes ('field_names', 'variables') if
    present; otherwise auto-generated as 'field_0', 'field_1', …

    Returns
    -------
    samples       : list of (H*W, C) float32 numpy arrays
    H, W          : spatial grid dimensions
    field_names   : list of C field name strings
    """
    import h5py

    if split in ['train', 'valid']:
        h5_file = os.path.join(base_path, 'raw', 'dynamic_stall_train.h5')
    else:
        h5_file = os.path.join(base_path, 'raw', 'dynamic_stall_test.h5')

    print(f"  Reading: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        # ---- print full HDF5 structure for diagnostics ----
        print("  HDF5 structure:")
        def _print_item(name, obj):
            if hasattr(obj, 'shape'):
                attrs = dict(obj.attrs)
                print(f"    {name}: shape={obj.shape}, dtype={obj.dtype}"
                      + (f", attrs={attrs}" if attrs else ""))
            else:
                attrs = dict(obj.attrs)
                print(f"    {name}/  [group]"
                      + (f"  attrs={attrs}" if attrs else ""))
        f.visititems(_print_item)
        top_attrs = dict(f.attrs)
        if top_attrs:
            print(f"  Root attrs: {top_attrs}")

        # ---- find the main data array ----
        CANDIDATE_KEYS = ['fields', 'data', 'u', 'velocity', 'vel', 'snapshots']
        data_key = None
        for k in CANDIDATE_KEYS:
            if k in f:
                data_key = k
                break
        if data_key is None:
            # fall back: first dataset with ndim >= 4
            for k in f.keys():
                if hasattr(f[k], 'shape') and f[k].ndim >= 4:
                    data_key = k
                    break
        if data_key is None:
            raise ValueError(
                f"Cannot find a 4-D+ dataset in {h5_file}. "
                f"Top-level keys: {list(f.keys())}. "
                "Run inspect_hdf5.py to see the full file layout and set "
                "--hdf5-key manually."
            )

        raw = f[data_key][:]   # load everything into RAM for flexibility

        # ---- try to read field names ----
        field_names: Optional[List[str]] = None
        for attr_name in ('field_names', 'variables', 'fields', 'channel_names'):
            if attr_name in f.attrs:
                raw_names = f.attrs[attr_name]
                field_names = [n.decode() if isinstance(n, bytes) else str(n)
                               for n in raw_names]
                break
        if field_names is None and data_key + '_names' in f:
            raw_names = f[data_key + '_names'][:]
            field_names = [n.decode() if isinstance(n, bytes) else str(n)
                           for n in raw_names]

    # ---- decode array layout ----
    shape = raw.shape
    print(f"  Data key: '{data_key}', raw shape: {shape}")

    if raw.ndim == 5:
        # (num_sims, T, C, H, W)
        num_sims, T, C, H, W = shape
        if split == 'train':
            sims = raw[:int(0.8 * num_sims)]
        elif split == 'valid':
            sims = raw[int(0.8 * num_sims):]
        else:
            sims = raw
        all_frames = sims.reshape(-1, C, H, W)        # (N', C, H, W)
    elif raw.ndim == 4:
        N, d1, d2, d3 = shape
        # Heuristic: if last dim is much smaller than others → channels-last
        if d3 <= min(d1, d2) // 4:
            C, H, W = d3, d1, d2
            all_frames = raw.transpose(0, 3, 1, 2)    # (N, C, H, W)
        else:
            C, H, W = d1, d2, d3
            all_frames = raw                            # (N, C, H, W)
        if split in ['train', 'valid']:
            n_total = len(all_frames)
            cut = int(0.8 * n_total)
            all_frames = all_frames[:cut] if split == 'train' else all_frames[cut:]
    else:
        raise ValueError(
            f"Unsupported data shape {shape} in {h5_file}. "
            "Expected ndim=4 or ndim=5."
        )

    if field_names is None:
        field_names = [f'field_{i}' for i in range(C)]

    print(f"  Detected: {len(all_frames)} frames, grid={H}×{W}, "
          f"C={C}, fields={field_names}")

    if num_samples is not None:
        all_frames = all_frames[:num_samples]

    samples: List[np.ndarray] = []
    for frame in all_frames:
        # (C, H, W) → (H, W, C) → (H*W, C)  float32
        x = frame.transpose(1, 2, 0).reshape(H * W, C).astype(np.float32)
        samples.append(x)

    return samples, H, W, field_names


# =============================================================================
# GPU-ACCELERATED BATCH PROCESSING
# =============================================================================

def _run_gpu_batch_process(
        data: List[Tuple[int, np.ndarray, int, int]],
        gpu_batch_size: int = 64,
) -> int:
    """
    GPU-accelerated batch Chebyshev preprocessing.

    Stacks `gpu_batch_size` signals per field into a (N, B) matrix, then
    calls apply_chebyshev_filters_with_L_tilde once per field per batch.
    This replaces batch_size sequential SpMV calls with a single SpMM call,
    giving significant GPU throughput gains.

    Requires _shared to be initialised and a CUDA device to be available.
    """
    global _shared
    import torch

    device = torch.device('cuda')

    # Move shared sparse tensors to GPU (done once per call)
    try:
        L_tilde_gpu    = _shared.L_sym_tilde.to(device).coalesce()
        coeffs_gpu     = _shared.chebyshev_coeffs.to(device)
    except Exception as e:
        print(f"  ⚠ GPU tensor transfer failed ({e}); falling back to CPU threadpool")
        return _run_threadpool(data, min(8, os.cpu_count() or 4))

    num_nodes   = _shared.num_nodes
    num_fields  = _shared.num_fields
    num_filters = _shared.num_filters
    completed   = 0

    total_batches = (len(data) + gpu_batch_size - 1) // gpu_batch_size

    with tqdm(total=len(data), desc="    GPU batches", unit="sample", leave=False) as pbar:
        for batch_start in range(0, len(data), gpu_batch_size):
            batch = data[batch_start: batch_start + gpu_batch_size]
            B     = len(batch)

            # Stack signals: (N, B, num_fields) — keep CPU until field loop
            x_list = []
            meta   = []
            for (sample_index, x_np, traj_idx, time_idx) in batch:
                x_list.append(torch.from_numpy(x_np).float())   # (N, num_fields)
                meta.append((sample_index, traj_idx, time_idx))

            X_cpu = torch.stack(x_list, dim=0)  # (B, N, num_fields)

            # Allocate coefficients buffer on CPU for saving
            all_coeffs = torch.zeros(
                B, num_nodes, num_fields, num_filters, dtype=torch.float32)

            for field_idx in range(num_fields):
                # (N, B) on GPU
                X_field = X_cpu[:, :, field_idx].T.contiguous().to(device)  # (N, B)

                with torch.no_grad():
                    Y = apply_chebyshev_filters_with_L_tilde(
                        x                   = X_field,
                        chebyshev_coeffs    = coeffs_gpu,
                        chebyshev_max_degree= GWT_CONFIG['chebyshev_max_degree'],
                        L_sym_tilde         = L_tilde_gpu,
                    )  # (F, N, B)  or (F, N) if B==1

                if Y.ndim == 2:          # B==1 → (F, N)
                    Y = Y.unsqueeze(-1)  # → (F, N, 1)

                # Y: (F, N, B) → transpose to (B, N, F)
                Y_cpu = Y.permute(2, 1, 0).cpu()  # (B, N, F)
                all_coeffs[:, :, field_idx, :] = Y_cpu

            del X_field, Y, Y_cpu
            torch.cuda.empty_cache()

            # Save individual sample files
            save_dtype = _shared.save_dtype
            for b_idx, (sample_index, traj_idx, time_idx) in enumerate(meta):
                x_b      = x_list[b_idx].to(save_dtype)         # (N, num_fields)
                coeffs_b = all_coeffs[b_idx].to(save_dtype)     # (N, num_fields, F)

                signal_metadata = {
                    "sample_index":     sample_index,
                    "trajectory_index": traj_idx,
                    "time_index":       time_idx,
                }
                from torch_geometric.data import Data as PygData
                data_obj  = PygData(x=x_b, coeffs=coeffs_b,
                                    signal_metadata=signal_metadata)
                save_path = os.path.join(
                    _shared.processed_dataset_dir, f'{sample_index}.pt')
                torch.save(data_obj, save_path)
                completed += 1

            pbar.update(B)

    return completed


# =============================================================================
# WORKER FUNCTION (runs in thread)
# =============================================================================

def process_single_sample(args: Tuple[int, np.ndarray, int, int]) -> int:
    """Compute GWT coefficients for one sample and save to disk."""
    global _shared

    sample_index, x_np, trajectory_index, time_index = args
    x = torch.from_numpy(x_np).float()

    coeffs = torch.zeros(
        _shared.num_nodes,
        _shared.num_fields,
        _shared.num_filters,
        dtype=torch.float32
    )

    for field_idx in range(_shared.num_fields):
        Y = apply_chebyshev_filters_with_L_tilde(
            x=x[:, field_idx],
            chebyshev_coeffs=_shared.chebyshev_coeffs,
            chebyshev_max_degree=GWT_CONFIG['chebyshev_max_degree'],
            L_sym_tilde=_shared.L_sym_tilde,
        )
        coeffs[:, field_idx, :] = Y.T

    signal_metadata = {
        "sample_index":    sample_index,
        "trajectory_index": trajectory_index,
        "time_index":      time_index,
    }

    # Cast to storage dtype (float16 by default → half disk usage)
    save_dtype = _shared.save_dtype
    data = Data(
        x=x.to(save_dtype),
        coeffs=coeffs.to(save_dtype),
        signal_metadata=signal_metadata,
    )
    save_path = os.path.join(_shared.processed_dataset_dir, f'{sample_index}.pt')
    torch.save(data, save_path)

    return sample_index


# =============================================================================
# SHARED DATA SAVE
# =============================================================================

def save_shared_data(output_dir: str) -> None:
    """Save the graph-level data shared by all samples."""
    global _shared

    shared_data = {
        'pos':                        _shared.pos,
        'edge_index':                 _shared.edge_index,
        'L_sym_sparse':               _shared.L_sym_sparse,
        'visualization_pos':          _shared.visualization_pos,
        'num_nodes':                  _shared.num_nodes,
        'num_rows':                   _shared.num_rows,
        'num_columns':                _shared.num_columns,
        'L_sym_tilde':                _shared.L_sym_tilde,
        'chebyshev_coeffs':           _shared.chebyshev_coeffs,
        'chebyshev_eigenvalue_samples': _shared.chebyshev_eigenvalue_samples,
        'cache_key':                  _shared.cache_key,
        'field_names':                _shared.field_names,
        'field_to_index':             _shared.field_to_index,
        'index_to_field':             _shared.index_to_field,
        'num_fields':                 _shared.num_fields,
        'num_filters':                _shared.num_filters,
        'gwt_config':                 GWT_CONFIG,
    }

    torch.save(shared_data, os.path.join(output_dir, 'shared_data.pt'))
    print(f"  ✓ Saved shared_data.pt")


# =============================================================================
# CONFIG FILES
# =============================================================================

def create_config_files(processed_data_dir: str) -> Dict[str, Any]:
    """Write chebyshev_synthesis_data.pt, graph_config.json,
    signal_config.json, gwt_config.json from shared_data.pt."""

    shared_data_path = os.path.join(processed_data_dir, 'shared_data.pt')
    if not os.path.exists(shared_data_path):
        raise FileNotFoundError(f"shared_data.pt not found in {processed_data_dir}")

    shared_data = torch.load(shared_data_path, map_location='cpu', weights_only=False)

    pt_files = sorted(
        [f for f in os.listdir(processed_data_dir)
         if f.endswith('.pt') and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0])
    )
    if not pt_files:
        raise FileNotFoundError(f"No sample .pt files found in {processed_data_dir}")

    print(f"  Found {len(pt_files)} processed samples")

    num_nodes   = shared_data['num_nodes']
    num_fields  = shared_data['num_fields']
    num_filters = shared_data['num_filters']
    num_edges   = shared_data['edge_index'].shape[1]
    gwt_config  = shared_data['gwt_config']

    # chebyshev_synthesis_data.pt (used at inference/reconstruction time)
    chebyshev_synthesis_data = {
        'chebyshev_coeffs':           shared_data['chebyshev_coeffs'],
        'chebyshev_eigenvalue_samples': shared_data['chebyshev_eigenvalue_samples'],
        'L_sym_sparse':               shared_data['L_sym_sparse'],
        'L_sym_tilde':                shared_data['L_sym_tilde'],
        'lambda_min':                 gwt_config['lambda_min'],
        'lambda_max':                 gwt_config['lambda_max'],
        'chebyshev_max_degree':       gwt_config['chebyshev_max_degree'],
    }
    torch.save(chebyshev_synthesis_data,
               os.path.join(processed_data_dir, 'chebyshev_synthesis_data.pt'))
    print(f"  ✓ Saved chebyshev_synthesis_data.pt")

    # graph_config.json
    graph_config = {
        'num_nodes':  num_nodes,
        'num_edges':  num_edges,
        'graph_type': 'grid',
        'grid_shape': [shared_data['num_rows'], shared_data['num_columns']],
    }
    with open(os.path.join(processed_data_dir, 'graph_config.json'), 'w') as f:
        json.dump(graph_config, f, indent=2)
    print(f"  ✓ Saved graph_config.json")

    # signal_config.json
    signal_config = {
        'num_fields':    num_fields,
        'field_names':   shared_data['field_names'],
        'field_to_index': shared_data['field_to_index'],
        'index_to_field': {str(k): v for k, v in shared_data['index_to_field'].items()},
        'signal_type':   'graph_signal',
    }
    with open(os.path.join(processed_data_dir, 'signal_config.json'), 'w') as f:
        json.dump(signal_config, f, indent=2)
    print(f"  ✓ Saved signal_config.json")

    # gwt_config.json
    gwt_config_json = {
        'num_filters':         num_filters,
        'num_scales':          gwt_config['num_scales'],
        'lambda_min':          float(gwt_config['lambda_min']),
        'lambda_max':          float(gwt_config['lambda_max']),
        'chebyshev_max_degree': int(gwt_config['chebyshev_max_degree']),
        'K':                   gwt_config['K'],
        'bandpass_lambda_min': float(gwt_config['bandpass_lambda_min']),
        'gwt_scales':          gwt_config['gwt_scales'],
    }
    with open(os.path.join(processed_data_dir, 'gwt_config.json'), 'w') as f:
        json.dump(gwt_config_json, f, indent=2)
    print(f"  ✓ Saved gwt_config.json")

    return {
        'num_samples': len(pt_files),
        'num_nodes':   num_nodes,
        'num_fields':  num_fields,
        'num_filters': num_filters,
        'num_edges':   num_edges,
    }


# =============================================================================
# WellDataset HELPERS
# =============================================================================

def _get_trajectory_and_timestep_info(sample_index: int, metadata) -> Tuple[int, int]:
    """Map flat sample index to (trajectory_index, time_index)."""
    cumulative = 0
    for file_idx in range(metadata.n_files):
        n_steps = metadata.n_steps_per_trajectory[file_idx]
        n_traj  = metadata.n_trajectories_per_file[file_idx]
        for _ in range(n_traj):
            if sample_index < cumulative + n_steps:
                return cumulative // max(n_steps, 1), sample_index - cumulative
            cumulative += n_steps
    raise IndexError(f"Sample index {sample_index} out of range")


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def preprocess_split(
        dataset_name: str,
        raw_dataset_dir: str,
        processed_dataset_dir: str,
        split: Literal["train", "valid", "test"],
        num_workers: Optional[int] = None,
        sample_indices: Optional[List[int]] = None,
        chunk_size: int = 500,
        use_gpu: bool = True,
        gpu_batch_size: int = 64,
        half_precision: bool = True,
) -> Dict[str, float]:
    """
    Preprocess one split of one dataset.

    Handles WellDataset (turbulent, viscoelastic) and HDF5 (pbfm, dynamic_stall).
    Viscoelastic crop is applied to both the graph AND every signal sample.
    HDF5 data is loaded once and processed via GPU batches (if available) or ThreadPoolExecutor.
    WellDataset data is loaded in chunks of `chunk_size` to avoid OOM.

    half_precision: If True (default), saves x and coeffs as float16 to halve disk usage.
      The dataset loader always casts back to float32 before returning to training code.

    GPU path (use_gpu=True and CUDA available):
      Signals are stacked into batches of gpu_batch_size and processed with a
      single SpMM per field per batch instead of one SpMV per sample.
    """
    global _shared

    os.makedirs(processed_dataset_dir, exist_ok=True)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 8, 48)  # use all available CPUs

    cfg        = DATASET_CONFIGS[dataset_name]
    loader     = cfg['loader']            # 'well', 'hdf5_kolmogorov', 'hdf5_dynamic'
    crop_rows  = cfg['crop_rows']         # None or (row_start, row_end_exclusive)

    gpu_available = use_gpu and torch.cuda.is_available()

    # Per-dataset override: some datasets have extreme raw values that overflow float16.
    if cfg.get('force_float32', False) and half_precision:
        print(f"  ⚠  {dataset_name} has extreme values — forcing float32 storage "
              f"(float16 would overflow and produce inf).")
        half_precision = False

    save_dtype_str = "float16 (half)" if half_precision else "float32 (full)"

    print("\n" + "=" * 80)
    print(f"PREPROCESSING  dataset={dataset_name}  split={split}")
    print("=" * 80)
    print(f"  Loader      : {loader}")
    print(f"  Crop rows   : {crop_rows}")
    print(f"  Workers     : {num_workers}")
    print(f"  GPU accel   : {gpu_available} (batch_size={gpu_batch_size})")
    print(f"  Storage     : {save_dtype_str}")
    print(f"  Output      : {processed_dataset_dir}")
    print("=" * 80)

    timings: Dict[str, float] = {}

    # =========================================================================
    # PHASE 1: Load dataset metadata + determine field names
    # =========================================================================
    print("\n[PHASE 1] Load dataset / determine fields")
    t0 = time.perf_counter()

    if loader == 'hdf5_kolmogorov':
        # PBFM: load all samples from HDF5 now
        num_limit = len(sample_indices) if sample_indices is not None else None
        samples_np, num_rows, num_columns = load_pbfm_samples(
            split=split,
            base_path=raw_dataset_dir,
            num_samples=num_limit,
        )
        field_names    = ['velocity_x', 'velocity_y']
        num_fields     = 2
        index_to_field = {0: 'velocity_x', 1: 'velocity_y'}
        field_to_index = {'velocity_x': 0, 'velocity_y': 1}
        num_nodes      = num_rows * num_columns
        # Preloaded: (idx, x_np, traj=0, time=i)
        preloaded_data = [(i, samples_np[i], 0, i) for i in range(len(samples_np))]
        indices_to_process = list(range(len(samples_np)))

    elif loader == 'hdf5_dynamic':
        # Dynamic stall: auto-detect fields and grid from HDF5
        num_limit = len(sample_indices) if sample_indices is not None else None
        samples_np, num_rows, num_columns, field_names = load_dynamic_stall_samples(
            split=split,
            base_path=raw_dataset_dir,
            num_samples=num_limit,
        )
        num_fields     = len(field_names)
        index_to_field = {i: n for i, n in enumerate(field_names)}
        field_to_index = {n: i for i, n in enumerate(field_names)}
        num_nodes      = num_rows * num_columns
        # Preloaded: (idx, x_np, traj=0, time=i)
        preloaded_data = [(i, samples_np[i], 0, i) for i in range(len(samples_np))]
        indices_to_process = list(range(len(samples_np)))

    else:  # 'well'
        try:
            from the_well.data import WellDataset
        except ImportError:
            raise ImportError("the_well package is required. Install with: pip install the-well")

        well_base_path = os.path.dirname(raw_dataset_dir)
        dataset_obj = WellDataset(
            well_base_path=well_base_path,
            well_dataset_name=dataset_name,
            well_split_name=split,
            n_steps_input=1,
            n_steps_output=0,
            use_normalization=False,
        )
        field_names    = [n for grp in dataset_obj.metadata.field_names.values() for n in grp]
        num_fields     = len(field_names)
        index_to_field = {idx: n for idx, n in enumerate(field_names)}
        field_to_index = {n: idx for idx, n in enumerate(field_names)}
        total_samples  = len(dataset_obj)

        if sample_indices is not None:
            indices_to_process = sample_indices
        else:
            indices_to_process = list(range(total_samples))

        preloaded_data = None  # Will be loaded in chunks below

    timings['load_dataset'] = time.perf_counter() - t0
    print(f"  ✓ Dataset info loaded in {timings['load_dataset']:.2f}s")
    print(f"  Fields: {field_names}")
    if loader in ('hdf5_kolmogorov', 'hdf5_dynamic'):
        print(f"  Samples: {len(preloaded_data)}")
    else:
        print(f"  Samples to process: {len(indices_to_process)}")

    # =========================================================================
    # PHASE 2: Construct graph (with crop for viscoelastic)
    # =========================================================================
    print("\n[PHASE 2] Construct graph")
    t0 = time.perf_counter()

    if loader == 'well':
        space_grid = dataset_obj[0]['space_grid']  # (H, W, 2)

        if crop_rows is not None:
            row_start, row_end = crop_rows
            print(f"  ⚙  Cropping viscoelastic grid: "
                  f"{space_grid.shape[0]}×{space_grid.shape[1]} → rows {row_start}-{row_end-1}")
            space_grid = space_grid[row_start:row_end, :, :]

        num_rows, num_columns = space_grid.shape[:2]
        num_nodes = num_rows * num_columns
        pos_raw   = space_grid.reshape(-1, 2)
        pos = pos_raw if isinstance(pos_raw, torch.Tensor) \
              else torch.tensor(pos_raw, dtype=torch.float32)
    else:
        # HDF5 datasets (PBFM / dynamic_stall): normalised grid positions
        row_idx = torch.arange(num_rows).repeat_interleave(num_columns).float() / max(num_rows - 1, 1)
        col_idx = torch.arange(num_columns).repeat(num_rows).float() / max(num_columns - 1, 1)
        pos = torch.stack([row_idx, col_idx], dim=1)

    edge_index, visualization_pos = grid(height=num_rows, width=num_columns)
    edge_index, _ = remove_self_loops(edge_index)
    if not is_undirected(edge_index=edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index=edge_index, num_nodes=num_nodes)

    L_sym_sparse = compute_sparse_L_sym(
        edge_index=edge_index, edge_weight=None, num_nodes=num_nodes)

    timings['construct_graph'] = time.perf_counter() - t0
    print(f"  ✓ Graph: {num_rows}×{num_columns} = {num_nodes} nodes "
          f"({timings['construct_graph']:.2f}s)")

    # =========================================================================
    # PHASE 3: Pre-compute Chebyshev polynomials
    # =========================================================================
    print("\n[PHASE 3] Pre-compute Chebyshev polynomials")
    t0 = time.perf_counter()

    if loader in ('hdf5_kolmogorov', 'hdf5_dynamic'):
        x_dummy = torch.zeros(num_nodes, dtype=torch.float32)
    else:
        # For WellDataset, also apply the crop to the dummy signal
        x_dummy_raw = dataset_obj[0]['input_fields'].squeeze()
        if crop_rows is not None:
            row_start, row_end = crop_rows
            x_dummy_raw = x_dummy_raw[row_start:row_end, :, :]
        x_dummy = torch.tensor(
            x_dummy_raw.reshape(num_nodes, num_fields)[:, 0], dtype=torch.float32)

    chebyshev_cache = {}
    _, cache_key, chebyshev_cache = chebyshev_gwt_analysis(
        x=x_dummy,
        L_sym_sparse=L_sym_sparse,
        edge_index=edge_index,
        num_nodes=num_nodes,
        gwt_scales=GWT_CONFIG['gwt_scales'],
        lambda_min=GWT_CONFIG['lambda_min'],
        lambda_max=GWT_CONFIG['lambda_max'],
        chebyshev_max_degree=GWT_CONFIG['chebyshev_max_degree'],
        cache=chebyshev_cache,
    )
    cached_data = chebyshev_cache[cache_key]

    timings['precompute_chebyshev'] = time.perf_counter() - t0
    print(f"  ✓ Chebyshev computed in {timings['precompute_chebyshev']:.2f}s  "
          f"shape={cached_data['chebyshev_coeffs'].shape}")

    # =========================================================================
    # PHASE 4: Initialize shared state and save shared_data.pt
    # =========================================================================
    print("\n[PHASE 4] Initialize shared state")
    t0 = time.perf_counter()

    _shared = SharedState()
    _shared.L_sym_sparse               = L_sym_sparse
    _shared.L_sym_tilde                = cached_data['L_sym_tilde']
    _shared.edge_index                 = edge_index
    _shared.pos                        = pos
    _shared.visualization_pos         = visualization_pos
    _shared.chebyshev_coeffs          = cached_data['chebyshev_coeffs']
    _shared.chebyshev_eigenvalue_samples = cached_data['chebyshev_eigenvalue_samples']
    _shared.cache_key                  = cache_key
    _shared.num_nodes                  = num_nodes
    _shared.num_fields                 = num_fields
    _shared.num_filters                = GWT_CONFIG['num_filters']
    _shared.num_rows                   = num_rows
    _shared.num_columns                = num_columns
    _shared.field_names                = field_names
    _shared.field_to_index             = field_to_index
    _shared.index_to_field             = index_to_field
    _shared.processed_dataset_dir     = processed_dataset_dir
    _shared.save_dtype                 = torch.float16 if half_precision else torch.float32

    save_shared_data(processed_dataset_dir)

    timings['init_shared'] = time.perf_counter() - t0
    print(f"  ✓ Shared state ready in {timings['init_shared']:.2f}s")

    # =========================================================================
    # PHASE 5 / 6: Process samples (chunked for WellDataset, direct for PBFM)
    # =========================================================================
    timings['preload_data']        = 0.0
    timings['parallel_processing'] = 0.0

    if loader in ('hdf5_kolmogorov', 'hdf5_dynamic'):
        # HDF5 datasets: already fully preloaded → single pass
        n_total = len(preloaded_data)
        print(f"\n[PHASE 5] Processing {n_total} preloaded samples "
              f"({'GPU batch' if gpu_available else 'CPU threadpool'})")
        t0 = time.perf_counter()

        if gpu_available:
            _run_gpu_batch_process(preloaded_data, gpu_batch_size=gpu_batch_size)
        else:
            _run_threadpool(preloaded_data, num_workers)

        timings['parallel_processing'] = time.perf_counter() - t0
        print(f"  ✓ Done in {timings['parallel_processing']:.2f}s  "
              f"({n_total / max(timings['parallel_processing'], 1e-6):.1f} samples/s)")

    else:
        # WellDataset: chunked loading to avoid OOM
        total_chunks = (len(indices_to_process) + chunk_size - 1) // chunk_size
        print(f"\n[PHASE 5] Chunked loading + processing "
              f"({total_chunks} chunks of ≤{chunk_size}, "
              f"{'GPU batch' if gpu_available else 'CPU threadpool'})")

        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end   = min(chunk_start + chunk_size, len(indices_to_process))
            chunk_idxs  = indices_to_process[chunk_start:chunk_end]

            print(f"\n  Chunk {chunk_idx+1}/{total_chunks} "
                  f"(samples {chunk_start}–{chunk_end-1})")

            # Load chunk into RAM
            t0 = time.perf_counter()
            chunk_data = []
            for idx in tqdm(chunk_idxs, desc="  Loading", unit="sample", leave=False):
                sample = dataset_obj[idx]
                traj_idx, time_idx = _get_trajectory_and_timestep_info(
                    idx, dataset_obj.metadata)

                x = sample['input_fields'].squeeze()

                # ===== VISCOELASTIC CROP — applied to every signal =====
                if crop_rows is not None:
                    row_start, row_end = crop_rows
                    x = x[row_start:row_end, :, :]   # (178, 512, num_fields)
                # =========================================================

                x = x.reshape(num_nodes, num_fields)
                chunk_data.append((idx, x.numpy(), traj_idx, time_idx))

            timings['preload_data'] += time.perf_counter() - t0
            print(f"    Loaded {len(chunk_data)} samples")

            # Process chunk (GPU batch or CPU threadpool)
            t0 = time.perf_counter()
            if gpu_available:
                _run_gpu_batch_process(chunk_data, gpu_batch_size=gpu_batch_size)
            else:
                _run_threadpool(chunk_data, num_workers)
            elapsed = time.perf_counter() - t0
            timings['parallel_processing'] += elapsed

            del chunk_data
            gc.collect()

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = sum(timings.values())
    n_done = (len(preloaded_data)
              if loader in ('hdf5_kolmogorov', 'hdf5_dynamic')
              else len(indices_to_process))
    print("\n" + "=" * 80)
    print("SPLIT PREPROCESSING COMPLETE")
    for k, v in timings.items():
        print(f"  {k:<30} {v:>8.2f}s")
    print(f"  {'total':<30} {total_time:>8.2f}s")
    if total_time > 0:
        print(f"  Throughput: {n_done / total_time:.2f} samples/sec")
    print("=" * 80)

    return timings


def _run_threadpool(data: List[Tuple], num_workers: int) -> int:
    """Run process_single_sample in parallel; return count of successes."""
    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_sample, item): item[0]
                   for item in data}
        with tqdm(total=len(futures), desc="    Processing", unit="sample",
                  leave=False) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                except Exception as e:
                    print(f"\n  ⚠ Error on sample {futures[future]}: {e}")
                pbar.update(1)
    return completed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_dataset(
        dataset_name: str,
        splits: Optional[List[str]] = None,
        verbose: bool = True,
        num_workers: Optional[int] = None,
        num_samples: Optional[int] = None,
        use_gpu: bool = True,
        gpu_batch_size: int = 64,
        output_dir_override: Optional[str] = None,
        half_precision: bool = True,
) -> Path:
    """
    Process all requested splits for `dataset_name`.

    Args:
        output_dir_override: If set, write the versioned output directly here
            instead of computing the path from DATASET_CONFIGS.  Useful when
            the project quota is tight and you want to write to /tmp first.
        half_precision: If True (default), saves x and coeffs as float16 to
            halve disk usage (~150 GB instead of ~300 GB for turbulent).
            Dataset loader casts back to float32 transparently.

    Returns:
        Path to the versioned output directory (e.g. .../processed/20260407_123456/).
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    raw_base, proc_base = get_dataset_paths(dataset_name)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir_override is not None:
        output_dir = Path(output_dir_override) / timestamp
    else:
        output_dir = proc_base / timestamp

    gpu_available = use_gpu and torch.cuda.is_available()

    if verbose:
        print("=" * 80)
        print(f"DATA PREPROCESSING  dataset={dataset_name}")
        print("=" * 80)
        print(f"  Timestamp   : {timestamp}")
        print(f"  Raw data    : {raw_base}")
        print(f"  Output      : {output_dir}")
        print(f"  Splits      : {splits}")
        print(f"  Workers     : {num_workers or 'auto'}")
        print(f"  GPU accel   : {gpu_available} (batch={gpu_batch_size})")
        print("=" * 80)

    cfg    = DATASET_CONFIGS[dataset_name]
    loader = cfg['loader']

    # Resolve raw dataset directory
    if loader == 'well':
        raw_dir = raw_base / "data"
        if not raw_dir.exists():
            raw_dir = raw_base / "datasets" / dataset_name / "data"
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found for {dataset_name}.\n"
                f"  Checked: {raw_base / 'data'}\n"
                f"           {raw_base / 'datasets' / dataset_name / 'data'}"
            )
        raw_dataset_dir = str(raw_dir.parent)   # parent of 'data/'
    else:
        # HDF5 datasets: base_path contains raw/ subdir
        raw_dataset_dir = str(raw_base)

    split_alias = cfg.get('split_alias', {})

    split_metadata = {}
    for split in splits:
        # Resolve the actual directory/WellDataset split name (handles val vs valid)
        actual_split = split_alias.get(split, split)

        if verbose:
            alias_note = f" (dir='{actual_split}')" if actual_split != split else ""
            print(f"\n{'='*40}\nProcessing split: {split}{alias_note}\n{'='*40}")

        # For WellDataset, verify the split directory exists
        if loader == 'well':
            split_raw_dir = raw_base / "data" / actual_split
            if not split_raw_dir.exists():
                split_raw_dir = raw_base / "datasets" / dataset_name / "data" / actual_split
            if not split_raw_dir.exists():
                print(f"  ⚠ Skipping {split}: raw directory '{actual_split}' not found "
                      f"under {raw_base}/data/")
                continue

        # Output always uses the canonical split name (train/valid/test)
        split_output_dir = output_dir / split
        sample_indices   = list(range(num_samples)) if num_samples is not None else None

        preprocess_split(
            dataset_name=dataset_name,
            raw_dataset_dir=raw_dataset_dir,
            processed_dataset_dir=str(split_output_dir),
            split=actual_split,       # actual dir/WellDataset name
            num_workers=num_workers,
            sample_indices=sample_indices,
            use_gpu=use_gpu,
            gpu_batch_size=gpu_batch_size,
            half_precision=half_precision,
        )

        if split_output_dir.exists() and any(split_output_dir.glob("*.pt")):
            metadata = create_config_files(str(split_output_dir))
            split_metadata[split] = metadata
            if verbose:
                print(f"  ✓ {split}: {metadata['num_samples']} samples, "
                      f"{metadata['num_nodes']} nodes, "
                      f"{metadata['num_fields']} fields")

    if not split_metadata:
        raise RuntimeError("No splits were processed successfully.")

    # Save summary JSON at version root
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'dataset':   dataset_name,
        'timestamp': timestamp,
        'splits':    split_metadata,
    }
    with open(output_dir / 'processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\n✓ Processing summary saved to {output_dir / 'processing_summary.json'}")

    return output_dir


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified data preprocessing for NIGWT experiments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset to preprocess',
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'valid', 'test'],
        help='Which splits to process',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker threads (default: auto = all CPUs, up to 48)',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Max samples per split (default: all). Useful for quick tests.',
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        default=False,
        help='Disable GPU-accelerated batch processing and use CPU threadpool only.',
    )
    parser.add_argument(
        '--gpu-batch-size',
        type=int,
        default=64,
        help='Number of samples per GPU batch (reduce if OOM).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=(
            'Override the output base directory for processed data. '
            'A timestamped sub-folder is created inside this path. '
            'Use this to write to /tmp when project quota is tight, '
            'then upload the result to S3.'
        ),
    )
    parser.add_argument(
        '--no-half',
        action='store_true',
        default=False,
        help=(
            'Disable float16 storage and save x/coeffs as float32. '
            'By default, float16 is used to halve disk usage (~150 GB instead of '
            '~300 GB for turbulent). Training code always receives float32 regardless.'
        ),
    )
    args = parser.parse_args()

    out = process_dataset(
        dataset_name=args.dataset,
        splits=args.splits,
        num_workers=args.workers,
        num_samples=args.num_samples,
        use_gpu=not args.no_gpu,
        gpu_batch_size=args.gpu_batch_size,
        output_dir_override=args.output_dir,
        half_precision=not args.no_half,
    )
    print(f"\nOutput directory: {out}")
