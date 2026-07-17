"""
pipeline.py — Unified Pipeline Orchestrator
============================================
Runs the full pipeline: [Download] → [Preprocess] → [Train]

Supports: turbulent_radiative_layer_2D | viscoelastic_instability | pbfm

Consolidates run_pipeline.py and run_pipelineV.py into one script.
Training is launched via torchrun (DDP). This script stays single-process.

Usage:
    # Turbulent: download + preprocess + train (4 GPU)
    python pipeline.py --dataset turbulent_radiative_layer_2D --num-gpus 4

    # Viscoelastic: skip download, run preprocess + train
    python pipeline.py --dataset viscoelastic_instability --skip-download --num-gpus 4

    # PBFM: skip download + preprocess, resume training
    python pipeline.py --dataset pbfm --skip-download --skip-preprocessing \\
                       --resume "2026-03-23_13-48-59" --num-gpus 4

    # Preprocess only
    python pipeline.py --dataset viscoelastic_instability --skip-download --skip-training
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import ROOT_PROJECT_PATH
from filebase_sync import (
    check_processed_data_exists_local,
    check_processed_data_exists_s3,
    download_processed_data,
    list_processed_data_timestamps,
    find_latest_processed_data_local,
    check_run_exists_local,
    check_run_exists_s3,
    download_run_checkpoint,
)

SCRIPT_DIR = Path(__file__).parent


# =============================================================================
# TERMINAL COLOURS
# =============================================================================

class C:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    WARN    = '\033[93m'
    FAIL    = '\033[91m'
    END     = '\033[0m'
    BOLD    = '\033[1m'


def hdr(msg):  print(f"\n{'='*80}\n{C.BOLD}{C.HEADER}{msg}{C.END}\n{'='*80}")
def ok(msg):   print(f"{C.GREEN}✓ {msg}{C.END}")
def err(msg):  print(f"{C.FAIL}✗ {msg}{C.END}")
def warn(msg): print(f"{C.WARN}⚠ {msg}{C.END}")
def info(msg): print(f"{C.CYAN}ℹ {msg}{C.END}")


# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def validate_environment() -> bool:
    info("Validating environment...")

    env_path = Path(ROOT_PROJECT_PATH) / ".env"
    if not env_path.exists():
        err(f".env file not found at {env_path}")
        return False
    ok(".env found")

    import dotenv
    dotenv.load_dotenv(env_path)

    try:
        import torch
        import torch_geometric
        ok("Required packages (torch, torch_geometric) installed")
    except ImportError as e:
        err(f"Missing package: {e}")
        return False

    return True


# =============================================================================
# STEP 1: DOWNLOAD  (turbulent / WellDataset only)
# =============================================================================

def step_download(test_mode: bool = False) -> bool:
    hdr("STEP 1: DATA DOWNLOAD")
    try:
        from data_collection_step import download_turbulent_dataset
        info(f"Test mode: {'ENABLED (first_only=True)' if test_mode else 'DISABLED'}")
        download_turbulent_dataset(
            splits=["train", "valid", "test"],
            verbose=True,
            first_only=test_mode,
            skip_if_exists=True,
        )
        ok("Data ready")
        return True
    except Exception as e:
        err(f"Download failed: {e}")
        import traceback; traceback.print_exc()
        return False


# =============================================================================
# STEP 2: PREPROCESS
# =============================================================================

def step_preprocess(
        dataset: str,
        test_mode: bool = False,
        num_workers: int = 8,
) -> bool:
    hdr("STEP 2: PREPROCESSING")
    try:
        from preprocess import process_dataset   # unified preprocess.py
        num_samples = 10 if test_mode else None
        info(f"dataset={dataset}  workers={num_workers}  "
             f"num_samples={'10 (test)' if test_mode else 'all'}")

        out = process_dataset(
            dataset_name=dataset,
            splits=["train", "valid", "test"],
            num_workers=num_workers,
            num_samples=num_samples,
        )
        ok(f"Preprocessing complete → {out}")
        return True
    except Exception as e:
        err(f"Preprocessing failed: {e}")
        import traceback; traceback.print_exc()
        return False


# =============================================================================
# STEP 3: TRAIN
# =============================================================================

def step_train(
        dataset: str,
        num_gpus: int = 1,
        test_mode: bool = False,
        resume: Optional[str] = None,
        use_normalization: bool = False,
        hidden_dim: int = 256,
        num_layers: int = 5,
        processed_dir: Optional[str] = None,
) -> bool:
    hdr("STEP 3: TRAINING")
    try:
        info(f"dataset={dataset}  gpus={num_gpus}  "
             f"resume={resume}  normalization={use_normalization}  "
             f"hidden_dim={hidden_dim}  num_layers={num_layers}")

        training_script = SCRIPT_DIR / 'train.py'   # unified train.py
        cmd = [
            'torchrun',
            f'--nproc_per_node={num_gpus}',
            str(training_script),
            '--dataset', dataset,
            '--hidden-dim', str(hidden_dim),
            '--num-layers', str(num_layers),
        ]
        if processed_dir is not None:
            cmd += ['--processed-dir', processed_dir]
        if test_mode:
            cmd += ['--max-samples', '10']
        if resume:
            cmd += ['--resume', resume]
        if use_normalization:
            cmd += ['--use-normalization']

        info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            err(f"torchrun exited with code {result.returncode}")
            return False
        ok("Training complete")
        return True
    except Exception as e:
        err(f"Training failed: {e}")
        import traceback; traceback.print_exc()
        return False


# =============================================================================
# PROCESSED DATA DISCOVERY (local + S3)
# =============================================================================

def find_processed_data(
        dataset: str,
        timestamp: Optional[str] = None,
        test_mode: bool = False,
) -> Optional[Path]:
    hdr("CHECKING FOR PROCESSED DATA")

    if timestamp:
        info(f"Looking for timestamp: {timestamp}")
        exists_local, local_path = check_processed_data_exists_local(
            timestamp, dataset=dataset)
        if exists_local:
            ok(f"Found locally: {local_path}")
            return local_path

        warn("Not found locally — checking S3...")
        if check_processed_data_exists_s3(timestamp, dataset=dataset, verbose=True):
            dl = download_processed_data(
                timestamp=timestamp, dataset=dataset,
                test_mode=test_mode, verbose=True)
            if dl:
                ok(f"Downloaded from S3: {dl}")
                return dl
        raise FileNotFoundError(
            f"Processed data '{timestamp}' not found locally or in S3")

    info("Looking for latest processed data...")
    latest = find_latest_processed_data_local(dataset=dataset)
    if latest:
        ts, path = latest
        ok(f"Found latest local: {ts} → {path}")
        return path

    warn("No local data — checking S3...")
    try:
        timestamps = list_processed_data_timestamps(dataset=dataset, verbose=True)
        if timestamps:
            dl = download_processed_data(
                timestamp=timestamps[0], dataset=dataset,
                test_mode=test_mode, verbose=True)
            if dl:
                ok(f"Downloaded latest from S3: {dl}")
                return dl
    except Exception as e:
        warn(f"S3 check failed: {e}")

    return None


def find_checkpoint(resume_timestamp: str) -> Path:
    hdr("CHECKING FOR TRAINING CHECKPOINT")
    info(f"Looking for run: {resume_timestamp}")

    exists_local, local_path = check_run_exists_local(resume_timestamp)
    if exists_local:
        ok(f"Found locally: {local_path}")
        return local_path

    warn("Not found locally — checking S3...")
    if check_run_exists_s3(resume_timestamp, verbose=True):
        dl = download_run_checkpoint(timestamp=resume_timestamp, verbose=True)
        if dl:
            ok(f"Downloaded from S3: {dl}")
            return dl

    raise FileNotFoundError(
        f"Run '{resume_timestamp}' not found locally or in S3")


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(
        dataset: str,
        skip_download: bool = False,
        skip_preprocessing: bool = False,
        skip_training: bool = False,
        test_mode: bool = False,
        num_workers: int = 48,
        num_gpus: int = 1,
        resume: Optional[str] = None,
        processed_data_timestamp: Optional[str] = None,
        use_normalization: bool = False,
        hidden_dim: int = 256,
        num_layers: int = 5,
        processed_dir: Optional[str] = None,
) -> bool:
    t0 = time.time()
    hdr(f"NIGWT PIPELINE  — dataset={dataset}")
    print(f"  Test mode:   {test_mode}")
    print(f"  Workers:     {num_workers}")
    print(f"  GPUs:        {num_gpus}")
    print(f"  Normalise:   {use_normalization}")
    print(f"  Hidden dim:  {hidden_dim}")
    print(f"  Num layers:  {num_layers}")

    if not validate_environment():
        err("Environment validation failed")
        return False
    ok("Environment valid\n")

    # STEP 1: Download (turbulent WellDataset only)
    if not skip_download:
        if dataset in ('turbulent_radiative_layer_2D',):
            if not step_download(test_mode=test_mode):
                return False
        else:
            info(f"Skipping download for dataset '{dataset}' (data already on disk)")
    else:
        warn("Skipping Step 1: Download")

    # STEP 2: Preprocess
    if not skip_preprocessing:
        if not step_preprocess(dataset=dataset, test_mode=test_mode,
                               num_workers=num_workers):
            return False
    else:
        warn("Skipping Step 2: Preprocessing")
        if processed_data_timestamp:
            try:
                find_processed_data(
                    dataset=dataset, timestamp=processed_data_timestamp,
                    test_mode=test_mode)
            except FileNotFoundError as e:
                err(str(e))
                return False

    # STEP 3: Train
    if not skip_training:
        if resume:
            try:
                find_checkpoint(resume)
            except FileNotFoundError as e:
                err(str(e))
                return False

        if not step_train(
                dataset=dataset,
                num_gpus=num_gpus,
                test_mode=test_mode,
                resume=resume,
                use_normalization=use_normalization,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                processed_dir=processed_dir,
        ):
            return False
    else:
        warn("Skipping Step 3: Training")

    elapsed = time.time() - t0
    hdr("PIPELINE COMPLETE")
    ok(f"Total time: {elapsed/60:.1f} min")
    return True


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run the NIGWT pipeline for any supported dataset.'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['turbulent_radiative_layer_2D', 'viscoelastic_instability',
                 'kolmogorov', 'dynamic_stall'],
        help='Dataset to process/train on',
    )
    parser.add_argument('--test-mode', action='store_true',
                        help='Quick test run (10 samples, first_only download)')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--skip-preprocessing', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--workers', type=int, default=48,
                        help='Preprocessing worker threads (default: 48)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='GPUs for DDP training (default: 1)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Run timestamp to resume (e.g. "2026-03-23_13-48-59")')
    parser.add_argument('--processed-data-timestamp', type=str, default=None,
                        help='Specific processed data version timestamp')
    parser.add_argument('--use-normalization', action='store_true',
                        help='Apply Z-score normalisation during training')
    parser.add_argument('--processed-dir', type=str, default=None,
                        help=(
                            'Override the default processed-data directory. '
                            'Must be the directory containing the timestamped version '
                            'subdirs (e.g. /path/to/datasets/viscoelastic/processed/). '
                            'Runs directory is still chosen by --dataset name.'
                        ))
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help=(
                            'GNN hidden dimension (default: 256 → ~993K params). '
                            'Use 96 for ~99K params matching GXN baseline.'
                        ))
    parser.add_argument('--num-layers', type=int, default=5,
                        help=(
                            'Number of GNN processor layers (default: 5). '
                            'Use 3 together with --hidden-dim 96 for ~99K params.'
                        ))

    args = parser.parse_args()

    success = run_pipeline(
        dataset=args.dataset,
        skip_download=args.skip_download,
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training,
        test_mode=args.test_mode,
        num_workers=args.workers,
        num_gpus=args.num_gpus,
        resume=args.resume,
        processed_data_timestamp=args.processed_data_timestamp,
        use_normalization=args.use_normalization,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        processed_dir=args.processed_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
