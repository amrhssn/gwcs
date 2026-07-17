"""
Filebase S3 Sync Utilities for Graph Wavelet Compressed Sensing
===============================================================

Handles syncing of processed data and training runs to/from Filebase S3.

Updated: 2026-03-24 - Added multi-dataset support (turbulent, pbfm, active_matter, shear_flow, viscoelastic)
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

from paths import ROOT_PROJECT_PATH

# Load environment variables
load_dotenv(Path(ROOT_PROJECT_PATH) / ".env")

# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://s3.filebase.com")
S3_BUCKET = os.getenv("S3_BUCKET", "graph-wavelet-compressed-sensing")

# Dataset path mapping
DATASET_PATHS = {
    'turbulent_radiative_layer_2D': 'local_storage/datasets/turbulent_radiative_layer_2D/processed',
    'active_matter': 'local_storage/datasets/active_matter/processed',
    'shear_flow': 'local_storage/datasets/shear_flow/processed',
    'viscoelastic_instability': 'local_storage/datasets/viscoelastic_instability/processed',
    'pbfm': 'local_storage/datasets/pbfm/processed',
}


def get_s3_client():
    """Get configured S3 client for Filebase."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise ValueError("AWS credentials not set. Check .env file.")
    
    return boto3.client(
        's3',
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def check_processed_data_exists_local(
    timestamp: str,
    dataset: str = 'turbulent_radiative_layer_2D'
) -> Tuple[bool, Optional[Path]]:
    """
    Check if processed data exists locally for a specific timestamp.
    
    Args:
        timestamp: Timestamp in format YYYYMMDD_HHMMSS
        dataset: Dataset name (turbulent_radiative_layer_2D, active_matter, etc.)
        
    Returns:
        Tuple of (exists: bool, path: Optional[Path])
    """
    if dataset not in DATASET_PATHS:
        return False, None
    
    base_storage = Path(ROOT_PROJECT_PATH) / "local_storage" / "datasets"

    # Build path: <ROOT_PROJECT_PATH>/local_storage/datasets/{dataset}/processed/{timestamp}
    processed_path = base_storage / dataset / "processed" / timestamp
    
    
    # ✅ ADD THIS DEBUG PRINT
    print(f"🔍 DEBUG: Checking for processed data:")
    print(f"   Dataset: {dataset}")
    print(f"   Timestamp: {timestamp}")
    #print(f"   DATASET_PATHS[dataset]: {DATASET_PATHS[dataset]}")
    #print(f"   ROOT_PROJECT_PATH: {ROOT_PROJECT_PATH}")
    print(f"   base_path: {base_storage}")
    print(f"   processed_path: {processed_path}")
    print(f"   processed_path.exists(): {processed_path.exists()}")
    
    #raw_path = Path(DATASET_PATHS[dataset])
    #if not raw_path.is_absolute():
    #    raw_path = Path(ROOT_PROJECT_PATH) / raw_path
    #processed_path = raw_path / timestamp
    
    # Check if directory exists and contains required files
    if not processed_path.exists():
        return False, None
    
    # Verify it has at least train + test splits ('valid' is optional for turbulent).
    has_all_splits = (
        (processed_path / 'train').exists() and
        (processed_path / 'test').exists()
    )
    
    if not has_all_splits:
        return False, None
    
    return True, processed_path


def find_latest_processed_data_local(
    dataset: str = 'turbulent_radiative_layer_2D'
) -> Optional[Tuple[str, Path]]:
    """
    Find the latest processed data directory locally.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Tuple of (timestamp: str, path: Path) or None if not found
    """
    if dataset not in DATASET_PATHS:
        return None
    
    base_path = Path(ROOT_PROJECT_PATH) / DATASET_PATHS[dataset]
    
    if not base_path.exists():
        return None
    
    # Find all timestamp directories
    timestamp_dirs = []
    for entry in base_path.iterdir():
        if entry.is_dir():
            parts = entry.name.split('_')
            if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
                try:
                    int(parts[0])
                    int(parts[1])
                    # Verify it has at least train + test splits.
                    # 'valid' is optional: turbulent_radiative_layer_2D only has
                    # train + test (its WellDataset uses 'val', not 'valid', and
                    # that directory is absent at this cluster path).
                    has_splits = (
                        (entry / 'train').exists() and
                        (entry / 'test').exists()
                    )
                    if has_splits:
                        timestamp_dirs.append(entry.name)
                except ValueError:
                    continue
    
    if not timestamp_dirs:
        return None
    
    # Sort descending (latest first)
    timestamp_dirs.sort(reverse=True)
    latest = timestamp_dirs[0]
    
    return latest, base_path / latest


def check_processed_data_exists_s3(
    timestamp: str,
    dataset: str = 'turbulent_radiative_layer_2D',
    verbose: bool = False
) -> bool:
    """
    Check if processed data exists in S3 for a specific timestamp.
    
    Args:
        timestamp: Timestamp in format YYYYMMDD_HHMMSS
        dataset: Dataset name
        verbose: Print detailed info
        
    Returns:
        True if exists in S3, False otherwise
    """
    try:
        s3 = get_s3_client()
        
        # S3 prefix for this dataset/timestamp
        prefix = f"processed_data/{dataset}/{timestamp}/"
        
        # List objects with this prefix
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=1
        )
        
        exists = response.get('KeyCount', 0) > 0
        
        if verbose:
            if exists:
                print(f"✓ Processed data {timestamp} found in S3")
            else:
                print(f"✗ Processed data {timestamp} not found in S3")
        
        return exists
        
    except Exception as e:
        if verbose:
            print(f"Error checking S3: {e}")
        return False


def list_processed_data_timestamps(
    dataset: str = 'turbulent_radiative_layer_2D',
    verbose: bool = False
) -> List[str]:
    """
    List all available processed data timestamps in S3.
    
    Args:
        dataset: Dataset name
        verbose: Print detailed info
        
    Returns:
        List of timestamps (sorted descending)
    """
    try:
        s3 = get_s3_client()
        
        prefix = f"processed_data/{dataset}/"
        
        # List all "directories" under this prefix
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            Delimiter='/'
        )
        
        timestamps = []
        for common_prefix in response.get('CommonPrefixes', []):
            # Extract timestamp from prefix like "processed_data/dataset/20260204_193616/"
            timestamp = common_prefix['Prefix'].rstrip('/').split('/')[-1]
            timestamps.append(timestamp)
        
        # Sort descending (latest first)
        timestamps.sort(reverse=True)
        
        if verbose:
            print(f"Found {len(timestamps)} processed data versions in S3 for {dataset}")
            for ts in timestamps[:5]:  # Show first 5
                print(f"  - {ts}")
            if len(timestamps) > 5:
                print(f"  ... and {len(timestamps) - 5} more")
        
        return timestamps
        
    except Exception as e:
        if verbose:
            print(f"Error listing S3 timestamps: {e}")
        return []


def download_processed_data(
    timestamp: str,
    dataset: str = 'turbulent_radiative_layer_2D',
    test_mode: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Download processed data from S3 to local storage.
    
    Args:
        timestamp: Timestamp to download
        dataset: Dataset name
        test_mode: If True, download only first 10 files per split
        verbose: Print detailed info
        
    Returns:
        Path to downloaded data, or None if failed
    """
    try:
        s3 = get_s3_client()
        
        if dataset not in DATASET_PATHS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        base_path = Path(ROOT_PROJECT_PATH) / DATASET_PATHS[dataset]
        local_dir = base_path / timestamp
        local_dir.mkdir(parents=True, exist_ok=True)
        
        s3_prefix = f"processed_data/{dataset}/{timestamp}/"
        
        if verbose:
            mode_str = "limited (test mode)" if test_mode else "full dataset"
            print(f"ℹ Downloading processed data {timestamp} from S3 ({mode_str})...")
        
        # List all files
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix)
        
        files_to_download = []
        for page in pages:
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                # Skip the prefix itself
                if s3_key == s3_prefix:
                    continue
                files_to_download.append(s3_key)
        
        if test_mode:
            # Limit to first 10 files per split
            limited_files = []
            for split in ['train', 'valid', 'test']:
                split_files = [f for f in files_to_download if f'/{split}/' in f]
                limited_files.extend(split_files[:10])
            files_to_download = limited_files
        
        if verbose:
            print(f"ℹ Downloading {len(files_to_download)} files...")
        
        # Download files with progress bar
        for s3_key in tqdm(files_to_download, desc="Downloading", disable=not verbose):
            # Compute local path
            rel_path = s3_key[len(s3_prefix):]
            local_file = local_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download
            s3.download_file(S3_BUCKET, s3_key, str(local_file))
        
        if verbose:
            print(f"✓ Downloaded to: {local_dir}")
        
        return local_dir
        
    except Exception as e:
        if verbose:
            print(f"Error downloading from S3: {e}")
        return None


def check_run_exists_local(timestamp: str) -> Tuple[bool, Optional[Path]]:
    """
    Check if training run exists locally.
    
    Args:
        timestamp: Run timestamp
        
    Returns:
        Tuple of (exists: bool, path: Optional[Path])
    """
    runs_dir = Path(ROOT_PROJECT_PATH) / "local_storage" / "runs_turbulent"
    run_path = runs_dir / timestamp
    
    if run_path.exists() and run_path.is_dir():
        return True, run_path
    
    return False, None


def check_run_exists_s3(timestamp: str, verbose: bool = False) -> bool:
    """
    Check if training run exists in S3.
    
    Args:
        timestamp: Run timestamp
        verbose: Print detailed info
        
    Returns:
        True if exists, False otherwise
    """
    try:
        s3 = get_s3_client()
        
        prefix = f"runs/{timestamp}/"
        
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=1
        )
        
        exists = response.get('KeyCount', 0) > 0
        
        if verbose:
            if exists:
                print(f"✓ Training run {timestamp} found in S3")
            else:
                print(f"✗ Training run {timestamp} not found in S3")
        
        return exists
        
    except Exception as e:
        if verbose:
            print(f"Error checking S3: {e}")
        return False


def download_run_checkpoint(
    timestamp: str,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Download training run from S3.
    
    Args:
        timestamp: Run timestamp
        verbose: Print detailed info
        
    Returns:
        Path to downloaded run, or None if failed
    """
    try:
        s3 = get_s3_client()
        
        runs_dir = Path(ROOT_PROJECT_PATH) / "local_storage" / "runs_turbulent"
        local_dir = runs_dir / timestamp
        local_dir.mkdir(parents=True, exist_ok=True)
        
        s3_prefix = f"runs/{timestamp}/"
        
        if verbose:
            print(f"ℹ Downloading run {timestamp} from S3...")
        
        # List all files
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix)
        
        files_to_download = []
        for page in pages:
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                if s3_key == s3_prefix:
                    continue
                files_to_download.append(s3_key)
        
        if verbose:
            print(f"ℹ Downloading {len(files_to_download)} files...")
        
        for s3_key in tqdm(files_to_download, desc="Downloading", disable=not verbose):
            rel_path = s3_key[len(s3_prefix):]
            local_file = local_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            s3.download_file(S3_BUCKET, s3_key, str(local_file))
        
        if verbose:
            print(f"✓ Downloaded to: {local_dir}")
        
        return local_dir
        
    except Exception as e:
        if verbose:
            print(f"Error downloading from S3: {e}")
        return None