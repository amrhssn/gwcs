"""
Download script for turbulent_radiative_layer_2D dataset from The Well.

Downloads train, valid, and test splits to:
    <project_root>/local_storage/datasets/turbulent_radiative_layer_2D/raw/[train|valid|test]

Configuration is read from .env file in project root.
"""

import os
import shutil
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from the_well.utils.download import well_download

from paths import ROOT_PROJECT_PATH


def get_base_path_from_env() -> Path:
    """
    Load base path from .env file and resolve to absolute path.

    Returns:
        Absolute path to: <project_root>/<TURBULENT_DATA_BASE_PATH>
    """
    env_path = Path(ROOT_PROJECT_PATH) / ".env"
    if not env_path.exists():
        raise FileNotFoundError(
            f".env file not found at {env_path}\n"
            f"Please create it with: TURBULENT_DATA_BASE_PATH=local_storage/datasets/turbulent_radiative_layer_2D"
        )

    load_dotenv(env_path)

    relative_path = os.getenv("TURBULENT_DATA_BASE_PATH")
    if relative_path is None:
        raise ValueError(
            "TURBULENT_DATA_BASE_PATH not found in .env file.\n"
            "Please add: TURBULENT_DATA_BASE_PATH=local_storage/datasets/turbulent_radiative_layer_2D"
        )

    base_path = Path(ROOT_PROJECT_PATH) / relative_path
    return base_path


def download_turbulent_dataset(
        splits: list[Literal["train", "valid", "test"]] = None,
        verbose: bool = True,
        first_only: bool = False,
        skip_if_exists: bool = True
) -> Path:
    """
    Download turbulent_radiative_layer_2D dataset splits.

    Args:
        splits: List of splits to download. Defaults to ["train", "valid", "test"].
        verbose: Print progress information.
        first_only: If True, download only the first file of each split (for testing).
        skip_if_exists: If True, skip downloading splits that already have data.

    Returns:
        Path to the base data directory.
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    base_path = get_base_path_from_env()

    if verbose:
        print("=" * 70)
        print("TURBULENT RADIATIVE LAYER 2D DATASET DOWNLOAD")
        print("=" * 70)
        print(f"Base path: {base_path}")
        print(f"Splits to download: {splits}")
        print(f"Skip if exists: {skip_if_exists}")
        print("=" * 70)

    # Create base directory if it doesn't exist
    base_path.mkdir(parents=True, exist_ok=True)

    # Download each split
    for split in splits:
        # Check if data already exists (check both possible locations)
        possible_paths = [
            base_path / "data" / split,  # Direct path: base_path/data/split
            base_path / "datasets" / "turbulent_radiative_layer_2D" / "data" / split,  # The Well default
        ]

        if skip_if_exists:
            for data_dir in possible_paths:
                if data_dir.exists():
                    h5_files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))
                    if h5_files:
                        if verbose:
                            print(f"\n✓ {split} split already exists at {data_dir} ({len(h5_files)} files) - skipping")
                        break
            else:
                # No existing data found, download it
                if verbose:
                    print(f"\nDownloading {split} split...")

                well_download(
                    first_only=first_only,
                    base_path=str(base_path),
                    dataset="turbulent_radiative_layer_2D",
                    split=split
                )

                if verbose:
                    print(f"✓ {split} split downloaded successfully")
            continue

        # If skip_if_exists is False, always download
        if verbose:
            print(f"\nDownloading {split} split...")

        well_download(
            first_only=first_only,
            base_path=str(base_path),
            dataset="turbulent_radiative_layer_2D",
            split=split
        )

        if verbose:
            print(f"✓ {split} split downloaded successfully")

    # Reorganize directory structure
    # The Well creates: base_path/datasets/turbulent_radiative_layer_2D/data/
    # We want:          base_path/data/
    data_dir = base_path / "data"
    if not data_dir.exists():
        data_dir = base_path / "datasets" / "turbulent_radiative_layer_2D" / "data"

    if verbose:
        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"Data location: {base_path}")
        print("=" * 70)

    return base_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download turbulent_radiative_layer_2D dataset from The Well"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
        help="Splits to download (default: all)"
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Download only the first file of each split (for testing)"
    )

    args = parser.parse_args()
    download_turbulent_dataset(splits=args.splits, first_only=args.first_only)
