#!/usr/bin/env python3
"""
Inspect all 5 datasets and generate comprehensive statistics table.
Reads field names DYNAMICALLY from actual data (shared_data.pt or WellDataset)
instead of hardcoding them.
"""
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import h5py
import torch

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_latest_processed_dir(base_path: str) -> Optional[Path]:
    """Find the latest timestamped processed directory under base_path."""
    base = Path(base_path)
    if not base.exists():
        return None

    # Look for timestamped directories that contain a train/ subfolder
    candidates = []
    for d in base.iterdir():
        if d.is_dir() and (d / "train").exists():
            # Check if name looks like a timestamp (digits, underscores, dashes)
            if d.name.replace('_', '').replace('-', '').isdigit():
                candidates.append(d)

    if not candidates:
        return None

    return sorted(candidates)[-1]


def inspect_processed_dataset(base_path: str, name: str) -> Dict[str, Any]:
    """
    Inspect a processed PyTorch dataset by reading shared_data.pt.
    Works for any dataset (turbulent, pbfm, viscoelastic, etc.)
    as long as processed data exists.
    """
    print(f"\nInspecting {name} dataset (processed)...")

    base = Path(base_path)
    version_dir = find_latest_processed_dir(base)

    if version_dir is None:
        return {"error": f"No processed data found in {base}"}

    stats = {
        "dataset": name,
        "processed_path": str(version_dir),
        "splits": {}
    }

    for split_name in ["train", "valid", "test"]:
        split_path = version_dir / split_name
        if not split_path.exists():
            continue

        shared_path = split_path / "shared_data.pt"
        if not shared_path.exists():
            continue

        shared = torch.load(shared_path, map_location="cpu", weights_only=False)
        sample_files = [f for f in split_path.glob("*.pt") if f.stem.isdigit()]

        stats["splits"][split_name] = {
            "num_samples": len(sample_files),
            "num_nodes": shared.get("num_nodes", 0),
            "grid_shape": [shared.get("num_rows", 0), shared.get("num_columns", 0)],
            "num_fields": len(shared.get("field_names", [])),
            "field_names": shared.get("field_names", []),
        }

    return stats


def get_well_field_names(base_path: str, dataset_name: str) -> List[str]:
    """
    Try to read field names from WellDataset metadata.
    Returns empty list if the_well is not installed or dataset not found.
    """
    try:
        from the_well.data import WellDataset
        well_base_path = os.path.dirname(base_path)
        dataset_obj = WellDataset(
            well_base_path=well_base_path,
            well_dataset_name=dataset_name,
            well_split_name='train',
            n_steps_input=1,
            n_steps_output=0,
            use_normalization=False,
        )
        field_names = [
            name
            for group in dataset_obj.metadata.field_names.values()
            for name in group
        ]
        return field_names
    except Exception as e:
        print(f"  Could not read WellDataset metadata for {dataset_name}: {e}")
        return []


def read_hdf5_field_info(h5_path: str) -> List[str]:
    """
    Read field structure from an HDF5 file (The Well format).
    Attempts to discover fields from t0_fields group keys.
    Multi-component fields (e.g., velocity with 2 channels) are expanded
    into separate entries (velocity_x, velocity_y).
    """
    field_names = []
    try:
        with h5py.File(h5_path, 'r') as f:
            # The Well format stores data under t0_fields/
            for group_key in ['t0_fields', 'fields']:
                if group_key in f:
                    grp = f[group_key]
                    for key in sorted(grp.keys()):
                        ds = grp[key]
                        if hasattr(ds, 'shape') and len(ds.shape) >= 3:
                            # shape is typically (n_traj, n_steps, C, H, W)
                            # C is the channel/component dimension
                            n_components = ds.shape[2] if len(ds.shape) >= 5 else 1
                            if n_components == 1:
                                field_names.append(key)
                            elif n_components == 2:
                                field_names.append(f"{key}_x")
                                field_names.append(f"{key}_y")
                            elif n_components == 4:
                                field_names.append(f"{key}_xx")
                                field_names.append(f"{key}_xy")
                                field_names.append(f"{key}_yx")
                                field_names.append(f"{key}_yy")
                            else:
                                for i in range(n_components):
                                    field_names.append(f"{key}_{i}")
                    break  # found field group, stop looking
    except Exception as e:
        print(f"  Error reading HDF5 fields from {h5_path}: {e}")
    return field_names


def inspect_hdf5_dataset(base_path: str, name: str,
                         well_dataset_name: str = None) -> Dict[str, Any]:
    """
    Inspect raw HDF5 dataset.
    First tries WellDataset metadata for field names.
    Falls back to reading HDF5 structure directly.
    """
    print(f"\nInspecting {name} dataset (raw HDF5)...")

    base = Path(base_path)
    data_dir = base / "data"

    if not data_dir.exists():
        return {"error": f"No data directory found at {data_dir}"}

    stats = {
        "dataset": name,
        "raw_path": str(base),
        "splits": {}
    }

    # Try to get field names from WellDataset metadata
    if well_dataset_name is None:
        well_dataset_name = name
    field_names = get_well_field_names(str(base), well_dataset_name)
    field_source = "WellDataset" if field_names else "unknown"

    for split in ["train", "valid", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        h5_files = list(split_dir.glob("*.hdf5")) + list(split_dir.glob("*.h5"))
        if not h5_files:
            continue

        # If we did not get field names from WellDataset, try HDF5 structure
        if not field_names:
            field_names = read_hdf5_field_info(str(h5_files[0]))
            field_source = "HDF5" if field_names else "unknown"

        with h5py.File(h5_files[0], 'r') as f:
            # Get grid dimensions
            x_dim = f['dimensions/x'].shape[0] if 'dimensions/x' in f else 0
            y_dim = f['dimensions/y'].shape[0] if 'dimensions/y' in f else 0
            grid_shape = [y_dim, x_dim]
            num_nodes = y_dim * x_dim

        # Count total samples across all HDF5 files in this split
        total_samples = 0
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as hf:
                    # Look for data arrays to count samples
                    for group_key in ['t0_fields', 'fields']:
                        if group_key in hf:
                            subkeys = list(hf[group_key].keys())
                            if subkeys:
                                shape = hf[f"{group_key}/{subkeys[0]}"].shape
                                n_traj = shape[0]
                                n_steps = shape[1] if len(shape) > 2 else 1
                                total_samples += n_traj * n_steps
                            break
            except Exception as e:
                print(f"  Warning: could not count samples in {h5_file}: {e}")

        stats["splits"][split] = {
            "num_samples": total_samples,
            "num_nodes": num_nodes,
            "grid_shape": grid_shape,
            "num_fields": len(field_names),
            "field_names": field_names,
            "field_source": field_source,
        }

    return stats


def generate_statistics_table(all_stats: Dict[str, Dict]) -> str:
    """Generate formatted ASCII table."""

    rows = []
    for dataset_name in ["turbulent", "pbfm", "active_matter", "shear_flow", "viscoelastic"]:
        stats = all_stats.get(dataset_name, {})

        if "error" in stats:
            rows.append({
                "Dataset": dataset_name,
                "Grid": "N/A",
                "Nodes": "N/A",
                "Fields": 0,
                "Field Names": f"ERROR: {stats['error'][:40]}",
                "Train": 0,
                "Val": 0,
                "Test": 0,
            })
            continue

        splits = stats.get("splits", {})
        train = splits.get("train", {})
        valid = splits.get("valid", splits.get("val", {}))
        test = splits.get("test", {})

        grid_shape = train.get("grid_shape", [0, 0])
        grid_str = f"{grid_shape[0]}x{grid_shape[1]}" if grid_shape[0] > 0 else "N/A"

        field_names = train.get("field_names", [])
        field_str = ", ".join(field_names[:3])
        if len(field_names) > 3:
            field_str += "..."

        rows.append({
            "Dataset": dataset_name,
            "Grid": grid_str,
            "Nodes": f"{train.get('num_nodes', 0):,}",
            "Fields": train.get("num_fields", 0),
            "Field Names": field_str,
            "Train": train.get("num_samples", 0),
            "Val": valid.get("num_samples", 0),
            "Test": test.get("num_samples", 0),
        })

    headers = ["Dataset", "Grid", "Nodes", "Fields", "Field Names", "Train", "Val", "Test"]
    col_widths = {h: len(h) for h in headers}

    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    sep = "=" * 120
    table = [sep, "DATASET STATISTICS TABLE", sep]

    # Header
    header_row = "  ".join(f"{h:<{col_widths[h]}}" for h in headers)
    table.append(header_row)
    table.append("-" * 120)

    # Data rows
    for row in rows:
        data_row = ""
        for h in headers:
            val = str(row[h])
            if h in ["Nodes", "Train", "Val", "Test"]:
                data_row += f"{val:>{col_widths[h]}}  "
            else:
                data_row += f"{val:<{col_widths[h]}}  "
        table.append(data_row.rstrip())

    table.append(sep)
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="Inspect all 5 datasets")
    parser.add_argument("--output-json", default="dataset_statistics.json")
    args = parser.parse_args()

    # Dataset paths
    # Processed data paths (checked first for every dataset)
    # EDIT ME: if turbulent's raw/processed data lives in a separate (e.g.
    # collaborator-owned) location, override COLLAB_PROJECT_ROOT below.
    collab_project_root = Path(os.environ.get("COLLAB_RAW_DATA_ROOT", str(PROJECT_ROOT)))
    turbulent_processed = str(collab_project_root / "local_storage/datasets/turbulent_radiative_layer_2D/processed")
    pbfm_processed = str(PROJECT_ROOT / "local_storage/datasets/pbfm/processed")
    viscoelastic_processed = str(PROJECT_ROOT / "local_storage/datasets/viscoelastic_instability/processed")

    all_stats = {}

    # Turbulent - always processed
    all_stats["turbulent"] = inspect_processed_dataset(turbulent_processed, "turbulent")

    # PBFM - always processed
    all_stats["pbfm"] = inspect_processed_dataset(pbfm_processed, "pbfm")

    # Viscoelastic - check processed first, fall back to raw HDF5
    vis_stats = inspect_processed_dataset(viscoelastic_processed, "viscoelastic")
    if "error" in vis_stats:
        print("  No processed viscoelastic data, trying raw HDF5...")
        vis_stats = inspect_hdf5_dataset(
            viscoelastic_raw, "viscoelastic",
            well_dataset_name="viscoelastic_instability"
        )
    all_stats["viscoelastic"] = vis_stats

    # Active matter - raw HDF5
    #all_stats["active_matter"] = inspect_hdf5_dataset(
    #    active_matter_raw, "active_matter",
    #    well_dataset_name="active_matter"
    #)

    # Shear flow - raw HDF5
    #all_stats["shear_flow"] = inspect_hdf5_dataset(
    #    shear_flow_raw, "shear_flow",
    #    well_dataset_name="shear_flow"
    #)

    table_str = generate_statistics_table(all_stats)
    print("\n" + table_str)

    with open(args.output_json, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved statistics to {args.output_json}")


if __name__ == "__main__":
    main()
