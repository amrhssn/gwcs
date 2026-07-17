"""
Data preparation script for turbulent radiative layer 2D dataset.

This script:
1. Reads existing preprocessed .pt files from local_storage/processed_data/turbulent_radiative_layer_2D/test/
2. Extracts metadata from one sample to create config files
3. Creates chebyshev_synthesis_data.pt with shared Chebyshev data
4. Creates raw/ subdirectory with symbolic links to the .pt files
5. Generates graph_config.json, signal_config.json, gwt_config.json
"""

import os
import json
from pathlib import Path

import torch


def prepare_turbulent_data(
        processed_data_dir: str = "local_storage/processed_data/turbulent_radiative_layer_2D/test",
        create_symlinks: bool = True,
):
    """
    Prepare turbulent data for training by creating config files and organizing data.

    Args:
        processed_data_dir: Path to directory containing preprocessed .pt files
        create_symlinks: Whether to create symbolic links in raw/ (True) or copy files (False)
    """
    from paths import ROOT_PROJECT_PATH

    # Resolve paths
    processed_data_dir = os.path.join(ROOT_PROJECT_PATH, processed_data_dir)
    raw_dir = os.path.join(processed_data_dir, 'raw')

    print("=" * 80)
    print("TURBULENT DATA PREPARATION")
    print("=" * 80)
    print(f"Processed data directory: {processed_data_dir}")
    print()

    # Get list of .pt files (excluding any files in raw/ subdirectory)
    pt_files = sorted(
        [f for f in os.listdir(processed_data_dir)
         if f.endswith('.pt') and os.path.isfile(os.path.join(processed_data_dir, f))],
        key=lambda x: int(x.split('.')[0])
    )

    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found in {processed_data_dir}")

    print(f"Found {len(pt_files)} .pt files: {pt_files[0]} to {pt_files[-1]}")
    print()

    # Load one sample to extract metadata
    sample_path = os.path.join(processed_data_dir, pt_files[0])
    print(f"Loading sample {pt_files[0]} to extract metadata...")
    sample_data = torch.load(sample_path, map_location='cpu', weights_only=False)

    # Extract shapes and metadata
    num_nodes = sample_data.x.shape[0]
    num_fields = sample_data.x.shape[1]
    num_filters = sample_data.coeffs.shape[2]
    num_edges = sample_data.edge_index.shape[1]

    print(f"  - num_nodes: {num_nodes}")
    print(f"  - num_fields: {num_fields}")
    print(f"  - num_filters: {num_filters}")
    print(f"  - num_edges: {num_edges}")
    print()

    # Extract Chebyshev synthesis data (shared across all samples)
    chebyshev_synthesis_data = {
        'chebyshev_coeffs': sample_data.gwt_metadata['chebyshev_coeffs'],
        'chebyshev_eigenvalue_samples': sample_data.gwt_metadata['chebyshev_eigenvalue_samples'],
        'L_sym_sparse': sample_data.graph_metadata['L_sym_sparse'],
        'lambda_min': sample_data.gwt_metadata['lambda_min'],
        'lambda_max': sample_data.gwt_metadata['lambda_max'],
        'chebyshev_max_degree': sample_data.gwt_metadata['chebyshev_max_degree'],
    }

    # Save Chebyshev synthesis data
    chebyshev_path = os.path.join(processed_data_dir, 'chebyshev_synthesis_data.pt')
    print(f"Saving Chebyshev synthesis data to {chebyshev_path}...")
    torch.save(chebyshev_synthesis_data, chebyshev_path)
    print(f"✓ Saved chebyshev_synthesis_data.pt")
    print()

    # Create config files
    # Note: We create minimal configs with essential information
    # The turbulent dataset doesn't have all the parameters of synthetic data

    graph_config = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'graph_type': 'turbulent_radiative_layer_2D',
        'grid_shape': [128, 384],  # Known from EDA
    }

    signal_config = {
        'num_fields': num_fields,
        'field_names': ['density', 'pressure', 'velocity_x', 'velocity_y'],
        'signal_type': 'turbulent_radiative_layer_2D',
    }

    gwt_config = {
        'num_filters': num_filters,
        'num_scales': num_filters - 1,  # Assuming J scales + 1 lowpass
        'lambda_min': float(sample_data.gwt_metadata['lambda_min']),
        'lambda_max': float(sample_data.gwt_metadata['lambda_max']),
        'chebyshev_max_degree': int(sample_data.gwt_metadata['chebyshev_max_degree']),
    }

    # Save config files
    print("Saving config files...")
    with open(os.path.join(processed_data_dir, 'graph_config.json'), 'w') as f:
        json.dump(graph_config, f, indent=2)
    print("✓ Saved graph_config.json")

    with open(os.path.join(processed_data_dir, 'signal_config.json'), 'w') as f:
        json.dump(signal_config, f, indent=2)
    print("✓ Saved signal_config.json")

    with open(os.path.join(processed_data_dir, 'gwt_config.json'), 'w') as f:
        json.dump(gwt_config, f, indent=2)
    print("✓ Saved gwt_config.json")
    print()

    # Create raw/ subdirectory
    print(f"Creating raw/ subdirectory: {raw_dir}")
    os.makedirs(raw_dir, exist_ok=True)

    # Create symlinks or copies in raw/
    if create_symlinks:
        print(f"Creating symbolic links in raw/ for {len(pt_files)} files...")
        for i, pt_file in enumerate(pt_files):
            src = os.path.join(processed_data_dir, pt_file)
            dst = os.path.join(raw_dir, f"{i}.pt")

            # Remove existing link if present
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)

            # Create symlink
            os.symlink(src, dst)

            if (i + 1) % 20 == 0:
                print(f"  Created {i + 1}/{len(pt_files)} symlinks")

        print(f"✓ Created {len(pt_files)} symbolic links")
    else:
        print(f"Copying {len(pt_files)} files to raw/ (this may take a while)...")
        import shutil
        for i, pt_file in enumerate(pt_files):
            src = os.path.join(processed_data_dir, pt_file)
            dst = os.path.join(raw_dir, f"{i}.pt")
            shutil.copy2(src, dst)

            if (i + 1) % 20 == 0:
                print(f"  Copied {i + 1}/{len(pt_files)} files")

        print(f"✓ Copied {len(pt_files)} files")

    print()
    print("=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Data directory: {processed_data_dir}")
    print(f"  - {len(pt_files)} samples in raw/")
    print(f"  - chebyshev_synthesis_data.pt")
    print(f"  - graph_config.json, signal_config.json, gwt_config.json")
    print()
    print("You can now run the training script with:")
    print(f"  python scripts/pretraining_reconstruction_turbulent.py")
    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare turbulent radiative layer 2D data for training')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='local_storage/processed_data/turbulent_radiative_layer_2D/test',
        help='Path to processed data directory (relative to project root)'
    )
    parser.add_argument(
        '--no-symlinks',
        action='store_true',
        help='Copy files instead of creating symbolic links'
    )

    args = parser.parse_args()

    prepare_turbulent_data(
        processed_data_dir=args.data_dir,
        create_symlinks=not args.no_symlinks,
    )
