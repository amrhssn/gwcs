import os
import json
from typing import Tuple, Dict, Literal, Optional, List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import grid, remove_self_loops, is_undirected, to_undirected
from the_well.data import WellDataset
from the_well.data.datasets import WellMetadata
import tensorflow as tf
from tqdm import tqdm

from paths import ROOT_PROJECT_PATH
from data.utils_graph import compute_sparse_L_sym
from graph_wavelet_transform.chebyshev import chebyshev_gwt_analysis
from visualization.gwt_coeffs import visualize_grid_graph_sgwt_coefficients


def get_raw_data_dir(
        raw_data_base_dir: str,
        dataset_name: str
) -> str:
    if dataset_name == "turbulent_radiative_layer_2D":
        raw_dataset_dir = os.path.join(raw_data_base_dir,
                                       "the_well",
                                       "datasets",
                                       "turbulent_radiative_layer_2D")
    elif dataset_name == "cylinder_flow":
        raw_dataset_dir = os.path.join(raw_data_base_dir,
                                       "MeshGraphNets",
                                       "datasets",
                                       "cylinder_flow")
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")
    return raw_dataset_dir


def get_trajectory_and_timestep_info_turbulent_radiative_layer_2D_dataset(
        sample_index: int,
        metadata: WellMetadata
) -> Tuple[int, int]:
    cumulative_samples = 0
    for file_idx in range(metadata.n_files):
        n_steps = metadata.n_steps_per_trajectory[file_idx]
        n_trajectories = metadata.n_trajectories_per_file[file_idx]

        for traj_index in range(n_trajectories):
            # Each trajectory has n_steps samples
            if sample_index < cumulative_samples + n_steps:
                timestep = sample_index - cumulative_samples
                trajectory_id = cumulative_samples // n_steps  # Global trajectory ID
                return trajectory_id, timestep
            cumulative_samples += n_steps
    raise IndexError(f"Sample index {sample_index} out of range")



def preprocess_turbulent_radiative_layer_2D_dataset(
        dataset_name: str,
        raw_dataset_dir: str,
        processed_data_dir: str,
        split: Literal["train", "valid", "test"],
        output_dir: str = None,
        sample_indices: Optional[List[int]] = None,
) -> None:
    """
    Sequential preprocessing for turbulent_radiative_layer_2D dataset.

    Args:
        dataset_name: Name of the dataset
        raw_dataset_dir: Path to raw data directory
        processed_data_dir: Base path for processed data
        split: Data split to process
        output_dir: Optional specific output directory (overrides processed_data_dir)
        sample_indices: Optional list of specific sample indices to process.
                        If None, processes all samples.
    """
    # configuring the directory for saving the processed Data objects
    if output_dir is not None:
        processed_dataset_dir = output_dir
    else:
        processed_dataset_dir = os.path.join(processed_data_dir, dataset_name, split)
    os.makedirs(processed_dataset_dir, exist_ok=True)

    # Print processing information
    print(f"\n{'=' * 70}")
    print(f"SEQUENTIAL Processing Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Raw data path: {raw_dataset_dir}")
    print(f"Processed data will be saved to: {processed_dataset_dir}")
    print(f"{'=' * 70}\n")

    # initializing the WellDataset class for turbulent_radiative_layer_2D_dataset
    well_base_path = os.path.dirname(raw_dataset_dir)
    dataset = WellDataset(
        well_base_path=well_base_path,
        well_dataset_name=dataset_name,
        well_split_name=split,
        n_steps_input=1,
        n_steps_output=0,
        use_normalization=False,
    )

    # extracting field metadata
    field_names = [
        name for group in dataset.metadata.field_names.values() for name in group
    ]
    num_fields = len(field_names)
    index_to_field = {idx: field for idx, field in enumerate(field_names)}
    field_to_index = {field: idx for idx, field in enumerate(field_names)}

    # extracting dataset's number of samples
    total_samples = len(dataset)

    # Determine which samples to process
    if sample_indices is not None:
        indices_to_process = sample_indices
        print(f"Processing {len(indices_to_process)} selected samples (out of {total_samples} total)")
    else:
        indices_to_process = list(range(total_samples))
        print(f"Total individual samples: {total_samples}")

    print(f"Number of trajectories: {len(dataset.metadata.n_steps_per_trajectory)}")
    print(f"Steps per trajectory: {dataset.metadata.n_steps_per_trajectory}")

    # constructing the graph
    pos = dataset[0]['space_grid'].reshape(-1, 2)
    num_rows, num_columns = dataset[0]['space_grid'].shape[:2]
    num_nodes = num_rows * num_columns
    edge_index, visualization_pos = grid(height=num_rows, width=num_columns)
    edge_index, _ = remove_self_loops(edge_index)
    if not is_undirected(edge_index=edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index=edge_index, num_nodes=num_nodes)
    L_sym_sparse = compute_sparse_L_sym(edge_index=edge_index,
                                        edge_weight=None,
                                        num_nodes=num_nodes)

    # GWT configs
    lambda_max = 2.0  # for L_sym, L_rw
    K = 1_000
    bandpass_lambda_min = lambda_max / K
    num_scales = 4  # Number of scales of the Mexican hat filters
    num_filters = num_scales + 1
    gwt_scales = np.logspace(np.log10(1 / lambda_max), np.log10(1 / bandpass_lambda_min), num_scales).tolist()

    # Chebyshev configs
    lambda_min = 0.0
    chebyshev_max_degree = 30
    chebyshev_cache = {}

    # iterating over the samples
    for sample_index in tqdm(iterable=indices_to_process,
                             desc=f"Processing {split} samples",
                             unit="sample"):
        # getting the sample data
        sample = dataset[sample_index]

        # computing the trajectory_index and time_index of the trajectory for this sample
        trajectory_index, time_index = get_trajectory_and_timestep_info_turbulent_radiative_layer_2D_dataset(
            sample_index=sample_index,
            metadata=dataset.metadata
        )

        # reshaping the fields data into (num_nodes, num_fields)
        x = sample['input_fields'].squeeze().reshape(num_nodes, num_fields)

        # initializing the GWT coefficients tensor
        coeffs = torch.zeros(num_nodes, num_fields, num_filters, dtype=torch.float32)

        # computing the Chebyshev approximation of GWT coefficients for each field
        chebyshev_cache_keys = set()
        for field_index in range(num_fields):
            approximate_gwt_coeffs, chebyshev_cache_key, chebyshev_cache = chebyshev_gwt_analysis(
                x=x[:, field_index],
                L_sym_sparse=L_sym_sparse,
                edge_index=edge_index,
                num_nodes=num_nodes,
                gwt_scales=gwt_scales,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                chebyshev_max_degree=chebyshev_max_degree,
                cache=chebyshev_cache,
            )
            coeffs[:, field_index, :] = approximate_gwt_coeffs
            chebyshev_cache_keys.add(chebyshev_cache_key)
        # ensuring that there is only one cache key for all fields: graph structure and GWT parameters are static
        assert len(chebyshev_cache_keys) == 1
        chebyshev_cache_key = chebyshev_cache_keys.pop()

        # constructing the PyG Data object for this sample
        # graph metadata
        graph_metadata = {
            "graph_type": "grid",
            "num_nodes": num_nodes,
            "num_edges": edge_index.size(1),
            "num_rows": num_rows,
            "num_columns": num_columns,
            "visualization_pos": visualization_pos.tolist(),
            "L_sym_sparse": L_sym_sparse,
        }
        # gwt metadata
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
            "chebyshev_cache_key": chebyshev_cache_key,
            "chebyshev_coeffs": chebyshev_cache[chebyshev_cache_key]["chebyshev_coeffs"],
            "chebyshev_eigenvalue_samples": chebyshev_cache[chebyshev_cache_key]["chebyshev_eigenvalue_samples"],
            "chebyshev_L_sym_tilde": chebyshev_cache[chebyshev_cache_key]["L_sym_tilde"],
        }
        # signal metadata
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
        data = Data(x=x,
                    coeffs=coeffs,
                    pos=pos,
                    edge_index=edge_index,
                    graph_metadata=graph_metadata,
                    gwt_metadata=gwt_metadata,
                    signal_metadata=signal_metadata)
        save_path = os.path.join(processed_dataset_dir, f'{sample_index}.pt')
        torch.save(data, save_path)

    print(f"\n✓ Completed! Saved {len(indices_to_process)} samples to {processed_dataset_dir}")


def extract_cylinder_flow_config(
        metadata: Dict
) -> Dict:
    config = {}
    # Basic metadata
    config['trajectory_length'] = metadata['trajectory_length']
    config['dt'] = metadata['dt']
    config['simulator'] = metadata['simulator']

    # Extract feature information
    features = metadata['features']

    # Static features (mesh structure)
    config['cells_dtype'] = features['cells']['dtype']
    config['cells_shape'] = features['cells']['shape']  # [1, -1, 3]
    config['cells_vertices_per_element'] = features['cells']['shape'][-1]  # 3 for triangles

    config['mesh_pos_dtype'] = features['mesh_pos']['dtype']
    config['mesh_pos_shape'] = features['mesh_pos']['shape']  # [1, -1, 2]
    config['mesh_spatial_dim'] = features['mesh_pos']['shape'][-1]  # 2 for 2D

    config['node_type_dtype'] = features['node_type']['dtype']
    config['node_type_shape'] = features['node_type']['shape']  # [1, -1, 1]

    # Dynamic features (time-varying fields)
    config['velocity_dtype'] = features['velocity']['dtype']
    config['velocity_shape'] = features['velocity']['shape']  # [600, -1, 2]
    config['velocity_components'] = features['velocity']['shape'][-1]  # 2 for (vx, vy)

    config['pressure_dtype'] = features['pressure']['dtype']
    config['pressure_shape'] = features['pressure']['shape']  # [600, -1, 1]
    config['pressure_components'] = features['pressure']['shape'][-1]  # 1 scalar field

    # Field names
    config['all_field_names'] = metadata['field_names']
    config['static_fields'] = [name for name, info in features.items() if info['type'] == 'static']
    config['dynamic_fields'] = [name for name, info in features.items() if info['type'] == 'dynamic']

    # Infer mesh representation type from cells shape
    if config['cells_vertices_per_element'] == 3:
        config['mesh_representation'] = 'triangular'
    elif config['cells_vertices_per_element'] == 4:
        config['mesh_representation'] = 'quadrilateral'
    else:
        config['mesh_representation'] = f"{config['cells_vertices_per_element']}-gon"

    # Node type descriptions (from cylinder flow physics)
    # These are standard for MeshGraphNets cylinder flow simulations
    config['node_type_descriptions'] = {
        0: 'NORMAL - interior fluid nodes (free to evolve)',
        4: 'INFLOW - inlet boundary condition',
        5: 'OUTFLOW - outlet boundary condition',
        6: 'WALL/OBSTACLE - no-slip boundary (wall + cylinder surface)'
    }

    return config


def parse_tfrecord_example(
        raw_example: bytes,
        metadata_config: Dict
) -> Dict:
    # Feature description for TFRecord parsing
    feature_description = {
        'cells': tf.io.FixedLenFeature([], tf.string),
        'mesh_pos': tf.io.FixedLenFeature([], tf.string),
        'node_type': tf.io.FixedLenFeature([], tf.string),
        'velocity': tf.io.FixedLenFeature([], tf.string),
        'pressure': tf.io.FixedLenFeature([], tf.string),
    }

    parsed = tf.io.parse_single_example(raw_example, feature_description)

    # Decode cells (int32)
    cells_raw = tf.io.decode_raw(parsed['cells'], tf.int32).numpy()
    num_cells = len(cells_raw) // metadata_config['cells_vertices_per_element']
    cells = cells_raw.reshape(num_cells, metadata_config['cells_vertices_per_element'])

    # Decode mesh_pos (float32)
    mesh_pos_raw = tf.io.decode_raw(parsed['mesh_pos'], tf.float32).numpy()
    num_nodes = len(mesh_pos_raw) // metadata_config['mesh_spatial_dim']
    mesh_pos = mesh_pos_raw.reshape(num_nodes, metadata_config['mesh_spatial_dim'])

    # Decode node_type (int32)
    node_type_raw = tf.io.decode_raw(parsed['node_type'], tf.int32).numpy()
    node_type = node_type_raw  # Already 1D

    # Decode velocity (float32)
    velocity_raw = tf.io.decode_raw(parsed['velocity'], tf.float32).numpy()
    trajectory_length = metadata_config['trajectory_length']
    velocity_components = metadata_config['velocity_components']
    velocity = velocity_raw.reshape(trajectory_length, num_nodes, velocity_components)

    # Decode pressure (float32)
    pressure_raw = tf.io.decode_raw(parsed['pressure'], tf.float32).numpy()
    pressure_components = metadata_config['pressure_components']
    if pressure_components == 1:
        pressure = pressure_raw.reshape(trajectory_length, num_nodes)
    else:
        pressure = pressure_raw.reshape(trajectory_length, num_nodes, pressure_components)

    # Convert to PyTorch tensors
    return {
        'cells': torch.from_numpy(cells).long(),
        'mesh_pos': torch.from_numpy(mesh_pos).float(),
        'node_type': torch.from_numpy(node_type).long(),
        'velocity': torch.from_numpy(velocity).float(),
        'pressure': torch.from_numpy(pressure).float(),
        'num_nodes': num_nodes,
        'num_cells': num_cells,
    }


def compute_avg_node_spacing(
        pos: torch.Tensor,
        edge_index: torch.Tensor
) -> float:
    """
    Compute average distance between connected nodes in the mesh.
    Useful for determining appropriate interpolation resolution.

    Args:
        pos: Node positions (num_nodes, spatial_dim)
        edge_index: Edge connectivity (2, num_edges)

    Returns:
        Average edge length
    """
    # Compute edge lengths
    edge_vectors = pos[edge_index[0]] - pos[edge_index[1]]
    edge_lengths = torch.norm(edge_vectors, dim=1)
    return edge_lengths.mean().item()


def compute_mesh_hash(
        cells: torch.Tensor,
        pos: torch.Tensor
) -> str:
    """
    Compute a unique hash for the mesh structure based on connectivity and positions.

    Args:
        cells: Cell connectivity tensor (num_cells, vertices_per_cell)
        pos: Node positions (num_nodes, spatial_dim)

    Returns:
        Hash string uniquely identifying this mesh structure
    """
    import hashlib

    # Combine cells and positions into a single bytes representation
    cells_bytes = cells.cpu().numpy().tobytes()
    pos_bytes = pos.cpu().numpy().tobytes()

    # Compute hash
    combined = cells_bytes + pos_bytes
    mesh_hash = hashlib.sha256(combined).hexdigest()

    return mesh_hash


def preprocess_cylinder_flow_dataset(
        dataset_name: str,
        raw_dataset_dir: str,
        processed_data_dir: str,
        split: Literal["train", "valid", "test"]
):
    # Configuring the directory for saving the processed Data objects
    processed_dataset_dir = os.path.join(processed_data_dir, dataset_name, split)
    os.makedirs(processed_dataset_dir, exist_ok=True)

    # Print processing information
    print(f"\n{'=' * 70}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Raw data path: {raw_dataset_dir}")
    print(f"Processed data will be saved to: {processed_dataset_dir}")
    print(f"{'=' * 70}\n")

    # Load meta.json
    metadata_path = os.path.join(raw_dataset_dir, "meta.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    trajectory_length = metadata.get('trajectory_length', None)
    assert trajectory_length is not None, "Trajectory length not found in meta.json"
    print(f"Trajectory length from metadata: {trajectory_length}")

    # Extract all configuration from metadata
    config = extract_cylinder_flow_config(metadata=metadata)

    print(f"Dataset Configuration:")
    print(f"  Trajectory length: {config['trajectory_length']}")
    print(f"  Time step (dt): {config['dt']}")
    print(f"  Simulator: {config['simulator']}")
    print(f"  Mesh representation: {config['mesh_representation']}")
    print(f"  Spatial dimensions: {config['mesh_spatial_dim']}D")
    print(f"  Velocity components: {config['velocity_components']}")
    print(f"  Pressure components: {config['pressure_components']}")
    print(f"  Static fields: {config['static_fields']}")
    print(f"  Dynamic fields: {config['dynamic_fields']}")

    # Load trajectories from TFRecord
    tfrecord_path = os.path.join(raw_dataset_dir, f"{split}.tfrecord")
    tf_dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    print("\nLoading trajectories from TFRecord...")
    trajectories = []
    for raw_record in tqdm(tf_dataset, desc="Loading trajectories", unit="trajectory"):
        trajectory = parse_tfrecord_example(raw_record, config)
        trajectories.append(trajectory)

    num_trajectories = len(trajectories)
    trajectory_length = config['trajectory_length']
    num_samples = num_trajectories * trajectory_length
    print(f"Total trajectories: {num_trajectories}")
    print(f"Timesteps per trajectory: {trajectory_length}")
    print(f"Total individual samples: {num_samples}")

    # Analyze mesh structure variability
    print("\nAnalyzing mesh structure across trajectories...")
    mesh_hash_to_trajectories = {}
    mesh_hash_to_shape = {}

    for traj_idx, traj in enumerate(trajectories):
        mesh_hash = compute_mesh_hash(traj['cells'], traj['mesh_pos'])

        if mesh_hash not in mesh_hash_to_trajectories:
            mesh_hash_to_trajectories[mesh_hash] = []
            mesh_hash_to_shape[mesh_hash] = (traj['num_nodes'], traj['num_cells'])

        mesh_hash_to_trajectories[mesh_hash].append(traj_idx)

    print(f"Found {len(mesh_hash_to_trajectories)} unique mesh structure(s):")
    for mesh_hash, traj_indices in mesh_hash_to_trajectories.items():
        num_nodes, num_cells = mesh_hash_to_shape[mesh_hash]
        print(f"  Mesh hash: {mesh_hash[:16]}... ({num_nodes} nodes, {num_cells} cells) - "
              f"{len(traj_indices)} trajectories")
        if len(traj_indices) <= 10:
            print(f"    Trajectory indices: {traj_indices}")
        else:
            print(f"    Trajectory indices: {traj_indices[:10]}... (and {len(traj_indices) - 10} more)")

    # Extract field metadata for dynamic fields
    field_names = ['pressure', 'velocity_x', 'velocity_y']
    num_fields = len(field_names)
    index_to_field = {idx: field for idx, field in enumerate(field_names)}
    field_to_index = {field: idx for idx, field in enumerate(field_names)}

    # GWT configs (same for all trajectories)
    lambda_max = 2.0
    K = 1_000
    bandpass_lambda_min = lambda_max / K
    num_scales = 4
    num_filters = num_scales + 1
    gwt_scales = np.logspace(np.log10(1 / lambda_max), np.log10(1 / bandpass_lambda_min), num_scales).tolist()

    # Chebyshev configs
    lambda_min = 0.0
    chebyshev_max_degree = 30

    print(f"\nGWT Configuration:")
    print(f"  Number of filters: {num_filters}")
    print(f"  Number of scales: {num_scales}")
    print(f"  Lambda max: {lambda_max}")
    print(f"  Chebyshev max degree: {chebyshev_max_degree}")

    # Cache for graph structures (keyed by mesh hash)
    graph_cache = {}

    # Iterate over all samples (trajectory, timestep pairs)
    sample_index = 0
    for trajectory_index in tqdm(range(num_trajectories),
                                 desc=f"Processing {split} trajectories",
                                 unit="trajectory"):
        trajectory = trajectories[trajectory_index]

        # Extract mesh structure for this trajectory
        pos = trajectory['mesh_pos']
        node_type = trajectory['node_type']
        cells = trajectory['cells']
        num_nodes = trajectory['num_nodes']
        num_cells = trajectory['num_cells']

        # Compute mesh hash for this trajectory
        mesh_hash = compute_mesh_hash(cells, pos)

        # Check if we've already processed this mesh structure
        if mesh_hash not in graph_cache:
            # Build graph structure for this mesh
            vertices_per_element = config['cells_vertices_per_element']
            edges_list = []
            for cell in cells:
                for i in range(vertices_per_element):
                    v_start = cell[i].item()
                    v_end = cell[(i + 1) % vertices_per_element].item()
                    edges_list.extend([[v_start, v_end], [v_end, v_start]])

            edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = torch.unique(edge_index, dim=1)

            if not is_undirected(edge_index=edge_index, num_nodes=num_nodes):
                edge_index = to_undirected(edge_index=edge_index, num_nodes=num_nodes)

            num_edges = edge_index.size(1)

            # Compute Laplacian
            L_sym_sparse = compute_sparse_L_sym(edge_index=edge_index,
                                                edge_weight=None,
                                                num_nodes=num_nodes)

            # Compute mesh statistics
            avg_spacing = compute_avg_node_spacing(pos, edge_index)
            spatial_extent = {
                "x_min": pos[:, 0].min().item(),
                "x_max": pos[:, 0].max().item(),
                "y_min": pos[:, 1].min().item(),
                "y_max": pos[:, 1].max().item(),
                "width": (pos[:, 0].max() - pos[:, 0].min()).item(),
                "height": (pos[:, 1].max() - pos[:, 1].min()).item(),
            }

            # Extract unique node types
            unique_node_types = torch.unique(node_type).tolist()
            node_type_info = {
                nt: config['node_type_descriptions'].get(nt, f'Unknown type {nt}')
                for nt in unique_node_types
            }

            # Cache this graph structure with all necessary data
            graph_cache[mesh_hash] = {
                'num_nodes': num_nodes,
                'num_cells': num_cells,
                'edge_index': edge_index,
                'num_edges': num_edges,
                'L_sym_sparse': L_sym_sparse,
                'avg_spacing': avg_spacing,
                'spatial_extent': spatial_extent,
                'node_type_info': node_type_info,
                'chebyshev_cache': {},  # Separate Chebyshev cache for each mesh
                'pos': pos.clone(),  # Store for reference
                'node_type': node_type.clone(),  # Store for reference
            }

            print(f"\n  New mesh structure discovered (hash: {mesh_hash[:16]}...):")
            print(f"    Number of nodes: {num_nodes}")
            print(f"    Number of edges: {num_edges}")
            print(f"    Number of {config['mesh_representation']} cells: {num_cells}")
            print(f"    Node types: {list(node_type_info.keys())}")
            print(f"    Spatial extent: {spatial_extent['width']:.4f} × {spatial_extent['height']:.4f}")
            print(f"    Average node spacing: {avg_spacing:.6f}")

        # Retrieve cached graph structure
        graph_data = graph_cache[mesh_hash]
        edge_index = graph_data['edge_index']
        num_edges = graph_data['num_edges']
        L_sym_sparse = graph_data['L_sym_sparse']
        avg_spacing = graph_data['avg_spacing']
        spatial_extent = graph_data['spatial_extent']
        node_type_info = graph_data['node_type_info']
        chebyshev_cache = graph_data['chebyshev_cache']

        # Verify dimensions match (safety check)
        assert num_nodes == graph_data['num_nodes'], \
            f"Node count mismatch for mesh hash {mesh_hash[:16]}: {num_nodes} vs {graph_data['num_nodes']}"

        # Process each timestep in this trajectory
        for time_index in range(trajectory_length):
            # Extract fields at this timestep
            velocity_t = trajectory['velocity'][time_index]  # (num_nodes, velocity_components)
            pressure_t = trajectory['pressure'][time_index]  # (num_nodes,) or (num_nodes, 1)

            # Ensure pressure is 1D
            if pressure_t.dim() > 1:
                pressure_t = pressure_t.squeeze(-1)

            # Reshape fields into (num_nodes, num_fields)
            x = torch.zeros(num_nodes, num_fields, dtype=torch.float32)
            x[:, 0] = pressure_t  # pressure
            x[:, 1] = velocity_t[:, 0]  # velocity_x
            x[:, 2] = velocity_t[:, 1]  # velocity_y

            # Initialize the GWT coefficients tensor
            coeffs = torch.zeros(num_nodes, num_fields, num_filters, dtype=torch.float32)

            # Compute the Chebyshev approximation of GWT coefficients for each field
            chebyshev_cache_keys = set()
            for field_index in range(num_fields):
                approximate_gwt_coeffs, chebyshev_cache_key, chebyshev_cache = chebyshev_gwt_analysis(
                    x=x[:, field_index],
                    L_sym_sparse=L_sym_sparse,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    gwt_scales=gwt_scales,
                    lambda_min=lambda_min,
                    lambda_max=lambda_max,
                    chebyshev_max_degree=chebyshev_max_degree,
                    cache=chebyshev_cache,
                )
                coeffs[:, field_index, :] = approximate_gwt_coeffs
                chebyshev_cache_keys.add(chebyshev_cache_key)

            # Update the cache in graph_data
            graph_data['chebyshev_cache'] = chebyshev_cache
            assert len(chebyshev_cache_keys) == 1
            chebyshev_cache_key = chebyshev_cache_keys.pop()

            # Construct the PyG Data object for this sample
            graph_metadata = {
                "graph_type": f"{config['mesh_representation']}_mesh",
                "mesh_representation": config['mesh_representation'],
                "spatial_dimension": config['mesh_spatial_dim'],
                "mesh_hash": mesh_hash,  # Include mesh hash for identification
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_cells": num_cells,
                "vertices_per_cell": config['cells_vertices_per_element'],
                "node_type_info": node_type_info,
                "simulator": config['simulator'],
                "L_sym_sparse": L_sym_sparse,
                "spatial_extent": spatial_extent,
                "avg_node_spacing": avg_spacing,
                "is_structured_grid": False,
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
                "chebyshev_cache_key": chebyshev_cache_key,
                "chebyshev_coeffs": chebyshev_cache[chebyshev_cache_key]["chebyshev_coeffs"],
                "chebyshev_eigenvalue_samples": chebyshev_cache[chebyshev_cache_key]["chebyshev_eigenvalue_samples"],
                "chebyshev_L_sym_tilde": chebyshev_cache[chebyshev_cache_key]["L_sym_tilde"],
            }

            signal_metadata = {
                "sample_index": sample_index,
                "trajectory_index": trajectory_index,
                "time_index": time_index,
                "dt": config['dt'],
                "physical_time": time_index * config['dt'],
                "trajectory_length": trajectory_length,
                "field_names": field_names,
                "num_fields": num_fields,
                "field_to_index": field_to_index,
                "index_to_field": index_to_field,
                "x_dims": "(num_nodes, num_fields)",
                "coeffs_dims": "(num_nodes, num_fields, num_filters)",
            }

            data = Data(
                x=x,
                coeffs=coeffs,
                pos=pos,
                edge_index=edge_index,
                node_type=node_type,
                graph_metadata=graph_metadata,
                gwt_metadata=gwt_metadata,
                signal_metadata=signal_metadata
            )

            # Save to disk
            save_path = os.path.join(processed_dataset_dir, f'{sample_index}.pt')
            torch.save(data, save_path)
            sample_index += 1

    print(f"\n{'=' * 70}")
    print(f"Processing complete!")
    print(f"Total samples saved: {sample_index}")
    print(f"Unique mesh structures processed: {len(graph_cache)}")
    for mesh_hash, graph_data in graph_cache.items():
        print(f"  Mesh {mesh_hash[:16]}...: {graph_data['num_nodes']} nodes, {graph_data['num_cells']} cells")
    print(f"Saved to: {processed_dataset_dir}")
    print(f"{'=' * 70}\n")


def main():
    # configuring the dataset and the raw data directories
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    processed_data_dir = os.path.join(local_storage_dir, 'processed_data')

    # current datasets are:
    # TheWell: "turbulent_radiative_layer_2D"
    # MeshGraphNets: "cylinder_flow"
    dataset_name = "cylinder_flow"
    raw_data_base_dir = "/Users/amirhossein/Projects/NJIT/Data"
    raw_dataset_dir = get_raw_data_dir(raw_data_base_dir=raw_data_base_dir,
                                       dataset_name=dataset_name)
    splits: list[Literal["train", "valid", "test"]] = ["test", "valid", "train"]
    if dataset_name == "turbulent_radiative_layer_2D":
        preprocess_turbulent_radiative_layer_2D_dataset(
            dataset_name=dataset_name,
            raw_dataset_dir=raw_dataset_dir,
            processed_data_dir=processed_data_dir,
            split=splits[0]
        )
    elif dataset_name == "cylinder_flow":
        preprocess_cylinder_flow_dataset(
            dataset_name=dataset_name,
            raw_dataset_dir=raw_dataset_dir,
            processed_data_dir=processed_data_dir,
            split=splits[0]
        )
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")


if __name__ == "__main__":
    main()
