import os
import json
from dataclasses import asdict
from typing import Dict, Any, Union

import torch

from data.data_types import (
    GraphGenerationConfig,
    SignalGenerationConfig,
    GraphWaveletTransformConfig,
    ReconstructorPretrainingConfig,
    SparseGAEConfig
)
from data.data_types_turbulent import (
    ReconstructorPretrainingConfigTurbulent,
    GraphGenerationConfigTurbulent,
    SignalGenerationConfigTurbulent,
    GraphWaveletTransformConfigTurbulent,
)
from data.data_types_turbulent import ReconstructorPretrainingConfigTurbulent


def save_graph_signal_as_torch(
        graph_signal: Dict[str, Any],
        sample_index: int,
        base_data_dir: str
):
    graph_signal_dir = os.path.join(base_data_dir, 'raw')
    os.makedirs(graph_signal_dir, exist_ok=True)

    graph = graph_signal['graph']
    signal = graph_signal['signal']
    gwt = graph_signal['gwt']

    # graph
    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
    L_sym = torch.tensor(graph['L_sym'], dtype=torch.float32)
    eigenvalues = torch.tensor(graph['eigenvalues'], dtype=torch.float32)
    eigenvectors = torch.tensor(graph['eigenvectors'], dtype=torch.float32)

    # signal
    x = torch.tensor(signal['x'], dtype=torch.float32)

    # gwt
    G = torch.tensor(gwt['G'], dtype=torch.float32)
    coeffs = torch.tensor(gwt['coeffs'], dtype=torch.float32)
    data_dict = {
        # Graph tensors
        'edge_index': edge_index,
        'L_sym': L_sym,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,

        # Signal tensor
        'x': x,

        # GWT tensors
        'G': G,
        'coeffs': coeffs,

        # Metadata (keep as native Python types)
        'graph_metadata': graph['metadata'],  # Dict with graph type info
        'signal_metadata': signal['metadata'],  # Signal metadata
    }
    data_path = os.path.join(graph_signal_dir, f"{sample_index}.pt")
    torch.save(data_dict, data_path)


def save_config(
        config: Union[
            GraphGenerationConfig,
            SignalGenerationConfig,
            GraphWaveletTransformConfig,
            ReconstructorPretrainingConfig,
            SparseGAEConfig,
            ReconstructorPretrainingConfigTurbulent
        ],
        config_name: str,
        save_dir: str
):
    config_path = os.path.join(save_dir, f'{config_name}.json')
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"[*] {config_name} saved to {config_path}")


def load_config(
        config_dir: str,
        config_name: str
) -> Union[
    GraphGenerationConfig,
    SignalGenerationConfig,
    GraphWaveletTransformConfig,
    ReconstructorPretrainingConfig,
    SparseGAEConfig,
    GraphGenerationConfigTurbulent,
    SignalGenerationConfigTurbulent,
    GraphWaveletTransformConfigTurbulent,
    ReconstructorPretrainingConfigTurbulent,
]:
    filename = os.path.join(config_dir, config_name)
    with open(filename, "r") as f:
        data = json.load(f)

    if config_name == "graph_config.json":
        # Mesh datasets (turbulent, kolmogorov, dynamic_stall, viscoelastic) write a
        # fixed 'num_nodes' field.  Synthetic datasets use 'num_nodes_range' instead.
        if 'num_nodes' in data:
            return GraphGenerationConfigTurbulent(**data)
        return GraphGenerationConfig(**data)

    if config_name == "signal_config.json":
        # Mesh datasets write 'num_fields' + 'field_names'.
        # Synthetic datasets write 'signal_types' + noise/coefficient ranges.
        if 'num_fields' in data:
            return SignalGenerationConfigTurbulent(**data)
        return SignalGenerationConfig(**data)

    if config_name == "gwt_config.json":
        # Mesh datasets use a Chebyshev-based GWT and always write 'chebyshev_max_degree'.
        # Synthetic datasets use a polynomial GWT and write 'scales' instead.
        if 'chebyshev_max_degree' in data:
            return GraphWaveletTransformConfigTurbulent(**data)
        return GraphWaveletTransformConfig(**data)

    if config_name == "reconstructor_pretraining_config.json":
        return ReconstructorPretrainingConfig(**data)
    if config_name == "reconstructor_pretraining_config_turbulent.json":
        return ReconstructorPretrainingConfigTurbulent(**data)
    if config_name == "sparse_gae_pretraining_config.json":
        return SparseGAEConfig(**data)

    raise ValueError(f"Unrecognized config file name: {filename}")



def save_json(
        data: Dict[str, Any],
        filename: str,
        save_dir: str
) -> None:
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[*] {filename} saved to {file_path}")


def load_json(
        filename: str,
        save_dir: str
) -> Dict[str, Any]:
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_best_checkpoint(
        checkpoints_dir: str,
) -> Dict[str, Any]:
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No .pt checkpoints found in: {checkpoints_dir}")
    last = max(files, key=lambda f: int(os.path.splitext(f)[0]))
    last_checkpoint = torch.load(os.path.join(checkpoints_dir, last),
                                 map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return last_checkpoint
