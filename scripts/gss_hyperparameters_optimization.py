import os
import itertools
import warnings
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from paths import ROOT_PROJECT_PATH
from data.dataset import (
    GraphSignalDataset,
    create_splits_from_indices
)
from data.utils_io import load_config, load_json
from baselines.graph_signal_sampling_utils import (
    get_hparams_dict
)
from baselines.graph_signal_sampling import (
    build_graph_from_edge_index,
    sampling_and_reconstruction
)

# Suppress the specific warning from thgsp
warnings.filterwarnings('ignore', message='.*Exited postprocessing with accuracies.*')


def search_hyperparameters(
        method_name: str,
        val_dataset,
        params: Dict,
        keep_ratio: float,
) -> Tuple[Dict, pd.DataFrame]:
    num_filters = 5
    # Generate all combinations of parameters
    params_names = list(params.keys())
    params_values = list(params.values())
    params_combinations = list(itertools.product(*params_values))

    results = []
    best_error = float('inf')
    best_params = None

    print(f"Testing {len(params_combinations)} parameter combinations...")
    print(f"Validation set size: {len(val_dataset)}")

    # Loop over all parameter combinations
    for param_combo in tqdm(params_combinations, desc="Parameter search"):
        # Create parameter dictionary
        params = dict(zip(params_names, param_combo))

        # Track errors for this parameter combination
        val_errors = []

        # Evaluate on validation set
        for i, data in enumerate(val_dataset):
            try:
                num_nodes = data.num_nodes
                edge_index = data.edge_index
                x = data.x
                x_clean = torch.tensor(data['signal_metadata']['x_clean'], dtype=torch.float32)
                bandwidth = data.signal_metadata['R']
                sample_size = min(int(num_nodes * num_filters * keep_ratio), num_nodes)

                # Build graph
                thgsp_graph, thgsp_adj = build_graph_from_edge_index(edge_index=edge_index,
                                                                     num_nodes=num_nodes,
                                                                     x=x)
                thgsp_L_sym = thgsp_graph.L("sym")
                thgsp_L_comb = thgsp_graph.L("comb")
                thgsp_eigenvalues, thgsp_eigenvectors = thgsp_graph.spectral(lap_type="sym")
                args = {
                    'x_clean': x_clean,
                    'thgsp_graph': thgsp_graph,
                    'thgsp_adj': thgsp_adj,
                    'thgsp_L_sym': thgsp_L_sym,
                    'thgsp_L_comb': thgsp_L_comb,
                    'thgsp_eigenvalues': thgsp_eigenvalues,
                    'thgsp_eigenvectors': thgsp_eigenvectors,
                    'signal_bandwidth': bandwidth,
                    'sample_size': sample_size,
                    'num_nodes': num_nodes,
                }
                # Sample and reconstruct with current parameters
                S, x_hat = sampling_and_reconstruction(
                    method_name=method_name,
                    x=x,
                    args=args,
                    search_params=params,
                )
                # Calculate error
                mse = ((x_hat.squeeze() - x.squeeze()) ** 2).mean().item()
                val_errors.append(mse)

            except Exception as e:
                raise e

        # Calculate mean validation error
        mean_val_error = np.mean(val_errors)
        std_val_error = np.std(val_errors)

        # Store results
        result = {
            **params,
            'mean_val_error': mean_val_error,
            'std_val_error': std_val_error,
        }
        results.append(result)

        # Update best parameters
        if mean_val_error < best_error:
            best_error = mean_val_error
            best_params = params.copy()
            print(f"\n✓ New best: {best_params} -> error: {best_error:.6f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_val_error')

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation error: {best_error:.6f}\n")

    print("Top 5 parameter combinations:")
    print(results_df.head())

    return best_params, results_df


def main():
    # load the data for optimization
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    run_dir = os.path.join(local_storage_dir, 'runs', '2025-10-31_21-42-33')
    pretraining_config = load_config(config_dir=run_dir,
                                     config_name='reconstructor_pretraining_config.json')
    base_data_dir = pretraining_config.base_data_dir
    split_indices = load_json(filename='split_indices.json', save_dir=run_dir)
    dataset = GraphSignalDataset(base_data_dir=base_data_dir)
    train_dataset, val_dataset, test_dataset = create_splits_from_indices(dataset=dataset,
                                                                          train_indices=split_indices['train'],
                                                                          val_indices=split_indices['val'],
                                                                          test_indices=split_indices['test'])
    hparam_optim_runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    keep_ratio = 0.01
    for method_name in ['ess', 'bsgda', 'fastgsss', 'rsbs']:
        # if method_name in ['ess', 'bsgda', 'fastgsss']:
        #     continue
        print(f" ****************** Method: {method_name} ****************** ")
        # configure artifacts directory
        artifacts_dir = os.path.join(local_storage_dir, 'gsss_hparams', hparam_optim_runtime, method_name)
        os.makedirs(artifacts_dir, exist_ok=True)

        # determine the method and its hyperparameters grid
        params = get_hparams_dict(method_name=method_name)

        # run the hparam optimization
        best_params, val_results = search_hyperparameters(
            method_name=method_name,
            val_dataset=val_dataset,
            params=params,
            keep_ratio=keep_ratio,
        )

        # Step 3: Save results
        val_results.to_csv(os.path.join(artifacts_dir, 'val_results.csv'), index=False)
        print(f"Best parameters: {best_params}")
        with open(os.path.join(artifacts_dir, 'best_params.txt'), 'w') as f:
            f.write(str(best_params))
        print("✓ Results saved to:")
        print("  - ess_validation_results.csv")
        print("  - ess_test_results.csv")
        print("  - ess_best_params.txt")


if __name__ == '__main__':
    main()
