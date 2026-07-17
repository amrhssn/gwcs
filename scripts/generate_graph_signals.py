import os
from datetime import datetime

import numpy as np

from paths import ROOT_PROJECT_PATH
from utils.random import set_random_seed
from data.data_types import (
    GraphGenerationConfig,
    SignalGenerationConfig,
    GraphWaveletTransformConfig
)
from processing_steps.data_generation import (
    data_generation_step
)

set_random_seed(random_seed=13)


def main():
    # =================== configs ===================
    # general configs
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_data_dir = os.path.join(local_storage_dir, 'data', run_timestamp)
    os.makedirs(base_data_dir, exist_ok=True)

    # graph generation configs
    # graph_types = ["grid"]
    graph_types = ["grid", "erdos_renyi", "barabasi_albert", "sbm"]
    num_nodes_range = (200, 2_000)
    num_graphs_per_type = 250
    barabasi_albert_min_num_edges_per_new_node_range = (2, 6)
    erdos_renyi_edge_prob_range = (0.002, 0.01)
    sbm_num_blocks_range = (3, 6)
    sbm_intra_cluster_prob_range = (0.02, 0.15)
    sbm_inter_cluster_prob_range = (0.001, 0.01)
    graph_config = GraphGenerationConfig(
        graph_types=graph_types,
        num_nodes_range=num_nodes_range,
        num_graphs_per_type=num_graphs_per_type,
        barabasi_albert_min_num_edges_per_new_node_range=barabasi_albert_min_num_edges_per_new_node_range,
        erdos_renyi_edge_prob_range=erdos_renyi_edge_prob_range,
        sbm_num_blocks_range=sbm_num_blocks_range,
        sbm_intra_cluster_prob_range=sbm_intra_cluster_prob_range,
        sbm_inter_cluster_prob_range=sbm_inter_cluster_prob_range,
    )

    # signal generation configs
    # signal_types = ["lowpass", "bandpass", "highpass", "R-sparse"]
    signal_types = ["lowpass"]
    R_range_max = 0.2
    coeffs_mean = 0.0
    coeffs_std_range = (1.0, 2.0)
    noise_level_range = (0.05, 0.2)
    signal_config = SignalGenerationConfig(
        signal_types=signal_types,
        R_range_max=R_range_max,
        coeffs_mean=coeffs_mean,
        coeffs_std_range=coeffs_std_range,
        noise_level_range=noise_level_range
    )

    # graph wavelet transform configs
    lambda_max = 2.0  # for L_sym, L_rw
    K = 50
    lambda_min = lambda_max / K
    num_scales = 4  # Number of scales of the Mexican hat filters
    num_filters = num_scales + 1
    scales = np.logspace(np.log10(1 / lambda_max), np.log10(1 / lambda_min), num_scales).tolist()
    gwt_config = GraphWaveletTransformConfig(lambda_max=lambda_max,
                                             K=K,
                                             lambda_min=lambda_min,
                                             num_scales=num_scales,
                                             num_filters=num_filters,
                                             scales=scales)

    # =================== data generation ===================
    data_generation_step(graph_config=graph_config,
                         signal_config=signal_config,
                         gwt_config=gwt_config,
                         base_data_dir=base_data_dir)


if __name__ == "__main__":
    main()
