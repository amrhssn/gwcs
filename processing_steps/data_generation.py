from data.data_types import (
    GraphGenerationConfig,
    SignalGenerationConfig,
    GraphWaveletTransformConfig
)
from data.graph_generation import generate_single_random_graph
from data.signal_generation import single_random_signal_generation_step
from graph_wavelet_transform.coeff_computation import single_gwt_coefficient_computation_step
from data.utils_io import (
    save_graph_signal_as_torch,
    save_config
)


def data_generation_step(
        graph_config: GraphGenerationConfig,
        signal_config: SignalGenerationConfig,
        gwt_config: GraphWaveletTransformConfig,
        base_data_dir: str,
) -> None:
    print(f"[*] Starting single-graph data generation process...")
    save_config(config=graph_config,
                config_name="graph_config",
                save_dir=base_data_dir)
    save_config(config=signal_config,
                config_name="signal_config",
                save_dir=base_data_dir)
    save_config(config=gwt_config,
                config_name="gwt_config",
                save_dir=base_data_dir)

    sample_index = 0
    for graph_type in graph_config.graph_types:
        for graph_idx in range(graph_config.num_graphs_per_type):
            if sample_index % 100 == 0:
                print(f"[*] Generating sample {sample_index} ...")
            graph = generate_single_random_graph(
                graph_type=graph_type,
                num_nodes_range=graph_config.num_nodes_range,
                erdos_renyi_edge_prob_range=graph_config.erdos_renyi_edge_prob_range,
                barabasi_albert_min_num_edges_per_new_node_range=graph_config.barabasi_albert_min_num_edges_per_new_node_range,
                sbm_num_blocks_range=graph_config.sbm_num_blocks_range,
                sbm_intra_cluster_prob_range=graph_config.sbm_intra_cluster_prob_range,
                sbm_inter_cluster_prob_range=graph_config.sbm_inter_cluster_prob_range
            )
            graph_signal = single_random_signal_generation_step(graph=graph,
                                                                signal_config=signal_config)
            gwt_augmented_graph_signal = single_gwt_coefficient_computation_step(graph_signal=graph_signal,
                                                                                 gwt_config=gwt_config)
            save_graph_signal_as_torch(graph_signal=gwt_augmented_graph_signal,
                                       sample_index=sample_index,
                                       base_data_dir=base_data_dir)
            sample_index += 1
