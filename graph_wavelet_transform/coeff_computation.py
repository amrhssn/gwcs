from typing import List, Dict, Any

import torch

from data.data_types import GraphWaveletTransformConfig
from graph_wavelet_transform.filterbank import construct_sgwt_filterbank
from graph_wavelet_transform.forward import gwt
from graph_wavelet_transform.utils import reshape_coeffs_to_2d


def single_gwt_coefficient_computation_step(
        graph_signal: Dict[str, Any],
        gwt_config: GraphWaveletTransformConfig,
):
    # print("Computing gwt coefficient for graph {}...".format(idx))
    num_nodes = graph_signal['graph']['metadata']['num_nodes']
    eigenvalues = torch.tensor(graph_signal['graph']['eigenvalues'], dtype=torch.float32)
    eigenvectors = torch.tensor(graph_signal['graph']['eigenvectors'], dtype=torch.float32)
    x = torch.tensor(graph_signal['signal']['x'], dtype=torch.float32)
    G = construct_sgwt_filterbank(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        scaling_function_minimum_eigenvalue=gwt_config.lambda_min,
        mexican_hat_filter_scales=gwt_config.scales,
    )
    coeffs = gwt(x=x, G=G)

    # Reshape coefficients to (1+J, num_nodes) and transposing to get (num_nodes, 1+J)
    input_coeffs = reshape_coeffs_to_2d(coeffs=coeffs,
                                        num_nodes=num_nodes,
                                        num_filters=gwt_config.num_filters)

    gwt_augmented_graph_signal = {
        'graph': graph_signal['graph'],
        'signal': graph_signal['signal'],
        'gwt': {
            'G': G.tolist(),
            'coeffs': input_coeffs.tolist()
        }
    }
    return gwt_augmented_graph_signal


def gwt_coefficient_computation_step(
        graph_signal_list: List[Dict[str, Any]],
        gwt_config: GraphWaveletTransformConfig,
) -> List[Dict[str, Any]]:
    # print("Computing gwt coefficients...")
    gwt_augmented_graph_signal_list = []
    for idx, graph_signal in enumerate(graph_signal_list):
        gwt_augmented_graph_signal = single_gwt_coefficient_computation_step(graph_signal=graph_signal,
                                                                             gwt_config=gwt_config)
        gwt_augmented_graph_signal_list.append(gwt_augmented_graph_signal)
    return gwt_augmented_graph_signal_list
