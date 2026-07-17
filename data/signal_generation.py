import random
from typing import Tuple, List, Dict, Any, Optional

import numpy as np

from data.data_types import SignalGenerationConfig


def generate_bandlimited_signal(
        eigenvectors: np.ndarray,
        signal_type: str,
        R: int,
        coeff_mean: float,
        coeff_std: float,
        add_noise: bool,
        noise_level: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Convert eigenvectors to numpy array
    U = np.array(eigenvectors)
    N = U.shape[0]  # Number of nodes

    # Validate R
    if R > N:
        raise ValueError(f"R ({R}) cannot be greater than number of nodes ({N})")

    # Select frequency indices based on signal type
    if signal_type == 'lowpass':
        # First R frequencies (low pass)
        selected_indices = np.arange(R)

    elif signal_type == 'highpass':
        # Last R frequencies (high pass)
        selected_indices = np.arange(N - R, N)

    elif signal_type == 'bandpass':
        # Default: use middle R frequencies
        start_idx = (N - R) // 2
        selected_indices = np.arange(start_idx, start_idx + R)

    elif signal_type == 'R-sparse':
        # Randomly select R frequency indices without replacement
        selected_indices = np.random.choice(N, size=R, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort for consistency
    else:
        raise ValueError(
            f"Invalid signal_type: {signal_type}. Must be one of 'lowpass', 'highpass', 'bandpass', 'R-sparse'")

    # Validate selected indices
    if np.any(selected_indices < 0) or np.any(selected_indices >= N):
        raise ValueError(f"Frequency indices must be in range [0, {N - 1}]")

    # Extract U_R (selected columns of U)
    U_R = U[:, selected_indices]

    # Generate frequency coefficients vector a
    coeff = np.random.normal(coeff_mean, coeff_std, size=R)

    # Generate bandlimited signal: x = U_R @ a
    x_clean = U_R @ coeff

    # Add noise if specified
    if add_noise:
        if noise_level is None:
            raise ValueError("noise_std must be provided when add_noise=True")
        noise_std = coeff_std * noise_level
        n = np.random.normal(0, noise_std, size=N)
        # Add noise: x = U_R @ a + n
        x = x_clean + n
    else:
        noise_std = None
        x = x_clean
    x = np.expand_dims(x, axis=-1).tolist()
    x_clean = np.expand_dims(x_clean, axis=-1)
    metadata = {
        'signal_type': signal_type,
        'R': R,
        'N': N,
        'selected_indices': selected_indices.tolist(),
        'coeff_mean': coeff_mean,
        'coeff_std': coeff_std,
        'coeff': coeff.tolist(),
        'U_R': U_R.tolist(),
        'add_noise': add_noise,
        'noise_std': noise_std if add_noise else None,
        'x_clean': x_clean.tolist(),
    }
    return x, metadata


def single_random_signal_generation_step(
        graph: Dict[str, Any],
        signal_config: SignalGenerationConfig
):
    num_nodes = graph['metadata']["num_nodes"]
    eigenvectors = np.array(graph["eigenvectors"])

    # generate random signal
    signal_type = random.choice(signal_config.signal_types)
    R = random.randint(3, int(num_nodes * signal_config.R_range_max))
    coeff_mean = signal_config.coeffs_mean
    coeff_std = random.uniform(signal_config.coeffs_std_range[0],
                               signal_config.coeffs_std_range[1])
    noise_level = random.uniform(signal_config.noise_level_range[0],
                                 signal_config.noise_level_range[1])
    x, signal_metadata = generate_bandlimited_signal(eigenvectors=eigenvectors,
                                                     signal_type=signal_type,
                                                     R=R,
                                                     coeff_mean=coeff_mean,
                                                     coeff_std=coeff_std,
                                                     add_noise=True,
                                                     noise_level=noise_level)
    graph_signal = {
        "graph": graph,
        "signal": {
            "x": x,
            "metadata": signal_metadata,
        }
    }
    return graph_signal


def random_signal_generation_step(
        graph_list: List[Dict[str, Any]],
        signal_config: SignalGenerationConfig
) -> List[Dict[str, Any]]:
    graph_signal_list = []
    for idx, graph in enumerate(graph_list):
        graph_signal = single_random_signal_generation_step(graph=graph,
                                                            signal_config=signal_config)
        graph_signal_list.append(graph_signal)
    return graph_signal_list
