import torch.nn as nn
import torch.nn.functional as F


def get_activation(
        activation_name: str,
        return_module: bool = False
):
    """
    Get activation function from torch.nn.functional or torch.nn based on string name.

    Args:
        activation_name: Name of the activation function (case-insensitive)
        return_module: If True, return nn.Module class; if False, return functional version

    Returns:
        Activation function from torch.nn.functional or torch.nn.Module class

    Raises:
        ValueError: If activation function name is not recognized
    """
    activation_name = activation_name.lower()

    # Functional activation map
    activation_map = {
        'relu': F.relu,
        'leaky_relu': F.leaky_relu,
        'prelu': F.prelu,
        'elu': F.elu,
        'selu': F.selu,
        'gelu': F.gelu,
        'tanh': F.tanh,
        'sigmoid': F.sigmoid,
        'softmax': F.softmax,
        'softplus': F.softplus,
        'softsign': F.softsign,
        'hardtanh': F.hardtanh,
        'relu6': F.relu6,
        'silu': F.silu,
        'swish': F.silu,  # silu and swish are the same
        'mish': F.mish,
        'hardswish': F.hardswish,
        'logsigmoid': F.logsigmoid,
        'celu': F.celu,
        'glu': F.glu,
    }

    # Module activation map
    module_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'prelu': nn.PReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(),
        'softplus': nn.Softplus(),
        'softsign': nn.Softsign(),
        'hardtanh': nn.Hardtanh(),
        'relu6': nn.ReLU6(),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),  # silu and swish are the same
        'mish': nn.Mish(),
        'hardswish': nn.Hardswish(),
        'logsigmoid': nn.LogSigmoid(),
        'celu': nn.CELU(),
        'glu': nn.GLU(),
    }

    if activation_name not in activation_map:
        raise ValueError(
            f"Activation function '{activation_name}' not recognized. "
            f"Available options: {list(activation_map.keys())}"
        )

    if return_module:
        return module_map[activation_name]
    else:
        return activation_map[activation_name]
