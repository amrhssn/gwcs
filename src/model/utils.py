from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
  GCNConv,
  SAGEConv,
  GATConv,
  GINConv
)
from torch_geometric.nn.conv import MessagePassing


_NAME_TO_FUNCTIONAL = {
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
_NAME_TO_MODULE = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'softmax': nn.Softmax,
    'softplus': nn.Softplus,
    'softsign': nn.Softsign,
    'hardtanh': nn.Hardtanh,
    'relu6': nn.ReLU6,
    'silu': nn.SiLU,
    'swish': nn.SiLU,  # silu and swish are the same
    'mish': nn.Mish,
    'hardswish': nn.Hardswish,
    'logsigmoid': nn.LogSigmoid,
    'celu': nn.CELU,
    'glu': nn.GLU,
}
_GCONV_TYPES = ('GCNConv', 'SAGEConv', 'GATConv', 'GINConv')


def get_activation_fn(name: str) -> Callable:
    name = name.lower()
    if name not in _NAME_TO_FUNCTIONAL:
        raise ValueError(
            f"Activation function '{name}' not recognized. "
            f"Available options: {list(_NAME_TO_FUNCTIONAL.keys())}"
        )
    return _NAME_TO_FUNCTIONAL[name]


def get_activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name not in _NAME_TO_MODULE:
        raise ValueError(
            f"Activation function '{name}' not recognized. "
            f"Available options: {list(_NAME_TO_MODULE.keys())}"
        )
    return _NAME_TO_MODULE[name]()


def get_gconv_layer(in_channels: int, out_channels: int, conv_layer_type: str) -> MessagePassing:
    if conv_layer_type == 'GCNConv':
        return GCNConv(in_channels=in_channels, out_channels=out_channels, improved=True)
    elif conv_layer_type == 'SAGEConv':
        return SAGEConv(in_channels=in_channels, out_channels=out_channels, normalize=False)
    elif conv_layer_type == 'GATConv':
        if out_channels % 4 == 0:
            return GATConv(in_channels=in_channels, out_channels=out_channels // 4, heads=4)
        return GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
    elif conv_layer_type == 'GINConv':
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        return GINConv(nn=mlp, eps=0.0, train_eps=False)
    else:
        raise ValueError(
            f"Graph conv layer '{conv_layer_type}' not recognized. "
            f"Available options: {list(_GCONV_TYPES)}"
        )
