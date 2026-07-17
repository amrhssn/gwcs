import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv
)
from torch_geometric.nn.conv import MessagePassing


def get_gconv_layer(
        in_channels: int,
        out_channels: int,
        conv_layer_type: str,
) -> MessagePassing:
    if conv_layer_type == "GCNConv":
        conv = GCNConv(in_channels=in_channels,
                       out_channels=out_channels,
                       improved=True)
    elif conv_layer_type == "SAGEConv":
        conv = SAGEConv(in_channels=in_channels,
                        out_channels=out_channels,
                        normalize=False)
    elif conv_layer_type == "GATConv":
        if out_channels // 4 != 0:
            conv = GATConv(in_channels=in_channels,
                           out_channels=out_channels // 4,
                           heads=4)
        else:
            conv = GATConv(in_channels=in_channels,
                           out_channels=out_channels,
                           heads=1)

    elif conv_layer_type == "GINConv":
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        conv = GINConv(nn=mlp, eps=0.0, train_eps=False)
    else:
        raise NotImplementedError
    return conv
