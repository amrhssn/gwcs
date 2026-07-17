from typing import Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
from torch_geometric.data import Data


def monte_carlo_averaging_reconstruction(
        data: Data,
        model: Union[Callable, nn.Module],
        k: int,
) -> torch.Tensor:
    model.eval()
    reconstructions = []
    for _ in range(k):
        # calling the Sparse GAE model
        with torch.no_grad():
            z_sparse = model.encode(data.x, data.edge_index)
            x_hat = model.decode(z_sparse, data.edge_index)

        reconstructions.append(x_hat)
    reconstructions = torch.stack(reconstructions, dim=0)
    x_hat_mc = reconstructions.mean(dim=0)
    return x_hat_mc

