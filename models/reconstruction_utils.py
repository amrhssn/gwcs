from typing import Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models.sparsification_utils import (
    weighted_random_per_scale,
    random_mask
)
from graph_wavelet_transform.inverse import igwt


def monte_carlo_averaging_reconstruction(
        data: Data,
        model: Union[Callable, nn.Module],
        igwt_mode: str,
        keep_ratio: float,
        k: int,
        mode: str = "random",
        mask_model: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert mode in ["weighted_random_per_scale", "random", "optimized"]
    model.eval()
    masks = []
    reconstructions = []
    for _ in range(k):
        # sampling
        if mode == "weighted_random_per_scale":
            sparsification_mask = weighted_random_per_scale(
                coeffs=data.coeffs,
                keep_ratio=keep_ratio
            )
        elif mode == "random":
            sparsification_mask = random_mask(
                coeffs=data.coeffs,
                keep_ratio=keep_ratio
            )
        elif mode == "optimized":
            if mask_model is None:
                raise ValueError("mask_model must be provided in optimized mode")
            sparsification_mask = mask_model.sample_mask_topk_gumbel()
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        # reconstruction
        if igwt_mode == "model":
            with torch.no_grad():
                x_hat = model(coeffs=data.coeffs,
                              sparsification_mask=sparsification_mask,
                              edge_index=data.edge_index,
                              gwt_matrices=[data.G])
        elif igwt_mode == "pseudo-inverse":
            sparse_coeffs = data.coeffs * sparsification_mask
            x_hat = igwt(coeffs=sparse_coeffs, G=data.G)
        else:
            raise ValueError(f"Unsupported igwt mode: {igwt_mode}")
        masks.append(sparsification_mask)
        reconstructions.append(x_hat)
    masks = torch.stack(masks, dim=0)
    reconstructions = torch.stack(reconstructions, dim=0)
    x_hat_mc = reconstructions.mean(dim=0)
    return x_hat_mc, masks

