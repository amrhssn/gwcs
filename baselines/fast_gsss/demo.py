import time
from typing import Tuple

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from baselines.fast_gsss.sampling import sampling_set_selection
from baselines.fast_gsss.reconstruction import reconstruction
from baselines.fast_gsss.utils import sgwt_cheby_coeff, sgwt_cheby_op2


def build_community_graph(
    num_nodes: int = 500, num_communities: int = 4, p_in: float = 0.1, p_out: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a simple stochastic block model ('community') graph.

    Returns
    -------
    W : np.ndarray, shape (N, N)
        Symmetric adjacency matrix (unweighted, 0/1).
    coords : np.ndarray, shape (N, 2)
        2D coordinates for visualization (communities placed on a circle).
    """
    sizes = [num_nodes // num_communities] * num_communities
    probs = np.full((num_communities, num_communities), p_out, dtype=float)
    np.fill_diagonal(probs, p_in)

    G = nx.stochastic_block_model(sizes, probs, seed=0)
    W = nx.to_numpy_array(G)

    # Place each community on a different point of a circle, add small noise
    coords = np.zeros((num_nodes, 2), dtype=float)
    start = 0
    radius = 1.0
    for i, size in enumerate(sizes):
        angle = 2 * np.pi * i / num_communities
        center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        coords[start:start + size] = center + 0.2 * np.random.randn(size, 2)
        start += size

    return W, coords


def build_sensor_graph(
    num_nodes: int = 500, radius: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 'sensor' graph: random points in 2D, edges between nearby points.

    Returns
    -------
    W : np.ndarray, shape (N, N)
        Symmetric adjacency matrix (0/1).
    coords : np.ndarray, shape (N, 2)
        2D coordinates of sensor locations (for plotting).
    """
    coords = np.random.rand(num_nodes, 2)
    G = nx.random_geometric_graph(num_nodes, radius, pos={i: coords[i] for i in range(num_nodes)})
    W = nx.to_numpy_array(G)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        idx = sorted(list(largest_cc))
        W = W[np.ix_(idx, idx)]
        coords = coords[idx]
    return W, coords


def demo_fastgsss(
    gname: str = "community",
    numnodes: int = 10,
    order: int = 12,
    nu: float = 75.0,
    bw: int = 100,
    device: str = "cpu",
) -> None:
    """
    Self-contained Python demo mirroring `gsp_demo_fastGSSS.m`, with extra
    visualization of ground-truth vs reconstruction.

    Steps:
    1. Build a graph (community or sensor) and compute its normalized Laplacian.
    2. Generate a bandlimited random graph signal f_or using the Laplacian eigenvectors.
    3. Run FastGSSS (Chebyshev mode) to select numnodes sampling nodes.
    4. Reconstruct the full signal from its samples.
    5. Report MSE and visualize:
        - Ground-truth signal on the graph,
        - Reconstructed signal,
        - Per-node reconstruction error,
        - Sampling locations.
    """
    torch_device = torch.device(device)

    # ----------------------------------------------------------------------
    # 1. Build graph (adjacency + coordinates)
    # ----------------------------------------------------------------------
    if gname == "community":
        W_np, coords = build_community_graph(num_nodes=100)
    elif gname == "sensor":
        W_np, coords = build_sensor_graph(num_nodes=500)
    else:
        raise ValueError(f"Unknown graph name: {gname}")

    N = W_np.shape[0]

    # ----------------------------------------------------------------------
    # 2. Laplacian and normalized Laplacian
    # ----------------------------------------------------------------------
    A = W_np
    d = A.sum(axis=1)              # (N,)
    D = np.diag(d)                 # (N, N)
    L = D - A                      # (N, N)

    with np.errstate(divide="ignore"):
        inv_sqrt_d = 1.0 / np.sqrt(d)
    inv_sqrt_d[np.isinf(inv_sqrt_d)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt_d)

    L_n_np = D_inv_sqrt @ L @ D_inv_sqrt  # (N, N)

    # ----------------------------------------------------------------------
    # 3. Eigen-decomposition and bandlimited signal generation
    # ----------------------------------------------------------------------
    E_vals, U_np = np.linalg.eigh(L_n_np)  # symmetric

    idx = np.argsort(E_vals)
    E_vals = E_vals[idx]
    U_np = U_np[:, idx]

    signs = np.sign(U_np[0, :])
    signs[signs == 0] = 1.0
    U_np = U_np * signs

    assert bw <= N, "bw must be <= N"

    coeff = np.zeros((N, 1), dtype=float)
    coeff[:bw, 0] = np.sqrt(0.2) * np.random.randn(bw)
    noise = np.sqrt(5 * 10 ** (-3)) * np.random.randn(N, 1)
    coeff_noisy = coeff + noise

    f_or_np = U_np @ coeff_noisy  # (N, 1)

    # To torch
    L_n = torch.from_numpy(L_n_np).float().to(torch_device)         # (N, N)
    f_or = torch.from_numpy(f_or_np).float().to(torch_device).squeeze(1)  # (N,)

    # ----------------------------------------------------------------------
    # 4. Sampling set selection (FastGSSS, Chebyshev mode)
    # ----------------------------------------------------------------------
    start = time.time()
    selected_nodes, T = sampling_set_selection(
        L_n=L_n,
        F=numnodes,
        bw=float(bw),
        nu=float(nu),
        eigen_decomp=False,
        m=order,
        cheby_coeff_fn=sgwt_cheby_coeff,
        cheby_op_fn=sgwt_cheby_op2,
    )
    elapsed = time.time() - start
    print(
        f"Execution time for selecting {numnodes} (out of {N}) vertices: {elapsed:.4f} seconds"
    )

    sampling_set = torch.zeros(N, dtype=torch.float32, device=torch_device)
    sampling_set[selected_nodes] = 1.0

    # ----------------------------------------------------------------------
    # 5. Reconstruction
    # ----------------------------------------------------------------------
    f_out = reconstruction(
        mem_fn=f_or,      # (N,)
        S_opt=sampling_set,
        T=T,
        k=order,
    )  # (N, 1)

    f_or_col = f_or.unsqueeze(1)  # (N, 1)
    mse = torch.mean((f_or_col - f_out) ** 2).item()
    print(f"{numnodes} vertices are selected from {N} vertices: MSE = {mse:g}")

    # ----------------------------------------------------------------------
    # 6. Visualization
    # ----------------------------------------------------------------------
    # Convert to numpy 1D arrays for plotting
    f_gt = f_or_col.squeeze(1).cpu().numpy()   # ground-truth (N,)
    f_rec = f_out.squeeze(1).detach().cpu().numpy()  # reconstruction (N,)
    err = (f_gt - f_rec)                       # error (N,)

    x, y = coords[:, 0], coords[:, 1]
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected_nodes] = True

    vmin = min(f_gt.min(), f_rec.min())
    vmax = max(f_gt.max(), f_rec.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # 1) Ground-truth signal
    sc0 = axes[0].scatter(
        x, y,
        c=f_gt,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=30,
        edgecolors="none",
    )
    axes[0].set_title("Ground-truth signal")
    axes[0].axis("equal")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Reconstructed signal (with sampling nodes marked)
    sc1 = axes[1].scatter(
        x, y,
        c=f_rec,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=30,
        edgecolors="none",
    )
    # overlay sampled nodes as black-edged circles
    axes[1].scatter(
        x[selected_mask],
        y[selected_mask],
        facecolors="none",
        edgecolors="red",
        s=80,
        linewidths=1.5,
        label="Sampled nodes",
    )
    axes[1].set_title("Reconstructed signal\n(red circles = sampled)")
    axes[1].axis("equal")
    axes[1].legend(loc="best")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Pointwise reconstruction error
    sc2 = axes[2].scatter(
        x, y,
        c=err,
        cmap="coolwarm",
        s=30,
        edgecolors="none",
    )
    axes[2].set_title("Reconstruction error (f_gt - f_rec)")
    axes[2].axis("equal")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"FastGSSS demo on '{gname}' graph\nMSE = {mse:.3e}, sampled nodes = {numnodes}",
        fontsize=12,
    )
    plt.show()


if __name__ == "__main__":
    demo_fastgsss(gname="sensor", numnodes=10, device="cpu")
