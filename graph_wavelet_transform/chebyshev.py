import math
import hashlib
from typing import Tuple, List, Dict, Callable, Optional, Any

import torch

from data.utils_graph import compute_sparse_L_sym
from graph_wavelet_transform.filterbank import (
    scaling_kernel,
    mexican_hat_kernel
)
from graph_wavelet_transform.utils import reshape_coeffs_to_2d


def compute_cache_key(
        edge_index: Optional[torch.Tensor],
        num_nodes: int,
        kernel_id: str,
        lambda_min: float,
        lambda_max: float,
        chebyshev_max_degree: int,
) -> str:
    """
    Compute a unique cache key based on graph structure and filter parameters.

    The key is constructed from a hash of the edge index (if provided) combined
    with the filter configuration parameters. This allows caching of computed
    Chebyshev coefficients and transformed Laplacians across calls.

    Parameters
    ----------
    edge_index :
        Edge index tensor of shape (2, E) where E is number of edges, or None.
    num_nodes :
        Number of nodes in the graph.
    kernel_id :
        User-defined identifier for the filter bank configuration. Used to
        distinguish cache entries when different spectral kernels are applied
        to the same graph. Change this value when using different wavelet
        types or filter configurations (default: "default").
    lambda_min :
        Minimum eigenvalue of the Laplacian spectrum.
    lambda_max :
        Maximum eigenvalue of the Laplacian spectrum.
    chebyshev_max_degree :
        Maximum degree K of Chebyshev polynomials.

    Returns
    -------
    key :
        A unique string key for caching purposes.
    """
    param_str = f"{num_nodes}|{kernel_id}|{lambda_min:.12g}|{lambda_max:.12g}|{chebyshev_max_degree}"
    if edge_index is not None:
        ei_cpu = edge_index.detach().cpu()
        try:
            bytes_data = ei_cpu.numpy().tobytes()
        except Exception:
            bytes_data = str(ei_cpu.tolist()).encode()
        digest = hashlib.sha1(bytes_data).hexdigest()[:12]
        key = f"{digest}|{param_str}"
    else:
        key = f"noedge|{param_str}"
    return key


def compute_tilde_L(
        L_sym_sparse: torch.Tensor,
        lambda_min: float,
        lambda_max: float,
        cache: Optional[Dict[str, Any]],
        cache_key: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute or fetch transformed ˜L = (L - bI)/a from cache.
    Returns (L_sym_tilde, new_cache) without mutating input cache.
    """
    new_cache = dict(cache) if cache is not None else {}

    if cache is not None and cache_key in cache and "L_sym_tilde" in cache[cache_key]:
        return cache[cache_key]["L_sym_tilde"], new_cache

    device = L_sym_sparse.device
    dtype = L_sym_sparse.dtype
    N = L_sym_sparse.shape[0]
    I_sparse = torch.eye(N, device=device, dtype=dtype).to_sparse()
    a = (lambda_max - lambda_min) / 2.0
    b = (lambda_max + lambda_min) / 2.0
    L_sym_tilde = (L_sym_sparse - b * I_sparse) * (1.0 / a)

    new_cache = dict(new_cache)
    new_cache.setdefault(cache_key, {})
    new_cache[cache_key]["L_sym_tilde"] = L_sym_tilde

    return L_sym_tilde, new_cache


def compute_chebyshev_coefficients(
        spectral_kernels: List[Callable[[torch.Tensor], torch.Tensor]],
        chebyshev_max_degree: int,
        lambda_min: float,
        lambda_max: float,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        cache: Optional[Dict[str, Any]],
        cache_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute or fetch Chebyshev polynomial coefficients for spectral filter kernels.

    Uses the Chebyshev-Gauss-Lobatto quadrature to approximate the spectral kernels
    as Chebyshev polynomial expansions. Results are cached to avoid recomputation.

    Parameters
    ----------
    spectral_kernels :
        List of F spectral filter functions, each mapping eigenvalues (tensor) to
        filter responses (tensor).
    chebyshev_max_degree :
        Maximum degree K of Chebyshev polynomials.
    lambda_min :
        Minimum eigenvalue of the Laplacian spectrum.
    lambda_max :
        Maximum eigenvalue of the Laplacian spectrum.
    device :
        Torch device for computation. Defaults to CPU if None.
    dtype :
        Torch dtype for computation. Defaults to float32 if None.
    cache :
        Optional cache dictionary to check for precomputed results.
    cache_key :
        Key to use for storing/retrieving cached results.

    Returns
    -------
    chebyshev_coeffs :
        Chebyshev coefficients tensor of shape (F, K+1) where F is number of
        filters and K = chebyshev_max_degree.
    chebyshev_eigenvalue_samples :
        Eigenvalue sample points used for coefficient computation, shape (K+1,).
    new_cache :
        Updated cache dictionary (does not mutate input cache).
    """

    # canonicalize device/dtype
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    # start from a shallow copy of cache so we never mutate the caller's object
    new_cache = dict(cache) if cache is not None else {}

    # return from cache if present
    if cache is not None and cache_key in cache and "chebyshev_coeffs" in cache[cache_key]:
        entry = cache[cache_key]
        return entry["chebyshev_coeffs"], entry["chebyshev_eigenvalue_samples"], new_cache

    # compute fresh
    num_chebyshev_points = chebyshev_max_degree + 1
    chebyshev_angles = torch.arange(num_chebyshev_points, device=device, dtype=dtype)
    chebyshev_angles = math.pi * (chebyshev_angles + 0.5) / float(num_chebyshev_points)
    cosine_chebyshev_angles = torch.cos(chebyshev_angles)
    scale_a = (lambda_max - lambda_min) / 2.0
    shift_b = (lambda_max + lambda_min) / 2.0
    chebyshev_eigenvalue_samples = scale_a * cosine_chebyshev_angles + shift_b

    num_filters = len(spectral_kernels)
    kernel_samples = torch.empty((num_filters, num_chebyshev_points), device=device, dtype=dtype)
    for i, kernel in enumerate(spectral_kernels):
        kernel_samples[i] = kernel(chebyshev_eigenvalue_samples).to(device=device, dtype=dtype)

    degree_index = torch.arange(chebyshev_max_degree + 1, device=device, dtype=dtype).unsqueeze(1)
    cosine_matrix = torch.cos(degree_index * chebyshev_angles.unsqueeze(0))
    chebyshev_coeffs = (2.0 / float(num_chebyshev_points)) * (kernel_samples @ cosine_matrix.T)

    # store into the copied cache (new_cache) under the cache_key
    new_cache = dict(new_cache)  # ensure we copy before nested set
    new_cache.setdefault(cache_key, {})
    new_cache[cache_key]["chebyshev_coeffs"] = chebyshev_coeffs
    new_cache[cache_key]["chebyshev_eigenvalue_samples"] = chebyshev_eigenvalue_samples
    new_cache[cache_key].setdefault("meta", {})["num_filters"] = num_filters
    new_cache[cache_key].setdefault("meta", {})["chebyshev_max_degree"] = chebyshev_max_degree
    new_cache[cache_key].setdefault("meta", {})["lambda_min"] = lambda_min
    new_cache[cache_key].setdefault("meta", {})["lambda_max"] = lambda_max

    return chebyshev_coeffs, chebyshev_eigenvalue_samples, new_cache


def apply_chebyshev_filters(
        x: torch.Tensor,
        chebyshev_coeffs: torch.Tensor,
        chebyshev_max_degree: int,
        L_sym_sparse: torch.Tensor,
        lambda_min: float,
        lambda_max: float,
) -> torch.Tensor:
    """
    Apply the Chebyshev polynomial approximation to a vertex-domain signal x
    using a shared Chebyshev basis T_k(˜L) x.

    Parameters
    ----------
    x :
        Input signal of shape (N,) or (N, M) where N is number of nodes and
        M is number of multiple right-hand-sides (signals).
    chebyshev_coeffs :
        Coefficients tensor of shape (F, K+1) where F is number of filters
        and K = chebyshev_max_degree.
    chebyshev_max_degree :
        Maximum degree K of Chebyshev polynomials.
    L_sym_sparse :
        Sparse symmetric graph Laplacian tensor (N x N) (torch sparse tensor).
    lambda_min :
        Minimum eigenvalue (used to construct tilde operator).
    lambda_max :
        Maximum eigenvalue (used to construct tilde operator).

    Returns
    -------
    Y :
        Filter responses of shape (F, N) where each row m is g_m(L) x (approximated).
    """
    # Device / dtype inferred from x (so results match input)
    device = x.device  # e.g., torch.device('cpu') or 'cuda'
    dtype = x.dtype  # e.g., torch.float32

    # Number of nodes N (graph size)
    num_nodes = x.shape[0]  # scalar

    # Number of filters F
    num_filters = chebyshev_coeffs.shape[0]  # scalar (F)

    # Build sparse identity matrix I (N x N) — memory-efficient, no dense allocation
    indices = torch.arange(num_nodes, device=device, dtype=torch.long)
    I_sparse = torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        torch.ones(num_nodes, device=device, dtype=dtype),
        size=(num_nodes, num_nodes),
    )

    # Build affine parameters to construct tilde operator ˜L = (L - b I) / a
    scale_a = (lambda_max - lambda_min) / 2.0  # a scalar
    shift_b = (lambda_max + lambda_min) / 2.0  # a scalar

    # Construct ˜L_sym_sparse = (L_sym_sparse - b * I) / a (sparse)
    # Note: operations with sparse tensors produce sparse results where supported.
    L_sym_tilde_sparse = (L_sym_sparse - shift_b * I_sparse) * (1.0 / scale_a)  # (N, N) sparse

    # Prepare input signal for sparse mm: ensure shape (N, M)
    if x.ndim == 1:
        x_mat = x.unsqueeze(1)  # (N, 1)
        single_vector = True
    else:
        x_mat = x  # (N, M)
        single_vector = False

    # Chebyshev basis recursion:
    # T0 = x
    T0 = x_mat  # (N, M)
    T_list: List[torch.Tensor] = [T0]  # list of length at least 1, each (N, M)

    # If K >= 1 compute T1 = ˜L x
    if chebyshev_max_degree >= 1:
        # sparse.mm expects (N, N) sparse, (N, M) dense -> (N, M) dense
        T1 = torch.sparse.mm(L_sym_tilde_sparse, x_mat)  # (N, M)
        T_list.append(T1)  # append T1

    # Recurrence for k = 1..K-1 to compute T_{k+1}
    for k in range(1, chebyshev_max_degree):
        # T_next = 2 * ˜L * T_k - T_{k-1}
        T_next = 2.0 * torch.sparse.mm(L_sym_tilde_sparse, T_list[k]) - T_list[
            k - 1
            ]  # (N, M)
        T_list.append(T_next)  # append T_{k+1}

    # Stack T_k across k into tensor of shape (K+1, N, M)
    # We will sum over k with coefficients: result shape (F, N, M)
    T = torch.stack(T_list, dim=0)  # (K+1, N, M)

    # For combination we want (K+1, N) if single_vector, else (K+1, N, M)
    # Prepare effective coefficients: half the k=0 term per Chebyshev series convention
    c_eff = chebyshev_coeffs.clone()  # (F, K+1)
    c_eff[:, 0] = 0.5 * c_eff[:, 0]  # (F, K+1) ; apply 1/2 to constant term

    # Combine coefficients with basis:
    # If single vector: T has shape (K+1, N, 1) -> squeeze to (K+1, N) and do matrix multiply
    if single_vector:
        T_stack = T.squeeze(-1)  # (K+1, N)
        # Y = c_eff @ T_stack  -> (F, K+1) @ (K+1, N) -> (F, N)
        Y = c_eff @ T_stack  # (F, N)
    else:
        # Multiple signals case: we compute Y for each signal separately and then sum.
        # T: (K+1, N, M). We want output shape (F, N, M). We'll do einsum:
        # Y[f,n,m] = sum_k c_eff[f,k] * T[k,n,m]
        Y = torch.einsum("fk,knm->fnm", c_eff, T)  # (F, N, M)
        # For compatibility with previous behavior (return (F, N) for single RHS),
        # if M==1 we squeeze.
        if Y.shape[-1] == 1:
            Y = Y.squeeze(-1)  # (F, N)

    # Return Y with shape (F, N) (or (F, N, M) if multiple RHS provided)
    return Y


def apply_chebyshev_filters_with_L_tilde(
        x: torch.Tensor,
        chebyshev_coeffs: torch.Tensor,
        chebyshev_max_degree: int,
        L_sym_tilde: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Chebyshev polynomial approximation using PRE-COMPUTED rescaled Laplacian.

    This is the fast version that skips the expensive L̃ = (L - bI)/a computation.
    Use this when you have already computed L_sym_tilde (e.g., from cache).

    Parameters
    ----------
    x :
        Input signal of shape (N,) or (N, M) where N is number of nodes.
    chebyshev_coeffs :
        Coefficients tensor of shape (F, K+1) where F is number of filters
        and K = chebyshev_max_degree.
    chebyshev_max_degree :
        Maximum degree K of Chebyshev polynomials.
    L_sym_tilde :
        PRE-COMPUTED rescaled Laplacian L̃ = (L - bI)/a, sparse tensor (N, N).

    Returns
    -------
    Y :
        Filter responses of shape (F, N) where each row m is g_m(L) x (approximated).
    """
    # Prepare input signal for sparse mm: ensure shape (N, M)
    if x.ndim == 1:
        x_mat = x.unsqueeze(1)  # (N, 1)
        single_vector = True
    else:
        x_mat = x  # (N, M)
        single_vector = False

    # Chebyshev basis recursion: T0 = x
    T0 = x_mat  # (N, M)
    T_list: List[torch.Tensor] = [T0]

    # If K >= 1 compute T1 = L̃ @ x
    if chebyshev_max_degree >= 1:
        T1 = torch.sparse.mm(L_sym_tilde, x_mat)  # (N, M)
        T_list.append(T1)

    # Recurrence for k = 1..K-1 to compute T_{k+1}
    for k in range(1, chebyshev_max_degree):
        T_next = 2.0 * torch.sparse.mm(L_sym_tilde, T_list[k]) - T_list[k - 1]  # (N, M)
        T_list.append(T_next)

    # Stack T_k into tensor of shape (K+1, N, M)
    T = torch.stack(T_list, dim=0)  # (K+1, N, M)

    # Prepare effective coefficients: half the k=0 term per Chebyshev series convention
    c_eff = chebyshev_coeffs.clone()  # (F, K+1)
    c_eff[:, 0] = 0.5 * c_eff[:, 0]

    # Combine coefficients with basis
    if single_vector:
        T_stack = T.squeeze(-1)  # (K+1, N)
        Y = c_eff @ T_stack  # (F, N)
    else:
        Y = torch.einsum("fk,knm->fnm", c_eff, T)  # (F, N, M)
        if Y.shape[-1] == 1:
            Y = Y.squeeze(-1)  # (F, N)

    return Y


def chebyshev_gwt_analysis(
        x: torch.Tensor,
        L_sym_sparse: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        num_nodes: Optional[int],
        gwt_scales: List[float],
        lambda_min: float,
        lambda_max: float,
        chebyshev_max_degree: int = 30,
        kernel_id: str = "default",
        cache: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    """
    Compute approximate Graph Wavelet Transform (analysis) coefficients using
    Chebyshev polynomial approximations of the spectral kernels.

    Computes approximate GWT coefficients by applying Chebyshev-approximated
    spectral filters (scaling function + wavelets at multiple scales) to the
    input graph signal.

    Parameters
    ----------
    x :
        Input graph signal of shape (N,) or (N, M) where N is number of nodes
        and M is number of signals.
    L_sym_sparse :
        Sparse symmetric normalized Laplacian (N, N). If None, will be computed
        from edge_index.
    edge_index :
        Edge index tensor of shape (2, E). Required if L_sym_sparse is None.
    num_nodes :
        Number of nodes in the graph. Required if L_sym_sparse is None.
    gwt_scales :
        List of wavelet scales for the Mexican hat wavelets.
    lambda_min :
        Minimum eigenvalue of the Laplacian spectrum.
    lambda_max :
        Maximum eigenvalue of the Laplacian spectrum.
    chebyshev_max_degree :
        Maximum degree K of Chebyshev polynomials (default: 30).
    kernel_id :
        User-defined identifier for the filter bank configuration. Used to
        distinguish cache entries when different spectral kernels are applied
        to the same graph. Change this value when using different wavelet
        types or filter configurations (default: "default").
    cache :
        Optional cache dictionary for reusing precomputed values.

    Returns
    -------
    approximate_gwt_coeffs :
        GWT coefficients tensor of shape (N, F).
    key :
        Cache key used to fetch cached values from the input cache dictionary.
    new_cache :
        Updated cache dictionary (does not mutate input cache). Contains cached
        values accessible via cache[cache_key], including 'chebyshev_coeffs',
        'chebyshev_eigenvalue_samples', and 'L_sym_tilde'.
    """
    # canonicalize device/dtype
    device = x.device
    dtype = x.dtype

    # ensure Laplacian is available: either provided or build from edge_index
    if L_sym_sparse is None:
        assert edge_index is not None and num_nodes is not None
        L_sym_sparse = compute_sparse_L_sym(
            edge_index=edge_index,
            edge_weight=None,
            num_nodes=int(num_nodes),
            normalization="sym",
        )

    # build cache key (prefer edge_index in key when available)
    key = compute_cache_key(edge_index=edge_index if edge_index is not None else None,
                            num_nodes=int(num_nodes or x.shape[0]),
                            kernel_id=kernel_id,
                            lambda_min=lambda_min,
                            lambda_max=lambda_max,
                            chebyshev_max_degree=chebyshev_max_degree)

    # prepare spectral_kernels (scaling + wavelets)
    spectral_kernels: List[Callable[[torch.Tensor], torch.Tensor]] = []

    def _scaling_kernel(eigenvalues: torch.Tensor) -> torch.Tensor:
        return scaling_kernel(eigenvalues=eigenvalues, minimum_eigenvalue=lambda_min)
    spectral_kernels.append(_scaling_kernel)

    for scale in gwt_scales:
        def _mexican_hat_kernel(s: float) -> Callable[[torch.Tensor], torch.Tensor]:
            return lambda eigenvalues: mexican_hat_kernel(eigenvalues=eigenvalues, scale=s)
        spectral_kernels.append(_mexican_hat_kernel(scale))

    # compute or fetch chebyshev coeffs (functional, returns new_cache)
    chebyshev_coeffs, chebyshev_eigenvalue_samples, new_cache = compute_chebyshev_coefficients(
        spectral_kernels=spectral_kernels,
        chebyshev_max_degree=chebyshev_max_degree,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        device=device,
        dtype=dtype,
        cache=cache,
        cache_key=key,
    )

    # compute or fetch tilde-L (functional)
    L_sym_tilde, new_cache = compute_tilde_L(
        L_sym_sparse=L_sym_sparse,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        cache=new_cache,
        cache_key=key,
    )

    # apply filters: here apply_chebyshev_filters expects the original L and will recompute tilde inside.
    # To avoid double-tilde computation, call a variant of the apply function that accepts pre-tilde L.
    # For simplicity we reuse apply_chebyshev_filters but pass original L_sym_sparse (so internal tilde recomputed).
    # If you want to avoid recomputing tilde inside apply_chebyshev_filters, adapt it to accept a flag.
    Y = apply_chebyshev_filters(
        x=x,
        chebyshev_coeffs=chebyshev_coeffs,
        chebyshev_max_degree=chebyshev_max_degree,
        L_sym_sparse=L_sym_sparse,  # pass original L
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )  # Y: (F,N) or (F,N,M)

    # flatten and reshape as before
    if Y.ndim == 2:
        w_flat = Y.reshape(-1)
    elif Y.ndim == 3:
        w_flat = Y.reshape(-1)
    else:
        w_flat = Y.flatten()

    approximate_gwt_coeffs = reshape_coeffs_to_2d(coeffs=w_flat,
                                                  num_nodes=int(x.shape[0]),
                                                  num_filters=chebyshev_coeffs.shape[0])
    return approximate_gwt_coeffs, key, new_cache


def _chebyshev_gwt_synthesis(
        approximate_gwt_coeffs: torch.Tensor,
        chebyshev_coeffs: torch.Tensor,
        chebyshev_eigenvalue_samples: torch.Tensor,
        L_sym_sparse: torch.Tensor,
        lambda_min: float,
        lambda_max: float,
        chebyshev_max_degree: int,
        eps: float = 1e-12,
) -> torch.Tensor:
    """
    Reconstruct (synthesize) a graph signal from approximate GWT coefficients
    using the canonical dual filters computed from Chebyshev-approximated kernels.

    The function computes the dual filters h_m(λ) = p_m(λ) / S(λ) where
    p_m is the Chebyshev-approximated analysis kernel and S(λ) = sum_m p_m(λ)^2.
    Then it applies h_m(L) to each per-filter coefficient vector w_m and sums
    the results to produce the reconstructed signal.

    Parameters
    ----------
    approximate_gwt_coeffs :
        Analysis coefficients produced by chebyshev_gwt_analysis. Shape is
        either (N, F) or (F, N). Columns correspond to filters (we will make
        a canonical (N, F) view).
    chebyshev_coeffs :
        Chebyshev coefficients used for approximating analysis kernels
        (F, K+1), where K = chebyshev_max_degree.
    chebyshev_eigenvalue_samples :
        Eigenvalue sample nodes used to compute coefficients, shape (K+1,).
    L_sym_sparse :
        Sparse symmetric Laplacian (N, N) (torch sparse tensor).
    lambda_min :
        Minimum eigenvalue of the Laplacian (float).
    lambda_max :
        Maximum eigenvalue of the Laplacian (float).
    chebyshev_max_degree :
        Degree K used for Chebyshev approximation.
    eps :
        Small epsilon to avoid division by zero when computing S(λ).

    Returns
    -------
    x_rec :
        Reconstructed vertex-domain signal (N,) as a dense tensor.
    """
    # --- normalize and canonicalize inputs ----------------------------------
    device = L_sym_sparse.device  # device of laplacian (prefer this)
    dtype = chebyshev_coeffs.dtype  # dtype from coefficients

    # ensure chebyshev_eigenvalue_samples is tensor on same device/dtype
    lambda_nodes = chebyshev_eigenvalue_samples.to(device=device, dtype=dtype)  # (K+1,)

    # canonical view of approximate_gwt_coeffs -> W (N, F)
    # handle (N,F) or (F,N)
    if approximate_gwt_coeffs.ndim != 2:
        raise ValueError("approximate_gwt_coeffs must be 2D (N,F) or (F,N).")
    if approximate_gwt_coeffs.shape[0] == chebyshev_coeffs.shape[0] and approximate_gwt_coeffs.shape[1] != chebyshev_coeffs.shape[0]:
        # likely (F, N) -> transpose
        # (F,N) and chebyshev_coeffs is (F,K+1)
        W = approximate_gwt_coeffs.T.to(device=device, dtype=dtype)  # (N, F)
    elif approximate_gwt_coeffs.shape[1] == chebyshev_coeffs.shape[0]:
        # likely (N, F)
        W = approximate_gwt_coeffs.to(device=device, dtype=dtype)  # (N, F)
    else:
        # ambiguous shapes: try to coerce to (N,F) where F = chebyshev_coeffs.shape[0]
        F = chebyshev_coeffs.shape[0]
        N = W_shape0 = approximate_gwt_coeffs.shape[0]
        if approximate_gwt_coeffs.numel() == N * F:
            W = approximate_gwt_coeffs.reshape(N, F).to(device=device, dtype=dtype)
        else:
            raise ValueError(
                f"Cannot infer orientation of approximate_gwt_coeffs {approximate_gwt_coeffs.shape} vs chebyshev_coeffs {chebyshev_coeffs.shape}."
            )

    # shapes
    N, F = W.shape  # N nodes, F filters
    K = chebyshev_max_degree  # maximum degree

    # --- compute sampled approx. frequency responses p_m(lambda_j) ---------
    # affine map params a,b
    a = (lambda_max - lambda_min) / 2.0  # scalar
    b = (lambda_max + lambda_min) / 2.0  # scalar

    # compute x_j = (lambda_j - b) / a  (in [-1,1]), then theta_j = arccos(x_j)
    x_nodes = (lambda_nodes - b) / a  # (K+1,)
    # guard numerical issues: clamp to [-1,1]
    x_nodes = torch.clamp(x_nodes, min=-1.0, max=1.0)
    theta = torch.acos(x_nodes)  # (K+1,)

    # build cos(k * theta_j) matrix: cos_k_theta shape (K+1, K+1) where rows k, cols j
    k_idx = torch.arange(K + 1, device=device, dtype=dtype).unsqueeze(1)  # (K+1,1)
    cos_k_theta = torch.cos(k_idx * theta.unsqueeze(0))  # (K+1, K+1)  (k, j)

    # evaluate p_m at each sampled node j: p = c @ cos_k_theta  -> (F, K+1)
    # note: chebyshev_coeffs shape (F, K+1), cos_k_theta.T shape (K+1, K+1) but evaluation at nodes uses cos(k theta_j)
    # compute p_samples[m,j] = 0.5*c_{m,0} + sum_{k>=1} c_{m,k} cos(k theta_j)
    # the matrix product below yields exactly that (because cos_k_theta has k varying along rows)
    # apply 1/2 to k=0 term before evaluating polynomial at nodes
    c_eff = chebyshev_coeffs.to(device=device, dtype=dtype).clone()  # (F, K+1)
    c_eff[:, 0] = 0.5 * c_eff[:, 0]  # (F, K+1)
    p_samples = c_eff @ cos_k_theta  # (F, K+1)

    # --- compute frame operator samples S_j = sum_m p_m(λ_j)^2 ----------------
    S_samples = torch.sum(p_samples * p_samples, dim=0)  # (K+1,)

    # clamp small values to avoid division by zero
    S_samples = torch.clamp(S_samples, min=eps)

    # --- compute sampled dual kernels d_m(λ_j) = p_m(λ_j) / S_j -------------
    d_samples = p_samples / S_samples.unsqueeze(0)  # (F, K+1)

    # --- project sampled dual kernels back to Chebyshev coefficients -------
    # d_coeffs_{m,k} = (2/N) sum_j d_samples[m,j] * cos(k theta_j)
    # reuse cos_k_theta for projection: cos_k_theta (k, j)
    d_chebyshev_coeffs = (2.0 / float(K + 1)) * (d_samples @ cos_k_theta.T)  # (F, K+1)

    # apply 1/2 factor to k=0 for evaluation code path (we will use c_eff below)
    d_chebyshev_coeffs = d_chebyshev_coeffs.to(device=device, dtype=dtype)  # (F, K+1)
    d_chebyshev_coeffs[:, 0] = 0.5 * d_chebyshev_coeffs[:, 0]  # (F, K+1)

    # --- compute Chebyshev basis T_k(˜L) applied to each column of W -----------
    # Build ˜L = (L_sym_sparse - b I) / a
    indices = torch.arange(N, device=device, dtype=torch.long)
    I_sparse = torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        torch.ones(N, device=device, dtype=dtype),
        size=(N, N),
    )
    L_sym_tilde = (L_sym_sparse - b * I_sparse) * (1.0 / a)  # (N, N) sparse

    # Prepare W as (N, F) dense for sparse mm
    W_mat = W  # already (N, F)

    # Recurrence: compute T0, T1, ..., TK applied to all RHS columns of W_mat
    # Use memory-efficient rolling buffer instead of storing all K+1 terms
    # We need: T_k for the dot product with d_chebyshev_coeffs, accumulated on-the-fly

    # Initialize accumulator for Y_per_filter: (F, N)
    # Y_per_filter[m, n] = sum_k d_eff[m,k] * T[k, n, m]
    # We accumulate contributions as we compute each T_k

    T_prev_prev = W_mat  # T_0 = W_mat, shape (N, F)

    # Accumulate k=0 contribution
    # d_chebyshev_coeffs[:, 0] already has the 0.5 factor applied
    # We need: for each filter m, Y[m, n] += d[m, 0] * T0[n, m]
    # T_prev_prev is (N, F), d_chebyshev_coeffs[:, 0] is (F,)
    Y_accum = T_prev_prev * d_chebyshev_coeffs[:, 0].unsqueeze(0)  # (N, F)

    if K >= 1:
        T_prev = torch.sparse.mm(L_sym_tilde, W_mat)  # T_1, shape (N, F)
        # Accumulate k=1 contribution
        Y_accum = Y_accum + T_prev * d_chebyshev_coeffs[:, 1].unsqueeze(0)  # (N, F)

        for k in range(2, K + 1):
            T_curr = 2.0 * torch.sparse.mm(L_sym_tilde, T_prev) - T_prev_prev  # (N, F)
            # Accumulate k-th contribution
            Y_accum = Y_accum + T_curr * d_chebyshev_coeffs[:, k].unsqueeze(0)  # (N, F)
            # Roll buffer forward
            T_prev_prev = T_prev
            T_prev = T_curr

        del T_prev, T_prev_prev, T_curr
    else:
        del T_prev_prev

    # Y_accum is (N, F) where Y_accum[n, m] = sum_k d[m,k] * T_k[n, m] = h_m(L) w_m at node n
    # Sum across filters to reconstruct x
    x_rec = torch.sum(Y_accum, dim=1)  # (N,)
    del Y_accum

    return x_rec


def chebyshev_gwt_synthesis(
        approximate_gwt_coeffs: torch.Tensor,
        chebyshev_coeffs: torch.Tensor,
        chebyshev_eigenvalue_samples: torch.Tensor,
        L_sym_sparse: torch.Tensor,
        lambda_min: float,
        lambda_max: float,
        chebyshev_max_degree: int,
        cache: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Reconstruct a graph signal from GWT coefficients using Chebyshev approximation.

    This is the cached wrapper around the core synthesis function. It attempts to
    reuse the transformed Laplacian from cache if available, then delegates to
    _chebyshev_gwt_synthesis for the actual reconstruction.

    Parameters
    ----------
    approximate_gwt_coeffs :
        Analysis coefficients produced by chebyshev_gwt_analysis. Shape is
        either (N, F) or (F, N) where N is nodes and F is filters.
    chebyshev_coeffs :
        Chebyshev coefficients used for approximating analysis kernels
        (F, K+1), where K = chebyshev_max_degree.
    chebyshev_eigenvalue_samples :
        Eigenvalue sample nodes used to compute coefficients, shape (K+1,).
    L_sym_sparse :
        Sparse symmetric Laplacian (N, N) (torch sparse tensor).
    lambda_min :
        Minimum eigenvalue of the Laplacian.
    lambda_max :
        Maximum eigenvalue of the Laplacian.
    chebyshev_max_degree :
        Degree K used for Chebyshev approximation.
    cache :
        Optional cache dictionary for reusing precomputed values.
    cache_key :
        Key to use for cache lookup/storage.
    eps :
        Small epsilon to avoid division by zero when computing dual filters.

    Returns
    -------
    x_rec :
        Reconstructed vertex-domain signal (N,) as a dense tensor.
    new_cache :
        Updated cache dictionary (does not mutate input cache).
    """

    # start from copy of cache
    new_cache = dict(cache) if cache is not None else {}

    # try to reuse L_sym_tilde from cache if available
    if cache is not None and cache_key is not None and cache_key in cache and "L_sym_tilde" in cache[cache_key]:
        L_sym_tilde = cache[cache_key]["L_sym_tilde"]
    else:
        # compute and update new_cache functionally
        L_sym_tilde, new_cache = compute_tilde_L(
            L_sym_sparse=L_sym_sparse,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            cache=new_cache,
            cache_key=(cache_key or "transient"),
        )

    # call the internal synthesis logic
    x_rec = _chebyshev_gwt_synthesis(
        approximate_gwt_coeffs=approximate_gwt_coeffs,
        chebyshev_coeffs=chebyshev_coeffs,
        chebyshev_eigenvalue_samples=chebyshev_eigenvalue_samples,
        L_sym_sparse=L_sym_sparse,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        chebyshev_max_degree=chebyshev_max_degree,
        eps=eps,
    )

    return x_rec, new_cache


def precompute_synthesis_data(
        chebyshev_coeffs: torch.Tensor,
        chebyshev_eigenvalue_samples: torch.Tensor,
        L_sym_sparse: torch.Tensor,
        lambda_min: float,
        lambda_max: float,
        chebyshev_max_degree: int,
        eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Precompute all graph-dependent synthesis data that is shared across samples.

    This computes the dual Chebyshev coefficients and the transformed Laplacian
    once, so they can be reused across all graphs in a batch (or across batches
    when the graph structure is shared).

    Returns a dict with 'L_sym_tilde' and 'd_chebyshev_coeffs' on the same device.
    """
    device = L_sym_sparse.device
    dtype = chebyshev_coeffs.dtype
    N = L_sym_sparse.shape[0]
    K = chebyshev_max_degree

    a = (lambda_max - lambda_min) / 2.0
    b = (lambda_max + lambda_min) / 2.0

    # Build sparse identity efficiently
    indices = torch.arange(N, device=device, dtype=torch.long)
    I_sparse = torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        torch.ones(N, device=device, dtype=dtype),
        size=(N, N),
    )
    L_sym_tilde = (L_sym_sparse - b * I_sparse) * (1.0 / a)

    # Compute dual Chebyshev coefficients (same as in _chebyshev_gwt_synthesis)
    lambda_nodes = chebyshev_eigenvalue_samples.to(device=device, dtype=dtype)
    x_nodes = (lambda_nodes - b) / a
    x_nodes = torch.clamp(x_nodes, min=-1.0, max=1.0)
    theta = torch.acos(x_nodes)

    k_idx = torch.arange(K + 1, device=device, dtype=dtype).unsqueeze(1)
    cos_k_theta = torch.cos(k_idx * theta.unsqueeze(0))

    c_eff = chebyshev_coeffs.to(device=device, dtype=dtype).clone()
    c_eff[:, 0] = 0.5 * c_eff[:, 0]
    p_samples = c_eff @ cos_k_theta

    S_samples = torch.sum(p_samples * p_samples, dim=0)
    S_samples = torch.clamp(S_samples, min=eps)
    d_samples = p_samples / S_samples.unsqueeze(0)

    d_chebyshev_coeffs = (2.0 / float(K + 1)) * (d_samples @ cos_k_theta.T)
    d_chebyshev_coeffs[:, 0] = 0.5 * d_chebyshev_coeffs[:, 0]

    return {
        'L_sym_tilde': L_sym_tilde,
        'd_chebyshev_coeffs': d_chebyshev_coeffs,
    }

def chebyshev_gwt_synthesis_fast(
        approximate_gwt_coeffs: torch.Tensor,
        chebyshev_coeffs: torch.Tensor,
        precomputed: Dict[str, Any],
        chebyshev_max_degree: int,
) -> torch.Tensor:
    """
    Fast Chebyshev IGWT synthesis using precomputed L_sym_tilde and dual coefficients.

    Args:
        approximate_gwt_coeffs: (N, F) GWT coefficients for a single graph
        chebyshev_coeffs: (F, K+1) original Chebyshev coefficients (used for shape inference)
        precomputed: dict from precompute_synthesis_data()
        chebyshev_max_degree: degree K

    Returns:
        x_rec: (N,) reconstructed signal
    """
    device = precomputed['L_sym_tilde'].device
    dtype = precomputed['d_chebyshev_coeffs'].dtype
    K = chebyshev_max_degree

    # Canonicalize W to (N, F)
    W = approximate_gwt_coeffs
    if W.shape[1] != chebyshev_coeffs.shape[0]:
        W = W.T
    W = W.to(device=device, dtype=dtype)

    L_sym_tilde = precomputed['L_sym_tilde']
    d_chebyshev_coeffs = precomputed['d_chebyshev_coeffs']

    # NaN CHECK 4a: Input to Chebyshev synthesis
    if not torch.isfinite(W).all():
        print(f"🔥 NaN in approximate_gwt_coeffs input (W)")
    if not torch.isfinite(d_chebyshev_coeffs).all():
        print(f"🔥 NaN in precomputed dual_filter_coeffs")

    # Rolling Chebyshev recurrence with on-the-fly accumulation
    T_prev_prev = W  # T_0
    Y_accum = T_prev_prev * d_chebyshev_coeffs[:, 0].unsqueeze(0)

    # NaN CHECK 4b: After T_0 accumulation
    if not torch.isfinite(Y_accum).all():
        print(f"🔥 NaN in Y_accum after T_0 (k=0)")
        print(
            f"   d_chebyshev_coeffs[:, 0] stats: min={d_chebyshev_coeffs[:, 0].min().item():.6f}, max={d_chebyshev_coeffs[:, 0].max().item():.6f}")

    if K >= 1:
        T_prev = torch.sparse.mm(L_sym_tilde, W)  # T_1
        # NaN CHECK 4c: After T_1 computation
        if not torch.isfinite(T_prev).all():
            print(f"🔥 NaN in T_prev (T_1) after sparse mm")

        Y_accum = Y_accum + T_prev * d_chebyshev_coeffs[:, 1].unsqueeze(0)

        # NaN CHECK 4d: After T_1 accumulation
        if not torch.isfinite(Y_accum).all():
            print(f"🔥 NaN in Y_accum after T_1 (k=1)")

        for k in range(2, K + 1):
            T_curr = 2.0 * torch.sparse.mm(L_sym_tilde, T_prev) - T_prev_prev

            # NaN CHECK 4e: After T_k computation
            if not torch.isfinite(T_curr).all():
                print(f"🔥 NaN in T_curr at k={k}")
                print(f"   T_prev finite: {torch.isfinite(T_prev).all()}")
                print(f"   T_prev_prev finite: {torch.isfinite(T_prev_prev).all()}")
                break  # Stop if NaN detected to prevent cascade

            Y_accum = Y_accum + T_curr * d_chebyshev_coeffs[:, k].unsqueeze(0)
            # NaN CHECK 4f: After T_k accumulation
            if not torch.isfinite(Y_accum).all():
                print(f"🔥 NaN in Y_accum after k={k}")
                print(
                    f"   d_chebyshev_coeffs[:, {k}] stats: min={d_chebyshev_coeffs[:, k].min().item():.6f}, max={d_chebyshev_coeffs[:, k].max().item():.6f}")
                break  # Stop if NaN detected

            T_prev_prev = T_prev
            T_prev = T_curr

        del T_prev, T_prev_prev

    x_rec = torch.sum(Y_accum, dim=1)
    # NaN CHECK 4g: Final output
    if not torch.isfinite(x_rec).all():
        print(f"🔥 NaN in reconstructed_signal output (x_rec)")
        print(f"   Y_accum finite: {torch.isfinite(Y_accum).all()}")

    del Y_accum
    return x_rec



def main():
    import torch
    from visualization.gwt_coeffs import (
        visualize_grid_graph_sgwt_coefficients,
        visualize_grid_graph_inverse_sgwt_reconstruction
    )

    # loading the sample data for testing
    data = torch.load('test_dataset_0_2025-10-25_11-45-48.pt', weights_only=False)

    # unpacking required values
    rows = data['graph_metadata']['height']
    cols = data['graph_metadata']['width']
    num_filters = data.coeffs.shape[1]

    # computing L_sym_sparse
    # Compute symmetric normalized Laplacian (sparse) using provided util
    L_sym_sparse = compute_sparse_L_sym(
        edge_index=data.edge_index,
        edge_weight=None,
        num_nodes=int(data.x.shape[0]),
        normalization="sym",
    )  # sparse tensor shape: (N, N)

    # the corresponding gwt scales that generated the exact gwt coeffs (from data generation configs)
    gwt_scale = [0.5, 1.8420157493201936, 6.786044041487268, 25.000000000000007]

    # computing the approximate gwt coeffs
    lambda_min = 0.0
    lambda_max = 2.0
    chebyshev_max_degree = 100
    cache: Dict[str, Any] = {}
    approximate_gwt_coeffs, cache_key, cache = chebyshev_gwt_analysis(
        x=data.x,
        L_sym_sparse=L_sym_sparse,
        edge_index=data.edge_index,
        num_nodes=int(data.x.shape[0]),
        gwt_scales=gwt_scale,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        chebyshev_max_degree=chebyshev_max_degree,
        cache=cache,
    )
    chebyshev_coeffs = cache[cache_key]['chebyshev_coeffs']
    chebyshev_eigenvalue_samples = cache[cache_key]['chebyshev_eigenvalue_samples']

    # exact gwt coeffs
    visualize_grid_graph_sgwt_coefficients(
        coeffs=data.coeffs,
        num_filters=num_filters,
        scales=gwt_scale,
        rows=rows,
        cols=cols,
    )
    # approximate gwt coeffs
    visualize_grid_graph_sgwt_coefficients(
        coeffs=approximate_gwt_coeffs,
        num_filters=num_filters,
        scales=gwt_scale,
        rows=rows,
        cols=cols,
    )

    # reconstructing the signal from the approximate gwt coeffs
    x_hat, cache = chebyshev_gwt_synthesis(
        approximate_gwt_coeffs=approximate_gwt_coeffs,
        chebyshev_coeffs=chebyshev_coeffs,
        chebyshev_eigenvalue_samples=chebyshev_eigenvalue_samples,
        L_sym_sparse=L_sym_sparse,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        chebyshev_max_degree=chebyshev_max_degree,
        cache=cache,
        cache_key=cache_key,
    )
    visualize_grid_graph_inverse_sgwt_reconstruction(
        x_recon=x_hat,
        x=data.x,
        rows=rows,
        cols=cols,
        save_fig=False
    )
    pass


if __name__ == "__main__":
    main()


