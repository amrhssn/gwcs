from typing import Tuple, List, Callable, Optional

import torch


def selection(
    selected: List[int],
    T_g_tmp: torch.Tensor,  # shape: (N, N)
) -> int:
    """
    Internal sampling selection function (Python port).

    MATLAB:
        function sensor = selection(selected,T_g_tmp)
            if ~isempty(selected)
                T=sum(T_g_tmp(:,selected),2);
                T2=mean(T)-T;
                T2(logical(T2<0))=0;
                T_g=(T_g_tmp)*(T2);
            else
                 T_g=sum(T_g_tmp);
            end

            T_g(selected) = 0;
            [~,sensor]  =max(T_g);
        end
    """
    if len(selected) > 0:
        # idx: tensor of selected indices, shape: (|S|,)
        idx = torch.tensor(selected, dtype=torch.long, device=T_g_tmp.device)

        # T_cov = sum over selected columns, shape: (N,)
        # MATLAB: T = sum(T_g_tmp(:, selected), 2);
        T_cov = T_g_tmp[:, idx].sum(dim=1)

        # T_mean: scalar (mean coverage)
        T_mean = T_cov.mean()

        # T2: coverage deficit per node, shape: (N,)
        # MATLAB: T2 = mean(T) - T;
        T2 = T_mean - T_cov

        # Clamp negative deficits to 0
        # MATLAB: T2(logical(T2<0))=0;
        T2 = torch.clamp(T2, min=0.0)

        # T_g: score per node, shape: (N,)
        # MATLAB: T_g = (T_g_tmp) * (T2);
        # matrix-vector product
        T_g = T_g_tmp @ T2
    else:
        # No selected nodes yet: score is total influence per node
        # MATLAB: T_g = sum(T_g_tmp);
        # Here we sum over rows to get a per-column score; resulting shape: (N,)
        T_g = T_g_tmp.sum(dim=0)

    # Zero out already selected nodes
    # MATLAB: T_g(selected) = 0;
    if len(selected) > 0:
        idx = torch.tensor(selected, dtype=torch.long, device=T_g_tmp.device)
        T_g[idx] = 0.0

    # sensor: int index of node with maximum score
    # MATLAB: [~, sensor] = max(T_g);
    sensor = int(torch.argmax(T_g).item())
    return sensor


def g(
    x: torch.Tensor,   # shape: arbitrary, elementwise
    nu: float,
    p: float,
    n_ratio: float,
    k_ratio: float,
) -> torch.Tensor:
    """
    Elementwise spectral filter:

        g(x) = exp(-nu * p * n * k * x / 2)

    x can be eigenvalues or a Laplacian-like spectrum; shape is arbitrary,
    broadcasting is used.
    """
    return torch.exp(-nu * p * n_ratio * k_ratio * x / 2.0)


def sampling_set_selection(
    L_n: torch.Tensor,  # shape: (N, N)
    F: int,
    bw: float,
    nu: float,
    eigen_decomp: bool,
    U: Optional[torch.Tensor] = None,  # shape: (N, N)
    E: Optional[torch.Tensor] = None,  # shape: (N,) or (N, 1)
    m: Optional[int] = None,
    cheby_coeff_fn: Optional[
        Callable[
            [Callable[[torch.Tensor], torch.Tensor], int, int, Tuple[float, float]],
            torch.Tensor,  # returns coefficients c, shape: (m+1,)
        ]
    ] = None,
    cheby_op_fn: Optional[
        Callable[
            [torch.Tensor, torch.Tensor, Tuple[float, float]],
            torch.Tensor,  # returns filtered operator, shape: (N, N)
        ]
    ] = None,
) -> Tuple[List[int], torch.Tensor]:
    r"""
    FastGSSS: Eigendecomposition-free sampling set selection for graph signals.

    This is a PyTorch port of the MATLAB function `fastGSSS.m`.

    Parameters
    ----------
    L_n : torch.Tensor, shape (N, N)
        Symmetric normalized graph Laplacian.
    F : int
        Number of vertices (nodes) to be selected (size of sampling set).
    bw : float
        Estimated bandwidth of the graph signals (number of low-frequency modes).
    nu : float
        Parameter controlling the width (localization) of the filter kernel.
    eigen_decomp : bool
        Flag indicating how to build the localization operator:
        - True: Use exact eigendecomposition (requires U and E).
        - False: Use Chebyshev polynomial approximation (requires m, cheby_coeff_fn, cheby_op_fn).
    U : torch.Tensor, optional, shape (N, N)
        Eigenvector matrix of L_n (graph Fourier basis). Required if eigen_decomp == True.
    E : torch.Tensor, optional, shape (N,) or (N, 1)
        Eigenvalues of L_n. Required if eigen_decomp == True.
    m : int, optional
        Polynomial order for Chebyshev approximation. Required if eigen_decomp == False.
    cheby_coeff_fn : callable, optional
        Function implementing the Chebyshev coefficient computation:

            c = cheby_coeff_fn(g_scalar, m, m_plus_1, arange)

        where:
            g_scalar : Callable[[torch.Tensor], torch.Tensor]
                Filter g applied elementwise to eigenvalues (same shape in, same shape out).
            m : int
                Polynomial order.
            m_plus_1 : int
                Typically m + 1.
            arange : (float, float)
                Range of eigenvalues (a, b).

        Should return c of shape (m_plus_1,) or compatible.
        Required if eigen_decomp == False.
    cheby_op_fn : callable, optional
        Function implementing the Chebyshev polynomial filter application:

            T_g_tmp1 = cheby_op_fn(L_n, c, arange)

        where:
            L_n : torch.Tensor, shape (N, N)
            c   : torch.Tensor, shape (m_plus_1,)
            arange : (float, float)

        Should return T_g_tmp1 of shape (N, N).
        Required if eigen_decomp == False.

    Returns
    -------
    selected_nodes : list[int]
        Indices (0-based) of selected vertices.
    T : torch.Tensor, shape (N, N)
        Localization operator / basis for reconstruction.

    (Docstring with mathematical and intuitive description omitted here for brevity;
     see previous version for full text.)
    """
    # ----------------------------------------------------------------------
    # 1. Input checks (mirror MATLAB logic)
    # ----------------------------------------------------------------------
    if eigen_decomp:
        # eigen_decomp == True: we expect U and E
        if U is None or E is None:
            raise ValueError("For eigen_decomp = True, U and E must be provided.")
        # Ensure E is a 1D tensor, shape: (N,)
        E = E.flatten()
    else:
        # eigen_decomp == False: we expect m and Chebyshev helpers
        if m is None:
            raise ValueError("For eigen_decomp = False, m must be provided.")
        if cheby_coeff_fn is None or cheby_op_fn is None:
            raise ValueError(
                "For eigen_decomp = False, cheby_coeff_fn and cheby_op_fn must be provided."
            )

    # ----------------------------------------------------------------------
    # 2. Graph parameters (N, p, n_ratio, k_ratio)
    # ----------------------------------------------------------------------
    # MATLAB:
    # N=length(L_n);
    # A_tmp = logical(L_n - diag(diag(L_n)));
    # numedge = sum(A_tmp(:))/2;
    # p = numedge/N; % edge probability
    # n = F/N; % sampling ratio
    # k = bw/N; %bandwidth

    # N: number of nodes
    N = L_n.shape[0]

    # A_tmp: adjacency-like mask (no diagonal), shape: (N, N)
    # non-zero off-diagonal entries are treated as edges
    A_tmp = (L_n - torch.diag(torch.diag(L_n))) != 0

    # num_edges: scalar, number of undirected edges
    num_edges = A_tmp.sum().item() / 2.0

    # p: scalar, "edge probability" (average edges per node)
    p = num_edges / N

    # n_ratio: scalar, sampling ratio F / N
    n_ratio = F / float(N)

    # k_ratio: scalar, normalized bandwidth bw / N
    k_ratio = bw / float(N)

    # ----------------------------------------------------------------------
    # 3. Build localization operator T_g_tmp1
    # ----------------------------------------------------------------------
    if eigen_decomp:
        # MATLAB:
        # lmax=max(E);
        # g_E=exp(-nu*p*n*k*E/lmax);
        # T_g_tmp1=U*diag(g_E)*U.';

        # lmax: scalar, largest eigenvalue
        lmax = torch.max(E)

        # g_E: filtered eigenvalues, shape: (N,)
        g_E = torch.exp(-nu * p * n_ratio * k_ratio * E / lmax)

        # D_g: diagonal matrix of g_E, shape: (N, N)
        D_g = torch.diag(g_E)

        # T_g_tmp1: localization operator, shape: (N, N)
        # U: (N, N), U.T: (N, N)
        T_g_tmp1 = U @ D_g @ U.T
    else:
        # MATLAB:
        # arange=[0,2];
        # g = @(x)(exp(-nu*p*n*k*x/2));
        # c = sgwt_cheby_coeff(g, m, m+1, arange);
        # T_g_tmp1 = sgwt_cheby_op2(L_n, c, arange);

        # arange: eigenvalue range (a, b)
        arange: Tuple[float, float] = (0.0, 2.0)

        # Wrap g into a scalar filter depending only on x, capturing parameters
        def g_scalar(x: torch.Tensor) -> torch.Tensor:
            # x: eigenvalues, shape: arbitrary
            return g(x, nu, p, n_ratio, k_ratio)

        # c: Chebyshev coefficients, shape: (m+1,)
        c = cheby_coeff_fn(g_scalar, m, m + 1, arange)

        # T_g_tmp1: localization operator via Chebyshev approximation, shape: (N, N)
        T_g_tmp1 = cheby_op_fn(L_n, c, arange)

    # ----------------------------------------------------------------------
    # 4. Sampling Set Selection (SSS) using `selection`
    # ----------------------------------------------------------------------
    # MATLAB:
    # T_g_tmp=abs(T_g_tmp1);

    # T_g_tmp: absolute influence matrix, shape: (N, N)
    T_g_tmp = T_g_tmp1.abs()

    # First sensor: selection([], T_g_tmp)
    # selected_nodes: list[int], length grows from 1 to F
    selected_nodes: List[int] = []
    first_sensor = selection([], T_g_tmp)
    selected_nodes.append(first_sensor)

    # MATLAB:
    # for i=1:F-1
    #     sensor = selection(selected_nodes,T_g_tmp);
    #     selected_nodes = [selected_nodes sensor];
    # end
    for _ in range(F - 1):
        s = selection(selected_nodes, T_g_tmp)
        selected_nodes.append(s)

    # T: final localization operator, shape: (N, N)
    # MATLAB: T = T_g_tmp1;
    T = T_g_tmp1

    return selected_nodes, T