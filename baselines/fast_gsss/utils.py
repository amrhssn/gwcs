import math
from typing import Callable, Tuple

import torch


def sgwt_cheby_coeff(
    g: Callable[[torch.Tensor], torch.Tensor],
    m: int,
    N: int,
    arange: Tuple[float, float],
) -> torch.Tensor:
    r"""
    Compute Chebyshev coefficients of a scalar spectral filter `g` on an interval.

    Intuition
    ---------
    We want to approximate a scalar function `g(λ)` (e.g., a spectral filter) on an
    interval `[a, b]` by a truncated Chebyshev series:

        g(λ) ≈ sum_{j=0}^m c_j T_j(λ̃),

    where `T_j` are Chebyshev polynomials of the first kind and `λ̃` is λ rescaled
    from `[a, b]` to the standard Chebyshev domain `[-1, 1]`. The coefficients `c_j`
    are computed by numerical quadrature at Chebyshev nodes. These coefficients are
    later used to build a polynomial of the Laplacian that approximates `g(L)`.

    Parameters
    ----------
    g : Callable[[torch.Tensor], torch.Tensor]
        Scalar filter function to approximate. It is evaluated elementwise on a
        tensor of Chebyshev nodes. Input shape and output shape match (broadcasting
        allowed).
    m : int
        Maximum Chebyshev order to compute (we compute coefficients c_0,...,c_m).
    N : int
        Number of quadrature points (grid order). In the original toolbox the
        default is `m + 1`. In practice, you typically pass `N = m + 1`.
    arange : (float, float)
        Interval `[a, b]` on which to approximate `g`.

    Returns
    -------
    c : torch.Tensor, shape (m+1,)
        Chebyshev coefficients, where `c[j]` corresponds to the j-th coefficient
        (i.e., `c[j]` is c_j in the series).

    MATLAB reference
    ----------------
    Corresponds to:

        function c=sgwt_cheby_coeff(g,m,N,arange)
          if ~exist('N','var')
            N=m+1;
          end
          if ~exist('arange','var')
              arange=[-1, 1];
          end
          a1=(arange(2)-arange(1))/2;
          a2=(arange(2)+arange(1))/2;
          for j=1:m+1
            c(j)=sum (g(a1* cos( (pi*((1:N)-0.5))/N) + a2).* ...
                       cos(pi*(j-1)*((1:N)-.5)/N) )*2/N;
          end

    """
    # Unpack interval [a, b]
    a, b = arange

    # Scale factors to map Chebyshev domain [-1, 1] to [a, b]
    # MATLAB:
    # a1 = (arange(2)-arange(1))/2;
    # a2 = (arange(2)+arange(1))/2;
    a1 = 0.5 * (b - a)
    a2 = 0.5 * (b + a)

    # Compute coefficients on CPU to avoid implicit device assumptions.
    # They will be moved to match the Laplacian's device inside sgwt_cheby_op2.
    device = torch.device("cpu")

    # k indices: 1..N in MATLAB; here 1..N in Python
    # shape: (N,)
    k = torch.arange(1, N + 1, dtype=torch.float32, device=device)

    # Chebyshev nodes in [a, b]:
    # MATLAB: x_k = a1 * cos(pi*((1:N)-0.5)/N) + a2;
    nodes = a1 * torch.cos(math.pi * (k - 0.5) / N) + a2  # shape: (N,)

    # Evaluate g at nodes, shape: (N,)
    g_vals = g(nodes)

    # Ensure g_vals is on the same device as nodes (in case g moved it)
    if g_vals.device != device:
        g_vals = g_vals.to(device)

    # Prepare cosines for all orders j and nodes k
    # For each j = 0..m:
    #   cos_term_jk = cos(pi*(j)*((1:N)-.5)/N)
    # We build matrix C of shape (m+1, N): C[j, k] = cos_term for that j and node k.
    j_idx = torch.arange(0, m + 1, dtype=torch.float32, device=device)  # (m+1,)
    # Broadcast to (m+1, N)
    # cos(pi * j * (k-0.5) / N)
    cos_terms = torch.cos(
        math.pi * j_idx.unsqueeze(1) * (k.unsqueeze(0) - 0.5) / N
    )  # shape: (m+1, N)

    # Multiply g(nodes) with cos_terms along N and sum over nodes:
    # MATLAB per j:
    #   c(j) = sum(g(nodes) .* cos_terms_jk) * 2/N
    # Here we compute for all j at once via matmul.
    c = (cos_terms @ g_vals) * (2.0 / N)  # shape: (m+1,)

    return c


def sgwt_cheby_op2(
    L: torch.Tensor,  # shape: (N, N)
    c: torch.Tensor,  # shape: (M,) where M >= 2
    arange: Tuple[float, float],
) -> torch.Tensor:
    r"""
    Apply a Chebyshev polynomial of the Laplacian (matrix version).

    Intuition
    ---------
    This computes a matrix polynomial T(L) where T is expressed in the Chebyshev
    basis:

        T(L) ≈ sum_{k=0}^{M-1} c_k T_k(L̃),

    with T_k the Chebyshev polynomials of the first kind, and L̃ a rescaled
    version of L so that its spectrum lies in [-1, 1]. Instead of explicitly
    forming T_k(L̃), we use a three-term recurrence:

        T_0(L̃) = I
        T_1(L̃) = L̃
        T_{k+1}(L̃) = 2 L̃ T_k(L̃) - T_{k-1}(L̃)

    The result is a matrix of the same shape as L, which can be used as a
    localization/filter operator (e.g., in FastGSSS).

    Parameters
    ----------
    L : torch.Tensor, shape (N, N)
        Graph Laplacian (or any symmetric matrix with spectrum in `[a, b]`).
    c : torch.Tensor, shape (M,)
        Chebyshev coefficients for the polynomial, where c[k] is coefficient
        of T_k. Must have M >= 2 (at least coefficients for T_0 and T_1).
    arange : (float, float)
        Interval `[a, b]` used when computing the Chebyshev coefficients;
        used here to rescale L consistently.

    Returns
    -------
    r : torch.Tensor, shape (N, N)
        The matrix polynomial applied to the identity (i.e., the corresponding
        operator). This matches the MATLAB `sgwt_cheby_op2` behavior where
        it builds a full operator `r` from the Laplacian.

    MATLAB reference (matrix version)
    ---------------------------------
        function r=sgwt_cheby_op2(L,c,arange)
        M = numel(c);
        assert(all(M>=2));

        maxM=max(M);
        %Twf_new = T_j(L) f
        %Twf_cur T_{j-1}(L) f
        %TWf_old T_{j-2}(L) f

        a1=(arange(2)-arange(1))/2;
        a2=(arange(2)+arange(1))/2;
        N = length(L);
        L_hat = L - a2*speye(N);
        Twf_old=speye(size(L)); %j=0;
        Twf_cur=L_hat/a1; % j=1;

        r = .5*c(1)*Twf_old + c(2)*Twf_cur;
        for k=2:maxM
            Twf_new = (2/a1)*L_hat*Twf_cur-Twf_old;
            if 1+k<=M
                r=r+c(k+1)*Twf_new;
            end
            Twf_old=Twf_cur;
            Twf_cur=Twf_new;
            clear Twf_new
        end
    """
    # Number of Chebyshev coefficients
    M = c.numel()
    if M < 2:
        raise ValueError("c must have at least 2 coefficients (for T_0 and T_1).")

    # Ensure coefficients live on the same device as L to avoid device mismatches
    c = c.to(L.device)

    # Unpack interval [a, b]
    a, b = arange

    # Scale factors as in MATLAB:
    # a1=(arange(2)-arange(1))/2;
    # a2=(arange(2)+arange(1))/2;
    a1 = 0.5 * (b - a)
    a2 = 0.5 * (b + a)

    N = L.shape[0]
    device = L.device
    dtype = L.dtype

    # L_hat = L - a2 * I, shape: (N, N)
    # MATLAB: L_hat = L - a2*speye(N);
    I = torch.eye(N, dtype=dtype, device=device)
    L_hat = L - a2 * I

    # Twf_old: T_0(L_hat / a1) applied to I => identity, shape: (N, N)
    # MATLAB: Twf_old=speye(size(L)); % j=0
    Twf_old = I.clone()

    # Twf_cur: T_1(L_hat / a1) applied to I => (L_hat / a1), shape: (N, N)
    # MATLAB: Twf_cur=L_hat/a1; % j=1
    Twf_cur = L_hat / a1

    # Initialize result:
    # r = 0.5 * c[0] * T_0 + c[1] * T_1
    # MATLAB: r = .5*c(1)*Twf_old + c(2)*Twf_cur;
    r = 0.5 * c[0] * Twf_old + c[1] * Twf_cur

    # Recurrence for higher-order terms:
    # for k = 2:maxM
    #   Twf_new = (2/a1)*L_hat*Twf_cur - Twf_old;
    #   if 1+k <= M
    #       r = r + c(k+1)*Twf_new;
    #   end
    #   Twf_old = Twf_cur;
    #   Twf_cur = Twf_new;
    # end
    for k in range(2, M):
        # Twf_new: T_k(L_hat / a1) applied to I, shape: (N, N)
        Twf_new = (2.0 / a1) * (L_hat @ Twf_cur) - Twf_old

        # Accumulate coefficient c[k] if within range
        r = r + c[k] * Twf_new

        # Shift for next iteration
        Twf_old, Twf_cur = Twf_cur, Twf_new

    return r