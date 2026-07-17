import torch


def reconstruction(
    mem_fn: torch.Tensor,  # shape: (N,) or (N, D)
    S_opt: torch.Tensor,   # shape: (N,) boolean or {0,1} indicator
    T: torch.Tensor,       # shape: (N, N)
    k: int,
) -> torch.Tensor:
    r"""
    Reconstruct graph signals from samples using the FastGSSS localization basis.

    Intuition
    ---------
    We assume the (unknown) full signal `f` on N graph nodes can be expressed as
    a linear combination of smooth “atoms” (columns of a powered localization
    operator `T^k`) centered at the sampled nodes. Concretely:

        f ≈ T^k(:, S) α,

    where:
      - `S` is the set of sampled node indices,
      - `T^k(:, S)` is the N×|S| matrix of localized basis functions at sampled nodes,
      - `α` are the coefficients in this basis.

    We know the true signal values at the sampled nodes, `f(S)`. To match these,
    we enforce:

        T^k(S, S) α = f(S),

    and solve this linear system for `α`. Once `α` is found, we reconstruct the
    full signal everywhere as:

        f_recon = T^k(:, S) α.

    This is a smooth interpolation of the samples under the model induced by T.

    Parameters
    ----------
    mem_fn : torch.Tensor, shape (N,) or (N, D)
        Measured signal(s) on the graph. N is the number of nodes,
        D is the number of signal dimensions / channels (D can be 1).
        Only values at indices where `S_opt != 0` are used.
    S_opt : torch.Tensor, shape (N,)
        Sampling indicator vector:
          - boolean or 0/1 tensor,
          - `S_opt[i] != 0` means node i is sampled.
    T : torch.Tensor, shape (N, N)
        Localization operator (typically output of `sampling_set_selection`,
        which corresponds to `T_g_tmp1` in MATLAB).
    k : int
        Power of T to use (computes T^k). Larger k corresponds to stronger
        diffusion/smoothing in the basis functions.

    Returns
    -------
    mem_fn_recon : torch.Tensor, shape (N, D)
        Reconstructed signal(s) on all N nodes. If the input `mem_fn` was 1D,
        the output will be 2D with shape (N, 1).

    MATLAB reference
    ----------------
        function mem_fn_recon = reconstruction_fastGSSS(mem_fn,S_opt,T,k)
        %RECONSTRUCTION_POCS Summary of this function goes here
        %   Detailed explanation goes here

        S_set=find(S_opt);
        T_k=T^k;
        mem_fn_recon=T_k(:,S_set)*((T_k(S_set,S_set))\mem_fn(S_set,:));

    """
    # Ensure mem_fn is 2D: (N, D)
    # If mem_fn is 1D: shape (N,) -> (N, 1)
    if mem_fn.dim() == 1:
        mem_fn = mem_fn.unsqueeze(1)  # (N,) -> (N, 1)

    # N: number of nodes, D: number of channels
    N, D = mem_fn.shape

    # ----------------------------------------------------------------------
    # 1. Extract indices of sampled nodes (S_set)
    # ----------------------------------------------------------------------
    # MATLAB:
    #   S_set = find(S_opt);
    # Here S_opt is a tensor of shape (N,), bool or {0,1},
    # we extract indices where S_opt != 0.
    S_set = torch.nonzero(S_opt != 0, as_tuple=False).flatten()  # shape: (|S|,)

    # ----------------------------------------------------------------------
    # 2. Compute T_k = T^k
    # ----------------------------------------------------------------------
    # MATLAB:
    #   T_k = T^k;
    # Use matrix_power for integer powers.
    T_k = torch.matrix_power(T, k)  # shape: (N, N)

    # ----------------------------------------------------------------------
    # 3. Build submatrices and solve for coefficients
    # ----------------------------------------------------------------------
    # MATLAB:
    #   mem_fn_recon = T_k(:,S_set)*((T_k(S_set,S_set))\mem_fn(S_set,:));
    #
    # shapes:
    #   T_k(:, S_set)      -> (N, |S|)
    #   T_k(S_set, S_set)  -> (|S|, |S|)
    #   mem_fn(S_set, :)   -> (|S|, D)
    #
    # Solve for alpha in:
    #   T_k(S_set, S_set) * alpha = mem_fn(S_set, :)
    # where alpha has shape (|S|, D).
    #
    # Then:
    #   mem_fn_recon = T_k(:, S_set) * alpha
    # which yields shape (N, D).

    # T_k_SS: T_k(S_set, S_set), shape: (|S|, |S|)
    T_k_SS = T_k.index_select(0, S_set).index_select(1, S_set)

    # f_S: mem_fn(S_set, :), shape: (|S|, D)
    f_S = mem_fn.index_select(0, S_set)

    # Solve linear system T_k_SS * alpha = f_S
    # Using torch.linalg.solve, which expects (..., n, n) and (..., n, k)
    alpha = torch.linalg.solve(T_k_SS, f_S)  # shape: (|S|, D)

    # ----------------------------------------------------------------------
    # 4. Reconstruct full signal
    # ----------------------------------------------------------------------
    # T_k_full_S: T_k(:, S_set), shape: (N, |S|)
    T_k_full_S = T_k.index_select(1, S_set)

    # mem_fn_recon: (N, |S|) @ (|S|, D) -> (N, D)
    mem_fn_recon = T_k_full_S @ alpha

    return mem_fn_recon
