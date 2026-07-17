# Graph Wavelet Transform (GWT)
# Overview
- Graph $G = (V, E), \quad N = |V|$.
- Graph signal $\mathbf{x} \in \mathbb{R}^N$.
- Adjacency $\mathbf{A} \in \mathbb{R}^{N\times N}$. 
- Degree matrix $\mathbf{D} = \mathrm{diag}(d_1,\dots,d_N)$.
- Graph operator $\mathbf{Z} \in \mathbb{R}^{N \times N} $:
  - Combinatorial Laplacian: $\mathbf{L} = \mathbf{D} - \mathbf{A}$ (symmetric).
  - Symmetric normalized Laplacian: 
  $\mathbf{L}_\text{sym} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ 
  (symmetric; spectrum in $[0, 2]$).
  - Random-walk Laplacian: $\mathbf{L}_{\text{rw}} = \mathbf{I} - \mathbf{D}^{-1}\mathbf{A}$ 
  (spectrum in $[0, 2]$ for undirected graphs, may be asymmetric unless symmetrized).
- Spectral decomposition: 
$\mathbf{Z}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$, 
$\mathbf{U} \in \mathbb{R}^{N \times N}$, 
$\mathbf{\Lambda}=\operatorname{diag}(\lambda_1, \dots, \lambda_N)$.

- **Graph Fourier Transform (GFT):**
  - For graph operator $\mathbf{Z}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$, the eigenvectors $\mathbf{U}=[\mathbf{u}_1, \dots, \mathbf{u}_N]$ form an orthonormal basis.
  - Each eigenvector $\mathbf{u}_i$ represents a **graph frequency mode**.
  - $\lambda_i$ (eigenvalue) indicates the **oscillation rate** on the graph.
  - **Small $\lambda_i$**: $\mathbf{u}_i$ varies smoothly across connected nodes (low frequency).
  - **Large $\lambda_i$**: $\mathbf{u}_i$ oscillates rapidly between neighbors (high frequency).
  - Projecting the graph signal $\mathbf{x}$ into the eigenvector basis $\mathbf{u}_i$ reveals how much $\mathbf{x}$ contains that frequency pattern (the i-th coefficient):
  $$\hat{x}_i = \langle\ \mathbf{u}_i, \mathbf{x}\rangle = \mathbf{u}_i^\top\mathbf{x}$$
  - **Energy Interpretation:** (Parseval's theorem) $||\mathbf{x}||^2 = \sum_{i=1}^N |\hat{x}_i|^2$, where each $|\hat{x}_i|^2$ is the energy at frequency $\lambda_i$.
  - **Graph Fourier Transform:** (Analysis) Projecting the graph signal into eigenvector basis of graph operator:
  $$\hat{\mathbf{x}} = \mathbf{U}^\top \mathbf{x}$$
    - **Basis Change**: From vertex representation to spectral representation
    - **Frequency Analysis**: Reveals graph frequency content
    - **Inverse GFT:** (Synthesis) $\mathbf{x} = \mathbf{U}\hat{\mathbf{x}}$ (perfect reconstruction/Invertibility)
    - **Convolution Theorem**: Filtering becomes multiplication in spectral domain.
  - The forward GFT is a linear map from the vector space of signals on graph nodes $\mathbb{R}^N$, to the vector space of
  coefficients in the eigenvector basis (`GFT: Vertex Space -> Spectral Space`):
  $$\text{GFT}:= \mathbf{U}^\top: \mathbb{R}^N \rightarrow \mathbb{R}^N$$
  - Inverse GFT (`IGFT: Spectral Space -> Vertex Space`):
  $$\text{IGFT}:= \mathbf{U}: \mathbb{R}^N \rightarrow \mathbb{R}^N$$
  - For filtering function $g(\lambda)$, filtering is a linear map from the spectral domain to itself:
  $$\text{Filtering}: \mathbf{G}=\text{diag}(g(\lambda_1), \dots, g(\lambda_N)): \mathbb{R}^N \rightarrow \mathbb{R}^N$$
  - Complete filtering operation: 
  $$\text{IGFT} \circ \text{Filtering} \circ \text{GFT} (\mathbf{x}) = \mathbf{U}\mathbf{G}\mathbf{U}^\top \mathbf{x}$$
  - `Signal → [GFT/Analysis] → Coefficients → [Filtering/Processing] → Modified Coefficients → [Inverse GFT/Synthesis] → Filtered/Processed Signal`
- Scalar kernels:
  - Scaling (lowpass) $h(\lambda)$.
  - Mother wavelet $g(\lambda)$. 
  - Scaled wavelet: $g_s(\lambda) = g(s\lambda)$.
- Center frequency: For a wavelet $g(\lambda)$ with center frequency $\lambda_{\text{center}}$,
the scaled version $g_s(\lambda) = g(s\lambda)$, has its center frequency at $\frac{\lambda_{\text{center}}}{s}$.
  - **Small scales** → **High frequencies** (fine details)
  - **Large scales** → **Low frequencies** (coarse features)
  - This inverse relationship allows wavelets to provide a **multi-resolution analysis** where different scales capture different frequency components of the graph signal, enabling effective sparse representation of signals with varying local and global structure.



- Per-scale filter operator: 
$$\mathbf{G}_s = g_s(\mathbf{Z}) = \mathbf{U} \, g_s(\mathbf{\Lambda}) \, \mathbf{U}^\top \in \mathbb{R}^{N \times N}$$
where $g_s(\mathbf{\Lambda})=\text{diag}(g_s(\mathbf{\lambda}_1), \dots, g_s(\mathbf{\lambda}_N))$ represents the band-pass filter.
- Impulse response (atom)- The graph wavelet basis function at scale $s$ and location node $i$:
$$\mathbf{a}_{i,s} = \mathbf{G}_s \mathbf{\delta}_i$$ ($ \in \mathbb{R}^N$, column $i$ of $\mathbf{G}_s$).
- Graph wavelet coefficients at scale $s$ and node $i$ are obtained through the dot product: $w_s(i)=\mathbf{a}_{i,s}^\top \mathbf{x}$.
- Analysis coefficients per scale: $\mathbf{w}_s = \mathbf{G}_s^\top \mathbf{x}$ ($\in \mathbb{R}^N$).
- SGWT (analysis) matrix: 

    The matrix of the spectral graph wavelet transform with respect to graph operator's GFT
  $$\mathbf{\mathcal{G}} = \begin{bmatrix} \mathbf{H}^\top \\ \mathbf{G}_{1}^\top \\ \vdots \\ \mathbf{G}_{J}^\top \end{bmatrix} \in \mathbb{R}^{(J+1)N \times N}$$

- Full coefficient vector: $\mathbf{w}=\mathbf{\mathcal{G}}\mathbf{x} \in \mathbb{R}^{(J+1)N}$.
- Inverse SGWT (synthesis):
$$\mathbf{x} = \mathbf{\mathcal{G}}^{\dagger} \mathbf{w} = (\mathbf{\mathcal{G}}^{\top} \mathbf{\mathcal{G}})^{-1} \mathbf{\mathcal{G}}^{\top} \mathbf{w}$$

where $\mathbf{\mathcal{G}}^{\dagger}$ is the Moore-Penrose pseudoinverse of $\mathbf{\mathcal{G}}$.

---------
```markdown
0. Input: Graph (A or adjacency), signal x
        │
        ▼
1. Build graph operator Z (choose: L_rw, L_sym, etc.)
        │
        ├─► Estimate λ_max (the spectral upper bound)
        │
        ├─► Choose filter design (scaling + mother wavelet) → h(λ), g(λ)
        │
        ├─► Choose scales t_1...t_J  (so 1/t_j cover [λ_min,λ_max])
        │
        ▼
2. Create filterbank kernels {h(λ), g_{t_1}(λ),...,g_{t_J}(λ)}
        │
        ├─────────────────────────────────────────────────────────────────────┐
        │                                                                     │
        │ Branch A: Exact spectral filtering (GFT)                            │
        │   - eigendecompose Z = U Λ Uᵀ                                       │
        │   - build G_s = U diag(g_s(Λ)) Uᵀ                                   │
        │   - compute atoms a_{i,s} = G_s δ_i  (columns)                      │
        │   - analysis: w_s = G_sᵀ x  (or G_s x if symmetric)                 │
        │                                                                     │
        │ Branch B: Chebyshev approximation (scalable)                        │
        │   - compute Chebyshev coeffs c_k for each kernel on [0,λ_max]       │
        │   - apply filters via three-term recurrence (no eigendecomposition) │
        │   - atoms or coefficient computations via matvecs                   │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
3. Stack analysis results into big transform 𝒢 (or compute per-scale w_s)
        │
        ▼
(Optionally) Visualize sc_freqs, bands, reorder rows for plotting
        │
        ▼
4. Inverse SGWT (synthesis / reconstruction)
        │
        ├─ Exact (small graphs): use pseudoinverse or eigendecomposition
        │     x = (𝒢ᵀ𝒢)^{-1} 𝒢ᵀ w  (or closed form if tight frame)
        │
        └─ Large graphs: iterative solver (CG / GMRES) using apply_filter routines
              - define A(x)=∑_j G_{s_j} G_{s_j}ᵀ x (use matvecs only)
              - compute b = ∑_j G_{s_j} w_{s_j}
              - solve A x = b (with preconditioning / regularization as needed)
```
### 1.1 Estimate or set $\lambda_{\max}$
- Spectral interval $[0, \lambda_{\max}]$ is used for filter design and Chebyshev scaling (mapping to $[-1,1]$).
- For normalized or random-walk Laplacian set $\lambda_{\max}=2$.
- If unknown, estimate largest eigenvalue using power method or `scipy.sparse.linalg.eigsh(k=1, which='LA')`.

### 1.2 Choose filter design and scales
- Goal: 
  - Specify scaling function $h(\lambda)$ and mother wavelet $g(\lambda)$; 
  - Choose number of scales $J$ and scale values $t_j\in\{t_1, \dots, t_J\}$.
- Recommendations:
  - Scaling function: $h(\lambda) = 1.2 e^{-1} \exp\!\Big(-\Big(\frac{\lambda}{0.4 \lambda_{\min}}\Big)^4\Big)$.
  - Mother wavelet (Mexican-hat): $g(\lambda) = \lambda e^{-\lambda}$
  - Typical J is $3 – 6$.
  - $\lambda_{\min} = \frac{\lambda_{\max}}{K}$
  - Choose scales $t_j$ so that center frequencies $\frac{1}{t_j}$ are log-spaced between
$t_j \in \text{logspace}\big(\log_{10}(1/\lambda_{\max}), \; \log_{10}(1/\lambda_{\min}), \; J\big)$

### 1.3 Build filterbank kernels (functions)
- Output: `List[Callable]` = `[h(·), g_{t_1}(·), ..., g_{t_J}(·)]`
- For each scale $g_{t_j}(\lambda) = g(t_j \lambda)$.

### 2.1 Branch A: Exact spectral filtering (GFT)
- Use for small graphs where eigendecomposition is affordable.
- **Eigendecomposition:** 
  - Compute $\mathbf{Z} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$ (cost $O(N^3)$ 
  for dense graphs, for sparse smaller $N$, `eigsh` computes all eigenpairs but may still be heavy).
- **Build per-scale filter matrices:**
  - For each scale $t_j$:
  $$\mathbf{G}_{j} = \mathbf{U} \, g_{t_j}(\mathbf{\Lambda})\, \mathbf{U}^\top$$
  where $g_{t_j}(\mathbf{\Lambda}) = \operatorname{diag}\big(g_{t_j}(\lambda_1),\dots,g_{t_j}(\lambda_N)\big)$.
- **Analysis (forward GWT):**
  - Per-scale coefficients: 
  $$\mathbf{w}_{j} = \mathbf{G}_{j}^\top \mathbf{x} \in \mathbb{R}^N$$
    (If $\mathbf{G}_{j}$ is symmetric, $\mathbf{w}_{j} = \mathbf{G}_{j} \mathbf{x}$).
  - Stack coefficients of all scales to get the full GWT:
  $$\mathbf{w} = \begin{bmatrix} \mathbf{w}_{1} \\ \vdots \\ \mathbf{w}_{J+1} \end{bmatrix} = \mathcal{G} \mathbf{x} \in \mathbb{R}^{(J+1)N}$$
  - How to compute:
    - Apply the GFT: $\hat{\mathbf{x}} = \mathbf{U}^\top \mathbf{x}$.
    - Compute the filtering in spectral domain: for each $j$ compute $\widehat{\mathbf{y}}_j = g_{t_j}(\Lambda) \, \hat{\mathbf{x}}$.
    - Apply the inverse GFT: $\mathbf{y}_j = \mathbf{U} \, \widehat{\mathbf{y}}_j$ gives $\mathbf{G}_{t_j} \mathbf{x}$,
    if symmetric, this is both filtered signal and coefficients.
- **Synthesis (inverse GWT):**
  - Exact reconstruction:
  $$\mathbf{x} = (\mathcal{G}^\top \mathcal{G})^{-1}\mathcal{G}^\top\mathbf{w} \in \mathbb{R}^N$$
  
### 2.2 Branch B: Chebyshev polynomial approximation
- The original filter $g_{t_j}(\lambda)$ is defined on the spectral axis $[0, \lambda_{\max}]$. 
Applying it exactly would need the graph Fourier transform (GFT), i.e. diagonalize $\mathbf{Z}$ that 
is expensive for large graphs.
- **Observation:** if a function of $\lambda$ can be well approximated by a **polynomial** $p(\lambda)$, 
then $p(\mathbf{Z})$ can be applied in the node domain using repeated multiplications by $\mathbf{Z}$, 
hence no diagonalization is needed.
- **Chebyshev polynomials** are a convenient orthogonal basis on $[-1, 1]$ and give numerically stable, 
fast approximations. 
- Chebyshev polynomials of the first kind:
$$\begin{align}
T_0(x) &= 1 \\
T_1(x) &= x \\
T_k(x) &= 2x T_{k-1}(x) - T_{k-2}(x) \quad &\text{for } k \geq 2
\end{align}
$$
- So we:
  1. Map $[0, \lambda_{\max}]$ to $[-1, 1]$.
  2. Approximate the mapped scalar function by Chebyshev polynomials.
  3. Evaluate the polynomial of the matrix via a stable three-term recurrence that only needs matrix-vector 
  multiplication with $\mathbf{Z}$.
- **Map variable**:
$$\tilde\lambda = \frac{2\lambda}{\lambda_{\max}} - 1,\qquad
\lambda=\frac{\lambda_{\max}}{2}(\tilde\lambda+1)$$
- Define the mapped scalar function for scale $t_j$:
$$\tilde{g}_{t_j}(\tilde{\lambda}):=g_{t_j}(\lambda(\tilde{\lambda}))= g_{t_j}(\frac{\lambda_{\max}}{2}(\tilde\lambda+1))$$
so now $\tilde{g}_{t_j}$ is a function in $[-1, 1]$.
- **Chebyshev expansion**:
$$\tilde{g}(\tilde{\lambda}) \approx \sum_{k=0}^M c_k T_k(\tilde{\lambda})$$
where $T_k$ are the Chebyshev polynomials of the first kind and $c_k$ are 
scalar coefficients (scale index is removed for better readability).
- **Chebyshev coefficients**:
  - Goal is to compute $c_0, c_1, \dots, c_M$ such that $\tilde{g}(\tilde{\lambda}) = \sum_{k=0}^M c_k T_k(\tilde{\lambda})$.
  - Build Chebyshev nodes on $[-1, 1]$ (Clenshaw–Curtis nodes):
  $$\tilde{\lambda}_\ell = \cos \bigl( \frac{(2\ell+1)\pi}{2N_{\text{samples}}} \bigr), \quad \ell=0, \dots, N_{\text{samples}}-1$$
  - Evaluate the mapped function at these nodes:
  $$f_\ell = \tilde{g}(\tilde{\lambda}_\ell) = g(\frac{\lambda_{\max}}{2}(\tilde{\lambda}_\ell+1))$$
  - Use a DCT (discrete cosine transform) on the vector $f_\ell$ to recover the coefficients $c_k$ up to degree $M$.
  - (Standard packages: `scipy.fftpack.dct` / `scipy.fft.dct` or `numpy.polynomial.chebyshev.chebfit`.)
- Apply the polynomial to a vector: the three-term recurrence:
  - Scaled operator (matrix):
  $$\tilde{\mathbf{Z}}=\frac{2}{\lambda_\max} \mathbf{Z} - \mathbf{I}$$
  - This ensures the eigenvalues of $\tilde{\mathbf{Z}}$ lie in $[-1, 1]$.
  - To compute $\mathbf{y} \approx g(\mathbf{Z})\mathbf{z}$:
    1. Initialization: $\mathbf{t}_0 = \mathbf{v}$, $\mathbf{t}_1 = \tilde{\mathbf{Z}} \mathbf{v}$.
    2. Recurrence: $\mathbf{t}_k = 2\tilde{\mathbf{Z}} \mathbf{t}_{k-1} - \mathbf{t}_{k-2}$.
    3. Accumulate: $\mathbf{y} = \sum_{k=0}^{M}c_k \mathbf{t}_k$.
- **Analysis (coefficients) vs synthesis (applying filters to coefficient vectors)**
  - If $\mathbf{Z}$ is symmetric and the scalar kernel is real, then the filter operator $\mathbf{G}=g(\mathbf{Z})$
  is symmetric, and the same `apply_chebyshev` routine computes both:
    - The filtered signal $\mathbf{G}\mathbf{x}$
    - The inner-product coefficients $\mathbf{w}=\mathbf{G}^\top\mathbf{x}=\mathbf{G}\mathbf{x}$.
  - If $\mathbf{Z}$ is not symmetric, we must be explicit:
    - To compute $\mathbf{G}\mathbf{v}$ us `apply_chebyshev(c, Z, v)`.
    - To compute $\mathbf{G}^\top\mathbf{x}$ we need either to:
      1. Run the same procedure on $\tilde{\mathbf{Z}}^\top$ (i.e. fit coefficients for $g$ and apply them to 
      $\tilde{\mathbf{Z}}^\top$).
      2. Symmetrize the operator for design, which often simplifies life.
  - Therefore, for each scale $t_j$:
    - Compute $\mathbf{w}_{t_j}=\mathbf{G}_{t_j}^\top\mathbf{x}$ by `apply_filter(s_j, x)` if symmetric; 
    otherwise compute apply_filter_transpose.
    - Concatenate.
    - No big matrix; only per-scale coefficients stored.
