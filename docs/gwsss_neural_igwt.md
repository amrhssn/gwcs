# 1. Neural Inverse Graph Wavelet Transform

## 1.1 Problem Setup

### Data
- $\mathbf{x} \in \mathbb{R}^N$ is the graph signal over the graph 
$\mathcal{G}=(\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ is the set of $N$ nodes, 
and $\mathcal{E}$ is the set of edges. 
The spectral decomposition of graph Laplacian is $\mathbf{L}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top \in \mathbb{R}^{N \times N}$
where $\mathbf{\Lambda}=\text{diag}(\lambda_0, \dots, \lambda_{N-1})$ is the diagonal matrix of eigenvalues and 
$\mathbf{U} = [\mathbf{u}_0, \dots, \mathbf{u}_{N-1}] \in \mathbb{R}^{N \times N}$ is the matrix of corresponding eigenvectors.
- $\mathbf{G} = \big[ \mathbf{H}^\top \; \mathbf{G}_1^\top \; \dots \; \mathbf{G}_J^\top \big]^\top \in \mathbb{R}^{(J+1)N \times N}$ is the graph wavelet transform (GWT) analysis matrix, 
and $J + 1$ is the number of filters 
($J$ bandpass filters $\{\mathbf{G}_k= \mathbf{U}g_k(\mathbf{\Lambda})\mathbf{U}^\top\in \mathbb{R}^{N \times N}\}_{k=1}^J$, 
and a single lowpass filter $\mathbf{H}=\mathbf{U}h(\mathbf{\Lambda})\mathbf{U}^\top \in \mathbb{R}^{N \times N}$, 
where $g_k$ is the bandpass filtering kernel at scale $k$ and $h$ is lowpass filter kernel).
- $\mathbf{w}=\mathbf{G}\mathbf{x} \in \mathbb{R}^{(J+1)N}$ is the signal's GWT coefficients.
- $S \subset \{0, 1, \dots, (J+1)N-1\}$ is the index set of observed GWT coefficients, 
and $\mathbf{s} \in \{0, 1\}^{(J+1)N}$ is its corresponding sparse mask such that $\mathbf{s}[i] = 1$ if 
$i \in S$ else $\mathbf{s}[i] = 0$.
- $\mathbf{w}_S = \mathbf{w} \odot \mathbf{s} $ is the observed GWT coefficients, 
where the subscript $_S$ denotes evaluation at the index set $S$ 
(equivalently, element-wise multiplication with the mask $\mathbf{s}$).
- Each sample of our training data is $(\mathbf{x}, \mathbf{s})$.

### Model
- $f_\theta: \mathbb{R}^{2(J+1)N} \rightarrow \mathbb{R}^N$ is the neural network model with parameters set $\theta$, 
that maps the observed GWT coefficients and the observation mask to the reconstructed signal.

### Objective:

Per-sample loss:
$$
\begin{equation}
    \theta^{*} = \arg \min_{\theta} \; 
    \big\| \mathbf{x} - f_\theta(\mathbf{w}_S, \mathbf{s}) \big\|_2^{2} + 
    \gamma \big\| \mathbf{w}_S - \big(\mathbf{G}f_\theta(\mathbf{w}_S, \mathbf{s})\big)_S \big\|_2^{2}
\end{equation}
$$
- First term is the reconstruction loss.
- Second term is the coefficient consistency loss that enforces that the reconstructed signal's 
GWT coefficients match the observed coefficients on their support.
- $\gamma \in \mathbb{R}_{\geq 0}$ is a scalar controlling the relative importance of coefficient consistency loss.
- $\hat{\mathbf{x}} = f_\theta(\mathbf{w}_S, \mathbf{s})$ is the reconstruction of the signal.
- $\hat{\mathbf{w}}_S=\big(\mathbf{G}f_\theta(\mathbf{w}_S, \mathbf{s})\big)_S$ is the GWT coefficients of the
reconstructed signal that is evaluated at the index set $S$.

## 1.2 Classical IGWT
We have:
- $\mathbf{x}=\mathbf{G}^\dagger \mathbf{w}=(\mathbf{G}^\top\mathbf{G})^{-1}\mathbf{G}^\top\mathbf{w}$
- $\mathbf{G}^\top = \big[ \mathbf{G}_0^\top, \mathbf{G}_1^\top, \dots \mathbf{G}_J^\top \big]$ 
where $\mathbf{G}_0 :=\mathbf{H}$
- $\mathbf{G}^\top \mathbf{w}= \sum_{k=0}^{J} \mathbf{G}_k^\top \mathbf{w}_k$, 
where $\mathbf{w}=[\mathbf{w}_0^\top, \mathbf{w}_1^\top, \dots, \mathbf{w}_k^\top]^\top$, $\mathbf{w}_k \in \mathbb{R}^N$.
- $\mathbf{G}^\top \mathbf{G}=\sum_{k=0}^{J} \mathbf{G}_k \mathbf{G}_k^\top := \mathbf{K} \in \mathbb{R}^{N\times N}$.

Hence, we the reconstructed graph signal is $\mathbf{x} = \mathbf{K}^{-1}\sum_{k=0}^{J} \mathbf{G}_k^\top \mathbf{w}_k$:
- $\mathbf{G}_k^\top \mathbf{w}_k$ is the per-scale synthesis operator, i.e., sum of wavelet atoms weighted by 
coefficients at the scale $k$.
- $\sum_{k=0}^{J} \mathbf{G}_k^\top \mathbf{w}_k$ is the naive synthesis that simply superposes scale-wise 
reconstructions.
- $\mathbf{K}^{-1}=(\mathbf{G}^\top \mathbf{G})^{-1}$ is the global correction term that 
enforces the optimal least-squares reconstruction given all atoms.

**Conclusion**. Classical synthesis is two-stage:
1. **Local** synthesis per scale (sum of atoms) and aggregation $\rightarrow$ $\sum_{k=0}^{J} \mathbf{G}_k^\top \mathbf{w}_k$
2. **Global** correction via applying $\mathbf{K}^{-1}$ to aggregated results $\rightarrow$ $\mathbf{K}^{-1}\sum_{k=0}^{J} \mathbf{G}_k^\top \mathbf{w}_k$

## 1.3 Designing Neural IGWT
We want to model $\mathbf{x} = \mathbf{K}^{-1}\sum_{k=0}^{J} \mathbf{G}_k^\top \mathbf{w}_k$ using 
a graph neural network (GNN) within the Encode-Process-Decode framework.

### (A) Baseline GNN
#### Input
- $\mathbf{W} \in \mathbb{R}^{N \times (J+1)}$: All GWT coefficients
- $\mathbf{S} \in \{0, 1\}^{N \times (J+1)}$: Sparsification mask
- $\mathbf{A} \in \mathbb{R}^{N \times N}$: Adjacency matrix

#### Encoder
- $\mathbf{W}_S=\mathbf{W} \odot \mathbf{S}$
- $\mathbf{Z}=[\mathbf{W}_S, \mathbf{S}] \in \mathbb{R}^{N \times 2(J+1)}$: Concatenated inputs.
- $\mathbf{Z}_0 = \text{MLP/Linear}(\mathbf{Z})$: Encoded inputs $\mathbf{Z}_0 \in \mathbb{R}^{N \times D}$.
- **[Optional]** Define learnable scale embeddings: $\mathbf{J} \in \mathbb{R}^{(J+1) \times D_{\text{scale}}}$.
Compute the per-node scale embedding $\mathbf{E} \in \mathbb{R}^{N \times D_{\text{scale}}}$ 
where $\mathbf{E}_i = \sum_{k=0}^J \mathbf{W}_{S, i, k} \mathbf{J}_{k}$.
Concatenate to encoded inputs: $\mathbf{Z}_0 = \text{CONCAT}(\text{MLP/Linear}(\mathbf{Z}), \mathbf{E})$


#### Processor
- A sequence of $L$ message-passing layers with skip-connections:
$$
\mathbf{Z}_{\ell+1} = \mathbf{Z}_\ell + \text{MPNN}(\mathbf{Z}_\ell , \mathbf{A}), \quad \ell=0, \dots, L-1
$$

#### Decoder
- $\hat{\mathbf{x}} = \text{MLP/Linear}(\mathbf{Z}_L) \in \mathbb{R}^N$: Reconstructed signal 

### (B) Scale-aware GNN
#### Input
- $\mathbf{w}_k \in \mathbb{R}^{N \times 1}, \quad k=0, 1, \dots, J$: Per-scale GWT coefficients
- $\mathbf{s}_k \in \mathbb{R}^{N \times 1}, \quad k=0, 1, \dots, J$: Per-scale sparsification map
- $\mathbf{A} \in \mathbb{R}^{N \times N}$: Adjacency matrix

#### Encoder
- $\mathbf{w}_k$ is the sparse observed coefficients.
- $\mathbf{Z}^{(k)} = \big[ \mathbf{w}_k, \mathbf{s}_k \big] \in \mathbb{R}^{R \times 2}, \quad k=0, 1, \dots, J$: 
Per-scale concatenated inputs
- $\mathbf{Z}^{(k)}_0 = \text{MLP}_k/\text{Linear}_k(\mathbf{Z}^{(k)})$: 
Per-scale encoded inputs $\mathbf{Z}^{(k)}_0 \in \mathbb{R}^{N \times D}$.
- **[Optional]** Define learnable scale embeddings: $\mathbf{J} \in \mathbb{R}^{(J+1) \times D_{\text{scale}}}$.
Compute the per-node per-scale scale embedding $\mathbf{E}^{(k)} \in \mathbb{R}^{N \times D_{\text{scale}}}$ 
where $\mathbf{E}^{(k)}_i = \mathbf{w}_{k}[i] \mathbf{J}_{k}$.
Concatenate to encoded inputs: $\mathbf{Z}^{(k)}_0 = \text{CONCAT}(\text{MLP/Linear}(\mathbf{Z}^{(k)}), \mathbf{E}^{(k)})$

#### Processor
- **Per-scale processor.** For scale $k$, run $L_k$ message-passing layers 
(to set receptive field proportional to scale).
$$
L_k = L_{\min} + \text{layer-factor} \times k 
$$
$$
\mathbf{Z}^{(k)}_{\ell+1}= \mathbf{Z}^{(k)}_{\ell} + \text{MPNN}(\mathbf{Z}^{(k)}_{\ell}, \mathbf{A}), 
\quad \ell= 0, \dots, L_k-1, \; \; k=0, \dots, J
$$
- **Fusion across scales**: 
  - [Option 1] : Concatenate 
  $\mathbf{Z}_0 = \text{CONCAT}(\mathbf{Z}^{(0)}_{L_0}, \mathbf{Z}^{(1)}_{L_1}, \dots, \mathbf{Z}^{(J)}_{L_J}) \in \mathbb{R}^{N \times (J+1)D}$.
  Apply a MLP/Linear layer to fuse across scales: $\mathbf{Y}_0 = \text{MLP/Linear}(\mathbf{Z}_0)$ with $\mathbf{Y}_0 \in \mathbb{R}^{N \times D}$.
  - [Option 2] : Per-node attention weights over scales
  $\mathbf{Y}_0[i, :]=\sum_{k=0}^{J} \mathbf{a}_i[k] \mathbf{Z}^{(k)}_{L_k}[i, :]$:
  - Initialize a learnable _missing-coefficient_ embedding vector:
  $$\mathbf{m} \in \mathbb{R}^D$$
  - Stack all the per-scale processed encodings:
  $$\mathbf{U} = \text{STACK}(\mathbf{Z}^{(0)}_{L_0}, \mathbf{Z}^{(1)}_{L_1}, \dots, \mathbf{Z}^{(J)}_{L_J}) \in \mathbb{R}^{N \times (J+1) \times D}$$
  - Set $\mathbf{U}[i, k] \leftarrow \mathbf{m}$ if $\mathbf{S}[i, k]=0$, 
  so that the network learns how to treat “missing” values.
  - Concatenate the mask so the attention is aware of the existing coefficients:
  $$\mathbf{U} = \text{CONCAT}(\mathbf{U}, \mathbf{S}) \in \mathbb{R}^{N \times (J+1) \times D+1}$$
  - Attention projection layer:
  $$\mathbf{V} = \tanh \big( \text{Linear}(\mathbf{U}) \big) \in \mathbb{R}^{N \times (J+1) \times D_a}$$
  - Score collapse:
  $$\mathbf{T} = \text{Linear}(\mathbf{V}) \in \mathbb{R}^{N \times (J+1) \times 1}$$
  - Softmax across scales to get the attention scores:
  $$\mathbf{a}_i = \text{Softmax} \big( \mathbf{T}[i] \big)$$
- **Global processor:** Apply a sequence of message-passing layers with skip-connections to mix cross-node information:
$$
\mathbf{Y}_{\ell+1} = \mathbf{Y}_\ell + \text{MP}(\mathbf{Y}_\ell , \mathbf{A}), \quad \ell=0, \dots, L-1
$$
  
#### Decoder
- $\hat{\mathbf{x}} = \text{MLP/Linear}(\mathbf{Y}_L) \in \mathbb{R}^N$: Reconstructed signal 

### (C) Graph U-Net encoder-decoder

#### Input
- $\mathbf{W} \in \mathbb{R}^{N \times (J+1)}$: All GWT coefficients
- $\mathbf{S} \in \{0, 1\}^{N \times (J+1)}$: Sparsification mask
- $\mathbf{A} \in \mathbb{R}^{N \times N}$: Adjacency matrix

#### Encoder
- $\mathbf{W}_S=\mathbf{W} \odot \mathbf{S}$
- $\mathbf{Z}=[\mathbf{W}_S, \mathbf{S}] \in \mathbb{R}^{N \times 2(J+1)}$: Concatenated inputs.
- $\mathbf{Z}_0 = \text{MLP/Linear}(\mathbf{Z})$: Encoded inputs $\mathbf{Z}_0 \in \mathbb{R}^{N \times D}$.
- **[Optional]** Define learnable scale embeddings: $\mathbf{J} \in \mathbb{R}^{(J+1) \times D_{\text{scale}}}$.
Compute the per-node scale embedding $\mathbf{E} \in \mathbb{R}^{N \times D_{\text{scale}}}$ 
where $\mathbf{E}_i = \sum_{k=0}^J \mathbf{W}_{S, i, k} \mathbf{J}_{k}$.
Concatenate to encoded inputs: $\mathbf{Z}_0 = \text{CONCAT}(\text{MLP/Linear}(\mathbf{Z}), \mathbf{E})$

#### Processor: Graph U-Net
- Graph pooling layers $\text{Graph-Pool}_l$ with downsample ratios $\alpha$ for each level $l=0, \dots, J-1$
  corresponding to the number of scale transitions ($l=0$ corresponds to the finest scale and $l=J-1$ corresponds to 
  the coarsest scale).
- **Down-path:** 
  - $\mathbf{Z}_0^{(0)}:= \mathbf{Z}_0$ and $\mathbf{A}^{(0)} := \mathbf{A}$.
  - For $\quad l=0, \dots, J - 1$: 
    - (1) Message-passing on initial node features of the level:
    $$
    \mathbf{Z}_{\ell + 1}^{(l)} = \mathbf{Z}_{\ell}^{(l)} + \text{MPNN}^{(l)}(\mathbf{Z}_{\ell}^{(l)}, \mathbf{A}^{(l)}),
    \quad \ell = 0, \dots L-1
    $$
    - (2) Graph pooling:
    $$
    \mathbf{Z}_{0}^{(l+1)}, \mathbf{A}^{(l+1)} = \text{Graph-Pool}_l(\mathbf{Z}_{L}^{(l)}, \mathbf{A}^{(l)})
    $$
- **Bottleneck:**
  - Message passing on the final level:
      $$
      \mathbf{Z}_{\ell + 1}^{(J)} = \mathbf{Z}_{\ell}^{(J)} + \text{MPNN}^{(J)}(\mathbf{Z}_{\ell}^{(J)}, \mathbf{A}^{(J)}),
      \quad \ell = 0, \dots L-1
      $$
- **Up-path:**
  - For $l = J-1, \dots, 0$:
    - Unpool the node features:
        $$
        \hat{\mathbf{Z}}^{(l)} = \text{Graph-Unpool}_l(\mathbf{Z}_{L}^{l+1}), 
        $$
    - Fuse the unpooled node features with the skip-connection from the corresponding donw-path layer:
        $$
        \tilde{\mathbf{Z}}^{(l)}_0 = \text{MLP}_l\text{\Linear}_l\big(\text{CONCAT}(\mathbf{Z}^{(l)}, \hat{\mathbf{Z}}^{(l)})\big)
        $$
    - Perform message passed on the fused node features:
        $$
        \tilde{\mathbf{Z}}^{(l)}_{\ell +1} = \tilde{\mathbf{Z}}^{(l)}_{\ell} + \text{MPNN}^{(l)}(\tilde{\mathbf{Z}}^{(l)}_\ell, \mathbf{A}^{(J)}),
        \quad \ell = 0, \dots L-1
        $$
#### Decoder
- $\hat{\mathbf{x}} = \text{MLP/Linear}(\tilde{\mathbf{Z}}^{(0)}_L) \in \mathbb{R}^N$: Reconstructed signal 
### (D)


-------
# 2. Graph Wavelet Coefficients Sampling Set Selection

## (A) Optimizing for the optimal GWCs given a trained Neural IGWT
## (B) Jointly optimize GWC-SSS and Neural IGWT 
$$
\mathbf{s}^* = \arg\min_{\mathbf{s} \in \{0,1\}^{  N}} \frac{1}{M}\sum_{i=1}^M \left[ \|\mathbf{x}_i - f_{\theta^*}\|_2^2 + \gamma \|\mathbf{w}_{i,S} - _S\|_2^2 \right] \text{subject to: } \|\mathbf{s}\|_1 = K
$$
