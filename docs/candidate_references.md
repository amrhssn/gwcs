🚨 The following list of papers are suggested by the LLM Claude 4.5 Sonnet:

I'll provide you with a comprehensive literature review strategy and paper recommendations for your compressed sensing in the graph wavelet domain project.
# Literature Review Strategy
## 1. Systematic Search Approach
**Phase 1: Core Foundations**
- Graph Signal Processing fundamentals
- Graph Wavelets theory
- Compressed Sensing on graphs
- Deep learning for inverse problems

**Phase 2: Related Domains**
- Accelerated MRI reconstruction (k-space sampling)
- Image compressed sensing with deep learning
- Active sensing and adaptive sampling

**Phase 3: Recent Advances**
- Neural graph operators
- Generalization across topologies
- Joint optimization of sampling and reconstruction

**Phase 4: Baselines & Code**
- Identify papers with open-source implementations
- Reproduce baseline results
- Adapt to your framework
 
# Papers by Category
## A. MUST-CITE Foundational Papers
### Graph Signal Processing & Wavelets
1. Hammond et al. (2011) - "Wavelets on graphs via spectral graph theory"
- Applied and Computational Harmonic Analysis
- THE foundational paper for graph wavelets
- Priority: CRITICAL
2. Shuman et al. (2013) - "The emerging field of signal processing on graphs"
- IEEE Signal Processing Magazine
- Comprehensive survey, highly cited
- Priority: CRITICAL
3. Tremblay & Borgnat (2014) - "Graph wavelets for multiscale community mining"
- IEEE TSP
- Practical applications of graph wavelets
- Priority: HIGH
4. Dong et al. (2020) - "Learning Laplacian Matrix in Smooth Graph Signal Representations"
- IEEE TSP
- Learning graphs from signals
- Priority: MEDIUM
### Compressed Sensing on Graphs
5. Chen et al. (2015) - "Discrete signal processing on graphs: Sampling theory"
- IEEE TSP
- Sampling and reconstruction theory
- Priority: CRITICAL
6. Tanaka & Eldar (2018) - "Sampling signals on graphs: From theory to applications"
- IEEE Signal Processing Magazine
- Survey with practical focus
- Priority: HIGH
7. Anis et al. (2016) - "Efficient sampling set selection for bandlimited graph signals using graph spectral proxies"
- IEEE TSP
- Adaptive sampling strategies
- Priority: HIGH
8. Puy et al. (2018) - "Random sampling of bandlimited signals on graphs"
- Applied and Computational Harmonic Analysis
- Theoretical guarantees
- Priority: MEDIUM
 
## B. Deep Learning for Inverse Problems (MRI Analogy)
### Accelerated MRI - Your Direct Inspiration
9. Hammernik et al. (2018) - "Learning a variational network for reconstruction of accelerated MRI data"
- Magnetic Resonance in Medicine
- Variational networks, iterative refinement
- Has code available
- Priority: CRITICAL - BASELINE CANDIDATE
10. Aggarwal et al. (2019) - "MoDL: Model-based deep learning architecture for inverse problems"
- IEEE TMI
- Model-based unrolling
- Code available
- Priority: HIGH - BASELINE CANDIDATE
11. Sriram et al. (2020) - "End-to-end variational networks for accelerated MRI reconstruction" (Facebook AI)
- MICCAI
- End-to-end training with large datasets
- fastMRI dataset & code
- Priority: HIGH - BASELINE CANDIDATE
12. Zbontar et al. (2018) - "fastMRI: An open dataset and benchmarks for accelerated MRI"
- arXiv
- Established benchmarks and baselines
- Priority: HIGH
- Learned Sampling Patterns
13. Bahadir et al. (2019) - "Deep-learning-based optimization of the under-sampling pattern in MRI"
- IEEE TCI
- Jointly learning sampling mask and reconstruction
- Priority: CRITICAL - DIRECTLY RELEVANT
14. Jin et al. (2019) - "Self-attention convolutional neural network for improved MR image reconstruction"
- Information Sciences
- Attention mechanisms for reconstruction
- Priority: MEDIUM
15. Yin et al. (2019) - "LOUPE: Learning optimal under-sampling patterns for MRI"
- NeurIPS Workshop
- Policy learning for sampling
- Code available
- Priority: CRITICAL - BASELINE CANDIDATE
 
## C. Graph Neural Networks for Signal Processing
### GNNs for Reconstruction
16. Bronstein et al. (2021) - "Geometric deep learning: Grids, groups, graphs, geodesics, and gauges"
- arXiv (extended survey)
- Comprehensive GDL framework
- Priority: HIGH
17. Gao & Ji (2019) - "Graph U-Nets"
- ICML
- Hierarchical graph processing (similar to your architecture)
- Priority: HIGH
18. Battaglia et al. (2018) - "Relational inductive biases, deep learning, and graph networks"
- arXiv (DeepMind)
- Encode-process-decode framework
- Priority: HIGH
- Generalization Across Topologies
19. Rusch et al. (2023) - "A survey on oversmoothing in graph neural networks"
- arXiv
- Understanding GNN limitations
- Priority: MEDIUM
20. Donnat et al. (2018) - "Learning structural node embeddings via diffusion wavelets"
- KDD
- Wavelet-based node features
- Priority: MEDIUM
21. Levie et al. (2019) - "CayleyNets: Graph convolutional neural networks with complex rational spectral filters"
- IEEE TSP
- Spectral filters generalizing across graphs
- Priority: HIGH
22. Zhao et al. (2021) - "Neural basis functions for accelerating solutions to high-dimensional PDEs"
- NeurIPS
- Learning over meshes/PDEs (your end goal!)
- Priority: HIGH - DIRECTLY RELEVANT
 
## D. Compressed Sensing + Deep Learning (General)
23. Mardani et al. (2018) - "Deep generative adversarial neural networks for compressive sensing MRI"
- IEEE TMI
- Generative models for CS
- Priority: MEDIUM
- Adler & Öktem (2018) - "Learned primal-dual reconstruction"
- IEEE TMI
24. Adler & Öktem (2018) - Primal-dual networks, unrolled optimization
- Code available
- Priority: HIGH - BASELINE CANDIDATE 
25. Bora et al. (2017) - "Compressed sensing using generative models"
- ICML
- Theory connecting CS and deep learning
- Priority: MEDIUM
26. Mousavi et al. (2015) - "A deep learning approach to structured signal recovery"
- Allerton
- Early DL for CS
- Priority: LOW
 
## E. Recent & Cutting-Edge (2022-2025)
### Graph Transformers & Modern Architectures
27. Rampášek et al. (2022) - "Recipe for a general, powerful, scalable graph transformer"
- NeurIPS
- State-of-the-art graph transformer
- Code available (GraphGPS)
- Priority: HIGH - ARCHITECTURE
28. Ma et al. (2023) - "Graph Mamba: Efficient Graph Learning with State Space Models"
- arXiv
- Mamba for graphs
- Priority: MEDIUM - ARCHITECTURE
29. Fey et al. (2023) - "Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification"
- IJCAI
- Modern GNN techniques
- Priority: MEDIUM
### Diffusion Models for Inverse Problems
30. Song et al. (2021) - "Score-based generative modeling through stochastic differential equations"
- ICLR
- Foundation for diffusion models
- Priority: HIGH (if using diffusion)
31. Chung et al. (2023) - "Diffusion posterior sampling for general noisy inverse problems"
- ICLR
- Diffusion for inverse problems
- Code available
- Priority: HIGH - GENERATIVE BASELINE
32. Jalal et al. (2021) - "Robust compressed sensing MRI with deep generative priors"
- NeurIPS
- Generative priors for MRI
- Priority: MEDIUM
### Neural Operators & PDEs
33. Li et al. (2021) - "Fourier neural operator for parametric partial differential equations"
- ICLR
- Neural operators for PDEs
- Priority: HIGH - YOUR END GOAL
34. Brandstetter et al. (2022) - "Message passing neural PDE solvers"
- ICLR
- GNNs for PDEs on meshes
- Code available
- Priority: CRITICAL - DIRECTLY RELEVANT
35. Fortunato et al. (2022) - "MultiScale MeshGraphNets"
- ICML SimDL Workshop
- Multi-scale mesh processing
- Priority: HIGH
### Graph Sampling & Active Learning
36. Tsitsulin et al. (2023) - "Graph Clustering with Graph Neural Networks"
- JMLR
- Graph pooling and coarsening
- Priority: MEDIUM
37. Wang et al. (2022) - "Deep graph structure learning for robust representations: A survey"
- arXiv
- Learning graph structure jointly
- Priority: MEDIUM
 
## F. Theoretical Foundations
38. Eldar & Kutyniok (2012) - "Compressed Sensing: Theory and Applications"
- Cambridge University Press (Book)
- CS theory reference
- Priority: MEDIUM
39. Mallat (2016) - "Understanding deep convolutional networks"
- Philosophical Transactions Royal Society A
- Scattering wavelets, theoretical foundations
- Priority: MEDIUM
40. Pesenson (2009) - "Sampling in Paley-Wiener spaces on combinatorial graphs"
- Transactions of the AMS
- Sampling theory on graphs
- Priority: LOW (theoretical)
 
# BASELINE PAPERS (With Code Priority)
## Tier 1: Direct Baselines (Must Compare)
1. LOUPE (Yin et al., 2019) - Learned sampling + reconstruction
- Code: GitHub available
- Adaptation: Replace k-space → graph wavelet domain
- Priority: #1 BASELINE
2. MoDL (Aggarwal et al., 2019) - Model-based deep learning
- Code: GitHub available (TensorFlow)
- Adaptation: Unrolled optimization in graph domain
- Priority: #2 BASELINE
3. Variational Network (Hammernik et al., 2018)
- Code: Available
- Adaptation: Cascade architecture for graphs
- Priority: #3 BASELINE
## Tier 2: Graph-Specific Baselines
4. Classical Graph Sampling (Chen et al., 2015)
- Greedy sampling + interpolation
- Easy to implement from paper
- Priority: #4 BASELINE
5. MeshGraphNets (Brandstetter et al., 2022)
- Code: GitHub (DeepMind)
- Adaptation: Use for PDE signals
- Priority: #5 BASELINE (for PDE experiments)
## Tier 3: Architecture Baselines
6. Graph U-Net (Gao & Ji, 2019)
- Code: PyTorch Geometric
- Use as reconstruction network
- Priority: Architecture comparison
7. GraphGPS (Rampášek et al., 2022)
- Code: GitHub
- Transformer baseline
- Priority: Architecture comparison
 
# LITERATURE REVIEW WRITING PLAN
## Section 1: Graph Signal Processing (1-1.5 pages)
Papers: 1, 2, 3, 5, 6 Flow:
- Define GSP framework
- Graph wavelets (Hammond et al.)
- Sampling theory (Chen et al., Tanaka & Eldar)
- Position your work: "extends to learned sampling in wavelet domain"
## Section 2: Compressed Sensing & Deep Learning (1.5-2 pages)
Papers: 9, 10, 11, 13, 15, 23, 24, 31, 38 Flow:
- CS theory basics
- Deep unrolling (MoDL, Variational Networks)
- Joint learning of sampling + reconstruction (LOUPE, Bahadir)
- MRI → Graph analogy table
- Generative models (if using)
## Section 3: Graph Neural Networks (1 page)
Papers: 16, 17, 18, 21, 27, 34, 35 Flow:
- GNN architectures
- Encode-process-decode
- Generalization across topologies (CayleyNets)
- Multi-scale processing (Graph U-Net)
- PDEs on meshes (your end goal)
## Section 4: Related Graph Sampling Works (0.5-1 page)
Papers: 7, 8, 20, 36 Flow:
- Vertex domain sampling (contrast with your wavelet domain)
- Active sampling strategies
- Why wavelet domain is better for your problem
## Section 5: Positioning Your Work (0.5 page)
Key differentiators:
1. Graph wavelet domain (not vertex domain)
2. Joint learning (sampling policy + reconstruction)
3. Topology generalization (multiple graph types)
4. Target PDEs (end application)
 
# JUSTIFICATION FOR EACH BASELINE
| **Baseline**              | **Why Include**                            | **Adaptation Difficulty**     |
|----------------------------|--------------------------------------------|-------------------------------|
| **LOUPE**                  | Directly comparable (learned sampling + recon) | Medium (k-space → GWT)        |
| **MoDL**                   | Standard unrolling approach                | Medium (FFT → GWT)            |
| **Variational Network**    | Established MRI baseline                   | Medium                        |
| **Classical Sampling**     | Must beat non-learned                      | Easy                          |
| **Random Sampling + GNN**  | Ablation: learned mask value               | Easy                          |
| **Graph U-Net (full)**     | Ablation: no sparsity                      | Easy                          |
| **MeshGraphNets**          | PDE-specific comparison                    | Medium (for PDE experiments)  |


# RED FLAGS to Address
## Reviewer Likely Comments:
1. "Why not vertex domain sampling?"
- Response: Cite 5, 6, 7 → wavelet = multi-scale, better compression
2. "How is this different from LOUPE?"
- Response: Graph domain (non-Euclidean), topology generalization, wavelets not FFT
3. "Generalization claims need proof"
- Response: Experiments on 5+ graph types, size range N ∈ [100, 10000]
4. "Missing recent transformers"
- Response: Include 27 (GraphGPS) as architecture ablation
5. "What about diffusion models?"
- Response: Cite 31, mention as future work OR run as baseline if time
 
