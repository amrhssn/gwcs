# Graph Wavelet Compressed Sensing — GWCS-Spatial

Spatial (per-frame) graph-wavelet compressed-sensing reconstruction model, extended to
four PDE datasets: **turbulent_radiative_layer_2D**, **kolmogorov**, **dynamic_stall**,
and **viscoelastic_instability**.

## Installation

### 1. Install PyTorch
Choose based on your hardware:
```bash
# CPU
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# CUDA 12.8
pip3 install torch
# CUDA 12.6 
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2. Install PyTorch Geometric
```bash
pip install torch_geometric
```

### 3. Install other dependencies
```bash
pip install -r requirements.txt
```

## Project structure
```
graph_wavelet_compressed_sensing/
├── baselines/                       # Comparison baselines (FastGSSS, GXN, sparse VGAE,
│                                     #   ConvLSTM/GRU/GraphGRU full-field comparisons)
├── data/                            # Dataset utilities
├── docs/                            # Documentation
├── graph_wavelet_transform/         # GWT implementation
├── models/                          # Neural network models
├── notebooks/                       # Jupyter notebooks
├── processing_steps/                # Data processing
├── scripts/                         # Training/evaluation scripts (per-dataset
│                                     #   pipelines + download_datasets/ for raw data)
├── utils/                           # Utility functions
├── visualization/                   # Plotting utilities
├── run_train_turbulent_99k.sh       # SLURM launcher: turbulent_radiative_layer_2D
├── run_train_kolmogorov_99k.sh      # SLURM launcher: kolmogorov
├── run_train_dynamic_99k.sh         # SLURM launcher: dynamic_stall
├── run_train_viscoelastic_99k.sh    # SLURM launcher: viscoelastic_instability
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

Each `run_train_*_99k.sh` script auto-resumes from the latest run directory if one
exists, otherwise starts a fresh training run via `pipeline.py`. Edit the
`# EDIT ME` markers in these scripts (project root path, SLURM account/QOS/partition,
cluster module names) before use on your own cluster.
