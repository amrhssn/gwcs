# Multilevel Graph Wavelet Compressed Sensing with Scale-Aware Neural Recovery

## Setup

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

For CUDA machine:
```bash
uv sync --extra gpu
```
For CPU machine:
```bash
uv sync --extra cpu
```

## Training GWCS

### Approximately Band-limited (ABL)

ABL data is synthetic and generated automatically on first run.

```bash
uv run python scripts/train_abl_gwcs.py
```

### Turbulent radiative layer 2D

Download the raw data first (from Hugging Face):

```bash
uv run python scripts/download_raw_data.py --dataset turbulent_radiative_layer_2D
```

Then train:

```bash
uv run python scripts/train_turbulent_gwcs.py
```

### Outputs

```
datasets/
├── ABL/                              # generated ABL data
└── turbulent_radiative_layer_2D/
    ├── raw/                          # downloaded HDF5 files
    └── processed/                    # preprocessed graphs

runs/
└── ABL_GWCS/<run_id>/                # example; turbulent_GWCS/<run_id>/ has the same layout
    ├── config.json                   # run metadata
    ├── train.log
    ├── checkpoint_best.pt
    ├── checkpoint_last.pt
    ├── history.json                  # per-epoch train/val loss
    ├── summary.json                  # final test metrics
    └── figs/                         # reconstruction figures
```