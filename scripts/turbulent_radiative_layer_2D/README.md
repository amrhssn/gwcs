# Turbulent Radiative Layer 2D - Pipeline Automation

This directory contains scripts for the complete data-to-model pipeline for the Turbulent Radiative Layer 2D dataset.

## Quick Start

### For Non-Technical Users

The entire pipeline can be run with a single command:

```bash
cd mgws/scripts/turbulent_radiative_layer_2D
python run_pipeline.py --test-mode
```

This will:
1. Download a small sample of the dataset (first file only)
2. Process 10 samples with GWT coefficients
3. Train a model on 10 samples

**Expected time:** ~5-10 minutes (depending on your hardware)

---

## Usage

### Full Production Pipeline

```bash
# Run the complete pipeline (downloads all data, processes everything)
python run_pipeline.py
```

⏱️ **Time:** Several hours (depending on dataset size and hardware)

### Test Mode (Recommended First)

```bash
# Quick validation run with limited samples
python run_pipeline.py --test-mode
```

⏱️ **Time:** ~5-10 minutes

### Skip Steps (If Data Already Exists)

```bash
# Skip download if data already exists
python run_pipeline.py --skip-download

# Skip download and preprocessing, only train
python run_pipeline.py --skip-download --skip-preprocessing

# Only download and preprocess (skip training)
python run_pipeline.py --skip-training
```

### Advanced Options

```bash
# Use more workers for faster preprocessing
python run_pipeline.py --workers 16

# Get help
python run_pipeline.py --help
```

---

## Pipeline Steps

The pipeline consists of three sequential steps:

### 1. **Data Collection** (`data_collection_step.py`)
- Downloads data from "The Well" dataset repository
- Output: Raw HDF5 files in `local_storage/datasets/turbulent_radiative_layer_2D/data/[train|valid|test]`

### 2. **Data Preprocessing** (`data_preprocessing_step.py`)
- Computes Graph Wavelet Transform (GWT) coefficients
- Creates graph structure from spatial grid
- Output: Processed `.pt` files in `local_storage/datasets/turbulent_radiative_layer_2D/processed/YYYYMMDD_HHMMSS/[train|valid|test]`

### 3. **Model Training** (`model_training.py`)
- Trains a graph neural network for signal reconstruction
- Uses GWT coefficients as input features
- Output: Model checkpoints and metrics in `local_storage/runs_turbulent/YYYY-MM-DD_HH-MM-SS_turbulent_density/`

---

## Output Paths

After running the pipeline, you'll see a summary like this:

```
OUTPUT PATHS
────────────────────────────────────────────────────────────────────────────────

📁 Raw Data (Downloaded):
   /path/to/local_storage/datasets/turbulent_radiative_layer_2D
   └── data/train/ (X files)
   └── data/valid/ (X files)
   └── data/test/ (X files)

📁 Processed Data (GWT Coefficients):
   /path/to/local_storage/datasets/turbulent_radiative_layer_2D/processed/YYYYMMDD_HHMMSS
   └── train/ (X samples)
       ├── shared_data.pt
       ├── chebyshev_synthesis_data.pt
       ├── graph_config.json
       ├── signal_config.json
       └── gwt_config.json
   └── valid/ (...)
   └── test/ (...)

📁 Training Run (Model Artifacts):
   /path/to/local_storage/runs_turbulent/YYYY-MM-DD_HH-MM-SS_turbulent_density
   └── checkpoints/ (X checkpoint(s))
   └── reconstructor_pretraining_config_turbulent.yaml
   └── dataset_info.json
   └── train_loss.json
   └── val_loss.json
   └── test_metrics.json
```

---

## Test Mode vs Production Mode

| Step | Test Mode | Production Mode |
|------|-----------|-----------------|
| **Data Collection** | Downloads only first file per split | Downloads all files |
| **Preprocessing** | Processes 10 samples | Processes all samples |
| **Training** | Uses 10 samples | Uses all samples |
| **Time** | ~5-10 minutes | Several hours |

---

## Requirements

Before running the pipeline, ensure:

1. **Environment file exists**: `.env` file in project root with:
   ```
   TURBULENT_DATA_BASE_PATH=local_storage/datasets/turbulent_radiative_layer_2D
   ```

2. **Required packages installed**:
   - `torch`
   - `torch_geometric`
   - `the-well` (for data download)
   - Other dependencies in `requirements.txt`

3. **Sufficient disk space**:
   - Test mode: ~500 MB
   - Full pipeline: ~50+ GB (depends on dataset size)

---

## Troubleshooting

### "Environment validation failed"
- Make sure `.env` file exists in project root
- Check that all required packages are installed: `pip install -r requirements.txt`

### "Data collection failed"
- Check internet connection
- Verify you have access to "The Well" dataset repository
- Try running with `--test-mode` first

### "Out of memory" during training
- Reduce batch size in `model_training.py` (line 392: `batch_size=100`)
- Use `--test-mode` to train on fewer samples
- Close other applications to free up GPU/RAM

---

## Running Individual Steps

You can also run each step independently:

```bash
# Step 1: Download data
python data_collection_step.py --first-only

# Step 2: Preprocess data
python data_preprocessing_step.py --num-samples 10 --workers 8

# Step 3: Train model
python model_training.py
```

---

## Questions?

If you encounter issues or have questions, please check:
1. The error messages in the terminal output
2. The `.env` file configuration
3. That all required packages are installed

For technical support, contact the development team or open an issue in the project repository.
