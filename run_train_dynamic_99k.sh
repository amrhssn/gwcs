#!/bin/bash
#SBATCH --job-name=nigwt_dynamic_99k
#SBATCH --output=%x.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=gpu   # EDIT ME: your cluster GPU partition name
#SBATCH --gpus-per-node=4
#SBATCH --account=mx6   # EDIT ME: your cluster account
#SBATCH --qos=standard   # EDIT ME: your cluster QOS
#SBATCH --time=48:00:00
#SBATCH --exclude=n0046,n0025,n0064,n0066

echo "Job started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi

# EDIT ME: absolute path to your own project root
export PROJECT_ROOT=/path/to/your/project_root
export SCRIPT_DIR=$PROJECT_ROOT/graph_wavelet/scripts/turbulent_radiative_layer_2D
export LOCAL=$PROJECT_ROOT/graph_wavelet/local_storage

module purge > /dev/null 2>&1
module load wulver
module load bright
module load GCC/13.3.0
module load CUDA/12.8.0
module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $PROJECT_ROOT/conda_envs/gwtenv

cd $SCRIPT_DIR

# ============================================================
# NIGWT ~99K params  (hidden_dim=96, num_layers=3)
# Dataset: Dynamic Stall
#
# Auto-resume: detects latest run dir and resumes via torchrun
# directly (bypasses pipeline.py's filebase_sync check which
# fails to find local checkpoints).
# ============================================================

RUNS_BASE=$LOCAL/runs_dynamic_stall
LATEST_RUN=$(ls -dt "$RUNS_BASE"/[0-9]* 2>/dev/null | head -1)

if [ -n "$LATEST_RUN" ]; then
    LATEST_TIMESTAMP=$(basename "$LATEST_RUN" | grep -oP '^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')
    if [ -n "$LATEST_TIMESTAMP" ]; then
        echo "Resuming from timestamp: $LATEST_TIMESTAMP  ($LATEST_RUN)"
        torchrun --nproc_per_node=4 train.py \
            --dataset dynamic_stall \
            --use-normalization \
            --hidden-dim 96 \
            --num-layers 3 \
            --resume "$LATEST_TIMESTAMP"
    else
        echo "Could not parse timestamp from: $(basename $LATEST_RUN) — starting fresh"
        python -u pipeline.py \
            --dataset dynamic_stall \
            --skip-download \
            --skip-preprocessing \
            --num-gpus 4 \
            --use-normalization \
            --hidden-dim 96 \
            --num-layers 3
    fi
else
    echo "No previous run found under $RUNS_BASE — starting fresh"
    python -u pipeline.py \
        --dataset dynamic_stall \
        --skip-download \
        --skip-preprocessing \
        --num-gpus 4 \
        --use-normalization \
        --hidden-dim 96 \
        --num-layers 3
fi

echo "Job finished at $(date)"
