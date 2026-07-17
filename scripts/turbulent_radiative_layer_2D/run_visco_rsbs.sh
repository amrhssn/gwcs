#!/bin/bash
# =============================================================================
#  run_visco_rsbs.sh
#  SLURM job: RSBS on Viscoelastic Instability, sample 48 (index 47)
#  Submit alongside run_visco_fastgsss.sh and run_visco_bsgda.sh for parallel runs.
#  NOTE: RSBS is very fast (~30-60s for N=91,136) — short time limit is fine.
# =============================================================================
#SBATCH --job-name=visco_rsbs
#SBATCH --output=visco_rsbs_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu   # EDIT ME: your cluster GPU partition name
#SBATCH --gpus-per-node=1
#SBATCH --account=mx6   # EDIT ME: your cluster account
#SBATCH --qos=standard   # EDIT ME: your cluster QOS
#SBATCH --time=01:00:00
#SBATCH --exclude=n0046,n0025,n0064,n0066

echo "Job started at $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
nvidia-smi

# ── Thread environment: maximise all CPU parallelism ────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ── Paths ────────────────────────────────────────────────────────────────────
# EDIT ME: absolute path to your own project root
export PROJECT_ROOT=/path/to/your/project_root
export SCRIPT_DIR=$PROJECT_ROOT/graph_wavelet/scripts/turbulent_radiative_layer_2D
export LOCAL=$PROJECT_ROOT/graph_wavelet/local_storage

# ── Environment ──────────────────────────────────────────────────────────────
module purge > /dev/null 2>&1
module load wulver
module load bright
module load GCC/13.3.0
module load CUDA/12.8.0
module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $PROJECT_ROOT/conda_envs/gwtenv

cd $SCRIPT_DIR

# ── Dataset settings ─────────────────────────────────────────────────────────
DATASET=viscoelastic_instability
SAMPLE_INDEX=47      # 0-based → 48th sample
FIELD_INDEX=0
KEEP_RATIO=0.01
METHOD=rsbs

PROC_BASE=$LOCAL/datasets/viscoelastic_instability/processed
RESULTS_DIR=$LOCAL/baseline_results/viscoelastic

# ── Resolve latest processed data folder ─────────────────────────────────────
PROCESSED_DIR=$(ls -dt "$PROC_BASE"/[0-9]* 2>/dev/null | head -1)
if [ -z "$PROCESSED_DIR" ]; then
    echo "ERROR: no processed data found under $PROC_BASE"
    exit 1
fi

OUTPUT_DIR=$RESULTS_DIR/plots/classical_sample48
echo ""
echo "Dataset      : $DATASET"
echo "Method       : $METHOD"
echo "Processed dir: $PROCESSED_DIR"
echo "Output dir   : $OUTPUT_DIR"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────
python -u plot_classical_sample_single.py \
    --test-dir      "$PROCESSED_DIR/test" \
    --method        $METHOD \
    --sample-index  $SAMPLE_INDEX \
    --field-index   $FIELD_INDEX \
    --keep-ratio    $KEEP_RATIO \
    --output-dir    "$OUTPUT_DIR"

echo ""
echo "Done → $OUTPUT_DIR"
echo "Job finished at $(date)"
