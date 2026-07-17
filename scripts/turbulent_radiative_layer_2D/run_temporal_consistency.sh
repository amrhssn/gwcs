#!/bin/bash
#SBATCH --job-name=temporal_gwcs
#SBATCH --output=temporal_gwcs_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu   # EDIT ME: your cluster GPU partition name
#SBATCH --gpus-per-node=1
#SBATCH --account=mx6   # EDIT ME: your cluster account
#SBATCH --qos=standard   # EDIT ME: your cluster QOS
#SBATCH --time=02:00:00
#SBATCH --exclude=n0046,n0025,n0064,n0066

echo "Job started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi

# EDIT ME: absolute path to your own project root
export PROJECT_ROOT=/path/to/your/project_root
export SCRIPT_DIR=$PROJECT_ROOT/graph_wavelet/scripts/turbulent_radiative_layer_2D
export LOCAL=$PROJECT_ROOT/graph_wavelet/local_storage

# ── Environment ─────────────────────────────────────────────────────────────
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
# CUSTOM GWCS CHECKPOINT OVERRIDES
# Set CUSTOM_GWCS_CKPT_<dataset> to use a specific .pt instead
# of auto-detecting the latest checkpoint from the runs folder.
# Leave empty ("") for auto-detection.
#
# Kolmogorov has a specific checkpoint from a completed run
# (saved in a different location from the other datasets).
# ============================================================

CUSTOM_GWCS_CKPT_kolmogorov="$PROJECT_ROOT/graph_wavelet/kolmogorov/396.pt"
CUSTOM_GWCS_RUNDIR_kolmogorov=""   # leave empty → auto-detect run dir

CUSTOM_GWCS_CKPT_turbulent_radiative_layer_2D=""
CUSTOM_GWCS_RUNDIR_turbulent_radiative_layer_2D=""

CUSTOM_GWCS_CKPT_viscoelastic_instability=""
CUSTOM_GWCS_RUNDIR_viscoelastic_instability=""

CUSTOM_GWCS_CKPT_dynamic_stall=""
CUSTOM_GWCS_RUNDIR_dynamic_stall=""

# ── How many test samples to evaluate per dataset ───────────────────────────
# Set to a smaller number (e.g. 100) to speed up a quick run.
# Leave unset ("") to evaluate ALL test samples.
NUM_SAMPLES=""
# NUM_SAMPLES="--num-samples 100"

# ── Dataset loop ─────────────────────────────────────────────────────────────
for DATASET in dynamic_stall turbulent_radiative_layer_2D viscoelastic_instability kolmogorov; do

    echo ""
    echo "###################################################################"
    echo "  DATASET: $DATASET"
    echo "###################################################################"

    # ── Per-dataset paths ─────────────────────────────────────────────────
    case $DATASET in
        dynamic_stall)
            RUNS_BASE=$LOCAL/runs_dynamic_stall
            PROC_BASE=$LOCAL/datasets/dynamic_stall/processed
            RESULTS_DIR=$LOCAL/baseline_results/dynamic_stall
            ;;
        kolmogorov)
            RUNS_BASE=$LOCAL/runs_kolmogorov
            PROC_BASE=$LOCAL/datasets/kolmogorov/processed
            RESULTS_DIR=$LOCAL/baseline_results/kolmogorov
            ;;
        turbulent_radiative_layer_2D)
            RUNS_BASE=$LOCAL/runs_turbulent
            PROC_BASE=$LOCAL/datasets/turbulent_radiative_layer_2D/processed
            RESULTS_DIR=$LOCAL/baseline_results/turbulent
            ;;
        viscoelastic_instability)
            RUNS_BASE=$LOCAL/runs_viscoelastic
            PROC_BASE=$LOCAL/datasets/viscoelastic_instability/processed
            RESULTS_DIR=$LOCAL/baseline_results/viscoelastic
            ;;
    esac

    # ── Processed data dir (latest timestamp subfolder) ───────────────────
    PROCESSED_DIR=$(ls -dt "$PROC_BASE"/[0-9]* 2>/dev/null | head -1)
    if [ -z "$PROCESSED_DIR" ]; then
        echo "  SKIP: no processed data found under $PROC_BASE"
        continue
    fi
    echo "  Processed data : $PROCESSED_DIR"

    # ── Baseline model checkpoints ────────────────────────────────────────
    GXN_CKPT=$RESULTS_DIR/gxn_99k/best_checkpoint.pt
    VGAE_99K_CKPT=$RESULTS_DIR/svgae_99k/best_checkpoint.pt
    VGAE_50K_CKPT=$RESULTS_DIR/svgae_50k/best_checkpoint.pt

    # ── GWCS checkpoint + run dir: auto-detect then apply custom override ──
    GWT_RUN_DIR=$(ls -dt "$RUNS_BASE"/[0-9]* 2>/dev/null | head -1)
    if [ -z "$GWT_RUN_DIR" ]; then
        echo "  SKIP: no GWCS run directory found under $RUNS_BASE"
        continue
    fi
    GWT_CKPT=$(ls "$GWT_RUN_DIR"/checkpoints/*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$GWT_CKPT" ]; then
        echo "  SKIP: no checkpoint found in $GWT_RUN_DIR/checkpoints/"
        continue
    fi

    # Apply per-dataset custom overrides via bash indirect variable lookup
    SAFE_DS="${DATASET//-/_}"
    CUSTOM_CKPT_VAR="CUSTOM_GWCS_CKPT_${SAFE_DS}"
    CUSTOM_RUNDIR_VAR="CUSTOM_GWCS_RUNDIR_${SAFE_DS}"
    CUSTOM_CKPT="${!CUSTOM_CKPT_VAR}"
    CUSTOM_RUNDIR="${!CUSTOM_RUNDIR_VAR}"

    if [ -n "$CUSTOM_CKPT" ]; then
        if [ -f "$CUSTOM_CKPT" ]; then
            GWT_CKPT="$CUSTOM_CKPT"
            echo "  GWCS checkpoint (CUSTOM): $GWT_CKPT"
        else
            echo "  WARNING: Custom checkpoint not found: $CUSTOM_CKPT"
            echo "           Falling back to: $GWT_CKPT"
        fi
    fi
    if [ -n "$CUSTOM_RUNDIR" ]; then
        if [ -d "$CUSTOM_RUNDIR" ]; then
            GWT_RUN_DIR="$CUSTOM_RUNDIR"
            echo "  GWCS run dir   (CUSTOM): $GWT_RUN_DIR"
        else
            echo "  WARNING: Custom run dir not found: $CUSTOM_RUNDIR"
            echo "           Falling back to: $GWT_RUN_DIR"
        fi
    fi

    echo "  GWCS run dir   : $GWT_RUN_DIR"
    echo "  GWCS checkpoint: $GWT_CKPT"

    # ── Build optional --num-samples flag ─────────────────────────────────
    NS_FLAG=""
    if [ -n "$NUM_SAMPLES" ]; then
        NS_FLAG="--num-samples $NUM_SAMPLES"
    fi

    # ── Run temporal consistency analysis ─────────────────────────────────
    echo "  ----------------------------------------------------------------"
    echo "  Temporal Consistency Analysis  ($DATASET)"
    echo "  ----------------------------------------------------------------"
    python -u plot_temporal_consistency.py \
        --test-dir            "$PROCESSED_DIR/test" \
        --gwt-checkpoint      "$GWT_CKPT" \
        --gwt-run-dir         "$GWT_RUN_DIR" \
        --vgae-checkpoint     "$VGAE_99K_CKPT" \
        --vgae-50k-checkpoint "$VGAE_50K_CKPT" \
        --gxn-checkpoint      "$GXN_CKPT" \
        --field-index         0 \
        --keep-ratio          0.01 \
        $NS_FLAG \
        --output-dir          "$RESULTS_DIR/plots/temporal_consistency"

    echo "  Done → $RESULTS_DIR/plots/temporal_consistency/"

done

echo ""
echo "All datasets complete."
echo "Job finished at $(date)"
