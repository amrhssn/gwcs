#!/bin/bash
#SBATCH --job-name=classical_sample48
#SBATCH --output=classical_sample48_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# ── Sample / field / keep-ratio settings ────────────────────────────────────
SAMPLE_INDEX=47    # 0-based → 48th sample
FIELD_INDEX=0
KEEP_RATIO=0.01

# ── Dataset loop (turbulent first, then the rest) ────────────────────────────
for DATASET in turbulent_radiative_layer_2D dynamic_stall viscoelastic_instability kolmogorov; do

    echo ""
    echo "###################################################################"
    echo "  DATASET: $DATASET"
    echo "###################################################################"

    # ── Per-dataset paths ─────────────────────────────────────────────────
    case $DATASET in
        dynamic_stall)
            PROC_BASE=$LOCAL/datasets/dynamic_stall/processed
            RESULTS_DIR=$LOCAL/baseline_results/dynamic_stall
            ;;
        kolmogorov)
            PROC_BASE=$LOCAL/datasets/kolmogorov/processed
            RESULTS_DIR=$LOCAL/baseline_results/kolmogorov
            ;;
        turbulent_radiative_layer_2D)
            PROC_BASE=$LOCAL/datasets/turbulent_radiative_layer_2D/processed
            RESULTS_DIR=$LOCAL/baseline_results/turbulent
            ;;
        viscoelastic_instability)
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

    OUTPUT_DIR=$RESULTS_DIR/plots/classical_sample48
    echo "  Output dir     : $OUTPUT_DIR"

    # ── Run classical sample reconstruction ───────────────────────────────
    echo "  ----------------------------------------------------------------"
    echo "  Classical GSS Sample Reconstruction  ($DATASET, sample index $SAMPLE_INDEX)"
    echo "  ----------------------------------------------------------------"
    python -u plot_classical_sample.py \
        --test-dir      "$PROCESSED_DIR/test" \
        --sample-index  $SAMPLE_INDEX \
        --field-index   $FIELD_INDEX \
        --keep-ratio    $KEEP_RATIO \
        --output-dir    "$OUTPUT_DIR"

    echo "  Done → $OUTPUT_DIR"

done

echo ""
echo "All datasets complete."
echo "Job finished at $(date)"
