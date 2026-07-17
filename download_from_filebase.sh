#!/bin/bash
#SBATCH --job-name=filebase_download
#SBATCH --partition=general
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=download_%j.out
#SBATCH --error=download_%j.err

# --- 1. Environment Setup ---
# EDIT ME: absolute path to your own project root
PROJECT_ROOT="/path/to/your/project_root"
cd $PROJECT_ROOT

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found at $PROJECT_ROOT/.env"
    exit 1
fi

# Load Conda to access 'aws' CLI
module purge
module load wulver Miniforge3   # EDIT ME: module names for your cluster
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $PROJECT_ROOT/myenv_312_v2

# --- 2. Configuration ---
ENDPOINT="${AWS_ENDPOINT_URL:-https://s3.filebase.com}"
BUCKET="${S3_BUCKET:-graph-wavelet-compressed-sensing}"

# --- 3. Sync Datasets ---
LOCAL_DATA="$PROJECT_ROOT/local_storage/datasets/turbulent_radiative_layer_2D/"
S3_DATA="s3://${BUCKET}/datasets/turbulent_radiative_layer_2D/"

echo "Syncing Datasets from: $S3_DATA"
mkdir -p "$LOCAL_DATA"
aws --endpoint-url "$ENDPOINT" s3 sync "$S3_DATA" "$LOCAL_DATA"

# --- 4. Sync Training Runs ---
LOCAL_RUNS="$PROJECT_ROOT/local_storage/runs_turbulent/"
S3_RUNS="s3://${BUCKET}/runs_turbulent/"

echo "Syncing Runs from: $S3_RUNS"
mkdir -p "$LOCAL_RUNS"
aws --endpoint-url "$ENDPOINT" s3 sync "$S3_RUNS" "$LOCAL_RUNS"

echo "Sync process complete!"