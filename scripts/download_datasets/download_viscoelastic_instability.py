"""
Download viscoelastic_instability dataset from Hugging Face Hub
"""
from pathlib import Path
from huggingface_hub import snapshot_download

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = PROJECT_ROOT / "local_storage" / "datasets" / "viscoelastic_instability"

print("=" * 80)
print("DOWNLOADING VISCOELASTIC INSTABILITY DATASET")
print("=" * 80)
print(f"Target directory: {TARGET_DIR}")

# Create target directory
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Download from Hugging Face
snapshot_download(
    repo_id="polymathic-ai/viscoelastic_instability",
    repo_type="dataset",
    local_dir=str(TARGET_DIR),
    allow_patterns=["data/train/*.hdf5", "data/valid/*.hdf5", "data/test/*.hdf5"],
)

print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print(f"Downloaded to: {TARGET_DIR}")
print(f"Total size: {sum(f.stat().st_size for f in TARGET_DIR.rglob('*') if f.is_file()) / 1e9:.2f} GB")
print("=" * 80)