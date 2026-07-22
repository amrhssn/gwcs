import argparse
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_REPO_IDS = {
  "turbulent_radiative_layer_2D": "polymathic-ai/turbulent_radiative_layer_2D",
}

Split = Literal["train", "valid", "test"]


def _split_allow_patterns(repo_id: str, split: Split, limit_files: int | None) -> list[str]:
  if limit_files is None:
    return [f"data/{split}/*.hdf5"]
  api = HfApi()
  split_files = sorted(f for f in api.list_repo_files(repo_id, repo_type="dataset")
                       if f.startswith(f"data/{split}/") and f.endswith(".hdf5"))
  return split_files[:limit_files]


def download_dataset(dataset: str, splits: list[Split], limit_files_per_split: int | None, force: bool) -> None:
  repo_id = DATASET_REPO_IDS[dataset]
  raw_dir = REPO_ROOT / "datasets" / dataset / "raw"

  for split in splits:
    split_dir = raw_dir / "data" / split
    if not force and split_dir.exists() and any(split_dir.glob("*.hdf5")):
      print(f"[{dataset}/{split}] already has raw data at {split_dir}; skipping (use --force to re-run)")
      continue

    allow_patterns = _split_allow_patterns(repo_id, split, limit_files_per_split)
    print(f"[{dataset}/{split}] downloading {len(allow_patterns)} file pattern(s) from {repo_id} into {split_dir}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(raw_dir), allow_patterns=allow_patterns)
    size_bytes = sum(f.stat().st_size for f in split_dir.glob("*.hdf5"))
    print(f"[{dataset}/{split}] done ({size_bytes / 1e6:.1f} MB)")


def main() -> None:
  parser = argparse.ArgumentParser(description="Download raw dataset files from Hugging Face Hub")
  parser.add_argument("--dataset", choices=sorted(DATASET_REPO_IDS), nargs="+", default=sorted(DATASET_REPO_IDS))
  parser.add_argument("--splits", choices=["train", "valid", "test"], nargs="+",
                      default=["train", "valid", "test"])
  parser.add_argument("--limit-files-per-split", type=int, default=None,
                      help="Only fetch the first N files of each split (for smoke-testing).")
  parser.add_argument("--force", action="store_true",
                      help="Re-run download even if the split directory already has raw files.")
  args = parser.parse_args()

  for dataset in args.dataset:
    download_dataset(dataset=dataset, splits=args.splits, limit_files_per_split=args.limit_files_per_split,
                     force=args.force)


if __name__ == "__main__":
  main()
