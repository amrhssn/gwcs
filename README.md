# Changes since previous repo:

Ran all four datasets via `run_train_turbulent_99k.sh`, `run_train_kolmogorov_99k.sh`, `run_train_dynamic_99k.sh`, and `run_train_viscoelastic_99k.sh` (runs across multigpu torchrun).

**Dataset accommodation** — replaced hardcoded turbulent-only field names and config
  checks (`data/dataset_turbulent.py`, `data/utils_io.py`, `models/reconstruction.py`)
  with dynamic lookups so kolmogorov, dynamic_stall, and viscoelastic_instability work
  correctly.
**Optimization**
- added GPU-batched Chebyshev preprocessing (`preprocess.py`), replacing one-sample-at-a-time CPU threading.
- rewrote normalization stats as a single-pass Chan's algorithm (`normalization_turbulent.py`), halving I/O and avoiding NCCL timeouts during DDP.
- added optional float16 storage for processed samples, roughly halving disk usage.
- added chunked WellDataset loading to avoid OOM on large runs.
