# -*- coding: utf-8 -*-
"""
Classical graph signal sampling baselines.
Evaluates FastGSSS, BSGDA, RSBS at a given keep_ratio (sample rate).
Classical methods work on normalized signals for fair comparison.

Speed improvements:
  - Graph objects (G, adj, L_comb, L_sym) built ONCE, reused for all samples.
  - Use --num-samples to cap evaluation count for quick runs.
  - Per-sample evaluation parallelized with ProcessPoolExecutor.
  - All three methods evaluated per sample in the same worker process.

Resume / checkpointing:
  - Each completed sample is checkpointed to {output_dir}/ckpt_{idx}.json
    immediately after it finishes — no progress is lost if the job is killed.
  - Visualizations (first 8 samples) are saved to
    {output_dir}/ckpt_{idx}_recon.pt and rendered immediately, so they
    are also safe across job restarts.
  - On re-run the script loads all existing checkpoints, skips those
    samples, and only computes the remaining ones.

RMSE comparability:
  - test_metrics.json saves both normalized MSE and physical RMSE.
  - physical_rmse = sqrt(normalized_mse) * signal_std
    → same physical units as NIGWT and SVGAE/GXN.

Saves 8 fixed test sample visualizations.
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paths import ROOT_PROJECT_PATH
from baselines.graph_signal_sampling import sampling_and_reconstruction

from thgsp.graphs.core import SparseTensor, GraphBase


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CKPT_PREFIX = "ckpt_"


def _ckpt_json_path(output_dir: str, idx: int) -> str:
    return os.path.join(output_dir, f"{_CKPT_PREFIX}{idx}.json")


def _ckpt_recon_path(output_dir: str, idx: int) -> str:
    return os.path.join(output_dir, f"{_CKPT_PREFIX}{idx}_recon.pt")


def save_sample_checkpoint(
    output_dir: str,
    sample_idx: int,
    mse_results: dict,
    recon_results: dict = None,
    vis_set: set = None,
):
    """
    Persist per-sample results to disk immediately after computation so that
    the job can be resumed without recomputing already-done samples.

    - mse_results: dict[method -> float|None]  (always saved as JSON)
    - recon_results: dict[method -> np.ndarray|None]  (saved as .pt for vis
      samples only, using float16 to keep file size small)
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(_ckpt_json_path(output_dir, sample_idx), "w") as f:
        json.dump({"sample_idx": sample_idx, "mse_results": mse_results}, f)

    if vis_set and sample_idx in vis_set and recon_results is not None:
        pt_path = _ckpt_recon_path(output_dir, sample_idx)
        torch.save(
            {
                m: torch.from_numpy(arr).half() if arr is not None else None
                for m, arr in recon_results.items()
            },
            pt_path,
        )


def load_completed_samples(output_dir: str) -> dict:
    """
    Scan output_dir for existing checkpoint files and return a dict:
        {sample_idx: (mse_results, recon_results)}
    where recon_results is None if no .pt checkpoint exists for that sample.
    """
    completed = {}
    if not output_dir or not os.path.isdir(output_dir):
        return completed

    for fname in os.listdir(output_dir):
        if not (fname.startswith(_CKPT_PREFIX) and fname.endswith(".json")):
            continue
        path = os.path.join(output_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            idx = data["sample_idx"]
            mse_r = data["mse_results"]

            recon_r = None
            pt_path = _ckpt_recon_path(output_dir, idx)
            if os.path.exists(pt_path):
                raw = torch.load(pt_path, map_location="cpu", weights_only=False)
                recon_r = {
                    m: arr.float().numpy() if arr is not None else None
                    for m, arr in raw.items()
                }

            completed[idx] = (mse_r, recon_r)
        except Exception as e:
            print(f"  Warning: could not load checkpoint {fname}: {e}")

    return completed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_signals(processed_split_dir: str, field_index: int = 0, max_samples: int = None):
    """Load all signals and shared graph data."""
    split_dir = Path(processed_split_dir)
    shared = torch.load(split_dir / "shared_data.pt", map_location='cpu', weights_only=False)
    edge_index = shared['edge_index']
    num_nodes  = shared['num_nodes']
    num_rows   = shared['num_rows']
    num_cols   = shared['num_columns']

    sample_files = sorted(
        [f for f in split_dir.glob("*.pt") if f.stem.isdigit()],
        key=lambda f: int(f.stem)
    )
    if max_samples:
        sample_files = sample_files[:max_samples]

    signals = []
    for f in sample_files:
        s = torch.load(f, map_location='cpu', weights_only=False)
        # .float() handles both float16 (half-precision storage) and float32
        signals.append(s.x[:, field_index].float())
    signals = torch.stack(signals, dim=1)  # (N, num_samples)
    return signals, edge_index, num_nodes, num_rows, num_cols, shared


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess — must be top-level for pickle)
# ---------------------------------------------------------------------------

def _evaluate_one_sample(args_tuple):
    """
    Worker function for ProcessPoolExecutor.

    Evaluates all three classical methods on a single sample and returns
    per-method MSE values (in normalized space).

    Returns:
        dict mapping method_name → mse (float) or None if failed
    """
    (
        sample_idx,
        x_orig_numpy,      # shape (N,)  — numpy array (picklable)
        sig_mean,
        sig_std,
        edge_index_numpy,  # (2, E) numpy
        num_nodes,
        sample_size,
    ) = args_tuple

    # Re-import inside worker (subprocess has fresh Python state)
    import sys, os
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    import torch
    import torch.nn.functional as F
    from thgsp.graphs.core import SparseTensor, GraphBase
    from baselines.graph_signal_sampling import sampling_and_reconstruction

    x_orig = torch.from_numpy(x_orig_numpy)
    sig_mean_t = torch.tensor(sig_mean)
    sig_std_t  = torch.tensor(sig_std)
    x_norm = (x_orig - sig_mean_t) / sig_std_t
    x_2d   = x_norm.unsqueeze(-1)  # (N, 1)

    # Rebuild thgsp objects inside worker
    row = torch.from_numpy(edge_index_numpy[0])
    col = torch.from_numpy(edge_index_numpy[1])
    value = torch.ones(row.numel(), dtype=x_orig.dtype)
    adj = SparseTensor(row=row, col=col, value=value,
                       sparse_sizes=(num_nodes, num_nodes))
    G = GraphBase(adj, dtype=x_orig.dtype, device=x_orig.device)

    deg = torch.zeros(num_nodes, dtype=x_orig.dtype)
    deg.scatter_add_(0, row, torch.ones(row.numel(), dtype=x_orig.dtype))
    L_row = torch.cat([row, torch.arange(num_nodes)])
    L_col = torch.cat([col, torch.arange(num_nodes)])
    L_val = torch.cat([-torch.ones(row.numel(), dtype=x_orig.dtype), deg])
    L_comb = SparseTensor(row=L_row, col=L_col, value=L_val,
                          sparse_sizes=(num_nodes, num_nodes))

    inv_sqrt_deg = deg.pow(-0.5)
    inv_sqrt_deg[deg == 0] = 0.0
    L_sym_val = -inv_sqrt_deg[L_row] * L_val.clone() * inv_sqrt_deg[L_col]
    diag_mask = L_row == L_col
    L_sym_val[diag_mask] = 1.0
    L_sym = SparseTensor(row=L_row, col=L_col, value=L_sym_val,
                         sparse_sizes=(num_nodes, num_nodes))

    search_params = {
        'fastgsss_bandwidth': 'exact',
        'fastgsss_nu': 75.0,
        'fastgsss_order': 12,
        'fastgsss_recon_order': 3,
        'bsgda_mu': 0.01,
        'bsgda_epsilon': 1e-5,
        'bsgda_p_hops': 12,
        'bsgda_reg_order': 1,
        'rsbs_bandwidth_mode': 'exact',
        'rsbs_num_estimation': 1,
        'rsbs_epsilon': 1e-2,
        'rsbs_order': 30,
        'rsbs_mu': 0.01,
        'rsbs_reg_order': 1,
        'rsbs_lap_type': 'comb',
    }

    mse_results = {}
    recon_results = {}  # store x_hat_norm for visualization return

    for method in ['fastgsss', 'bsgda', 'rsbs']:
        args_dict = {
            'num_nodes': num_nodes,
            'sample_size': sample_size,
            'thgsp_graph': G,
            'thgsp_adj': adj,
            'thgsp_L_comb': L_comb,
            'thgsp_L_sym': L_sym,
            'signal_bandwidth': sample_size,
        }
        try:
            S, x_hat_norm = sampling_and_reconstruction(
                method_name=method,
                x=x_2d,
                args=args_dict,
                search_params=search_params,
            )
            mse = F.mse_loss(x_hat_norm.squeeze(), x_norm).item()
            mse_results[method] = mse
            recon_results[method] = x_hat_norm.squeeze().numpy()
            print(f"  [{method}] sample {sample_idx}: MSE (norm)={mse:.6f}", flush=True)
        except Exception as e:
            print(f"  [{method}] sample {sample_idx}: FAILED - {e}", flush=True)
            mse_results[method] = None
            recon_results[method] = None

    return sample_idx, mse_results, recon_results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _save_visualization(
    sample_idx: int,
    mse_r: dict,
    recon_r: dict,
    all_signals,
    sig_mean,
    sig_std,
    num_rows: int,
    num_cols: int,
    output_dir: str,
):
    """Save per-method reconstruction images for one sample. Closes figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.gwt_coeffs import visualize_grid_graph_inverse_sgwt_reconstruction

    x_orig = all_signals[:, sample_idx]
    for method in ['fastgsss', 'bsgda', 'rsbs']:
        png_path = os.path.join(output_dir, f"{method}_sample{sample_idx}.png")
        if os.path.exists(png_path):
            continue  # already saved in a previous run
        if recon_r is None:
            continue
        x_hat_numpy = recon_r.get(method)
        if x_hat_numpy is None:
            continue
        x_hat_orig = torch.from_numpy(x_hat_numpy) * sig_std + sig_mean
        visualize_grid_graph_inverse_sgwt_reconstruction(
            x_recon=x_hat_orig,
            x=x_orig.cpu(),
            rows=num_rows,
            cols=num_cols,
            save_fig=True,
            filename_prefix=f"{method}_sample{sample_idx}",
            save_dir=output_dir,
        )
    plt.close("all")


def run_all_classical_baselines(
    processed_split_dir: str,
    sample_indices,
    keep_ratio: float = 0.01,
    field_index: int = 0,
    output_dir: str = None,
    num_workers: int = 4,
):
    """
    Run FastGSSS, BSGDA, RSBS on normalized signals.

    Supports automatic resume: if output_dir already contains per-sample
    checkpoint files from a previous run, those samples are skipped and
    their results are merged into the final test_metrics.json.

    Args:
        processed_split_dir: Path to test split directory.
        sample_indices:      List of sample indices to evaluate.
        keep_ratio:          Fraction of nodes to sample.
        field_index:         Field index to extract from each sample.
        output_dir:          Where to save visualizations and test_metrics.json.
        num_workers:         Number of parallel worker processes (default 4).
                             Set to 1 to disable parallelism for debugging.
    """
    # Load all needed signals at once
    all_signals, edge_index, num_nodes, num_rows, num_cols, shared = \
        load_all_signals(processed_split_dir, field_index,
                         max_samples=max(sample_indices) + 1 if sample_indices else None)

    # Z-score normalize (using stats from THIS split for classical methods)
    sig_mean = all_signals.mean()
    sig_std  = all_signals.std()
    if sig_std < 1e-8:
        sig_std = torch.tensor(1.0)
    print(f"Normalization: mean={sig_mean:.4f}, std={sig_std:.4f}")

    sample_size = max(1, int(keep_ratio * num_nodes))
    print(f"Nodes: {num_nodes}, sample_size={sample_size} ({keep_ratio*100:.1f}%)")

    # Auto-cap parallelism for large graphs.
    # Each worker rebuilds the full graph's sparse matrices; at 4× concurrency
    # this OOM-kills workers on grids with 49K+ nodes even on A100 nodes.
    #   > 64K nodes  → 1 worker  (viscoelastic: 91136)
    #   > 32K nodes  → 2 workers (turbulent:    49152)
    #   ≤ 32K nodes  → use as-specified (kolmogorov/dynamic_stall: 16384)
    if num_nodes > 64_000 and num_workers > 1:
        print(f"  ⚠  Large graph ({num_nodes:,} nodes): "
              f"reducing workers {num_workers}→1 to prevent OOM.")
        num_workers = 1
    elif num_nodes > 32_000 and num_workers > 2:
        print(f"  ⚠  Large graph ({num_nodes:,} nodes): "
              f"reducing workers {num_workers}→2 to prevent OOM.")
        num_workers = 2

    print(f"Parallel workers: {num_workers}")

    # ------------------------------------------------------------------
    # Resume: load already-completed sample checkpoints
    # ------------------------------------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    completed = load_completed_samples(output_dir) if output_dir else {}
    if completed:
        done_ids = sorted(completed.keys())
        print(f"  ✓ Resuming: {len(completed)} sample(s) already checkpointed "
              f"(indices {done_ids[:10]}{'...' if len(done_ids) > 10 else ''}).")

    # Merge checkpointed results into sample_outputs immediately
    sample_outputs = dict(completed)

    vis_set = set(range(min(8, len(sample_indices))))

    # Build picklable argument tuples only for samples not yet done
    edge_index_numpy = edge_index.numpy()  # (2, E)
    worker_args = []
    for sample_idx in sample_indices:
        if sample_idx in completed:
            continue
        x_orig_numpy = all_signals[:, sample_idx].numpy()
        worker_args.append((
            sample_idx,
            x_orig_numpy,
            sig_mean.item(),
            sig_std.item(),
            edge_index_numpy,
            num_nodes,
            sample_size,
        ))

    remaining = len(worker_args)
    skipped   = len(sample_indices) - remaining
    if skipped:
        print(f"  → Skipping {skipped} already-done sample(s), "
              f"computing {remaining} remaining.")
    else:
        print(f"  → Computing all {remaining} samples.")

    # ------------------------------------------------------------------
    # Compute remaining samples
    # ------------------------------------------------------------------
    if num_workers == 1:
        # Sequential: checkpoint + visualize immediately after each sample
        for args_tuple in worker_args:
            idx, mse_r, recon_r = _evaluate_one_sample(args_tuple)
            sample_outputs[idx] = (mse_r, recon_r)
            # Checkpoint MSE + recon (for vis samples) right away
            if output_dir:
                save_sample_checkpoint(output_dir, idx, mse_r, recon_r, vis_set)
                if idx in vis_set:
                    _save_visualization(
                        idx, mse_r, recon_r,
                        all_signals, sig_mean, sig_std,
                        num_rows, num_cols, output_dir,
                    )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_evaluate_one_sample, arg): arg[0]
                for arg in worker_args
            }
            for future in as_completed(futures):
                try:
                    idx, mse_r, recon_r = future.result()
                    sample_outputs[idx] = (mse_r, recon_r)
                    if output_dir:
                        save_sample_checkpoint(output_dir, idx, mse_r, recon_r, vis_set)
                except Exception as e:
                    sample_idx = futures[future]
                    print(f"  Worker for sample {sample_idx} raised: {e}")
                    sample_outputs[sample_idx] = (
                        {m: None for m in ['fastgsss', 'bsgda', 'rsbs']},
                        {m: None for m in ['fastgsss', 'bsgda', 'rsbs']},
                    )

    # ------------------------------------------------------------------
    # Visualizations (deferred path for parallel mode; PNG-existence check
    # also handles sequential-mode samples loaded from checkpoint)
    # ------------------------------------------------------------------
    if output_dir:
        for sample_idx in sample_indices:
            if sample_idx not in vis_set:
                continue
            mse_r, recon_r = sample_outputs.get(
                sample_idx,
                ({}, {m: None for m in ['fastgsss', 'bsgda', 'rsbs']})
            )
            _save_visualization(
                sample_idx, mse_r, recon_r,
                all_signals, sig_mean, sig_std,
                num_rows, num_cols, output_dir,
            )

    # ------------------------------------------------------------------
    # Aggregate and save final results
    # ------------------------------------------------------------------
    results = {m: [] for m in ['fastgsss', 'bsgda', 'rsbs']}
    for sample_idx in sample_indices:
        mse_r, _ = sample_outputs.get(
            sample_idx,
            ({m: None for m in results}, None)
        )
        for method in ['fastgsss', 'bsgda', 'rsbs']:
            if mse_r.get(method) is not None:
                results[method].append(mse_r[method])

    print("\n=== CLASSICAL BASELINE RESULTS ===")
    all_results = {}
    for method, mses in results.items():
        if mses:
            mean_m  = float(np.mean(mses))
            std_m   = float(np.std(mses))
            # Physical RMSE: sqrt(normalized_mse) * signal_std → comparable with NIGWT/SVGAE/GXN
            rmses_physical = [float(m ** 0.5) * float(sig_std) for m in mses]
            mean_rmse_phys = float(np.mean(rmses_physical))
            std_rmse_phys  = float(np.std(rmses_physical))
            print(f"  {method}: "
                  f"MSE(norm)={mean_m:.6f}±{std_m:.6f}  "
                  f"RMSE(physical)={mean_rmse_phys:.6f}±{std_rmse_phys:.6f}  "
                  f"(n={len(mses)} samples)")
            all_results[method] = {
                'mean_mse':            mean_m,
                'std_mse':             std_m,
                'all_mses':            mses,
                'mean_rmse_physical':  mean_rmse_phys,
                'std_rmse_physical':   std_rmse_phys,
                'all_rmses_physical':  rmses_physical,
                'signal_std':          float(sig_std),
                'num_samples':         len(mses),
            }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {metrics_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-split-dir', required=True)
    parser.add_argument('--field-index', type=int, default=0)
    parser.add_argument('--keep-ratio', type=float, default=0.01)
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of test samples to evaluate (default: 10). '
                             'Increase for thorough evaluation; reduce for speed.')
    parser.add_argument('--output-dir', type=str, default='baseline_results/classical')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Parallel worker processes for per-sample evaluation '
                             '(default: 4). Set to 1 for sequential/debug mode. '
                             'Auto-capped for large graphs: >64K→1, >32K→2.')
    args = parser.parse_args()

    run_all_classical_baselines(
        processed_split_dir=args.processed_split_dir,
        sample_indices=list(range(args.num_samples)),
        keep_ratio=args.keep_ratio,
        field_index=args.field_index,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
