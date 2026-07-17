#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_hdf5.py — HDF5 file structure inspector
================================================
Run this on the cluster BEFORE preprocessing a new HDF5 dataset to verify
field names, shapes, dtypes, and attributes.

Usage:
    python inspect_hdf5.py /path/to/dynamic_stall_train.h5
    python inspect_hdf5.py /path/to/kolmogorov_train.h5 --plot-sample 0 --field 0
"""

import sys
import argparse
from pathlib import Path


def inspect_hdf5(filepath: str, plot_sample: int = -1, field_idx: int = 0,
                 max_array_print: int = 8):
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is not installed. Run: pip install h5py --break-system-packages")

    path = Path(filepath)
    if not path.exists():
        sys.exit(f"File not found: {filepath}")

    print("=" * 80)
    print(f"HDF5 Inspection: {path.name}")
    print(f"Full path: {path}")
    print(f"Size: {path.stat().st_size / 1024**2:.1f} MB")
    print("=" * 80)

    with h5py.File(filepath, 'r') as f:
        # Root-level attributes
        root_attrs = dict(f.attrs)
        if root_attrs:
            print("\n[Root attributes]")
            for k, v in root_attrs.items():
                if hasattr(v, '__len__') and len(v) > max_array_print:
                    print(f"  {k}: {type(v).__name__} len={len(v)}  first={list(v[:4])}...")
                else:
                    print(f"  {k}: {v!r}")
        else:
            print("\n[Root attributes] (none)")

        # Recursive visitor
        print("\n[Datasets and groups]")

        candidates = []

        def _visit(name, obj):
            indent = "  " * name.count('/')
            if hasattr(obj, 'shape'):
                attrs = dict(obj.attrs)
                attr_str = ""
                if attrs:
                    attr_str = "  attrs=" + str({k: (list(v) if hasattr(v, '__len__') else v)
                                                  for k, v in attrs.items()})
                print(f"  {indent}{name}:  shape={obj.shape}  dtype={obj.dtype}{attr_str}")
                if obj.ndim >= 3:
                    candidates.append((name, obj.shape, str(obj.dtype)))
            else:
                attrs = dict(obj.attrs)
                print(f"  {indent}{name}/  [group]"
                      + (f"  attrs={dict(obj.attrs)}" if attrs else ""))

        f.visititems(_visit)

        print("\n[Candidate data arrays (ndim ≥ 3)]")
        if candidates:
            for name, shape, dtype in candidates:
                print(f"  {name}: shape={shape}, dtype={dtype}")
        else:
            print("  (none found)")

        # Decode layout of first viable array
        for key, shape, dtype in candidates:
            if len(shape) >= 4:
                print(f"\n[Layout analysis for '{key}']")
                if len(shape) == 5:
                    ns, T, C, H, W = shape
                    print(f"  Likely layout: (num_sims={ns}, T={T}, C={C}, H={H}, W={W})")
                    print(f"  Total frames: {ns * T}")
                    print(f"  Grid: {H} × {W} = {H*W} nodes")
                    print(f"  Fields (C): {C}")
                elif len(shape) == 4:
                    N, d1, d2, d3 = shape
                    if d3 <= min(d1, d2) // 4:
                        print(f"  Likely layout: (N={N}, H={d1}, W={d2}, C={d3})  [channels-last]")
                    else:
                        print(f"  Likely layout: (N={N}, C={d1}, H={d2}, W={d3})")
                        print(f"  Total frames: {N}")
                        print(f"  Grid: {d2} × {d3} = {d2*d3} nodes")
                        print(f"  Fields (C): {d1}")

                # Value statistics from first frame
                print(f"\n[Value statistics for '{key}' (first frame only)]")
                try:
                    import numpy as np
                    first = f[key][0]   # first sim or frame
                    if first.ndim >= 3:
                        first = first.reshape(-1)
                    print(f"  min={first.min():.4g}  max={first.max():.4g}  "
                          f"mean={first.mean():.4g}  std={first.std():.4g}")
                    if first.std() < 1e-10:
                        print("  ⚠  All-zero first frame — check if data is valid!")
                except Exception as e:
                    print(f"  Could not read first frame: {e}")
                break   # only analyse first candidate

        # ---- Optional visualisation ----
        if plot_sample >= 0 and candidates:
            print(f"\n[Plotting sample {plot_sample}, field {field_idx}]")
            try:
                import numpy as np
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                key = candidates[0][0]
                shape = candidates[0][1]

                with h5py.File(filepath, 'r') as f2:
                    raw = f2[key]
                    if len(shape) == 5:
                        ns, T, C, H, W = shape
                        flat_idx = plot_sample
                        sim_idx  = flat_idx // T
                        t_idx    = flat_idx % T
                        frame    = raw[sim_idx, t_idx, field_idx, :, :]   # (H, W)
                    elif len(shape) == 4:
                        N, C, H, W = shape
                        frame = raw[plot_sample, field_idx, :, :]
                    else:
                        frame = None

                if frame is not None:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    im = ax.imshow(frame, cmap='RdBu_r', aspect='auto')
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"{path.stem}  sample={plot_sample}  field={field_idx}")
                    out_png = path.parent / f"inspect_{path.stem}_s{plot_sample}_f{field_idx}.png"
                    plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved: {out_png}")

            except Exception as e:
                print(f"  Plot failed: {e}")

    print("\n" + "=" * 80)
    print("Inspection complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect HDF5 file structure for NIGWT preprocessing.'
    )
    parser.add_argument('filepath', help='Path to HDF5 file')
    parser.add_argument('--plot-sample', type=int, default=-1,
                        help='If >= 0, save a PNG of this flat sample index')
    parser.add_argument('--field', type=int, default=0,
                        help='Field (channel) index to plot (default: 0)')
    args = parser.parse_args()
    inspect_hdf5(args.filepath, plot_sample=args.plot_sample, field_idx=args.field)
