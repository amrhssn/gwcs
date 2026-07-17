import h5py
import numpy as np
import argparse
from pathlib import Path

def inspect(path):
    print(f"\n{'='*60}")
    print(f"FILE: {path.name}")
    if not path.exists():
        print(f"  ERROR: File not found at {path}")
        return
    with h5py.File(path, 'r') as f:
        def _print(name, obj):
            shape = obj.shape if hasattr(obj, 'shape') else ''
            dtype = obj.dtype if hasattr(obj, 'dtype') else ''
            print(f"  {name}: shape={shape}, dtype={dtype}")
        f.visititems(_print)
        # Print top-level keys
        print(f"\n  Top-level keys: {list(f.keys())}")
        for k in f.keys():
            v = f[k]
            if hasattr(v, 'shape'):
                print(f"    '{k}': shape={v.shape}, dtype={v.dtype}, min={v[0].min():.4f}, max={v[0].max():.4f}")

parser = argparse.ArgumentParser()
parser.add_argument('--raw-dir', required=True,
                    help='Absolute path to the folder containing kolmogorov_train.h5 and kolmogorov_test.h5')
args = parser.parse_args()

raw_dir = Path(args.raw_dir)
for fname in ['kolmogorov_train.h5', 'kolmogorov_test.h5']:
    inspect(raw_dir / fname)