"""
Inspect shear_flow HDF5 files
"""
import h5py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
dataset_path = PROJECT_ROOT / "local_storage" / "datasets" / "shear_flow" / "data"

for split in ['train', 'valid', 'test']:
    split_dir = dataset_path / split
    if not split_dir.exists():
        print(f"? {split} directory not found")
        continue
    
    h5_files = list(split_dir.glob("*.hdf5"))
    if not h5_files:
        print(f"? No HDF5 files in {split}")
        continue
    
    print("=" * 60)
    print(f"SPLIT: {split.upper()} ({len(h5_files)} files)")
    print("=" * 60)
    
    # Inspect first file
    first_file = h5_files[0]
    with h5py.File(first_file, 'r') as f:
        print(f"\nFile: {first_file.name}")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        
        f.visititems(print_structure)
        
        # Print sample data stats
        if 'fields' in f:
            fields = f['fields'][:]
            print(f"\n  Top-level keys: {list(f.keys())}")
            print(f"    'fields': shape={fields.shape}, dtype={fields.dtype}, min={fields.min():.4f}, max={fields.max():.4f}")