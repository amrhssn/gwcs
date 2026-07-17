#!/usr/bin/env python3
"""
plot_pbfm_frames.py - Creates GIF frames for PBFM velocity_x field.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import ROOT_PROJECT_PATH

# EDIT ME: set to the processed-data timestamp directory you want to plot
# (or override entirely via the PBFM_PROCESSED_PATH env var).
_PBFM_TIMESTAMP = "20260321_235358"
PBFM_PROCESSED_PATH = Path(os.environ.get(
    "PBFM_PROCESSED_PATH",
    str(Path(ROOT_PROJECT_PATH) / "local_storage" / "datasets" / "pbfm" / "processed" / _PBFM_TIMESTAMP)
))
OUTPUT_DIR = Path('./pbfm_gif_frames')

def plot_pbfm_frames(
    num_samples: int = 500,
    split: str = 'test',
):
    """
    Plot frames for PBFM dataset showing only velocity_x component.
    Reads from **processed .pt files** (same format as turbulent).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    split_dir = PBFM_PROCESSED_PATH / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Load shared data (grid shape)
    shared_data = torch.load(split_dir / 'shared_data.pt', weights_only=False)
    H = shared_data['num_rows']       # 128
    W = shared_data['num_columns']    # 128
    
    print(f"Loading PBFM processed data from {split_dir}...")
    print(f"Grid: {H}x{W} (rowsxcols)")
    
    # Get sample files
    sample_files = sorted(
        [f for f in split_dir.glob('*.pt') if f.stem.isdigit()],
        key=lambda x: int(x.stem)
    )
    num_samples = min(num_samples, len(sample_files))
    
    print(f"\nPlotting {num_samples} frames...")
    
    for idx in tqdm(range(num_samples), desc="Creating frames"):
        # Load sample
        data = torch.load(sample_files[idx], weights_only=False)
        
        # Extract velocity_x (first field, index 0)
        # data.x shape: (N, 2) ? [:, 0] gives velocity_x
        velocity_x = data.x[:, 0].float().reshape(H, W)  # (H, W)
        
        # Create figure with correct aspect ratio
        aspect = W / H
        fig_height = 5
        fig_width = fig_height * aspect
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Plot signal
        im = ax.imshow(velocity_x, aspect='auto', cmap='RdBu_r', origin='lower')
        #ax.set_title(f'PBFM Velocity X - Sample {idx}', fontsize=14, fontweight='bold')
        #ax.set_xlabel('X (columns)', fontsize=12)
        #ax.set_ylabel('Y (rows)', fontsize=12)
        
        # Add colorbar
        #cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_label('Velocity X', rotation=270, labelpad=20, fontsize=11)
        
        # Add grid info as text
        #ax.text(0.02, 0.98, f'Grid: {H}x{W}', 
        #        transform=ax.transAxes,
        #        fontsize=10, 
        #        verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        #plt.tight_layout()
        
        # Save frame
        frame_path = OUTPUT_DIR / f'pbfm_velocity_x_frame_{idx:04d}.png'
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n? Saved {num_samples} frames to {OUTPUT_DIR}")
    print(f"\nTo create GIF, run:")
    print(f"  convert -delay 10 -loop 0 {OUTPUT_DIR}/pbfm_velocity_x_frame_*.png pbfm_velocity_x.gif")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot PBFM velocity_x frames for GIF')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of frames to generate')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test', help='Dataset split')
    args = parser.parse_args()
    
    plot_pbfm_frames(num_samples=args.num_samples, split=args.split)