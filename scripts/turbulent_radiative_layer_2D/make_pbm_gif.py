#!/usr/bin/env python3
"""
make_pbfm_gif.py

Creates a GIF from PBFM velocity_x frames.
"""
from pathlib import Path
import imageio.v2 as imageio

# Frames are written by plot_pbfm_original.py into this script's own directory by default.
FRAMES_DIR = Path(__file__).resolve().parent / 'pbfm_gif_frames'
OUTPUT_GIF = Path('./pbfm_velocity_x.gif')

def create_gif(frames_dir: Path, output_path: Path, duration: float = 0.1):
    """
    Create GIF from PNG frames.
    
    Args:
        frames_dir: Directory containing frame_XXXX.png files
        output_path: Output GIF path
        duration: Duration per frame in seconds (default: 0.1s = 10 FPS)
    """
    # Get all frames sorted
    frame_files = sorted(frames_dir.glob('pbfm_velocity_x_frame_*.png'))
    
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    
    print(f"Found {len(frame_files)} frames")
    print(f"Creating GIF: {output_path}")
    
    # Read images
    images = [imageio.imread(str(f)) for f in frame_files]
    
    # Save as GIF
    imageio.mimsave(
        str(output_path),
        images,
        duration=duration,
        loop=0  # Infinite loop
    )
    
    print(f"? GIF saved: {output_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")

if __name__ == '__main__':
    create_gif(FRAMES_DIR, OUTPUT_GIF, duration=0.1)