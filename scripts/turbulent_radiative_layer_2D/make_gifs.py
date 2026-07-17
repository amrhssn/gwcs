"""
make_gifs.py

Creates animated GIFs from the per-sample reconstruction frames saved by
plot_50_frames.py.  One GIF per model, each frame shows:
    Original Signal | Reconstruction | Absolute Error

Output files:
    {output_dir}/gwcs.gif
    {output_dir}/svgae_99k.gif
    {output_dir}/svgae_50k.gif
    {output_dir}/gxn.gif

Usage:
    python make_gifs.py \
        --frames-dir  <path/to/plots/frames> \
        --output-dir  <path/to/plots/gifs>   \
        --num-frames  50                      \
        --fps         5
"""

import os
import argparse
from pathlib import Path
from PIL import Image

MODELS = [
    ('gwcs',      'GWCS'),
    ('svgae_99k', 'SVGAE-99K'),
    ('svgae_50k', 'SVGAE-50K'),
    ('gxn',       'GXN'),
]


def make_gif(frames_dir: Path, output_path: str, model_prefix: str,
             num_frames: int, fps: float, resize_width: int):
    duration_ms = int(1000 / fps)

    images = []
    for i in range(num_frames):
        fp = frames_dir / f'{model_prefix}_sample{i}.png'
        if not fp.exists():
            print(f"    Warning: {fp.name} not found — stopping at frame {i}")
            break
        img = Image.open(fp).convert('RGB')
        if resize_width and resize_width != img.width:
            h = int(img.height * resize_width / img.width)
            img = img.resize((resize_width, h), Image.LANCZOS)
        images.append(img.copy())
        img.close()

    if not images:
        print(f"    No frames found for {model_prefix}, skipping")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert to palette mode for proper GIF encoding
    palette_images = [im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0)
                      for im in images]

    palette_images[0].save(
        output_path,
        save_all=True,
        append_images=palette_images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    Saved {len(images)}-frame GIF ({size_mb:.1f} MB): {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIFs from per-sample reconstruction frames")
    parser.add_argument('--frames-dir',   required=True,
                        help="Directory containing the PNG frames from plot_50_frames.py")
    parser.add_argument('--output-dir',   required=True,
                        help="Output directory for GIFs")
    parser.add_argument('--num-frames',   type=int,   default=50,
                        help="Number of frames to include (default 50)")
    parser.add_argument('--fps',          type=float, default=5.0,
                        help="Playback speed in frames per second (default 5)")
    parser.add_argument('--resize-width', type=int,   default=1350,
                        help="Resize frames to this width in pixels before encoding "
                             "(default 1350 = half of the 2700px originals). "
                             "Set 0 to keep original resolution (large file).")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"ERROR: frames directory not found: {frames_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    resize = args.resize_width if args.resize_width > 0 else None

    print(f"Frames dir  : {frames_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Frames/GIF  : {args.num_frames}")
    print(f"FPS         : {args.fps}  ({int(1000/args.fps)} ms/frame)")
    print(f"Resize width: {resize if resize else 'original'}")
    print()

    for model_prefix, model_label in MODELS:
        print(f"  [{model_label}]")
        out = os.path.join(args.output_dir, f'{model_prefix}.gif')
        make_gif(frames_dir, out, model_prefix, args.num_frames, args.fps, resize)

    print(f"\nAll GIFs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
