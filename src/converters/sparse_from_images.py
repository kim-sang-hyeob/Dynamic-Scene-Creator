#!/usr/bin/env python
"""
Create COLMAP sparse files from image directory.

For 4DGS training when you have images but no COLMAP reconstruction.
Creates dummy camera poses (fixed camera) suitable for object-centric scenes.

Usage:
    python src/create_sparse_from_images.py <images_dir> [--fov 50]

Example:
    python src/create_sparse_from_images.py data/black_cat_alpha/images
"""

import os
import sys
import argparse
import math
import json
from pathlib import Path
from PIL import Image


def create_colmap_sparse(images_dir, fov=50.0, output_dir=None):
    """
    Create COLMAP sparse files from images directory.

    Args:
        images_dir: Directory containing images
        fov: Field of view in degrees (default: 50)
        output_dir: Output directory for sparse files (default: ../sparse/0)
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    images = sorted([f for f in images_dir.iterdir() if f.suffix in image_extensions])

    if not images:
        raise ValueError(f"No images found in {images_dir}")

    print(f"[Sparse] Found {len(images)} images")

    # Get image dimensions from first image
    first_img = Image.open(images[0])
    width, height = first_img.size
    print(f"[Sparse] Image size: {width}x{height}")

    # Calculate focal length from FOV
    # fov = 2 * atan(sensor_size / (2 * focal_length))
    # For normalized: focal_length = width / (2 * tan(fov/2))
    fov_rad = math.radians(fov)
    focal_length = width / (2 * math.tan(fov_rad / 2))
    cx = width / 2.0
    cy = height / 2.0

    print(f"[Sparse] FOV: {fov}°, Focal length: {focal_length:.2f}")

    # Output directory
    if output_dir is None:
        output_dir = images_dir.parent / "sparse" / "0"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cameras.txt (single camera for all images)
    cameras_file = output_dir / "cameras.txt"
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {focal_length} {focal_length} {cx} {cy}\n")

    print(f"[Sparse] Created {cameras_file}")

    # Create images.txt with fixed camera pose (identity rotation, looking at origin)
    # Default pose: camera at (0, 0, -3) looking at origin
    # Rotation: identity quaternion (1, 0, 0, 0)
    # Translation: (0, 0, 3) - note: COLMAP uses camera-to-world inverse

    images_file = output_dir / "images.txt"
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, img_path in enumerate(images):
            image_id = idx + 1
            # Fixed camera pose (can be modified for orbiting cameras)
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0  # Identity rotation
            tx, ty, tz = 0.0, 0.0, 3.0  # Camera 3 units away

            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_path.name}\n")
            f.write("\n")  # Empty POINTS2D line

    print(f"[Sparse] Created {images_file}")

    # Create empty points3D.txt (SfM-free)
    points3d_file = output_dir / "points3D.txt"
    with open(points3d_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    print(f"[Sparse] Created {points3d_file}")

    # Create timestamps.json for 4DGS
    timestamps_file = images_dir.parent / "timestamps.json"
    timestamps = {}
    for idx, img_path in enumerate(images):
        # Normalized time [0, 1]
        t = idx / (len(images) - 1) if len(images) > 1 else 0.0
        timestamps[img_path.name] = t

    with open(timestamps_file, 'w') as f:
        json.dump(timestamps, f, indent=2)

    print(f"[Sparse] Created {timestamps_file}")

    return len(images)


def main():
    parser = argparse.ArgumentParser(
        description="Create COLMAP sparse files from images directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect sparse directory)
  python src/create_sparse_from_images.py data/black_cat_alpha/images

  # Custom FOV
  python src/create_sparse_from_images.py data/my_scene/images --fov 60

  # Custom output directory
  python src/create_sparse_from_images.py data/my_scene/images --output data/my_scene/sparse/0
        """
    )

    parser.add_argument("images_dir", help="Directory containing images")
    parser.add_argument("--fov", type=float, default=50.0,
                        help="Camera field of view in degrees (default: 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for sparse files")

    args = parser.parse_args()

    try:
        num_images = create_colmap_sparse(
            args.images_dir,
            fov=args.fov,
            output_dir=args.output
        )
        print(f"\n[Done] Created sparse files for {num_images} images")
        print(f"Ready for 4DGS training!")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
