"""
Convert SPZ file to .splat format.

This script converts Niantic's compressed SPZ files to the .splat format
used by the antimatter15/splat WebGL viewer.

SPZ is a compressed 3D Gaussian Splatting format by Niantic Labs that achieves
~90% compression compared to PLY with virtually no perceptible quality loss.

The .splat format stores each Gaussian as 32 bytes:
- position: float32[3] (12 bytes)
- scale: float32[3] (12 bytes)
- color: uint8[4] (4 bytes - RGBA)
- rotation: uint8[4] (4 bytes - normalized quaternion)

Requirements:
    pip install spz numpy

Usage:
    python convert_spz_to_splat.py input.spz -o output.splat
    python convert_spz_to_splat.py file1.spz file2.spz  # Creates .splat for each

References:
    - https://github.com/nianticlabs/spz
    - https://scaniverse.com/spz
"""

import numpy as np
import argparse
from io import BytesIO

try:
    import spz
except ImportError:
    print("Error: spz library not found.")
    print("Install it with:")
    print("  git clone https://github.com/nianticlabs/spz.git")
    print("  cd spz && pip install .")
    exit(1)


def sigmoid(x):
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def process_spz_to_splat(spz_file_path):
    """
    Convert an SPZ file to splat format.

    Args:
        spz_file_path: Path to the input SPZ file

    Returns:
        bytes: The splat file data
    """
    # Load SPZ file
    cloud = spz.load_spz(spz_file_path)

    num_points = cloud.num_points
    print(f"  Loaded {num_points:,} gaussians from SPZ")

    # Extract data from GaussianCloud
    # positions: float32 (num_points, 3)
    positions = np.array(cloud.positions).reshape(-1, 3)

    # scales: float32 (num_points, 3) - log scale, need exp()
    scales_log = np.array(cloud.scales).reshape(-1, 3)
    scales = np.exp(scales_log)

    # rotations: float32 (num_points, 4) - quaternion (x, y, z, w)
    rotations = np.array(cloud.rotations).reshape(-1, 4)

    # alphas: float32 (num_points) - pre-sigmoid opacity
    alphas = np.array(cloud.alphas)
    opacity = sigmoid(alphas)

    # colors: float32 (num_points, 3) - RGB [0, 1]
    colors = np.array(cloud.colors).reshape(-1, 3)

    # Sort by importance (scale * opacity) - larger/more opaque first for progressive loading
    importance = np.prod(scales, axis=1) * opacity
    sorted_indices = np.argsort(-importance)

    # Apply sorting
    positions = positions[sorted_indices]
    scales = scales[sorted_indices]
    rotations = rotations[sorted_indices]
    opacity = opacity[sorted_indices]
    colors = colors[sorted_indices]

    # Build splat buffer (32 bytes per gaussian)
    buffer = BytesIO()

    for i in range(num_points):
        # Position (float32[3] = 12 bytes)
        pos = positions[i].astype(np.float32)
        buffer.write(pos.tobytes())

        # Scale (float32[3] = 12 bytes)
        scale = scales[i].astype(np.float32)
        buffer.write(scale.tobytes())

        # Color RGBA (uint8[4] = 4 bytes)
        # colors are [0, 1] range, convert to [0, 255]
        rgba = np.array([
            colors[i, 0],  # R
            colors[i, 1],  # G
            colors[i, 2],  # B
            opacity[i],    # A
        ])
        rgba_uint8 = (rgba * 255).clip(0, 255).astype(np.uint8)
        buffer.write(rgba_uint8.tobytes())

        # Rotation quaternion (uint8[4] = 4 bytes)
        # SPZ uses (x, y, z, w) order
        # Normalize and map [-1, 1] to [0, 255]
        rot = rotations[i]
        rot_norm = rot / (np.linalg.norm(rot) + 1e-10)
        # Reorder to (w, x, y, z) for .splat format compatibility
        rot_wxyz = np.array([rot_norm[3], rot_norm[0], rot_norm[1], rot_norm[2]])
        rot_uint8 = ((rot_wxyz * 128) + 128).clip(0, 255).astype(np.uint8)
        buffer.write(rot_uint8.tobytes())

    return buffer.getvalue()


def process_spz_to_splat_vectorized(spz_file_path):
    """
    Convert an SPZ file to splat format (vectorized version for speed).

    Args:
        spz_file_path: Path to the input SPZ file

    Returns:
        bytes: The splat file data
    """
    # Load SPZ file
    cloud = spz.load_spz(spz_file_path)

    num_points = cloud.num_points
    print(f"  Loaded {num_points:,} gaussians from SPZ")

    # Extract data from GaussianCloud
    positions = np.array(cloud.positions).reshape(-1, 3).astype(np.float32)
    scales_log = np.array(cloud.scales).reshape(-1, 3)
    scales = np.exp(scales_log).astype(np.float32)
    rotations = np.array(cloud.rotations).reshape(-1, 4)
    alphas = np.array(cloud.alphas)
    opacity = sigmoid(alphas)
    colors = np.array(cloud.colors).reshape(-1, 3)

    # Sort by importance
    importance = np.prod(scales, axis=1) * opacity
    sorted_indices = np.argsort(-importance)

    positions = positions[sorted_indices]
    scales = scales[sorted_indices]
    rotations = rotations[sorted_indices]
    opacity = opacity[sorted_indices]
    colors = colors[sorted_indices]

    # Normalize rotations
    rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-10
    rotations = rotations / rot_norms

    # Reorder quaternion (x,y,z,w) -> (w,x,y,z)
    rotations_wxyz = rotations[:, [3, 0, 1, 2]]

    # Prepare RGBA (uint8)
    rgba = np.column_stack([colors, opacity])
    rgba_uint8 = (rgba * 255).clip(0, 255).astype(np.uint8)

    # Prepare rotation uint8
    rot_uint8 = ((rotations_wxyz * 128) + 128).clip(0, 255).astype(np.uint8)

    # Build output buffer
    # Each gaussian: 12 (pos) + 12 (scale) + 4 (rgba) + 4 (rot) = 32 bytes
    output = np.zeros((num_points, 32), dtype=np.uint8)

    # Position (bytes 0-11)
    output[:, 0:12] = positions.view(np.uint8).reshape(-1, 12)

    # Scale (bytes 12-23)
    output[:, 12:24] = scales.view(np.uint8).reshape(-1, 12)

    # RGBA (bytes 24-27)
    output[:, 24:28] = rgba_uint8

    # Rotation (bytes 28-31)
    output[:, 28:32] = rot_uint8

    return output.tobytes()


def save_splat_file(splat_data, output_path):
    """
    Save splat data to a file.

    Args:
        splat_data: bytes data to write
        output_path: Path to the output file
    """
    with open(output_path, "wb") as f:
        f.write(splat_data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SPZ files to .splat format for WebGL viewing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_spz_to_splat.py model.spz -o model.splat
    python convert_spz_to_splat.py *.spz  # Creates .splat for each file

The output .splat file can be viewed at https://antimatter15.com/splat/
or with the web_viewer_final in this project.

To merge with a dynamic 4DGS object:
    python merge_splat_files.py background.splat object.splatv -o merged.splatv
        """
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input SPZ file(s) to convert"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .splat file path (only valid for single input file)"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slower but more memory-efficient conversion (for very large files)"
    )

    args = parser.parse_args()

    if args.output and len(args.input_files) > 1:
        print("Warning: --output is ignored when processing multiple files")
        print("Each file will be saved as <input_file>.splat")

    for input_file in args.input_files:
        print(f"Processing {input_file}...")

        try:
            # Choose conversion method
            if args.slow:
                splat_data = process_spz_to_splat(input_file)
            else:
                splat_data = process_spz_to_splat_vectorized(input_file)

            # Determine output path
            if args.output and len(args.input_files) == 1:
                output_file = args.output
            else:
                # Replace .spz extension or append .splat
                if input_file.lower().endswith('.spz'):
                    output_file = input_file[:-4] + ".splat"
                else:
                    output_file = input_file + ".splat"

            save_splat_file(splat_data, output_file)

            # Calculate and display statistics
            num_gaussians = len(splat_data) // 32  # 32 bytes per gaussian
            file_size_mb = len(splat_data) / (1024 * 1024)

            print(f"  → Saved {output_file}")
            print(f"    Gaussians: {num_gaussians:,}")
            print(f"    File size: {file_size_mb:.2f} MB")

        except FileNotFoundError:
            print(f"  Error: File not found: {input_file}")
            continue
        except Exception as e:
            print(f"  Error processing {input_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nDone!")


if __name__ == "__main__":
    main()
