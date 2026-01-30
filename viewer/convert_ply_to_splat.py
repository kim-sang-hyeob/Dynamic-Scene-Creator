"""
Convert PLY file to .splat format.

This script converts 3D Gaussian Splatting PLY files to the .splat format
used by the antimatter15/splat WebGL viewer.

The .splat format stores each Gaussian as 32 bytes:
- position: float32[3] (12 bytes)
- scale: float32[3] (12 bytes)  
- color: uint8[4] (4 bytes - RGBA)
- rotation: uint8[4] (4 bytes - normalized quaternion)

Reference: https://github.com/antimatter15/splat

Usage:
    python convert_ply_to_splat.py input.ply -o output.splat
    python convert_ply_to_splat.py file1.ply file2.ply  # Creates file1.ply.splat, file2.ply.splat
"""

from plyfile import PlyData
import numpy as np
import argparse
from io import BytesIO


def process_ply_to_splat(ply_file_path):
    """
    Convert a 3DGS PLY file to splat format.
    
    Args:
        ply_file_path: Path to the input PLY file
        
    Returns:
        bytes: The splat file data
    """
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    
    # Sort by importance (scale * opacity)
    # Larger gaussians with higher opacity come first for progressive loading
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        
        # Position (float32[3])
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        
        # Scale (exp of log-scale values)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        
        # Rotation quaternion
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        
        # Convert SH DC coefficient to RGB color
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),  # Sigmoid for alpha
            ]
        )
        
        # Write position (12 bytes)
        buffer.write(position.tobytes())
        
        # Write scales (12 bytes)
        buffer.write(scales.tobytes())
        
        # Write color as RGBA uint8 (4 bytes)
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        
        # Write rotation as normalized uint8[4] (4 bytes)
        # Normalize quaternion and map [-1, 1] to [0, 255]
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    return buffer.getvalue()


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
        description="Convert 3DGS PLY files to .splat format for WebGL viewing.",
        epilog="""
Examples:
    python convert_ply_to_splat.py model.ply -o model.splat
    python convert_ply_to_splat.py *.ply  # Creates .ply.splat for each file
    
The output .splat file can be viewed at https://antimatter15.com/splat/
or with any compatible WebGL Gaussian Splatting viewer.
        """
    )
    parser.add_argument(
        "input_files", 
        nargs="+", 
        help="Input PLY file(s) to convert"
    )
    parser.add_argument(
        "--output", "-o", 
        default=None,
        help="Output .splat file path (only valid for single input file)"
    )
    
    args = parser.parse_args()
    
    if args.output and len(args.input_files) > 1:
        print("Warning: --output is ignored when processing multiple files")
        print("Each file will be saved as <input_file>.splat")
    
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        
        try:
            splat_data = process_ply_to_splat(input_file)
            
            if args.output and len(args.input_files) == 1:
                output_file = args.output
            else:
                # Replace .ply extension or append .splat
                if input_file.lower().endswith('.ply'):
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
            
        except Exception as e:
            print(f"  Error processing {input_file}: {e}")
            continue
    
    print("\nDone!")


if __name__ == "__main__":
    main()
