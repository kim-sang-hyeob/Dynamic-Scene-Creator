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

# ==================== Math Helpers ====================
def rotation_matrix(rx, ry, rz):
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def euler_to_quat(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)

def q_mult(q1, q2):
    # WXYZ convention
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)


def process_ply_to_splat(ply_file_path, rotate_args=None, extra_yaw=0):
    """
    Convert a 3DGS PLY file to splat format with optional rectification.
    
    Args:
        ply_file_path: Path to the input PLY file
        rotate_args: (rx, ry, rz) tuple for base rectification
        extra_yaw: Extra yaw rotation in degrees
        
    Returns:
        bytes: The splat file data
    """
    R_mat = np.eye(3, dtype=np.float32)
    q_rot = np.array([1, 0, 0, 0], dtype=np.float32) # Identity w,x,y,z
    
    if rotate_args or extra_yaw != 0:
        rx, ry, rz = rotate_args if rotate_args else (0, 0, 0)
        
        # Base Rectification
        R_base = rotation_matrix(rx, ry, rz)
        q_base = euler_to_quat(rx, ry, rz)
        
        # Extra Yaw (Global Y)
        R_yaw = rotation_matrix(0, extra_yaw, 0)
        q_yaw = euler_to_quat(0, extra_yaw, 0)
        
        # Combine: Yaw * Base
        R_mat = (R_yaw @ R_base).astype(np.float32)
        q_rot = q_mult(q_yaw, q_base)
        print(f"Applying Rotation: Base={rotate_args}, ExtraYaw={extra_yaw}")
    
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
        
        # Rotation quaternion [w, x, y, z] standard 3DGS
        rot_curr = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        
        # Apply Transformation
        position = (R_mat @ position)
        rot = q_mult(q_rot, rot_curr) # Apply global rotation to local rotation
        
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
    parser.add_argument("--rotate", nargs=3, type=float, default=None, help="Rotation degrees (x y z) for rectification")
    parser.add_argument("--extra-yaw", type=float, default=0, help="Additional global rotation around Y axis")
    
    args = parser.parse_args()
    
    if args.output and len(args.input_files) > 1:
        print("Warning: --output is ignored when processing multiple files")
        print("Each file will be saved as <input_file>.splat")
    
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        
        try:
            splat_data = process_ply_to_splat(input_file, args.rotate, args.extra_yaw)
            
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
