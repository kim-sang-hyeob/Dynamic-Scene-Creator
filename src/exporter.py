import numpy as np
import os
import struct
from plyfile import PlyData

def convert_ply_to_splat(ply_path, splat_path):
    """
    Converts a 3DGS PLY file to the .splat format used by many web viewers (e.g. splat.tv, Luma).
    Format: x, y, z, scale[3], color[4], rotation[4] (44 bytes total per splat)
    """
    print(f"[Export] Converting {os.path.basename(ply_path)} to {os.path.basename(splat_path)}...")
    
    try:
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        
        num_gaussians = len(v)
        
        # Prepare data
        xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
        
        # Scaling
        scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32)
        # Note: scales in PLY are usually log-scales
        scales = np.exp(scales)
        
        # Rotation (quaternion)
        rotations = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32)
        # Re-normalize just in case
        rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
        
        # Colors & Opacity
        # 3DGS PLY stores SH coefficients. DC part is f_dc_0,1,2
        C0 = 0.28209479177387814
        r = (0.5 + C0 * v['f_dc_0']) * 255
        g = (0.5 + C0 * v['f_dc_1']) * 255
        b = (0.5 + C0 * v['f_dc_2']) * 255
        # Opacity (Sigmoid)
        a = (1 / (1 + np.exp(-v['opacity']))) * 255
        
        colors = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        
        # Write binary .splat file
        with open(splat_path, 'wb') as f:
            for i in range(num_gaussians):
                # position (3 floats)
                f.write(xyz[i].tobytes())
                # scale (3 floats)
                f.write(scales[i].tobytes())
                # color (4 uint8)
                f.write(colors[i].tobytes())
                # rotation (4 floats)
                f.write(rotations[i].tobytes())
                
        print(f"[Export] Saved to {splat_path}")
        return True
        
    except Exception as e:
        print(f"[Error] Conversion failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", help="Output .splat file")
    args = parser.parse_args()
    convert_ply_to_splat(args.input, args.output)
