
import argparse
import numpy as np
import struct
from pathlib import Path

# Reuse logic from merge_splat_files.py
def read_splat_file(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    num_gaussians = len(data) // 32
    f_buffer = np.frombuffer(data, dtype=np.float32).reshape(-1, 8)
    
    positions = f_buffer[:, 0:3]
    scales = f_buffer[:, 3:6]
    colors = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)[:, 24:28]
    rotations = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)[:, 28:32]
    
    return {
        'positions': positions,
        'scales': scales,
        'colors': colors,
        'rotations': rotations,
        'count': num_gaussians
    }

def rotation_matrix(rx, ry, rz):
    # Degrees to radians
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def euler_to_quat(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])

def float_to_half(f):
    return np.float16(f).view(np.uint16)

def pack_half2x16(f1, f2):
    h1 = float_to_half(f1)
    h2 = float_to_half(f2)
    return (int(h2) << 16) | int(h1)

def save_rectified_splatv(input_path, output_path, rotation, extra_yaw=0):
    print(f"Reading {input_path}...")
    data = read_splat_file(input_path)
    count = data['count']
    
    # Texture dimensions
    texwidth = 1024 * 4 # Shader expects 4096 width
    texheight = (count * 4 + texwidth - 1) // texwidth
    texheight = max(texheight, 1)
    
    print(f"Rectifying with rotation {rotation} and extra yaw {extra_yaw}...")
    R = rotation_matrix(*rotation)
    q_rot = euler_to_quat(*rotation)
    
    # Apply extra yaw (global Y rotation)
    if extra_yaw != 0:
        R_yaw = rotation_matrix(0, extra_yaw, 0)
        R = R_yaw @ R # Matrix multiplication order: Apply Yaw *after* (globally) or *before*? 
                      # We want to rotate the *rectified result* around the new vertical Y.
                      # So P_new = R_yaw @ (R_rect @ P_old)
                      # So R_total = R_yaw @ R_rect
        
        q_yaw = euler_to_quat(0, extra_yaw, 0)
        q_rot = q_mult(q_yaw, q_rot)

    # Prepare buffer
    texdata = np.zeros(texwidth * texheight * 4, dtype=np.uint32)
    texdata_f = texdata.view(np.float32)
    
    positions = data['positions']
    scales = data['scales']
    colors = data['colors']
    rotations = data['rotations']
    
    # Batch Rotate Positions
    rotated_pos = positions @ R.T
    
    print(f"DEBUG: Sample Transformation")
    print(f"  Pos[0] Old: {positions[0]}")
    print(f"  Pos[0] New: {rotated_pos[0]}")
    print(f"  Calculated Rotation Matrix R:\n{R}")
    
    print("Processing gaussians...")
    for i in range(count):
        # 1. Position
        pos = rotated_pos[i]
        texdata_f[16 * i + 0] = pos[0]
        texdata_f[16 * i + 1] = pos[1]
        texdata_f[16 * i + 2] = pos[2]
        
        # 2. Rotation (Quaternion multiplication)
        # Decode original rotation (mapping 0-255 back to range)
        # Re-encoding logic might be lossy if not careful.
        # But for .splat files, the rotation is u8u8u8u8.
        # Standard .splat mapping: (val - 128) / 128
        rot_u8 = rotations[i]
        q_old = (rot_u8.astype(np.float32) - 128) / 128.0
        # q_old order usually x, y, z, w in .splat
        
        # Apply global rotation
        q_new = q_mult(q_rot, q_old)
        
        # Pack to half floats for .splatv
        texdata[16 * i + 3] = pack_half2x16(q_new[0], q_new[1])
        texdata[16 * i + 4] = pack_half2x16(q_new[2], q_new[3])
        
        # 3. Scale (Packed Half Float)
        s = scales[i]
        texdata[16 * i + 5] = pack_half2x16(s[0], s[1])
        texdata[16 * i + 6] = pack_half2x16(s[2], 0)
        
        # 4. Color (RGBA Packed in uint32 at index 7)
        c = colors[i]
        # We need to write bytes to the memory of texdata[16*i + 7]
        # Easier to construct the uint32 value:
        # R | (G << 8) | (B << 16) | (A << 24)
        # assuming Little Endian (standard for splat files/webgl)
        parts = int(c[0]) | (int(c[1]) << 8) | (int(c[2]) << 16) | (int(c[3]) << 24)
        texdata[16 * i + 7] = parts

    # Write file
    if output_path.lower().endswith('.splat'):
        print(f"Saving as standard .splat format to {output_path}")
        with open(output_path, 'wb') as f:
            for i in range(count):
                # 1. Position (3 floats)
                f.write(struct.pack('<fff', *rotated_pos[i]))
                
                # 2. Scale (3 floats)
                f.write(struct.pack('<fff', *scales[i]))
                
                # 3. Color (4 bytes RGBA)
                f.write(colors[i].tobytes())
                
                # Re-calculate q_new
                rot_u8 = rotations[i]
                q_old = (rot_u8.astype(np.float32) - 128) / 128.0
                q_new = q_mult(q_rot, q_old)
                
                # Normalize q_new
                norm = np.linalg.norm(q_new)
                if norm > 0: q_new /= norm
                
                # Encode back to uint8 (0-255)
                # q = (val - 128) / 128 => val = q * 128 + 128
                q_u8 = np.clip(q_new * 128 + 128, 0, 255).astype(np.uint8)
                
                f.write(q_u8.tobytes())
    else:
        # Existing .splatv logic
        with open(output_path, 'wb') as f:
            f.write(struct.pack('<I', 0x674b)) # Kg (Standard splatv magic)
            
            # Metadata
            meta = [{
                "type": "splat",
                "size": texdata.nbytes,
                "texwidth": texwidth,
                "texheight": texheight,
                 "cameras": [{
                    "id": 0,
                    "img_name": "00001",
                    "width": 1959,
                    "height": 1090,
                    "position": [-3, 0, -3],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "fy": 1164.66,
                    "fx": 1159.58,
                }]
            }]
            import json
            json_str = json.dumps(meta).encode('utf-8')
            f.write(struct.pack('<I', len(json_str)))
            f.write(json_str)
            
            # Data
            f.write(texdata.tobytes())
        
    print(f"Saved rectified map to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .splat file")
    parser.add_argument("output", help="Output .splatv file")
    parser.add_argument("--rotate", nargs=3, type=float, required=True, help="Rotation degrees (x y z)")
    parser.add_argument("--extra-yaw", type=float, default=0, help="Additional global rotation around Y axis (degrees)")
    args = parser.parse_args()
    
    save_rectified_splatv(args.input, args.output, args.rotate, args.extra_yaw)
