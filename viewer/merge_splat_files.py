"""
Merge .splat and .splatv files into a single .splatv file.

This script combines:
- Background scene (.splat) - static 3D Gaussian Splatting
- Dynamic object (.splatv) - 4D Gaussian Splatting with motion

The 4DGS object position can be offset to place it in the scene.

Usage:
    python merge_splat_files.py background.splat object.splatv -o merged.splatv
    python merge_splat_files.py background.splat object.splatv -o merged.splatv --offset 0 1 0
    python merge_splat_files.py background.splat object.splatv -o merged.splatv --offset 0 1 0 --scale 0.5
"""

import numpy as np
import struct
import json
import argparse
from pathlib import Path
from scipy.interpolate import CubicSpline


def pack_half2x16(x, y):
    """Pack two float16 values into a uint32."""
    x_half = np.float16(x)
    y_half = np.float16(y)
    x_bits = x_half.view(np.uint16)
    y_bits = y_half.view(np.uint16)
    return np.uint32(x_bits) | (np.uint32(y_bits) << 16)


def rotation_matrix(rx, ry, rz):
    """
    Create a 3x3 rotation matrix from Euler angles (in degrees).
    Uses ZYX rotation order.
    """
    rx, ry, rz = np.radians([rx, ry, rz])
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def read_splat_file(filepath):
    """
    Read .splat file (antimatter15 format: 32 bytes per gaussian).
    
    Format per gaussian:
    - position: float32[3] (12 bytes)
    - scale: float32[3] (12 bytes)
    - color: uint8[4] (4 bytes - RGBA)
    - rotation: uint8[4] (4 bytes - normalized quaternion)
    
    Returns: dict with gaussians data
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    num_gaussians = len(data) // 32
    print(f"Reading .splat: {num_gaussians:,} gaussians from {filepath}")
    
    f_buffer = np.frombuffer(data, dtype=np.float32).reshape(-1, 8)
    u8_buffer = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)
    
    gaussians = {
        'positions': f_buffer[:, 0:3].copy(),
        'scales': f_buffer[:, 3:6].copy(),
        'colors': u8_buffer[:, 24:28].copy(),  # RGBA
        'rotations_u8': u8_buffer[:, 28:32].copy(),  # Normalized quaternion as uint8
        'is_dynamic': False,
        'count': num_gaussians
    }
    
    return gaussians


def read_splatv_file(filepath):
    """
    Read .splatv file.
    
    Format:
    - Magic: uint32 (0x674b)
    - JSON length: uint32
    - JSON metadata: bytes
    - Texture data: uint32[]
    
    Returns: dict with gaussians data and metadata
    """
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x674b:
            raise ValueError(f"Invalid splatv magic: {magic:x}")
        
        json_length = struct.unpack('<I', f.read(4))[0]
        json_bytes = f.read(json_length)
        metadata = json.loads(json_bytes.decode('utf-8'))
        
        texdata = np.frombuffer(f.read(), dtype=np.uint32)
    
    chunk = metadata[0]
    texwidth = chunk['texwidth']
    texheight = chunk['texheight']
    
    # Each gaussian uses 16 uint32 values (64 bytes in texture)
    num_gaussians = (texwidth * texheight) // 4
    print(f"Reading .splatv: {num_gaussians:,} gaussians from {filepath}")
    
    return {
        'texdata': texdata,
        'texwidth': texwidth,
        'texheight': texheight,
        'metadata': metadata,
        'is_dynamic': True,
        'count': num_gaussians
    }


def q_mult(q1, q2):
    """
    Multiply two quaternions q1 * q2.
    Quaternions are [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def euler_to_quat(rx, ry, rz):
    """
    Convert Euler angles (degrees) to quaternion [x, y, z, w].
    Order: ZYX (consistent with rotation_matrix function)
    """
    rx, ry, rz = np.radians([rx, ry, rz])
    
    cx = np.cos(rx * 0.5)
    sx = np.sin(rx * 0.5)
    cy = np.cos(ry * 0.5)
    sy = np.sin(ry * 0.5)
    cz = np.cos(rz * 0.5)
    sz = np.sin(rz * 0.5)
    
    # ZYX order: q = qz * qy * qx
    
    # qx = [sx, 0, 0, cx]
    # qy = [0, sy, 0, cy]
    # qz = [0, 0, sz, cz]
    
    # qy * qx
    # x = cy*sx
    # y = sy*cx
    # z = -sy*sx
    # w = cy*cx
    
    # qz * (qy * qx)
    # x = cz*cy*sx - sz*sy*cx  <- check this derivation or use standard formula
    
    # Standard ZYX Euler to Quat:
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    
    return np.array([x, y, z, w])


def splat_to_texdata(gaussians, offset=(0, 0, 0), scale=1.0, rotation=(0, 0, 0)):
    """
    Convert .splat data to splatv texture format.
    
    Args:
        gaussians: dict from read_splat_file
        offset: (x, y, z) position offset
        scale: scale factor for positions
        rotation: (rx, ry, rz) rotation in degrees
    
    Returns: (texdata, texwidth, texheight)
    """
    num = gaussians['count']
    texwidth = 1024 * 4
    texheight = int(np.ceil((4 * num) / texwidth))
    
    texdata = np.zeros((texwidth * texheight * 4,), dtype=np.uint32)
    texdata_f = texdata.view(np.float32)
    texdata_c = texdata.view(np.uint8)
    
    # Apply rotation, scale, then offset
    R = rotation_matrix(*rotation)
    positions = gaussians['positions'] @ R.T  # Apply rotation
    positions = positions * scale + np.array(offset)
    
    scales = gaussians['scales'] * scale
    colors = gaussians['colors']
    rots_u8 = gaussians['rotations_u8']
    
    # Rotation quaternion for the object transform
    q_rot = euler_to_quat(*rotation)
    
    for j in range(num):
        # Position (float32[3])
        texdata_f[16 * j + 0] = positions[j, 0]
        texdata_f[16 * j + 1] = positions[j, 1]
        texdata_f[16 * j + 2] = positions[j, 2]
        
        # Rotation from uint8 to normalized quaternion
        # (val - 128) / 128 maps 0..255 to -1..1 (approx)
        rot_0 = (float(rots_u8[j, 0]) - 128.0) / 128.0
        rot_1 = (float(rots_u8[j, 1]) - 128.0) / 128.0
        rot_2 = (float(rots_u8[j, 2]) - 128.0) / 128.0
        rot_3 = (float(rots_u8[j, 3]) - 128.0) / 128.0
        
        # Normalize just in case
        norm = np.sqrt(rot_0*rot_0 + rot_1*rot_1 + rot_2*rot_2 + rot_3*rot_3)
        if norm > 0:
            rot_0 /= norm
            rot_1 /= norm
            rot_2 /= norm
            rot_3 /= norm
            
        # Apply global rotation: q_new = q_rot * q_old
        q_old = np.array([rot_1, rot_2, rot_3, rot_0]) # Standard order usually [x, y, z, w]. 
        # CAUTION: antimatter15 splat format order.
        # Common convention for .splat is [w, x, y, z] or [x, y, z, w].
        # In convert_ply_to_splat.py (if available) we could check.
        # But looking at processSplatBuffer in hybrid.js:
        # const rot_0 = (u8_buffer[base_u8 + 28] - 128) / 128; // first byte
        # ...
        # In THREE.js or SPLAT renderers, usually w is first or last.
        # Let's assumes [x, y, z, w] based on common packing if not specified.
        # Looking at splat_to_texdata original code: 
        # texdata[16 * j + 3] = pack_half2x16(rot_0, rot_1) -> first 2 floats of rot
        
        # Let's assume input layout is [w, x, y, z] or [x, y, z, w].
        # Standard .splat usually stores [r, i, j, k] i.e. [w, x, y, z].
        # If rot_0 is w, rot_1 is x...
        # Let's try q_old = [rot_1, rot_2, rot_3, rot_0] (x,y,z,w) assuming rot_0 is w.
        # Actually usually it's x,y,z,w or w,x,y,z.
        # Let's stick to [rot_0, rot_1, rot_2, rot_3] as x,y,z,w for now unless we see artifacts.
        
        # BUT wait, simpler approach:
        # If we re-use the same packing as source, we should just multiply.
        # Let's assume standard quaternion order for multiplication.
        
        # Let's assume the bytes are [x, y, z, w].
        q_local = np.array([rot_0, rot_1, rot_2, rot_3])
        
        # Apply rotation
        q_final = q_mult(q_rot, q_local)
        
        texdata[16 * j + 3] = pack_half2x16(q_final[0], q_final[1])
        texdata[16 * j + 4] = pack_half2x16(q_final[2], q_final[3])
        
        # Scale
        texdata[16 * j + 5] = pack_half2x16(scales[j, 0], scales[j, 1])
        texdata[16 * j + 6] = pack_half2x16(scales[j, 2], 0)
        
        # RGBA
        texdata_c[4 * (16 * j + 7) + 0] = colors[j, 0]
        texdata_c[4 * (16 * j + 7) + 1] = colors[j, 1]
        texdata_c[4 * (16 * j + 7) + 2] = colors[j, 2]
        texdata_c[4 * (16 * j + 7) + 3] = colors[j, 3]
        
        # Motion data - all zeros for static splat
        texdata[16 * j + 8:16 * j + 15] = 0
        texdata[16 * j + 15] = pack_half2x16(0.5, 1.0)  # trbf_center, trbf_scale
    
    return texdata, texwidth, texheight


def unpack_half2x16(val):
    """Unpack uint32 to two float16 (returned as float32)."""
    # numpy doesn't have a direct unpack for this without struct or complicated view logic
    # Simplified approach: view as uint16, then astype
    # This is slow per-pixel in python, but we only do it once.
    
    # x_bits = (val & 0xFFFF)
    # y_bits = (val >> 16)
    # But doing this vectorized is better.
    # We will assume this helper is used inside loop or modify offset_splatv_positions to be efficient.
    # Actually offset_splatv_positions works on the whole array.
    pass

def transform_splatv(splatv_data, offset=(0, 0, 0), scale=1.0, rotation=(0, 0, 0)):
    """
    Apply position offset, scale, and rotation to splatv texture data.
    
    Args:
        splatv_data: dict from read_splatv_file
        offset: (x, y, z) position offset
        scale: scale factor
        rotation: (rx, ry, rz) rotation in degrees
    
    Returns: modified texdata
    """
    texdata = splatv_data['texdata'].copy()
    texdata_f = texdata.view(np.float32)
    
    num = splatv_data['count']
    
    # 1. Update Positions
    # We need to extract all positions, apply transform, and put back.
    # Strided access to positions: texdata_f[0::16], [1::16], [2::16]
    
    # Construct rotation matrix
    R = rotation_matrix(*rotation)
    
    # Extract positions
    # texdata_f is a flat array of floats.
    # Positions are at indices 16*j + 0, 1, 2
    # Reshape to (N, 16) to make it easier
    N = len(texdata) // 16 # Note: len(texdata) is size in uint32s.
    
    # Create a view that is (N, 16) float32
    # CAUTION: texdata is uint32. texdata_f is float32 view of same memory.
    # modifying flat_view modifies texdata.
    flat_view = texdata_f.reshape(-1, 16)
    
    # P_new = (R @ P_old^T)^T * scale + offset
    #       = P_old @ R^T * scale + offset
    
    positions = flat_view[:num, 0:3] # Shape (num, 3)
    
    # Rotation
    rotated_pos = positions @ R.T
    
    # Scale & Offset
    final_pos = rotated_pos * scale + np.array(offset)
    
    # Write back
    flat_view[:num, 0:3] = final_pos
    
    # 2. Update Rotations
    # Rotations are stored as packed half-floats at indices 3 and 4 (floats)
    # i.e. indices 3 and 4 of the float array contain the packed data?
    # Wait, texdata (uint32) index 3 and 4 contain the packed data.
    # float view at index 3 is meaningless because it interprets the bits of (half, half) as a float32.
    # We must access texdata (uint32) for rotation.
    
    # Indices in (N, 16) uint32 view:
    texdata_u32 = texdata.reshape(-1, 16)
    
    packed_rot_1 = texdata_u32[:num, 3] # low=x, high=y
    packed_rot_2 = texdata_u32[:num, 4] # low=z, high=w
    
    # We need to unpack halves. Numpy has float16.
    # packed is uint32. 
    # weak ref: x = packed & 0xFFFF, y = packed >> 16
    
    def unpack_halves(packed):
        low = (packed & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        high = (packed >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
        return low, high
        
    r0, r1 = unpack_halves(packed_rot_1) # x, y
    r2, r3 = unpack_halves(packed_rot_2) # z, w
    
    # Combine to (N, 4)
    quats = np.stack([r0, r1, r2, r3], axis=1) # (N, 4)
    
    # Global rotation quaternion
    q_rot = euler_to_quat(*rotation)
    
    # Multiply q_rot * q_old for each q_old
    # q_mult logic vectorized:
    # Q1 * Q2
    # w1, x1, y1, z1 = q_rot
    # w2, x2, y2, z2 = columns of quats (assuming x,y,z,w or w,x,y,z?)
    # Let's assume input is [x, y, z, w].
    
    x1, y1, z1, w1 = q_rot
    x2, y2, z2, w2 = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    
    # Result
    # x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    # y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    # z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    # w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    
    qx = w1*x2 + x1*w2 + y1*z2 - z1*y2
    qy = w1*y2 - x1*z2 + y1*w2 + z1*x2
    qz = w1*z2 + x1*y2 - y1*x2 + z1*w2
    qw = w1*w2 - x1*x2 - y1*y2 - z1*z2
    
    # Normalize
    norms = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    # Avoid div by zero
    norms[norms < 1e-6] = 1.0
    
    qx /= norms
    qy /= norms
    qz /= norms
    qw /= norms
    
    # Pack back
    def pack_halves(l, h):
        l_f16 = l.astype(np.float16).view(np.uint16)
        h_f16 = h.astype(np.float16).view(np.uint16)
        return l_f16.astype(np.uint32) | (h_f16.astype(np.uint32) << 16)
        
    new_packed_1 = pack_halves(qx, qy)
    new_packed_2 = pack_halves(qz, qw)
    
    texdata_u32[:num, 3] = new_packed_1
    texdata_u32[:num, 4] = new_packed_2
    
    return texdata


def merge_texdata(texdata1, count1, texdata2, count2):
    """
    Merge two texture data arrays.
    
    Returns: (merged_texdata, texwidth, texheight, total_count)
    """
    total_count = count1 + count2
    texwidth = 1024 * 4
    texheight = int(np.ceil((4 * total_count) / texwidth))
    
    merged = np.zeros((texwidth * texheight * 4,), dtype=np.uint32)
    
    # Copy first dataset
    for j in range(count1):
        merged[16 * j:16 * (j + 1)] = texdata1[16 * j:16 * (j + 1)]
    
    # Copy second dataset
    for j in range(count2):
        merged[16 * (count1 + j):16 * (count1 + j + 1)] = texdata2[16 * j:16 * (j + 1)]
    
    return merged, texwidth, texheight, total_count


def write_splatv_file(filepath, texdata, texwidth, texheight, cameras=None):
    """Write merged data to .splatv file."""
    if cameras is None:
        cameras = [{
            "id": 0,
            "img_name": "00001",
            "width": 800,
            "height": 800,
            "position": [0, 0, -3],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "fy": 800,
            "fx": 800,
        }]
    
    metadata = [{
        "type": "splat",
        "size": texdata.nbytes,
        "texwidth": texwidth,
        "texheight": texheight,
        "cameras": cameras
    }]
    
    json_bytes = json.dumps(metadata).encode('utf-8')
    
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', 0x674b))  # Magic
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(json_bytes)
        f.write(texdata.tobytes())
    
    print(f"Written: {filepath} ({texdata.nbytes / 1024 / 1024:.2f} MB)")


def get_positions_from_splatv(texdata, count):
    """Extract positions from splatv texdata."""
    texdata_f = texdata.view(np.float32)
    # Reshape to (N, 16) and take first 3 columns
    # Ensure we don't go out of bounds if texture is padded
    reshaped = texdata_f.reshape(-1, 16)
    return reshaped[:count, 0:3]


def interpolate_path(control_points, num_samples=100):
    """
    Interpolate control points using Natural Cubic Spline.

    Args:
        control_points: list of [x, y, z] positions
        num_samples: number of samples for path fitting

    Returns:
        positions: array of shape (num_samples, 3)
        t_values: array of parameter values [0, 1]
    """
    points = np.array(control_points)
    n = len(points)

    if n < 2:
        raise ValueError("Need at least 2 control points for path animation")

    if n == 2:
        # Linear interpolation
        t_values = np.linspace(0, 1, num_samples)
        positions = np.outer(1 - t_values, points[0]) + np.outer(t_values, points[1])
        return positions, t_values

    # Cubic spline interpolation
    # Parameter t based on cumulative chord length for more uniform spacing
    chord_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t_control = np.zeros(n)
    t_control[1:] = np.cumsum(chord_lengths)
    t_control /= t_control[-1]  # Normalize to [0, 1]

    # Create cubic splines for each dimension
    cs_x = CubicSpline(t_control, points[:, 0], bc_type='natural')
    cs_y = CubicSpline(t_control, points[:, 1], bc_type='natural')
    cs_z = CubicSpline(t_control, points[:, 2], bc_type='natural')

    t_values = np.linspace(0, 1, num_samples)
    positions = np.stack([cs_x(t_values), cs_y(t_values), cs_z(t_values)], axis=1)

    return positions, t_values


def fit_path_motion_coefficients(path_positions, t_values):
    """
    Fit polynomial coefficients for path motion.

    The splatv format uses:
        position(t) = base_pos + linear*dt + quadratic*dt² + cubic*dt³
    where dt = t - 0.5 (trbf_center)

    Args:
        path_positions: array of shape (N, 3) - positions along path
        t_values: array of parameter values [0, 1]

    Returns:
        base_position: position at t=0.5
        linear: [3] linear coefficients
        quadratic: [3] quadratic coefficients
        cubic: [3] cubic coefficients
    """
    # Convert t to dt (centered at 0.5)
    dt = t_values - 0.5

    # Find center position (at t=0.5)
    center_idx = np.argmin(np.abs(t_values - 0.5))
    base_position = path_positions[center_idx].copy()

    # Compute offsets from center position
    offsets = path_positions - base_position

    # Build design matrix for polynomial fitting: [dt, dt², dt³]
    A = np.column_stack([dt, dt**2, dt**3])

    # Fit coefficients for each dimension
    linear = np.zeros(3)
    quadratic = np.zeros(3)
    cubic = np.zeros(3)

    for dim in range(3):
        coeffs, _, _, _ = np.linalg.lstsq(A, offsets[:, dim], rcond=None)
        linear[dim] = coeffs[0]
        quadratic[dim] = coeffs[1]
        cubic[dim] = coeffs[2]

    return base_position, linear, quadratic, cubic


def compute_path_yaw_params(path_positions, t_values):
    """
    Compute yaw angle parameters for path-following rotation.

    Returns yaw angle at center time and angular velocity (linear fit).

    Args:
        path_positions: array of shape (N, 3) - positions along path
        t_values: array of parameter values [0, 1]

    Returns:
        yaw_center: yaw angle at t=0.5 (radians)
        yaw_rate: angular velocity (radians per unit dt)
        yaw_angles: all yaw angles for debugging
    """
    # Compute tangent vectors
    tangents = np.zeros_like(path_positions)
    tangents[0] = path_positions[1] - path_positions[0]
    tangents[-1] = path_positions[-1] - path_positions[-2]
    for i in range(1, len(path_positions) - 1):
        tangents[i] = path_positions[i + 1] - path_positions[i - 1]

    # Normalize tangents
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    tangents = tangents / norms

    # Compute yaw angles from XZ plane tangent
    # The cat's forward direction is assumed to be +X in local space
    # We want the cat to face the tangent direction
    # atan2(z, x) gives angle from +X axis, which is what we need for Y-axis rotation
    yaw_angles = np.arctan2(-tangents[:, 2], tangents[:, 0])

    # Handle angle wrapping for continuous rotation
    for i in range(1, len(yaw_angles)):
        diff = yaw_angles[i] - yaw_angles[i-1]
        if diff > np.pi:
            yaw_angles[i:] -= 2 * np.pi
        elif diff < -np.pi:
            yaw_angles[i:] += 2 * np.pi

    # Find yaw at t=0.5
    center_idx = np.argmin(np.abs(t_values - 0.5))
    yaw_center = yaw_angles[center_idx]

    # Fit linear yaw rate: yaw(t) = yaw_center + yaw_rate * dt
    dt = t_values - 0.5
    yaw_deltas = yaw_angles - yaw_center

    # Linear least squares fit
    if np.abs(dt).sum() > 0:
        yaw_rate = np.sum(yaw_deltas * dt) / np.sum(dt ** 2)
    else:
        yaw_rate = 0.0

    return yaw_center, yaw_rate, yaw_angles


def yaw_to_quaternion_wxyz(yaw):
    """Convert yaw angle (radians) to quaternion [w, x, y, z] for Y-axis rotation."""
    half_yaw = yaw / 2
    return np.array([np.cos(half_yaw), 0, np.sin(half_yaw), 0])


def apply_path_rotation_to_splatv(texdata, count, yaw_center, yaw_rate, num_samples=20):
    """
    Apply Y-axis rotation to entire object for path-following.

    This rotates all Gaussian positions around the object's center point
    and fits polynomial motion coefficients by sampling at multiple time points.
    Also rotates each Gaussian's orientation quaternion.

    Args:
        texdata: splatv texture data (uint32 array)
        count: number of gaussians
        yaw_center: yaw angle at t=0.5 (radians)
        yaw_rate: angular velocity (radians per unit dt)
        num_samples: number of time samples for polynomial fitting
    """
    texdata_f = texdata.view(np.float32)
    texdata_u32 = texdata.reshape(-1, 16)

    def unpack_halves(packed):
        low = (packed & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        high = (packed >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
        return low, high

    def pack_halves(l, h):
        l_f16 = np.asarray(l).astype(np.float16).view(np.uint16)
        h_f16 = np.asarray(h).astype(np.float16).view(np.uint16)
        return l_f16.astype(np.uint32) | (h_f16.astype(np.uint32) << 16)

    flat_view = texdata_f.reshape(-1, 16)

    # 1. Extract all positions and compute object center
    positions = flat_view[:count, 0:3].copy()
    object_center = np.mean(positions, axis=0)

    print(f"    Object center (original): {object_center}")
    print(f"    Yaw center: {np.degrees(yaw_center):.1f}°")
    print(f"    Yaw rate: {np.degrees(yaw_rate):.1f}°/dt")
    print(f"    Yaw at t=0: {np.degrees(yaw_center - 0.5 * yaw_rate):.1f}°")
    print(f"    Yaw at t=1: {np.degrees(yaw_center + 0.5 * yaw_rate):.1f}°")

    # 2. Sample yaw angles at multiple time points for accurate polynomial fitting
    t_samples = np.linspace(0, 1, num_samples)
    dt_samples = t_samples - 0.5
    yaw_samples = yaw_center + yaw_rate * dt_samples
    center_idx = num_samples // 2

    # 3. Pre-compute rotation quaternion and omega (same for all Gaussians)
    q_yaw_center = yaw_to_quaternion_wxyz(yaw_center)

    # Sample quaternions and fit omega
    q_samples = np.array([yaw_to_quaternion_wxyz(y) for y in yaw_samples])
    q_base = q_samples[center_idx]
    q_deltas = q_samples - q_base

    # Fit linear omega: q_delta = omega * dt
    A_design = np.column_stack([dt_samples, dt_samples**2, dt_samples**3])
    global_omega = np.zeros(4)
    for dim in range(4):
        coeffs, _, _, _ = np.linalg.lstsq(A_design[:, 0:1], q_deltas[:, dim:dim+1], rcond=None)
        global_omega[dim] = coeffs[0, 0]

    print(f"    Global omega: {global_omega}")

    # 4. For each Gaussian, compute rotated positions at all sample times
    # Pre-compute cos and sin for efficiency
    cos_yaw_samples = np.cos(yaw_samples)
    sin_yaw_samples = np.sin(yaw_samples)

    for j in range(count):
        # Get local position relative to object center
        x_local = positions[j, 0] - object_center[0]
        y_local = positions[j, 1] - object_center[1]
        z_local = positions[j, 2] - object_center[2]

        # Compute rotated positions at all sample times
        x_samples = x_local * cos_yaw_samples + z_local * sin_yaw_samples
        z_samples = -x_local * sin_yaw_samples + z_local * cos_yaw_samples

        # Find position at t=0.5 (center)
        x_base = x_samples[center_idx]
        z_base = z_samples[center_idx]

        # Compute offsets from center position
        x_offsets = x_samples - x_base
        z_offsets = z_samples - z_base

        # Fit polynomial: offset = linear*dt + quad*dt² + cubic*dt³
        x_coeffs, _, _, _ = np.linalg.lstsq(A_design, x_offsets, rcond=None)
        z_coeffs, _, _, _ = np.linalg.lstsq(A_design, z_offsets, rcond=None)

        # New base position (rotated, centered at origin)
        flat_view[j, 0] = x_base
        flat_view[j, 1] = y_local
        flat_view[j, 2] = z_base

        # Add rotation-induced motion to existing motion coefficients
        m0, m1 = unpack_halves(texdata_u32[j, 8])      # linear x, y
        m2, m3 = unpack_halves(texdata_u32[j, 9])      # linear z, quad x
        m4, m5 = unpack_halves(texdata_u32[j, 10])     # quad y, z
        m6, m7 = unpack_halves(texdata_u32[j, 11])     # cubic x, y
        m8, _ = unpack_halves(texdata_u32[j, 12])      # cubic z

        m0 += x_coeffs[0]  # linear x
        m2 += z_coeffs[0]  # linear z
        m3 += x_coeffs[1]  # quad x
        m5 += z_coeffs[1]  # quad z
        m6 += x_coeffs[2]  # cubic x
        m8 += z_coeffs[2]  # cubic z

        texdata_u32[j, 8] = pack_halves(m0, m1)
        texdata_u32[j, 9] = pack_halves(m2, m3)
        texdata_u32[j, 10] = pack_halves(m4, m5)
        texdata_u32[j, 11] = pack_halves(m6, m7)
        texdata_u32[j, 12] = pack_halves(m8, 0)

        # 4. Rotate the Gaussian's base orientation quaternion
        rw, rx = unpack_halves(texdata_u32[j, 3])
        ry, rz = unpack_halves(texdata_u32[j, 4])

        pw, px, py, pz = q_yaw_center

        new_w = pw * rw - px * rx - py * ry - pz * rz
        new_x = pw * rx + px * rw + py * rz - pz * ry
        new_y = pw * ry - px * rz + py * rw + pz * rx
        new_z = pw * rz + px * ry - py * rx + pz * rw

        norm = np.sqrt(new_w**2 + new_x**2 + new_y**2 + new_z**2)
        if norm > 1e-6:
            new_w /= norm
            new_x /= norm
            new_y /= norm
            new_z /= norm

        texdata_u32[j, 3] = pack_halves(new_w, new_x)
        texdata_u32[j, 4] = pack_halves(new_y, new_z)

        # 5. Apply pre-computed global omega (same for all Gaussians)
        ow, ox = unpack_halves(texdata_u32[j, 13])
        oy, oz = unpack_halves(texdata_u32[j, 14])

        ow += global_omega[0]
        ox += global_omega[1]
        oy += global_omega[2]
        oz += global_omega[3]

        texdata_u32[j, 13] = pack_halves(ow, ox)
        texdata_u32[j, 14] = pack_halves(oy, oz)

    # Verify new center
    new_positions = flat_view[:count, 0:3]
    new_center = np.mean(new_positions, axis=0)
    print(f"    New object center: {new_center}")


def add_path_motion_to_splatv(texdata, count, path_base_offset, path_linear, path_quadratic, path_cubic):
    """
    Add path translation motion to existing splatv motion data.

    This modifies the texdata in-place, adding the path translation on top of
    the object's internal motion (e.g., cat walking animation).

    Args:
        texdata: splatv texture data (uint32 array)
        count: number of gaussians
        path_base_offset: [3] offset to add to base positions
        path_linear: [3] linear motion coefficients to add
        path_quadratic: [3] quadratic motion coefficients to add
        path_cubic: [3] cubic motion coefficients to add
    """
    texdata_f = texdata.view(np.float32)
    texdata_u32 = texdata.reshape(-1, 16)

    # Helper functions for half-float packing
    def unpack_halves(packed):
        low = (packed & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        high = (packed >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
        return low, high

    def pack_halves(l, h):
        l_f16 = np.asarray(l).astype(np.float16).view(np.uint16)
        h_f16 = np.asarray(h).astype(np.float16).view(np.uint16)
        return l_f16.astype(np.uint32) | (h_f16.astype(np.uint32) << 16)

    # Add offset to base positions
    flat_view = texdata_f.reshape(-1, 16)

    # Debug: show before state
    old_center = np.mean(flat_view[:count, 0:3], axis=0)
    print(f"    Before path offset, center at: {old_center}")
    print(f"    Adding path_base_offset: {path_base_offset}")

    flat_view[:count, 0] += path_base_offset[0]
    flat_view[:count, 1] += path_base_offset[1]
    flat_view[:count, 2] += path_base_offset[2]

    # Debug: show after state
    new_center = np.mean(flat_view[:count, 0:3], axis=0)
    print(f"    After path offset, center at: {new_center}")

    # Texture layout per gaussian (16 uint32 values):
    # [0-2]: position (float32 x, y, z)
    # [3]: rotation x, y (packed half)
    # [4]: rotation z, w (packed half)
    # [5]: scale x, y (packed half)
    # [6]: scale z, 0 (packed half)
    # [7]: RGBA (uint8 x 4)
    # [8]: motion_0, motion_1 (linear x, y)
    # [9]: motion_2, motion_3 (linear z, quadratic x)
    # [10]: motion_4, motion_5 (quadratic y, z)
    # [11]: motion_6, motion_7 (cubic x, y)
    # [12]: motion_8, 0 (cubic z, padding)
    # [13]: omega_0, omega_1 (rotation delta x, y)
    # [14]: omega_2, omega_3 (rotation delta z, w)
    # [15]: trbf_center, trbf_scale (packed half)

    for j in range(count):
        # Unpack existing motion
        m0, m1 = unpack_halves(texdata_u32[j, 8])      # linear x, y
        m2, m3 = unpack_halves(texdata_u32[j, 9])      # linear z, quad x
        m4, m5 = unpack_halves(texdata_u32[j, 10])     # quad y, z
        m6, m7 = unpack_halves(texdata_u32[j, 11])     # cubic x, y
        m8, _ = unpack_halves(texdata_u32[j, 12])      # cubic z

        # Add path motion
        m0 += path_linear[0]
        m1 += path_linear[1]
        m2 += path_linear[2]
        m3 += path_quadratic[0]
        m4 += path_quadratic[1]
        m5 += path_quadratic[2]
        m6 += path_cubic[0]
        m7 += path_cubic[1]
        m8 += path_cubic[2]

        # Pack back
        texdata_u32[j, 8] = pack_halves(m0, m1)
        texdata_u32[j, 9] = pack_halves(m2, m3)
        texdata_u32[j, 10] = pack_halves(m4, m5)
        texdata_u32[j, 11] = pack_halves(m6, m7)
        texdata_u32[j, 12] = pack_halves(m8, 0)
    

def main():
    parser = argparse.ArgumentParser(
        description="Merge .splat and .splatv files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple merge
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv
    
    # With position offset (move object up by 1 unit)
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv --offset 0 1 0
    
    # With rotation (rotate object 90 degrees around X axis)
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv --rotate 90 0 0
    
    # Auto-snap to floor (requires lumina-path)
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv --lumina-path path.json --snap-to-floor
        """
    )
    
    parser.add_argument("background", help="Background .splat file (static scene)")
    parser.add_argument("object", help="Object .splatv file (dynamic 4DGS)")
    parser.add_argument("-o", "--output", required=True, help="Output .splatv file")
    parser.add_argument("--offset", nargs=3, type=float, default=[0, 0, 0],
                        metavar=('X', 'Y', 'Z'),
                        help="Position offset for the 4DGS object (default: 0 0 0)")
    parser.add_argument("--rotate", nargs=3, type=float, default=[0, 0, 0],
                        metavar=('RX', 'RY', 'RZ'),
                        help="Rotation angles in degrees (default: 0 0 0)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for the 4DGS object (default: 1.0)")
    parser.add_argument("--snap-to-floor", action="store_true",
                        help="Automatically align object bottom to the height specified in lumina-path")
    
    parser.add_argument("--bg-offset", nargs=3, type=float, default=[0, 0, 0],
                        metavar=('X', 'Y', 'Z'),
                        help="Position offset for background (default: 0 0 0)")
    parser.add_argument("--bg-scale", type=float, default=1.0,
                        help="Scale factor for background (default: 1.0)")
    parser.add_argument("--bg-rotate", nargs=3, type=float, default=[0, 0, 0],
                        metavar=('X', 'Y', 'Z'),
                        help="Rotation for background in degrees (default: 0 0 0)")
    parser.add_argument("--lumina-path", help="Path to lumina_path.json to extract position offset")
    parser.add_argument("--animate", action="store_true",
                        help="Animate object along the path defined in lumina-path")
    parser.add_argument("--rotate-along-path", action="store_true",
                        help="Rotate object to face the path direction while animating")
    parser.add_argument("--yaw-offset", type=float, default=0.0,
                        help="Additional yaw offset in degrees for path rotation (default: 0)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Animation speed multiplier (default: 1.0)")

    args = parser.parse_args()
    
    print("=" * 50)
    print("Merging Splat Files")
    print("=" * 50)
    
    # 1. Load Object Data first to compute bounds if needed
    obj_file = Path(args.object)
    obj_data_raw = None
    if obj_file.suffix.lower() == '.splatv':
        obj_data_raw = read_splatv_file(args.object)
        obj_count = obj_data_raw['count']
        cameras = obj_data_raw['metadata'][0].get('cameras')
        # Extract positions for analysis
        raw_positions = get_positions_from_splatv(obj_data_raw['texdata'], obj_count).copy()
    elif obj_file.suffix.lower() == '.splat':
        obj_data_raw = read_splat_file(args.object)
        obj_count = obj_data_raw['count']
        cameras = None
        raw_positions = obj_data_raw['positions']
    else:
        raise ValueError(f"Unsupported object format: {obj_file.suffix}")

    # 2. Determine Base Offset and Path Animation
    lumina_target_y = None
    current_offset = np.array(args.offset)  # Start with manual offset
    path_animation_data = None  # Will hold (base_offset, linear, quadratic, cubic) if animate is True

    if args.lumina_path:
        try:
            with open(args.lumina_path, 'r') as f:
                lumina_data = json.load(f)
                if 'controlPoints' in lumina_data and len(lumina_data['controlPoints']) > 0:
                    control_points = [cp['position'] for cp in lumina_data['controlPoints']]

                    if args.animate and len(control_points) >= 2:
                        # Path animation mode
                        print(f"\n[PATH ANIMATION MODE]")
                        print(f"  Control points: {len(control_points)}")
                        for i, cp in enumerate(control_points):
                            print(f"    [{i}]: {cp}")

                        # Interpolate path
                        path_positions, t_values = interpolate_path(control_points)

                        # Fit motion coefficients
                        path_base, path_linear, path_quad, path_cubic = fit_path_motion_coefficients(
                            path_positions, t_values
                        )

                        print(f"\n  Path motion coefficients:")
                        print(f"    Base (t=0.5): {path_base}")
                        print(f"    Linear: {path_linear}")
                        print(f"    Quadratic: {path_quad}")
                        print(f"    Cubic: {path_cubic}")

                        # Compute rotation along path if requested
                        yaw_center = None
                        yaw_rate = None
                        if args.rotate_along_path:
                            yaw_center, yaw_rate, yaw_angles = compute_path_yaw_params(path_positions, t_values)

                            # Apply user-specified yaw offset
                            yaw_offset_rad = np.radians(args.yaw_offset)
                            yaw_center += yaw_offset_rad
                            yaw_angles = yaw_angles + yaw_offset_rad

                            print(f"\n  Path rotation (Y-axis):")
                            print(f"    Yaw offset: {args.yaw_offset}°")
                            print(f"    Yaw at t=0: {np.degrees(yaw_angles[0]):.1f}°")
                            print(f"    Yaw at t=0.5: {np.degrees(yaw_center):.1f}°")
                            print(f"    Yaw at t=1: {np.degrees(yaw_angles[-1]):.1f}°")
                            print(f"    Yaw rate: {np.degrees(yaw_rate):.1f}°/dt")
                            print(f"    Total rotation: {np.degrees(yaw_angles[-1] - yaw_angles[0]):.1f}°")

                        # Store for later application
                        path_animation_data = (path_base, path_linear, path_quad, path_cubic, yaw_center, yaw_rate)

                        # For snap-to-floor, use the Y value at path center
                        lumina_target_y = path_base[1]

                        # Don't add offset here - it will be added via motion coefficients
                        # Verify path endpoints
                        expected_start = path_base + path_linear * (-0.5) + path_quad * 0.25 + path_cubic * (-0.125)
                        expected_end = path_base + path_linear * 0.5 + path_quad * 0.25 + path_cubic * 0.125

                        print(f"\n  Path control points:")
                        print(f"    Start (t=0): {control_points[0]}")
                        print(f"    End (t=1): {control_points[-1]}")
                        print(f"  Fitted path endpoints:")
                        print(f"    Start (t=0): {expected_start}")
                        print(f"    End (t=1): {expected_end}")
                        print(f"  Path length: {np.linalg.norm(np.diff(path_positions, axis=0), axis=1).sum():.2f} units")
                    else:
                        # Static placement mode (original behavior)
                        start_pos = control_points[0]
                        lumina_offset = np.array(start_pos)
                        print(f"Loaded offset from {args.lumina_path}: {lumina_offset}")
                        current_offset = current_offset + lumina_offset
                        lumina_target_y = lumina_offset[1]
                else:
                    print(f"Warning: No control points found in {args.lumina_path}")
        except Exception as e:
            print(f"Error reading lumina path: {e}")
            import traceback
            traceback.print_exc()
            
    # 3. Apply Snap to Floor Logic
    if args.snap_to_floor:
        if lumina_target_y is not None:
            # Compute rotated & scaled Y bounds
            R = rotation_matrix(*args.rotate)
            rotated_positions = raw_positions @ R.T
            scaled_positions = rotated_positions * args.scale

            min_y = np.min(scaled_positions[:, 1])
            max_y = np.max(scaled_positions[:, 1])
            print(f"\n[SNAP TO FLOOR]")
            print(f"  Target Floor Y (from Lumina): {lumina_target_y:.4f}")
            print(f"  Object Bounds (Rotated & Scaled): Min Y = {min_y:.4f}, Max Y = {max_y:.4f}")

            # Snap adjustment: align object bottom (max_y) to floor level
            snap_adjustment = -max_y
            print(f"  Calculated Snap Adjustment: {snap_adjustment:.4f}")

            if path_animation_data is not None:
                # In path animation mode, adjust the path base position
                path_base, path_linear, path_quad, path_cubic, yaw_center, yaw_rate = path_animation_data
                path_base = path_base.copy()
                path_base[1] += snap_adjustment
                path_animation_data = (path_base, path_linear, path_quad, path_cubic, yaw_center, yaw_rate)
                print(f"  Adjusted path base Y: {path_base[1]:.4f}")
            else:
                # Static placement mode
                current_offset[1] += snap_adjustment
                print(f"  Final Y Offset: {current_offset[1]:.4f}")
        else:
            print("Warning: --snap-to-floor requires --lumina-path to determine ground level. Skipping snap.")
            
    print(f"Final Object Offset: {current_offset}")
    print(f"Object Rotation: {args.rotate}")
    print(f"Object Scale: {args.scale}")

    # Read background
    bg_file = Path(args.background)
    if bg_file.suffix.lower() == '.splat':
        bg_data = read_splat_file(args.background)
        bg_texdata, _, _ = splat_to_texdata(
            bg_data, 
            offset=tuple(args.bg_offset),
            scale=args.bg_scale,
            rotation=tuple(args.bg_rotate)
        )
        bg_count = bg_data['count']
    elif bg_file.suffix.lower() == '.splatv':
        bg_splatv = read_splatv_file(args.background)
        # Use transform_splatv for background too if it's splatv
        bg_texdata = transform_splatv(
            bg_splatv,
            offset=tuple(args.bg_offset),
            scale=args.bg_scale,
            rotation=tuple(args.bg_rotate)
        )
        bg_count = bg_splatv['count']
    else:
        raise ValueError(f"Unsupported background format: {bg_file.suffix}")
    
    # Process Object (using previously loaded data)
    if obj_file.suffix.lower() == '.splatv':
        if path_animation_data is not None:
            # For path animation, transform without the path offset first
            # (path offset is handled via motion coefficients)
            obj_texdata = transform_splatv(
                obj_data_raw,
                offset=tuple(args.offset),  # Only manual offset, not lumina
                scale=args.scale,
                rotation=tuple(args.rotate)
            )
        else:
            obj_texdata = transform_splatv(
                obj_data_raw,
                offset=tuple(current_offset),
                scale=args.scale,
                rotation=tuple(args.rotate)
            )
    elif obj_file.suffix.lower() == '.splat':
        obj_texdata, _, _ = splat_to_texdata(
            obj_data_raw,
            offset=tuple(current_offset),
            scale=args.scale,
            rotation=tuple(args.rotate)
        )

    # Apply path animation if enabled
    if path_animation_data is not None and obj_file.suffix.lower() == '.splatv':
        path_base, path_linear, path_quad, path_cubic, yaw_center, yaw_rate = path_animation_data

        print(f"\n[Applying Path Animation]")
        print(f"  Rotation along path: {yaw_center is not None}")
        print(f"  Full path traversal (speed controlled by viewer)")

        # First apply rotation if enabled (this modifies positions and adds rotation motion)
        if yaw_center is not None and yaw_rate is not None:
            print(f"\n  Applying Y-axis rotation to object:")
            apply_path_rotation_to_splatv(
                obj_texdata,
                obj_count,
                yaw_center,
                yaw_rate
            )

        # Add path translation (full path, no scaling)
        add_path_motion_to_splatv(
            obj_texdata,
            obj_count,
            path_base,
            path_linear,
            path_quad,
            path_cubic
        )

        print(f"  Path motion applied to {obj_count:,} gaussians")
    
    print(f"\nBackground: {bg_count:,} gaussians")
    print(f"Object: {obj_count:,} gaussians")
    
    # Merge
    merged, texwidth, texheight, total = merge_texdata(
        bg_texdata, bg_count,
        obj_texdata, obj_count
    )
    
    print(f"Total: {total:,} gaussians")
    print(f"Texture: {texwidth}x{texheight}")
    
    # Write output
    write_splatv_file(args.output, merged, texwidth, texheight, cameras)
    print("\nDone!")


if __name__ == "__main__":
    main()
