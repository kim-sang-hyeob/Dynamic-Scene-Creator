"""
Unity ↔ NeRF/COLMAP coordinate system transformations.

Unity: Left-handed, Y-up, Z-forward
NeRF/COLMAP: Right-handed, Y-up, Z-backward
"""

import os
import json
import math
import numpy as np


# Default map transform (can be overridden via map_transform.json in project directory)
DEFAULT_MAP_TRANSFORM = {
    'position': [-150.85, -30.0, 3.66],
    'rotation': [0, 0, 0],
    'scale': [3, 3, 3]
}


def get_json_value(frame, new_key, old_key, default=None):
    """Get value from frame dict supporting both new (snake_case) and old (camelCase) key formats."""
    if new_key in frame:
        return frame[new_key]
    if old_key in frame:
        return frame[old_key]
    return default


def euler_to_quaternion(euler_dict):
    """Convert Euler angles (in degrees) to quaternion {x, y, z, w}.
    Uses Unity's rotation order: ZXY (yaw, pitch, roll).
    """
    # Convert degrees to radians
    pitch = math.radians(euler_dict.get('x', 0))  # X-axis rotation
    yaw = math.radians(euler_dict.get('y', 0))    # Y-axis rotation
    roll = math.radians(euler_dict.get('z', 0))   # Z-axis rotation

    # Unity uses ZXY order (yaw, pitch, roll)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # ZXY order quaternion multiplication
    w = cy * cp * cr + sy * sp * sr
    x = cy * sp * cr + sy * cp * sr
    y = sy * cp * cr - cy * sp * sr
    z = cy * cp * sr - sy * sp * cr

    return {'x': x, 'y': y, 'z': z, 'w': w}


def get_cam_pos(frame):
    """Get camera position from frame, supporting both formats."""
    pos = get_json_value(frame, 'cam_pos', 'camPos')
    if pos is None:
        raise KeyError("Neither 'cam_pos' nor 'camPos' found in frame")
    return pos


def get_cam_rot(frame):
    """Get camera rotation as quaternion from frame.
    Supports both quaternion format (camRot with w) and Euler format (cam_rot without w).
    """
    rot = get_json_value(frame, 'cam_rot', 'camRot')
    if rot is None:
        raise KeyError("Neither 'cam_rot' nor 'camRot' found in frame")

    # Check if it's Euler angles (no 'w' key) or quaternion (has 'w' key)
    if 'w' not in rot:
        # Convert Euler to quaternion
        return euler_to_quaternion(rot)
    return rot


def get_obj_pos(frame):
    """Get object position from frame, supporting both formats."""
    pos = get_json_value(frame, 'obj_pos', 'objPos', {'x': 0, 'y': 0, 'z': 0})
    return pos


def load_map_transform(project_dir):
    """Load map transform from file if exists, otherwise return None."""
    map_transform_path = os.path.join(project_dir, "map_transform.json")
    if os.path.exists(map_transform_path):
        with open(map_transform_path, 'r') as f:
            data = json.load(f)
            print(f"[Coordinate] Loaded map_transform from {map_transform_path}")
            return {
                'position': np.array(data['position']),
                'rotation': np.array(data['rotation']),
                'scale': np.array(data['scale'])
            }
    return None


def save_map_transform(project_dir, map_transform):
    """Save map transform to file for reproducibility and camera rotation patch."""
    map_transform_path = os.path.join(project_dir, "map_transform.json")
    data = {
        'position': map_transform['position'].tolist() if isinstance(map_transform['position'], np.ndarray) else list(map_transform['position']),
        'rotation': map_transform['rotation'].tolist() if isinstance(map_transform['rotation'], np.ndarray) else list(map_transform['rotation']),
        'scale': map_transform['scale'].tolist() if isinstance(map_transform['scale'], np.ndarray) else list(map_transform['scale']),
        '_description': 'Unity map transform for coordinate conversion. Used by camera rotation patch.'
    }
    with open(map_transform_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[Coordinate] Saved map_transform to {map_transform_path}")


def get_map_transform(project_dir=None):
    """Returns the Unity map transform for normalization.

    If project_dir is provided, tries to load from map_transform.json first.
    Otherwise returns the default transform.

    CRITICAL: The position must center on the OBJECT, not the camera!
    D-NeRF/4DGS expects the object at the origin with cameras orbiting around it.
    """
    # Try to load from file first
    if project_dir:
        loaded = load_map_transform(project_dir)
        if loaded:
            return loaded

    # Return default
    return {
        'position': np.array(DEFAULT_MAP_TRANSFORM['position']),
        'rotation': np.array(DEFAULT_MAP_TRANSFORM['rotation']),
        'scale': np.array(DEFAULT_MAP_TRANSFORM['scale'])
    }


def normalize_position(pos, map_transform=None):
    """Normalize a Unity world position to local coordinates."""
    if map_transform is None:
        map_transform = get_map_transform()

    map_pos = np.array(map_transform['position'])
    map_rot = np.array(map_transform['rotation'])
    map_scale = np.array(map_transform['scale'])

    # Create rotation matrix from Euler angles
    rx, ry, rz = np.radians(map_rot)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_map = Rz @ Ry @ Rx
    R_map_inv = R_map.T

    # Undo map transform
    C_local = pos - map_pos
    C_local = R_map_inv @ C_local
    C_local = C_local / map_scale

    return C_local


def quat_to_mat(x, y, z, w):
    """Converts a quaternion to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])


def mat_to_quat(R):
    """Converts a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return qw, qx, qy, qz


def build_map_rotation_matrix(map_transform):
    """Build rotation matrix from map transform for reuse."""
    map_rot = np.array(map_transform['rotation'])
    rx, ry, rz = np.radians(map_rot)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_map = Rz @ Ry @ Rx
    return R_map, R_map.T  # R_map, R_map_inv
