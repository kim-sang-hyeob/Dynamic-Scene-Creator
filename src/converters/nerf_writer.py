"""
NeRF/D-NeRF format transforms JSON generation.

Generates transforms_train.json, transforms_test.json, transforms_val.json
in Blender/D-NeRF format for 4DGS compatibility.
"""

import os
import json
import math
import numpy as np
import cv2

from .coordinate import (
    normalize_position,
    get_map_transform,
    quat_to_mat,
    build_map_rotation_matrix,
)


def write_transforms_json(frames, output_dir, map_transform=None):
    """
    Writes transforms_train.json, transforms_test.json, transforms_val.json
    in D-NeRF/Blender format for 4DGS compatibility.

    Args:
        frames: List of frame data with camPos, camRot, time
        output_dir: Output directory
        map_transform: Optional dict with Unity map transform to undo
    """
    # Camera intrinsics - get actual dimensions from images
    width, height = 1280, 720  # Default fallback
    img_dir = os.path.join(output_dir, "images")

    if os.path.exists(img_dir):
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    break

    # Unity default vertical FOV is 60 degrees
    # Convert to focal length: focal = height / (2 * tan(vfov/2))
    unity_vfov_deg = 60
    unity_vfov = math.radians(unity_vfov_deg)
    focal = height / (2 * math.tan(unity_vfov / 2))
    camera_angle_x = 2 * math.atan(width / (2 * focal))

    # Use shared map transform
    if map_transform is None:
        map_transform = get_map_transform()

    # Build inverse map transform to normalize camera positions
    R_map, R_map_inv = build_map_rotation_matrix(map_transform)

    nerf_frames = []
    for i, frame in enumerate(frames):
        p = frame['camPos']
        q = frame['camRot']

        # Convert Unity camera pose to NeRF/Blender transform_matrix
        qx, qy, qz, qw = q['x'], q['y'], q['z'], q['w']
        R_unity = quat_to_mat(qx, qy, qz, qw)
        C_unity = np.array([p['x'], p['y'], p['z']])

        # 1. Normalize position using shared function
        C_local = normalize_position(C_unity, map_transform)

        # 2. Normalize rotation (undo map rotation)
        R_local = R_map_inv @ R_unity

        # 2. Unity (LHS, Y-up) to NeRF/Blender (RHS, Y-up, Z-backward)
        # NeRF convention: camera looks along -Z in its local frame
        # Unity: X-right, Y-up, Z-forward (LHS)
        # NeRF:  X-right, Y-up, Z-backward (RHS)
        flip = np.array([
            [1,  0,  0],
            [0,  1,  0],
            [0,  0, -1]
        ])

        R_nerf = flip @ R_local @ flip.T
        C_nerf = flip @ C_local

        # Build 4x4 camera-to-world transform matrix
        transform = np.eye(4)
        transform[:3, :3] = R_nerf
        transform[:3, 3] = C_nerf

        # File path (relative, without extension for Blender format compatibility)
        base_name = os.path.splitext(frame['file_path'])[0]
        file_path = f"./images/{base_name}"

        nerf_frames.append({
            "file_path": file_path,
            "rotation": 0.0,
            "time": frame['time'],
            "transform_matrix": transform.tolist()
        })

    output_data = {
        "camera_angle_x": camera_angle_x,
        "frames": nerf_frames
    }

    # Write all three splits (train, test, val)
    for split in ['train', 'test', 'val']:
        json_path = os.path.join(output_dir, f"transforms_{split}.json")
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
