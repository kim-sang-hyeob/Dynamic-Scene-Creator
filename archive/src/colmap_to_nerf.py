"""
Converts COLMAP sparse reconstruction to D-NeRF/Blender format (transforms_train.json)
for 4DGS compatibility.
"""
import os
import json
import numpy as np
import math

def quat_to_rotmat(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

def parse_colmap_cameras(cameras_txt):
    """Parse cameras.txt to get camera intrinsics."""
    cameras = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def parse_colmap_images(images_txt):
    """Parse images.txt to get camera poses."""
    images = []
    with open(images_txt, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]

            images.append({
                'id': img_id,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': cam_id,
                'name': name
            })
        i += 1

    return images

def colmap_to_nerf_transform(qw, qx, qy, qz, tx, ty, tz):
    """
    Convert COLMAP camera pose to NeRF/Blender transform_matrix.

    COLMAP: world-to-camera (R, t) where t = -R @ C
    NeRF: camera-to-world 4x4 matrix
    """
    # COLMAP rotation matrix (world-to-camera)
    R_w2c = quat_to_rotmat(qw, qx, qy, qz)
    t_w2c = np.array([tx, ty, tz])

    # Convert to camera-to-world
    R_c2w = R_w2c.T
    C = -R_w2c.T @ t_w2c  # Camera center in world coordinates

    # NeRF/Blender uses a different convention:
    # OpenGL: X-right, Y-up, Z-backward (out of screen)
    # COLMAP: X-right, Y-down, Z-forward
    # We need to flip Y and Z axes

    # Flip matrix for coordinate conversion
    flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    R_nerf = flip @ R_c2w
    C_nerf = flip @ C

    # Build 4x4 transform matrix
    transform = np.eye(4)
    transform[:3, :3] = R_nerf
    transform[:3, 3] = C_nerf

    return transform.tolist()

def compute_camera_angle_x(camera):
    """Compute horizontal field of view from camera intrinsics."""
    if camera['model'] == 'PINHOLE':
        fx = camera['params'][0]
        width = camera['width']
        fov_x = 2 * math.atan(width / (2 * fx))
        return fov_x
    elif camera['model'] == 'SIMPLE_PINHOLE':
        f = camera['params'][0]
        width = camera['width']
        fov_x = 2 * math.atan(width / (2 * f))
        return fov_x
    else:
        # Default FOV if model not recognized
        return 0.6911  # ~39.6 degrees, common default

def convert_colmap_to_nerf(scene_dir, output_json=None, timestamps_json=None):
    """
    Main conversion function.

    Args:
        scene_dir: Directory containing sparse/0/ and images/
        output_json: Output path for transforms_train.json (default: scene_dir/transforms_train.json)
        timestamps_json: Path to timestamps.json for 4DGS time values
    """
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    images_dir = os.path.join(scene_dir, "images")

    if not os.path.exists(sparse_dir):
        print(f"[Error] Sparse directory not found: {sparse_dir}")
        return False

    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    images_txt = os.path.join(sparse_dir, "images.txt")

    if not os.path.exists(cameras_txt) or not os.path.exists(images_txt):
        print(f"[Error] cameras.txt or images.txt not found in {sparse_dir}")
        return False

    # Parse COLMAP data
    cameras = parse_colmap_cameras(cameras_txt)
    images = parse_colmap_images(images_txt)

    if not cameras or not images:
        print("[Error] Failed to parse COLMAP data")
        return False

    # Load timestamps if available (for 4DGS)
    timestamps = {}
    if timestamps_json is None:
        timestamps_json = os.path.join(scene_dir, "timestamps.json")

    if os.path.exists(timestamps_json):
        with open(timestamps_json, 'r') as f:
            ts_data = json.load(f)
            for item in ts_data:
                timestamps[item['file_path']] = item['timestamp']
        print(f"[Info] Loaded {len(timestamps)} timestamps from {timestamps_json}")

    # Get camera FOV from first camera
    first_cam = list(cameras.values())[0]
    camera_angle_x = compute_camera_angle_x(first_cam)

    # Sort images by name for consistent ordering
    images = sorted(images, key=lambda x: x['name'])

    # Build frames list
    frames = []
    for i, img in enumerate(images):
        # Get transform matrix
        transform = colmap_to_nerf_transform(
            img['qw'], img['qx'], img['qy'], img['qz'],
            img['tx'], img['ty'], img['tz']
        )

        # File path (relative, without extension for Blender format)
        name = img['name']
        base_name = os.path.splitext(name)[0]
        file_path = f"./images/{base_name}"

        # Time value for 4DGS
        time_val = timestamps.get(name, i / max(len(images) - 1, 1))

        frame = {
            "file_path": file_path,
            "rotation": 0.0,  # Not used by 4DGS
            "time": time_val,
            "transform_matrix": transform
        }
        frames.append(frame)

    # Build output JSON
    output = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }

    # Write output
    if output_json is None:
        output_json = os.path.join(scene_dir, "transforms_train.json")

    with open(output_json, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"[Success] Created {output_json} with {len(frames)} frames")
    print(f"[Info] camera_angle_x: {camera_angle_x:.4f} rad ({math.degrees(camera_angle_x):.1f} deg)")

    # Also create transforms_test.json and transforms_val.json (can be same as train for now)
    for split in ['test', 'val']:
        split_json = os.path.join(scene_dir, f"transforms_{split}.json")
        with open(split_json, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"[Info] Created {split_json}")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert COLMAP to D-NeRF format for 4DGS")
    parser.add_argument("scene_dir", help="Scene directory containing sparse/0/ and images/")
    parser.add_argument("--output", help="Output transforms_train.json path")
    parser.add_argument("--timestamps", help="Path to timestamps.json")
    args = parser.parse_args()

    convert_colmap_to_nerf(args.scene_dir, args.output, args.timestamps)
