#!/usr/bin/env python3
"""
Patch 4DGS dataset_readers.py to support camera rotation via environment variables.

This simplified version generates a cleaner patch using the camera_transform module.

Environment variables:
    CAMERA_ANGLE_OFFSET: Rotation angle in degrees (default: 0)
    CAMERA_ROTATION_CENTER: Rotation center as "x,y,z" (optional, auto-detected if not set)

Usage:
    python patch_4dgs_camera_offset_v2.py /path/to/4dgs/scene/dataset_readers.py
"""

import sys
import re
import shutil
from pathlib import Path

# The patch code to inject into dataset_readers.py
PATCH_CODE = '''
# === CAMERA_ANGLE_OFFSET PATCH v2 ===
import json as _json

def _rotate_video_cameras(train_cam_infos, angle_offset, source_path):
    """
    Rotate training cameras around object center by angle_offset degrees.

    Uses camera_transform module for coordinate conversion and rotation.
    Falls back to inline implementation if module not available.
    """
    if angle_offset == 0:
        return train_cam_infos

    print(f"[PATCH] Rotating cameras by {angle_offset} degrees", flush=True)

    # Try to import camera_transform module
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(source_path).parent.parent / 'src'))
        from camera_transform import (
            CameraTransformer,
            CoordinateConverter,
            CameraFormatConverter,
            load_rotation_center_from_sync_metadata,
            estimate_rotation_center_from_camera
        )
        _use_module = True
    except ImportError:
        _use_module = False
        print("[PATCH] camera_transform module not found, using inline implementation", flush=True)

    # Get rotation center
    rotation_center = None

    # 1. Try environment variable
    center_env = os.environ.get('CAMERA_ROTATION_CENTER', '')
    if center_env:
        try:
            coords = [float(x.strip()) for x in center_env.split(',')]
            if len(coords) == 3:
                rotation_center = np.array(coords)
                print(f"[PATCH] Rotation center from env: {rotation_center}", flush=True)
        except:
            pass

    # 2. Try sync_metadata.json
    if rotation_center is None:
        sync_meta_path = os.path.join(source_path, 'sync_metadata.json')
        if _use_module:
            rotation_center = load_rotation_center_from_sync_metadata(sync_meta_path)
            if rotation_center is not None:
                print(f"[PATCH] Rotation center from sync_metadata: {rotation_center}", flush=True)
        elif os.path.exists(sync_meta_path):
            try:
                with open(sync_meta_path) as f:
                    sync_data = _json.load(f)
                obj_positions = []
                for frame in sync_data[:10]:
                    if 'objPos' in frame:
                        obj_positions.append([frame['objPos']['x'], frame['objPos']['y'], frame['objPos']['z']])
                if obj_positions:
                    avg_obj = np.mean(obj_positions, axis=0)
                    map_pos = np.array([-150.85, -30.0, 3.66])
                    map_scale = np.array([3, 3, 3])
                    obj_local = (avg_obj - map_pos) / map_scale
                    rotation_center = np.array([obj_local[0], obj_local[1], -obj_local[2]])
                    print(f"[PATCH] Rotation center from sync_metadata: {rotation_center}", flush=True)
            except Exception as e:
                print(f"[PATCH] Warning: Could not read sync_metadata: {e}", flush=True)

    # 3. Fallback: estimate from camera direction
    if rotation_center is None:
        first_cam = train_cam_infos[0]
        if _use_module:
            rotation_center = estimate_rotation_center_from_camera(
                np.array(first_cam.R), np.array(first_cam.T), look_distance=4.5
            )
        else:
            R_stored = np.array(first_cam.R)
            T_stored = np.array(first_cam.T)
            R_tmp = R_stored.copy()
            R_tmp[:,0] = -R_tmp[:,0]
            w2c_rot = -R_tmp.T
            w2c_trans = -T_stored
            w2c = np.eye(4)
            w2c[:3,:3] = w2c_rot
            w2c[:3,3] = w2c_trans
            c2w = np.linalg.inv(w2c)
            cam_pos = c2w[:3,3]
            look_dir = -c2w[:3,2]
            look_dir = look_dir / np.linalg.norm(look_dir)
            rotation_center = cam_pos + look_dir * 4.5
        print(f"[PATCH] Rotation center estimated: {rotation_center}", flush=True)

    # Rotate cameras
    if _use_module:
        transformer = CameraTransformer(rotation_center)
        cam_infos = []
        for idx, cam in enumerate(train_cam_infos):
            new_R, new_T = transformer.rotate_camera(
                np.array(cam.R), np.array(cam.T), angle_offset, axis='y'
            )
            cam_infos.append(CameraInfo(
                uid=idx, R=new_R, T=new_T,
                FovY=cam.FovY, FovX=cam.FovX,
                image=cam.image,
                image_path=cam.image_path, image_name=cam.image_name,
                width=cam.width, height=cam.height,
                time=cam.time, mask=cam.mask
            ))
    else:
        # Inline rotation implementation
        offset_rad = angle_offset * np.pi / 180.0
        cos_a, sin_a = np.cos(offset_rad), np.sin(offset_rad)
        rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float64)

        cam_infos = []
        for idx, cam in enumerate(train_cam_infos):
            R_stored = np.array(cam.R)
            T_stored = np.array(cam.T)
            R_tmp = R_stored.copy()
            R_tmp[:,0] = -R_tmp[:,0]
            w2c_rot = -R_tmp.T
            w2c_trans = -T_stored
            w2c = np.eye(4)
            w2c[:3,:3] = w2c_rot
            w2c[:3,3] = w2c_trans
            c2w = np.linalg.inv(w2c)
            cam_pos = c2w[:3,3]

            relative_pos = cam_pos - rotation_center
            new_cam_pos = rotation_center + rot_y @ relative_pos

            new_c2w = np.eye(4)
            new_c2w[:3,:3] = rot_y @ c2w[:3,:3]
            new_c2w[:3,3] = new_cam_pos

            new_w2c = np.linalg.inv(new_c2w)
            new_R = -new_w2c[:3,:3].T
            new_R[:,0] = -new_R[:,0]
            new_T = -new_w2c[:3,3]

            cam_infos.append(CameraInfo(
                uid=idx, R=new_R, T=new_T,
                FovY=cam.FovY, FovX=cam.FovX,
                image=cam.image,
                image_path=cam.image_path, image_name=cam.image_name,
                width=cam.width, height=cam.height,
                time=cam.time, mask=cam.mask
            ))

    print(f"[PATCH] Rotated {len(cam_infos)} cameras", flush=True)
    return cam_infos
# === END PATCH v2 ===
'''


def patch_file(file_path: Path) -> bool:
    """Apply camera rotation patch to dataset_readers.py."""

    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return False

    content = file_path.read_text()

    # Check if already patched
    if "CAMERA_ANGLE_OFFSET PATCH" in content:
        print("[Info] File already patched. Removing old patch first...")
        # Remove old patch
        content = re.sub(
            r'# === CAMERA_ANGLE_OFFSET PATCH.*?# === END PATCH.*?\n',
            '',
            content,
            flags=re.DOTALL
        )

    # Create backup
    backup_path = file_path.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy(file_path, backup_path)
        print(f"[Patch] Created backup: {backup_path}")

    # Insert patch code before CameraInfo class
    insert_marker = "class CameraInfo"
    if insert_marker not in content:
        print(f"[Error] Could not find '{insert_marker}'")
        return False

    content = content.replace(insert_marker, PATCH_CODE + "\n" + insert_marker)
    print("[Patch] Added rotation function")

    # Patch Colmap loader
    colmap_old = "video_cameras=train_cam_infos,"
    colmap_new = "video_cameras=_rotate_video_cameras(train_cam_infos, float(os.environ.get('CAMERA_ANGLE_OFFSET', 0)), path),"
    if colmap_old in content:
        content = content.replace(colmap_old, colmap_new)
        print("[Patch] Patched Colmap loader")

    # Patch Blender loader - find the generateCamerasFromTransforms call for video
    blender_pattern = r'video_cam_infos = generateCamerasFromTransforms\(path, "transforms_train.json", extension, max_time\)'
    blender_new = "video_cam_infos = _rotate_video_cameras(generateCamerasFromTransforms(path, 'transforms_train.json', extension, max_time), float(os.environ.get('CAMERA_ANGLE_OFFSET', 0)), path)"
    if re.search(blender_pattern, content):
        content = re.sub(blender_pattern, blender_new, content, count=1)
        print("[Patch] Patched Blender loader")

    file_path.write_text(content)
    print(f"[Patch] Successfully patched {file_path.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nDefault path:")
        print("  python patch_4dgs_camera_offset_v2.py ../external/4dgs/scene/dataset_readers.py")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    success = patch_file(file_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
