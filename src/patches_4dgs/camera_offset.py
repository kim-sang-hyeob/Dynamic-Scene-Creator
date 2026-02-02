#!/usr/bin/env python3
"""
Patch 4DGS dataset_readers.py to support CAMERA_ANGLE_OFFSET environment variable.

This script adds support for rotating video cameras used in rendering.
Works with both Colmap and Blender data loaders.

Usage:
    python patch_4dgs_camera_offset.py /path/to/4dgs/scene/dataset_readers.py

Or for this project:
    python patch_4dgs_camera_offset.py ../external/4dgs/scene/dataset_readers.py
"""

import sys
import re
import shutil
from pathlib import Path

# Helper function to add to dataset_readers.py
ROTATE_CAM_FUNCTION = '''
# === CAMERA_ANGLE_OFFSET PATCH ===
import math as _math
import json as _json

def _generate_video_cameras_with_offset(path, template_transformsfile, extension, maxtime, angle_offset):
    """Generate 360-degree orbit cameras with angle offset.

    This replaces generateCamerasFromTransforms when CAMERA_ANGLE_OFFSET is set.
    The orbit starts at -180 + angle_offset instead of -180.
    """
    import torch as _torch
    from PIL import Image as _Image
    from pathlib import Path as _Path

    print(f"[PATCH] Generating video cameras with {angle_offset} degree offset", flush=True)

    trans_t = lambda t : _torch.Tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).float()

    rot_phi = lambda phi : _torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : _torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = _torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w

    # Apply offset to the orbit angles
    start_angle = -180 + angle_offset
    end_angle = 180 + angle_offset
    render_poses = _torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(start_angle, end_angle, 160+1)[:-1]], 0)
    render_times = _torch.linspace(0, maxtime, render_poses.shape[0])

    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = _json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])

    # Load a single image to get image info
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image = _Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image, (800, 800))
        break

    cam_infos = []
    for idx, (time, poses) in enumerate(zip(render_times, render_poses)):
        time = time / maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                        time=time, mask=None))

    print(f"[PATCH] Generated {len(cam_infos)} cameras, orbit from {start_angle} to {end_angle}", flush=True)
    return cam_infos

def _generate_colmap_video_cameras(train_cam_infos, angle_offset, source_path):
    """Rotate training cameras around the object center by angle_offset degrees.

    The rotation center can be specified via CAMERA_ROTATION_CENTER env var (x,y,z format).
    If not specified, it reads from sync_metadata.json to find the object position.
    Camera rotates around this point (the object) on Y-axis.
    """
    print(f"[PATCH] Rotating training cameras by {angle_offset} degrees", flush=True)

    if angle_offset == 0:
        return train_cam_infos

    # Y-axis rotation matrix (3x3)
    offset_rad = angle_offset * np.pi / 180.0
    cos_a = np.cos(offset_rad)
    sin_a = np.sin(offset_rad)
    rot_y = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float64)

    # Get rotation center from environment variable or sync_metadata.json
    rotation_center = None
    center_env = os.environ.get('CAMERA_ROTATION_CENTER', '')
    if center_env:
        try:
            coords = [float(x.strip()) for x in center_env.split(',')]
            if len(coords) == 3:
                rotation_center = np.array(coords)
                print(f"[PATCH] Using rotation center from env: {rotation_center}", flush=True)
        except:
            pass

    if rotation_center is None:
        # Try to read from sync_metadata.json + map_transform.json
        sync_meta_path = os.path.join(source_path, 'sync_metadata.json')
        map_transform_path = os.path.join(source_path, 'map_transform.json')
        if os.path.exists(sync_meta_path):
            try:
                with open(sync_meta_path) as f:
                    sync_data = _json.load(f)
                # Get average object position from first few frames
                obj_positions = []
                for frame in sync_data[:10]:
                    if 'objPos' in frame:
                        obj_positions.append([frame['objPos']['x'], frame['objPos']['y'], frame['objPos']['z']])
                if obj_positions:
                    avg_obj_unity = np.mean(obj_positions, axis=0)
                    # Load map transform from file or use defaults
                    if os.path.exists(map_transform_path):
                        with open(map_transform_path) as f:
                            mt = _json.load(f)
                            map_pos = np.array(mt['position'])
                            map_scale = np.array(mt['scale'])
                        print(f"[PATCH] Loaded map_transform from {map_transform_path}", flush=True)
                    else:
                        # Fallback to defaults
                        map_pos = np.array([-150.85, -30.0, 3.66])
                        map_scale = np.array([3, 3, 3])
                        print(f"[PATCH] Using default map_transform (no map_transform.json found)", flush=True)
                    obj_local = (avg_obj_unity - map_pos) / map_scale
                    # Apply Z flip for NeRF convention
                    rotation_center = np.array([obj_local[0], obj_local[1], -obj_local[2]])
                    print(f"[PATCH] Computed rotation center from sync_metadata: {rotation_center}", flush=True)
            except Exception as e:
                print(f"[PATCH] Warning: Could not read sync_metadata.json: {e}", flush=True)

    if rotation_center is None:
        # Fallback: estimate from camera direction
        first_cam = train_cam_infos[0]
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
        look_distance = 4.5
        rotation_center = cam_pos + look_dir * look_distance
        print(f"[PATCH] Estimated rotation center from camera direction: {rotation_center}", flush=True)

    print(f"[PATCH] Final rotation center: {rotation_center}", flush=True)

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

        # Rotate camera position around rotation_center (object position)
        relative_pos = cam_pos - rotation_center
        new_relative_pos = rot_y @ relative_pos
        new_cam_pos = rotation_center + new_relative_pos

        # Rotate camera orientation
        new_c2w = np.eye(4)
        new_c2w[:3,:3] = rot_y @ c2w[:3,:3]
        new_c2w[:3,3] = new_cam_pos

        # Convert back to w2c
        new_w2c = np.linalg.inv(new_c2w)
        new_w2c_rot = new_w2c[:3,:3]
        new_w2c_trans = new_w2c[:3,3]

        # Apply 4DGS storage transform
        new_R = -new_w2c_rot.T
        new_R[:,0] = -new_R[:,0]
        new_T = -new_w2c_trans

        cam_infos.append(CameraInfo(
            uid=idx, R=new_R, T=new_T,
            FovY=cam.FovY, FovX=cam.FovX,
            image=cam.image,
            image_path=cam.image_path, image_name=cam.image_name,
            width=cam.width, height=cam.height,
            time=cam.time, mask=cam.mask))

    print(f"[PATCH] Rotated {len(cam_infos)} cameras by {angle_offset} degrees around object center", flush=True)
    return cam_infos
# === END PATCH ===
'''

def patch_dataset_readers(file_path):
    """Patch dataset_readers.py to support CAMERA_ANGLE_OFFSET env var for Colmap."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return False

    content = file_path.read_text()

    # Check if already patched
    if "CAMERA_ANGLE_OFFSET" in content:
        print("[Patch] Already applied: CAMERA_ANGLE_OFFSET support")
        return True

    # Create backup
    backup_path = file_path.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy(file_path, backup_path)
        print(f"[Patch] Created backup: {backup_path}")

    # 1. Add the rotate function after imports
    # Find a good place to insert (after the imports, before first class)
    insert_marker = "class CameraInfo"
    if insert_marker not in content:
        print(f"[Error] Could not find '{insert_marker}' to insert helper function")
        return False

    content = content.replace(insert_marker, ROTATE_CAM_FUNCTION + "\n" + insert_marker)
    print("[Patch] Added _rotate_cam_infos helper function")

    # 2. Patch Colmap loader: video_cameras=train_cam_infos -> use orbit generation with offset
    # Colmap uses 'path' variable for source path
    colmap_pattern = r'video_cameras=train_cam_infos,'
    colmap_replacement = "video_cameras=_generate_colmap_video_cameras(train_cam_infos, float(os.environ.get('CAMERA_ANGLE_OFFSET', 0)), path),"

    if re.search(colmap_pattern, content):
        content = re.sub(colmap_pattern, colmap_replacement, content)
        print("[Patch] Patched Colmap loader (readColmapSceneInfo)")
    else:
        print("[Warning] Could not find Colmap video_cameras=train_cam_infos pattern")

    # 3. Patch Blender loader (readNerfSyntheticInfo): replace generateCamerasFromTransforms call
    # Find: video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    # Replace with our offset version
    blender_pattern = r'video_cam_infos = generateCamerasFromTransforms\(path, "transforms_train.json", extension, max_time\)'
    blender_replacement = "video_cam_infos = _generate_video_cameras_with_offset(path, 'transforms_train.json', extension, max_time, float(os.environ.get('CAMERA_ANGLE_OFFSET', 0)))"

    if re.search(blender_pattern, content):
        content = re.sub(blender_pattern, blender_replacement, content, count=1)
        print("[Patch] Patched Blender loader (readNerfSyntheticInfo) - video camera generation")
    else:
        print("[Warning] Could not find Blender generateCamerasFromTransforms pattern")

    # Write patched content
    file_path.write_text(content)
    print(f"[Patch] Successfully patched {file_path.name}")
    return True

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nDefault path for this project:")
        print("  /data/ephemeral/home/4dgs_project/external/4dgs/scene/dataset_readers.py")
        sys.exit(1)

    file_path = sys.argv[1]
    success = patch_dataset_readers(file_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
