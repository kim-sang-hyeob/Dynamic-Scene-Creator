import os
import json
import cv2
import numpy as np
import math

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
            print(f"[JSON-Sync] Loaded map_transform from {map_transform_path}")
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
    print(f"[JSON-Sync] Saved map_transform to {map_transform_path}")

def sync_video_with_json(video_path, json_path, original_video_path, output_dir, map_transform=None, resize=None, max_frames=None, remove_bg=False, use_midas=True):
    """
    Synchronizes the JSON tracking data (from original video) with the Diffusion-generated video.
    Diffusion video is often faster (sped up) than the original tracking sequence.

    Args:
        resize: Optional tuple (width, height) or float scale factor (e.g., 0.5 for half size)
        max_frames: Optional int to limit number of frames (uniformly sampled, includes first and last)
        remove_bg: If True, remove background using BiRefNet (creates transparent PNGs)
        use_midas: If True, use MiDaS for depth-based point initialization (default: True)
    """
    # Ensure absolute paths for safety on server
    output_dir = os.path.abspath(output_dir)
    video_path = os.path.abspath(video_path)
    json_path = os.path.abspath(json_path)
    original_video_path = os.path.abspath(original_video_path)

    print(f"[JSON-Sync] Base output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Initialize background remover if requested
    bg_remover = None
    if remove_bg:
        print(f"[JSON-Sync] Background removal enabled, loading BiRefNet...")
        try:
            from src.adapters.background_remover import BiRefNetRemover
            bg_remover = BiRefNetRemover()
        except ImportError as e:
            print(f"[Error] Failed to load BiRefNet: {e}")
            print("[Error] Install with: pip install transformers<4.40 timm einops kornia")
            return False

    # 1. Load JSON data
    if not os.path.exists(json_path):
        print(f"[Error] JSON file not found: {json_path}")
        return False
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    json_frames = json_data['frames']
    
    # 2. Get Video Info
    cap_out = cv2.VideoCapture(video_path)
    if not cap_out.isOpened():
        print(f"[Error] Could not open output video: {video_path}")
        return False
        
    fps_out = cap_out.get(cv2.CAP_PROP_FPS)
    total_frames_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_out = total_frames_out / fps_out
    
    cap_orig = cv2.VideoCapture(original_video_path)
    if not cap_orig.isOpened():
        print(f"[Error] Could not open original video: {original_video_path}")
        cap_out.release()
        return False
        
    fps_orig = cap_orig.get(cv2.CAP_PROP_FPS)
    total_frames_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_orig = total_frames_orig / fps_orig
    
    print(f"[JSON-Sync] Original: {duration_orig:.2f}s ({total_frames_orig} frames)")
    print(f"[JSON-Sync] Output: {duration_out:.2f}s ({total_frames_out} frames)")

    # Time mapping: Video diffusion compresses entire original into shorter output
    time_scale = duration_orig / duration_out
    print(f"[JSON-Sync] Time scale: {time_scale:.3f}x (diffusion sped up the video)")

    # Parse resize parameter
    resize_dims = None
    if resize is not None:
        if isinstance(resize, (int, float)) and resize != 1.0:
            # Scale factor (e.g., 0.5)
            orig_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resize_dims = (int(orig_w * resize), int(orig_h * resize))
            print(f"[JSON-Sync] Resizing frames: {orig_w}x{orig_h} -> {resize_dims[0]}x{resize_dims[1]} (scale={resize})")
        elif isinstance(resize, (tuple, list)) and len(resize) == 2:
            # Explicit dimensions (width, height)
            resize_dims = (int(resize[0]), int(resize[1]))
            print(f"[JSON-Sync] Resizing frames to: {resize_dims[0]}x{resize_dims[1]}")

    # Calculate which frames to extract (uniform sampling if max_frames specified)
    if max_frames is not None and max_frames < total_frames_out:
        # Uniform sampling: include first and last frame
        # For N frames from M total: indices = [0, M/(N-1), 2*M/(N-1), ..., M-1]
        frame_indices = []
        for i in range(max_frames):
            idx = int(i * (total_frames_out - 1) / (max_frames - 1)) if max_frames > 1 else 0
            frame_indices.append(idx)
        print(f"[JSON-Sync] Sampling {max_frames} frames uniformly from {total_frames_out} (first={frame_indices[0]}, last={frame_indices[-1]})")
    else:
        frame_indices = list(range(total_frames_out))

    # 3. Process Frames
    extracted_frames = []
    print(f"[JSON-Sync] Extracting {len(frame_indices)} frames to {img_dir}...")

    # Read all frames first if we're sampling
    all_frames_data = []
    for i in range(total_frames_out):
        ret, frame = cap_out.read()
        if not ret:
            print(f"[Warning] Failed to read frame {i} from video.")
            break
        all_frames_data.append(frame)

    for out_idx, i in enumerate(frame_indices):
        if i >= len(all_frames_data):
            print(f"[Warning] Frame index {i} out of range.")
            break
        frame = all_frames_data[i]

        # Current time in output video (based on original frame index)
        t_out = i / fps_out
        # Corresponding time in the JSON/Original scale
        t_orig = t_out * time_scale

        # Find closest frame in JSON (Binary search or simple look-up)
        # Assuming JSON is ordered by time
        closest_idx = 0
        min_diff = float('inf')
        for j, (f_data) in enumerate(json_frames):
            diff = abs(f_data['time'] - t_orig)
            if diff < min_diff:
                min_diff = diff
                closest_idx = j
            elif diff > min_diff: # Optimization: already passed the closest point
                break

        target_json = json_frames[closest_idx]

        # Resize frame if requested
        if resize_dims is not None:
            frame = cv2.resize(frame, resize_dims, interpolation=cv2.INTER_AREA)

        # Apply background removal if enabled
        if bg_remover is not None:
            rgba = bg_remover.remove_background(frame)
            # Save as PNG with alpha channel
            frame_name = f"{out_idx:04d}.png"
            save_path = os.path.join(img_dir, frame_name)
            from PIL import Image
            Image.fromarray(rgba).save(save_path)
            success = True
        else:
            # Save Frame (use sequential output index for clean naming)
            frame_name = f"{out_idx:04d}.png"
            save_path = os.path.join(img_dir, frame_name)
            success = cv2.imwrite(save_path, frame)

        if not success:
            print(f"[Error] Failed to write frame to {save_path}")
        elif out_idx == 0:
            print(f"[JSON-Sync] First frame saved successfully: {save_path}")

        # Progress reporting (especially useful for slow background removal)
        if bg_remover is not None and (out_idx + 1) % 10 == 0:
            print(f"[JSON-Sync] Progress: {out_idx + 1}/{len(frame_indices)} frames processed")

        # Normalized time for 4DGS (0 to 1 based on output sequence)
        normalized_time = out_idx / (len(frame_indices) - 1) if len(frame_indices) > 1 else 0.0

        # Convert Camera Pose to COLMAP/3DGS format if needed
        extracted_frames.append({
            "file_path": frame_name,
            "time": normalized_time, # Normalized [0,1] for 4DGS
            "original_time": t_orig,
            "camPos": get_cam_pos(target_json),
            "camRot": get_cam_rot(target_json),
            "objPos": get_obj_pos(target_json)
        })

    cap_out.release()
    cap_orig.release()
    
    # Save the synchronized metadata for 4DGS
    sync_meta = os.path.join(output_dir, "sync_metadata.json")
    with open(sync_meta, "w") as f:
        json.dump(extracted_frames, f, indent=4)
        
    # Generate 4DGS timestamps.json (normalized to [0, 1])
    # 4DGS expects timestamps in [0, 1] range for proper temporal modeling
    n_frames = len(extracted_frames)
    timestamps_dict = {}
    for i, frame in enumerate(extracted_frames):
        # Normalize time to [0, 1] based on frame index
        normalized_time = i / (n_frames - 1) if n_frames > 1 else 0.0
        timestamps_dict[frame["file_path"]] = normalized_time

    with open(os.path.join(output_dir, "timestamps.json"), "w") as f:
        json.dump(timestamps_dict, f, indent=4)

    print(f"[Timestamps] Normalized {n_frames} frames to [0, 1] range")
        
    # 4. Determine map transform (load from file, use provided, or default)
    if map_transform is None:
        map_transform = get_map_transform(output_dir)

    # Save map_transform.json for reproducibility and camera rotation patch
    save_map_transform(output_dir, map_transform)

    # 5. Generate COLMAP sparse/0 (SfM) files to bypass VGGT
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    write_colmap_text(extracted_frames, sparse_dir, img_dir, map_transform)

    # 6. Generate transforms_train.json for 4DGS (D-NeRF format)
    write_transforms_json(extracted_frames, output_dir, map_transform)

    print(f"[JSON-Sync] Successfully synced {len(extracted_frames)} frames.")
    print(f"[JSON-Sync] Generated SfM (COLMAP) files in {sparse_dir}. You can skip SfM!")
    print(f"[JSON-Sync] Generated transforms_train.json for 4DGS.")
    return sync_meta

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

def write_colmap_text(frames, output_dir, img_dir=None, map_transform=None):
    """
    Writes cameras.txt, images.txt, and points3D.txt in COLMAP format.
    Includes proper Unity (LHS) to COLMAP (RHS) conversion with map transform normalization.

    For images with alpha channel, initial points are generated by back-projecting
    foreground pixels to 3D, ensuring Gaussians start where the object is.
    """
    if map_transform is None:
        map_transform = get_map_transform()

    # 1. cameras.txt - Get actual image dimensions
    width, height = 1280, 720  # Default fallback
    has_alpha = False
    if img_dir is None:
        img_dir = os.path.join(os.path.dirname(output_dir), "images")

    # Try to read first image to get actual dimensions and check for alpha
    if os.path.exists(img_dir):
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    height, width = img.shape[:2]
                    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
                    print(f"[COLMAP] Using actual image dimensions: {width}x{height}")
                    if has_alpha:
                        print(f"[COLMAP] Alpha channel detected - will use foreground-based point initialization")
                    break

    # Unity default vertical FOV is 60 degrees
    # focal = height / (2 * tan(vfov/2))
    unity_vfov = math.radians(60)
    focal = height / (2 * math.tan(unity_vfov / 2))
    with open(os.path.join(output_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}\n")

    # Build map rotation matrix for camera rotation normalization
    map_rot = np.array(map_transform['rotation'])
    rx, ry, rz = np.radians(map_rot)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_map = Rz @ Ry @ Rx
    R_map_inv = R_map.T

    # 2. images.txt
    with open(os.path.join(output_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        for i, frame in enumerate(frames):
            p = frame['camPos']
            q = frame['camRot']

            # --- Coordinate Conversion with Normalization ---
            # 1. Convert Unity Quaternion to Rotation Matrix (LHS)
            qx, qy, qz, qw = q['x'], q['y'], q['z'], q['w']
            R_unity = quat_to_mat(qx, qy, qz, qw)
            C_unity = np.array([p['x'], p['y'], p['z']])

            # 2. Normalize position (undo map transform)
            C_local = normalize_position(C_unity, map_transform)

            # 3. Normalize rotation (undo map rotation)
            R_local = R_map_inv @ R_unity

            # 4. Convert LHS to RHS (COLMAP: Y-down)
            S = np.diag([1, -1, 1])
            R_rhs = S @ R_local @ S
            C_rhs = S @ C_local

            # 5. Convert Camera-to-World to World-to-Camera
            R_w2c = R_rhs.T
            t_w2c = -R_w2c @ C_rhs

            # 6. Convert Matrix back to Quaternion for COLMAP images.txt
            qw_c, qx_c, qy_c, qz_c = mat_to_quat(R_w2c)

            # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            f.write(f"{i+1} {qw_c} {qx_c} {qy_c} {qz_c} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {frame['file_path']}\n\n")

    # 3. points3D.txt (Foreground-based point cloud seeding)
    # If images have alpha channel, back-project foreground pixels to 3D
    # This ensures initial Gaussians are placed where the object actually is
    with open(os.path.join(output_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        pt_idx = 1

        if has_alpha and os.path.exists(img_dir):
            print(f"[COLMAP] Generating foreground-based initial points from alpha masks...")

            # Try to load MiDaS depth estimator for better depth estimation
            depth_estimator = None
            if use_midas:
                try:
                    from src.adapters.depth_estimator import DepthEstimator
                    depth_estimator = DepthEstimator("MiDaS_small")
                    print(f"[COLMAP] Using MiDaS for depth estimation (better point quality)")
                except Exception as e:
                    print(f"[COLMAP] MiDaS not available ({e}), using distance-based depth")
            else:
                print(f"[COLMAP] MiDaS disabled, using distance-based depth")

            all_points = []

            # Sample from more frames for better coverage (10 frames instead of 5)
            sample_frames = frames[::max(1, len(frames)//10)]

            for frame_idx, frame in enumerate(sample_frames):
                img_path = os.path.join(img_dir, frame['file_path'])
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None or img.shape[2] != 4:
                    continue

                alpha = img[:, :, 3]
                foreground_mask = alpha > 127

                # Get foreground pixel coordinates
                fy, fx = np.where(foreground_mask)
                if len(fx) == 0:
                    continue

                # Sample up to 500 foreground pixels per frame (increased from 200)
                n_samples = min(500, len(fx))
                indices = np.random.choice(len(fx), n_samples, replace=False)
                sampled_x = fx[indices]
                sampled_y = fy[indices]

                # Get camera pose for this frame
                p = frame['camPos']
                q = frame['camRot']
                qx, qy, qz, qw = q['x'], q['y'], q['z'], q['w']
                R_unity = quat_to_mat(qx, qy, qz, qw)
                C_unity = np.array([p['x'], p['y'], p['z']])

                # Normalize camera position and rotation
                C_local = normalize_position(C_unity, map_transform)
                map_rot = np.array(map_transform['rotation'])
                rx, ry, rz = np.radians(map_rot)
                Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                R_map = Rz @ Ry @ Rx
                R_map_inv = R_map.T
                R_local = R_map_inv @ R_unity

                # Convert to NeRF convention (flip Z)
                flip = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
                R_nerf = flip @ R_local @ flip.T
                C_nerf = flip @ C_local

                # Get reference depth (distance to object center)
                obj_pos = frame.get('objPos', {'x': 0, 'y': 0, 'z': 0})
                obj_unity = np.array([obj_pos['x'], obj_pos['y'], obj_pos['z']])
                obj_local = normalize_position(obj_unity, map_transform)
                obj_nerf = flip @ obj_local
                ref_depth = np.linalg.norm(obj_nerf - C_nerf)
                if ref_depth < 0.1:
                    ref_depth = 2.0  # Fallback depth

                # Get per-pixel depth using MiDaS or fallback to uniform depth
                if depth_estimator is not None:
                    try:
                        # Get relative depth map
                        rel_depth_map = depth_estimator.estimate(img, normalize=True)

                        # Scale relative depth to metric depth using reference distance
                        # Median foreground depth should match reference depth
                        fg_depths = rel_depth_map[foreground_mask]
                        if len(fg_depths) > 0:
                            median_rel = np.median(fg_depths)
                            if median_rel > 0.01:
                                depth_scale = ref_depth / median_rel
                            else:
                                depth_scale = ref_depth
                        else:
                            depth_scale = ref_depth

                        use_midas_depth = True
                    except Exception as e:
                        print(f"[COLMAP] Depth estimation failed for {frame['file_path']}: {e}")
                        use_midas_depth = False
                else:
                    use_midas_depth = False

                # Back-project pixels to 3D
                cx, cy = width / 2, height / 2
                for px, py in zip(sampled_x, sampled_y):
                    # Normalized image coordinates
                    x_norm = (px - cx) / focal
                    y_norm = (py - cy) / focal

                    # Ray direction in camera space (looking along +Z in NeRF convention)
                    ray_dir_cam = np.array([x_norm, y_norm, 1.0])
                    ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)

                    # Transform to world space
                    ray_dir_world = R_nerf @ ray_dir_cam

                    # Get depth for this pixel
                    if use_midas_depth:
                        # Use MiDaS depth with small random variation
                        pixel_depth = rel_depth_map[py, px] * depth_scale
                        pixel_depth = pixel_depth * (0.95 + 0.1 * np.random.random())
                    else:
                        # Fallback: uniform depth with larger random variation
                        pixel_depth = ref_depth * (0.8 + 0.4 * np.random.random())

                    point_3d = C_nerf + ray_dir_world * pixel_depth

                    # Get color from image
                    b, g, r = img[py, px, :3]
                    all_points.append((point_3d, r, g, b))

                print(f"[COLMAP] Processed frame {frame_idx+1}/{len(sample_frames)}: {frame['file_path']}")

            # Write unique points (remove duplicates by rounding)
            seen = set()
            for pt, r, g, b in all_points:
                key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))  # Coarser rounding for more points
                if key not in seen:
                    seen.add(key)
                    f.write(f"{pt_idx} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0\n")
                    pt_idx += 1

            print(f"[COLMAP] Generated {pt_idx - 1} foreground-based initial points")

        else:
            # Fallback: use object position-based seeding (original behavior)
            print(f"[COLMAP] Using object position-based point initialization (no alpha)")
            for frame in frames:
                obj_pos = frame.get('objPos', {'x':0, 'y':0, 'z':0})
                obj_unity = np.array([obj_pos['x'], obj_pos['y'], obj_pos['z']])

                # Normalize object position (same as camera)
                obj_local = normalize_position(obj_unity, map_transform)

                # Apply same flip as transforms_json (Z flip for NeRF convention)
                ox, oy, oz = obj_local[0], obj_local[1], -obj_local[2]

                # Sample a few points around the object to give 4DGS a dense start
                for dx in [-0.1, 0, 0.1]:
                    for dy in [-0.1, 0, 0.1]:
                        for dz in [-0.1, 0, 0.1]:
                            f.write(f"{pt_idx} {ox+dx} {oy+dy} {oz+dz} 128 128 128 0\n")
                            pt_idx += 1

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
    map_rot = np.array(map_transform['rotation'])

    # Create rotation matrix from Euler angles (XYZ order, Unity convention)
    rx, ry, rz = np.radians(map_rot)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_map = Rz @ Ry @ Rx  # Unity uses ZYX order for applying rotations
    R_map_inv = R_map.T

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
