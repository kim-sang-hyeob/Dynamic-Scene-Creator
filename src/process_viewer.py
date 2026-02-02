#!/usr/bin/env python
"""
Process viewer output (video + transforms_train.json) → 4DGS training dataset.

Takes the output from web_viewer_final (or any video + NeRF-format transforms)
and creates a complete dataset ready for 4DGS training.

Pipeline:
  1. Extract frames from video (WebM/MP4)
  2. Subsample frames + camera data (--frames)
  3. Resize images (--resize)
  4. Create COLMAP sparse files from real camera poses
  5. Create timestamps.json for 4DGS

Usage:
    python src/process_viewer.py video.webm transforms_train.json --output data/my_scene
    python src/process_viewer.py video.webm transforms_train.json --output data/my_scene --frames 40 --resize 0.5
"""

import os
import sys
import json
import math
import argparse
import struct
import numpy as np
from pathlib import Path


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0][0] + R[1][1] + R[2][2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2][1] - R[1][2]) * s
        y = (R[0][2] - R[2][0]) * s
        z = (R[1][0] - R[0][1]) * s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s

    return w, x, y, z


def c2w_to_colmap(transform_matrix):
    """
    Convert camera-to-world (NeRF format) to COLMAP format (world-to-camera).

    NeRF transform_matrix: 4x4 camera-to-world
    COLMAP: quaternion (qw, qx, qy, qz) + translation (tx, ty, tz) in world-to-camera
    """
    c2w = np.array(transform_matrix)
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]

    # World-to-camera: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
    tx, ty, tz = t_w2c

    return qw, qx, qy, qz, tx, ty, tz


def extract_frames(video_path, output_dir, frame_indices=None, resize=None):
    """
    Extract frames from video using OpenCV.

    Args:
        video_path: Path to video file (WebM, MP4, etc.)
        output_dir: Directory to save frames
        frame_indices: List of frame indices to extract (None = all)
        resize: Scale factor (float) or (width, height) tuple

    Returns:
        List of saved file paths
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {video_path}")
    print(f"[Video] {total_frames} frames, {fps:.1f} fps, {width}x{height}")

    if frame_indices is None:
        frame_indices = list(range(total_frames))

    # Calculate resize dimensions
    new_size = None
    if resize is not None:
        if isinstance(resize, (int, float)):
            new_size = (int(width * resize), int(height * resize))
        else:
            new_size = resize
        print(f"[Video] Resizing to {new_size[0]}x{new_size[1]}")

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    frame_set = set(frame_indices)

    frame_idx = 0
    output_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_set:
            if new_size:
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            filename = f"frame_{output_idx:05d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)
            output_idx += 1

            if output_idx % 10 == 0:
                print(f"[Video] Extracted {output_idx}/{len(frame_indices)} frames")

        frame_idx += 1

    cap.release()
    print(f"[Video] Extracted {len(saved_paths)} frames to {output_dir}")
    return saved_paths


def uniform_sample_indices(total, max_count):
    """Uniformly sample indices, always including first and last."""
    if max_count >= total:
        return list(range(total))

    indices = [int(round(i * (total - 1) / (max_count - 1))) for i in range(max_count)]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def process_viewer_output(video_path, transforms_path, output_dir,
                          max_frames=None, resize=None, remove_bg=False):
    """
    Process viewer output into 4DGS training dataset.

    Args:
        video_path: Path to video file (WebM/MP4)
        transforms_path: Path to transforms_train.json
        output_dir: Output directory for dataset
        max_frames: Max number of frames (uniform sampling)
        resize: Scale factor or (w, h) tuple
        remove_bg: Remove background using BiRefNet
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load transforms_train.json
    print(f"\n[1/5] Loading camera data...")
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    frames_data = transforms['frames']
    total_json_frames = len(frames_data)
    print(f"[Camera] {total_json_frames} frames in transforms_train.json")

    # Camera intrinsics
    fl_x = transforms.get('fl_x', 800)
    fl_y = transforms.get('fl_y', fl_x)
    cx = transforms.get('cx', transforms.get('w', 1800) / 2)
    cy = transforms.get('cy', transforms.get('h', 985) / 2)
    w = transforms.get('w', 1800)
    h = transforms.get('h', 985)
    duration = transforms.get('duration', 5)
    fps = transforms.get('fps', 30)

    print(f"[Camera] Resolution: {w}x{h}, focal: {fl_x:.1f}, duration: {duration}s, fps: {fps}")

    # 2. Determine frame indices to use
    print(f"\n[2/5] Selecting frames...")
    if max_frames and max_frames < total_json_frames:
        selected_indices = uniform_sample_indices(total_json_frames, max_frames)
        print(f"[Frames] Subsampled {total_json_frames} → {len(selected_indices)} frames")
    else:
        selected_indices = list(range(total_json_frames))
        print(f"[Frames] Using all {len(selected_indices)} frames")

    selected_frames = [frames_data[i] for i in selected_indices]

    # 3. Extract frames from video
    print(f"\n[3/5] Extracting frames from video...")
    img_dir = os.path.join(output_dir, "images")
    saved_paths = extract_frames(video_path, img_dir, selected_indices, resize)

    if len(saved_paths) != len(selected_frames):
        print(f"[Warning] Frame count mismatch: video={len(saved_paths)}, json={len(selected_frames)}")
        # Use the minimum
        count = min(len(saved_paths), len(selected_frames))
        saved_paths = saved_paths[:count]
        selected_frames = selected_frames[:count]

    # Remove background if requested
    if remove_bg:
        print(f"\n[3.5/5] Removing background...")
        try:
            from src.background_remover import BiRefNetRemover
            remover = BiRefNetRemover()
            for path in saved_paths:
                remover.process_file(path, path)  # overwrite with transparent
            print(f"[BG] Processed {len(saved_paths)} images")
        except ImportError as e:
            print(f"[Warning] BiRefNet not available: {e}")

    # Calculate actual image size (after resize)
    if resize is not None:
        if isinstance(resize, (int, float)):
            actual_w = int(w * resize)
            actual_h = int(h * resize)
            actual_fl_x = fl_x * resize
            actual_fl_y = fl_y * resize
            actual_cx = cx * resize
            actual_cy = cy * resize
        else:
            actual_w, actual_h = resize
            scale_x = actual_w / w
            scale_y = actual_h / h
            actual_fl_x = fl_x * scale_x
            actual_fl_y = fl_y * scale_y
            actual_cx = cx * scale_x
            actual_cy = cy * scale_y
    else:
        actual_w, actual_h = w, h
        actual_fl_x, actual_fl_y = fl_x, fl_y
        actual_cx, actual_cy = cx, cy

    # 4. Create COLMAP sparse files
    print(f"\n[4/5] Creating COLMAP sparse files...")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # cameras.txt
    cameras_file = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {actual_w} {actual_h} {actual_fl_x} {actual_fl_y} {actual_cx} {actual_cy}\n")
    print(f"[Sparse] cameras.txt (PINHOLE {actual_w}x{actual_h}, f={actual_fl_x:.1f})")

    # images.txt - with REAL camera poses from transforms_train.json
    images_file = os.path.join(sparse_dir, "images.txt")
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, frame in enumerate(selected_frames):
            image_id = idx + 1
            tm = frame['transform_matrix']
            qw, qx, qy, qz, tx, ty, tz = c2w_to_colmap(tm)
            filename = f"frame_{idx:05d}.png"
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {filename}\n")
            f.write("\n")
    print(f"[Sparse] images.txt ({len(selected_frames)} images with real camera poses)")

    # points3D.txt (empty - SfM-free)
    points3d_file = os.path.join(sparse_dir, "points3D.txt")
    with open(points3d_file, 'w') as f:
        f.write("# 3D point list\n")
    print(f"[Sparse] points3D.txt (empty)")

    # 5. Create timestamps.json + updated transforms_train.json
    print(f"\n[5/5] Creating metadata files...")

    # timestamps.json
    timestamps = {}
    for idx, frame in enumerate(selected_frames):
        filename = f"frame_{idx:05d}.png"
        t = frame.get('time_normalized', idx / max(len(selected_frames) - 1, 1))
        timestamps[filename] = t

    timestamps_file = os.path.join(output_dir, "timestamps.json")
    with open(timestamps_file, 'w') as f:
        json.dump(timestamps, f, indent=2)
    print(f"[Meta] timestamps.json ({len(timestamps)} entries)")

    # Updated transforms_train.json (with correct file paths and intrinsics)
    output_transforms = {
        'camera_angle_x': transforms.get('camera_angle_x', 2 * math.atan(actual_w / (2 * actual_fl_x))),
        'camera_angle_y': transforms.get('camera_angle_y', 2 * math.atan(actual_h / (2 * actual_fl_y))),
        'fl_x': actual_fl_x,
        'fl_y': actual_fl_y,
        'cx': actual_cx,
        'cy': actual_cy,
        'w': actual_w,
        'h': actual_h,
        'duration': duration,
        'fps': fps,
        'frames': []
    }

    for idx, frame in enumerate(selected_frames):
        output_transforms['frames'].append({
            'file_path': f"./images/frame_{idx:05d}",
            'time': frame.get('time', 0),
            'time_normalized': frame.get('time_normalized', idx / max(len(selected_frames) - 1, 1)),
            'transform_matrix': frame['transform_matrix']
        })

    transforms_output = os.path.join(output_dir, "transforms_train.json")
    with open(transforms_output, 'w') as f:
        json.dump(output_transforms, f, indent=2)
    print(f"[Meta] transforms_train.json ({len(output_transforms['frames'])} frames)")

    # Summary
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Dataset created at: {output_dir}")
    print(f"{'='*60}")
    print(f"  images/             : {len(saved_paths)} PNG frames ({actual_w}x{actual_h})")
    print(f"  sparse/0/           : COLMAP format (real camera poses)")
    print(f"  transforms_train.json : NeRF format camera data")
    print(f"  timestamps.json     : Normalized timestamps for 4DGS")
    print(f"\nNext step:")
    print(f"  python manage.py train {os.path.relpath(output_dir)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process viewer output → 4DGS training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: extract all frames
  python src/process_viewer.py video.webm transforms_train.json --output data/my_scene

  # Subsample to 40 frames + resize to half
  python src/process_viewer.py video.webm transforms_train.json --output data/my_scene --frames 40 --resize 0.5

  # With background removal
  python src/process_viewer.py composited.mp4 transforms_train.json --output data/my_scene --frames 40 --remove-bg
        """
    )

    parser.add_argument("video", help="Input video (WebM, MP4, etc.)")
    parser.add_argument("transforms", help="transforms_train.json from viewer")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--frames", type=int, default=None,
                        help="Max number of frames (uniform sampling, includes first/last)")
    parser.add_argument("--resize", type=str, default=None,
                        help="Resize: scale (e.g., 0.5) or WxH (e.g., 384x216)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Remove background using BiRefNet")

    args = parser.parse_args()

    # Parse resize
    resize = None
    if args.resize:
        if 'x' in args.resize.lower():
            w, h = args.resize.lower().split('x')
            resize = (int(w), int(h))
        else:
            resize = float(args.resize)

    process_viewer_output(
        args.video,
        args.transforms,
        args.output,
        max_frames=args.frames,
        resize=resize,
        remove_bg=args.remove_bg
    )


if __name__ == "__main__":
    main()
