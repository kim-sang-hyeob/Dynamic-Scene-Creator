"""
Video frame extraction with Unity JSON synchronization.

Main entry point for the Unity → 4DGS data conversion pipeline.
"""

import os
import json
import cv2
import numpy as np

from .coordinate import (
    get_cam_pos,
    get_cam_rot,
    get_obj_pos,
    get_map_transform,
    save_map_transform,
)
from .colmap_writer import write_colmap_text
from .nerf_writer import write_transforms_json


def sync_video_with_json(video_path, json_path, original_video_path, output_dir,
                          map_transform=None, resize=None, max_frames=None,
                          remove_bg=False, use_midas=True):
    """
    Synchronizes the JSON tracking data (from original video) with the Diffusion-generated video.
    Diffusion video is often faster (sped up) than the original tracking sequence.

    Args:
        video_path: Path to the diffusion-generated video
        json_path: Path to Unity tracking JSON
        original_video_path: Path to original video (for timing reference)
        output_dir: Output directory for 4DGS dataset
        map_transform: Optional Unity map transform override
        resize: Optional tuple (width, height) or float scale factor (e.g., 0.5 for half size)
        max_frames: Optional int to limit number of frames (uniformly sampled, includes first and last)
        remove_bg: If True, remove background using BiRefNet (creates transparent PNGs)
        use_midas: If True, use MiDaS for depth-based point initialization (default: True)

    Returns:
        Path to sync_metadata.json if successful, False otherwise
    """
    # Ensure absolute paths for safety on server
    output_dir = os.path.abspath(output_dir)
    video_path = os.path.abspath(video_path)
    json_path = os.path.abspath(json_path)
    original_video_path = os.path.abspath(original_video_path)

    print(f"[FrameExtractor] Base output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Initialize background remover if requested
    bg_remover = None
    if remove_bg:
        print(f"[FrameExtractor] Background removal enabled, loading BiRefNet...")
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

    print(f"[FrameExtractor] Original: {duration_orig:.2f}s ({total_frames_orig} frames)")
    print(f"[FrameExtractor] Output: {duration_out:.2f}s ({total_frames_out} frames)")

    # Time mapping: Video diffusion compresses entire original into shorter output
    time_scale = duration_orig / duration_out
    print(f"[FrameExtractor] Time scale: {time_scale:.3f}x (diffusion sped up the video)")

    # Parse resize parameter
    resize_dims = None
    if resize is not None:
        if isinstance(resize, (int, float)) and resize != 1.0:
            # Scale factor (e.g., 0.5)
            orig_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resize_dims = (int(orig_w * resize), int(orig_h * resize))
            print(f"[FrameExtractor] Resizing frames: {orig_w}x{orig_h} -> {resize_dims[0]}x{resize_dims[1]} (scale={resize})")
        elif isinstance(resize, (tuple, list)) and len(resize) == 2:
            # Explicit dimensions (width, height)
            resize_dims = (int(resize[0]), int(resize[1]))
            print(f"[FrameExtractor] Resizing frames to: {resize_dims[0]}x{resize_dims[1]}")

    # Calculate which frames to extract (uniform sampling if max_frames specified)
    if max_frames is not None and max_frames < total_frames_out:
        frame_indices = []
        for i in range(max_frames):
            idx = int(i * (total_frames_out - 1) / (max_frames - 1)) if max_frames > 1 else 0
            frame_indices.append(idx)
        print(f"[FrameExtractor] Sampling {max_frames} frames uniformly from {total_frames_out} (first={frame_indices[0]}, last={frame_indices[-1]})")
    else:
        frame_indices = list(range(total_frames_out))

    # 3. Process Frames
    extracted_frames = []
    print(f"[FrameExtractor] Extracting {len(frame_indices)} frames to {img_dir}...")

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

        # Find closest frame in JSON
        closest_idx = 0
        min_diff = float('inf')
        for j, f_data in enumerate(json_frames):
            diff = abs(f_data['time'] - t_orig)
            if diff < min_diff:
                min_diff = diff
                closest_idx = j
            elif diff > min_diff:
                break

        target_json = json_frames[closest_idx]

        # Resize frame if requested
        if resize_dims is not None:
            frame = cv2.resize(frame, resize_dims, interpolation=cv2.INTER_AREA)

        # Apply background removal if enabled
        if bg_remover is not None:
            rgba = bg_remover.remove_background(frame)
            frame_name = f"{out_idx:04d}.png"
            save_path = os.path.join(img_dir, frame_name)
            from PIL import Image
            Image.fromarray(rgba).save(save_path)
            success = True
        else:
            frame_name = f"{out_idx:04d}.png"
            save_path = os.path.join(img_dir, frame_name)
            success = cv2.imwrite(save_path, frame)

        if not success:
            print(f"[Error] Failed to write frame to {save_path}")
        elif out_idx == 0:
            print(f"[FrameExtractor] First frame saved successfully: {save_path}")

        # Progress reporting
        if bg_remover is not None and (out_idx + 1) % 10 == 0:
            print(f"[FrameExtractor] Progress: {out_idx + 1}/{len(frame_indices)} frames processed")

        # Normalized time for 4DGS (0 to 1 based on output sequence)
        normalized_time = out_idx / (len(frame_indices) - 1) if len(frame_indices) > 1 else 0.0

        extracted_frames.append({
            "file_path": frame_name,
            "time": normalized_time,
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
    n_frames = len(extracted_frames)
    timestamps_dict = {}
    for i, frame in enumerate(extracted_frames):
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

    write_colmap_text(extracted_frames, sparse_dir, img_dir, map_transform, use_midas)

    # 6. Generate transforms_train.json for 4DGS (D-NeRF format)
    write_transforms_json(extracted_frames, output_dir, map_transform)

    print(f"[FrameExtractor] Successfully synced {len(extracted_frames)} frames.")
    print(f"[FrameExtractor] Generated SfM (COLMAP) files in {sparse_dir}. You can skip SfM!")
    print(f"[FrameExtractor] Generated transforms_train.json for 4DGS.")
    return sync_meta
