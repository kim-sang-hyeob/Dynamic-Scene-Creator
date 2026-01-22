"""
4DGS Custom Camera Renderer
Renders with rotated camera (e.g., 45 degrees left)
"""
import os
import sys
import json
import numpy as np
import math

def rotation_matrix_y(angle_deg):
    """Create Y-axis rotation matrix."""
    angle = math.radians(angle_deg)
    return np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)]
    ])

def rotate_camera_transforms(input_json, output_json, angle_deg=-45, pivot=None):
    """
    Rotate all camera transforms around Y-axis, centered on object (pivot).
    Negative angle = rotate left (counter-clockwise when viewed from above)

    Args:
        pivot: Center of rotation. If None, uses origin (0,0,0).
               For our black_cat scene, object is near origin after normalization.
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    R_y = rotation_matrix_y(angle_deg)

    if pivot is None:
        pivot = np.array([0.0, 0.0, 0.0])
    else:
        pivot = np.array(pivot)

    for frame in data['frames']:
        T = np.array(frame['transform_matrix'])

        # Extract rotation and position
        R = T[:3, :3]
        pos = T[:3, 3]

        # Rotate camera position around pivot (object center)
        # 1. Translate to pivot-centered coordinates
        pos_centered = pos - pivot
        # 2. Rotate around Y-axis
        pos_rotated = R_y @ pos_centered
        # 3. Translate back
        new_pos = pos_rotated + pivot

        # Rotate camera orientation to keep looking at object
        new_R = R_y @ R

        # Rebuild transform matrix
        T[:3, :3] = new_R
        T[:3, 3] = new_pos

        frame['transform_matrix'] = T.tolist()

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"[Done] Rotated cameras by {angle_deg} degrees around pivot {pivot}")
    print(f"[Done] Saved to {output_json}")

def render_rotated(scene_dir, model_dir, angle_deg=-45, output_video=None):
    """
    Full pipeline: render with rotated camera using CAMERA_ANGLE_OFFSET env var.

    This approach uses an environment variable to offset the spiral camera path
    in 4DGS's generateCamerasFromTransforms function. Requires a one-line change
    in dataset_readers.py (see setup instructions below).
    """
    import subprocess
    from datetime import datetime

    # Get scene name for output filename
    scene_name = os.path.basename(scene_dir.rstrip('/'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Render with CAMERA_ANGLE_OFFSET environment variable
    print(f"[1/2] Rendering with {angle_deg} degree camera offset...")
    render_script = os.path.join(os.getcwd(), "external/4dgs/render.py")

    # Set environment variable for camera angle offset
    env = os.environ.copy()
    env["CAMERA_ANGLE_OFFSET"] = str(angle_deg)

    subprocess.run([
        sys.executable, render_script,
        "-m", model_dir,
        "--skip_train", "--skip_test"
    ], cwd=os.path.join(os.getcwd(), "external/4dgs"), env=env)

    # Create video
    print("[2/2] Creating video...")
    video_dirs = sorted([d for d in os.listdir(os.path.join(model_dir, "video")) if d.startswith("ours_")])
    if video_dirs:
        latest = video_dirs[-1]
        renders_path = os.path.join(model_dir, "video", latest, "renders", "%05d.png")

        if output_video is None:
            # Include scene name, angle, and timestamp to avoid overwriting
            output_video = os.path.join(os.getcwd(), f"{scene_name}_rot{int(angle_deg)}deg_{timestamp}.mp4")

        subprocess.run([
            "ffmpeg", "-y", "-framerate", "24",
            "-i", renders_path,
            "-c:v", "mpeg4", "-q:v", "5",
            output_video
        ])
        print(f"\n[Done] Video saved to: {output_video}")
    else:
        print("[Error] No renders found")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rotate 4DGS camera transforms and render")
    parser.add_argument("scene_dir", help="Scene directory containing transforms_train.json")
    parser.add_argument("--angle", type=float, default=-45, help="Rotation angle in degrees (negative=left)")
    parser.add_argument("--model", default=None, help="Model output directory")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--render", action="store_true", help="Auto render and create video")
    args = parser.parse_args()

    if args.render:
        scene_name = os.path.basename(args.scene_dir.rstrip('/'))
        model_dir = args.model or os.path.join(os.getcwd(), "output/4dgs", scene_name)
        render_rotated(args.scene_dir, model_dir, args.angle, args.output)
    else:
        input_json = os.path.join(args.scene_dir, "transforms_train.json")
        output_json = os.path.join(args.scene_dir, "transforms_rotated.json")
        rotate_camera_transforms(input_json, output_json, args.angle)

if __name__ == "__main__":
    main()
