import os
import json
import math
import numpy as np
import subprocess

def extract_frames_from_video(video_path, output_dir, num_frames=12, interval=None):
    """
    Extracts evenly spaced frames from a video using ffmpeg.
    If interval is provided, it uses that instead of calculating from num_frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total frames
    cmd_dur = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", 
               "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video_path]
    try:
        total_frames = int(subprocess.check_output(cmd_dur).decode().strip())
    except:
        # Fallback if packet count fails
        total_frames = 144 # Default for 6s @ 24fps
    
    if interval is None:
        interval = max(1, total_frames // num_frames)
    else:
        num_frames = total_frames // interval
    
    # ffmpeg index is 1-based usually in select filter
    select_filter = "+".join([f"eq(n,{i*interval})" for i in range(num_frames)])
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='{select_filter}'",
        "-vsync", "vfr",
        os.path.join(output_dir, "%d.png")
    ]
    subprocess.check_call(cmd)
    
    # Rename 1.png, 2.png -> 0000.png, 0001.png for consistency and better SFM sorting
    for i in range(num_frames):
        src = os.path.join(output_dir, f"{i+1}.png")
        dst = os.path.join(output_dir, f"{i:04d}.png")
        if os.path.exists(src):
            os.rename(src, dst)

    print(f"Extracted {num_frames} frames to {output_dir}")

def get_isometric_matrix(angle_deg):
    """
    Generates a 4x4 matrix for an isometric camera at a specific rotation.
    Assumes: 
    - Elevation: 45 degrees (Matching user's prompt)
    - Distance: 10.0 (Further back for more parallel look)
    """
    elevation = math.radians(45)
    azimuth = math.radians(angle_deg)
    
    # Camera position in world space
    dist = 10.0
    x = dist * math.cos(elevation) * math.sin(azimuth)
    y = dist * math.sin(elevation) # Y is UP in Blender world
    z = dist * math.cos(elevation) * math.cos(azimuth)
    
    # Simple Look-at matrix towards origin (0,0,0)
    def normalize(v):
        return v / np.linalg.norm(v)

    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    
    z_axis = normalize(camera_pos - target)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    
    # Create 4x4 transform matrix (Camera to World)
    c2w = np.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = camera_pos
    
    return c2w.tolist()

def generate_isometric_transforms(output_dir, image_dir):
    """ Legacy: 4-view isometric (N, E, S, W) """
    angles = [0, 90, 180, 270]
    _generate_any_transforms(output_dir, image_dir, angles)

def generate_orbital_transforms(output_dir, image_dir, num_frames, span_degrees=360):
    """ New: Orbital rotation for N frames extracted from video with custom span """
    # Spread the frames across the specified span
    if num_frames > 1:
        step = span_degrees / (num_frames - 1)
    else:
        step = 0
    angles = [ step * i for i in range(num_frames) ]
    _generate_any_transforms(output_dir, image_dir, angles)

def _generate_any_transforms(output_dir, image_dir, angles):
    frames = []
    for i, angle in enumerate(angles):
        # We assume files are named 0.png, 1.png, etc.
        file_path = f"input/{i}"
        frames.append({
            "file_path": file_path,
            "rotation": 0,
            "transform_matrix": get_isometric_matrix(angle)
        })
    
    data = {
        "camera_angle_x": 0.7, # Changed from 0.2 to a more standard FOV (~40 deg)
        "frames": frames
    }
    
    train_json = os.path.join(output_dir, "transforms_train.json")
    with open(train_json, 'w') as f:
        json.dump(data, f, indent=4)
    
    test_json = os.path.join(output_dir, "transforms_test.json")
    with open(test_json, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Generated transforms for {len(angles)} views in {output_dir}")
