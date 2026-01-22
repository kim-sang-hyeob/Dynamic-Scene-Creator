import os
import cv2
import json
import numpy as np

def extract_4dgs_data(video_path, output_dir, num_frames=None, fps=None):
    """
    Extracts frames and generates timestamps for 4DGS.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_dir = os.path.join(output_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if num_frames is None:
        num_frames = total_frames
    
    # Calculate step to match num_frames if specified
    step = max(1, total_frames // num_frames)
    
    timestamps = []
    count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % step == 0:
            frame_name = f"{extracted_count:04d}.png"
            cv2.imwrite(os.path.join(img_dir, frame_name), frame)
            
            # Normalized timestamp [0, 1]
            t = count / total_frames
            timestamps.append({
                "file_path": frame_name,
                "timestamp": t,
                "frame_idx": count
            })
            extracted_count += 1
            
        count += 1
    
    cap.release()
    
    # Save timestamps.json
    with open(os.path.join(output_dir, "timestamps.json"), "w") as f:
        json.dump(timestamps, f, indent=4)
        
    print(f"[4DGS] Extracted {extracted_count} frames to {img_dir}")
    print(f"[4DGS] Saved timestamps to {os.path.join(output_dir, 'timestamps.json')}")
    return True

def generate_4dgs_metadata(output_dir, video_info):
    """
    Optional: Generate additional metadata like duration, fps, etc.
    """
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(video_info, f, indent=4)
    return meta_path
