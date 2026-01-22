import os
import torch
import numpy as np
import rerun as rr
import argparse
from plyfile import PlyData

# Note: This script assumes you have the deformation.pth and point_cloud.ply
# However, to be implementation-agnostic, we can also work on the exported PLY sequence 
# from the 'gaussian_pertimestamp' folder.

def analyze_motion(input_dir, threshold=0.01):
    """
    Analyzes a sequence of PLY files to calculate the motion 'energy' of each Gaussian.
    """
    ply_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.ply')])
    if len(ply_files) < 2:
        print("Need at least 2 frames to analyze motion.")
        return

    print(f"Analyzing {len(ply_files)} frames for motion...")

    # Initialize accumulators
    all_positions = []
    
    # Load first frame to get structure
    first_ply = PlyData.read(ply_files[0])
    num_points = len(first_ply['vertex']['x'])
    
    # Sample every N frames if the sequence is too long
    step = max(1, len(ply_files) // 10) 
    sampled_files = ply_files[::step]

    for ply_path in sampled_files:
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        pos = np.stack([v['x'], v['y'], v['z']], axis=1)
        all_positions.append(pos)
    
    # Convert to numpy array [Time, Points, 3]
    all_positions = np.stack(all_positions, axis=0)
    
    # --- Robust Outlier Filtering ---
    # Calculate median position and spread of the base frame
    base_pos = all_positions[0]
    median_pos = np.median(base_pos, axis=0)
    dist_from_center = np.linalg.norm(base_pos - median_pos, axis=1)
    
    # Keep only points within a reasonable distance (e.g., 99th percentile of first frame)
    # This removes points that are effectively 'at infinity'
    valid_mask = dist_from_center < np.percentile(dist_from_center, 99.5)
    
    print(f"Filtering {np.sum(~valid_mask)} spatial outliers...")
    all_positions = all_positions[:, valid_mask, :]
    base_pos = base_pos[valid_mask, :]
    
    # Calculate displacement: Standard deviation over time for each point
    motion_energy = np.std(all_positions, axis=0).mean(axis=-1)
    
    # Further filter points that "fly off" to infinity during the sequence
    # (Checking if max motion energy is crazy high)
    sane_motion_mask = motion_energy < (np.median(motion_energy) + 10 * np.std(motion_energy))
    print(f"Filtering {np.sum(~sane_motion_mask)} motion outliers...")
    
    all_positions = all_positions[:, sane_motion_mask, :]
    motion_energy = motion_energy[sane_motion_mask]
    base_pos = base_pos[sane_motion_mask, :]
    
    # Normalize for visualization [0, 1]
    max_energy = np.percentile(motion_energy, 98) 
    norm_energy = np.clip(motion_energy / (max_energy + 1e-6), 0, 1)

    print(f"Motion analysis complete.")
    return motion_energy, norm_energy, all_positions

def log_to_rerun(pts_seq, raw_energy, norm_energy, fps, output_file="motion_analysis.rrd"):
    rr.init("4DGS Motion Analysis", spawn=False)
    rr.save(output_file)
    
    # 1. Log a Static Heatmap for reference
    colors_static = np.zeros((len(norm_energy), 3), dtype=np.uint8)
    colors_static[:, 0] = (norm_energy * 255).astype(np.uint8) # Red channel
    colors_static[:, 2] = ((1 - norm_energy) * 255).astype(np.uint8) # Blue channel
    
    try:
        rr.log("world/static_map", rr.Points3D(pts_seq[0], colors=colors_static, radii=0.006), static=True)
    except TypeError:
        rr.log("world/static_map", rr.Points3D(pts_seq[0], colors=colors_static, radii=0.006), timeless=True)

    # 2. Log the Dynamic Sequence colored by energy
    print(f"Logging dynamic sequence to Rerun...")
    for i, pts in enumerate(pts_seq):
        try:
            rr.set_time("frame", sequence=i)
        except (AttributeError, TypeError):
            rr.set_time_sequence("frame", i)

        # We keep the SAME colors for each point across the sequence
        # This helps track if a 'Red' point actually belongs to the moving object
        rr.log(
            "world/dynamic_stream",
            rr.Points3D(pts, colors=colors_static, radii=0.005)
        )
    
    print(f"Motion analysis (.rrd) saved to {output_file}")
    print("In Rerun, compared 'static_map' and 'dynamic_stream' side by side!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to gaussian_pertimestamp folder")
    parser.add_argument("--threshold", type=float, default=0.005, help="Energy threshold for 'static'")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output", type=str, default="motion_analysis.rrd")
    args = parser.parse_args()
    
    energy_raw, energy_norm, pts_seq = analyze_motion(args.input_dir, args.threshold)
    if pts_seq is not None:
        log_to_rerun(pts_seq, energy_raw, energy_norm, args.fps, args.output)
        
        # Count static vs dynamic
        is_static = energy_raw < args.threshold
        n_static = np.sum(is_static)
        n_dynamic = len(is_static) - n_static
        print(f"Static points (<{args.threshold}): {n_static} ({n_static/len(is_static)*100:.1f}%)")
        print(f"Dynamic points: {n_dynamic} ({n_dynamic/len(is_static)*100:.1f}%)")
