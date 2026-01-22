import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

def freeze_static_points(input_dir, output_dir, threshold=0.005):
    """
    Identifies static points and 'freezes' them by overriding their positions 
    across all frames with their positions from the first frame.
    """
    ply_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.ply')])
    if len(ply_files) < 2:
        print("Need at least 2 frames.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load all frames to calculate motion energy (Simplified version of analyze_motion)
    print("Calculating motion energy to identify points to freeze...")
    all_positions = []
    for f in ply_files:
        p = PlyData.read(f)
        v = p['vertex']
        all_positions.append(np.stack([v['x'], v['y'], v['z']], axis=1))
    
    all_positions = np.stack(all_positions, axis=0)
    motion_energy = np.std(all_positions, axis=0).mean(axis=-1)
    
    # 2. Identify static mask
    is_static = motion_energy < threshold
    print(f"Total points: {len(is_static)}")
    print(f"Freezing {np.sum(is_static)} static points (Threshold: {threshold})...")

    # 3. Reference positions from Frame 0
    reference_pos = all_positions[0]

    # 4. Generate new PLY files
    for i, f in enumerate(ply_files):
        print(f"Processing frame {i}...")
        plydata = PlyData.read(f)
        v = plydata['vertex'].data.copy()
        
        # Override positions for static points
        # Note: v is a structured array, we update x, y, z fields
        v['x'][is_static] = reference_pos[is_static, 0]
        v['y'][is_static] = reference_pos[is_static, 1]
        v['z'][is_static] = reference_pos[is_static, 2]
        
        # Save new PLY
        new_el = PlyElement.describe(v, 'vertex')
        new_path = os.path.join(output_dir, os.path.basename(f))
        PlyData([new_el], text=False).write(new_path)

    print(f"Baking complete! Baked files saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()
    
    freeze_static_points(args.input_dir, args.output_dir, args.threshold)
