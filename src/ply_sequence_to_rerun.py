import os
import sys
import glob
import re
import argparse
import numpy as np
import rerun as rr
from plyfile import PlyData

def parse_args():
    parser = argparse.ArgumentParser(description="Log a sequence of PLY files to Rerun")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PLY files (frame_0.ply, etc.)")
    parser.add_argument("--output", type=str, default="4dgs_recording.rrd", help="Output .rrd file path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for timeline")
    parser.add_argument("--max_points", type=int, default=100000, help="Max points to log per frame for performance")
    return parser.parse_args()

def log_ply_sequence(input_dir, output_file, fps, max_points):
    # Find all PLY files
    ply_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    if not ply_files:
        print(f"Error: No PLY files found in {input_dir}")
        return

    print(f"Found {len(ply_files)} PLY files. Beginning export to {output_file}...")

    # Initialize Rerun
    rr.init("4DGS Visualization", spawn=False)
    rr.save(output_file)

    for i, ply_path in enumerate(ply_files):
        # Extract frame number if possible, else use index
        frame_num = i
        match = re.search(r"(\d+)", os.path.basename(ply_path))
        if match:
            frame_num = int(match.group(1))

        # Set the time for this frame
        try:
            rr.set_time("frame", sequence=frame_num)
        except (AttributeError, TypeError):
            rr.set_time_sequence("frame", frame_num)

        # Load PLY
        try:
            plydata = PlyData.read(ply_path)
            v = plydata['vertex']
            
            # Extract positions
            pts = np.stack([v['x'], v['y'], v['z']], axis=1)
            
            # Extract colors if available
            colors = None
            if 'red' in v and 'green' in v and 'blue' in v:
                colors = np.stack([v['red'], v['green'], v['blue']], axis=1)
            elif 'f_dc_0' in v:
                # 3DGS/4DGS specific: convert SH to RGB (simplified)
                f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
                colors = (0.5 + 0.28209 * f_dc) * 255
                colors = np.clip(colors, 0, 255).astype(np.uint8)

            # Subsample for performance if needed
            if len(pts) > max_points:
                indices = np.random.choice(len(pts), max_points, replace=False)
                pts = pts[indices]
                if colors is not None:
                    colors = colors[indices]

            # Log to Rerun
            rr.log(
                "world/gaussians",
                rr.Points3D(pts, colors=colors, radii=0.005)
            )
            
            print(f"Logged frame {frame_num} ({len(pts)} points)")

        except Exception as e:
            print(f"Error loading {ply_path}: {e}")

    print(f"Export complete: {output_file}")
    print("Download this file to your local machine and open with Rerun Viewer!")

if __name__ == "__main__":
    args = parse_args()
    log_ply_sequence(args.input_dir, args.output, args.fps, args.max_points)
