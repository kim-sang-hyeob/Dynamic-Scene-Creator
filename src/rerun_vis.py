import rerun as rr
import numpy as np
import os
import time
import threading
from plyfile import PlyData

def log_ply_to_rerun(ply_path, entity_path="world/gaussians"):
    """Logs a single 3DGS PLY file to Rerun using 0.28.x API."""
    try:
        if not os.path.exists(ply_path):
            print(f"[Error] File not found: {ply_path}")
            return

        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        
        positions = np.stack([v['x'], v['y'], v['z']], axis=-1)
        
        # Colors: SH coefficients (DC part) to RGB
        if 'red' in v:
            colors = np.stack([v['red'], v['green'], v['blue']], axis=-1)
        elif 'f_dc_0' in v:
            C0 = 0.28209479177387814
            colors = 0.5 + C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
            colors = np.clip(colors, 0, 1)
        else:
            colors = None

        # Radii
        if 'scale_0' in v:
            scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1))
            radii = np.mean(scales, axis=-1) * 0.5
        else:
            radii = np.ones(len(positions)) * 0.02

        if colors is not None:
            colors = (colors * 255).astype(np.uint8)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log(entity_path, rr.Points3D(positions, colors=colors, radii=radii))
        print(f"[Vis] Successfully logged {len(positions)} points from {os.path.basename(ply_path)}")
        
    except Exception as e:
        print(f"[Error] Failed to log {ply_path}: {e}")

def find_latest_ply_files(base_dir):
    """Recursively finds all .ply files in the latest iteration folders."""
    ply_files = []
    if not os.path.exists(base_dir):
        return []

    # 1. 4DGS structure check
    pc_path = os.path.join(base_dir, "point_cloud")
    if os.path.exists(pc_path):
        iters = [d for d in os.listdir(pc_path) if os.path.isdir(os.path.join(pc_path, d))]
        if iters:
            iters.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            for it in iters:
                it_path = os.path.join(pc_path, it)
                files = sorted([os.path.join(it_path, f) for f in os.listdir(it_path) if f.endswith('.ply')])
                ply_files.extend(files)
            return ply_files

    # 2. General recursive search
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.ply'):
                ply_files.append(os.path.join(root, f))
    
    return sorted(ply_files)

def _logging_thread(sequence_dir, watch=False, interval=2.0, recording_stream=None):
    """Internal thread for background logging."""
    if recording_stream:
        rr.set_thread_local_data_recording(recording_stream)
        
    print(f"[Vis] Background logging thread active for {sequence_dir}")
    
    # Log a test point to verify connectivity immediately
    rr.log("world/test_point", rr.Points3D([[0, 0, 0]], colors=[[255, 0, 0]], radii=[1.0]))
    
    seen_files = set()
    try:
        while True:
            current_files = find_latest_ply_files(sequence_dir)
            new_files = [f for f in current_files if f not in seen_files]
            
            if new_files:
                for f_path in new_files:
                    try:
                        it_num = int(''.join(filter(str.isdigit, os.path.basename(os.path.dirname(f_path)))) or 0)
                        rr.set_time("frame_idx", sequence=it_num)
                    except:
                        rr.set_time("frame_idx", sequence=len(seen_files))
                    
                    log_ply_to_rerun(f_path)
                    seen_files.add(f_path)
                print(f"[Vis] Processed {len(new_files)} new files. Total: {len(seen_files)}")
            
            if not watch:
                break
            time.sleep(interval)
    except Exception as e:
        print(f"[Vis] Thread Error: {e}")

def run_visualization(sequence_dir, watch=False, web=False, connect=None, save_path=None):
    """Orchestrates visualization using 0.28.2 API."""
    rec = rr.init("4DGS Visualization", spawn=False)
    
    # Enable global recording
    if hasattr(rr, "set_global_data_recording"):
        rr.set_global_data_recording(rec)

    # 1. Setup Sink
    if save_path:
        print(f"[Vis] Saving recording to {save_path}...")
        rr.save(save_path)
    elif web:
        print("[Vis] Starting Rerun Web Viewer...")
        print("[Note] PLEASE FORWARD BOTH PORTS 9876 (Web) AND 9877 (Data) in VS Code!")
        # Default ports work best for the Wasm viewer auto-discovery
        rr.serve_web_viewer(web_port=9876)
    elif connect:
        print(f"[Vis] Connecting to Rerun viewer at {connect} via gRPC...")
        rr.connect_grpc(connect)
    
    # 2. Start Background Logging
    thread = threading.Thread(target=_logging_thread, args=(sequence_dir, watch, 2.0, rec), daemon=True)
    thread.start()
    
    # 3. Persistence loop
    if watch or web or connect or (save_path and watch):
        print(f"[Vis] Monitoring {sequence_dir}... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
                if not watch and not thread.is_alive():
                    break
        except KeyboardInterrupt:
            print("[Vis] Stopped.")
    else:
        # If just a one-shot save, wait for thread to finish
        thread.join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing per-frame PLY files")
    parser.add_argument("--watch", action="store_true", help="Watch directory")
    parser.add_argument("--web", action="store_true", help="Serve Rerun over web")
    parser.add_argument("--connect", help="Connect to a remote gRPC Rerun viewer (IP:Port)")
    parser.add_argument("--save", help="Save recording to an .rrd file")
    args = parser.parse_args()
    
    run_visualization(args.dir, watch=args.watch, web=args.web, connect=args.connect, save_path=args.save)
