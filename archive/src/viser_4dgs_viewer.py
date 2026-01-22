"""
4DGS Interactive Web Viewer using Viser
Allows real-time viewing of 4D Gaussian Splatting models with time control.
"""
import os
import sys
import torch
import numpy as np
import viser
import viser.transforms as tf
import time
from pathlib import Path

def load_4dgs_model(model_path):
    """Load 4DGS model from checkpoint."""
    # Find the latest checkpoint
    chkpt_files = list(Path(model_path).glob("chkpnt*.pth"))
    if not chkpt_files:
        print(f"[Error] No checkpoint found in {model_path}")
        return None

    latest_chkpt = sorted(chkpt_files)[-1]
    print(f"[Viewer] Loading checkpoint: {latest_chkpt}")

    checkpoint = torch.load(latest_chkpt, map_location='cuda')
    return checkpoint

def load_ply_gaussians(model_path):
    """Load Gaussians from PLY file (simpler approach)."""
    from plyfile import PlyData

    # Find point cloud PLY
    ply_paths = [
        Path(model_path) / "point_cloud" / "iteration_10000" / "point_cloud.ply",
        Path(model_path) / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        Path(model_path) / "point_cloud" / "iteration_3000" / "point_cloud.ply",
    ]

    ply_path = None
    for p in ply_paths:
        if p.exists():
            ply_path = p
            break

    if not ply_path:
        print(f"[Error] No point_cloud.ply found in {model_path}")
        return None

    print(f"[Viewer] Loading PLY: {ply_path}")
    plydata = PlyData.read(str(ply_path))
    vertex = plydata['vertex']

    # Extract positions
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Extract colors (SH coefficients or direct RGB)
    if 'red' in vertex._property_lookup:
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1)
        colors = colors / 255.0 if colors.max() > 1.0 else colors
    elif 'f_dc_0' in vertex._property_lookup:
        # Convert SH DC component to RGB
        SH_C0 = 0.28209479177387814
        f_dc_0 = vertex['f_dc_0']
        f_dc_1 = vertex['f_dc_1']
        f_dc_2 = vertex['f_dc_2']
        colors = np.stack([
            0.5 + SH_C0 * f_dc_0,
            0.5 + SH_C0 * f_dc_1,
            0.5 + SH_C0 * f_dc_2
        ], axis=1)
        colors = np.clip(colors, 0, 1)
    else:
        colors = np.ones((xyz.shape[0], 3)) * 0.5

    # Extract opacity
    if 'opacity' in vertex._property_lookup:
        opacity = 1.0 / (1.0 + np.exp(-vertex['opacity']))  # sigmoid
    else:
        opacity = np.ones(xyz.shape[0])

    # Extract scales
    if 'scale_0' in vertex._property_lookup:
        scales = np.stack([
            np.exp(vertex['scale_0']),
            np.exp(vertex['scale_1']),
            np.exp(vertex['scale_2'])
        ], axis=1)
    else:
        scales = np.ones((xyz.shape[0], 3)) * 0.01

    return {
        'xyz': xyz,
        'colors': colors,
        'opacity': opacity,
        'scales': scales,
        'count': xyz.shape[0]
    }

def run_viewer(model_path, host="0.0.0.0", port=8080):
    """Run the interactive viser viewer."""
    print(f"[Viewer] Starting 4DGS Viewer...")
    print(f"[Viewer] Model: {model_path}")

    # Load gaussians
    gaussians = load_ply_gaussians(model_path)
    if gaussians is None:
        return

    print(f"[Viewer] Loaded {gaussians['count']} Gaussians")

    # Create viser server
    server = viser.ViserServer(host=host, port=port)
    print(f"[Viewer] Server running at http://{host}:{port}")

    # Filter by opacity for visualization (show top N points)
    opacity = gaussians['opacity']
    visible_mask = opacity > 0.1  # Filter low opacity

    xyz = gaussians['xyz'][visible_mask]
    colors = gaussians['colors'][visible_mask]

    # Subsample if too many points
    max_points = 100000
    if len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        colors = colors[indices]

    print(f"[Viewer] Displaying {len(xyz)} points")

    # Add point cloud to scene
    server.scene.add_point_cloud(
        name="/gaussians",
        points=xyz.astype(np.float32),
        colors=(colors * 255).astype(np.uint8),
        point_size=0.01,
    )

    # Add coordinate frame
    server.scene.add_frame(
        name="/origin",
        wxyz=tf.SO3.identity().wxyz,
        position=(0, 0, 0),
        axes_length=1.0,
        axes_radius=0.02,
    )

    # Add GUI controls
    with server.gui.add_folder("4DGS Controls"):
        time_slider = server.gui.add_slider(
            "Time",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.0,
        )

        point_size = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.01,
        )

        opacity_threshold = server.gui.add_slider(
            "Opacity Threshold",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=0.1,
        )

        play_button = server.gui.add_button("Play Animation")
        reset_button = server.gui.add_button("Reset View")

    # State for animation
    animation_state = {"playing": False}

    @play_button.on_click
    def _(_):
        animation_state["playing"] = not animation_state["playing"]
        play_button.name = "Pause" if animation_state["playing"] else "Play Animation"

    @point_size.on_update
    def _(_):
        server.scene.add_point_cloud(
            name="/gaussians",
            points=xyz.astype(np.float32),
            colors=(colors * 255).astype(np.uint8),
            point_size=point_size.value,
        )

    @opacity_threshold.on_update
    def _(_):
        # Re-filter points
        visible = gaussians['opacity'] > opacity_threshold.value
        new_xyz = gaussians['xyz'][visible]
        new_colors = gaussians['colors'][visible]

        if len(new_xyz) > max_points:
            indices = np.random.choice(len(new_xyz), max_points, replace=False)
            new_xyz = new_xyz[indices]
            new_colors = new_colors[indices]

        server.scene.add_point_cloud(
            name="/gaussians",
            points=new_xyz.astype(np.float32),
            colors=(new_colors * 255).astype(np.uint8),
            point_size=point_size.value,
        )

    # Animation loop
    print("[Viewer] Ready! Open browser to interact.")
    print("[Viewer] Press Ctrl+C to stop.")

    try:
        while True:
            if animation_state["playing"]:
                current_time = time_slider.value
                new_time = (current_time + 0.02) % 1.0
                time_slider.value = new_time
                time.sleep(0.05)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Viewer] Shutting down...")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="4DGS Interactive Viewer")
    parser.add_argument("model_path", help="Path to 4DGS output directory")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()

    run_viewer(args.model_path, args.host, args.port)

if __name__ == "__main__":
    main()
