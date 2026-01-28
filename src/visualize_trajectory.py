#!/usr/bin/env python
"""
Visualize 4DGS Gaussian trajectories over time.

Shows how each Gaussian point moves through the deformation field.
Useful for debugging whether the model learns proper motion or creates new Gaussians.

Usage:
    python src/visualize_trajectory.py output/4dgs/black_cat_alpha --num-points 500 --num-steps 10
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add 4DGS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', '4dgs'))


def get_default_deform_args():
    """Create default args for deformation network."""
    from argparse import Namespace
    args = Namespace(
        net_width=64,
        timebase_pe=4,
        defor_depth=1,
        posebase_pe=10,
        scale_rotation_pe=2,
        opacity_pe=2,
        timenet_width=64,
        timenet_output=32,
        bounds=1.6,
        kplanes_config={
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64, 64, 64, 25]
        },
        multires=[1, 2, 4, 8],
        no_dx=False,
        no_grid=False,
        no_ds=False,
        no_dr=False,
        no_do=True,
        no_dshs=True,
        empty_voxel=False,
        grid_pe=0,
        static_mlp=False,
        apply_rotation=False,
    )
    return args


def load_ply_points(model_path):
    """Load point cloud from PLY file directly (without full GaussianModel)."""
    from plyfile import PlyData

    # Find the latest checkpoint
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(point_cloud_dir):
        raise FileNotFoundError(f"Point cloud directory not found: {point_cloud_dir}")

    # Find iteration folders
    iterations = []
    for item in os.listdir(point_cloud_dir):
        if item.startswith("iteration_"):
            try:
                iter_num = int(item.split("_")[1])
                iterations.append(iter_num)
            except:
                pass

    if not iterations:
        raise FileNotFoundError("No iteration checkpoints found")

    latest_iter = max(iterations)
    ply_path = os.path.join(point_cloud_dir, f"iteration_{latest_iter}", "point_cloud.ply")

    print(f"[Trajectory] Loading PLY from iteration {latest_iter}")
    print(f"[Trajectory] PLY path: {ply_path}")

    # Load PLY directly
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    # Try to get opacity if available
    opacity = None
    if 'opacity' in vertex.data.dtype.names:
        opacity = vertex['opacity']

    return xyz, opacity, latest_iter


def load_deformation_model(model_path, iteration):
    """Load the deformation network from checkpoint."""
    from scene.deformation import deform_network

    deform_path = os.path.join(model_path, "deformation", f"iteration_{iteration}", "deformation.pth")
    if not os.path.exists(deform_path):
        print(f"[Warning] No deformation model found at {deform_path}")
        return None

    print(f"[Trajectory] Loading deformation model from {deform_path}")

    # Create deformation network with default args
    args = get_default_deform_args()
    deform = deform_network(args)

    # Load weights
    checkpoint = torch.load(deform_path, map_location='cpu')
    deform.load_state_dict(checkpoint)
    deform.eval()

    return deform


def compute_trajectories(xyz, deform, num_points=500, num_time_steps=10, device='cpu'):
    """Compute Gaussian trajectories over time using deformation network."""
    total_points = len(xyz)

    # Sample points
    if num_points < total_points:
        indices = np.random.choice(total_points, size=num_points, replace=False)
    else:
        indices = np.arange(total_points)
        num_points = total_points

    sampled_xyz = xyz[indices]

    # Time steps
    times = np.linspace(0, 1, num_time_steps)

    # Initialize trajectories
    trajectories = np.zeros((num_points, num_time_steps, 3))

    if deform is None:
        # No deformation - static points
        print("[Warning] No deformation model - showing static positions")
        for t_idx in range(num_time_steps):
            trajectories[:, t_idx, :] = sampled_xyz
        return trajectories, times, indices

    # Move deform to device
    deform = deform.to(device)

    # Query deformation at each time step
    xyz_tensor = torch.tensor(sampled_xyz, device=device, dtype=torch.float32)

    # Create dummy scales, rotations for the forward pass
    scales = torch.zeros(num_points, 3, device=device)
    rotations = torch.zeros(num_points, 4, device=device)
    rotations[:, 0] = 1.0  # Identity quaternion
    opacity = torch.ones(num_points, 1, device=device)
    shs = torch.zeros(num_points, 16, 3, device=device)

    for t_idx, t in enumerate(times):
        time_tensor = torch.full((num_points, 1), t, device=device, dtype=torch.float32)

        with torch.no_grad():
            try:
                # The deformation network returns (means3D, scales, rotations, opacity, shs)
                deformed_xyz, _, _, _, _ = deform(xyz_tensor, scales, rotations, opacity, shs, time_tensor)
                trajectories[:, t_idx, :] = deformed_xyz.cpu().numpy()
            except Exception as e:
                print(f"[Warning] Deformation query failed at t={t:.2f}: {e}")
                trajectories[:, t_idx, :] = sampled_xyz

        if (t_idx + 1) % 5 == 0:
            print(f"[Trajectory] Computed t={t:.2f} ({t_idx+1}/{num_time_steps})")

    return trajectories, times, indices


def analyze_point_distribution(xyz, opacity=None):
    """Analyze point cloud distribution to detect 'shadow clone' problem."""

    total_points = len(xyz)
    print(f"\n[Analysis] Total Gaussians: {total_points}")

    # Compute bounding box
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_range = xyz_max - xyz_min

    print(f"\n[Analysis] Bounding Box:")
    print(f"  X: {xyz_min[0]:.3f} ~ {xyz_max[0]:.3f} (range: {xyz_range[0]:.3f})")
    print(f"  Y: {xyz_min[1]:.3f} ~ {xyz_max[1]:.3f} (range: {xyz_range[1]:.3f})")
    print(f"  Z: {xyz_min[2]:.3f} ~ {xyz_max[2]:.3f} (range: {xyz_range[2]:.3f})")

    # Compute center and spread
    center = xyz.mean(axis=0)
    distances_from_center = np.linalg.norm(xyz - center, axis=1)

    print(f"\n[Analysis] Point Distribution:")
    print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  Mean distance from center: {distances_from_center.mean():.3f}")
    print(f"  Max distance from center: {distances_from_center.max():.3f}")
    print(f"  Std of distances: {distances_from_center.std():.3f}")

    # Check for elongated distribution (shadow clone symptom)
    # If one axis is much larger than others, points may be spread along camera path
    aspect_ratios = xyz_range / (xyz_range.min() + 1e-6)
    max_aspect = aspect_ratios.max()
    elongated_axis = ['X', 'Y', 'Z'][aspect_ratios.argmax()]

    print(f"\n[Analysis] Shape Analysis:")
    print(f"  Aspect ratios: X={aspect_ratios[0]:.2f}, Y={aspect_ratios[1]:.2f}, Z={aspect_ratios[2]:.2f}")
    print(f"  Most elongated axis: {elongated_axis} ({max_aspect:.2f}x)")

    if max_aspect > 5:
        print(f"\n  ⚠️  WARNING: Point cloud is highly elongated along {elongated_axis} axis!")
        print(f"     This may indicate 'shadow clone' problem (Gaussians spread along camera path)")
        print(f"     Expected: compact object shape, Got: elongated distribution")
    elif max_aspect > 3:
        print(f"\n  ⚠️  CAUTION: Moderate elongation along {elongated_axis} axis")
    else:
        print(f"\n  ✓ Point cloud appears reasonably compact")

    # Cluster analysis (simplified)
    # If there are multiple distinct clusters, it might indicate shadow clones
    from scipy.cluster.hierarchy import fclusterdata
    try:
        if total_points > 100:
            sample_idx = np.random.choice(total_points, min(1000, total_points), replace=False)
            sample_xyz = xyz[sample_idx]
        else:
            sample_xyz = xyz

        # Cluster with distance threshold
        clusters = fclusterdata(sample_xyz, t=xyz_range.max() * 0.1, criterion='distance')
        n_clusters = len(np.unique(clusters))

        print(f"\n[Analysis] Cluster Analysis:")
        print(f"  Detected clusters (threshold=10% of max range): {n_clusters}")

        if n_clusters > 5:
            print(f"  ⚠️  Multiple clusters detected - possible shadow clones")
        else:
            print(f"  ✓ Point cloud appears to be a single connected object")
    except Exception as e:
        print(f"\n[Analysis] Cluster analysis skipped: {e}")

    return {
        'total_points': total_points,
        'bbox_range': xyz_range,
        'center': center,
        'max_aspect_ratio': max_aspect,
        'elongated_axis': elongated_axis
    }


def save_trajectories_ply(trajectories, times, output_path):
    """Save trajectories as PLY with lines."""
    from plyfile import PlyData, PlyElement

    num_points, num_times, _ = trajectories.shape

    # Create vertices (all positions at all times)
    vertices = trajectories.reshape(-1, 3)

    # Create colors based on time (blue -> red)
    colors = np.zeros((len(vertices), 3), dtype=np.uint8)
    for t_idx in range(num_times):
        t = times[t_idx]
        start_idx = t_idx * num_points
        end_idx = (t_idx + 1) * num_points
        # Blue (early) to Red (late)
        colors[start_idx:end_idx, 0] = int(255 * t)  # Red
        colors[start_idx:end_idx, 2] = int(255 * (1 - t))  # Blue

    # Save as PLY
    vertex_data = np.zeros(len(vertices), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el]).write(output_path)
    print(f"[Trajectory] Saved trajectory points to {output_path}")


def visualize_with_rerun(trajectories, times, indices):
    """Visualize trajectories using Rerun."""
    try:
        import rerun as rr
    except ImportError:
        print("[Error] Rerun not installed. Install with: pip install rerun-sdk")
        return

    num_points, num_times, _ = trajectories.shape

    rr.init("4DGS Trajectories", spawn=True)

    # Log trajectories as line strips
    for p_idx in range(min(num_points, 200)):  # Limit for performance
        positions = trajectories[p_idx]  # (num_times, 3)

        # Color based on movement magnitude
        movement = np.linalg.norm(positions[-1] - positions[0])

        # Log as line strip
        rr.log(
            f"trajectories/point_{p_idx}",
            rr.LineStrips3D([positions], colors=[[255, int(255 * (1 - min(movement, 1))), 0, 255]])
        )

    # Log point clouds at different times
    for t_idx, t in enumerate(times):
        rr.set_time_sequence("frame", t_idx)
        positions = trajectories[:, t_idx, :]

        # Color by time
        color = [int(255 * t), 100, int(255 * (1 - t)), 255]

        rr.log(
            "gaussians",
            rr.Points3D(positions, colors=[color] * len(positions), radii=0.01)
        )

    print("[Trajectory] Rerun visualization started")


def compute_movement_stats(trajectories, times):
    """Compute statistics about Gaussian movements."""
    num_points, num_times, _ = trajectories.shape

    # Total displacement (start to end)
    start_pos = trajectories[:, 0, :]
    end_pos = trajectories[:, -1, :]
    total_displacement = np.linalg.norm(end_pos - start_pos, axis=1)

    # Path length (cumulative distance)
    path_lengths = np.zeros(num_points)
    for t_idx in range(1, num_times):
        step_distance = np.linalg.norm(
            trajectories[:, t_idx, :] - trajectories[:, t_idx - 1, :],
            axis=1
        )
        path_lengths += step_distance

    print("\n" + "=" * 50)
    print("Movement Statistics")
    print("=" * 50)
    print(f"Number of tracked points: {num_points}")
    print(f"Time steps: {num_times}")
    print(f"\nTotal Displacement (start to end):")
    print(f"  Mean: {total_displacement.mean():.4f}")
    print(f"  Max:  {total_displacement.max():.4f}")
    print(f"  Min:  {total_displacement.min():.4f}")
    print(f"  Std:  {total_displacement.std():.4f}")
    print(f"\nPath Length (cumulative):")
    print(f"  Mean: {path_lengths.mean():.4f}")
    print(f"  Max:  {path_lengths.max():.4f}")
    print(f"\nPoints with significant movement (>0.1): {(total_displacement > 0.1).sum()}")
    print(f"Points with minimal movement (<0.01): {(total_displacement < 0.01).sum()}")
    print("=" * 50)

    return {
        'total_displacement': total_displacement,
        'path_lengths': path_lengths
    }


def visualize_with_rerun(xyz, stats, model_name="4DGS", output_path=None, trajectories=None, times=None):
    """Visualize point cloud distribution and trajectories with Rerun."""
    try:
        import rerun as rr
    except ImportError:
        print("[Error] Rerun not installed. Install with: pip install rerun-sdk")
        return

    rr.init(f"4DGS Trajectory Visualization - {model_name}")

    # If output_path specified, save to file. Otherwise try to spawn viewer
    if output_path:
        rr.save(output_path)
        print(f"[Rerun] Saving to {output_path}")
    else:
        try:
            rr.spawn()
        except Exception as e:
            output_path = f"{model_name}_trajectories.rrd"
            rr.save(output_path)
            print(f"[Rerun] No display available, saving to {output_path}")

    # If we have trajectories, visualize them
    if trajectories is not None and times is not None:
        num_points, num_times, _ = trajectories.shape

        # Log trajectory lines (static, showing full path)
        print(f"[Rerun] Logging {num_points} trajectory lines...")
        for p_idx in range(num_points):
            positions = trajectories[p_idx]  # (num_times, 3)

            # Color based on total movement magnitude
            movement = np.linalg.norm(positions[-1] - positions[0])
            # Green (low movement) to Red (high movement)
            red = min(255, int(movement * 500))
            green = max(0, 255 - red)

            rr.log(
                f"trajectories/line_{p_idx}",
                rr.LineStrips3D([positions], colors=[[red, green, 0, 200]])
            )

        # Animate points over time
        print(f"[Rerun] Logging animation frames...")
        for t_idx, t in enumerate(times):
            rr.set_time_sequence("frame", t_idx)
            rr.set_time_seconds("time", t)

            positions = trajectories[:, t_idx, :]

            # Color by time: Blue (t=0) to Red (t=1)
            color = [int(255 * t), 50, int(255 * (1 - t)), 255]

            rr.log(
                "gaussians/animated",
                rr.Points3D(positions, colors=[color] * len(positions), radii=0.03)
            )

        print(f"[Rerun] Animation: {num_times} frames logged")

    else:
        # Static visualization (no trajectories)
        axis_idx = ['X', 'Y', 'Z'].index(stats['elongated_axis'])
        axis_vals = xyz[:, axis_idx]
        normalized = (axis_vals - axis_vals.min()) / (axis_vals.max() - axis_vals.min() + 1e-6)

        colors = np.zeros((len(xyz), 4), dtype=np.uint8)
        colors[:, 0] = (normalized * 255).astype(np.uint8)
        colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)
        colors[:, 3] = 255

        rr.log(
            "gaussians/all_points",
            rr.Points3D(xyz, colors=colors, radii=0.02)
        )

    # Log bounding box
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    half_sizes = (bbox_max - bbox_min) / 2

    rr.log(
        "scene/bounding_box",
        rr.Boxes3D(centers=[center], half_sizes=[half_sizes], colors=[[255, 255, 0, 50]])
    )

    # Log axes
    axis_length = max(half_sizes) * 1.5
    rr.log("axes/x", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[axis_length, 0, 0]], colors=[[255, 0, 0, 255]]))
    rr.log("axes/y", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, axis_length, 0]], colors=[[0, 255, 0, 255]]))
    rr.log("axes/z", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, 0, axis_length]], colors=[[0, 0, 255, 255]]))

    print(f"\n[Rerun] Visualization complete")
    if trajectories is not None:
        print(f"  - Trajectory lines: Green (static) to Red (moving)")
        print(f"  - Animation: Blue (t=0) to Red (t=1)")
    else:
        print(f"  - Points colored by {stats['elongated_axis']} position")
    if output_path:
        print(f"\n  To view: rerun {output_path}")
        print(f"  Or download and open locally")


def main():
    parser = argparse.ArgumentParser(description="Analyze 4DGS Gaussian point cloud and visualize trajectories")
    parser.add_argument("model_path", help="Path to trained model (e.g., output/4dgs/black_cat)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PLY file for visualization")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute statistics, no file output")
    parser.add_argument("--rerun", action="store_true",
                        help="Visualize with Rerun")
    parser.add_argument("--trajectories", action="store_true",
                        help="Compute and visualize trajectories (requires deformation model)")
    parser.add_argument("--num-points", type=int, default=300,
                        help="Number of points to track for trajectories (default: 300)")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of time steps for trajectories (default: 20)")

    args = parser.parse_args()

    # Load PLY directly
    try:
        xyz, opacity, iteration = load_ply_points(args.model_path)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Analyze point distribution
    stats = analyze_point_distribution(xyz, opacity)

    if args.stats_only:
        return

    # Compute trajectories if requested
    trajectories = None
    times = None
    if args.trajectories or args.rerun:
        print(f"\n[Trajectory] Loading deformation model...")
        try:
            deform = load_deformation_model(args.model_path, iteration)
            if deform is not None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"[Trajectory] Computing trajectories on {device}...")
                trajectories, times, traj_indices = compute_trajectories(
                    xyz, deform,
                    num_points=args.num_points,
                    num_time_steps=args.num_steps,
                    device=device
                )

                # Compute movement stats
                total_displacement = np.linalg.norm(trajectories[:, -1] - trajectories[:, 0], axis=1)
                print(f"\n[Trajectory] Movement Statistics:")
                print(f"  Mean displacement: {total_displacement.mean():.4f}")
                print(f"  Max displacement:  {total_displacement.max():.4f}")
                print(f"  Min displacement:  {total_displacement.min():.4f}")
                print(f"  Points with movement > 0.1: {(total_displacement > 0.1).sum()}/{len(total_displacement)}")
        except Exception as e:
            print(f"[Warning] Could not load deformation model: {e}")
            import traceback
            traceback.print_exc()

    # Visualize with Rerun if requested
    if args.rerun:
        model_name = os.path.basename(args.model_path)
        rrd_output = os.path.join(args.model_path, f"{model_name}_trajectories.rrd")
        visualize_with_rerun(xyz, stats, model_name, rrd_output, trajectories, times)
        return

    # Save colored PLY for visualization
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.model_path, f"analysis_iter{iteration}.ply")

    # Color points by position along elongated axis
    from plyfile import PlyData, PlyElement

    axis_idx = ['X', 'Y', 'Z'].index(stats['elongated_axis'])
    axis_vals = xyz[:, axis_idx]
    normalized = (axis_vals - axis_vals.min()) / (axis_vals.max() - axis_vals.min() + 1e-6)

    # Blue (low) to Red (high)
    colors = np.zeros((len(xyz), 3), dtype=np.uint8)
    colors[:, 0] = (normalized * 255).astype(np.uint8)  # Red
    colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)  # Blue

    vertex_data = np.zeros(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el]).write(output_path)
    print(f"\n[Output] Saved colored point cloud to {output_path}")
    print(f"  Colors: Blue (low {stats['elongated_axis']}) → Red (high {stats['elongated_axis']})")


if __name__ == "__main__":
    main()
