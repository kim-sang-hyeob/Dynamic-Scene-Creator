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


def load_gaussian_model(model_path):
    """Load trained 4DGS model."""
    from scene.gaussian_model import GaussianModel
    from argparse import Namespace

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

    print(f"[Trajectory] Loading model from iteration {latest_iter}")
    print(f"[Trajectory] PLY path: {ply_path}")

    # Initialize Gaussian model
    # Load hyperparameters if available
    sh_degree = 3  # Default
    hyper_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(hyper_path):
        import pickle
        try:
            with open(hyper_path, 'rb') as f:
                cfg_args = pickle.load(f)
            sh_degree = getattr(cfg_args, 'sh_degree', 3)
        except Exception as e:
            print(f"[Warning] Could not load cfg_args: {e}, using default sh_degree=3")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)

    # Load deformation model if exists
    deform_path = os.path.join(model_path, "deformation", f"iteration_{latest_iter}", "deformation.pth")
    if os.path.exists(deform_path):
        print(f"[Trajectory] Loading deformation model...")
        from scene.deformation import DeformModel
        deform = DeformModel()
        deform.load_weights(model_path, latest_iter)
        return gaussians, deform, latest_iter
    else:
        print(f"[Warning] No deformation model found at {deform_path}")
        return gaussians, None, latest_iter


def compute_trajectories(gaussians, deform, num_points=500, num_time_steps=10):
    """Compute Gaussian trajectories over time."""

    # Get base positions
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    total_points = len(xyz)

    print(f"[Trajectory] Total Gaussians: {total_points}")

    # Sample points (random or by importance)
    if num_points < total_points:
        # Sample points with higher opacity
        opacities = gaussians.get_opacity.detach().cpu().numpy().squeeze()
        # Prioritize visible points
        probs = np.clip(opacities, 0, 1)
        probs = probs / probs.sum()
        indices = np.random.choice(total_points, size=num_points, replace=False, p=probs)
    else:
        indices = np.arange(total_points)
        num_points = total_points

    print(f"[Trajectory] Sampling {num_points} points")

    # Time steps
    times = np.linspace(0, 1, num_time_steps)

    # Compute trajectories
    trajectories = np.zeros((num_points, num_time_steps, 3))

    if deform is None:
        # No deformation - static points
        print("[Warning] No deformation model - showing static positions")
        for t_idx in range(num_time_steps):
            trajectories[:, t_idx, :] = xyz[indices]
    else:
        # Query deformation at each time step
        device = next(deform.deformation.parameters()).device

        for t_idx, t in enumerate(times):
            # Create time tensor
            time_tensor = torch.tensor([t], device=device).float()

            # Get positions for sampled points
            xyz_tensor = torch.tensor(xyz[indices], device=device).float()

            # Query deformation
            with torch.no_grad():
                # The deformation network takes (N, 3) positions and time
                # and returns position offsets
                try:
                    d_xyz, _, _ = deform.step(xyz_tensor, time_tensor.expand(len(xyz_tensor), 1))
                    deformed_xyz = xyz_tensor + d_xyz
                    trajectories[:, t_idx, :] = deformed_xyz.cpu().numpy()
                except Exception as e:
                    print(f"[Warning] Deformation query failed at t={t:.2f}: {e}")
                    trajectories[:, t_idx, :] = xyz[indices]

            print(f"[Trajectory] Computed t={t:.2f} ({t_idx+1}/{num_time_steps})")

    return trajectories, times, indices


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


def main():
    parser = argparse.ArgumentParser(description="Visualize 4DGS Gaussian trajectories")
    parser.add_argument("model_path", help="Path to trained model (e.g., output/4dgs/black_cat)")
    parser.add_argument("--num-points", type=int, default=500,
                        help="Number of points to track (default: 500)")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of time steps (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PLY file for trajectories")
    parser.add_argument("--rerun", action="store_true",
                        help="Visualize with Rerun")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute statistics, no visualization")

    args = parser.parse_args()

    # Load model
    try:
        gaussians, deform, iteration = load_gaussian_model(args.model_path)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compute trajectories
    trajectories, times, indices = compute_trajectories(
        gaussians, deform,
        num_points=args.num_points,
        num_time_steps=args.num_steps
    )

    # Compute stats
    stats = compute_movement_stats(trajectories, times)

    if args.stats_only:
        return

    # Save PLY
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.model_path, f"trajectories_iter{iteration}.ply")

    save_trajectories_ply(trajectories, times, output_path)

    # Visualize with Rerun
    if args.rerun:
        visualize_with_rerun(trajectories, times, indices)


if __name__ == "__main__":
    main()
