#!/usr/bin/env python3
"""
Verify camera rotation by analyzing transforms_train.json.
Calculate where the cat is (look-at point) and test Y-axis rotation.
"""

import json
import numpy as np

def load_transforms(path):
    with open(path) as f:
        return json.load(f)

def analyze_camera(transform_matrix):
    """Analyze a c2w transform matrix."""
    c2w = np.array(transform_matrix)

    # Camera position (4th column)
    cam_pos = c2w[:3, 3]

    # Camera axes in world space
    right = c2w[:3, 0]  # X axis
    up = c2w[:3, 1]     # Y axis
    forward = -c2w[:3, 2]  # Camera looks along -Z

    return cam_pos, forward, right, up

def estimate_look_at_point(cam_pos, forward, distance=4.5):
    """Estimate where camera is looking at."""
    forward_norm = forward / np.linalg.norm(forward)
    return cam_pos + forward_norm * distance

def rotate_around_point(point, center, angle_deg, axis='y'):
    """Rotate a point around a center point by angle degrees around axis."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Y-axis rotation matrix
    if axis == 'y':
        rot = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])

    relative = point - center
    rotated = rot @ relative
    return center + rotated

def rotate_c2w_around_point(c2w, center, angle_deg):
    """Rotate entire c2w matrix around a center point."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Y-axis rotation matrix
    rot_y = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    # Get camera position
    cam_pos = c2w[:3, 3]

    # Rotate position around center
    relative_pos = cam_pos - center
    new_relative_pos = rot_y @ relative_pos
    new_cam_pos = center + new_relative_pos

    # Rotate orientation
    new_c2w = np.eye(4)
    new_c2w[:3, :3] = rot_y @ c2w[:3, :3]
    new_c2w[:3, 3] = new_cam_pos

    return new_c2w

def main():
    # Load transforms
    transforms = load_transforms('/Users/blakelee/Desktop/4dgs_project/data/black_cat/transforms_train.json')

    print("=" * 60)
    print("Camera Rotation Analysis for transforms_train.json")
    print("=" * 60)

    # First, understand the coordinate system
    # In NeRF/Blender format: camera looks along -Z axis in camera space
    # c2w transforms camera space to world space
    # So camera looks at -c2w[:3,2] direction in world space

    # Analyze first frame (frame 0)
    frame0 = transforms['frames'][0]
    c2w_0 = np.array(frame0['transform_matrix'])

    cam_pos_0, forward_0, right_0, up_0 = analyze_camera(c2w_0)

    print("\n[Frame 0 - Original Camera]")
    print(f"  Position: ({cam_pos_0[0]:.4f}, {cam_pos_0[1]:.4f}, {cam_pos_0[2]:.4f})")
    print(f"  Forward:  ({forward_0[0]:.4f}, {forward_0[1]:.4f}, {forward_0[2]:.4f})")

    # Calculate distance from origin
    dist_from_origin = np.linalg.norm(cam_pos_0)
    print(f"  Distance from origin: {dist_from_origin:.4f}")

    # Try different look-at distances to find where the cat might be
    print("\n[Estimating Look-At Point (Cat Position)]")
    for dist in [2.0, 3.0, 4.0, 4.5, 5.0]:
        look_at = estimate_look_at_point(cam_pos_0, forward_0, dist)
        print(f"  Distance {dist}: look-at = ({look_at[0]:.4f}, {look_at[1]:.4f}, {look_at[2]:.4f})")

    # Use distance 4.5 as estimated look-at point
    look_distance = 4.5
    look_at_point = estimate_look_at_point(cam_pos_0, forward_0, look_distance)

    print(f"\n[Selected Look-At Point (distance={look_distance})]")
    print(f"  Cat position: ({look_at_point[0]:.4f}, {look_at_point[1]:.4f}, {look_at_point[2]:.4f})")

    # Now rotate camera 45 degrees around Y-axis centered at look_at_point
    angle = 45
    print(f"\n[Rotating Camera {angle} degrees around Y-axis]")
    print(f"  Rotation center: ({look_at_point[0]:.4f}, {look_at_point[1]:.4f}, {look_at_point[2]:.4f})")

    # Rotate camera position
    new_cam_pos = rotate_around_point(cam_pos_0, look_at_point, angle)
    print(f"\n  Original camera pos: ({cam_pos_0[0]:.4f}, {cam_pos_0[1]:.4f}, {cam_pos_0[2]:.4f})")
    print(f"  Rotated camera pos:  ({new_cam_pos[0]:.4f}, {new_cam_pos[1]:.4f}, {new_cam_pos[2]:.4f})")

    # Full c2w rotation
    new_c2w = rotate_c2w_around_point(c2w_0, look_at_point, angle)
    new_cam_pos_full, new_forward, _, _ = analyze_camera(new_c2w)

    print(f"\n[Rotated c2w Matrix]")
    print(f"  New position: ({new_cam_pos_full[0]:.4f}, {new_cam_pos_full[1]:.4f}, {new_cam_pos_full[2]:.4f})")
    print(f"  New forward:  ({new_forward[0]:.4f}, {new_forward[1]:.4f}, {new_forward[2]:.4f})")

    # Verify: new camera should still look at the same point
    new_look_at = estimate_look_at_point(new_cam_pos_full, new_forward, look_distance)
    print(f"\n[Verification]")
    print(f"  Original look-at: ({look_at_point[0]:.4f}, {look_at_point[1]:.4f}, {look_at_point[2]:.4f})")
    print(f"  New look-at:      ({new_look_at[0]:.4f}, {new_look_at[1]:.4f}, {new_look_at[2]:.4f})")
    look_at_diff = np.linalg.norm(new_look_at - look_at_point)
    print(f"  Look-at difference: {look_at_diff:.6f} (should be ~0)")

    # Distance check
    new_dist = np.linalg.norm(new_cam_pos_full - look_at_point)
    print(f"\n  Original distance to look-at: {look_distance:.4f}")
    print(f"  New distance to look-at:      {new_dist:.4f}")

    # Print full rotated transform matrix for reference
    print(f"\n[Rotated Transform Matrix (for first frame)]")
    for row in new_c2w.tolist():
        print(f"  {row}")

    # Compare with original
    print(f"\n[Original Transform Matrix (for reference)]")
    for row in c2w_0.tolist():
        print(f"  {row}")

    # Now analyze the Z translation pattern (cat movement along camera's Z)
    print("\n" + "=" * 60)
    print("Cat Movement Analysis (Z translation changes)")
    print("=" * 60)

    z_values = []
    for i, frame in enumerate(transforms['frames'][:10]):  # First 10 frames
        c2w = np.array(frame['transform_matrix'])
        z = c2w[2, 3]  # Z translation
        z_values.append(z)
        print(f"  Frame {i:02d}: Z = {z:.4f}")

    print(f"\n  Z range: {min(z_values):.4f} to {max(z_values):.4f}")

    # The cat moves along camera's view direction
    # When Z becomes more positive, cat moves away from camera
    # This matches what we see - cat walks forward (away)

    print("\n" + "=" * 60)
    print("Unity -> NeRF Coordinate Verification")
    print("=" * 60)

    # Unity original data (from sync_metadata.json)
    cam_unity = np.array([-156.3583221435547, -23.88477325439453, 15.85699462890625])
    obj_unity = np.array([-152.6715087890625, -30.0, 25.959999084472656])
    map_pos = np.array([-150.85, -30.0, 3.66])
    map_scale = np.array([3, 3, 3])

    print(f"\n[Unity Coordinates]")
    print(f"  Camera: {cam_unity}")
    print(f"  Object (cat): {obj_unity}")
    print(f"  Map transform: pos={map_pos}, scale={map_scale}")

    # Apply normalization (like json_sync_utils.py does)
    # 1. Subtract map position
    # 2. Divide by scale
    # 3. Flip Z for NeRF convention

    cam_local = (cam_unity - map_pos) / map_scale
    obj_local = (obj_unity - map_pos) / map_scale

    print(f"\n[After Map Transform (local coords)]")
    print(f"  Camera: ({cam_local[0]:.4f}, {cam_local[1]:.4f}, {cam_local[2]:.4f})")
    print(f"  Object: ({obj_local[0]:.4f}, {obj_local[1]:.4f}, {obj_local[2]:.4f})")

    # Apply Z flip for NeRF
    cam_nerf = np.array([cam_local[0], cam_local[1], -cam_local[2]])
    obj_nerf = np.array([obj_local[0], obj_local[1], -obj_local[2]])

    print(f"\n[After Z-flip (NeRF coords)]")
    print(f"  Camera: ({cam_nerf[0]:.4f}, {cam_nerf[1]:.4f}, {cam_nerf[2]:.4f})")
    print(f"  Object: ({obj_nerf[0]:.4f}, {obj_nerf[1]:.4f}, {obj_nerf[2]:.4f})")

    # Compare with transforms_train.json
    print(f"\n[From transforms_train.json]")
    print(f"  Camera position (frame 0): ({cam_pos_0[0]:.4f}, {cam_pos_0[1]:.4f}, {cam_pos_0[2]:.4f})")

    # Key insight: object position is the correct rotation center!
    print(f"\n" + "=" * 60)
    print("CORRECT ROTATION CENTER = Object Position in NeRF coords")
    print("=" * 60)
    print(f"  Rotation center should be: ({obj_nerf[0]:.4f}, {obj_nerf[1]:.4f}, {obj_nerf[2]:.4f})")

    # Now let's verify rotation with correct center
    rotation_center = obj_nerf
    print(f"\n[Rotating Camera {angle} degrees around correct center]")

    new_c2w_correct = rotate_c2w_around_point(c2w_0, rotation_center, angle)
    new_pos_correct, new_fwd_correct, _, _ = analyze_camera(new_c2w_correct)

    print(f"\n  Original camera pos: ({cam_pos_0[0]:.4f}, {cam_pos_0[1]:.4f}, {cam_pos_0[2]:.4f})")
    print(f"  Rotated camera pos:  ({new_pos_correct[0]:.4f}, {new_pos_correct[1]:.4f}, {new_pos_correct[2]:.4f})")
    print(f"  Original forward:    ({forward_0[0]:.4f}, {forward_0[1]:.4f}, {forward_0[2]:.4f})")
    print(f"  Rotated forward:     ({new_fwd_correct[0]:.4f}, {new_fwd_correct[1]:.4f}, {new_fwd_correct[2]:.4f})")

    # Verify distances
    orig_dist_to_obj = np.linalg.norm(cam_pos_0 - rotation_center)
    new_dist_to_obj = np.linalg.norm(new_pos_correct - rotation_center)
    print(f"\n[Distance Verification]")
    print(f"  Original distance to object: {orig_dist_to_obj:.4f}")
    print(f"  New distance to object:      {new_dist_to_obj:.4f}")

    # Print the corrected transform matrix
    print(f"\n[Corrected Rotated Transform Matrix]")
    for row in new_c2w_correct.tolist():
        print(f"  {row}")

if __name__ == "__main__":
    main()
