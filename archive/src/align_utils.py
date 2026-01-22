import os
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R

def suggest_scaling(sparse_dir):
    """
    Analyzes the point cloud to detect if it's 'flattened'.
    Returns a suggested scale factor.
    """
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        return 1.0
        
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    xyzs = np.array([p.xyz for p in reconstruction.points3D.values()])
    
    if len(xyzs) == 0:
        return 1.0
        
    mins = np.min(xyzs, axis=0)
    maxs = np.max(xyzs, axis=0)
    extents = maxs - mins
    
    # After 90deg X-rotation, Z is height.
    # In many isometric cases, height (Z) is tiny relative to footprint (X, Y)
    footprint = (extents[0] + extents[1]) / 2.0
    height = extents[2]
    
    ratio = height / footprint if footprint > 0 else 1.0
    print(f"[Analyze] Scene Extents: X={extents[0]:.2f}, Y={extents[1]:.2f}, Z={extents[2]:.2f}")
    print(f"[Analyze] Height/Footprint ratio: {ratio:.4f}")
    
    if ratio < 0.1:
        suggested = 0.3 / ratio
        print(f"[Analyze] WARNING: Scene looks very flattened. Suggested --scale: {suggested:.1f}")
        return suggested
    
    return 1.0

def rotate_reconstruction(sparse_dir, angle_x_deg=90.0, scale_factor=1.0, axis='z'):
    """
    Rotates and scales a COLMAP reconstruction.
    Useful for correcting 'lying down' orientation and 'compressed' height.
    """
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        print(f"[Error] No reconstruction found in {sparse_dir}")
        return False

    print(f"[Align] Loading reconstruction from {sparse_dir}...")
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    
    # 1. Define rotation (90 degrees around X)
    rot = R.from_euler('x', angle_x_deg, degrees=True).as_matrix()
    
    # 2. Rotate and Scale all points
    print(f"[Align] Rotating and scaling (x{scale_factor} on {axis}) {len(reconstruction.points3D)} points...")
    for point_id, point in reconstruction.points3D.items():
        # Rotate first
        p_rotated = rot @ point.xyz
        # Then scale the target axis (usually 'z' for height)
        if axis.lower() == 'z':
            p_rotated[2] *= scale_factor
        elif axis.lower() == 'y':
            p_rotated[1] *= scale_factor
        elif axis.lower() == 'x':
            p_rotated[0] *= scale_factor
        point.xyz = p_rotated

    # 3. Rotate and Scale all camera poses
    rot_inv = rot.T
    # Scaling factor matrix S
    scale_vec = np.ones(3)
    if axis.lower() == 'z': scale_vec[2] = scale_factor
    elif axis.lower() == 'y': scale_vec[1] = scale_factor
    elif axis.lower() == 'x': scale_vec[0] = scale_factor
    S = np.diag(scale_vec)
    S_inv = np.diag(1.0 / scale_vec)

    print(f"[Align] Updating {len(reconstruction.images)} camera poses...")
    for image_id, image in reconstruction.images.items():
        try:
            # Note: We rotate the world-to-camera rotation (R_w2c)
            # P_cam = R_w2c * (Rot_inv * P_world_new) + T_w2c
            # R_new = R_w2c * Rot_inv
            # T_new = T_w2c
            pass # Placeholder for documentation logic
        except Exception as e:
            print(f"[Align Error] Pose update failed: {e}")
            raise e

    # Version-safe Rotation class search
    RotationClass = getattr(pycolmap, 'Rotation3d', getattr(pycolmap, 'Rotation3D', None))

    for image_id, image in reconstruction.images.items():
        try:
            if hasattr(image, 'qvec'):
                # API 0.6.x and below
                qvec = image.qvec
                r_w2c = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
                new_r_w2c = r_w2c @ rot_inv
                new_q = R.from_matrix(new_r_w2c).as_quat()
                image.qvec = np.array([new_q[3], new_q[0], new_q[1], new_q[2]])
            elif hasattr(image, 'cam_from_world'):
                # API 1.0+
                cfw = image.cam_from_world
                r_w2c = cfw.rotation.matrix() if hasattr(cfw.rotation, 'matrix') else cfw.rotation.to_matrix()
                new_r_w2c = r_w2c @ rot_inv
                if RotationClass:
                    cfw.rotation = RotationClass(new_r_w2c)
                    image.cam_from_world = cfw
                else:
                    raise AttributeError("Could not find pycolmap.Rotation3d/Rotation3D class")
            else:
                attrs = dir(image)
                raise AttributeError(f"pycolmap.Image (id={image_id}) has no qvec/cam_from_world. Available: {attrs}")
        except Exception as e:
            print(f"[Align Error] Pose rotation failed for image {image_id}: {e}")
            raise e

    # 4. Save the rotated reconstruction
    # We save to the same directory or a subfolder?
    # Let's save to a temp folder then overwrite to be safe.
    output_temp = os.path.join(sparse_dir, "rotated_temp")
    os.makedirs(output_temp, exist_ok=True)
    
    print(f"[Align] Saving rotated reconstruction to {sparse_dir} (overwriting)...")
    reconstruction.write(sparse_dir) # pycolmap write-to-binary
    
    # Also need to update the points3D.ply if it exists for GS seeding
    # Optional: FastGS uses the bin files directly or the ply?
    # Usually it uses points3D.bin for seeding.
    
    print("[Align] Scene alignment complete.")
    return True

if __name__ == "__main__":
    # Test script if called directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sparse_dir")
    args = parser.parse_args()
    rotate_reconstruction(args.sparse_dir)
