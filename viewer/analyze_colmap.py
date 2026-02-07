
import struct
import numpy as np
import collections
from pathlib import Path

# COLMAP binary file helpers
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        print(f"Reading {num_reg_images} images...", flush=True)
        for image_index in range(num_reg_images):
            if image_index % 10 == 0:
                print(f"  Processing image {image_index}/{num_reg_images}...", end='\r', flush=True)
            binary_image_properties = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            
            # Read points2D
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_points2D):
                fid.read(24) # Skip x, y, point3D_id
            
            images[image_id] = {
                "qvec": qvec, 
                "tvec": tvec, 
                "camera_id": camera_id, 
                "name": image_name
            }
    return images

def analyze_colmap_up(sparse_path):
    images_bin = Path(sparse_path) / "images.bin"
    if not images_bin.exists():
        print(f"File not found: {images_bin}")
        return

    print(f"Reading {images_bin}...")
    images = read_images_binary(images_bin)
    
    ups = []
    centers = []
    
    for img_id, data in images.items():
        R_w2c = qvec2rotmat(data["qvec"])
        R_c2w = R_w2c.T
        
        # In OpenCV (COLMAP):
        # Y-axis (Col 1) is "Down"
        # So -Col 1 is "Up" in world space
        up_world = -R_c2w[:, 1]
        ups.append(up_world)
        
        # Optical center in world space
        center = -R_c2w @ data["tvec"]
        centers.append(center)
        
    ups = np.array(ups)
    centers = np.array(centers)
    
    mean_up = np.mean(ups, axis=0)
    std_up = np.std(ups, axis=0)
    
    print("\n[COLMAP Poses Analysis]")
    print(f"Number of cameras: {len(images)}")
    print(f"Mean Up Vector: {mean_up} (Magnitude: {np.linalg.norm(mean_up):.4f})")
    print(f"Std Up Vector:  {std_up}")
    
    # Orientation classification
    dom_axis = np.argmax(np.abs(mean_up))
    sign = np.sign(mean_up[dom_axis])
    axis_name = ["X", "Y", "Z"][dom_axis]
    print(f"Result: Gravity Up is aligned with {sign:+.0f}{axis_name}")

    # Centroid
    mean_center = np.mean(centers, axis=0)
    print(f"Mean Camera Center: {mean_center}")

    # Suggested rotation to make Y-up
    # We want to rotate 'mean_up' to [0, 1, 0]
    target_up = np.array([0, 1, 0])
    
    # Rotation matrix via cross/dot
    def rotation_matrix_from_vectors(vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-6: return np.eye(3) if c > 0 else -np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-9))

    R = rotation_matrix_from_vectors(mean_up, target_up)
    
    # Euler angles for --bg-rotate
    def euler_from_matrix(matrix):
        sy = np.sqrt(matrix[0,0]**2 + matrix[1,0]**2)
        if sy > 1e-6:
            x, y, z = np.arctan2(matrix[2,1], matrix[2,2]), np.arctan2(-matrix[2,0], sy), np.arctan2(matrix[1,0], matrix[0,0])
        else:
            x, y, z = np.arctan2(-matrix[1,2], matrix[1,1]), np.arctan2(-matrix[2,0], sy), 0
        return np.degrees([x, y, z])

    euler = euler_from_matrix(R)
    print("\n" + "="*40)
    print(f"SUGGESTED MAP ROTATION: {euler}")
    print(f"Cmd: --bg-rotate {euler[0]:.2f} {euler[1]:.2f} {euler[2]:.2f}")
    print("="*40)

if __name__ == "__main__":
    analyze_colmap_up("forest/sparse/0")
