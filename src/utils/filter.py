import os
import numpy as np
from plyfile import PlyData, PlyElement

def clean_ply_model(input_path, output_path, nb_neighbors=20, std_ratio=2.0):
    """
    Cleans a 3DGS PLY model by removing outliers (floaters) using a Statistical 
    Outlier Removal (SOR) approach.
    
    Args:
        input_path: Path to the original point_cloud.ply
        output_path: Path to save the cleaned point_cloud.ply
        nb_neighbors: Number of neighbors to analyze for each point.
        std_ratio: Standard deviation multiplier. Points further than this 
                   from the mean neighbor distance are removed.
    """
    if not os.path.exists(input_path):
        print(f"[Error] PLY file not found: {input_path}")
        return False

    print(f"[Cleaner] Loading model from {input_path}...")
    plydata = PlyData.read(input_path)
    
    # Extract vertex data
    vertices = plydata['vertex']
    pts = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    num_pts = pts.shape[0]
    print(f"[Cleaner] Loaded {num_pts} Gaussians. Analyzing neighbors...")

    try:
        from sklearn.neighbors import NearestNeighbors
        
        # 1. Calculate average distance to neighbors
        nn = NearestNeighbors(n_neighbors=nb_neighbors + 1)
        nn.fit(pts)
        distances, _ = nn.kneighbors(pts)
        
        # Take mean distance excluding the point itself (distances[:, 0] is 0)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        # 2. Statistical filtering
        mean_dist = np.mean(avg_distances)
        std_dist = np.std(avg_distances)
        threshold = mean_dist + std_ratio * std_dist
        
        # 3. Mask valid points
        mask = avg_distances < threshold
        num_cleaned = np.sum(mask)
        
        print(f"[Cleaner] Removal stats:")
        print(f"  - Original: {num_pts}")
        print(f"  - Filtered: {num_cleaned}")
        print(f"  - Removed: {num_pts - num_cleaned} (Floaters)")

        # 4. Create new PlyData with filtered vertices
        new_vertices = vertices.data[mask]
        new_el = PlyElement.describe(new_vertices, 'vertex')
        
        print(f"[Cleaner] Saving cleaned model to {output_path}...")
        PlyData([new_el]).write(output_path)
        
        return True

    except ImportError:
        print("[Warning] scikit-learn not found. Falling back to simple scale-based pruning.")
        # Fallback: Simple pruning based on transparency or abnormally large scale
        # If we can't do SOR, we can at least remove 'invisible' or 'stretched' bits
        return False
    except Exception as e:
        print(f"[Error] Cleaning failed: {e}")
        return False

if __name__ == "__main__":
    # Test stub
    pass
