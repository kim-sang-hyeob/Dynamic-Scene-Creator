import os
import subprocess
import sys
import shutil

class SfmManager:
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.input_dir = os.path.join(self.data_dir, "input")
        self.database_path = os.path.join(self.data_dir, "database.db")
        self.sparse_raw_dir = os.path.join(self.data_dir, "sparse_raw")
        self.sparse_dir = os.path.join(self.data_dir, "sparse")

    def run_full_pipeline(self):
        """
        Runs the full COLMAP pipeline from images to undistorted sparse reconstruction.
        """
        print(f"[SFM] Starting COLMAP pipeline in {self.data_dir}...")
        
        # 1. Feature Extraction
        self._run_command([
            "colmap", "feature_extractor",
            "--database_path", self.database_path,
            "--image_path", self.input_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV", # More robust distortion modeling
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.max_num_features", "16384",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1"
        ])

        # 2. Matching
        self._run_command([
            "colmap", "sequential_matcher",
            "--database_path", self.database_path,
            "--SiftMatching.use_gpu", "0",
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7"
        ])

        # 3. Mapping (Raw Sparse Reconstruction)
        os.makedirs(self.sparse_raw_dir, exist_ok=True)
        self._run_command([
            "colmap", "mapper",
            "--database_path", self.database_path,
            "--image_path", self.input_dir,
            "--output_path", self.sparse_raw_dir,
            "--Mapper.init_min_num_inliers", "15",
            "--Mapper.init_min_tri_angle", "8",
            "--Mapper.ba_global_max_num_iterations", "50"
        ])

        # 4. Undistortion
        if not os.path.exists(os.path.join(self.sparse_raw_dir, "0")):
            print("[Error] COLMAP mapping failed. No reconstruction found.")
            return False

        print("[SFM] COLMAP mapping successful. Undistorting to PINHOLE...")
        
        # Create a temporary directory for undistortion
        temp_undistorted_dir = os.path.join(self.data_dir, "temp_undistorted")
        os.makedirs(temp_undistorted_dir, exist_ok=True)

        # Undistort images and convert camera model to PINHOLE (required by FastGS)
        self._run_command([
            "colmap", "image_undistorter",
            "--image_path", self.input_dir,
            "--input_path", os.path.join(self.sparse_raw_dir, "0"),
            "--output_path", temp_undistorted_dir,
            "--output_type", "COLMAP"
        ])
        
        # Move files to the final structure expected by 3DGS
        # 1. Move images to data_dir/images
        source_images = os.path.join(temp_undistorted_dir, "images")
        dest_images = os.path.join(self.data_dir, "images")
        if os.path.exists(dest_images):
            shutil.rmtree(dest_images)
        shutil.move(source_images, dest_images)

        # 2. Move sparse files to data_dir/sparse/0/
        os.makedirs(os.path.join(self.sparse_dir, "0"), exist_ok=True)
        source_sparse = os.path.join(temp_undistorted_dir, "sparse")
        # The undistorter puts cameras.bin etc directly in 'sparse' folder
        for f in os.listdir(source_sparse):
            shutil.move(os.path.join(source_sparse, f), os.path.join(self.sparse_dir, "0", f))

        # Cleanup temp
        shutil.rmtree(temp_undistorted_dir)
        
        print(f"[SFM] Pipeline complete. Ready for training in {self.data_dir}")
        return True
        
        # Note: image_undistorter creates 'images' and 'sparse' in the output_path.
        # It might conflict if we aren't careful, but 3DGS usually wants them there.
        print(f"[SFM] Pipeline complete. Ready for training in {self.data_dir}")
        return True

    def _run_command(self, cmd):
        print(f"  [Exec] {' '.join(cmd)}")
        # Set environment variable to avoid Qt GUI errors on headless servers
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        try:
            subprocess.check_call(cmd, env=env)
        except subprocess.CalledProcessError as e:
            print(f"  [Error] Command failed: {e}")
            raise e

def auto_process_video(video_path, project_dir, num_frames=50):
    """
    High-level entry point: Video -> Frames -> COLMAP -> 3DGS Ready
    """
    from src.isometric_utils import extract_frames_from_video
    
    # 1. Prepare directory
    os.makedirs(project_dir, exist_ok=True)
    image_dir = os.path.join(project_dir, "input")
    
    # 2. Extract frames
    print(f"[Auto] Extracting {num_frames} frames from {video_path}...")
    extract_frames_from_video(video_path, image_dir, num_frames=num_frames)
    
    # 3. Run SfM
    sfm = SfmManager(project_dir)
    return sfm.run_full_pipeline()
