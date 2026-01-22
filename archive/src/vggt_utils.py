import os
import sys
import shutil
import subprocess
import cv2
import numpy as np

class VggtManager:
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.images_dir = os.path.join(self.data_dir, "images")
        self.vggt_path = os.path.join(os.getcwd(), "external/vggt")
        
    def check_environment(self):
        """ Checks if VGGT repository is present. """
        if not os.path.exists(self.vggt_path):
            print(f"[Error] VGGT not found at {self.vggt_path}")
            return False
        return True

    def run_inference(self, use_ba=True, camera_type="PINHOLE", clean_sparse=True, align_scene=True, low_vram=False, resize=None):
        """
        Runs VGGT inference via its official demo_colmap.py script.
        Stage 1: Added tuning parameters (use_ba, camera_type).
        Stage 2: Added optional cleaning of the SfM result.
        Stage 5: Added scene alignment (rotation) for isometric correction.
        """
        if not self.check_environment():
            return False
            
        # Stage 1 Fix: VGGT internal wrapper is case-sensitive and expects UPPERCASE.
        camera_type = camera_type.upper() if camera_type else "PINHOLE"
        
        print(f"[VGGT] Starting high-fidelity pose estimation (BA={use_ba}, Type={camera_type}, Resize={resize})...")
        
        # Optional: Resize images to save memory
        if resize:
            print(f"[VGGT] Resizing images to {resize}px for SfM...")
            for img_name in os.listdir(self.images_dir):
                if img_name.lower().endswith('.png'):
                    img_path = os.path.join(self.images_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        # Maintain aspect ratio
                        if w > h:
                            new_w = resize
                            new_h = int(h * (resize / w))
                        else:
                            new_h = resize
                            new_w = int(w * (resize / h))
                        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(img_path, img_resized)
        
        try:
            # VGGT's demo_colmap.py command
            cmd = [
                sys.executable,
                "demo_colmap.py",
                "--scene_dir", self.data_dir,
            ]
            
            if use_ba:
                cmd.append("--use_ba")
            if camera_type:
                cmd.extend(["--camera_type", camera_type])
            
            if low_vram:
                # vggt specific flags to save memory
                cmd.extend(["--query_frame_num", "2"])
                cmd.extend(["--max_query_pts", "2048"]) # Increased from 512 for better 4DGS seeds
                cmd.append("--shared_camera")
                print("[VGGT] Low VRAM mode: query_frame_num=2, max_query_pts=2048, shared_camera=True")
            
            print(f">>> [VGGT DEBUG] EXECUTING COMMAND: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.vggt_path, capture_output=False)
            
            if result.returncode != 0:
                print(f"[Error] VGGT execution failed with return code {result.returncode}")
                return False
                
            # Standard 3DGS (FastGS) expects sparse/0/
            sparse_root = os.path.join(self.data_dir, "sparse")
            sparse_zero = os.path.join(sparse_root, "0")
            
            if os.path.exists(sparse_root):
                # Restructure to sparse/0 if needed
                if not os.path.exists(sparse_zero):
                    print("[VGGT] Restructuring output to sparse/0 format...")
                    temp_sparse = os.path.join(self.data_dir, "sparse_temp")
                    os.rename(sparse_root, temp_sparse)
                    os.makedirs(sparse_root, exist_ok=True)
                    os.rename(temp_sparse, sparse_zero)
                
                # STAGE 2: Pre-cleaning the SfM result
                if clean_sparse:
                    from src.filter_utils import clean_ply_model
                    ply_path = os.path.join(sparse_zero, "points3D.ply")
                    if os.path.exists(ply_path):
                        print("[Stage 2] Cleaning SfM point cloud before training...")
                        clean_ply_model(ply_path, ply_path, nb_neighbors=20, std_ratio=2.0)
                
                # STAGE 5: Scene Alignment (Isometric Correction)
                if align_scene:
                    from src.align_utils import rotate_reconstruction
                    print("[Stage 5] Aligning scene orientation (90deg X-rotation)...")
                    rotate_reconstruction(sparse_zero, angle_x_deg=90.0)

                print("[VGGT] Processing complete.")
                return True
            else:
                print(f"[Error] VGGT ran but did not generate {sparse_root}")
                return False
            
        except Exception as e:
            print(f"[Error] VGGT execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def vggt_process_video(video_path, project_dir, num_frames=24, use_ba=True, camera_type="PINHOLE", clean_sparse=True, align_scene=True, low_vram=False, resize=None):
    """
    Orchestrates the VGGT video processing pipeline.
    """
    from src.isometric_utils import extract_frames_from_video
    
    # 1. Prepare directory
    # Do NOT rmtree project_dir as it may contain 4DGS timestamps.json
    images_dir = os.path.join(project_dir, "images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)
    
    # 2. Extract frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    interval = max(1, total_frames // num_frames)
    
    extract_frames_from_video(video_path, os.path.join(project_dir, "images"), interval=interval)
    
    # 3. Run VGGT
    vm = VggtManager(project_dir)
    return vm.run_inference(use_ba=use_ba, camera_type=camera_type, clean_sparse=clean_sparse, align_scene=align_scene, low_vram=low_vram, resize=resize)
