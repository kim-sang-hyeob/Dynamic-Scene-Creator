import os
import sys
import shutil
import subprocess
import numpy as np
import torch

class Dust3rManager:
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.input_dir = os.path.join(self.data_dir, "input")
        self.dust3r_path = os.path.join(os.getcwd(), "external/dust3r")
        self.weights_path = os.path.join(self.dust3r_path, "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        
    def check_environment(self):
        """ Checks if Dust3R repository and dependencies are present. """
        if not os.path.exists(self.dust3r_path):
            print(f"[Error] Dust3R not found at {self.dust3r_path}")
            return False
        if not os.path.exists(self.weights_path):
            print(f"[Error] Dust3R weights not found at {self.weights_path}")
            return False
        return True

    def run_inference(self):
        """
        Runs Dust3R inference to get poses and point cloud.
        """
        if not self.check_environment():
            return False
            
        print("[Dust3R] Starting modern pose estimation...")
        
        # Add Dust3R and its submodules to path
        sys.path.append(self.dust3r_path)
        sys.path.append(os.path.join(self.dust3r_path, "croco"))
        
        try:
            from dust3r.inference import inference
            try:
                from dust3r.model import AsymmetricCroCo3DStereo as Dust3rModel
            except ImportError:
                try:
                    from dust3r.model import AsymmetricCroppedInference as Dust3rModel
                except ImportError:
                    print("[Error] Could not find AsymmetricCroCo3DStereo or AsymmetricCroppedInference in dust3r.model")
                    return False

            from dust3r.utils.image import load_images
            try:
                from dust3r.cloud_opt import GlobalAligner
            except ImportError:
                # Direct import if not exposed in __init__
                try:
                    from dust3r.cloud_opt.global_aligner import GlobalAligner
                except ImportError:
                    from dust3r.cloud_opt import global_aligner as GlobalAligner
            
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 1. Load Model
            print(f"[Dust3R] Loading model onto {device}...")
            model = Dust3rModel.from_pretrained(self.weights_path).to(device)
            
            # 2. Load Images
            images = load_images(self.input_dir, size=512)
            if len(images) < 2:
                print("[Error] Not enough images for Dust3R.")
                return False
                
            # 3. Pairwise Inference (All-to-All matching for small frame counts to ELIMINATE ghosting)
            num_images = len(images)
            print(f"[Dust3R] Running intensive All-to-All matching on {num_images} frames...")
            pairs = []
            for i in range(num_images):
                for j in range(i + 1, num_images):
                    pairs.append((images[i], images[j]))
            
            print(f"[Dust3R] Total matching pairs: {len(pairs)}")
            output = inference(pairs, model, device, batch_size=1)
            
            # 4. Global Alignment (Increased iterations for perfect convergence)
            print("[Dust3R] Running global alignment (Maximum Precision Mode)...")
            scene = GlobalAligner(output, device=device)
            loss = scene.compute_global_alignment(init='mst', niter=800, schedule='linear')
            
            # 5. Extract Poses and Points
            print("[Dust3R] Exporting results to 3DGS format...")
            
            # Poses handle list/tensor
            poses = scene.get_im_poses()
            if isinstance(poses, list): poses = torch.stack(poses)
            poses = poses.detach().cpu().numpy() # [N, 4, 4]
            
            # Points handle list/tensor
            pts3d = scene.get_pts3d()
            if isinstance(pts3d, list): pts3d = torch.stack(pts3d)
            pts3d = pts3d.detach().cpu().numpy()

            # Focals extraction
            focals = scene.get_focals()
            if isinstance(focals, list): focals = torch.stack(focals)
            
            # 1. Process Transforms (OpenCV -> OpenGL)
            c2ws, fov_x = self._save_transforms(poses, images, focals, image_size=512)
            
            # 2. Process PLY (Centering, Scaling, Filtering)
            self._save_ply(scene, c2ws, fov_x, images)
            
            # 3. Save debug visualizations
            self._save_debug_visuals(scene)
            
            print("[Dust3R] Processing complete.")
            return True
            
        except ImportError as e:
            print(f"[Error] Dust3R dependencies missing: {e}")
            return False
        except Exception as e:
            print(f"[Error] Dust3R execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_transforms(self, poses, images, focals, image_size):
        import json
        import math
        frames = []
        
        avg_focal = focals.mean().item()
        sample_img = images[0]['img'] if isinstance(images[0], dict) else images[0]
        w, h = sample_img.shape[3], sample_img.shape[2] 
        fov_x = 2 * math.atan(w / (2 * avg_focal))
        
        # 1. Convert to C2W
        c2ws = [np.linalg.inv(p) for p in poses]
        
        # 2. Extract camera positions
        cam_centers = np.array([c[0:3, 3] for c in c2ws])
        
        # Diagnostics
        print(f"[Dust3R] Camera center range: min={cam_centers.min(axis=0)}, max={cam_centers.max(axis=0)}")
        
        # 3. Standardization (OpenCV -> OpenGL)
        # NeRF/3DGS convention: camera looks along -Z, Y is Up.
        # This is a 180-deg rotation around X-axis for each camera local frame.
        for c2w in c2ws:
            c2w[0:3, 1:3] *= -1 
            
        # We will let the PLY export handle the global centering/scaling
        # to ensure points and cameras are in perfect sync.
        return c2ws, fov_x

    def _save_ply(self, scene, c2ws_opengl, fov_x, images):
        """
        Saves the reconstruction in both Blender (transforms.json) and 
        COLMAP (sparse/0/*.txt) formats to ensure compatibility.
        """
        import json
        
        # 0. Define pathing
        ply_path = os.path.join(self.data_dir, "points3d.ply") # Blender naming
        sparse_dir = os.path.join(self.data_dir, "sparse/0")
        os.makedirs(sparse_dir, exist_ok=True)
        
        try:
            # 1. Extract Raw Point Cloud Data
            pts_raw = scene.get_pts3d()
            if isinstance(pts_raw, list): pts_raw = torch.stack(pts_raw)
            pts = pts_raw.detach().cpu().numpy().reshape(-1, 3)
            
            # 2. Extract Confidence and Filter
            conf = scene.get_conf()
            if isinstance(conf, list): conf = torch.stack(conf)
            conf = conf.detach().cpu().numpy().reshape(-1)
            
            # 3. Handle Images/Colors (Sync with PTS)
            imgs = getattr(scene, 'imgs', None)
            if imgs is None: imgs = scene.get_images()
            
            if isinstance(imgs, list):
                if isinstance(imgs[0], np.ndarray): imgs = np.stack(imgs)
                else: imgs = torch.stack(imgs).detach().cpu().numpy()
            elif isinstance(imgs, torch.Tensor):
                imgs = imgs.detach().cpu().numpy()
            
            if len(imgs.shape) == 4 and imgs.shape[1] == 3:
                imgs = imgs.transpose(0, 2, 3, 1) # (N, 3, H, W) -> (N, H, W, 3)
            colors = imgs.reshape(-1, 3)
            
            # Filter
            mask = conf > 2.0
            if mask.sum() < 1000: mask = conf > 1.0 # Fallback
            
            pts_filtered = pts[mask]
            colors_filtered = colors[mask]
            
            # 4. Global Transformation (Centering on POINTS)
            scene_center = np.median(pts_filtered, axis=0)
            pts_ready = pts_filtered - scene_center
            
            # Scaling
            scale = 1.0 / (np.abs(pts_ready).max() + 1e-8)
            pts_ready *= scale
            
            # 5. Save Blender Format (OpenGL/NeRF)
            image_dir = os.path.join(self.data_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            
            from PIL import Image
            frames = []
            for i, (c2w, img_data) in enumerate(zip(c2ws_opengl, images)):
                # Apply normalization
                c2w[0:3, 3] = (c2w[0:3, 3] - scene_center) * scale
                
                # Save Image
                img_path = os.path.join(image_dir, f"{i}.png")
                img_tensor = img_data['img'] if isinstance(img_data, dict) else img_data
                if isinstance(img_tensor, torch.Tensor):
                    img_np = (img_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                else:
                    img_np = img_tensor
                    if img_np.shape[0] == 3: img_np = img_np.transpose(1, 2, 0)
                    if img_np.max() <= 1.0: img_np = (img_np * 255).astype(np.uint8)
                Image.fromarray(img_np).save(img_path)
                
                frames.append({
                    "file_path": f"images/{i}",
                    "rotation": 0,
                    "transform_matrix": c2w.tolist()
                })
            
            json_data = {"camera_angle_x": fov_x, "frames": frames}
            for name in ["transforms_train.json", "transforms_test.json"]:
                with open(os.path.join(self.data_dir, name), 'w') as f:
                    json.dump(json_data, f, indent=4)
            
            # 6. Save COLMAP Format (OpenCV) text files
            poses_w2c = scene.get_im_poses()
            if isinstance(poses_w2c, list): poses_w2c = torch.stack(poses_w2c)
            poses_w2c = poses_w2c.detach().cpu().numpy()
            
            def rotmat2q(R):
                tr = np.trace(R)
                if tr > 0:
                    S = np.sqrt(tr + 1.0) * 2; qw = 0.25 * S; qx = (R[2, 1] - R[1, 2]) / S
                    qy = (R[0, 2] - R[2, 0]) / S; qz = (R[1, 0] - R[0, 1]) / S
                elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                    S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; qw = (R[2, 1] - R[1, 2]) / S
                    qx = 0.25 * S; qy = (R[0, 1] + R[1, 0]) / S; qz = (R[0, 2] + R[2, 0]) / S
                elif R[1, 1] > R[2, 2]:
                    S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; qw = (R[0, 2] - R[2, 0]) / S
                    qx = (R[0, 1] + R[1, 0]) / S; qy = 0.25 * S; qz = (R[1, 2] + R[2, 1]) / S
                else:
                    S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; qw = (R[1, 0] - R[0, 1]) / S
                    qx = (R[0, 2] + R[2, 0]) / S; qy = (R[1, 2] + R[2, 1]) / S; qz = 0.25 * S
                return np.array([qw, qx, qy, qz])

            focal = (512 / (2 * np.tan(fov_x / 2)))
            with open(os.path.join(sparse_dir, "cameras.txt"), 'w') as f:
                f.write(f"1 PINHOLE 512 512 {focal} {focal} 256 256\n")
            
            with open(os.path.join(sparse_dir, "images.txt"), 'w') as f:
                for i, w2c in enumerate(poses_w2c):
                    c2w_cv = np.linalg.inv(w2c)
                    c2w_cv[0:3, 3] = (c2w_cv[0:3, 3] - scene_center) * scale
                    w2c_norm = np.linalg.inv(c2w_cv)
                    q = rotmat2q(w2c_norm[0:3, 0:3]); t = w2c_norm[0:3, 3]
                    f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {i}.png\n\n")
            
            with open(os.path.join(sparse_dir, "points3D.txt"), 'w') as f:
                for i, (p, c) in enumerate(zip(pts_ready, colors_filtered)):
                    f.write(f"{i} {p[0]} {p[1]} {p[2]} {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)} 0\n")

            # 7. Write PLY (CRITICAL: Include Normals for 3DGS/FastGS reader)
            from plyfile import PlyData, PlyElement
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            
            vertex = np.empty(len(pts_ready), dtype=dtype)
            vertex['x'], vertex['y'], vertex['z'] = pts_ready[:, 0], pts_ready[:, 1], pts_ready[:, 2]
            vertex['nx'], vertex['ny'], vertex['nz'] = 0, 0, 0 # Zero normals
            vertex['red'] = (colors_filtered[:, 0]*255).astype(np.uint8)
            vertex['green'] = (colors_filtered[:, 1]*255).astype(np.uint8)
            vertex['blue'] = (colors_filtered[:, 2]*255).astype(np.uint8)
            
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el]).write(ply_path) # Root (Blender)
            PlyData([el]).write(os.path.join(sparse_dir, "points3D.ply")) # Sparse (COLMAP)

            print(f"[Dust3R] Normalized dataset saved: {len(pts_ready)} points (with normals)")
        except Exception as e:
            print(f"[Error] Failed to save 3DGS data: {e}")
            import traceback
            traceback.print_exc()

    def _save_debug_visuals(self, scene):
        """
        Saves correspondence/confidence heatmaps for debugging.
        """
        debug_dir = os.path.join(self.data_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            conf = scene.get_conf()
            if isinstance(conf, list): conf = torch.stack(conf)
            conf = conf.detach().cpu().numpy() # [N, H, W]
            
            for i in range(min(5, len(conf))): # Save first 5 frames
                plt.figure(figsize=(10, 5))
                plt.imshow(conf[i], cmap='jet')
                plt.colorbar(label='Confidence')
                plt.title(f"Confidence Map - Frame {i}")
                plt.axis('off')
                plt.savefig(os.path.join(debug_dir, f"confidence_{i}.png"))
                plt.close()
            
            print(f"[Dust3R] Debug visualizations saved to {debug_dir}")
        except ImportError:
            print("[Warning] matplotlib not found, skipping debug visualizations.")
        except Exception as e:
            print(f"[Warning] Failed to save debug visuals: {e}")

def dust3r_process_video(video_path, project_dir, num_frames=24):
    """
    Orchestrates the Dust3R video processing pipeline.
    """
    from src.isometric_utils import extract_frames_from_video
    
    # 1. Prepare directory
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    os.makedirs(os.path.join(project_dir, "input"), exist_ok=True)
    
    # 2. Extract frames
    extract_frames_from_video(video_path, os.path.join(project_dir, "input"), num_frames=num_frames)
    
    # 3. Run Dust3R
    dm = Dust3rManager(project_dir)
    return dm.run_inference()
