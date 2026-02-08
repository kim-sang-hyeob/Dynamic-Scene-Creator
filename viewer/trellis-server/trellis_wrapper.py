"""
TRELLIS Model Wrapper
Mock mode: generates dummy .splat sphere data for dev/test without GPU
Real mode: runs TRELLIS image-to-3D pipeline for actual Gaussian generation
"""
import io
import os
import sys
import tempfile
import numpy as np
from PIL import Image
from typing import Optional

# Add TRELLIS to Python path
TRELLIS_DIR = os.path.join(os.path.expanduser("~"), "TRELLIS")
if os.path.isdir(TRELLIS_DIR) and TRELLIS_DIR not in sys.path:
    sys.path.insert(0, TRELLIS_DIR)


class TrellisGenerator:
    def __init__(self, settings):
        self.is_mock = os.environ.get("TRELLIS_MOCK", "").lower() in ("true", "1", "yes")
        self.model = None

        if not self.is_mock:
            self._load_model(settings)
        else:
            print("Running in MOCK mode (no GPU needed)")

        if self.model is None:
            self.is_mock = True

    def _load_model(self, settings):
        try:
            os.environ.setdefault('SPCONV_ALGO', 'native')
            import torch
            from trellis.pipelines import TrellisImageTo3DPipeline
            model_name = settings.model_name
            print(f"Loading TRELLIS model: {model_name} ...")
            self.model = TrellisImageTo3DPipeline.from_pretrained(model_name)
            self.model.cuda()
            print(f"Loaded TRELLIS: {model_name} on {torch.cuda.get_device_name(0)}")
        except ImportError as e:
            print(f"TRELLIS not installed -> falling back to mock mode ({e})")
        except Exception as e:
            print(f"TRELLIS load failed: {e} -> falling back to mock mode")

    def check_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def generate_from_text(self, prompt_text, seed=-1, params=None):
        """Text -> 3D (mock returns sphere, real mode raises error)"""
        if self.is_mock:
            return self._generate_mock_splat(color_hint=[150, 150, 200])
        raise NotImplementedError(
            "Text-to-3D requires an image generation step. Use image mode."
        )

    def generate_from_image(self, image_bytes, seed=-1, params=None):
        """Image -> 3D Gaussian"""
        params = params or {}
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        if self.is_mock:
            avg_color = self._get_avg_color(image)
            return self._generate_mock_splat(color_hint=avg_color)

        # Real TRELLIS generation
        import torch
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)

        actual_seed = seed if seed >= 0 else 42
        outputs = self.model.run(
            image,
            seed=actual_seed,
            sparse_structure_sampler_params={
                "steps": params.get("sparse_structure_steps", 12),
                "cfg_strength": params.get("sparse_structure_cfg_strength", 7.5),
            },
            slat_sampler_params={
                "steps": params.get("slat_steps", 12),
                "cfg_strength": params.get("slat_cfg_strength", 3.0),
            },
            formats=["gaussian"],
        )

        gaussians = outputs["gaussian"][0]

        # Convert to .splat format (32 bytes per gaussian)
        # This is much smaller than PLY and the frontend parser handles it reliably
        import torch as _torch
        import struct

        xyz = gaussians.get_xyz.detach().cpu().numpy()            # (N, 3) world pos
        scales = gaussians.get_scaling.detach().cpu().numpy()     # (N, 3) already exp'd
        rotations = gaussians.get_rotation.detach().cpu().numpy() # (N, 4) normalized quat
        opacity = gaussians.get_opacity.detach().cpu().numpy()    # (N, 1) sigmoid'd [0,1]
        sh_dc = gaussians._features_dc.detach().cpu().numpy()     # (N, 1, 3) or (N, 3, 1)

        # Apply coordinate transform (same as save_ply default: Y↔Z swap)
        transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        xyz = xyz @ transform.T

        num_gaussians = xyz.shape[0]

        # SH DC to RGB: color = SH_C0 * f_dc + 0.5, then clamp to [0,1]
        SH_C0 = 0.28209479177387814
        sh_flat = sh_dc.reshape(num_gaussians, 3)
        colors = np.clip(SH_C0 * sh_flat + 0.5, 0.0, 1.0)
        colors_u8 = (colors * 255).astype(np.uint8)
        alpha_u8 = (np.clip(opacity, 0.0, 1.0).reshape(-1) * 255).astype(np.uint8)

        # Build .splat buffer: pos(12) + scale(12) + rgba(4) + quat(4) = 32 bytes
        splat_data = np.empty(num_gaussians, dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('sx', '<f4'), ('sy', '<f4'), ('sz', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
            ('q0', 'u1'), ('q1', 'u1'), ('q2', 'u1'), ('q3', 'u1'),
        ])
        splat_data['x'] = xyz[:, 0]
        splat_data['y'] = xyz[:, 1]
        splat_data['z'] = xyz[:, 2]
        splat_data['sx'] = scales[:, 0]
        splat_data['sy'] = scales[:, 1]
        splat_data['sz'] = scales[:, 2]
        splat_data['r'] = colors_u8[:, 0]
        splat_data['g'] = colors_u8[:, 1]
        splat_data['b'] = colors_u8[:, 2]
        splat_data['a'] = alpha_u8

        # Quaternion: normalize and pack to uint8 [0,255] (128=0)
        rot_norm = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-10)
        rot_u8 = np.clip((rot_norm * 128 + 128), 0, 255).astype(np.uint8)
        splat_data['q0'] = rot_u8[:, 0]
        splat_data['q1'] = rot_u8[:, 1]
        splat_data['q2'] = rot_u8[:, 2]
        splat_data['q3'] = rot_u8[:, 3]

        splat_bytes = splat_data.tobytes()

        return {
            "ply_bytes": splat_bytes,
            "gaussian_count": num_gaussians,
            "format": "splat",
            "thumbnail_base64": None,
        }

    # ── Mock mode ──────────────────────────────────────────────────────

    def _generate_mock_splat(self, color_hint=None, num_points=1000):
        """
        Mock mode: generate a sphere of dummy .splat data
        for frontend dev/testing
        """
        color = color_hint or [200, 150, 100]

        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        r = np.random.uniform(0.8, 1.0, num_points)

        positions = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ]).astype(np.float32)

        # .splat binary (32 bytes per gaussian)
        buffer = bytearray()
        for i in range(num_points):
            buffer += positions[i].tobytes()                                    # pos: 12B
            buffer += np.array([0.02, 0.02, 0.02], dtype=np.float32).tobytes()  # scale: 12B
            c = [int(color[0] + np.random.randint(-20, 20)),
                 int(color[1] + np.random.randint(-20, 20)),
                 int(color[2] + np.random.randint(-20, 20)),
                 200]
            buffer += np.clip(c, 0, 255).astype(np.uint8).tobytes()             # RGBA: 4B
            buffer += np.array([255, 128, 128, 128], dtype=np.uint8).tobytes()   # quat: 4B

        return {
            "ply_bytes": bytes(buffer),
            "gaussian_count": num_points,
            "format": "splat",
            "thumbnail_base64": None,
        }

    def _get_avg_color(self, image):
        arr = np.array(image.resize((8, 8)).convert("RGB"))
        return arr.mean(axis=(0, 1)).astype(int).tolist()
