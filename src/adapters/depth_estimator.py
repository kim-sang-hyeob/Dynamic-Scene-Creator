#!/usr/bin/env python
"""
Monocular depth estimation using MiDaS for better initial point cloud generation.

Usage:
    from src.adapters.depth_estimator import DepthEstimator

    estimator = DepthEstimator()
    depth_map = estimator.estimate(image)  # Returns relative depth map
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F


class DepthEstimator:
    """MiDaS-based monocular depth estimation."""

    def __init__(self, model_type="MiDaS_small", device=None):
        """
        Initialize depth estimator.

        Args:
            model_type: MiDaS model variant
                - "DPT_Large": Highest quality, slowest
                - "DPT_Hybrid": Good balance
                - "MiDaS_small": Fastest, good enough for point init
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.transform = None

    def _load_model(self):
        """Lazy load MiDaS model."""
        if self.model is not None:
            return

        print(f"[Depth] Loading MiDaS model: {self.model_type}")

        # Load MiDaS from torch hub
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print(f"[Depth] Model loaded on {self.device}")

    def estimate(self, image, normalize=True):
        """
        Estimate depth from a single image.

        Args:
            image: numpy array (H, W, 3) BGR or RGB, or PIL Image
            normalize: If True, normalize depth to [0, 1] range

        Returns:
            depth_map: numpy array (H, W) with relative depth values
                       Higher values = farther from camera
        """
        self._load_model()

        # Convert to RGB numpy if needed
        if isinstance(image, Image.Image):
            img_rgb = np.array(image)
        elif image.shape[-1] == 4:  # RGBA
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:  # BGR
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_size = (img_rgb.shape[1], img_rgb.shape[0])  # (W, H)

        # Apply MiDaS transform
        input_batch = self.transform(img_rgb).to(self.device)

        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original size
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=original_size[::-1],  # (H, W)
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # MiDaS outputs inverse depth (closer = higher values)
        # Convert to regular depth (closer = lower values)
        depth_map = depth_map.max() - depth_map

        if normalize:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros_like(depth_map)

        return depth_map

    def estimate_metric_depth(self, image, camera_distance, alpha_mask=None):
        """
        Estimate metric depth using camera distance as reference.

        Args:
            image: Input image (H, W, 3) or (H, W, 4)
            camera_distance: Known distance from camera to object center
            alpha_mask: Optional (H, W) mask, depth estimated only for foreground

        Returns:
            depth_map: (H, W) metric depth values
        """
        # Get relative depth
        rel_depth = self.estimate(image, normalize=True)

        if alpha_mask is not None:
            # Use median depth of foreground as reference
            fg_mask = alpha_mask > 127
            if fg_mask.sum() > 0:
                fg_depths = rel_depth[fg_mask]
                median_rel_depth = np.median(fg_depths)

                # Scale so that median foreground depth = camera_distance
                if median_rel_depth > 0:
                    scale = camera_distance / median_rel_depth
                else:
                    scale = camera_distance

                metric_depth = rel_depth * scale
            else:
                metric_depth = rel_depth * camera_distance
        else:
            # Use center depth as reference
            h, w = rel_depth.shape
            center_depth = rel_depth[h//2, w//2]
            if center_depth > 0:
                scale = camera_distance / center_depth
            else:
                scale = camera_distance
            metric_depth = rel_depth * scale

        return metric_depth


def estimate_depth_for_frames(img_dir, frames, output_dir=None, model_type="MiDaS_small"):
    """
    Estimate depth for multiple frames.

    Args:
        img_dir: Directory containing images
        frames: List of frame dicts with 'file_path'
        output_dir: Optional directory to save depth maps
        model_type: MiDaS model type

    Returns:
        depth_maps: Dict mapping filename to depth map
    """
    estimator = DepthEstimator(model_type)
    depth_maps = {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        img_path = os.path.join(img_dir, frame['file_path'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        print(f"[Depth] Estimating depth for {frame['file_path']} ({i+1}/{len(frames)})")

        # Get alpha mask if available
        alpha_mask = img[:, :, 3] if img.shape[-1] == 4 else None

        # Estimate depth
        depth = estimator.estimate(img, normalize=True)
        depth_maps[frame['file_path']] = depth

        # Save depth map if output_dir specified
        if output_dir:
            depth_vis = (depth * 255).astype(np.uint8)
            depth_path = os.path.join(output_dir, frame['file_path'].replace('.png', '_depth.png'))
            cv2.imwrite(depth_path, depth_vis)

    return depth_maps


# Standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate depth from images")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("--output", "-o", help="Output directory for depth maps")
    parser.add_argument("--model", default="MiDaS_small",
                        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                        help="MiDaS model type")

    args = parser.parse_args()

    estimator = DepthEstimator(args.model)

    if os.path.isfile(args.input):
        # Single image
        img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        depth = estimator.estimate(img)

        # Visualize
        depth_vis = (depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        if args.output:
            cv2.imwrite(args.output, depth_color)
            print(f"Saved to {args.output}")
        else:
            cv2.imshow("Depth", depth_color)
            cv2.waitKey(0)
    else:
        # Directory
        output_dir = args.output or os.path.join(args.input, "depth")
        os.makedirs(output_dir, exist_ok=True)

        for fname in sorted(os.listdir(args.input)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(args.input, fname)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                if img is not None:
                    print(f"Processing {fname}...")
                    depth = estimator.estimate(img)
                    depth_vis = (depth * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

                    out_path = os.path.join(output_dir, fname.replace('.png', '_depth.png').replace('.jpg', '_depth.png'))
                    cv2.imwrite(out_path, depth_color)

        print(f"Saved depth maps to {output_dir}")
