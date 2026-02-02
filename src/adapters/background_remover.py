#!/usr/bin/env python
"""
Background Remover using BiRefNet

Automatically removes background from video frames using BiRefNet model.
Creates transparent PNG images suitable for 4DGS training.

Usage:
    python src/background_remover.py <video_path> <output_dir> [--model birefnet]

Example:
    python src/background_remover.py data/black_cat/output_cat.mp4 data/black_cat_alpha/images
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Check for required packages
def check_dependencies():
    """Check and report missing dependencies."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    try:
        from transformers import AutoModelForImageSegmentation
    except ImportError:
        missing.append("transformers")

    try:
        import torchvision
    except ImportError:
        missing.append("torchvision")

    if missing:
        print(f"[Error] Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


class BiRefNetRemover:
    """Background remover using BiRefNet model."""

    def __init__(self, device=None):
        import torch
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BiRefNet] Loading model on {self.device}...")

        # Load BiRefNet model from HuggingFace
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("[BiRefNet] Model loaded successfully")

    def remove_background(self, image):
        """
        Remove background from image.

        Args:
            image: PIL Image or numpy array (BGR)

        Returns:
            RGBA numpy array with transparent background
        """
        import torch
        from PIL import Image

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        original_size = image.size

        # Prepare input
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid()

        # Get mask
        mask = preds[0].squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        # Resize mask to original size
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)

        # Apply mask to create RGBA image
        # Set RGB to WHITE where mask is low (transparent) - like lego dataset
        rgb = np.array(image)
        mask_binary = (mask > 127).astype(np.float32)[:, :, np.newaxis]
        white_bg = np.ones_like(rgb) * 255
        rgb_composited = (rgb * mask_binary + white_bg * (1 - mask_binary)).astype(np.uint8)
        rgba = np.dstack([rgb_composited, mask])

        return rgba


class RembgRemover:
    """Background remover using rembg library (alternative)."""

    def __init__(self):
        try:
            from rembg import remove, new_session
            self.remove = remove
            # Use u2net for better quality
            self.session = new_session("u2net")
            print("[rembg] Model loaded successfully")
        except ImportError:
            raise ImportError("rembg not installed. Install with: pip install rembg[gpu]")

    def remove_background(self, image):
        """Remove background from image."""
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        result = self.remove(image, session=self.session)
        return np.array(result)


def process_video(video_path, output_dir, model_type="birefnet", max_frames=None, resize=None):
    """
    Process video and remove background from all frames.

    Args:
        video_path: Path to input video
        output_dir: Directory to save output images
        model_type: "birefnet" or "rembg"
        max_frames: Optional max number of frames to process
        resize: Optional resize factor (e.g., 0.5) or tuple (width, height)
    """
    # Initialize remover
    if model_type == "birefnet":
        remover = BiRefNetRemover()
    elif model_type == "rembg":
        remover = RembgRemover()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {video_path}")
    print(f"[Video] {total_frames} frames, {fps:.2f} FPS, {width}x{height}")

    # Calculate output size
    output_size = None
    if resize is not None:
        if isinstance(resize, (int, float)):
            output_size = (int(width * resize), int(height * resize))
        else:
            output_size = resize
        print(f"[Video] Output size: {output_size[0]}x{output_size[1]}")

    # Calculate frame indices
    if max_frames is not None and max_frames < total_frames:
        # Uniform sampling
        frame_indices = []
        for i in range(max_frames):
            idx = int(i * (total_frames - 1) / (max_frames - 1)) if max_frames > 1 else 0
            frame_indices.append(idx)
        print(f"[Video] Sampling {max_frames} frames uniformly")
    else:
        frame_indices = list(range(total_frames))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process frames
    processed = 0
    frame_idx = 0
    output_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_indices:
            # Remove background
            rgba = remover.remove_background(frame)

            # Resize if needed
            if output_size is not None:
                from PIL import Image
                img = Image.fromarray(rgba)
                img = img.resize(output_size, Image.LANCZOS)
                rgba = np.array(img)

            # Save as PNG with alpha
            output_path = os.path.join(output_dir, f"{output_idx:04d}.png")
            from PIL import Image
            Image.fromarray(rgba).save(output_path)

            processed += 1
            output_idx += 1

            if processed % 10 == 0 or processed == len(frame_indices):
                print(f"[Progress] {processed}/{len(frame_indices)} frames processed")

        frame_idx += 1

    cap.release()
    print(f"[Done] Saved {processed} transparent images to {output_dir}")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Remove background from video frames using BiRefNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/background_remover.py data/black_cat/output_cat.mp4 data/black_cat_alpha/images

  # With frame limit and resize
  python src/background_remover.py data/black_cat/output_cat.mp4 data/black_cat_alpha/images \\
      --frames 40 --resize 0.5

  # Use rembg instead of BiRefNet
  python src/background_remover.py data/black_cat/output_cat.mp4 data/black_cat_alpha/images \\
      --model rembg
        """
    )

    parser.add_argument("video", help="Input video path")
    parser.add_argument("output", help="Output directory for transparent images")
    parser.add_argument("--model", choices=["birefnet", "rembg"], default="birefnet",
                        help="Background removal model (default: birefnet)")
    parser.add_argument("--frames", type=int, default=None,
                        help="Max number of frames to process (uniform sampling)")
    parser.add_argument("--resize", type=str, default=None,
                        help="Resize factor (e.g., 0.5) or WxH (e.g., 512x295)")

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Parse resize
    resize = None
    if args.resize:
        if 'x' in args.resize.lower():
            w, h = args.resize.lower().split('x')
            resize = (int(w), int(h))
        else:
            resize = float(args.resize)

    # Process video
    try:
        process_video(
            args.video,
            args.output,
            model_type=args.model,
            max_frames=args.frames,
            resize=resize
        )
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
