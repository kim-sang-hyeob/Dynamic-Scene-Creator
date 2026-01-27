#!/usr/bin/env python
"""
Patch 4DGS to properly use alpha channel from transparent PNG images.

Problem: 4DGS ignores alpha channel - Gaussians learn background colors.
Solution: Extract alpha from RGBA images and use as mask during training.

This patch modifies:
1. camera_utils.py - Extract alpha channel from 4-channel images
2. cameras.py - Composite GT image onto white background (matches --white_background render)

Usage:
    python src/patch_4dgs_alpha_mask.py external/4dgs

After patching, train with:
    python manage.py train data/your_scene --extra="--white_background"
"""

import sys
import os
import re

PATCH_MARKER = "[ALPHA PATCH]"


def patch_camera_utils(file_path):
    """Patch camera_utils.py to extract alpha channel."""

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if PATCH_MARKER in content:
        print(f"[ALPHA] Already patched: {file_path}")
        return True

    # Use regex to find and replace gt_alpha_mask=None with gt_alpha_mask=gt_alpha_mask
    # And add the alpha extraction code before the return statement

    # Pattern to find the return Camera(...gt_alpha_mask=None...) in loadCam function
    if "gt_alpha_mask=None" not in content:
        print(f"[Error] Could not find gt_alpha_mask=None in {file_path}")
        return False

    # Find the loadCam function and add alpha extraction before return
    # Match: def loadCam(...): followed by anything up to return Camera(
    pattern = r'(def loadCam\(args, id, cam_info, resolution_scale\):)(.*?)(return Camera\()'

    def replacement(match):
        func_def = match.group(1)
        middle = match.group(2)
        return_stmt = match.group(3)

        # Add alpha extraction code
        alpha_code = """
    # [ALPHA PATCH] Extract alpha channel from RGBA images for background masking
    gt_alpha_mask = None
    if cam_info.image.shape[0] == 4:
        gt_alpha_mask = cam_info.image[3:4, :, :].clone()  # Clone to avoid memory issues
"""
        return func_def + alpha_code + "\n    " + return_stmt

    patched_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Replace gt_alpha_mask=None with gt_alpha_mask=gt_alpha_mask
    patched_content = patched_content.replace("gt_alpha_mask=None", "gt_alpha_mask=gt_alpha_mask")

    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"[ALPHA] Patched: {file_path}")
    return True


def patch_cameras(file_path):
    """Patch cameras.py to composite GT onto white background."""

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if PATCH_MARKER in content:
        print(f"[ALPHA] Already patched: {file_path}")
        return True

    # Find the gt_alpha_mask handling code
    # Original: self.original_image *= gt_alpha_mask (makes background BLACK)
    # New: Composite onto white background (matches --white_background render)
    old_code = """        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask"""

    new_code = """        if gt_alpha_mask is not None:
            # [ALPHA PATCH] Composite onto white background (matches --white_background render)
            # Formula: RGB * alpha + white * (1 - alpha)
            self.original_image = self.original_image * gt_alpha_mask + (1.0 - gt_alpha_mask)"""

    if old_code not in content:
        print(f"[Error] Could not find gt_alpha_mask handling in {file_path}")
        print("        Looking for: 'self.original_image *= gt_alpha_mask'")
        return False

    patched_content = content.replace(old_code, new_code)

    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"[ALPHA] Patched: {file_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_4dgs_alpha_mask.py <path_to_4dgs>")
        print("")
        print("Example:")
        print("  python src/patch_4dgs_alpha_mask.py external/4dgs")
        sys.exit(1)

    base_path = sys.argv[1]

    # Patch both files
    camera_utils_path = os.path.join(base_path, "utils", "camera_utils.py")
    cameras_path = os.path.join(base_path, "scene", "cameras.py")

    success1 = patch_camera_utils(camera_utils_path)
    success2 = patch_cameras(cameras_path)

    if success1 and success2:
        print("")
        print("[ALPHA] Successfully patched 4DGS for alpha channel support!")
        print("[ALPHA] Train with: --white_background flag")
        print("[ALPHA] Example: python manage.py train data/scene --extra=\"--white_background\"")

    sys.exit(0 if (success1 and success2) else 1)


if __name__ == "__main__":
    main()
