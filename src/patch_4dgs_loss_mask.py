#!/usr/bin/env python
"""
Patch 4DGS to apply loss masking using alpha channel.

Problem: Even with white background compositing, Gaussians form in background
         regions because loss still flows to those areas.
Solution: Mask the loss computation so background pixels don't contribute.

This patch modifies:
1. cameras.py - Store gt_alpha_mask for later use in training
2. train.py - Apply alpha mask to loss computation

Usage:
    python src/patch_4dgs_loss_mask.py external/4dgs

After patching, train with:
    python manage.py train data/your_scene --extra="--white_background"
"""

import sys
import os
import re

PATCH_MARKER = "[LOSS MASK PATCH]"


def patch_cameras(file_path):
    """Patch cameras.py to store gt_alpha_mask."""

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if PATCH_MARKER in content:
        print(f"[LOSS MASK] Already patched: {file_path}")
        return True

    # Find where self.mask = mask is set and add alpha mask storage
    if "self.mask = mask" not in content:
        print(f"[Error] Could not find 'self.mask = mask' in {file_path}")
        return False

    # Add storage for gt_alpha_mask after self.mask = mask
    old_code = "self.mask = mask"
    new_code = """self.mask = mask
        # [LOSS MASK PATCH] Store alpha mask for loss masking during training
        self.gt_alpha_mask = gt_alpha_mask"""

    patched_content = content.replace(old_code, new_code)

    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"[LOSS MASK] Patched: {file_path}")
    return True


def patch_train(file_path):
    """Patch train.py to apply loss masking."""

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if PATCH_MARKER in content:
        print(f"[LOSS MASK] Already patched: {file_path}")
        return True

    # Find the loss computation section and add masking
    # Original: Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
    if "Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])" not in content:
        print(f"[Error] Could not find L1 loss computation in {file_path}")
        return False

    # We need to:
    # 1. Collect alpha masks along with images
    # 2. Apply mask to loss computation

    # First, add alpha mask collection in the render loop
    # Find: gt_images.append(gt_image.unsqueeze(0))
    old_collect = "gt_images.append(gt_image.unsqueeze(0))"
    new_collect = """gt_images.append(gt_image.unsqueeze(0))
            # [LOSS MASK PATCH] Collect alpha masks for loss masking
            if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                alpha_masks.append(viewpoint_cam.gt_alpha_mask.cuda().unsqueeze(0))
            else:
                alpha_masks.append(None)"""

    if old_collect not in content:
        print(f"[Error] Could not find gt_images collection in {file_path}")
        return False

    patched_content = content.replace(old_collect, new_collect)

    # Add alpha_masks list initialization
    # Find: gt_images = []
    old_init = """images = []
        gt_images = []"""
    new_init = """images = []
        gt_images = []
        alpha_masks = []  # [LOSS MASK PATCH] For loss masking"""

    patched_content = patched_content.replace(old_init, new_init)

    # Now modify the loss computation to use masking
    # Find: Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
    old_loss = "Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])"
    new_loss = """# [LOSS MASK PATCH] Apply alpha mask to loss computation
        # Combine alpha masks if any exist
        combined_mask = None
        valid_masks = [m for m in alpha_masks if m is not None]
        if valid_masks:
            combined_mask = torch.cat(valid_masks, 0)

        if combined_mask is not None:
            # Masked L1: only compute loss where alpha > 0.5
            mask_binary = (combined_mask > 0.5).float()
            masked_render = image_tensor * mask_binary
            masked_gt = gt_image_tensor[:,:3,:,:] * mask_binary
            # Normalize by number of valid pixels to maintain loss scale
            valid_pixels = mask_binary.sum() + 1e-8
            Ll1 = torch.abs(masked_render - masked_gt).sum() / valid_pixels
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])"""

    patched_content = patched_content.replace(old_loss, new_loss)

    # Also modify SSIM loss if lambda_dssim is used
    old_ssim = "ssim_loss = ssim(image_tensor,gt_image_tensor)"
    new_ssim = """# [LOSS MASK PATCH] Apply mask to SSIM if available
            if combined_mask is not None:
                # For SSIM with masking, apply mask and compute on masked regions
                mask_binary = (combined_mask > 0.5).float()
                ssim_loss = ssim(image_tensor * mask_binary, gt_image_tensor[:,:3,:,:] * mask_binary)
            else:
                ssim_loss = ssim(image_tensor, gt_image_tensor)"""

    patched_content = patched_content.replace(old_ssim, new_ssim)

    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"[LOSS MASK] Patched: {file_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_4dgs_loss_mask.py <path_to_4dgs>")
        print("")
        print("Example:")
        print("  python src/patch_4dgs_loss_mask.py external/4dgs")
        sys.exit(1)

    base_path = sys.argv[1]

    # Patch both files
    cameras_path = os.path.join(base_path, "scene", "cameras.py")
    train_path = os.path.join(base_path, "train.py")

    success1 = patch_cameras(cameras_path)
    success2 = patch_train(train_path)

    if success1 and success2:
        print("")
        print("[LOSS MASK] Successfully patched 4DGS for loss masking!")
        print("[LOSS MASK] Background pixels will be excluded from loss computation")
        print("[LOSS MASK] This prevents Gaussians from forming in transparent regions")

    sys.exit(0 if (success1 and success2) else 1)


if __name__ == "__main__":
    main()
