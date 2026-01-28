#!/usr/bin/env python
"""
Comprehensive alpha channel patch for 4DGS with transparent PNG images.

This patch enables proper handling of transparent PNG images by:
1. Extracting alpha channel from RGBA images
2. Compositing GT image onto white background (matches --white_background render)
3. Storing alpha mask for loss masking
4. Masking loss computation to exclude background pixels

Files modified:
- camera_utils.py: Extract alpha channel with .clone() to avoid memory issues
- cameras.py: Composite onto white + store alpha mask
- train.py: Apply alpha mask to L1 and SSIM loss

Usage:
    python src/patch_4dgs_alpha.py external/4dgs
    python manage.py train data/scene --extra="--white_background"
"""

import sys
import os

PATCH_MARKER = "[ALPHA PATCH]"


def patch_camera_utils(file_path):
    """Patch camera_utils.py to extract alpha channel."""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    if PATCH_MARKER in content:
        print(f"[ALPHA] Already patched: {file_path}")
        return True

    if "gt_alpha_mask=None" not in content:
        print(f"[Error] Could not find gt_alpha_mask=None in {file_path}")
        return False

    # Add alpha extraction before return Camera(...)
    alpha_code = '''
    # [ALPHA PATCH] Extract alpha channel from RGBA images for background masking
    gt_alpha_mask = None
    if cam_info.image.shape[0] == 4:
        gt_alpha_mask = cam_info.image[3:4, :, :].clone()  # Clone to avoid memory issues

    '''

    # Find "def loadCam" and insert alpha extraction before return
    if "def loadCam(args, id, cam_info, resolution_scale):" in content:
        # Insert after function definition, before return
        old_pattern = "def loadCam(args, id, cam_info, resolution_scale):\n"
        # Find what comes after the function def
        idx = content.find(old_pattern)
        if idx != -1:
            idx += len(old_pattern)
            # Find the return statement
            return_idx = content.find("return Camera(", idx)
            if return_idx != -1:
                # Insert alpha code before return
                content = content[:return_idx] + alpha_code + content[return_idx:]

    # Replace gt_alpha_mask=None with gt_alpha_mask=gt_alpha_mask
    content = content.replace("gt_alpha_mask=None", "gt_alpha_mask=gt_alpha_mask")

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"[ALPHA] Patched: {file_path}")
    return True


def patch_cameras(file_path):
    """Patch cameras.py to composite onto white background and store alpha mask."""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    if PATCH_MARKER in content:
        print(f"[ALPHA] Already patched: {file_path}")
        return True

    # 1. Change alpha handling from multiply to white background compositing
    old_alpha = """        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask"""

    new_alpha = """        if gt_alpha_mask is not None:
            # [ALPHA PATCH] Composite onto white background (matches --white_background render)
            # Formula: RGB * alpha + white * (1 - alpha)
            self.original_image = self.original_image * gt_alpha_mask + (1.0 - gt_alpha_mask)"""

    if old_alpha in content:
        content = content.replace(old_alpha, new_alpha)
    else:
        print(f"[Warning] Could not find original alpha handling in {file_path}")

    # 2. Store gt_alpha_mask for loss masking
    if "self.mask = mask" in content and "self.gt_alpha_mask" not in content:
        content = content.replace(
            "self.mask = mask",
            """self.mask = mask
        # [ALPHA PATCH] Store alpha mask for loss masking during training
        self.gt_alpha_mask = gt_alpha_mask"""
        )

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"[ALPHA] Patched: {file_path}")
    return True


def patch_train(file_path):
    """Patch train.py to apply loss masking."""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    if PATCH_MARKER in content:
        print(f"[ALPHA] Already patched: {file_path}")
        return True

    if "Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])" not in content:
        print(f"[Error] Could not find L1 loss computation in {file_path}")
        return False

    # 1. Add alpha_masks list initialization
    old_init = """images = []
        gt_images = []"""
    new_init = """images = []
        gt_images = []
        alpha_masks = []  # [ALPHA PATCH] For loss masking"""

    content = content.replace(old_init, new_init)

    # 2. Collect alpha masks in render loop
    old_collect = "gt_images.append(gt_image.unsqueeze(0))"
    new_collect = """gt_images.append(gt_image.unsqueeze(0))
            # [ALPHA PATCH] Collect alpha masks for loss masking
            if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                alpha_masks.append(viewpoint_cam.gt_alpha_mask.cuda().unsqueeze(0))
            else:
                alpha_masks.append(None)"""

    content = content.replace(old_collect, new_collect)

    # 3. Apply mask to L1 loss
    old_loss = "Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])"
    new_loss = """# [ALPHA PATCH] Apply alpha mask to loss computation
        combined_mask = None
        valid_masks = [m for m in alpha_masks if m is not None]
        if valid_masks:
            combined_mask = torch.cat(valid_masks, 0)

        if combined_mask is not None:
            # Masked L1: only compute loss where alpha > 0.5
            mask_binary = (combined_mask > 0.5).float()
            masked_render = image_tensor * mask_binary
            masked_gt = gt_image_tensor[:,:3,:,:] * mask_binary
            valid_pixels = mask_binary.sum() + 1e-8
            Ll1 = torch.abs(masked_render - masked_gt).sum() / valid_pixels
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])"""

    content = content.replace(old_loss, new_loss)

    # 4. Apply mask to SSIM loss
    old_ssim = "ssim_loss = ssim(image_tensor,gt_image_tensor)"
    new_ssim = """# [ALPHA PATCH] Apply mask to SSIM
            if combined_mask is not None:
                mask_binary = (combined_mask > 0.5).float()
                ssim_loss = ssim(image_tensor * mask_binary, gt_image_tensor[:,:3,:,:] * mask_binary)
            else:
                ssim_loss = ssim(image_tensor, gt_image_tensor)"""

    content = content.replace(old_ssim, new_ssim)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"[ALPHA] Patched: {file_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_4dgs_alpha.py <path_to_4dgs>")
        print("")
        print("Example:")
        print("  python src/patch_4dgs_alpha.py external/4dgs")
        sys.exit(1)

    base_path = sys.argv[1]

    camera_utils_path = os.path.join(base_path, "utils", "camera_utils.py")
    cameras_path = os.path.join(base_path, "scene", "cameras.py")
    train_path = os.path.join(base_path, "train.py")

    success1 = patch_camera_utils(camera_utils_path)
    success2 = patch_cameras(cameras_path)
    success3 = patch_train(train_path)

    if success1 and success2 and success3:
        print("")
        print("[ALPHA] Successfully patched 4DGS for transparent PNG support!")
        print("[ALPHA] - Alpha channel extraction with .clone()")
        print("[ALPHA] - White background compositing")
        print("[ALPHA] - Loss masking (background excluded)")
        print("")
        print("[ALPHA] Train with: --white_background flag")

    sys.exit(0 if (success1 and success2 and success3) else 1)


if __name__ == "__main__":
    main()
