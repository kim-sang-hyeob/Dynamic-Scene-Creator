#!/usr/bin/env python
"""
Patch 4DGS dataset_readers.py for SfM-free operation.

When no SfM points exist (empty points3D.txt), this patch generates
random initial points so training can proceed without COLMAP.

Usage:
    python src/patch_4dgs_sfm_free.py <path_to_dataset_readers.py>

Example:
    python src/patch_4dgs_sfm_free.py external/4dgs/scene/dataset_readers.py
"""

import sys
import os
import re

PATCH_MARKER = "[SfM-FREE PATCH]"

# Patch code template - {indent} will be replaced with actual indentation
PATCH_CODE_TEMPLATE = '''
{indent}# [SfM-FREE PATCH] Generate random initial points if no SfM points exist
{indent}try:
{indent}    _xyz_exists = xyz is not None and len(xyz) > 0
{indent}except (NameError, UnboundLocalError):
{indent}    _xyz_exists = False
{indent}if not _xyz_exists:
{indent}    import numpy as np
{indent}    print("[SfM-FREE] No SfM points found. Generating random initial points...")
{indent}    np.random.seed(42)
{indent}    num_pts = 1000
{indent}    xyz = np.random.randn(num_pts, 3).astype(np.float32) * 0.5
{indent}    rgb = np.random.randint(128, 255, size=(num_pts, 3)).astype(np.uint8)
{indent}    print(f"[SfM-FREE] Generated {{num_pts}} random initial points")
'''


def patch_dataset_readers(file_path):
    """Patch dataset_readers.py for SfM-free operation."""

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if PATCH_MARKER in content:
        print(f"[SfM-FREE] Already patched: {file_path}")
        return True

    # Find the storePly call in readColmapSceneInfo function
    # We need to insert the patch before storePly(ply_path, xyz, rgb)
    pattern = r'(\s+)(storePly\(ply_path, xyz, rgb\))'
    match = re.search(pattern, content)

    if not match:
        print(f"[Error] Could not find 'storePly(ply_path, xyz, rgb)' in {file_path}")
        print("       The 4DGS code structure may have changed.")
        return False

    indent = match.group(1)
    store_ply_call = match.group(2)

    # Remove leading newline from indent (keep only spaces/tabs)
    base_indent = indent.lstrip('\n')

    # Generate patch code with correct indentation
    patch_code = PATCH_CODE_TEMPLATE.format(indent=base_indent)

    # Insert the patch code before storePly
    patched_content = content.replace(
        f"{indent}{store_ply_call}",
        f"{patch_code}\n{indent}{store_ply_call}"
    )

    # Write the patched file
    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"[SfM-FREE] Successfully patched: {file_path}")
    print(f"[SfM-FREE] 4DGS will now generate random initial points when points3D is empty.")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_4dgs_sfm_free.py <path_to_dataset_readers.py>")
        print("")
        print("Example:")
        print("  python src/patch_4dgs_sfm_free.py external/4dgs/scene/dataset_readers.py")
        sys.exit(1)

    file_path = sys.argv[1]
    success = patch_dataset_readers(file_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
