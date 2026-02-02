#!/usr/bin/env python
"""
Remove unused open3d import from gaussian_model.py.
open3d is imported but never used, and causes installation issues.
"""

import sys
import os

def patch_gaussian_model(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    if "import open3d" not in content:
        print(f"[OPEN3D] No open3d import found (already patched or not needed)")
        return True

    # Remove the open3d import line
    patched_content = content.replace("import open3d as o3d\n", "# import open3d as o3d  # Removed - not used\n")

    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"[OPEN3D] Removed unused open3d import from: {file_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_4dgs_open3d.py <path_to_4dgs>")
        sys.exit(1)

    base_path = sys.argv[1]
    gaussian_model_path = os.path.join(base_path, "scene", "gaussian_model.py")

    success = patch_gaussian_model(gaussian_model_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
