#!/usr/bin/env python
"""
Patches 4DGS gaussian_model.py to work without open3d.

open3d is only used for creating BasicPointCloud, which can be replaced
with a simple namedtuple. This avoids the need to install open3d which
causes server crashes on some environments.
"""
import sys
import os
import re

def patch_gaussian_model(file_path):
    """Patch gaussian_model.py to remove open3d dependency."""

    if not os.path.exists(file_path):
        print(f"[PATCH] Error: File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if "# PATCHED: open3d removed" in content:
        print(f"[PATCH] Already patched: {file_path}")
        return True

    # Replace open3d import with a mock BasicPointCloud
    old_import = "import open3d as o3d"
    new_import = """# PATCHED: open3d removed - using namedtuple instead
from collections import namedtuple
# Mock BasicPointCloud to avoid open3d dependency
BasicPointCloud = namedtuple('BasicPointCloud', ['points', 'colors', 'normals'])"""

    if old_import in content:
        content = content.replace(old_import, new_import)
        print(f"[PATCH] Replaced open3d import with namedtuple BasicPointCloud")
    else:
        print(f"[PATCH] Warning: Could not find 'import open3d as o3d' in {file_path}")
        return False

    # Remove any usage of o3d.geometry.PointCloud if present
    # Usually BasicPointCloud is defined like:
    # BasicPointCloud = namedtuple("BasicPointCloud", ["points", "colors", "normals"])
    # which is already what we want, so we just need to remove the import

    # Check if there's a BasicPointCloud definition we need to remove
    # (since we're providing our own)
    basic_pc_pattern = r'BasicPointCloud\s*=\s*namedtuple\s*\(\s*["\']BasicPointCloud["\']\s*,\s*\[[^\]]+\]\s*\)'
    if re.search(basic_pc_pattern, content):
        # There's already a namedtuple definition, our import will conflict
        # Remove our duplicate definition
        content = content.replace(new_import, """# PATCHED: open3d removed
from collections import namedtuple""")
        print(f"[PATCH] Kept existing BasicPointCloud namedtuple definition")

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"[PATCH] Successfully patched: {file_path}")
    return True


def patch_dataset_readers_open3d(file_path):
    """Patch dataset_readers.py to remove open3d dependency if present."""

    if not os.path.exists(file_path):
        print(f"[PATCH] File not found (skipping): {file_path}")
        return True

    with open(file_path, 'r') as f:
        content = f.read()

    modified = False

    # Check for open3d import
    if "import open3d" in content and "# PATCHED" not in content:
        content = re.sub(
            r'import open3d[^\n]*\n',
            '# PATCHED: open3d import removed\n',
            content
        )
        modified = True
        print(f"[PATCH] Removed open3d import from dataset_readers.py")

    if modified:
        with open(file_path, 'w') as f:
            f.write(content)

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_4dgs_open3d.py <4dgs_install_dir>")
        print("Example:")
        print("  python patch_4dgs_open3d.py external/4dgs")
        sys.exit(1)

    install_dir = sys.argv[1]

    # Patch gaussian_model.py
    gaussian_model_path = os.path.join(install_dir, "scene", "gaussian_model.py")
    success1 = patch_gaussian_model(gaussian_model_path)

    # Also check dataset_readers.py for open3d usage
    dataset_readers_path = os.path.join(install_dir, "scene", "dataset_readers.py")
    success2 = patch_dataset_readers_open3d(dataset_readers_path)

    if success1 and success2:
        print(f"\n[PATCH] All patches applied successfully!")
        print(f"[PATCH] 4DGS should now work without open3d.")
        sys.exit(0)
    else:
        print(f"\n[PATCH] Some patches failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
