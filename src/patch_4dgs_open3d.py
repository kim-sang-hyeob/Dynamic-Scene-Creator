import os
import sys

def patch_file(file_path):
    if not os.path.exists(file_path):
        print(f"[Patch] Skip: {file_path} not found.")
        return False
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    patched = False
    for line in lines:
        if "import open3d as o3d" in line:
            new_lines.append(f"# {line}")  # Comment out the import
            patched = True
        else:
            new_lines.append(line)
    
    if patched:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print(f"[Patch] Successfully patched {file_path}")
    else:
        print(f"[Patch] No open3d import found in {file_path}")
    return patched

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_4dgs_open3d.py <install_dir>")
        sys.exit(1)
    
    install_dir = sys.argv[1]
    
    # Target files known to import open3d
    targets = [
        os.path.join(install_dir, "scene", "gaussian_model.py"),
        os.path.join(install_dir, "scripts", "downsample_point.py")
    ]
    
    for target in targets:
        patch_file(target)
