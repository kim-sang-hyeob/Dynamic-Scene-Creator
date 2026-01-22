import pycolmap
import os
import sys

def debug_pycolmap():
    print(f"pycolmap version: {getattr(pycolmap, '__version__', 'unknown')}")
    
    # Try to load a reconstruction if it exists to check real objects
    path = "/data/ephemeral/home/3dgs_project/data/seoul_reconstruction_vggt_pro/sparse/0"
    if os.path.exists(os.path.join(path, "cameras.bin")):
        recon = pycolmap.Reconstruction(path)
        if len(recon.images) > 0:
            img_id = next(iter(recon.images))
            img = recon.images[img_id]
            print(f"Image object type: {type(img)}")
            print(f"Attributes: {dir(img)}")
            if hasattr(img, 'qvec'):
                print(f"Found qvec: {img.qvec}")
            elif hasattr(img, 'cam_from_world'):
                print(f"Found cam_from_world: {img.cam_from_world}")
                print(f"cam_from_world attributes: {dir(img.cam_from_world)}")
        else:
            print("No images in reconstruction.")
    else:
        print(f"Reconstruction not found at {path}")

if __name__ == "__main__":
    debug_pycolmap()
