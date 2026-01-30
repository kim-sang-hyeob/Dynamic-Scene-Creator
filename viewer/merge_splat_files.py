"""
Merge .splat and .splatv files into a single .splatv file.

This script combines:
- Background scene (.splat) - static 3D Gaussian Splatting
- Dynamic object (.splatv) - 4D Gaussian Splatting with motion

The 4DGS object position can be offset to place it in the scene.

Usage:
    python merge_splat_files.py background.splat object.splatv -o merged.splatv
    python merge_splat_files.py background.splat object.splatv -o merged.splatv --offset 0 1 0
    python merge_splat_files.py background.splat object.splatv -o merged.splatv --offset 0 1 0 --scale 0.5
"""

import numpy as np
import struct
import json
import argparse
from pathlib import Path


def pack_half2x16(x, y):
    """Pack two float16 values into a uint32."""
    x_half = np.float16(x)
    y_half = np.float16(y)
    x_bits = x_half.view(np.uint16)
    y_bits = y_half.view(np.uint16)
    return np.uint32(x_bits) | (np.uint32(y_bits) << 16)


def read_splat_file(filepath):
    """
    Read .splat file (antimatter15 format: 32 bytes per gaussian).
    
    Format per gaussian:
    - position: float32[3] (12 bytes)
    - scale: float32[3] (12 bytes)
    - color: uint8[4] (4 bytes - RGBA)
    - rotation: uint8[4] (4 bytes - normalized quaternion)
    
    Returns: dict with gaussians data
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    num_gaussians = len(data) // 32
    print(f"Reading .splat: {num_gaussians:,} gaussians from {filepath}")
    
    f_buffer = np.frombuffer(data, dtype=np.float32).reshape(-1, 8)
    u8_buffer = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)
    
    gaussians = {
        'positions': f_buffer[:, 0:3].copy(),
        'scales': f_buffer[:, 3:6].copy(),
        'colors': u8_buffer[:, 24:28].copy(),  # RGBA
        'rotations_u8': u8_buffer[:, 28:32].copy(),  # Normalized quaternion as uint8
        'is_dynamic': False,
        'count': num_gaussians
    }
    
    return gaussians


def read_splatv_file(filepath):
    """
    Read .splatv file.
    
    Format:
    - Magic: uint32 (0x674b)
    - JSON length: uint32
    - JSON metadata: bytes
    - Texture data: uint32[]
    
    Returns: dict with gaussians data and metadata
    """
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x674b:
            raise ValueError(f"Invalid splatv magic: {magic:x}")
        
        json_length = struct.unpack('<I', f.read(4))[0]
        json_bytes = f.read(json_length)
        metadata = json.loads(json_bytes.decode('utf-8'))
        
        texdata = np.frombuffer(f.read(), dtype=np.uint32)
    
    chunk = metadata[0]
    texwidth = chunk['texwidth']
    texheight = chunk['texheight']
    
    # Each gaussian uses 16 uint32 values (64 bytes in texture)
    num_gaussians = (texwidth * texheight) // 4
    print(f"Reading .splatv: {num_gaussians:,} gaussians from {filepath}")
    
    return {
        'texdata': texdata,
        'texwidth': texwidth,
        'texheight': texheight,
        'metadata': metadata,
        'is_dynamic': True,
        'count': num_gaussians
    }


def splat_to_texdata(gaussians, offset=(0, 0, 0), scale=1.0):
    """
    Convert .splat data to splatv texture format.
    
    Args:
        gaussians: dict from read_splat_file
        offset: (x, y, z) position offset
        scale: scale factor for positions
    
    Returns: (texdata, texwidth, texheight)
    """
    num = gaussians['count']
    texwidth = 1024 * 4
    texheight = int(np.ceil((4 * num) / texwidth))
    
    texdata = np.zeros((texwidth * texheight * 4,), dtype=np.uint32)
    texdata_f = texdata.view(np.float32)
    texdata_c = texdata.view(np.uint8)
    
    positions = gaussians['positions'] * scale + np.array(offset)
    scales = gaussians['scales'] * scale
    colors = gaussians['colors']
    rots_u8 = gaussians['rotations_u8']
    
    for j in range(num):
        # Position (float32[3])
        texdata_f[16 * j + 0] = positions[j, 0]
        texdata_f[16 * j + 1] = positions[j, 1]
        texdata_f[16 * j + 2] = positions[j, 2]
        
        # Rotation from uint8 to half
        rot_0 = (rots_u8[j, 0] - 128) / 128
        rot_1 = (rots_u8[j, 1] - 128) / 128
        rot_2 = (rots_u8[j, 2] - 128) / 128
        rot_3 = (rots_u8[j, 3] - 128) / 128
        
        texdata[16 * j + 3] = pack_half2x16(rot_0, rot_1)
        texdata[16 * j + 4] = pack_half2x16(rot_2, rot_3)
        
        # Scale
        texdata[16 * j + 5] = pack_half2x16(scales[j, 0], scales[j, 1])
        texdata[16 * j + 6] = pack_half2x16(scales[j, 2], 0)
        
        # RGBA
        texdata_c[4 * (16 * j + 7) + 0] = colors[j, 0]
        texdata_c[4 * (16 * j + 7) + 1] = colors[j, 1]
        texdata_c[4 * (16 * j + 7) + 2] = colors[j, 2]
        texdata_c[4 * (16 * j + 7) + 3] = colors[j, 3]
        
        # Motion data - all zeros for static splat
        texdata[16 * j + 8:16 * j + 15] = 0
        texdata[16 * j + 15] = pack_half2x16(0.5, 1.0)  # trbf_center, trbf_scale
    
    return texdata, texwidth, texheight


def offset_splatv_positions(splatv_data, offset=(0, 0, 0), scale=1.0):
    """
    Apply position offset and scale to splatv texture data.
    
    Args:
        splatv_data: dict from read_splatv_file
        offset: (x, y, z) position offset
        scale: scale factor
    
    Returns: modified texdata
    """
    texdata = splatv_data['texdata'].copy()
    texdata_f = texdata.view(np.float32)
    
    num = splatv_data['count']
    for j in range(num):
        # Apply scale and offset to position
        texdata_f[16 * j + 0] = texdata_f[16 * j + 0] * scale + offset[0]
        texdata_f[16 * j + 1] = texdata_f[16 * j + 1] * scale + offset[1]
        texdata_f[16 * j + 2] = texdata_f[16 * j + 2] * scale + offset[2]
    
    return texdata


def merge_texdata(texdata1, count1, texdata2, count2):
    """
    Merge two texture data arrays.
    
    Returns: (merged_texdata, texwidth, texheight, total_count)
    """
    total_count = count1 + count2
    texwidth = 1024 * 4
    texheight = int(np.ceil((4 * total_count) / texwidth))
    
    merged = np.zeros((texwidth * texheight * 4,), dtype=np.uint32)
    
    # Copy first dataset
    for j in range(count1):
        merged[16 * j:16 * (j + 1)] = texdata1[16 * j:16 * (j + 1)]
    
    # Copy second dataset
    for j in range(count2):
        merged[16 * (count1 + j):16 * (count1 + j + 1)] = texdata2[16 * j:16 * (j + 1)]
    
    return merged, texwidth, texheight, total_count


def write_splatv_file(filepath, texdata, texwidth, texheight, cameras=None):
    """Write merged data to .splatv file."""
    if cameras is None:
        cameras = [{
            "id": 0,
            "img_name": "00001",
            "width": 800,
            "height": 800,
            "position": [0, 0, -3],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "fy": 800,
            "fx": 800,
        }]
    
    metadata = [{
        "type": "splat",
        "size": texdata.nbytes,
        "texwidth": texwidth,
        "texheight": texheight,
        "cameras": cameras
    }]
    
    json_bytes = json.dumps(metadata).encode('utf-8')
    
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', 0x674b))  # Magic
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(json_bytes)
        f.write(texdata.tobytes())
    
    print(f"Written: {filepath} ({texdata.nbytes / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge .splat and .splatv files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple merge
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv
    
    # With position offset (move object up by 1 unit)
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv --offset 0 1 0
    
    # With scale (make object half size)
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv --scale 0.5
    
    # Combined offset and scale
    python merge_splat_files.py scene.splat object.splatv -o merged.splatv --offset 1 2 0 --scale 0.3
        """
    )
    
    parser.add_argument("background", help="Background .splat file (static scene)")
    parser.add_argument("object", help="Object .splatv file (dynamic 4DGS)")
    parser.add_argument("-o", "--output", required=True, help="Output .splatv file")
    parser.add_argument("--offset", nargs=3, type=float, default=[0, 0, 0],
                        metavar=('X', 'Y', 'Z'),
                        help="Position offset for the 4DGS object (default: 0 0 0)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for the 4DGS object (default: 1.0)")
    parser.add_argument("--bg-offset", nargs=3, type=float, default=[0, 0, 0],
                        metavar=('X', 'Y', 'Z'),
                        help="Position offset for background (default: 0 0 0)")
    parser.add_argument("--bg-scale", type=float, default=1.0,
                        help="Scale factor for background (default: 1.0)")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Merging Splat Files")
    print("=" * 50)
    
    # Read background
    bg_file = Path(args.background)
    if bg_file.suffix.lower() == '.splat':
        bg_data = read_splat_file(args.background)
        bg_texdata, _, _ = splat_to_texdata(
            bg_data, 
            offset=tuple(args.bg_offset),
            scale=args.bg_scale
        )
        bg_count = bg_data['count']
    elif bg_file.suffix.lower() == '.splatv':
        bg_splatv = read_splatv_file(args.background)
        bg_texdata = offset_splatv_positions(
            bg_splatv,
            offset=tuple(args.bg_offset),
            scale=args.bg_scale
        )
        bg_count = bg_splatv['count']
    else:
        raise ValueError(f"Unsupported background format: {bg_file.suffix}")
    
    # Read object
    obj_file = Path(args.object)
    if obj_file.suffix.lower() == '.splatv':
        obj_splatv = read_splatv_file(args.object)
        obj_texdata = offset_splatv_positions(
            obj_splatv,
            offset=tuple(args.offset),
            scale=args.scale
        )
        obj_count = obj_splatv['count']
        cameras = obj_splatv['metadata'][0].get('cameras')
    elif obj_file.suffix.lower() == '.splat':
        obj_data = read_splat_file(args.object)
        obj_texdata, _, _ = splat_to_texdata(
            obj_data,
            offset=tuple(args.offset),
            scale=args.scale
        )
        obj_count = obj_data['count']
        cameras = None
    else:
        raise ValueError(f"Unsupported object format: {obj_file.suffix}")
    
    print(f"\nBackground: {bg_count:,} gaussians")
    print(f"Object: {obj_count:,} gaussians (offset: {args.offset}, scale: {args.scale})")
    
    # Merge
    merged, texwidth, texheight, total = merge_texdata(
        bg_texdata, bg_count,
        obj_texdata, obj_count
    )
    
    print(f"Total: {total:,} gaussians")
    print(f"Texture: {texwidth}x{texheight}")
    
    # Write output
    write_splatv_file(args.output, merged, texwidth, texheight, cameras)
    print("\nDone!")


if __name__ == "__main__":
    main()
