
import struct
import json
import numpy as np
import argparse
from pathlib import Path

def read_splat_file(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    num_gaussians = len(data) // 32
    f_buffer = np.frombuffer(data, dtype=np.float32).reshape(-1, 8)
    return f_buffer[:, 0:3]

def read_splatv_file(filepath):
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        json_length = struct.unpack('<I', f.read(4))[0]
        json_bytes = f.read(json_length)
        metadata = json.loads(json_bytes.decode('utf-8'))
        texdata = np.frombuffer(f.read(), dtype=np.uint32)
    
    texwidth = metadata[0]['texwidth']
    texheight = metadata[0]['texheight']
    num_gaussians = (texwidth * texheight) // 4
    
    texdata_f = texdata.view(np.float32)
    positions = np.zeros((num_gaussians, 3), dtype=np.float32)
    
    for j in range(num_gaussians):
        positions[j, 0] = texdata_f[16 * j + 0]
        positions[j, 1] = texdata_f[16 * j + 1]
        positions[j, 2] = texdata_f[16 * j + 2]
    
    return positions

def create_bbox_splat(min_pos, max_pos, num_points_per_edge=50):
    """Generate splat data for a bounding box wireframe."""
    # 8 Corners
    corners = [
        [min_pos[0], min_pos[1], min_pos[2]],
        [max_pos[0], min_pos[1], min_pos[2]],
        [min_pos[0], max_pos[1], min_pos[2]],
        [max_pos[0], max_pos[1], min_pos[2]],
        [min_pos[0], min_pos[1], max_pos[2]],
        [max_pos[0], min_pos[1], max_pos[2]],
        [min_pos[0], max_pos[1], max_pos[2]],
        [max_pos[0], max_pos[1], max_pos[2]],
    ]
    
    # 12 Edges (indices of corners)
    edges = [
        (0, 1), (0, 2), (0, 4), # From 0
        (7, 3), (7, 6), (7, 5), # From 7
        (1, 3), (1, 5),         # From 1
        (2, 3), (2, 6),         # From 2
        (4, 5), (4, 6)          # From 4
    ]
    
    positions = []
    
    # Generate points along edges
    for start_idx, end_idx in edges:
        start = np.array(corners[start_idx])
        end = np.array(corners[end_idx])
        
        for k in range(num_points_per_edge + 1):
            t = k / num_points_per_edge
            pos = start * (1 - t) + end * t
            positions.append(pos)
            
    positions = np.array(positions, dtype=np.float32)
    
    # Create simple splat data
    count = len(positions)
    print(f"Generating BBox Splat: {count} points")
    
    # Buffer structure: 32 bytes per splat
    # pos(3f), scale(3f), color(4u8), rot(4u8)
    buffer = bytearray(count * 32)
    
    # Color: Red by default (255, 0, 0, 255)
    color_r, color_g, color_b, color_a = 255, 0, 0, 255
    
    # Scale: Larger size for visibility
    scale_val = 0.1
    
    # Rotation: Identity (0,0,0,128) -> mapped to approx 0,0,0,1 if standard mapping used
    # Or just 128 for w?
    # read_splat_file: (val - 128) / 128. So 128 maps to 0.
    # We want identity quaternion (0,0,0,1).
    # so x,y,z = 128 (0), w = 255 (approx 1). 
    rot_x, rot_y, rot_z, rot_w = 128, 128, 128, 255
    
    for i in range(count):
        offset = i * 32
        
        # Position
        struct.pack_into('<fff', buffer, offset, positions[i, 0], positions[i, 1], positions[i, 2])
        
        # Scale
        struct.pack_into('<fff', buffer, offset + 12, scale_val, scale_val, scale_val)
        
        # Color (RGBA)
        struct.pack_into('BBBB', buffer, offset + 24, color_r, color_g, color_b, color_a)
        
        # Rotation
        struct.pack_into('BBBB', buffer, offset + 28, rot_x, rot_y, rot_z, rot_w)
        
    return buffer

def create_axes_splat(origin, size, num_points=100):
    """Generate splat data for XYZ axes."""
    # Axes endpoints relative to origin
    # X: (size, 0, 0), Y: (0, size, 0), Z: (0, 0, size)
    
    positions = []
    colors = [] # Store (R, G, B, A)
    
    # X Axis (Red)
    for k in range(num_points + 1):
        t = k / num_points
        pos = origin + np.array([size * t, 0, 0])
        positions.append(pos)
        colors.append((255, 0, 0, 255))
        
    # Y Axis (Green)
    for k in range(num_points + 1):
        t = k / num_points
        pos = origin + np.array([0, size * t, 0])
        positions.append(pos)
        colors.append((0, 255, 0, 255))
        
    # Z Axis (Blue)
    for k in range(num_points + 1):
        t = k / num_points
        pos = origin + np.array([0, 0, size * t])
        positions.append(pos)
        colors.append((0, 0, 255, 255))
        
    positions = np.array(positions, dtype=np.float32)
    count = len(positions)
    print(f"Generating Axes Splat: {count} points (Size: {size:.2f})")
    
    buffer = bytearray(count * 32)
    
    # Scale: Slightly thicker than bbox
    scale_val = 0.15
    # Rotation: Identity
    rot_x, rot_y, rot_z, rot_w = 128, 128, 128, 255
    
    for i in range(count):
        offset = i * 32
        
        # Position
        struct.pack_into('<fff', buffer, offset, positions[i, 0], positions[i, 1], positions[i, 2])
        
        # Scale
        struct.pack_into('<fff', buffer, offset + 12, scale_val, scale_val, scale_val)
        
        # Color (RGBA)
        c = colors[i]
        struct.pack_into('BBBB', buffer, offset + 24, c[0], c[1], c[2], c[3])
        
        # Rotation
        struct.pack_into('BBBB', buffer, offset + 28, rot_x, rot_y, rot_z, rot_w)
        
    return buffer

def main():
    parser = argparse.ArgumentParser(description="Analyze coordinates for debugging")
    parser.add_argument("--bg", required=True, help="Background .splat file")
    parser.add_argument("--obj", help="Object .splatv file")
    parser.add_argument("--lumina", help="lumina_path.json file (optional)")
    parser.add_argument("--export-bbox", nargs='?', const="bbox.splat", help="Export bounding box visualization to file (default: bbox.splat)")
    parser.add_argument("--export-axes", nargs='?', const="axes.splat", help="Export coordinate axes visualization to file (default: axes.splat)")
    parser.add_argument("--export-obj-axes", nargs='?', const="obj_axes.splat", help="Export object coordinate axes visualization to file (default: obj_axes.splat)")
    args = parser.parse_args()

    print("="*50)
    print("COORDINATE ANALYSIS")
    print("="*50)

    # 1. Analyze Background
    if args.bg.endswith('.splat'):
        bg_pos = read_splat_file(args.bg)
    else:
        print("Background must be .splat for this test script")
        return

    bg_min = bg_pos.min(axis=0)
    bg_max = bg_pos.max(axis=0)
    bg_center = (bg_min + bg_max) / 2
    bg_dims = bg_max - bg_min
    max_dim = np.max(bg_dims)
    
    print(f"BACKGROUND ({args.bg})")
    print(f"  Count: {len(bg_pos):,}")
    print(f"  Bounds Min: {bg_min}")
    print(f"  Bounds Max: {bg_max}")
    print(f"  Center:     {bg_center}")
    print(f"  Dims:       {bg_dims}")
    print(f"  Sample (first 5):")
    print(bg_pos[:5])
    print("-" * 30)

    if args.export_bbox:
        bbox_data = create_bbox_splat(bg_min, bg_max)
        with open(args.export_bbox, 'wb') as f:
            f.write(bbox_data)
        print(f"Exported Bounding Box to: {args.export_bbox}")
    
    if args.export_axes:
        # Draw axes starting from center, length = half of max dimension
        # Or start from (0,0,0) if it's within bounds?
        # Usually seeing axes at (0,0,0) defines the world origin.
        # But if the map is offset far away, (0,0,0) might be invisible.
        # Let's draw at BG Center to show LOCAL orientation, 
        # AND maybe specific markers for (0,0,0) if requested?
        # User asked "How the map is tilted in XYZ frame".
        # So visualization at the Center of the map is most useful to see how the map aligns with axes.
        axes_data = create_axes_splat(bg_center, max_dim * 0.5)
        with open(args.export_axes, 'wb') as f:
            f.write(axes_data)
        print(f"Exported Axes to: {args.export_axes} (Origin: Map Center)")
        print("  Red: X, Green: Y, Blue: Z")

    # 2. Analyze Lumina Path
    if args.lumina:
        try:
            with open(args.lumina, 'r') as f:
                lumina_data = json.load(f)
            
            if 'controlPoints' in lumina_data and len(lumina_data['controlPoints']) > 0:
                target_pos = np.array(lumina_data['controlPoints'][0]['position'])
                print(f"LUMINA PATH ({args.lumina})")
                print(f"  Target Position: {target_pos}")
                
                # Check if target is inside background bounds
                inside = np.all(target_pos >= bg_min) and np.all(target_pos <= bg_max)
                print(f"  Is inside BG bounds? {inside}")
                if not inside:
                    print(f"  WARNING: Picked point is OUTSIDE the background bounding box!")
            else:
                print("No control points in lumina file!")
        except Exception as e:
            print(f"Error reading lumina path: {e}")
        print("-" * 30)

    # 3. Analyze Object
    if args.obj:
        obj_pos = read_splatv_file(args.obj)
        obj_min = obj_pos.min(axis=0)
        obj_max = obj_pos.max(axis=0)
        obj_center = (obj_min + obj_max) / 2
        obj_dims = obj_max - obj_min
        obj_max_dim = np.max(obj_dims)
        
        print(f"OBJECT ({args.obj})")
        print(f"  Count: {len(obj_pos):,}")
        print(f"  Bounds Min: {obj_min}")
        print(f"  Bounds Max: {obj_max}")
        print(f"  Center:     {obj_center}")
        
        if args.export_obj_axes:
            axes_data = create_axes_splat(obj_center, obj_max_dim * 0.8) # 80% of size
            with open(args.export_obj_axes, 'wb') as f:
                f.write(axes_data)
            print(f"Exported Object Axes to: {args.export_obj_axes} (Origin: Object Center)")

        # Predict Result
        if args.lumina and 'target_pos' in locals():
            print("-" * 30)
            print("PREDICTION")
            final_pos = obj_center + target_pos
            print(f"  If merged, object center will move to: {final_pos} (Offset: {target_pos})")
            print(f"  Object vertical range will be: {obj_min[1] + target_pos[1]} to {obj_max[1] + target_pos[1]}")
            print(f"  Picked Ground Height (Y): {target_pos[1]}")

if __name__ == "__main__":
    main()
