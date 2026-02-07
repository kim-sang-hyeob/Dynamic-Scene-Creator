#!/usr/bin/env python
"""
Unified Format Manager for 3DGS/4DGS file conversions and merging.

Commands:
    convert     Convert between different 3DGS/4DGS formats
    merge       Merge background and object splat files
    list        List supported formats

Supported Conversions:
    PLY      → .splat   (3DGS PLY to web viewer format)
    SPZ      → .splat   (Niantic compressed to web viewer format)
    HexPlane → .splatv  (HexPlane-based 4DGS to animated format)
    MLP      → .splatv  (MLP-based 4DGS to animated format)

Usage:
    python format_manage.py convert input.spz -o output.splat
    python format_manage.py convert input.ply -o output.splat
    python format_manage.py convert --type hexplane --model-path <path> -o output.splatv
    python format_manage.py convert --type mlp --model-dir <path> -o output.splatv
    python format_manage.py merge background.splat object.splatv -o merged.splatv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_file_format(filepath):
    """Detect file format from extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return {'.ply': 'ply', '.spz': 'spz', '.splat': 'splat', '.splatv': 'splatv'}.get(ext)


def convert_command(args):
    """Handle convert subcommand."""
    if args.type == 'hexplane':
        convert_hexplane(args)
    elif args.type == 'mlp':
        convert_mlp(args)
    elif args.input:
        input_format = get_file_format(args.input)
        if input_format == 'spz':
            convert_spz(args.input, args.output, args.slow)
        elif input_format == 'ply':
            convert_ply(args.input, args.output)
        else:
            print(f"Error: Unsupported input format: {input_format}")
            print("Supported: .ply, .spz, or use --type hexplane/mlp")
            sys.exit(1)
    else:
        print("Error: Specify input file or --type")
        sys.exit(1)


def convert_spz(input_path, output_path, slow=False):
    """Convert SPZ to splat."""
    from convert_spz_to_splat import process_spz_to_splat_vectorized, process_spz_to_splat, save_splat_file

    output_path = output_path or input_path.rsplit('.', 1)[0] + '.splat'
    print(f"Converting: {input_path} → {output_path}")

    splat_data = process_spz_to_splat(input_path) if slow else process_spz_to_splat_vectorized(input_path)
    save_splat_file(splat_data, output_path)

    print(f"  Gaussians: {len(splat_data) // 32:,}")
    print(f"  Size: {len(splat_data) / 1024 / 1024:.2f} MB")


def convert_ply(input_path, output_path):
    """Convert PLY to splat."""
    from convert_ply_to_splat import process_ply_to_splat, save_splat_file

    output_path = output_path or input_path.rsplit('.', 1)[0] + '.splat'
    print(f"Converting: {input_path} → {output_path}")

    splat_data = process_ply_to_splat(input_path)
    save_splat_file(splat_data, output_path)

    print(f"  Gaussians: {len(splat_data) // 32:,}")
    print(f"  Size: {len(splat_data) / 1024 / 1024:.2f} MB")


def convert_hexplane(args):
    """Convert HexPlane 4DGS to splatv."""
    if not args.model_path:
        print("Error: --model-path required for HexPlane")
        sys.exit(1)
    if not args.output:
        print("Error: -o/--output required")
        sys.exit(1)

    print(f"Converting HexPlane: {args.model_path} → {args.output}")
    print("Note: Requires PYTHONPATH=external/4dgs")

    # Import and run the existing converter's main logic
    import subprocess
    cmd = [
        sys.executable, 'convert_hexplane_to_splatv.py',
        '--model_path', args.model_path,
        '--output', args.output,
        '--iteration', str(args.iteration),
        '--num_samples', str(args.num_samples),
    ]
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def convert_mlp(args):
    """Convert MLP 4DGS to splatv."""
    if not args.model_dir:
        print("Error: --model-dir required for MLP")
        sys.exit(1)
    if not args.output:
        print("Error: -o/--output required")
        sys.exit(1)

    print(f"Converting MLP: {args.model_dir} → {args.output}")

    from convert_mlp_to_splatv import convert_mlp_to_splatv
    convert_mlp_to_splatv(
        args.model_dir,
        args.output,
        num_samples=args.num_samples,
        iteration=args.mlp_iteration
    )


def merge_command(args):
    """Handle merge subcommand."""
    from merge_splat_files import (
        read_splat_file, read_splatv_file,
        splat_to_texdata, offset_splatv_positions,
        merge_texdata, write_splatv_file
    )
    from pathlib import Path

    print("=" * 50)
    print("Merge Splat Files")
    print("=" * 50)

    bg_path, obj_path = Path(args.background), Path(args.object)

    # Read background
    if bg_path.suffix.lower() == '.splat':
        bg_data = read_splat_file(args.background)
        bg_tex, _, _ = splat_to_texdata(bg_data, tuple(args.bg_offset), args.bg_scale, tuple(args.bg_rotate))
        bg_count = bg_data['count']
    else:
        bg_splatv = read_splatv_file(args.background)
        bg_tex = offset_splatv_positions(bg_splatv, tuple(args.bg_offset), args.bg_scale)
        bg_count = bg_splatv['count']

    # Read object
    cameras = None
    if obj_path.suffix.lower() == '.splatv':
        obj_splatv = read_splatv_file(args.object)
        obj_tex = offset_splatv_positions(obj_splatv, tuple(args.offset), args.scale)
        obj_count = obj_splatv['count']
        cameras = obj_splatv['metadata'][0].get('cameras')
    else:
        obj_data = read_splat_file(args.object)
        obj_tex, _, _ = splat_to_texdata(obj_data, tuple(args.offset), args.scale)
        obj_count = obj_data['count']

    print(f"Background: {bg_count:,} gaussians")
    print(f"Object: {obj_count:,} gaussians (offset={args.offset}, scale={args.scale})")

    merged, tw, th, total = merge_texdata(bg_tex, bg_count, obj_tex, obj_count)
    write_splatv_file(args.output, merged, tw, th, cameras)

    print(f"Total: {total:,} gaussians")
    print("Done!")


def list_command(args):
    """List supported formats."""
    print("""
Supported Formats
=================

Input:                      Output:
  .ply   (3DGS PLY)           .splat  (static web viewer)
  .spz   (Niantic compressed) .splatv (animated web viewer)
  .splat (web viewer)
  .splatv (animated)

Conversions
===========
  .ply  → .splat     python format_manage.py convert input.ply -o out.splat
  .spz  → .splat     python format_manage.py convert input.spz -o out.splat
  HexPlane → .splatv python format_manage.py convert --type hexplane --model-path <path> -o out.splatv
  MLP    → .splatv   python format_manage.py convert --type mlp --model-dir <path> -o out.splatv

Merge
=====
  python format_manage.py merge bg.splat obj.splatv -o merged.splatv
  python format_manage.py merge bg.splat obj.splatv -o merged.splatv --offset 0 1.5 -2 --scale 0.5
""")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Format Manager for 3DGS/4DGS",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest='command')

    # convert
    cp = sub.add_parser('convert', help='Convert formats')
    cp.add_argument('input', nargs='?', help='Input file (.ply, .spz)')
    cp.add_argument('-o', '--output', help='Output file')
    cp.add_argument('--type', choices=['hexplane', 'mlp'], help='4DGS model type')
    cp.add_argument('--model-path', help='HexPlane model path')
    cp.add_argument('--model-dir', help='MLP model directory')
    cp.add_argument('--iteration', type=int, default=-1, help='Checkpoint iteration')
    cp.add_argument('--mlp-iteration', type=int, help='MLP iteration')
    cp.add_argument('--num-samples', type=int, default=20, help='Motion samples')
    cp.add_argument('--slow', action='store_true', help='Memory-efficient mode')

    # merge
    mp = sub.add_parser('merge', help='Merge splat files')
    mp.add_argument('background', help='Background .splat/.splatv')
    mp.add_argument('object', help='Object .splat/.splatv')
    mp.add_argument('-o', '--output', required=True, help='Output .splatv')
    mp.add_argument('--offset', nargs=3, type=float, default=[0, 0, 0], metavar=('X', 'Y', 'Z'))
    mp.add_argument('--scale', type=float, default=1.0)
    mp.add_argument('--bg-offset', nargs=3, type=float, default=[0, 0, 0], metavar=('X', 'Y', 'Z'))
    mp.add_argument('--bg-scale', type=float, default=1.0)
    mp.add_argument('--bg-rotate', nargs=3, type=float, default=[0, 0, 0], metavar=('X', 'Y', 'Z'))

    # list
    sub.add_parser('list', help='List supported formats')

    args = parser.parse_args()

    if args.command == 'convert':
        convert_command(args)
    elif args.command == 'merge':
        merge_command(args)
    elif args.command == 'list':
        list_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
