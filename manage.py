#!/usr/bin/env python
"""
4DGS Project Manager - Colmap-free Unity→4DGS Pipeline

Commands:
    setup           Setup 4DGS environment
    process-unity   Sync Unity JSON + Video → 4DGS dataset (bypass SfM)
    train           Train 4DGS model
    render          Render with camera rotation
    visualize       Visualize in Rerun
    clean-model     Remove floaters from PLY
    list-models     Show available models
"""
import argparse
import sys
import yaml
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.setup import SetupManager
from src.dataset import DatasetManager
from src.runner import Runner
from src.model_registry import ModelRegistry

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    registry = ModelRegistry()
    available_models = registry.list_models()

    # Default to 4dgs for this project
    default_model = "4dgs" if "4dgs" in available_models else (available_models[0] if available_models else None)

    parser = argparse.ArgumentParser(
        description="4DGS Project Manager - Colmap-free Unity→4DGS Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup 4DGS environment
  python manage.py setup --model 4dgs

  # Process Unity data (bypass SfM!)
  python manage.py process-unity data/black_cat/output_cat.mp4 data/black_cat/full_data.json data/black_cat/original_catvideo.mp4 --output black_cat

  # Train 4DGS
  python manage.py train data/black_cat

  # Render with 45 degree camera rotation
  CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py -m output/4dgs/black_cat --skip_train --skip_test

  # Visualize in Rerun
  python manage.py visualize output/4dgs/black_cat/point_cloud
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common args
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--model", default=default_model, choices=available_models,
                               help=f"Select model (default: {default_model})")

    # =====================
    # CORE COMMANDS
    # =====================

    # Command: setup
    subparsers.add_parser("setup", parents=[parent_parser],
                          help="Setup environment for selected model")

    # Command: process-unity (renamed from process-json, THE KEY COMMAND)
    parser_unity = subparsers.add_parser("process-unity",
        help="[CORE] Sync Unity JSON + Diffusion Video → 4DGS dataset (bypasses SfM!)")
    parser_unity.add_argument("video", help="Diffusion-generated video (e.g., output_cat.mp4)")
    parser_unity.add_argument("json", help="Unity tracking JSON (e.g., full_data.json)")
    parser_unity.add_argument("original_video", help="Original Unity video (for timing sync)")
    parser_unity.add_argument("--output", required=True, help="Output dataset name (e.g., black_cat)")
    parser_unity.add_argument("--map-pos", type=str, default=None,
                              help="Map position override: x,y,z (default: from map_transform.json or built-in)")
    parser_unity.add_argument("--map-scale", type=str, default=None,
                              help="Map scale override: x,y,z (default: from map_transform.json or built-in)")

    # Command: train
    parser_train = subparsers.add_parser("train", parents=[parent_parser],
                                         help="Train 4DGS on prepared dataset")
    parser_train.add_argument("scene_path", help="Path to scene folder (e.g., data/black_cat)")
    parser_train.add_argument("--extra", default="", help="Extra arguments for training script")

    # Command: clean-model
    parser_clean = subparsers.add_parser("clean-model",
                                         help="Remove floaters from PLY using Statistical Outlier Removal")
    parser_clean.add_argument("ply_path", help="Path to point_cloud.ply")
    parser_clean.add_argument("--output", help="Output path (default: *_cleaned.ply)")
    parser_clean.add_argument("--neighbors", type=int, default=20, help="Neighbors for SOR (default: 20)")

    # Command: visualize (renamed from visualize-4dgs)
    parser_vis = subparsers.add_parser("visualize",
                                       help="Visualize 4DGS in Rerun")
    parser_vis.add_argument("dir", help="Directory containing PLY files")
    parser_vis.add_argument("--watch", action="store_true", help="Watch for new files")
    parser_vis.add_argument("--web", action="store_true", help="Serve over web")
    parser_vis.add_argument("--save", help="Save to .rrd file")

    # Command: list-models
    subparsers.add_parser("list-models", help="List available model configurations")

    # =====================
    # UTILITY COMMANDS
    # =====================

    # Command: download_data
    parser_data = subparsers.add_parser("download_data", help="Download benchmark datasets")
    parser_data.add_argument("--dataset", default="tandt", help="Dataset name")

    # Command: export-splat
    parser_export = subparsers.add_parser("export-splat", help="Convert PLY to .splat for web viewers")
    parser_export.add_argument("input", help="Input PLY file")
    parser_export.add_argument("output", help="Output .splat file")

    # Command: run-api (server)
    subparsers.add_parser("run-api", help="Start FastAPI server")

    # Command: setup-server (full server environment setup)
    parser_server = subparsers.add_parser("setup-server",
        help="[SERVER] Install all dependencies and setup environment")
    parser_server.add_argument("--with-vggt", action="store_true",
                               help="Also install VGGT model")
    parser_server.add_argument("--skip-system", action="store_true",
                               help="Skip system package installation (requires sudo)")

    args = parser.parse_args()

    # Load global config
    default_config_path = "configs/default.yaml"
    if os.path.exists(default_config_path):
        global_config = load_config(default_config_path)
    else:
        global_config = {'paths': {'output_root': 'output', 'data_root': 'data'}}

    data_root = global_config['paths'].get('data_root', 'data')

    # =====================
    # COMMAND HANDLERS
    # =====================

    if args.command == "setup":
        model_config = registry.get_model(args.model)
        SetupManager(global_config, model_config).run()

    elif args.command == "process-unity":
        from src.json_sync_utils import sync_video_with_json, DEFAULT_MAP_TRANSFORM
        import numpy as np

        project_dir = os.path.join(data_root, args.output)

        # Handle optional map transform overrides
        map_transform = None
        if args.map_pos or args.map_scale:
            map_transform = {
                'position': np.array([float(x) for x in args.map_pos.split(',')]) if args.map_pos else np.array(DEFAULT_MAP_TRANSFORM['position']),
                'rotation': np.array([0, 0, 0]),
                'scale': np.array([float(x) for x in args.map_scale.split(',')]) if args.map_scale else np.array(DEFAULT_MAP_TRANSFORM['scale'])
            }

        result = sync_video_with_json(args.video, args.json, args.original_video, project_dir, map_transform)

        if result:
            print(f"\n{'='*60}")
            print(f"[SUCCESS] Dataset created at: {project_dir}")
            print(f"{'='*60}")
            print(f"\nGenerated files:")
            print(f"  - images/           : Extracted frames")
            print(f"  - sync_metadata.json: Frame-by-frame Unity data")
            print(f"  - transforms_*.json : Camera matrices (NeRF format)")
            print(f"  - timestamps.json   : Frame timestamps for 4DGS")
            print(f"  - map_transform.json: Coordinate conversion params")
            print(f"  - sparse/0/         : COLMAP-format (for compatibility)")
            print(f"\nNext steps:")
            print(f"  1. Train: python manage.py train {args.output}")
            print(f"  2. Render: CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py -m output/4dgs/{args.output} --skip_train --skip_test")
        else:
            print("[Error] Unity data processing failed.")
            sys.exit(1)

    elif args.command == "train":
        model_config = registry.get_model(args.model)
        Runner(global_config, model_config).train(args.scene_path, args.extra)

    elif args.command == "clean-model":
        from src.filter_utils import clean_ply_model
        output = args.output if args.output else args.ply_path.replace(".ply", "_cleaned.ply")
        clean_ply_model(args.ply_path, output, nb_neighbors=args.neighbors)

    elif args.command == "visualize":
        from src.rerun_vis import run_visualization
        run_visualization(args.dir, watch=args.watch, web=args.web, save_path=args.save)

    elif args.command == "list-models":
        print(f"\n{'Model':<15} | {'VRAM':<10} | Description")
        print("-" * 70)
        for name in available_models:
            cfg = registry.get_model(name)
            print(f"{name:<15} | {cfg.get('vram_usage', 'N/A'):<10} | {cfg.get('description', '')}")

    elif args.command == "download_data":
        DatasetManager(global_config).download(args.dataset)

    elif args.command == "export-splat":
        from src.exporter import convert_ply_to_splat
        convert_ply_to_splat(args.input, args.output)

    elif args.command == "run-api":
        import uvicorn
        uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

    elif args.command == "setup-server":
        import subprocess
        import shutil

        print("=" * 60)
        print(" 4DGS Server Setup")
        print("=" * 60)

        # Step 1: System packages (requires sudo)
        if not args.skip_system:
            print("\n[1/4] Installing system dependencies (requires sudo)...")
            system_cmds = [
                "apt-get update",
                "apt-get install -y git wget ninja-build libx11-6 libgl1 libglib2.0-0",
            ]

            # Check if CUDA 11.8 nvcc exists
            if not os.path.exists("/usr/local/cuda-11.8/bin/nvcc"):
                print("Installing CUDA 11.8 toolkit...")
                system_cmds.extend([
                    "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin",
                    "mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600",
                    "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb",
                    "dpkg -i cuda-keyring_1.1-1_all.deb",
                    "rm -f cuda-keyring_1.1-1_all.deb",
                    "apt-get update",
                    "apt-get install -y cuda-nvcc-11-8",
                ])

            for cmd in system_cmds:
                print(f"  $ sudo {cmd}")
                try:
                    subprocess.run(f"sudo {cmd}", shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  [Warning] Command failed: {e}")
        else:
            print("\n[1/4] Skipping system packages (--skip-system)")

        # Step 2: Set CUDA environment
        print("\n[2/4] Setting up CUDA environment...")
        cuda_home = "/usr/local/cuda-11.8"
        if os.path.exists(cuda_home):
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
            print(f"  CUDA_HOME={cuda_home}")
        else:
            print("  [Warning] CUDA 11.8 not found, using system default")

        # Step 3: Python dependencies
        print("\n[3/4] Installing Python dependencies...")
        pip_cmds = [
            'pip install --upgrade pip',
            'pip install "numpy<2.0"',
            'pip install websockets',
        ]
        for cmd in pip_cmds:
            print(f"  $ {cmd}")
            subprocess.run(cmd, shell=True)

        # Step 4: Setup models
        print("\n[4/4] Setting up 4DGS model...")
        model_config = registry.get_model("4dgs")
        SetupManager(global_config, model_config).run()

        if args.with_vggt and "vggt" in available_models:
            print("\n[4/4+] Setting up VGGT model...")
            vggt_config = registry.get_model("vggt")
            SetupManager(global_config, vggt_config).run()

        print("\n" + "=" * 60)
        print(" Setup Complete!")
        print("=" * 60)
        print("""
Next steps:
  1. Process Unity data:
     python manage.py process-unity <video> <json> <original> --output <name>

  2. Train 4DGS:
     python manage.py train data/<name>

  3. Render with camera rotation:
     CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py -m output/4dgs/<name> --skip_train --skip_test

Add to ~/.bashrc for persistent CUDA:
  export CUDA_HOME=/usr/local/cuda-11.8
  export PATH=$CUDA_HOME/bin:$PATH
        """)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
