# Dynamic Scene Creator

**A Colmap-free 4D Gaussian Splatting pipeline for composing dynamic objects on high-quality static 3DGS maps.**

Leverages Unity camera tracking data to train 4D Gaussian Splatting models without Structure from Motion (SfM), plus a browser-based scene composer for multi-layer Gaussian splatting composition with AI-powered 3D generation.

---

## Key Features

### 1. Large Translational Motion

Most existing 4DGS work targets **quasi-static motion** (hand waving, facial expressions). This pipeline handles **large translational motion** — objects moving across the scene (running animals, walking people, etc.).

| | Conventional 4DGS | This Pipeline |
|---|-----------|---------------|
| Target motion | Quasi-static | Large translational |
| Deformation | Small displacement | Large spatial movement + shape deformation |
| Initial points | SfM points | MiDaS depth-based foreground initialization |

### 2. Static-Dynamic Composition

Instead of learning the entire scene as one 4DGS, we **separate static background and dynamic objects**.

```
Conventional: Entire scene → Single 4DGS → Background + objects mixed
Ours:         Static background → High-quality 3DGS map (reusable)
              Dynamic object → Background-removed 4DGS (swappable/editable)
```

**Benefits:**
- Swap/edit dynamic objects without retraining the entire scene
- Reuse existing high-quality 3DGS maps
- Improved training efficiency by removing background Gaussians (PSNR 33+)

### 3. Colmap-free Pipeline

Uses Unity camera ground truth directly, completely bypassing COLMAP SfM.

| | COLMAP-based | This Pipeline |
|---|------------|---------------|
| Camera pose | Estimated via SfM (can fail) | Unity GT directly |
| Initial points | SfM points | Alpha mask + MiDaS depth back-projection |
| Preprocessing time | Minutes to hours | Under a few minutes |
| Dynamic scenes | SfM failure risk | No issues |

### 4. Scene Composer (Browser-based)

A multi-layer 4DGS composition tool running in the browser:
- **Multi-layer system**: Load multiple .splat/.ply/.splatv files as separate layers
- **Transform gizmos**: Translate / Rotate / Scale each layer independently
- **AI 3D generation**: Text/image → 3D via TRELLIS integration
- **Path editor**: Define motion paths for dynamic objects with Bezier curves
- **Export**: Merge composed scenes into a single .splat file

---

## Pipeline Overview

```
Unity Scene → Camera Tracking JSON → Diffusion Video Generation
                    ↓
            process-unity (SfM bypass)
              ├─ BiRefNet background removal
              ├─ MiDaS depth estimation → foreground point initialization
              └─ Unity→COLMAP/NeRF coordinate transform
                    ↓
              4DGS Training (background removal + Loss Masking)
                    ↓
         Novel View Rendering (any angle)
                    ↓
         Scene Composer (multi-layer composition in browser)
```

## Capabilities

- **SfM-free**: Direct use of Unity camera data (no COLMAP needed)
- **Background removal training**: BiRefNet background removal (`--remove-bg`) + Alpha-aware Loss Masking
- **MiDaS depth initialization** (optional): Monocular depth estimation for foreground point cloud generation (`--no-midas` to disable)
- **Camera rotation rendering**: Render trained models from various angles
- **Configurable coordinate transform**: Unity↔NeRF coordinate transform parameter settings
- **Trajectory visualization**: Rerun-based Gaussian trajectory visualization

## Requirements

- Python 3.8+
- CUDA 11.8 (V100 GPU compatible)
- PyTorch (CUDA 11.8 build)
- numpy < 2.0

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/kim-sang-hyeob/Dynamic-Scene-Creator.git
cd Dynamic-Scene-Creator

# 2. Automated setup (recommended, run as root)
chmod +x scripts/setup_server.sh
./scripts/setup_server.sh

# Or via Python CLI
python manage.py setup-server
```

### Manual Installation

```bash
# CUDA 11.8 (required for V100, run as root)
apt-get update
apt-get install -y cuda-nvcc-11-8

# Environment variables
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# Python dependencies
pip install "numpy<2.0"
pip install websockets

# 4DGS model setup
python manage.py setup --model 4dgs
```

## Quick Start

```bash
# 1. Process data (limit frames + downscale to save VRAM)
python manage.py process-unity \
    data/black_cat/output_cat.mp4 \
    data/black_cat/full_data.json \
    data/black_cat/original_catvideo.mp4 \
    --output black_cat \
    --frames 40 \
    --resize 0.5

# 2. Train (--low-vram: batch_size=1)
python manage.py train data/black_cat --low-vram

# 3. Render (45-degree rotation)
CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py \
    -m output/4dgs/black_cat \
    --skip_train --skip_test
```

## Usage

### 1. Unity Data Processing

Convert Unity camera tracking JSON and video into a 4DGS dataset.

```bash
# Basic usage
python manage.py process-unity \
    data/black_cat/output_cat.mp4 \
    data/black_cat/full_data.json \
    data/black_cat/original_catvideo.mp4 \
    --output black_cat \
    --frames 40 \
    --resize 0.5

# With background removal (BiRefNet)
python manage.py process-unity \
    data/black_cat/output_cat.mp4 \
    data/black_cat/full_data.json \
    data/black_cat/original_catvideo.mp4 \
    --output black_cat_alpha \
    --frames 40 \
    --resize 0.5 \
    --remove-bg
```

**Options:**
- `--frames 40` — Uniform sampling of 40 frames (includes first and last)
- `--resize 0.5` — Downscale images to 50% (or specify resolution like `384x216`)
- `--remove-bg` — Background removal with BiRefNet (produces transparent PNGs; requires `--white_background` during training)
- `--no-midas` — Disable MiDaS depth estimation

**Input files:**
- `output_cat.mp4` — Video generated by diffusion model
- `full_data.json` — Unity camera tracking data
- `original_catvideo.mp4` — Original Unity video (for timing sync)

**Generated files:**
- `images/` — Extracted frames
- `sync_metadata.json` — Per-frame Unity data
- `transforms_train.json` — Camera matrices (NeRF format)
- `timestamps.json` — Frame timestamps for 4DGS
- `map_transform.json` — Coordinate transform parameters
- `sparse/0/` — COLMAP-compatible format

### 2. Background Removal Pipeline (without Unity)

Generate a 4DGS dataset from video alone, without Unity camera data.

```bash
# Full pipeline (background removal + sparse generation)
python manage.py prepare-alpha data/my_video.mp4 \
    --output my_scene_alpha \
    --frames 40 \
    --resize 512x295

# Train (--white_background required for transparent backgrounds)
python manage.py train data/my_scene_alpha --extra="--white_background"
```

### 3. Training

```bash
python manage.py train data/black_cat
```

### 4. Rendering (Camera Rotation)

```bash
# Render at 45-degree rotation
CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py \
    -m output/4dgs/black_cat \
    --skip_train --skip_test

# Front view (0 degrees)
CAMERA_ANGLE_OFFSET=0 python external/4dgs/render.py \
    -m output/4dgs/black_cat \
    --skip_train --skip_test
```

### 5. Scene Composer

```bash
# Start the scene composer server
cd viewer/scene-composer && python3 server.py --port 8080

# (Optional) Start TRELLIS server for AI 3D generation
cd viewer/trellis-server && python3 server.py --port 8000
```

Open `http://localhost:8080` in a browser. Drag & drop `.splat` / `.ply` files to compose scenes.

### 6. Visualization (Optional)

```bash
# Visualize in Rerun
python manage.py visualize output/4dgs/black_cat/point_cloud

# Web viewer
python manage.py visualize output/4dgs/black_cat/point_cloud --web
```

## Technical Highlights

### Problem-Solution Flow

```
Problem: Background artifacts appear even with transparent PNGs
         ↓
Cause 1: 4DGS ignores Alpha → Applied Alpha Patch
         ↓
Cause 2: Loss computed on background → Applied Loss Masking Patch
         ↓
Cause 3: Initial points not on foreground → MiDaS depth-based foreground init
         ↓
Side issue: CUDA crash → Fixed Tensor View→Clone
         ↓
Result: Clean 4D Gaussian training without background artifacts
```

### Quantitative Results

| Metric | Baseline (with BG) | Loss Masking only | Final (all patches) |
|------|-----------------|----------------|-----------------|
| Initial Points | 1080 | 1080 | 1000 |
| Final Points | 168000+ | 1080 (no change) | 19000+ |
| PSNR | ~25 | 18.4 (training failed) | **33+** |
| Background Gaussians | Present | N/A | **None** |

## Project Structure

```
Dynamic-Scene-Creator/
├── manage.py                  # Main CLI
├── configs/
│   ├── default.yaml           # Global config
│   └── models/
│       └── 4dgs.yaml          # 4DGS model config
├── src/
│   ├── converters/            # Unity → 4DGS data conversion (core)
│   │   ├── frame_extractor.py
│   │   ├── coordinate.py
│   │   ├── colmap_writer.py
│   │   ├── nerf_writer.py
│   │   └── sparse_from_images.py
│   ├── adapters/              # External model/tool wrappers
│   │   ├── background_remover.py
│   │   ├── depth_estimator.py
│   │   ├── camera_transform.py
│   │   ├── rerun_vis.py
│   │   └── visualize_trajectory.py
│   ├── patches_4dgs/          # 4DGS patches (applied during setup)
│   │   ├── alpha.py
│   │   ├── sfm_free.py
│   │   ├── open3d.py
│   │   └── camera_offset.py
│   ├── utils/
│   │   ├── filter.py
│   │   └── exporter.py
│   ├── runner.py
│   ├── setup.py
│   ├── dataset.py
│   └── model_registry.py
├── viewer/
│   ├── scene-composer/        # Browser-based scene composition tool
│   └── trellis-server/        # TRELLIS AI 3D generation server
├── scripts/
│   └── setup_server.sh
├── inputs/                    # Raw input files
├── data/                      # Converted datasets (gitignored)
├── external/4dgs/             # 4DGS repository (gitignored)
└── output/                    # Training output (gitignored)
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `setup` | Install 4DGS environment |
| `process-unity` | Unity JSON + Video → 4DGS dataset |
| `prepare-alpha` | Video → background removal + sparse generation (no Unity) |
| `remove-bg` | Remove background from video (BiRefNet) |
| `create-sparse` | Image folder → COLMAP sparse files |
| `train` | Train 4DGS model |
| `visualize` | Rerun visualization |
| `clean-model` | Remove PLY floaters |
| `list-models` | List available models |
| `setup-server` | Automated server setup |

## License

MIT

## References

- [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians) — Dynamic scene reconstruction
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Static scene reconstruction
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — Background removal
- [MiDaS](https://github.com/isl-org/MiDaS) — Monocular depth estimation
- [TRELLIS](https://github.com/microsoft/TRELLIS) — Image-to-3D generation
- [Rerun](https://github.com/rerun-io/rerun) — 3D visualization
