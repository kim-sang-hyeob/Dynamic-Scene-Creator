# Archived Files

These files were archived during project cleanup (2026-01-22).
They are not needed for the core Unity→4DGS pipeline but may be useful for reference.

## src/ files

| File | Original Purpose |
|------|------------------|
| sfm_utils.py | COLMAP-based SfM pipeline |
| dust3r_utils.py | Dust3R SfM-free pose estimation |
| vggt_utils.py | VGGT pose estimation (CVPR 2025) |
| colmap_to_nerf.py | COLMAP → transforms.json conversion |
| align_utils.py | Scene rotation using pycolmap |
| patch_4dgs_camera_offset_v2.py | Alternative camera patch (duplicate) |
| render_custom_camera.py | Custom camera rendering |
| verify_camera_rotation.py | Camera rotation debugging |
| split_images.py | Train/val/test splitting |
| analyze_motion.py | Gaussian motion analysis |
| freeze_static_points.py | Static point freezing |
| viser_4dgs_viewer.py | Viser-based 4DGS viewer |

## configs/models/ files

| File | Original Purpose |
|------|------------------|
| dust3r.yaml | Dust3R model config |
| vggt.yaml | VGGT model config |
| compgs.yaml | Compressed 3DGS config |
| flashgs.yaml | FlashGS config |
| gsplat.yaml | Gsplat config |

## To restore

```bash
mv archive/src/*.py src/
mv archive/configs/models/*.yaml configs/models/
```
