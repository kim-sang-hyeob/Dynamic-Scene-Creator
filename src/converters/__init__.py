"""
Data converters for Unity → 4DGS pipeline.

Modules:
    - coordinate: Unity ↔ NeRF coordinate system transformations
    - colmap_writer: COLMAP sparse format generation
    - nerf_writer: transforms_train.json generation
    - frame_extractor: Video frame extraction with JSON synchronization
"""

from .frame_extractor import sync_video_with_json
from .coordinate import (
    normalize_position,
    get_map_transform,
    save_map_transform,
    quat_to_mat,
    mat_to_quat,
)
from .colmap_writer import write_colmap_text
from .nerf_writer import write_transforms_json

__all__ = [
    "sync_video_with_json",
    "normalize_position",
    "get_map_transform",
    "save_map_transform",
    "quat_to_mat",
    "mat_to_quat",
    "write_colmap_text",
    "write_transforms_json",
]
