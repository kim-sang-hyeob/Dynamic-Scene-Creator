#!/usr/bin/env python3
"""
Camera Transformation Module for 4DGS Pipeline

Provides utilities for:
- Unity to NeRF coordinate conversion
- Camera rotation around objects
- 4DGS camera format conversion (c2w <-> w2c <-> stored format)

Usage:
    from camera_transform import CameraTransformer, CoordinateConverter

    # Convert Unity coords to NeRF
    converter = CoordinateConverter(map_pos=[-150.85, -30.0, 3.66], map_scale=[3, 3, 3])
    nerf_pos = converter.unity_to_nerf([x, y, z])

    # Rotate camera around object
    transformer = CameraTransformer(rotation_center=nerf_pos)
    new_R, new_T = transformer.rotate_camera(R, T, angle_degrees=45)
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass


@dataclass
class MapTransform:
    """Unity map transform parameters."""
    position: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_dict(cls, data: dict) -> 'MapTransform':
        """Create from dictionary with 'position' and 'scale' keys."""
        return cls(
            position=np.array([data['position']['x'], data['position']['y'], data['position']['z']]),
            scale=np.array([data['scale']['x'], data['scale']['y'], data['scale']['z']])
        )

    @classmethod
    def default_black_cat(cls) -> 'MapTransform':
        """Default transform for black_cat project."""
        return cls(
            position=np.array([-150.85, -30.0, 3.66]),
            scale=np.array([3.0, 3.0, 3.0])
        )


class CoordinateConverter:
    """Convert coordinates between Unity and NeRF coordinate systems."""

    def __init__(self, map_pos: Union[List[float], np.ndarray],
                 map_scale: Union[List[float], np.ndarray]):
        """
        Initialize converter with map transform.

        Args:
            map_pos: Unity map position [x, y, z]
            map_scale: Unity map scale [x, y, z]
        """
        self.map_pos = np.array(map_pos)
        self.map_scale = np.array(map_scale)

    @classmethod
    def from_map_transform(cls, transform: MapTransform) -> 'CoordinateConverter':
        """Create from MapTransform object."""
        return cls(transform.position, transform.scale)

    def unity_to_nerf(self, unity_pos: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert Unity world position to NeRF coordinates.

        Transform: (unity_pos - map_pos) / map_scale, then flip Z

        Args:
            unity_pos: Position in Unity coordinates [x, y, z]

        Returns:
            Position in NeRF coordinates [x, y, -z]
        """
        unity_pos = np.array(unity_pos)
        local = (unity_pos - self.map_pos) / self.map_scale
        return np.array([local[0], local[1], -local[2]])

    def nerf_to_unity(self, nerf_pos: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert NeRF coordinates back to Unity world position.

        Args:
            nerf_pos: Position in NeRF coordinates [x, y, z]

        Returns:
            Position in Unity world coordinates
        """
        nerf_pos = np.array(nerf_pos)
        # Reverse Z flip
        local = np.array([nerf_pos[0], nerf_pos[1], -nerf_pos[2]])
        # Reverse transform
        return local * self.map_scale + self.map_pos


class CameraFormatConverter:
    """
    Convert between different camera matrix formats used in 4DGS.

    4DGS storage format:
        R_stored = -w2c_rot.T  with R_stored[:,0] negated
        T_stored = -w2c_trans
    """

    @staticmethod
    def stored_to_w2c(R_stored: np.ndarray, T_stored: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 4DGS stored format to world-to-camera (w2c).

        Args:
            R_stored: Stored rotation matrix (3x3)
            T_stored: Stored translation vector (3,)

        Returns:
            (w2c_rot, w2c_trans): World-to-camera rotation and translation
        """
        R_tmp = R_stored.copy()
        R_tmp[:, 0] = -R_tmp[:, 0]  # Undo first column negation
        w2c_rot = -R_tmp.T
        w2c_trans = -T_stored
        return w2c_rot, w2c_trans

    @staticmethod
    def w2c_to_stored(w2c_rot: np.ndarray, w2c_trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert world-to-camera (w2c) to 4DGS stored format.

        Args:
            w2c_rot: World-to-camera rotation matrix (3x3)
            w2c_trans: World-to-camera translation vector (3,)

        Returns:
            (R_stored, T_stored): 4DGS stored rotation and translation
        """
        R_stored = -w2c_rot.T
        R_stored[:, 0] = -R_stored[:, 0]
        T_stored = -w2c_trans
        return R_stored, T_stored

    @staticmethod
    def stored_to_c2w(R_stored: np.ndarray, T_stored: np.ndarray) -> np.ndarray:
        """
        Convert 4DGS stored format to camera-to-world (c2w) 4x4 matrix.

        Args:
            R_stored: Stored rotation matrix (3x3)
            T_stored: Stored translation vector (3,)

        Returns:
            c2w: Camera-to-world 4x4 transformation matrix
        """
        w2c_rot, w2c_trans = CameraFormatConverter.stored_to_w2c(R_stored, T_stored)
        w2c = np.eye(4)
        w2c[:3, :3] = w2c_rot
        w2c[:3, 3] = w2c_trans
        c2w = np.linalg.inv(w2c)
        return c2w

    @staticmethod
    def c2w_to_stored(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert camera-to-world (c2w) 4x4 matrix to 4DGS stored format.

        Args:
            c2w: Camera-to-world 4x4 transformation matrix

        Returns:
            (R_stored, T_stored): 4DGS stored rotation and translation
        """
        w2c = np.linalg.inv(c2w)
        w2c_rot = w2c[:3, :3]
        w2c_trans = w2c[:3, 3]
        return CameraFormatConverter.w2c_to_stored(w2c_rot, w2c_trans)

    @staticmethod
    def get_camera_position(R_stored: np.ndarray, T_stored: np.ndarray) -> np.ndarray:
        """
        Get camera position in world coordinates from stored format.

        Args:
            R_stored: Stored rotation matrix (3x3)
            T_stored: Stored translation vector (3,)

        Returns:
            Camera position in world coordinates (3,)
        """
        c2w = CameraFormatConverter.stored_to_c2w(R_stored, T_stored)
        return c2w[:3, 3]


class CameraTransformer:
    """Transform cameras in 4DGS format."""

    def __init__(self, rotation_center: Optional[np.ndarray] = None):
        """
        Initialize camera transformer.

        Args:
            rotation_center: Center point for rotation in NeRF coordinates.
                           If None, rotates around origin.
        """
        self.rotation_center = rotation_center if rotation_center is not None else np.zeros(3)

    @staticmethod
    def make_rotation_matrix_y(angle_degrees: float) -> np.ndarray:
        """
        Create Y-axis rotation matrix.

        Args:
            angle_degrees: Rotation angle in degrees.
                          Positive = counterclockwise when viewed from above

        Returns:
            3x3 rotation matrix
        """
        rad = np.radians(angle_degrees)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float64)

    @staticmethod
    def make_rotation_matrix_x(angle_degrees: float) -> np.ndarray:
        """Create X-axis rotation matrix."""
        rad = np.radians(angle_degrees)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ], dtype=np.float64)

    @staticmethod
    def make_rotation_matrix_z(angle_degrees: float) -> np.ndarray:
        """Create Z-axis rotation matrix."""
        rad = np.radians(angle_degrees)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float64)

    def rotate_camera(self, R_stored: np.ndarray, T_stored: np.ndarray,
                      angle_degrees: float, axis: str = 'y') -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate camera around rotation_center.

        Camera position orbits around the center while orientation also rotates
        to keep looking at the same relative direction.

        Args:
            R_stored: 4DGS stored rotation matrix (3x3)
            T_stored: 4DGS stored translation vector (3,)
            angle_degrees: Rotation angle in degrees
            axis: Rotation axis ('x', 'y', or 'z')

        Returns:
            (new_R, new_T): Rotated camera in 4DGS stored format
        """
        # Get rotation matrix for the axis
        if axis.lower() == 'x':
            rot_mat = self.make_rotation_matrix_x(angle_degrees)
        elif axis.lower() == 'y':
            rot_mat = self.make_rotation_matrix_y(angle_degrees)
        elif axis.lower() == 'z':
            rot_mat = self.make_rotation_matrix_z(angle_degrees)
        else:
            raise ValueError(f"Unknown axis: {axis}. Use 'x', 'y', or 'z'.")

        # Convert to c2w
        c2w = CameraFormatConverter.stored_to_c2w(R_stored, T_stored)
        cam_pos = c2w[:3, 3]

        # Rotate position around center
        relative_pos = cam_pos - self.rotation_center
        new_relative_pos = rot_mat @ relative_pos
        new_cam_pos = self.rotation_center + new_relative_pos

        # Rotate orientation
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = rot_mat @ c2w[:3, :3]
        new_c2w[:3, 3] = new_cam_pos

        # Convert back to stored format
        return CameraFormatConverter.c2w_to_stored(new_c2w)

    def translate_camera(self, R_stored: np.ndarray, T_stored: np.ndarray,
                         translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Translate camera position.

        Args:
            R_stored: 4DGS stored rotation matrix (3x3)
            T_stored: 4DGS stored translation vector (3,)
            translation: Translation vector in world coordinates (3,)

        Returns:
            (new_R, new_T): Translated camera in 4DGS stored format
        """
        c2w = CameraFormatConverter.stored_to_c2w(R_stored, T_stored)
        c2w[:3, 3] += translation
        return CameraFormatConverter.c2w_to_stored(c2w)


def load_rotation_center_from_sync_metadata(
    sync_metadata_path: str,
    map_pos: Union[List[float], np.ndarray] = [-150.85, -30.0, 3.66],
    map_scale: Union[List[float], np.ndarray] = [3.0, 3.0, 3.0],
    num_frames: int = 10
) -> Optional[np.ndarray]:
    """
    Load rotation center from sync_metadata.json by averaging object positions.

    Args:
        sync_metadata_path: Path to sync_metadata.json
        map_pos: Unity map position
        map_scale: Unity map scale
        num_frames: Number of frames to average

    Returns:
        Rotation center in NeRF coordinates, or None if failed
    """
    import json
    import os

    if not os.path.exists(sync_metadata_path):
        return None

    try:
        with open(sync_metadata_path) as f:
            sync_data = json.load(f)

        obj_positions = []
        for frame in sync_data[:num_frames]:
            if 'objPos' in frame:
                obj_positions.append([
                    frame['objPos']['x'],
                    frame['objPos']['y'],
                    frame['objPos']['z']
                ])

        if not obj_positions:
            return None

        avg_obj_unity = np.mean(obj_positions, axis=0)
        converter = CoordinateConverter(map_pos, map_scale)
        return converter.unity_to_nerf(avg_obj_unity)

    except Exception as e:
        print(f"Warning: Could not load rotation center from {sync_metadata_path}: {e}")
        return None


def estimate_rotation_center_from_camera(
    R_stored: np.ndarray,
    T_stored: np.ndarray,
    look_distance: float = 4.5
) -> np.ndarray:
    """
    Estimate rotation center by projecting from camera position along look direction.

    Args:
        R_stored: 4DGS stored rotation matrix
        T_stored: 4DGS stored translation vector
        look_distance: Distance from camera to estimated object center

    Returns:
        Estimated rotation center in NeRF coordinates
    """
    c2w = CameraFormatConverter.stored_to_c2w(R_stored, T_stored)
    cam_pos = c2w[:3, 3]
    look_dir = -c2w[:3, 2]  # Camera looks along -Z in camera space
    look_dir = look_dir / np.linalg.norm(look_dir)
    return cam_pos + look_dir * look_distance


# Convenience functions
def rotate_cameras_around_object(
    cam_infos: list,
    angle_degrees: float,
    rotation_center: np.ndarray,
    CameraInfo: type
) -> list:
    """
    Rotate a list of CameraInfo objects around an object center.

    Args:
        cam_infos: List of CameraInfo named tuples
        angle_degrees: Rotation angle in degrees
        rotation_center: Center of rotation in NeRF coordinates
        CameraInfo: The CameraInfo namedtuple class

    Returns:
        List of new CameraInfo objects with rotated cameras
    """
    if angle_degrees == 0:
        return cam_infos

    transformer = CameraTransformer(rotation_center)
    rotated = []

    for idx, cam in enumerate(cam_infos):
        R_stored = np.array(cam.R)
        T_stored = np.array(cam.T)

        new_R, new_T = transformer.rotate_camera(R_stored, T_stored, angle_degrees, axis='y')

        rotated.append(CameraInfo(
            uid=idx,
            R=new_R,
            T=new_T,
            FovY=cam.FovY,
            FovX=cam.FovX,
            image=cam.image,
            image_path=cam.image_path,
            image_name=cam.image_name,
            width=cam.width,
            height=cam.height,
            time=cam.time,
            mask=cam.mask
        ))

    return rotated


if __name__ == "__main__":
    # Example usage
    print("Camera Transform Module")
    print("=" * 50)

    # Example: Convert Unity to NeRF coordinates
    converter = CoordinateConverter(
        map_pos=[-150.85, -30.0, 3.66],
        map_scale=[3.0, 3.0, 3.0]
    )

    # Black cat Unity position
    unity_pos = [-149.0284, -30.0, 25.96]
    nerf_pos = converter.unity_to_nerf(unity_pos)
    print(f"Unity position: {unity_pos}")
    print(f"NeRF position:  {nerf_pos}")

    # Example: Create rotation matrix
    rot_45 = CameraTransformer.make_rotation_matrix_y(45)
    print(f"\n45-degree Y rotation matrix:\n{rot_45}")

    # Example: Verify round-trip
    back_to_unity = converter.nerf_to_unity(nerf_pos)
    print(f"\nRound-trip Unity: {back_to_unity}")
    print(f"Original Unity:   {unity_pos}")
    print(f"Difference:       {np.abs(back_to_unity - unity_pos).max():.10f}")
