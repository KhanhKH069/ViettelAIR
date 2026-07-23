"""
COLMAP binary format readers and competition data parsers.

COLMAP convention (important for downstream correctness):
  - Rotation:    R  maps world→camera  (i.e. w2c rotation)
  - Translation: t  is camera position in world, expressed in camera frame
                    i.e.  t = -R @ camera_center_in_world
  - So camera_center_world = -R.T @ t
  - Coordinate system: OpenCV (x-right, y-down, z-forward in camera space)

Internal "c2w" convention used throughout this codebase:
  - 4×4 matrix that maps points from camera space → world space
  - c2w = [[R.T, -R.T@t], [0,0,0,1]]

References:
  https://colmap.github.io/format.html
  https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
"""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class ColmapCamera:
    """Single camera model (intrinsics)."""
    camera_id: int
    model: str          # e.g. "PINHOLE", "SIMPLE_RADIAL", ...
    width: int
    height: int
    params: np.ndarray  # model-dependent (fx, fy, cx, cy, ...)

    @property
    def fx(self) -> float:
        return float(self.params[0])

    @property
    def fy(self) -> float:
        if self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL":
            return float(self.params[0])
        return float(self.params[1])

    @property
    def cx(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            return float(self.params[1])
        return float(self.params[2])

    @property
    def cy(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            return float(self.params[2])
        return float(self.params[3])

    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0,    self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,    1.0],
        ], dtype=np.float64)


@dataclass
class ColmapImage:
    """Single registered image (extrinsics + metadata)."""
    image_id: int
    qvec: np.ndarray    # (qw, qx, qy, qz) — w2c rotation quaternion
    tvec: np.ndarray    # (tx, ty, tz)      — w2c translation
    camera_id: int
    name: str           # relative image path
    xys: np.ndarray     # 2-D feature points, shape (N, 2) — not always needed
    point3D_ids: np.ndarray  # corresponding 3-D point IDs, shape (N,)

    def R(self) -> np.ndarray:
        """3×3 world-to-camera rotation matrix."""
        return qvec_to_rotmat(self.qvec)

    def c2w(self) -> np.ndarray:
        """4×4 camera-to-world matrix (OpenCV convention)."""
        return colmap_to_c2w(self.R(), self.tvec)


@dataclass
class ColmapPoint3D:
    """Single 3-D point from sparse reconstruction."""
    point3D_id: int
    xyz: np.ndarray     # (3,)
    rgb: np.ndarray     # (3,) uint8
    error: float
    track: List[Tuple[int, int]] = field(default_factory=list)  # (image_id, point2D_idx)


@dataclass
class ColmapScene:
    """Full parsed COLMAP scene."""
    cameras: Dict[int, ColmapCamera]
    images: Dict[int, ColmapImage]
    points3D: Dict[int, ColmapPoint3D]


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP quaternion (qw, qx, qy, qz) to 3×3 rotation matrix.
    The result R maps world→camera (w2c).
    """
    qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    return R


def colmap_to_c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP (w2c) rotation + translation to 4×4 camera-to-world matrix.

    COLMAP: X_cam = R @ X_world + t
    Inverse: X_world = R.T @ (X_cam - t) = R.T @ X_cam + (-R.T @ t)

    So c2w[:3,:3] = R.T,  c2w[:3,3] = -R.T @ t
    """
    Rt = R.T
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Rt
    c2w[:3, 3] = -Rt @ t
    return c2w


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# COLMAP binary readers
# (format spec: https://colmap.github.io/format.html#binary-format)
# ---------------------------------------------------------------------------

# Mapping from model name string to number of parameters
_CAMERA_MODEL_PARAMS: Dict[str, int] = {
    "SIMPLE_PINHOLE":   3,  # f, cx, cy
    "PINHOLE":          4,  # fx, fy, cx, cy
    "SIMPLE_RADIAL":    4,  # f, cx, cy, k
    "RADIAL":           5,  # f, cx, cy, k1, k2
    "OPENCV":           8,  # fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV_FISHEYE":   8,
    "FULL_OPENCV":     12,
    "FOV":              5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE":   5,
    "THIN_PRISM_FISHEYE": 12,
}

_CAMERA_MODEL_IDS: Dict[int, str] = {
    0:  "SIMPLE_PINHOLE",
    1:  "PINHOLE",
    2:  "SIMPLE_RADIAL",
    3:  "RADIAL",
    4:  "OPENCV",
    5:  "OPENCV_FISHEYE",
    6:  "FULL_OPENCV",
    7:  "FOV",
    8:  "SIMPLE_RADIAL_FISHEYE",
    9:  "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def read_cameras_binary(path: str | Path) -> Dict[int, ColmapCamera]:
    """Parse COLMAP cameras.bin → dict[camera_id → ColmapCamera]."""
    cameras: Dict[int, ColmapCamera] = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id  = struct.unpack("<i", f.read(4))[0]
            width     = struct.unpack("<Q", f.read(8))[0]
            height    = struct.unpack("<Q", f.read(8))[0]

            model_name = _CAMERA_MODEL_IDS.get(model_id, f"UNKNOWN_{model_id}")
            num_params = _CAMERA_MODEL_PARAMS.get(model_name, 0)
            params = np.array(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))

            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model_name,
                width=int(width),
                height=int(height),
                params=params,
            )
    return cameras


def read_images_binary(path: str | Path) -> Dict[int, ColmapImage]:
    """Parse COLMAP images.bin → dict[image_id → ColmapImage]."""
    images: Dict[int, ColmapImage] = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec     = np.array(struct.unpack("<4d", f.read(32)))   # qw,qx,qy,qz
            tvec     = np.array(struct.unpack("<3d", f.read(24)))   # tx,ty,tz
            camera_id = struct.unpack("<i", f.read(4))[0]

            # read null-terminated image name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            if num_points2D > 0:
                data = struct.unpack(f"<{num_points2D * 3}d", f.read(num_points2D * 24))
                # interleaved: x, y, point3D_id (as double, but actually int64 packed)
                # Actually: x(f64), y(f64), point3D_id(i64) per point
                # Re-read correctly:
            # Seek back and reread with correct types
            # Each 2D point: x(double), y(double), point3D_id(int64)
            # We already consumed num_points2D * 24 bytes above — need to redo
            # Let's redo: seek back
            f.seek(-num_points2D * 24, 1)
            xys = np.empty((num_points2D, 2), dtype=np.float64)
            point3D_ids = np.empty(num_points2D, dtype=np.int64)
            for i in range(num_points2D):
                x, y = struct.unpack("<2d", f.read(16))
                pid  = struct.unpack("<q", f.read(8))[0]
                xys[i] = (x, y)
                point3D_ids[i] = pid

            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path: str | Path) -> Dict[int, ColmapPoint3D]:
    """Parse COLMAP points3D.bin → dict[point3D_id → ColmapPoint3D]."""
    points3D: Dict[int, ColmapPoint3D] = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz   = np.array(struct.unpack("<3d", f.read(24)))
            rgb   = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]

            track_len = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_len):
                img_id, pt2d_idx = struct.unpack("<2i", f.read(8))
                track.append((img_id, pt2d_idx))

            points3D[point3D_id] = ColmapPoint3D(
                point3D_id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                track=track,
            )
    return points3D


# ---------------------------------------------------------------------------
# High-level scene parser
# ---------------------------------------------------------------------------

def parse_colmap_scene(sparse_dir: str | Path) -> ColmapScene:
    """
    Parse a full COLMAP sparse reconstruction directory.

    Expected layout:
        sparse_dir/
            cameras.bin
            images.bin
            points3D.bin

    Returns a ColmapScene with all cameras, images, and 3-D points.
    """
    sparse_dir = Path(sparse_dir)
    cameras  = read_cameras_binary(sparse_dir / "cameras.bin")
    images   = read_images_binary(sparse_dir / "images.bin")
    points3D = read_points3D_binary(sparse_dir / "points3D.bin")
    return ColmapScene(cameras=cameras, images=images, points3D=points3D)


def get_sfm_points(scene: ColmapScene) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SfM point cloud from a parsed ColmapScene.

    Returns:
        xyz:    float64 array (N, 3) — 3-D positions in world space
        colors: float32 array (N, 3) — RGB in [0, 1]
    """
    pts = list(scene.points3D.values())
    if not pts:
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.float32)
    xyz    = np.stack([p.xyz for p in pts], axis=0)
    colors = np.stack([p.rgb for p in pts], axis=0).astype(np.float32) / 255.0
    return xyz, colors


# ---------------------------------------------------------------------------
# test_poses.csv parser
# ---------------------------------------------------------------------------

@dataclass
class TestPose:
    """Single test camera pose parsed from test_poses.csv."""
    image_name: str
    qvec: np.ndarray    # (qw, qx, qy, qz)
    tvec: np.ndarray    # (tx, ty, tz)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,    self.cx],
            [0.0,    self.fy, self.cy],
            [0.0,    0.0,    1.0],
        ], dtype=np.float64)

    def c2w(self) -> np.ndarray:
        R = qvec_to_rotmat(self.qvec)
        return colmap_to_c2w(R, self.tvec)


def parse_test_poses_csv(csv_path: str | Path) -> List[TestPose]:
    """
    Parse competition test_poses.csv.

    Expected columns (with or without spaces after comma):
        image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
    """
    poses: List[TestPose] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize header keys (strip whitespace)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]  # type: ignore[arg-type]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            poses.append(TestPose(
                image_name=row["image_name"],
                qvec=np.array([float(row["qw"]), float(row["qx"]),
                                float(row["qy"]), float(row["qz"])]),
                tvec=np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])]),
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                width=int(row["width"]),
                height=int(row["height"]),
            ))
    return poses
