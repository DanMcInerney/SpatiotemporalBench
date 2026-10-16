"""Camera geometry over agent-facing data: map frame <-> OpenCV camera frame, projection and unprojection.

Conventions (docs/format.md, "Frames and units"): a pose `rot` is a unit quaternion [w, x, y, z] mapping the body
frame (x forward, y left, z up) to the map frame; the OpenCV camera frame is x right, y down, z forward, so
x_cv = -y_body, y_cv = -z_body, z_cv = x_body. Image coordinates are continuous: pixel column j spans u in
[j, j + 1), and the principal point (cx, cy) is in the same coordinates. Depth is planar (camera-frame z).

Everything a system under test may use lives here, so baselines and the compiler share one transform.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Rows map body coordinates to OpenCV camera coordinates.
CV_FROM_BODY = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

# A depth image is a 16-bit PNG: 1..DEPTH_STEPS counts of the unit in `camera.json`'s `depth_unit`, 0 where there is no
# surface or it lies beyond the range. The units and the metres each stands for; a robot names its own
# (`robots.DepthSpec.unit`, written to `camera.json`).
DEPTH_UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0}
DEPTH_STEPS = 65535


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_unit: str = "mm"
    near_m: float = 0.3  # nearer surfaces read 0 in the depth image
    far_m: float = 150.0  # farther surfaces read 0 (a file without the keys keeps these)

    @classmethod
    def load(cls, source: str | Path | dict) -> Camera:
        data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
        return cls(int(data["width"]), int(data["height"]), float(data["fx"]), float(data["fy"]),
                   float(data["cx"]), float(data["cy"]), str(data.get("depth_unit", "mm")),
                   float(data.get("near_m", cls.near_m)), float(data.get("far_m", cls.far_m)))

    @property
    def depth_scale(self) -> float:
        """Metres per depth-image unit."""
        return DEPTH_UNITS[self.depth_unit]


def rot_matrix(rot: Sequence[float]) -> np.ndarray:
    """3x3 matrix of a unit quaternion [w, x, y, z] (body -> map)."""
    w, x, y, z = (float(c) for c in rot)
    n = w * w + x * x + y * y + z * z
    if not 0.98 < n < 1.02:
        raise ValueError(f"rotation is not a unit quaternion (norm^2 {n:.4f}): {list(rot)}")
    w, x, y, z = (c / n**0.5 for c in (w, x, y, z))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def map_to_cam(points_map: Sequence[float] | np.ndarray, pos: Sequence[float], rot: Sequence[float]) -> np.ndarray:
    """Map-frame point(s) (3,) or (N, 3) -> OpenCV camera frame, for a camera at `pos` with orientation `rot`."""
    p = np.asarray(points_map, dtype=float)
    body = (p - np.asarray(pos, dtype=float)) @ rot_matrix(rot)  # R^T (p - pos), row-wise
    return body @ CV_FROM_BODY.T


def cam_to_map(points_cam: Sequence[float] | np.ndarray, pos: Sequence[float], rot: Sequence[float]) -> np.ndarray:
    """OpenCV camera-frame point(s) (3,) or (N, 3) -> map frame. Inverse of `map_to_cam`."""
    p = np.asarray(points_cam, dtype=float)
    body = p @ CV_FROM_BODY  # CV_FROM_BODY^T applied row-wise
    return body @ rot_matrix(rot).T + np.asarray(pos, dtype=float)


def project(point_cam: Sequence[float], cam: Camera) -> tuple[float, float] | None:
    """Continuous image coordinates (u, v) of a camera-frame point, or None if it is not in front of the camera."""
    x, y, z = (float(c) for c in point_cam)
    if z <= 1e-9:
        return None
    return cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy


def in_image(uv: tuple[float, float] | None, cam: Camera) -> bool:
    return uv is not None and 0.0 <= uv[0] < cam.width and 0.0 <= uv[1] < cam.height


def unproject(u: float, v: float, depth_m: float, cam: Camera) -> np.ndarray:
    """Camera-frame point at continuous image coordinates (u, v) with planar depth `depth_m`."""
    return np.array([(u - cam.cx) * depth_m / cam.fx, (v - cam.cy) * depth_m / cam.fy, depth_m])
