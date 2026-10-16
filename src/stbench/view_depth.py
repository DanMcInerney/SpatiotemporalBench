"""A depth image drawn for the eye, for `stbench view`'s frame-by-frame stream.

A depth image of the dataset is a 16-bit PNG of counts of the camera's unit (`camera.json`'s `depth_unit`), 0 where the
sensor reads nothing: a browser shows it nearly black, since the counts sit in the bottom of the 16-bit range. The
file is the scorer's (`private/<sid>/depth/`), and the viewer reads it from there for a stream that sends depth, the
`detections` gate channel's. It draws it as an 8-bit grey PNG: no reading is black (0), a surface at
`near_m` white (255) and one at `far_m` black, linear in metres between. The agent is fed the camera and sees no depth
picture (0.6.0); the harness copies the 16-bit file to the `detections` system alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import Camera
from .render_encode import read_png16


def depth_view(counts: np.ndarray, cam: Camera) -> np.ndarray:
    """The 8-bit grey picture of a depth image's `counts` (see the module docstring); the array keeps its shape. A
    reading outside `near_m` to `far_m` is clamped to the range. ValueError for an empty range."""
    if cam.far_m <= cam.near_m:
        raise ValueError(f"the depth range is empty: near {cam.near_m:g} m, far {cam.far_m:g} m")
    metres = np.asarray(counts, dtype=np.float64) * cam.depth_scale
    level = np.rint(255.0 * (cam.far_m - np.clip(metres, cam.near_m, cam.far_m)) / (cam.far_m - cam.near_m))
    return np.where(np.asarray(counts) > 0, level, 0.0).astype(np.uint8)


def write_depth_view(source: Path, target: Path, cam: Camera) -> bool:
    """Draw the depth image `source` at `target` (its folder made) and say so; a `target` that is not older than
    `source` is left as it is (False), so a second view of a dataset redraws nothing. ValueError for a file that is not
    a 16-bit PNG."""
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return False
    from PIL import Image  # as in render_encode: Pillow loads where it is used

    picture = depth_view(read_png16(source), cam)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(picture).save(target, format="PNG")
    return True
