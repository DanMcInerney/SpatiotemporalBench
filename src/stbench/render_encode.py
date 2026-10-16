"""Turn Blender's raw float passes into the dataset's 16-bit PNGs, and strip metadata from images.

- Depth: planar camera-z in metres (float) -> `cut_depth` to the robot's range (0 nearer than `near_m` or farther than
  `far_m`), then uint16 counts of the robot's depth unit (`camera.DEPTH_UNITS`: millimetres, centimetres or metres; the
  unit, `near_m` and `far_m` are in `sensors/camera.json`); 0 = no surface, nearer than one unit or farther than 65,535
  units (65.535 m in millimetres, 655.35 m in centimetres).
- Instances: the object-index pass (float, integral) -> uint16 `pass_index`; 0 = structure, ground or sky.
- The robot's own body and payload render under the reserved index `SELF_ID`; `self_filter` zeroes both passes there,
  as a real robot's self-filter does, so depth and free space never see the robot itself (the RGB still shows it).
- The RGB's near clip is `rgb_clip_m(near_m)`: nearer than any depth the robot reports, so a close surface is whole.
- JPEG: `camera_model` writes the developed frames with the JFIF header only; `strip_jpeg` drops every other segment of
  a JPEG that has some, losslessly, so the file carries nothing and is byte-stable.

This module is neutral: the agent side (`llm_locate`) and the visibility log read `read_png16`, so it imports nothing of
the truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import DEPTH_STEPS, DEPTH_UNITS

MAX_U16 = 65535  # the largest value of a uint16 pass
SELF_ID = 65535  # the robot's own body and payload (from M3); never in instance_map.json
NEAR_CLIP_MAX_M = 0.005  # the RGB's near clip is at most this: the camera of a robot 2 cm under a shelf sees it whole
ID_TOLERANCE = 1e-3  # the index pass is exactly integral; anything else means the renderer blended or mis-wrote it


def depth_to_units(depth_m: np.ndarray, unit: str) -> np.ndarray:
    """Metres (float, background is huge or non-finite) -> uint16 counts of `unit` ("mm", "cm" or "m": the map's
    `depth_unit`), rounded to nearest; 0 where there is no depth, under one unit or beyond `DEPTH_STEPS` units."""
    if unit not in DEPTH_UNITS:
        raise ValueError(f"unknown depth unit {unit!r}; known: {', '.join(DEPTH_UNITS)}")
    d = np.asarray(depth_m, dtype=np.float64)
    ok = np.isfinite(d) & (d > 0)
    counts = np.zeros(d.shape, dtype=np.uint16)
    scaled = np.rint(np.where(ok, d, 0.0) * round(1 / DEPTH_UNITS[unit]))
    keep = ok & (scaled >= 1) & (scaled <= DEPTH_STEPS)
    counts[keep] = scaled[keep].astype(np.uint16)
    return counts


def rgb_clip_m(near_m: float) -> float:
    """The renderer's near clip for a robot whose depth reads from `near_m`: min(near_m, `NEAR_CLIP_MAX_M`), so the
    RGB is never cut where depth reads."""
    return min(near_m, NEAR_CLIP_MAX_M)


def cut_depth(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """Depth in metres with what the robot's sensor cannot read set to 0: nearer than `near_m` or farther than
    `far_m` (both bounds themselves are kept). Not-a-number and infinite values are left for `depth_to_units`, which
    zeroes them."""
    d = np.asarray(depth_m)
    return np.where((d < near_m) | (d > far_m), 0.0, d)


def self_filter(depth: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth and the instance index with both set to 0 where the index is `SELF_ID` (the robot's own body and
    payload)."""
    own = np.asarray(index) == SELF_ID
    return np.where(own, 0, depth), np.where(own, 0, index)


def depth_to_mm(depth_m: np.ndarray) -> np.ndarray:
    """`depth_to_units` in millimetres: what every depth image of the apartment and the warehouse holds."""
    return depth_to_units(depth_m, "mm")


def index_to_u16(index: np.ndarray) -> np.ndarray:
    """The float object-index pass -> uint16 pass_index. Raises if a value is not an integer in [0, 65535]."""
    a = np.asarray(index, dtype=np.float64)
    finite = np.isfinite(a)
    r = np.rint(np.where(finite, a, 0.0))
    bad = ~finite | (np.abs(a - r) > ID_TOLERANCE) | (r < 0) | (r > MAX_U16)
    if bad.any():
        raise ValueError(f"object-index pass holds {int(bad.sum())} values that are not integers in 0..{MAX_U16}, "
                         f"for example {a[bad][0]!r}")
    return r.astype(np.uint16)


def write_png16(path: str | Path, arr: np.ndarray) -> None:
    """A 16-bit greyscale PNG with no text or ancillary chunks."""
    from PIL import Image  # not at the top: Blender's own Python has no Pillow, and `render_episode` imports this

    if arr.dtype != np.uint16 or arr.ndim != 2:
        raise ValueError(f"write_png16 needs a 2-D uint16 array, got {arr.dtype} {arr.shape}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(p, format="PNG")


def read_png16(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        a = np.asarray(im)
    if a.dtype != np.uint16:
        raise ValueError(f"{path} is not a 16-bit PNG (got {a.dtype})")
    return a


def strip_jpeg(path: str | Path) -> None:
    """Rewrite a JPEG without its comment and APPn segments (JFIF's APP0 stays); the image data is untouched."""
    p = Path(path)
    data = p.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"{path} is not a JPEG")
    out, i = [data[:2]], 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError(f"{path}: bad JPEG segment at byte {i}")
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is entropy-coded data up to the end of the image
            out.append(data[i:])
            break
        n = int.from_bytes(data[i + 2:i + 4], "big")
        if not (marker == 0xFE or 0xE1 <= marker <= 0xEF):
            out.append(data[i:i + 2 + n])
        i += 2 + n
    p.write_bytes(b"".join(out))
