"""The camera's ground frame (0.6.0): where a point of the map is from the camera, along the floor.

An `answers.Offset` is `(ahead, left)` in metres: ahead along the camera's heading and to its left, both measured on the
floor, whatever the height of the camera or the point. The heading is the base's `yaw_deg` (counter-clockwise from +x,
so +y is ahead at 90 degrees and left is positive): the camera sits on the base and looks the way the base heads, and a
drone's camera, which hangs below its base, shares the base's heading as well. The 3-D slant range `slant_m` is what the
distance credit's rungs scale with (`rules.relative_ladder`), and `band` names the range bands a report splits by.

Neutral and standard-library only: the compiler (gold), the scripted baselines and the report share one transform, so a
sign cannot differ between the question and its answerers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

NEAR_UNITS, FAR_UNITS = 6, 16   # a range under 6 place units is `near`, from 16 `far`, between `mid`
BANDS = ("near", "mid", "far")


def to_offset(point_xy: Sequence[float], cam_xy: Sequence[float], yaw_deg: float) -> tuple[float, float]:
    """`(ahead, left)`, in metres, of the map point `point_xy` from the camera at `cam_xy` heading `yaw_deg`."""
    dx, dy = point_xy[0] - cam_xy[0], point_xy[1] - cam_xy[1]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (dx * c + dy * s, -dx * s + dy * c)


def from_offset(offset: Sequence[float], cam_xy: Sequence[float], yaw_deg: float) -> tuple[float, float]:
    """The map point `(x, y)` at `offset` = `(ahead, left)` from the camera at `cam_xy` heading `yaw_deg`:
    `to_offset`'s inverse."""
    ahead, left = offset[0], offset[1]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (cam_xy[0] + ahead * c - left * s, cam_xy[1] + ahead * s + left * c)


def slant_m(cam_pos: Sequence[float], point: Sequence[float]) -> float:
    """The straight-line (3-D) distance in metres from the camera at `cam_pos` to `point`, both `(x, y, z)`."""
    return math.dist(cam_pos[:3], point[:3])


def band(r_m: float, unit_m: float) -> str:
    """The range band of a distance `r_m` for a place whose unit is `unit_m`: `near` under 6 units, `far` from 16, `mid`
    between. For reports only: no question or credit reads it."""
    if r_m < NEAR_UNITS * unit_m:
        return BANDS[0]
    return BANDS[2] if r_m >= FAR_UNITS * unit_m else BANDS[1]
