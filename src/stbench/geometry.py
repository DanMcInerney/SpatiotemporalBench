"""2D geometry in the map frame (metres, x east, y north, yaw counter-clockwise from +x), and the rotation helpers every
frame convention shares (quaternions `[w, x, y, z]`, the quantiser `q`). Stdlib plus numpy; no stbench imports."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

Point = tuple[float, float]
DECIMALS = 6
Quat = tuple[float, float, float, float]
# A piece on legs stands on four corner squares of this side. 0.15 m covers every leg in `render/furniture.py` (insets
# and thicknesses at most 0.06 m); `under_obstacles` is the one place that reads it.
LEG_SQUARE_M = 0.15


class Rect(NamedTuple):
    """An oriented rectangle: centre, full extents along its local x and y, yaw of local x."""

    cx: float
    cy: float
    sx: float
    sy: float
    yaw: float


def to_world(rect: Rect, lx: float, ly: float) -> Point:
    """Local coordinates of the rectangle (origin at its centre) to the map frame."""
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    return (rect.cx + c * lx - s * ly, rect.cy + s * lx + c * ly)


def to_local(rect: Rect, x: float, y: float) -> Point:
    """Map-frame point to the rectangle's local coordinates."""
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    dx, dy = x - rect.cx, y - rect.cy
    return (c * dx + s * dy, -s * dx + c * dy)


def corners(rect: Rect, shrink: float = 0.0) -> list[Point]:
    """The four corners, counter-clockwise, optionally pulled in by `shrink` on every side."""
    hx, hy = rect.sx / 2 - shrink, rect.sy / 2 - shrink
    return [to_world(rect, lx, ly) for lx, ly in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


def _axes(rect: Rect) -> list[Point]:
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    return [(c, s), (-s, c)]


def _project(pts: list[Point], axis: Point) -> tuple[float, float]:
    vals = [p[0] * axis[0] + p[1] * axis[1] for p in pts]
    return min(vals), max(vals)


def rects_overlap(a: Rect, b: Rect, margin: float = 0.0) -> bool:
    """True if the rectangles intersect or come within `margin` of each other along a separating axis.

    Rectangles that merely touch (within 1e-9) do not overlap.
    """
    pa, pb = corners(a), corners(b)
    for axis in _axes(a) + _axes(b):
        amin, amax = _project(pa, axis)
        bmin, bmax = _project(pb, axis)
        if amax + margin <= bmin + 1e-9 or bmax + margin <= amin + 1e-9:
            return False
    return True


def point_in_rect(rect: Rect, x: float, y: float, tol: float = 0.0) -> bool:
    lx, ly = to_local(rect, x, y)
    return abs(lx) <= rect.sx / 2 + tol and abs(ly) <= rect.sy / 2 + tol


def rect_inside_rect(inner: Rect, outer: Rect, tol: float = 1e-9) -> bool:
    return all(point_in_rect(outer, x, y, tol) for x, y in corners(inner))


def point_in_polygon(x: float, y: float, poly: list[Point] | list[list[float]]) -> bool:
    """Ray casting; points on the boundary may fall either way, so callers pull shapes in by a tolerance."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def dist_to_segment(x: float, y: float, a: Point, b: Point) -> float:
    """Distance from (x, y) to the closed segment a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else min(1.0, max(0.0, ((x - a[0]) * dx + (y - a[1]) * dy) / length2))
    return math.hypot(x - (a[0] + t * dx), y - (a[1] + t * dy))


def dist_to_polygon(x: float, y: float, poly: list[Point] | list[list[float]]) -> float:
    """0 if the point is inside the polygon (`point_in_polygon`), else its distance to the polygon's outline."""
    if point_in_polygon(x, y, poly):
        return 0.0
    return min(dist_to_segment(x, y, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: list[list[float]]) -> np.ndarray:
    """Vectorised ray casting over arrays of points."""
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > ys) != (y2 > ys)
        x_at = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= crosses & (xs < x_at)
    return inside


# `grid_in_polygon` tests the polygon's box grown by this much (a crossing's x is a few ulps off the edge's).
BOX_TOL_M = 1e-6


def grid_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: list[list[float]]) -> np.ndarray:
    """`points_in_polygon` over the grid whose columns sit at the ascending `xs` and rows at the ascending `ys`: the
    (len(ys), len(xs)) mask of the points inside, equal to testing the whole grid but tested only over the polygon's
    bounding box (grown by `BOX_TOL_M`). A point beyond the box is outside: above or below it no edge crosses its
    row, and to either side every edge that crosses the row passes the point on the same side, so the ray test toggles
    nothing or an even number of times (a closed ring crosses a row an even number of times)."""
    inside = np.zeros((len(ys), len(xs)), dtype=bool)
    lo, hi = np.min(poly, axis=0) - BOX_TOL_M, np.max(poly, axis=0) + BOX_TOL_M
    i0, i1 = np.searchsorted(xs, lo[0], side="left"), np.searchsorted(xs, hi[0], side="right")
    j0, j1 = np.searchsorted(ys, lo[1], side="left"), np.searchsorted(ys, hi[1], side="right")
    gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
    inside[j0:j1, i0:i1] = points_in_polygon(gx, gy, poly)
    return inside


def dist_to_rect(xs: np.ndarray, ys: np.ndarray, rect: Rect) -> np.ndarray:
    """Distance from each point to the rectangle (0 inside)."""
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    dx, dy = xs - rect.cx, ys - rect.cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    ex = np.maximum(np.abs(lx) - rect.sx / 2, 0.0)
    ey = np.maximum(np.abs(ly) - rect.sy / 2, 0.0)
    return np.hypot(ex, ey)


def nearest_on_rect(rect: Rect, x: float, y: float) -> tuple[Point, float]:
    """The point of the rectangle closest to (x, y), and the distance to it (0 inside)."""
    lx, ly = to_local(rect, x, y)
    px = min(max(lx, -rect.sx / 2), rect.sx / 2)
    py = min(max(ly, -rect.sy / 2), rect.sy / 2)
    return to_world(rect, px, py), math.hypot(lx - px, ly - py)


def _orient(p: Point, q: Point, r: Point) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if the closed segments p1-p2 and q1-q2 properly cross (touching ends do not count)."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rect_distance(a: Rect, b: Rect) -> float:
    """The least distance between two oriented rectangles; 0 when they overlap or touch."""
    if rects_overlap(a, b):
        return 0.0
    # Apart, the nearest points are a corner of one and an edge of the other.
    pa, pb = corners(a), corners(b)
    return min(dist_to_segment(x, y, other[i], other[(i + 1) % 4])
               for pts, other in ((pa, pb), (pb, pa)) for x, y in pts for i in range(4))


def under_obstacles(rect: Rect, under_m: float) -> list[Rect]:
    """The parts of a piece that obstruct at every height: [rect] when under_m <= 0 (solid to the floor); else the four
    LEG_SQUARE_M squares in its corners, inside it, at its yaw (a side shorter than a square gives a shorter leg). A
    caller adds the whole rect for a robot that cannot pass under (`robots.passes_under`)."""
    if under_m <= 0:
        return [rect]
    lx, ly = min(LEG_SQUARE_M, rect.sx), min(LEG_SQUARE_M, rect.sy)
    ox, oy = (rect.sx - lx) / 2, (rect.sy - ly) / 2
    return [Rect(*to_world(rect, sx * ox, sy * oy), lx, ly, rect.yaw) for sx in (1, -1) for sy in (1, -1)]


# -- rotations ---------------------------------------------------------------------------------------------------
def q(x: float) -> float:
    """Quantise a float to the precision stored in JSON, normalising -0.0, so values survive a save and load."""
    return round(float(x), DECIMALS) + 0.0


def quat_z(yaw: float) -> Quat:
    """Rotation about +z by yaw (radians, counter-clockwise), as [w, x, y, z]."""
    return (q(math.cos(yaw / 2)), 0.0, 0.0, q(math.sin(yaw / 2)))


def quat_yaw_pitch(yaw: float, pitch: float) -> Quat:
    """Camera body rotation: yaw about z, then pitch up about the body y axis; as [w, x, y, z]."""
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return (q(cy * cp), q(sy * sp), q(-cy * sp), q(sy * cp))


def forward(rot: Quat) -> tuple[float, float, float]:
    """The body x axis (where the entity or camera faces) in the map frame."""
    w, x, y, z = rot
    return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))


def yaw_of(rot: Quat) -> float:
    fx, fy, _ = forward(rot)
    return math.atan2(fy, fx)


def pitch_of(rot: Quat) -> float:
    return math.asin(max(-1.0, min(1.0, forward(rot)[2])))
