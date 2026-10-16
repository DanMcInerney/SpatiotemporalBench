"""The boxes `dressing_zones` and `dressing_robots` are made of: axis-aligned volumes in the
map frame, whether a box lies wholly inside the union of others, and a rectilinear map's open floor.

A box is `(xmin, ymin, zmin, xmax, ymax, zmax)`, as `render/dressing_spec.Dressing.box`. Where an axis-aligned box would
be too permissive, a wall, region or footprint that is not turned a quarter turn is an error: both maps are
rectilinear, and a map that is not needs this module extended.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .geometry import Rect, corners
from .mapspec import MapSpec

Box = tuple[float, float, float, float, float, float]  # xmin, ymin, zmin, xmax, ymax, zmax
Zone = tuple[str, Box]  # why, where
Rect2 = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax

WALL_PROUD_MAX_M = 0.12  # dressing on a wall stands at most this far from its face
FLUSH_M = 0.02  # a plate or strip this thin on the floor is flush: nothing tracked or the camera can meet it
EPS = 1e-9  # boxes that touch, or differ by less than this, share no volume


def overlaps(a: Box, b: Box, eps: float = EPS) -> bool:
    """Whether two boxes share volume (touching faces do not): the rule of `render/dressing_spec.overlaps`."""
    return all(a[i] < b[i + 3] - eps and b[i] < a[i + 3] - eps for i in range(3))


def _cuts(lo: float, hi: float, faces: Iterable[float]) -> list[float]:
    """The box's faces on one axis and the band faces inside it, in order, those closer than `EPS` merged."""
    out: list[float] = []
    for v in sorted({lo, hi, *(min(max(f, lo), hi) for f in faces)}):
        if not out or v - out[-1] > EPS:
            out.append(v)
    return out


def covered(box: Box, bands: Iterable[Box]) -> bool:
    """Whether the box lies wholly inside the union of the bands (it may cross from one band into the next: a cabinet
    run over several counters). Cuts the box at every band face inside it and asks of each piece whether a band holds
    it."""
    near = [b for b in bands if overlaps(box, b)]
    cuts = [_cuts(box[a], box[a + 3], (c for b in near for c in (b[a], b[a + 3]))) for a in range(3)]
    for ix in range(len(cuts[0]) - 1):
        for iy in range(len(cuts[1]) - 1):
            for iz in range(len(cuts[2]) - 1):
                mid = [(cuts[a][i] + cuts[a][i + 1]) / 2 for a, i in zip(range(3), (ix, iy, iz), strict=True)]
                if not any(all(b[a] - EPS <= mid[a] <= b[a + 3] + EPS for a in range(3)) for b in near):
                    return False
    return True


def aabb(rect: Rect) -> Rect2:
    pts = corners(rect)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _quarter(yaw: float) -> bool:
    return abs(math.sin(2 * yaw)) < 1e-6


def square(rect: Rect, what: str) -> Rect2:
    """The axis-aligned rectangle of a rectangle turned a quarter turn: the shapes whose box would be too permissive."""
    if not _quarter(rect.yaw):
        raise ValueError(f"{what} is turned {math.degrees(rect.yaw):.1f} degrees: a rectilinear map is needed")
    return aabb(rect)


def rooms(m: MapSpec) -> list[Rect2]:
    out = []
    for r in m.regions:
        xs, ys = [p[0] for p in r.polygon], [p[1] for p in r.polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if len(r.polygon) != 4 or any(abs(x - x0) > 1e-9 and abs(x - x1) > 1e-9 for x in xs) \
                or any(abs(y - y0) > 1e-9 and abs(y - y1) > 1e-9 for y in ys):
            raise ValueError(f"region {r.id} is not an axis-aligned rectangle: dressing_zones needs a rectilinear map")
        out.append((x0, y0, x1, y1))
    return out


def _cross(r: Rect2, d: float) -> list[Rect2]:
    """The rectangle grown by `d` along x and, separately, along y: a plus that sits inside the rectangle grown by `d`
    on every side (rounded corners), so the floor it leaves is a little too much, never too little."""
    return [(r[0] - d, r[1], r[2] + d, r[3]), (r[0], r[1] - d, r[2], r[3] + d)]


def open_floor(m: MapSpec, inset: float = WALL_PROUD_MAX_M) -> list[Rect2]:
    """The rooms' floor at least `inset` from every wall face and every blocking fixed footprint (the light furniture a
    room holds stands on it too, so it needs no entry), as disjoint rectangles. A room boundary that is no wall (the
    warehouse's four regions are one hall) leaves the floor whole."""
    solids = [square(r, "a wall piece") for r in m.wall_pieces()]
    solids += [square(f.footprint(), f"furniture {f.id}") for f in m.furniture if f.mobility == "fixed" and f.blocks]
    cuts = [c for s in solids for c in _cross(s, inset)]
    room_rects = rooms(m)
    allr = [*room_rects, *cuts]
    xs = np.unique(np.round([v for r in allr for v in (r[0], r[2])], 9))
    ys = np.unique(np.round([v for r in allr for v in (r[1], r[3])], 9))
    cx, cy = np.meshgrid((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)  # rows are y
    hold = np.zeros(cx.shape, dtype=bool)
    for r in room_rects:
        hold |= (cx > r[0]) & (cx < r[2]) & (cy > r[1]) & (cy < r[3])
    for r in cuts:
        hold &= ~((cx > r[0]) & (cx < r[2]) & (cy > r[1]) & (cy < r[3]))
    out: list[Rect2] = []
    live: dict[tuple[float, float], float] = {}  # x span -> where its run of rows began
    for iy in range(len(ys) - 1):
        spans, ix = [], 0
        while ix < len(xs) - 1:
            if hold[iy, ix]:
                j = ix
                while j + 1 < len(xs) - 1 and hold[iy, j + 1]:
                    j += 1
                spans.append((float(xs[ix]), float(xs[j + 1])))
                ix = j + 1
            else:
                ix += 1
        for span in [s for s in live if s not in spans]:
            out.append((span[0], live.pop(span), span[1], float(ys[iy])))
        for span in spans:
            live.setdefault(span, float(ys[iy]))
    out += [(s[0], y0, s[1], float(ys[-1])) for s, y0 in live.items()]
    return out
