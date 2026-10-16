"""The free-space rule over a depth image's points, once for many bands: `grid.observe`'s rule (the fine cells' states a
sensor's points certify, V0 §10, `grid`'s docstring) as an `Observer` that works out what does not depend on the floor
or the band once, and a `CellFilter` that restricts it to the cells a reader asks about.

A step's image is read under the robot's band and under every zone's own (`table.Mapper`), and the baselines run over
a case each read the same images, so what one image costs is paid once a band here and not once a point cloud.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .grid import EPS, MIN_FLOOR_PX, SUB_M, SUB_PER_FINE, Occupied


class CellFilter:
    """A set of fine cells as a boolean grid over their bounding box, to ask of many points at once which of them lie in
    a cell of the set (`Observer`'s `keep`)."""

    def __init__(self, cells: Iterable[tuple[int, int]]):
        found = np.array(sorted(cells), dtype=np.int64).reshape(-1, 2)
        self.lo = tuple(found.min(axis=0).tolist()) if len(found) else (0, 0)
        self.grid = np.zeros(tuple((found.max(axis=0) - found.min(axis=0) + 1).tolist()) if len(found) else (0, 0),
                             dtype=bool)
        self.grid[found[:, 0] - self.lo[0], found[:, 1] - self.lo[1]] = True

    def holds(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Which of the fine cells (u[i], v[i]) are in the set."""
        ru, rv = u - self.lo[0], v - self.lo[1]
        inside = np.flatnonzero((ru >= 0) & (ru < self.grid.shape[0]) & (rv >= 0) & (rv < self.grid.shape[1]))
        out = np.zeros(len(u), dtype=bool)
        out[inside] = self.grid[ru[inside], rv[inside]]
        return out


DENSE_MAX = 1 << 21   # the most cells of a bounding box `_distinct` counts into a plain array, else it sorts


def _distinct(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distinct pairs (u[i], v[i]) of two integer arrays in order (by u, then v) as two arrays, and how often each
    occurs. A small bounding box is counted into an array, a large one sorted: the same result."""
    u0, v0 = int(u.min()), int(v.min())
    span = int(v.max()) - v0 + 1
    area = (int(u.max()) - u0 + 1) * span
    key = (u - u0) * span + (v - v0)
    if area <= DENSE_MAX:
        counts = np.bincount(key, minlength=area)
        keys = np.flatnonzero(counts)
        counts = counts[keys]
    else:
        keys, counts = np.unique(key, return_counts=True)
    return keys // span + u0, keys % span + v0, counts


class Observer:
    """`observe`'s rule over one depth image's map-frame points `pts` ((N, 3), `points`'s output), with what does not
    depend on the floor or the band worked out once: each point's 0.02 m sub-cell (`su`, `sv`) and its fine cell (`cu`,
    `cv`). A step's image is read under the robot's band and under every zone's own (`table.Mapper`), each its own
    `observe` call."""

    def __init__(self, pts: np.ndarray):
        self.z = pts[:, 2]
        subs = np.floor(pts[:, :2] / SUB_M + EPS).astype(np.int64)
        self.su, self.sv = np.ascontiguousarray(subs[:, 0]), np.ascontiguousarray(subs[:, 1])
        self.cu, self.cv = self.su // SUB_PER_FINE, self.sv // SUB_PER_FINE   # a point's fine cell is its sub-cell's

    def within(self, keep: CellFilter) -> Observer:
        """The points that lie in a cell of `keep`, as an `Observer` of their own: what a reader of those cells alone
        (a zone) asks of many bands is then asked of these points and not of the whole image."""
        at = np.flatnonzero(keep.holds(self.cu, self.cv))
        out = object.__new__(Observer)
        out.z, out.su, out.sv, out.cu, out.cv = (a[at] for a in (self.z, self.su, self.sv, self.cu, self.cv))
        return out

    def observe(self, floor_z: float, band: tuple[float, float], keep: CellFilter | None = None
                ) -> dict[tuple[int, int], str]:
        """The state of every fine cell the points say something about: `occupied` (an `Occupied`, which also names the
        0.02 m sub-cells holding the obstacle's points) when any point's height above the floor is over `band[0]` and
        at most `band[1]`; else `free` when at least `MIN_FLOOR_PX` points are at or under `band[0]` (the floor); a cell
        with neither is absent (unobserved). A point above the band is overhead and ignored, so the floor under a table
        is seen between its legs; a cell with no floor point and no obstacle point (the shadow of a box, anything past
        the sensor's range) is not claimed either way. A point's cell is the fine cell of its sub-cell, `floor(x / 0.02
        + 1e-9)` along each axis, and a height within 1e-9 of a band edge is on it (the edge's decimal, as for the
        cells). The cells come in order, by u and then v. With `keep` only the cells in it are returned, with the states
        the whole dict gives them (a cell's state is a function of its own points), and the points of the others are not
        looked at: a zone's step never builds the tens of thousands of cells it does not read.

        Known limit: an overhang whose underside lies inside the band, seen only edge-on from the camera's own height
        (the sofa's base from the vacuum's 0.08 m), marks its front cells occupied and the floor under it free, so a
        corridor that meets it stops at the front, which is right."""
        climb, top = band
        height = self.z - floor_z
        held = height <= top + EPS
        if keep is not None:
            held &= keep.holds(self.cu, self.cv)
        low = height <= climb + EPS
        stands: dict[tuple[int, int], list[tuple[int, int]]] = {}
        if (hit := np.flatnonzero(held & ~low)).size:
            a, b, _ = _distinct(self.su[hit], self.sv[hit])
            for sub in zip(a.tolist(), b.tolist(), strict=True):
                stands.setdefault((sub[0] // SUB_PER_FINE, sub[1] // SUB_PER_FINE), []).append(sub)
        floor: list[tuple[int, int]] = []
        if (hit := np.flatnonzero(held & low)).size:
            u, v, n = _distinct(self.cu[hit], self.cv[hit])
            enough = n >= MIN_FLOOR_PX
            floor = [c for c in zip(u[enough].tolist(), v[enough].tolist(), strict=True) if c not in stands]
        return {c: Occupied(stands[c]) if c in stands else "free" for c in sorted([*stands, *floor])}
