"""Dense places: where a pool of `position` and `positions` golds puts things most often, and which of them the camera
faced.

The prior systems (`blind`, `path_log`) answer a `position` or `positions` question from other scripts' golds. The fixed
furniture of a fixed map holds things in most scripts, so the best such answer is the few places that hold a thing in
many of them. `dense_places` finds them: greedily, the place with the most gold things within `match_radius` of it,
then those things are dropped and the next one is found, up to `MAX_PLACES`; each place is the mean of the things it
gathered. `Viewpoints` says whether a place was in the camera's view at some pose (`path_log` keeps only those: a place
the camera never faced cannot have held a thing it saw).

This module reads gold rows and poses only, so the baselines may import it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import numpy as np

from .camera import CV_FROM_BODY, Camera, rot_matrix
from .rules import MATCH_M, Z_RANGE

MAX_PLACES = 8  # the most places a candidate answer holds
HEIGHTS = (0.0, 0.5, 1.0)  # metres above a place's base that a thing standing there is looked for at


def gold_points(golds: Sequence[dict]) -> list[tuple[float, float, float]]:
    """(x, y, z) of the point of every `position` gold and of every item of the `positions` golds in `golds` (a gold
    row's `gold`); z is 0 for an `xy` point. Golds of other types (`unknown` among them) hold none."""
    out = []
    for g in golds:
        value = g["gold"]
        if value["type"] == "position":
            items = [value]
        elif value["type"] == "positions":
            items = value["items"]
        else:
            continue
        for item in items:
            c = item.get("xyz") or [*item["xy"], 0.0]
            out.append((float(c[0]), float(c[1]), float(c[2])))
    return out


def match_radius(golds: Sequence[dict]) -> float:
    """The median matching radius (`tolerances.match_m`, `rules.MATCH_M` where a gold has none) of the golds."""
    return statistics.median((g.get("tolerances") or {}).get("match_m", MATCH_M) for g in golds) if golds else MATCH_M


def dense_places(golds: Sequence[dict], k: int = MAX_PLACES) -> list[tuple[float, float, float]]:
    """Up to `k` dense places of the `positions` golds, densest first, each as (x, y, z) (the mean of its things).
    Distance is the floor-plane distance, as the scorer matches places."""
    pts = np.array(gold_points(golds), dtype=float).reshape(-1, 3)
    if not len(pts):
        return []
    near = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1]) <= match_radius(golds) + 1e-9
    alive, out = np.ones(len(pts), dtype=bool), []
    while alive.any() and len(out) < k:
        count = np.where(alive, (near & alive[None, :]).sum(axis=1), -1)
        members = near[int(np.argmax(count))] & alive
        out.append(tuple(float(v) for v in pts[members].mean(axis=0)))
        alive &= ~members
    return out


class Viewpoints:
    """The poses of one visit, and whether a place stood in the camera's view at any of them: some point from the
    place's base up to `HEIGHTS` projects into the image with camera z inside `rules.Z_RANGE` (no depth: a wall in
    front is not known). `poses` are `{pos, rot}` pose-log entries."""

    def __init__(self, poses: Sequence[dict], cam: Camera):
        self.cam = cam
        self._pos = np.array([p["pos"] for p in poses], dtype=float).reshape(-1, 3)
        self._cv = np.array([CV_FROM_BODY @ rot_matrix(p["rot"]).T for p in poses]).reshape(-1, 3, 3)

    def sees(self, place: Sequence[float]) -> bool:
        if not len(self._pos):
            return False
        points = np.array([[place[0], place[1], place[2] + dz] for dz in HEIGHTS])
        cp = np.einsum("nij,nhj->nhi", self._cv, points[None, :, :] - self._pos[:, None, :])
        z = cp[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = self.cam.fx * cp[..., 0] / z + self.cam.cx, self.cam.fy * cp[..., 1] / z + self.cam.cy
            ok = (z > Z_RANGE[0]) & (z < Z_RANGE[1]) & (u >= 0) & (u < self.cam.width) & (v >= 0) & (
                v < self.cam.height)
        return bool(ok.any())
