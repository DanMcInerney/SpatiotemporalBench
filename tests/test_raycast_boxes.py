"""`grid.raycast` looks only at the sub-cells of each rect's box, within the corridor's: the answer is the one the
whole corridor box gives, which is written out here the plain way."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stbench import grid
from stbench.geometry import Rect
from stbench.robots import Body

BOX = Body("rect", 0.4, 0.5, 0.3, 0.0, 0.1, 0.2)
DISC = Body("disc", 0.35, 0.35, 0.3, 0.0, 0.1, 0.2)


def whole_box(rects, start, yaw, body, cap) -> float:
    """The rule over every sub-cell of the corridor's bounding box at once, a rect after another into one mask."""
    sub, eps = grid.SUB_M, grid.EPS
    half, back = body.width_m / 2, body.width_m / 2 if body.shape == "disc" else 0.0
    axis = (math.cos(yaw), math.sin(yaw))
    origin = (start[0] - back * axis[0], start[1] - back * axis[1])
    corners = np.array([(origin[0] + t * axis[0] - s * axis[1], origin[1] + t * axis[1] + s * axis[0])
                        for t in (0.0, cap + back) for s in (-half, half)])
    lo = np.floor(corners.min(axis=0) / sub).astype(int) - 1
    hi = np.floor(corners.max(axis=0) / sub).astype(int) + 2
    a, b = (g.ravel() for g in np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij"))
    x, y = (a + 0.5) * sub, (b + 0.5) * sub
    dx, dy = x - start[0], y - start[1]
    along, across = dx * axis[0] + dy * axis[1], dy * axis[0] - dx * axis[1]
    t = along + half - np.sqrt(np.maximum(half * half - across * across, 0.0)) if body.shape == "disc" else along
    stands = (t >= -eps) & (t <= cap + eps) & (np.abs(across) <= half + eps)
    inside = np.zeros_like(stands)
    for r in rects:
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        inside |= ((np.abs(c * (x - r.cx) + s * (y - r.cy)) <= r.sx / 2 + eps)
                   & (np.abs(c * (y - r.cy) - s * (x - r.cx)) <= r.sy / 2 + eps))
    stands &= inside
    if not stands.any():
        return cap
    return min(cap, round(max(0.0, float(np.round(t[stands], 9).min()) - sub / 2), 3))


@pytest.mark.parametrize("seed", range(10))
def test_the_boxes_of_the_rects_give_the_corridor_boxs_answer(seed):
    """Contract: for random rects (thin walls and blocks, turned every way, some across the corridor, some beside it,
    some on its start or past its end), starts, headings and caps, for a box and a disc body, `raycast` returns what
    the rule gives over the whole corridor box: the same millimetre, `cap` when nothing is in reach. Regression caught:
    a rect's box taken too tight (its turned corner, or the sub-cell a tolerance puts a point in, lost), or a rect
    that the corridor's box clips left out."""
    rng = np.random.default_rng(seed)
    for _ in range(30):
        rects = [Rect(*rng.uniform(-6, 6, 2), *rng.uniform(0.05, 5.0, 2), rng.uniform(0, math.pi))
                 for _ in range(int(rng.integers(0, 6)))]
        rects += [Rect(float(rng.integers(-4, 4)), float(rng.integers(-4, 4)), 0.1, 3.0, 0.0)]   # an aligned wall
        start, yaw = tuple(rng.uniform(-3, 3, 2)), float(rng.uniform(-math.pi, math.pi))
        cap = float(rng.uniform(0.5, 9))
        for body in (BOX, DISC):
            assert grid.raycast(rects, start, yaw, body, cap) == whole_box(
                [r for r in rects if math.dist((r.cx, r.cy), start) <= cap + body.width_m + math.hypot(r.sx, r.sy) / 2],
                start, yaw, body, cap)
