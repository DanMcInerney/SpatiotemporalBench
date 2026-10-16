"""One mover's walk: legs of dwell, waypoint and A*, advanced a keyframe at a time (`people` draws a resident's with it;
ported from the retired T3 movers, `redesign-base:src/stbench/mover_walk.py`).

A mover waits for a time drawn from its kind's `dwell`, picks a waypoint (in the room it stands in with probability
`same_room`, else in another room), follows the A* path there at one constant speed drawn from the kind's `speed`, and
starts again. `Walker.advance` moves a `Walk` state on by `dt` seconds; the state is a value, so a caller that finds the
new pose blocked keeps the old state (the mover pauses) and the walk resumes from it.

The walker knows nothing of the robot: a waypoint never depends on where the robot is, which would let the robot's own
pose predict a mover's place (a `pose_only` leak). Clearance from the robot and its camera is the caller's.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .catalog import MoverKind
from .geometry import Point
from .route import FreeSpace

MIN_LEG_M = 1.0  # a waypoint is at least this far from where the leg starts, if the room has such a cell
WAYPOINT_TRIES = 6


class Walk(NamedTuple):
    pos: Point
    yaw: float
    dwell: float  # seconds still to wait before the next leg
    path: tuple[Point, ...]  # the rest of the polyline after `pos`
    speed: float  # m/s on this leg


class Walker:
    """The walks of one mover kind on one free grid, from one rng."""

    def __init__(self, kind: MoverKind, free: FreeSpace, rng: np.random.Generator):
        self.kind, self.free, self.rng = kind, free, rng
        room = np.full(free.free.shape, -1)
        for i, r in enumerate(free.m.regions):
            room[free.polygon_mask(r.polygon)] = i
        reach = _largest_component(free)
        self.rx, self.ry = free.cells_xy(reach)  # the cells a walk may start or end in
        self.rroom = room[reach]
        self._room = room

    # -- starting --------------------------------------------------------------------------------------------
    def start(self, keepout: list[tuple[Point, float]]) -> Walk | None:
        """A walk that waits at a random reachable cell at least the given distance from each keepout point, facing
        a random way; None if no cell is far enough."""
        ok = np.ones(len(self.rx), dtype=bool)
        for (x, y), d in keepout:
            ok &= np.hypot(self.rx - x, self.ry - y) >= d
        cells = np.flatnonzero(ok)
        if not len(cells):
            return None
        i = int(cells[int(self.rng.integers(0, len(cells)))])
        return Walk((float(self.rx[i]), float(self.ry[i])), float(self.rng.uniform(-math.pi, math.pi)), 0.0, (), 0.0)

    # -- moving ----------------------------------------------------------------------------------------------
    def advance(self, st: Walk, dt: float) -> Walk:
        """The state `dt` seconds later: finish the dwell, follow the path, and draw a new leg whenever one ends."""
        pos, yaw, dwell, path, speed = st
        left = dt
        while left > 1e-9:
            if not path and dwell <= 0:
                dwell, path, speed = self._leg(pos)
            if dwell > 0:
                used = min(dwell, left)
                dwell, left = dwell - used, left - used
                continue
            nxt = path[0]
            seg = math.dist(pos, nxt)
            if seg > 0:
                yaw = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0])
            if speed * left < seg:
                f = speed * left / seg
                pos, left = (pos[0] + (nxt[0] - pos[0]) * f, pos[1] + (nxt[1] - pos[1]) * f), 0.0
            else:
                pos, left, path = nxt, left - seg / speed, path[1:]
        return Walk(pos, yaw, dwell, path, speed)

    def _leg(self, pos: Point) -> tuple[float, tuple[Point, ...], float]:
        """(dwell, path after `pos`, speed) of the next leg. If no waypoint is reachable the mover just waits."""
        dwell = float(self.rng.uniform(*self.kind.dwell))
        speed = float(self.rng.uniform(*self.kind.speed))
        for _ in range(WAYPOINT_TRIES):
            wp = self._waypoint(pos)
            path = self.free.path(pos, wp) if wp is not None else None
            if path is not None and len(path) > 1:
                return dwell, tuple(path[1:]), speed
        return max(dwell, 1.0), (), speed

    def _waypoint(self, pos: Point) -> Point | None:
        """A reachable cell: in the room the mover stands in with probability `same_room`, else in another room."""
        ix, iy = self.free.cell(pos)
        here = int(self._room[iy, ix])
        far = np.hypot(self.rx - pos[0], self.ry - pos[1]) >= MIN_LEG_M
        in_room = self.rroom >= 0  # a cell in a doorway belongs to no room and is never a waypoint
        want = (self.rroom == here) if self.rng.random() < self.kind.same_room else (self.rroom != here) & in_room
        for mask in (want & far, want, in_room & far, in_room):  # the wish first, then anything reachable
            cells = np.flatnonzero(mask)
            if len(cells):
                i = int(cells[int(self.rng.integers(0, len(cells)))])
                return (float(self.rx[i]), float(self.ry[i]))
        return None


def _largest_component(free: FreeSpace) -> np.ndarray:
    """The biggest set of free cells that connect to each other: a mover never starts in a pocket it cannot leave."""
    remaining, best = free.free.copy(), np.zeros_like(free.free)
    while remaining.any():
        iy, ix = (int(v) for v in np.argwhere(remaining)[0])
        comp = free.reachable_from(free.center(ix, iy))
        remaining &= ~comp
        if comp.sum() > best.sum():
            best = comp
    return best
