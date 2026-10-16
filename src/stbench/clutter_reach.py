"""Keeping the floor connected under clutter (`clutter.place`) and under the world's later moves (`clutter.floor_spot`,
`clutter.move_floor`, light furniture's new places): a thing left on the floor must not cut a room in two for a robot
that is planned to drive there (`dressing_zones.PLANNED_ROBOTS`).

`route` is the one owner of where a robot can go: `route.robot_free_space` is the floor its planner walks, the cells
where the base keeps the robot's clearance from every wall and every obstacle `route.robot_obstacles` lists. A floor
thing taller than the robot climbs is such an obstacle, and one that fills the gap between a piece of furniture and a
wall can leave a pocket the robot can never reach, or a room it cannot enter. `FloorGuard` takes that free space of the
world as it stands, once per planned ground robot, and answers whether one more obstacle would split any of them: it
blocks the cells within the clearance of the thing (the rule `FreeSpace` itself applies to an obstacle), and the free
cells that touch the blocked patch must stay joined along it. Any path that crossed the patch can then go round it, so
no two other cells are cut apart. That is conservative (a detour that leaves the patch's edge does not count, so a gap
between two pieces of furniture is never plugged) and it never refuses a thing in the open.

`FreeSpace` steps to eight neighbours but never cuts a corner, which joins exactly the cells that four-neighbour steps
join (a diagonal step needs both cells beside it free, and those are a four-step way round).
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .catalog import FLOOR_ITEM_Z
from .changes import World
from .dressing_robots import PLANNED_ROBOTS
from .geometry import Rect, corners, dist_to_rect, under_obstacles
from .robots import ROBOTS, Robot, passes_under
from .route import FreeSpace, robot_free_space


def _grow8(a: np.ndarray) -> np.ndarray:
    """`a` and every cell beside or diagonal to one of its cells."""
    r = a.copy()
    r[:, 1:] |= a[:, :-1]
    r[:, :-1] |= a[:, 1:]
    g = r.copy()
    g[1:] |= r[:-1]
    g[:-1] |= r[1:]
    return g


def _joined(cells: set[tuple[int, int]]) -> bool:
    """Whether `cells` form one piece under four-neighbour steps."""
    start = next(iter(cells))
    seen, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if p in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return len(seen) == len(cells)


class _Space:
    """One robot's free cells, and what a new obstacle does to them. The cells are taken (25 ms) when the first thing
    that is an obstacle to this robot is asked about, from the world as it stands then: a thing that is not one (a mat,
    a sheet, for the vacuum) never builds them, and a thing that is one is asked about before it is put in the world."""

    def __init__(self, world: World, robot: Robot, t: float = 0.0):
        self.robot = robot
        self.world, self.t = world, t

    @cached_property
    def fs(self) -> FreeSpace:
        return robot_free_space(self.world.m, self.world, self.robot, self.t)

    def _blocked(self, box: Rect) -> tuple[slice, slice, np.ndarray] | None:
        """The window of cells that `box` would stop being free, as (rows, columns, mask); None if it blocks none."""
        fs, c = self.fs, self.fs.min_clearance
        xs, ys = [p[0] for p in corners(box)], [p[1] for p in corners(box)]
        ix0 = max(int((min(xs) - c - fs.x0) / fs.res), 0)
        ix1 = min(int((max(xs) + c - fs.x0) / fs.res) + 1, fs.nx)
        iy0 = max(int((min(ys) - c - fs.y0) / fs.res), 0)
        iy1 = min(int((max(ys) + c - fs.y0) / fs.res) + 1, fs.ny)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        gx, gy = np.meshgrid(fs.x0 + (np.arange(ix0, ix1) + 0.5) * fs.res,
                             fs.y0 + (np.arange(iy0, iy1) + 0.5) * fs.res)
        mask = fs.free[iy0:iy1, ix0:ix1] & (dist_to_rect(gx, gy, box) < c)
        return (slice(iy0, iy1), slice(ix0, ix1), mask) if mask.any() else None

    def _linked(self, rows: slice, cols: slice, blocked: np.ndarray) -> bool:
        """Whether the free cells that touch the blocked patch are still one piece along the patch's edge."""
        fs = self.fs
        y0, y1 = max(rows.start - 1, 0), min(rows.stop + 1, fs.ny)
        x0, x1 = max(cols.start - 1, 0), min(cols.stop + 1, fs.nx)
        patch = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        patch[rows.start - y0:rows.stop - y0, cols.start - x0:cols.stop - x0] = blocked
        beside = _grow8(patch) & ~patch  # the cells beside the patch: a ring round a lone blob
        ring = beside & fs.free[y0:y1, x0:x1]
        n = int(ring.sum())
        if n <= 1 or n == int(beside.sum()):
            return True  # nothing to join, or the whole ring is free and so joined
        return _joined({(int(x), int(y)) for y, x in zip(*np.nonzero(ring), strict=True)})

    def _obstacle(self, height: float, base: float = 0.0) -> bool:
        """Whether a thing from `base` to `base + height` on the floor is an obstacle to this robot, as
        `route.robot_obstacles` reads it: low enough to start on the floor and taller than its climb."""
        return base < FLOOR_ITEM_Z and height > self.robot.body.climb_m

    def allows(self, box: Rect, height: float) -> bool:
        if not self._obstacle(height) or (hit := self._blocked(box)) is None:
            return True
        return self._linked(*hit)

    def block(self, box: Rect, height: float) -> None:
        if self._obstacle(height) and (hit := self._blocked(box)) is not None:
            rows, cols, mask = hit
            self.fs.free[rows, cols] &= ~mask

    def allows_together(self, boxes: list[Rect], height: float) -> bool:
        """Whether boxes that stand together (a chair's legs) split nothing: each is allowed once the ones before it are
        blocked, which the free cells are given back from after."""
        if not self._obstacle(height):
            return True
        free = self.fs.free.copy()
        try:
            for box in boxes:
                if not self.allows(box, height):
                    return False
                self.block(box, height)
            return True
        finally:
            self.fs.free = free


class FloorGuard:
    """The planned ground robots' free cells of a world (`route.robot_free_space`, each door leaf in its state at `t`),
    for `clutter.place` and the world's processes between tasks: `allows` a thing of this footprint and height on the
    floor if it splits none of them, `block` it once placed."""

    def __init__(self, world: World, t: float = 0.0):
        robots = [ROBOTS[r] for r in PLANNED_ROBOTS.get(world.catalog.name, ())]
        self.spaces = [_Space(world, r, t) for r in robots if not r.flies]

    def allows(self, box: Rect, height: float) -> bool:
        return all(s.allows(box, height) for s in self.spaces)

    def allows_piece(self, box: Rect, height: float, under_m: float) -> bool:
        """Whether a piece of light furniture splits none of the floors: a robot that passes under it (`under_m` is the
        height of its open underside) meets its leg squares (`geometry.under_obstacles`), any other its whole
        footprint."""
        def parts(robot: Robot) -> list[Rect]:
            return under_obstacles(box, under_m) if under_m > 0 and passes_under(robot, under_m) else [box]

        return all(s.allows_together(parts(s.robot), height) for s in self.spaces)

    def block(self, box: Rect, height: float) -> None:
        for s in self.spaces:
            s.block(box, height)
