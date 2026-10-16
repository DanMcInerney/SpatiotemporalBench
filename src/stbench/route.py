"""The floor-plan planner's base: obstacles, the free-space grid and A*, and polyline resampling (V0 §4).

A job plans its robot's path on `FreeSpace`, a GRID_RES grid of the floor where the robot's base may stand: at least
`clearance` (the robot's own `body.clearance_m` plus its body's radius, or a world process's
`changes.WORLD_CLEARANCE_M`) from every wall piece and obstacle. `blocking_rects` lists the obstacles a path never
enters: blocking fixed furniture and the light furniture the caller passes in, less what a flier passes over
(`robots.flies_over`). `robot_obstacles` and `robot_free_space` give a ground robot's own: what `validate`'s body rule
keeps its base clear of, so the planner and the rule cannot disagree. The tour planner of the body-free camera is gone;
each job plans its own path (`job_*.py`).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .catalog import FLOOR_ITEM_Z
from .geometry import (
    Point,
    Rect,
    corners,
    dist_to_rect,
    distance,
    grid_in_polygon,
    nearest_on_rect,
    under_obstacles,
)
from .mapspec import MapSpec
from .placement import occupied_entry
from .robots import Robot, flies_over, passes_under
from .validate import FLOOR_COVERING_HEIGHT, door_leaf_rect

if TYPE_CHECKING:
    from .changes import World

GRID_RES = 0.1  # the default cell side
ROBOT_GRID_RES = 0.025  # the cell side of a ground robot's own free space (`robot_free_space`)
MIN_STEP = 0.05  # a path never ends with a hop shorter than this
Pose = tuple[Point, float]  # floor-plan position, yaw


def blocking_rects(m: MapSpec, light: dict[str, Rect] | None = None, over: float | None = None) -> list[Rect]:
    """Obstacles for a robot's path and for furniture shifts: blocking fixed furniture plus the given light furniture.

    A flier passes over what is lower than `over` (`robots.over_m`, `robots.flies_over`): only fixed furniture whose top
    is at `over` or above obstructs. The caller hands in just the light pieces that do (it knows their heights, `light`
    is footprints only). `over` None keeps every blocking piece, as a ground robot does."""
    rects = [f.footprint() for f in m.furniture
             if f.mobility == "fixed" and f.blocks and not flies_over(over, f.size[2])]
    return rects + list((light or {}).values())


def robot_obstacles(m: MapSpec, w: World, robot: Robot, t: float = 0.0) -> list[Rect]:
    """What a ground robot's base keeps clear of besides the walls (`FreeSpace` adds those): what `validate`'s body rule
    reads (`validate_ego`'s docstring), in the world as `w` holds it, with each door leaf in its state at `t`.
    - fixed and light pieces: their four leg squares when they have an open underside the robot passes under
      (`robots.passes_under`), else the whole footprint; floor coverings (rugs, mats) never. The dock is one: the
      scripted undock and dock leave and reach it, and the planner starts and ends in front of it;
    - every door leaf, open or shut;
    - carried items and clutter on the floor (base under `FLOOR_ITEM_Z`) taller than the robot's `climb_m`."""
    def parts(rect: Rect, under_m: float) -> list[Rect]:
        return under_obstacles(rect, under_m) if under_m > 0 and passes_under(robot, under_m) else [rect]

    out: list[Rect] = []
    for f in m.furniture:
        if f.mobility == "fixed" and f.size[2] >= FLOOR_COVERING_HEIGHT:
            out += parts(f.footprint(), f.under_m)
    entity = {e.id: e for e in w.entities}
    out += [part for eid, rect in w.light_rects.items() for part in parts(rect, entity[eid].under_m)]
    items = [occupied_entry(c.spec, c.spot.pos, c.spot.yaw) for c in w.carried.values()] + list(w.clutter.values())
    out += [rect for rect, z0, z1 in items if z0 < FLOOR_ITEM_Z and z1 - z0 > robot.body.climb_m]
    states = w.ledger.finish()[1]
    for d in m.doors():
        eid = w.door_entity[d.id]
        is_open = next((bool(s.value) for s in states if s.entity == eid and s.key == "open" and s.covers(t)), False)
        out.append(door_leaf_rect(m, d, is_open, entity[eid].size[1]))
    return out


def robot_free_space(m: MapSpec, w: World, robot: Robot, t: float = 0.0, shut: Iterable[str] = (),
                     keep_out: Iterable[Rect] = ()) -> FreeSpace:
    """`FreeSpace` for a ground robot, on a `ROBOT_GRID_RES` grid: its base keeps `body.clearance_m` from
    `robot_obstacles` and the walls. A free cell certifies only its centre, so the clearance also holds the cell's
    half-diagonal: every point of a free cell, not only its centre, keeps the body's clearance. That costs a passage up
    to a cell's diagonal on each side, which is why the grid is fine: at 0.1 m it closed the passage between an open
    door leaf and the kitchen counter (0.8 m wide, for a robot needing 0.38 m). A rectangular body is planned as its
    circumscribed disc.

    Each door of `shut` (opening ids, taken sorted) is an obstacle in both of its leaf positions, shut and open,
    whatever its state at `t`: no state of it can then change a path planned here. A job passes the doors off the ways
    of its task, so a path never detours round a door whose state it would leak (generic: on another map an off-way
    door may swing into a room the robot cleans). `shut=()` is the free space of `robot_obstacles` alone.

    Each rectangle of `keep_out` is an obstacle whatever the world holds there (a place the robot keeps out of, loaded
    or not: the AMR's staging lanes, so that no lane's load can change a path). `keep_out=()` adds nothing."""
    body = robot.body
    radius = body.length_m / 2 if body.shape == "disc" else math.hypot(body.length_m, body.width_m) / 2
    clearance = radius + body.clearance_m + ROBOT_GRID_RES * math.sqrt(2) / 2
    obstacles = robot_obstacles(m, w, robot, t)
    if shut:
        entity = {e.id: e for e in w.entities}
        doors = {d.id: d for d in m.doors()}
        unknown = sorted(set(shut) - doors.keys())
        if unknown:
            raise KeyError(f"map {m.name} has no door {unknown[0]!r} to shut")
        for door_id in sorted(set(shut)):
            thickness = entity[w.door_entity[door_id]].size[1]
            obstacles += [door_leaf_rect(m, doors[door_id], is_open, thickness) for is_open in (False, True)]
    obstacles += list(keep_out)
    return FreeSpace(m, obstacles, clearance, ROBOT_GRID_RES)


class FreeSpace:
    """Occupancy of the floor on a grid of `res` metres a cell (GRID_RES unless a caller asks for finer); a cell is free
    if the base may stand there (its centre is at least `clearance` from every wall piece and obstacle)."""

    def __init__(self, m: MapSpec, obstacles: list[Rect], clearance: float, res: float = GRID_RES):
        self.min_clearance, self.res = clearance, res
        x0, y0, x1, y1 = m.bounds
        self.x0, self.y0 = x0, y0
        self.nx, self.ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        self.xs, self.ys = x0 + (np.arange(self.nx) + 0.5) * res, y0 + (np.arange(self.ny) + 0.5) * res  # cell centres
        gx, gy = np.meshgrid(self.xs, self.ys)
        inside = np.zeros(gx.shape, dtype=bool)
        for r in m.regions:
            inside |= self.polygon_mask(r.polygon)
        for o in m.openings:
            inside |= dist_to_rect(gx, gy, o.rect(m.wall_thickness)) <= 0.0
        self.m = m
        self.rects = m.wall_pieces() + obstacles
        for rect in self.rects:  # only the cells within the clearance of the rectangle's box can fall short of it
            rx, ry = [c[0] for c in corners(rect)], [c[1] for c in corners(rect)]
            lo_x, hi_x = int((min(rx) - clearance - x0) / res) - 1, int((max(rx) + clearance - x0) / res) + 2
            lo_y, hi_y = int((min(ry) - clearance - y0) / res) - 1, int((max(ry) + clearance - y0) / res) + 2
            window = (slice(max(lo_y, 0), max(hi_y, 0)), slice(max(lo_x, 0), max(hi_x, 0)))
            inside[window] &= dist_to_rect(gx[window], gy[window], rect) >= clearance
        self.free = inside
        self._reached: dict[tuple[int, int], np.ndarray] = {}

    @cached_property
    def _columns(self) -> list[bool]:
        """`free` as one list, column after column (x major), in a grid a ring of blocked cells wider, which the search
        below reads faster than the array."""
        padded = np.zeros((self.nx + 2, self.ny + 2), dtype=bool)
        padded[1:-1, 1:-1] = self.free.T
        return padded.ravel().tolist()

    # -- cells -----------------------------------------------------------------------------------------------
    def cell(self, p: Point) -> tuple[int, int]:
        ix = min(max(int((p[0] - self.x0) / self.res), 0), self.nx - 1)
        iy = min(max(int((p[1] - self.y0) / self.res), 0), self.ny - 1)
        return ix, iy

    def center(self, ix: int, iy: int) -> Point:
        return (self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res)

    def cells_xy(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(mask)
        return self.x0 + (xs + 0.5) * self.res, self.y0 + (ys + 0.5) * self.res

    def polygon_mask(self, polygon: Iterable[Iterable[float]]) -> np.ndarray:
        """A mask over the grid of the cells whose centres lie in the polygon (free or not)."""
        return grid_in_polygon(self.xs, self.ys, [list(p) for p in polygon])

    def is_free(self, p: Point) -> bool:
        ix, iy = self.cell(p)
        return bool(self.free[iy, ix])

    def reachable_from(self, p: Point) -> np.ndarray:
        """Boolean mask of the free cells connected to p (8-neighbourhood, no corner cutting)."""
        ix, iy = self.cell(p)
        if (ix, iy) not in self._reached:
            self._reached[ix, iy] = self._flood(ix, iy)
        return self._reached[ix, iy].copy()

    @cached_property
    def _padded(self) -> np.ndarray:
        """`free` flat, in a grid a ring of blocked cells wider, so that a step from a free cell never leaves the array
        and a move of one cell along a row never wraps to the next."""
        padded = np.zeros((self.ny + 2, self.nx + 2), dtype=bool)
        padded[1:-1, 1:-1] = self.free
        return padded.ravel()

    def _flood(self, ix: int, iy: int) -> np.ndarray:
        """The free cells connected to cell (ix, iy) (8-neighbourhood, no corner cutting: a diagonal step needs both
        cells beside it free), a front at a time: the cells one step from the last front's, not seen before."""
        width, free = self.nx + 2, self._padded
        seen = np.zeros(free.shape, dtype=bool)
        if self.free[iy, ix]:
            start = (iy + 1) * width + ix + 1
            seen[start] = True
            front = np.array([start])
            while front.size:
                found = []
                for d in (1, -1, width, -width):
                    to = front + d
                    found.append(to[free[to]])
                for a, b in ((1, width), (1, -width), (-1, width), (-1, -width)):
                    to = front + (a + b)
                    found.append(to[free[to] & free[front + a] & free[front + b]])
                front = np.concatenate(found)
                front = np.unique(front[~seen[front]])
                seen[front] = True
        return seen.reshape(self.ny + 2, self.nx + 2)[1:-1, 1:-1].copy()

    def line_free(self, a: Point, b: Point) -> bool:
        """Whether a point every half cell along the segment, its ends included, lies in a free cell."""
        n = max(int(distance(a, b) / (self.res / 2)), 1)
        i = np.arange(n + 1)   # the points a + (b - a) * i / n, each cell as `cell` gives it, in the array's own floats
        ix = np.clip(((a[0] + (b[0] - a[0]) * i / n - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((a[1] + (b[1] - a[1]) * i / n - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return bool(self.free[iy, ix].all())

    def line_clear(self, a: Point, b: Point) -> bool:
        """The finer check: a point every tenth of a cell along the segment lies in a free cell. `line_free` samples
        every half cell, which can miss the corner of a blocked cell that a diagonal chord clips."""
        n = max(math.ceil(distance(a, b) / (self.res / 10)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        ix = np.clip(((a[0] + (b[0] - a[0]) * t - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((a[1] + (b[1] - a[1]) * t - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return bool(self.free[iy, ix].all())

    # -- clearance -------------------------------------------------------------------------------------------
    def clearance(self, p: Point) -> tuple[float, Point]:
        """Distance from p to the nearest wall or obstacle, and the unit vector pointing away from it."""
        best, away = math.inf, (0.0, 0.0)
        for r in self.rects:
            near, d = nearest_on_rect(r, p[0], p[1])
            if d < best:
                best = d
                away = ((p[0] - near[0]) / d, (p[1] - near[1]) / d) if d > 1e-12 else (0.0, 0.0)
        return best, away

    def nudge(self, p: Point, want: float | None = None) -> Point | None:
        """Move p away from geometry until it keeps `want` clearance (the grid's own clearance plus 1e-4 by default);
        None if that does not converge."""
        want = self.min_clearance + 1e-4 if want is None else want
        for _ in range(30):
            d, away = self.clearance(p)
            if d >= want:
                return p
            if away == (0.0, 0.0):
                return None
            p = (p[0] + away[0] * (want - d + 1e-4), p[1] + away[1] * (want - d + 1e-4))
        return p if self.clearance(p)[0] >= want else None

    # -- A* --------------------------------------------------------------------------------------------------
    def path(self, start: Point, goal: Point, strict: bool = False) -> list[Point] | None:
        """Shortest smoothed polyline from start to goal through free cells, or None if unreachable. With `strict` the
        smoothing keeps only chords `line_clear` passes, so every point of the path lies in a free cell."""
        s, g = self.cell(start), self.cell(goal)
        if not (self.free[s[1], s[0]] and self.free[g[1], g[0]]):
            return None
        # A cell (x, y) is the number (x + 1) * stride + y + 1 here: in the order of (x, y), which ties in the heap
        # break by, and on a grid with a ring of blocked cells round it, so a step needs no bounds test.
        stride, open_cells = self.ny + 2, self._columns
        moves = [(dx, dy, dx * stride + dy, math.sqrt(2.0) if dx and dy else 1.0)
                 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]
        first, last = (s[0] + 1) * stride + s[1] + 1, (g[0] + 1) * stride + g[1] + 1
        open_heap = [(0.0, 0.0, first)]
        best = {first: 0.0}
        parent: dict[int, int] = {}
        while open_heap:
            _, cost, cur = heapq.heappop(open_heap)
            if cur == last:
                break
            if cost > best.get(cur, math.inf):
                continue
            x, y = (v - 1 for v in divmod(cur, stride))
            for dx, dy, move, step in moves:
                nxt = cur + move
                if not open_cells[nxt] or (dx and dy and not (open_cells[cur + dx * stride] and open_cells[cur + dy])):
                    continue
                c = cost + step
                if c < best.get(nxt, math.inf):
                    best[nxt] = c
                    parent[nxt] = cur
                    heapq.heappush(open_heap, (c + math.hypot(x + dx - g[0], y + dy - g[1]), c, nxt))
        if last not in best:
            return None
        found = [last]
        while found[-1] != first:
            found.append(parent[found[-1]])
        found.reverse()
        return self._smooth([start] + [self.center(c // stride - 1, c % stride - 1) for c in found[1:-1]] + [goal],
                            strict)

    def _smooth(self, pts: list[Point], strict: bool = False) -> list[Point]:
        fits = self.line_clear if strict else self.line_free
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not fits(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out


def _resample(poly: list[Point], spacing: float) -> list[Pose]:
    """Points every `spacing` along the polyline (not the start), with the travel direction at each; the end last."""
    total = sum(distance(a, b) for a, b in zip(poly, poly[1:], strict=False))
    targets = [spacing * i for i in range(1, int(total / spacing) + 1)]
    if targets and total - targets[-1] < MIN_STEP:
        targets[-1] = total  # fold a sliver into the last hop rather than add a near-duplicate keyframe
    elif not targets or total - targets[-1] > 1e-6:
        targets.append(total)
    out: list[Pose] = []
    seg_start, seg = 0.0, 0
    for target in targets:
        while seg < len(poly) - 2 and target > seg_start + distance(poly[seg], poly[seg + 1]) + 1e-9:
            seg_start += distance(poly[seg], poly[seg + 1])
            seg += 1
        a, b = poly[seg], poly[seg + 1]
        length = distance(a, b)
        f = 0.0 if length == 0 else min(max((target - seg_start) / length, 0.0), 1.0)
        out.append(((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), math.atan2(b[1] - a[1], b[0] - a[0])))
    return out
