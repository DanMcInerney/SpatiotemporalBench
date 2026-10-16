"""The way a floor-cleaning robot covers a room: a pass along its edge, then parallel strips over the rest.

Both read a room's reachable free floor on a robot's `FreeSpace` (`route.robot_free_space`): the free cells whose
centres lie in the room's polygon and that the robot can drive to from `entry` (through any room: a pocket of a room
reached only by another room still counts). The paths they return are polylines whose every point lies in a free cell,
so the robot keeps its body's clearance everywhere on them, not only at keyframes. A job turns a polyline into
primitives (`keyframes`); this module names no robot, phase or kind and reads no timeline.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .geometry import Point, dist_to_segment, distance
from .mapspec import Region
from .route import FreeSpace

SIMPLIFY_M = 0.08  # the edge pass drops a vertex that lies this near the chord of its neighbours
EDGE_INSET_M = 0.05  # the edge pass runs this far inside the free floor, so it cuts no nook and rounds no staircase
MIN_RUN_M = 0.15  # a strip shorter than this is not worth turning into: its cells lie within reach of its neighbours
MIN_OVERLAP_M = 0.2  # runs on neighbouring strips chain when they overlap along the strips by this much
SAMPLE_M = 0.02  # strips are cut into runs on a line sampled this finely (and no coarser than a cell)
_CLOCKWISE = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))  # (dx, dy), y up


def room_floor(free: FreeSpace, room: Region) -> np.ndarray:
    """A mask over `free`'s grid of the free cells whose centres lie in the room's polygon."""
    return free.free & free.polygon_mask(room.polygon)


def reachable_floor(free: FreeSpace, room: Region, entry: Point) -> np.ndarray:
    """The room's reachable free floor: `room_floor` less what is not connected to the cell of `entry` through free
    cells. Empty when `entry` is not free."""
    return free.reachable_from(entry) & room_floor(free, room)


def perimeter(free: FreeSpace, room: Region, entry: Point) -> list[Point]:
    """The outer contour of the room's reachable free floor around `entry`, taken `EDGE_INSET_M` inside it, simplified,
    counter-clockwise, as a closed polyline that starts and ends at the contour's cell nearest `entry`. Furniture
    standing free of the walls leaves the contour alone (its edges are inner holes); a piece against a wall is part of
    the outer edge. A chord is kept only where `FreeSpace.line_clear` holds. Raises ValueError when the room has no
    reachable free floor at `entry`."""
    floor = reachable_floor(free, room, entry)
    if not floor[free.cell(entry)[1], free.cell(entry)[0]]:
        raise ValueError(f"{room.name}: the entry {entry} is not on the room's reachable free floor")
    mask = _erode(floor, math.ceil(EDGE_INSET_M / free.res))
    if not mask.any():
        mask = floor
    ys, xs = mask.nonzero()
    near = min(range(len(xs)), key=lambda i: distance(free.center(xs[i], ys[i]), entry))
    ring = [free.center(*c) for c in _outline(_blob(mask, (int(xs[near]), int(ys[near]))))]
    if _area(ring) < 0:
        ring.reverse()
    first = min(range(len(ring)), key=lambda i: distance(ring[i], entry))
    ring = ring[first:] + ring[:first]
    if len(ring) < 3:
        return [ring[0], ring[0]]
    far = max(range(len(ring)), key=lambda i: distance(ring[i], ring[0]))
    return _simplify(ring[:far + 1], free)[:-1] + _simplify(ring[far:] + [ring[0]], free)


def strips(free: FreeSpace, room: Region, entry: Point, spacing: float, start: Point,
           mask: np.ndarray | None = None) -> list[Point]:
    """The polyline from `start` (a free point: where the robot is) over the room's reachable free floor in strips
    `spacing` apart, parallel to the longer side of the room's bounding box. Each strip is cut into the runs of
    reachable free floor it crosses, and runs on neighbouring strips that overlap along them form a chain (an obstacle
    cuts the strips it stands on into a chain each side of it, and one under it). The robot cleans a chain at a time,
    back and forth: it enters at the nearest end of a chain's first or last strip, goes to the nearer end of each next
    run and along it, and `FreeSpace.path` joins the ends (a chain's neighbouring strips are a hop apart; the chains,
    the way across). The next chain is the one with the nearest end. Just `[start]` when no strip fits.

    `mask` (a boolean array over `free`'s grid) keeps only the runs that cross it, a run being kept whole when any of
    its samples is on the mask: a robot finishing a room covers the runs that still hold floor it has not swept, and
    no other. `None` keeps every run."""
    floor = reachable_floor(free, room, entry)
    xs, ys = [p[0] for p in room.polygon], [p[1] for p in room.polygon]
    along_x = max(xs) - min(xs) >= max(ys) - min(ys)
    lo, hi = (min(ys), max(ys)) if along_x else (min(xs), max(xs))
    a0, a1 = (min(xs), max(xs)) if along_x else (min(ys), max(ys))
    step = min(SAMPLE_M, free.res)  # finer than a cell, so consecutive free samples never skip a blocked one
    along = np.arange(a0, a1 + step / 2, step)
    lines = [lo + spacing * (i + 0.5) for i in range(math.ceil((hi - lo) / spacing - 1e-9))]
    chains = _chains([_runs(free, floor, along, c, along_x, mask) for c in lines])
    out = [start]
    while chains:
        ends = [(min(distance(out[-1], p) for p in (run.lo, run.hi)), i, forward)
                for i, chain in enumerate(chains) for forward, run in ((True, chain[0]), (False, chain[-1]))]
        _, i, forward = min(ends)
        for run in chains.pop(i)[:: 1 if forward else -1]:
            a, b = sorted((run.lo, run.hi), key=lambda p: distance(out[-1], p))
            out += _join(free, out[-1], a)[1:] + [b]
    return _dedupe(out)


# -- strips ------------------------------------------------------------------------------------------------------
class _Run(NamedTuple):
    u0: float  # where along the strip the run starts and ends
    u1: float
    lo: Point  # its two ends
    hi: Point


def _runs(free: FreeSpace, floor: np.ndarray, along: np.ndarray, c: float, along_x: bool,
          want: np.ndarray | None = None) -> list[_Run]:
    """The runs of the strip at offset `c`: each maximal stretch of samples on reachable free floor that is at least
    MIN_RUN_M long, from the smaller position, and (with `want`) holds a sample on it."""
    x, y = (along, np.full_like(along, c)) if along_x else (np.full_like(along, c), along)
    ix = np.clip(((x - free.x0) / free.res).astype(int), 0, free.nx - 1)
    iy = np.clip(((y - free.y0) / free.res).astype(int), 0, free.ny - 1)
    on = np.concatenate(([False], floor[iy, ix], [False]))
    edges = np.flatnonzero(on[1:] != on[:-1])
    spans = [(s, e) for s, e in zip(edges[::2], edges[1::2], strict=True)
             if want is None or want[iy[s:e], ix[s:e]].any()]
    ends = [(float(along[s]), float(along[e - 1])) for s, e in spans]
    return [_Run(u0, u1, (u0, c) if along_x else (c, u0), (u1, c) if along_x else (c, u1))
            for u0, u1 in ends if u1 - u0 >= MIN_RUN_M]


def _chains(lines: list[list[_Run]]) -> list[list[_Run]]:
    """The runs of consecutive strips grouped into chains, one run a strip: a run joins the chain of the previous strip
    whose last run overlaps it most (at least MIN_OVERLAP_M), else it starts a chain."""
    chains: list[list[_Run]] = []
    active: list[list[_Run]] = []  # the chains whose last run is on the previous strip
    for line in lines:
        taken: set[int] = set()
        extended: list[list[_Run]] = []
        for run in line:
            fits = [(min(c[-1].u1, run.u1) - max(c[-1].u0, run.u0), -i) for i, c in enumerate(active) if i not in taken]
            best = max(fits, default=(0.0, 0))
            if best[0] >= MIN_OVERLAP_M:
                taken.add(-best[1])
                active[-best[1]].append(run)
                extended.append(active[-best[1]])
            else:
                chains.append([run])
                extended.append(chains[-1])
        active = extended
    return chains


def _join(free: FreeSpace, a: Point, b: Point) -> list[Point]:
    """The way from `a` to `b` through free cells, `a` first and `b` last."""
    if distance(a, b) < 1e-9:
        return [a]
    way = free.path(a, b, strict=True)
    if way is None:
        raise ValueError(f"no way from {a} to {b} on the robot's free space")
    return way


def _dedupe(poly: list[Point]) -> list[Point]:
    return [p for i, p in enumerate(poly) if i == 0 or distance(p, poly[i - 1]) > 1e-9]


# -- the edge pass -----------------------------------------------------------------------------------------------
def _erode(mask: np.ndarray, cells: int) -> np.ndarray:
    """The mask less every cell within `cells` cells (a square around it) of one that is not in it."""
    pad, (h, w) = np.pad(mask, cells), mask.shape
    out = mask.copy()
    for dy in range(2 * cells + 1):
        for dx in range(2 * cells + 1):
            out &= pad[dy:dy + h, dx:dx + w]
    return out


def _blob(mask: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    """The 8-connected blob of `mask` holding `cell`."""
    seen = np.zeros_like(mask)
    seen[cell[1], cell[0]] = True
    todo = [cell]
    while todo:
        x, y = todo.pop()
        for dx, dy in _CLOCKWISE:
            nx, ny = x + dx, y + dy
            if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0] and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                todo.append((nx, ny))
    return seen


def _outline(blob: np.ndarray) -> list[tuple[int, int]]:
    """The cells of the blob's outer boundary in order (Moore-neighbour tracing from its lowest, leftmost cell, which
    is always on it). A one-cell-wide spur is walked out and back."""
    pad = np.pad(blob, 1)  # a background rim, so the tracing needs no bounds checks
    iy, ix = np.nonzero(pad)
    start = (int(ix[iy == iy.min()].min()), int(iy.min()))
    cur, back = start, 6  # `back` indexes _CLOCKWISE: the direction from `cur` to the background cell it came from
    out = [start]
    for _ in range(8 * pad.size):
        for i in range(1, 9):
            d = _CLOCKWISE[(back + i) % 8]
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if pad[nxt[1], nxt[0]]:
                break
        else:
            return [(start[0] - 1, start[1] - 1)]  # a single cell
        if cur == start and len(out) > 1 and nxt == out[1]:  # back at the start, about to repeat the first move
            out.pop()
            break
        bg = _CLOCKWISE[(back + i - 1) % 8]
        back = _CLOCKWISE.index((cur[0] + bg[0] - nxt[0], cur[1] + bg[1] - nxt[1]))
        cur = nxt
        out.append(cur)
    return [(x - 1, y - 1) for x, y in out]


def _area(ring: list[Point]) -> float:
    """The shoelace area of a closed ring: positive when counter-clockwise."""
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1], strict=True)) / 2


def _simplify(pts: list[Point], free: FreeSpace) -> list[Point]:
    """Douglas-Peucker over an open polyline, keeping a chord only where it is within SIMPLIFY_M of every point it
    replaces and `free.line_clear`; the ends stay."""
    a, b = pts[0], pts[-1]
    if len(pts) <= 2:
        return [a, b]
    d, i = max((dist_to_segment(p[0], p[1], a, b), i) for i, p in enumerate(pts[1:-1], 1))
    if d <= SIMPLIFY_M and free.line_clear(a, b):
        return [a, b]
    if d <= SIMPLIFY_M:
        i = len(pts) // 2  # near the chord, but the chord clips a blocked cell: split in the middle
    return _simplify(pts[:i + 1], free)[:-1] + _simplify(pts[i:], free)
