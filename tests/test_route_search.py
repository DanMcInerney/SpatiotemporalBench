"""`route.FreeSpace`'s searches (flood fill, A*, the half-cell line test) on random grids, against the searches written
out the plain way: cells as pairs, a generator of neighbours, a point at a time."""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
import pytest

from stbench.route import FreeSpace

STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def grid_of(free: np.ndarray, res: float = 0.1, x0: float = -0.35, y0: float = 1.2) -> FreeSpace:
    """A `FreeSpace` over the boolean array `free` (rows are y), with no map: only the searches read it."""
    out = object.__new__(FreeSpace)
    out.free, out.res, out.x0, out.y0 = free, res, x0, y0
    out.ny, out.nx = free.shape
    out._reached = {}
    return out


def random_grid(seed: int) -> FreeSpace:
    """Walls and pockets: about a third of the cells blocked, in blobs and lines, so that there are several pieces and
    diagonal gaps between blocked cells."""
    rng = np.random.default_rng(seed)
    ny, nx = int(rng.integers(6, 40)), int(rng.integers(6, 40))
    free = rng.random((ny, nx)) > rng.uniform(0.15, 0.45)
    free[rng.integers(0, ny), :] &= rng.random(nx) > 0.5
    free[:, rng.integers(0, nx)] &= rng.random(ny) > 0.5
    return grid_of(free)


def reached(g: FreeSpace, ix: int, iy: int) -> np.ndarray:
    """The cells connected to (ix, iy) by 8-neighbour moves that need both cells beside a diagonal one free, a cell at
    a time."""
    seen = np.zeros_like(g.free)
    if g.free[iy, ix]:
        seen[iy, ix] = True
        queue = deque([(ix, iy)])
        while queue:
            x, y = queue.popleft()
            for dx, dy in STEPS:
                a, b = x + dx, y + dy
                if not (0 <= a < g.nx and 0 <= b < g.ny) or not g.free[b, a] or seen[b, a]:
                    continue
                if dx and dy and not (g.free[y, a] and g.free[b, x]):
                    continue
                seen[b, a] = True
                queue.append((a, b))
    return seen


def cells_path(g: FreeSpace, s: tuple[int, int], t: tuple[int, int]) -> list[tuple[int, int]] | None:
    """A*'s cells from s to t as the search was first written: neighbours in `STEPS` order, the heap ordered by
    (cost plus distance to t, cost, cell), a cell kept at its first best cost."""
    heap, best, parent = [(0.0, 0.0, s)], {s: 0.0}, {}
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == t:
            break
        if cost > best.get(cur, math.inf):
            continue
        for dx, dy in STEPS:
            a, b = cur[0] + dx, cur[1] + dy
            if not (0 <= a < g.nx and 0 <= b < g.ny) or not g.free[b, a]:
                continue
            if dx and dy and not (g.free[cur[1], a] and g.free[b, cur[0]]):
                continue
            c = cost + (math.sqrt(2.0) if dx and dy else 1.0)
            if c < best.get((a, b), math.inf):
                best[a, b], parent[a, b] = c, cur
                heapq.heappush(heap, (c + math.hypot(a - t[0], b - t[1]), c, (a, b)))
    if t not in best:
        return None
    out = [t]
    while out[-1] != s:
        out.append(parent[out[-1]])
    return out[::-1]


def line_free(g: FreeSpace, a, b) -> bool:
    n = max(int(math.dist(a, b) / (g.res / 2)), 1)
    return all(g.is_free((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)) for i in range(n + 1))


@pytest.mark.parametrize("seed", range(12))
def test_the_flood_fill_is_the_pieces_of_the_free_cells(seed):
    """Contract: `reachable_from` is the set of free cells connected to the start by 8-neighbour moves, a diagonal one
    only past two free cells, whatever the grid; from a blocked cell it is empty; and the mask is the caller's to
    change. Hand check at the end: a free cell between two blocked ones on a diagonal does not join them. Regression
    caught: a front-at-a-time fill that leaks across a corner, wraps from the end of a row to the next, or drops a
    cell reached twice in one front."""
    g = random_grid(seed)
    for ix, iy in {(int(x), int(y)) for y, x in np.argwhere(np.ones_like(g.free))[::17]}:
        got = g.reachable_from(g.center(ix, iy))
        assert np.array_equal(got, reached(g, ix, iy))
        got[:] = ~got   # a copy: the next call is unaffected
        assert np.array_equal(g.reachable_from(g.center(ix, iy)), reached(g, ix, iy))
    corner = grid_of(np.array([[True, False], [False, True]]))
    assert corner.reachable_from(corner.center(0, 0)).sum() == 1


@pytest.mark.parametrize("seed", range(12))
def test_a_star_takes_the_same_cells_through_ties(seed):
    """Contract: `path` is the A* of the plain search, cell for cell: the same ties broken the same way (by cost, then
    by the cell as (x, y)), so the route a seed's world was planned on is the one it always was; the plain search's
    cells, smoothed as the planner smooths them, are the polyline `path` returns. The grids have many routes of equal
    cost. Regression caught: a re-numbering of the cells that orders ties by (y, x), or a diagonal that cuts a
    corner."""
    g = random_grid(100 + seed)
    rng = np.random.default_rng(seed)
    free_cells = np.argwhere(g.free)
    for _ in range(12):
        (sy, sx), (ty, tx) = (free_cells[int(rng.integers(0, len(free_cells)))] for _ in range(2))
        a, b = g.center(int(sx), int(sy)), g.center(int(tx), int(ty))
        want = cells_path(g, (int(sx), int(sy)), (int(tx), int(ty)))
        got = g.path(a, b)
        if want is None:
            assert got is None
            continue
        assert got is not None and got[0] == a and got[-1] == b
        plain = [a, *(g.center(*c) for c in want[1:-1]), b]   # smoothed as the planner smooths it
        assert got == g._smooth(plain)


@pytest.mark.parametrize("seed", range(12))
def test_the_half_cell_line_test_is_the_point_at_a_time_one(seed):
    """Contract: `line_free` agrees with the test written a point at a time (a point every half cell along the segment,
    ends included, each in a free cell, a point off the grid taken at the nearest cell), for segments inside the grid,
    off it, and of no length. Regression caught: a vectorised test that rounds a point to the next cell, or drops an
    end."""
    g = random_grid(200 + seed)
    rng = np.random.default_rng(seed)
    lo, hi = (g.x0 - 0.5, g.y0 - 0.5), (g.x0 + g.nx * g.res + 0.5, g.y0 + g.ny * g.res + 0.5)
    for _ in range(60):
        a = tuple(float(v) for v in rng.uniform(lo, hi))
        b = a if rng.random() < 0.1 else tuple(float(v) for v in rng.uniform(lo, hi))
        assert g.line_free(a, b) == line_free(g, a, b)
