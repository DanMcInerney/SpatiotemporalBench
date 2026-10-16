"""The corridor's run at every heading the robot could turn to in place (0.9.0, docs/families/open_heading.md): the
profile `open_heading`'s gold, the table systems and the baselines read.

`profile` is `grid.corridor_clear` at each heading of the circle, `step_deg` apart from the robot's own, from the turned
body's edge (`grid.front_edge` on that heading) over the same free-space states, the cells counted from the edge on
(`blind` 0: the body has turned to face them), so the run at a sampled heading is exactly the gold of a `clear_path`
corridor turned to that heading (the tests hold it to `grid.corridor_clear`, heading by heading). It is vectorised: the
looped function would read every cell of the 5 m corridor 360 times, and a profile is wanted at every look point of a
case. A fine cell or a 2 cm sub-cell only matters to the headings within the angle its distance leaves the body's half
width (`asin(half / distance)`), so each is paired with those headings, one bin wider on each side, and the pairs are
tested with `grid`'s own arithmetic: the same expressions in the same order over the same floats, which numpy evaluates
as Python does, so a run is the looped function's to the last digit (a tie between sub-cells is broken as `grid.stop`
breaks it: the nearest touched, then the fine cell level with the edge, then u, then v; the stopping cell fixes how far
the counted cells reach). A final rounding is Python's.

`None` is a heading whose corridor was not seen empty (`corridor_clear` says `unknown`): its run is unknown and a
chosen heading into it earns the run 0 (`score.regret`). It is neutral baseline-and-compile code: numpy, `grid`,
`robots`; it reads no truth and no file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from . import grid
from .robots import Robot

STEP_DEG = 1       # the gold's profile: a run for every degree of the circle
EDGE_BINS = 1   # the bins added to each side of a point's headings, so a float at a window's edge is still tried
BEST_BAND_M = grid.SUB_M   # a run this close to the longest is the longest: the corridor is told to the sub-cell
FOOTPRINT_TOL_M = 1e-6     # a sub-cell centre this near the footprint's edge is on it (the pose's rounding)


def bins_of(step_deg: float) -> int:
    """The headings of the circle at `step_deg` degrees: 360 / step. ValueError unless the step divides 360."""
    n = round(360 / step_deg)
    if n < 1 or abs(n * step_deg - 360) > 1e-9:
        raise ValueError(f"the headings step must divide 360 degrees, got {step_deg!r}")
    return n


def deg_of(i: int, step_deg: float) -> float:
    """The answer's heading, degrees to the left of the robot's own in (-180, 180], of profile index `i`."""
    deg = (i * step_deg) % 360
    return deg - 360 if deg > 180 else deg


def best_index(runs: Sequence[float | None]) -> int | None:
    """The best heading of a profile: among the headings whose run is within `BEST_BAND_M` of the longest, the one
    nearest the robot's own (the left one when two are as near); None when no heading was seen."""
    known = [r for r in runs if r is not None]
    if not known:
        return None
    n, longest = len(runs), max(known)
    near = [i for i, r in enumerate(runs) if r is not None and r >= longest - BEST_BAND_M - grid.EPS]
    return min(near, key=lambda i: (min(i, n - i), i))


def turn_swept(states: dict[tuple[int, int], tuple[str, int]], robot: Robot, base: Sequence[float], yaw: float) -> bool:
    """Whether a box body turning in place would hit something it saw: an obstacle sub-cell inside the disc of the
    body's half-diagonal around the base and outside the footprint at the delivered heading `yaw` (a corner would sweep
    it). A disc body sweeps nothing new."""
    body = robot.body
    if body.shape == "disc":
        return False
    radius = math.hypot(body.length_m / 2, body.width_m / 2)
    ax, ay = math.cos(yaw), math.sin(yaw)
    for u in range(math.floor((base[0] - radius) / grid.FINE_M), math.floor((base[0] + radius) / grid.FINE_M) + 1):
        for v in range(math.floor((base[1] - radius) / grid.FINE_M), math.floor((base[1] + radius) / grid.FINE_M) + 1):
            state = states.get((u, v), ("", 0))[0]
            if state != "occupied":
                continue
            for a, b in grid.sub_cells_of((u, v), state):
                x, y = (a + 0.5) * grid.SUB_M - base[0], (b + 0.5) * grid.SUB_M - base[1]
                along, across = x * ax + y * ay, y * ax - x * ay
                if (math.hypot(x, y) <= radius and (abs(along) > body.length_m / 2 + FOOTPRINT_TOL_M
                                                    or abs(across) > body.width_m / 2 + FOOTPRINT_TOL_M)):
                    return True
    return False


def _pairs(rho: np.ndarray, theta: np.ndarray, half: float, step: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(point index, heading index) of every heading that can reach each point: those within `asin(half / rho)` of
    the point's bearing `theta` (radians from the robot's heading), one bin wider each side; every heading for a point
    within `half` of the base. Each heading index is in [0, n)."""
    near = rho <= half + grid.EPS
    reach = np.arcsin(np.minimum(1.0, (half + grid.EPS) / np.maximum(rho, 1e-12)))
    first = np.floor((theta - reach) / step).astype(np.int64) - EDGE_BINS
    count = np.ceil((theta + reach) / step).astype(np.int64) + EDGE_BINS - first + 1
    wide = near | (count >= n)
    first[wide], count[wide] = 0, n
    point = np.repeat(np.arange(len(rho)), count)
    start = np.repeat(np.cumsum(count) - count, count)
    return point, (np.repeat(first, count) + np.arange(len(point)) - start) % n


def _reach(shape: str, half: float, along: np.ndarray, across: np.ndarray) -> np.ndarray:
    """`grid.reach` (widen 0 or infinite) over arrays, for points the caller has put within the body's width."""
    if shape == "disc":
        return along + half - np.sqrt(np.maximum(half * half - across * across, 0.0))
    return along


def profile(states: dict[tuple[int, int], tuple[str, int]], robot: Robot, base: Sequence[float], yaw: float,
            cap: float, step_deg: float = 1) -> list[float | None]:
    """The corridor's run, in metres, for each heading `yaw + radians(i * step_deg)` (i = 0 .. 360 / step_deg - 1: the
    robot's own, then turning left), from the body's edge at that heading, over `states` (`grid.latest`'s output), `cap`
    long: `grid.corridor_clear(states, robot, front_edge, heading, cap, 0.0)`'s number, None where it is `unknown`.
    `base` is the delivered base (x, y) and `yaw` its heading in radians. ValueError unless `step_deg` divides 360."""
    n = bins_of(step_deg)
    body = robot.body
    half, shape, bx, by = body.width_m / 2, body.shape, float(base[0]), float(base[1])
    heads = [yaw + math.radians(i * step_deg) for i in range(n)]
    ax, ay = np.array([math.cos(h) for h in heads]), np.array([math.sin(h) for h in heads])
    edges = [grid.front_edge(robot, base, h) for h in heads]
    sx, sy = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
    step = math.radians(step_deg)
    reach_m = math.hypot(cap + body.length_m / 2, half) + 2 * grid.FINE_M   # farthest a cell centre can matter

    # the fine cells around the base, their states, and the 2 cm sub-cells of the occupied ones
    u0, u1 = math.floor((bx - reach_m) / grid.FINE_M), math.floor((bx + reach_m) / grid.FINE_M)
    v0, v1 = math.floor((by - reach_m) / grid.FINE_M), math.floor((by + reach_m) / grid.FINE_M)
    u, v = (g.ravel() for g in np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1), indexing="ij"))
    cx, cy = (u + 0.5) * grid.FINE_M, (v + 0.5) * grid.FINE_M
    keep = np.hypot(cx - bx, cy - by) <= reach_m
    u, v, cx, cy = u[keep], v[keep], cx[keep], cy[keep]
    held = [states.get(c, ("", 0))[0] for c in zip(u.tolist(), v.tolist(), strict=True)]
    free = np.array([s == "free" for s in held], dtype=bool)
    occupied = np.array([s == "occupied" for s in held], dtype=bool)
    parent, sa, sb = [], [], []
    for i in np.flatnonzero(occupied).tolist():
        subs = list(grid.sub_cells_of((int(u[i]), int(v[i])), held[i]))
        parent += [i] * len(subs)
        sa += [s[0] for s in subs]
        sb += [s[1] for s in subs]
    parent, sa, sb = np.array(parent, dtype=np.int64), np.array(sa, dtype=np.int64), np.array(sb, dtype=np.int64)
    px, py = (sa + 0.5) * grid.SUB_M, (sb + 0.5) * grid.SUB_M

    # the obstacle sub-cell each heading's body touches first (`grid.stop`)
    stop_t = np.full(n, np.nan)
    limit = np.full(n, math.inf)
    if len(parent):
        pt, hd = _pairs(np.hypot(px - bx, py - by), np.arctan2(py - by, px - bx) - yaw, half, step, n)
        dx, dy = px[pt] - sx[hd], py[pt] - sy[hd]
        along, across = dx * ax[hd] + dy * ay[hd], dy * ax[hd] - dx * ay[hd]
        t = _reach(shape, half, along, across)
        ok = (np.abs(across) <= half + grid.EPS) & (t >= -grid.EPS) & (t <= cap + grid.EPS)
        pt, hd, t = pt[ok], hd[ok], t[ok]
        if len(pt):
            cell = parent[pt]
            cdx, cdy = cx[cell] - sx[hd], cy[cell] - sy[hd]
            level = np.round(cdx * ax[hd] + cdy * ay[hd], 9)
            order = np.lexsort((v[cell], u[cell], level, np.round(t, 9), hd))
            heading, first = np.unique(hd[order], return_index=True)
            win = order[first]
            stop_t[heading] = t[win]
            wc = cell[win]
            wdx, wdy = cx[wc] - sx[heading], cy[wc] - sy[heading]
            walong, wacross = wdx * ax[heading] + wdy * ay[heading], wdy * ax[heading] - wdx * ay[heading]
            limit[heading] = np.round(_reach(shape, half, walong, wacross), 9)

    # the cells the body sweeps before it reaches that cell: seen free, or the corridor was not seen empty
    pt, hd = _pairs(np.hypot(cx - bx, cy - by), np.arctan2(cy - by, cx - bx) - yaw, half, step, n)
    dx, dy = cx[pt] - sx[hd], cy[pt] - sy[hd]
    along, across = dx * ax[hd] + dy * ay[hd], dy * ax[hd] - dx * ay[hd]
    inside = (along >= -grid.EPS) & (along <= cap + grid.EPS) & (np.abs(across) <= half + grid.EPS)
    pt, hd, along, across = pt[inside], hd[inside], along[inside], across[inside]
    d = np.round(along, 9)
    t = np.round(_reach(shape, half, d, across), 9)
    counted = (d > 0.0) & (t < limit[hd]) & (t <= cap + 2 * grid.EPS) & ~occupied[pt]
    seen = np.bincount(hd[counted & free[pt]], minlength=n)
    total = np.bincount(hd[counted], minlength=n)

    out: list[float | None] = []
    for i in range(n):
        if total[i] and seen[i] < grid.FREE_SHARE * total[i] - grid.EPS:
            out.append(None)
        elif math.isnan(stop_t[i]):
            out.append(cap)
        else:
            out.append(min(cap, round(max(0.0, round(float(stop_t[i]), 9) - grid.SUB_M / 2), 3)))
    return out
