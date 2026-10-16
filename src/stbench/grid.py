"""The answer grid's conventions (the 0.5 m cells a `cells` answer names and the 0.1 m fine cells they are built from)
and the free-space rule that reads depth through them: what a sensor recorded of the floor, as fine-cell states, the
corridor a robot would drive and the swath a body covers.

A cell (i, j) is the half-open square [0.5i, 0.5i + 0.5) x [0.5j, 0.5j + 0.5) of the map frame, negative coordinates
included; a fine cell (u, v) is [0.1u, 0.1u + 0.1) x [0.1v, 0.1v + 0.1); a cell holds 25 fine cells, and a fine cell 25
sub-cells (a, b) of 0.02 m, in which an occupied fine cell says where its obstacle stands. The gold and the baselines'
mapper both call these functions, so they agree on every boundary and every rule. A claim that space is empty needs
space seen to be empty (VISION, truth rule 2): `observe` makes a fine cell `free` only from floor points the depth
image recorded, so a pixel of depth 0 (no surface, too near, past `far_m`, the robot's own body) certifies nothing. This
module is neutral (numpy, `camera` and `robots`): it reads no truth and no file.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import NamedTuple

import numpy as np

from .camera import Camera, cam_to_map, unproject
from .geometry import Rect
from .robots import Body, Robot, blind_m

FINE_M, CELL_M = 0.1, 0.5
FINE_PER_CELL = round(CELL_M / FINE_M)  # 5 fine cells along each side of a cell
SUB_M = 0.02  # the sub-cells an occupied fine cell is told in: the corridor's width is judged to this, not to 0.1 m
SUB_PER_FINE = round(FINE_M / SUB_M)  # 5 sub-cells along each side of a fine cell
EPS = 1e-9   # 0.3 / 0.1 is 2.9999999999999996 in floats: a decimal on a boundary lands in the cell it opens
MIN_FLOOR_PX = 4  # floor points a fine cell needs before it is `free`: fewer is a glimpse of the floor, not a view
FREE_SHARE = 0.8  # the share of a corridor's counted cells that must be `free` before `path_ahead` answers
# A depth image is millimetres and a pose is rounded, so a surface within this of the climb is floor to the free-space
# rule, for every robot: a rug as high as the helper's climb is driven over, a cable 4 mm over it is not.
DEPTH_QUANTUM_M = 0.003


def fine_cell(x: float, y: float) -> tuple[int, int]:
    """The fine cell (u, v) holding the point."""
    return (math.floor(x / FINE_M + EPS), math.floor(y / FINE_M + EPS))


def cell(x: float, y: float) -> tuple[int, int]:
    """The cell (i, j) holding the point: i = floor(x / 0.5), j = floor(y / 0.5)."""
    return (math.floor(x / CELL_M + EPS), math.floor(y / CELL_M + EPS))


def fine_cells_of(c: tuple[int, int]) -> list[tuple[int, int]]:
    """The 25 fine cells of cell `c`, row by row: v from the south edge up, and u from the west edge along each."""
    n = FINE_PER_CELL
    u0, v0 = n * c[0], n * c[1]
    return [(u0 + du, v0 + dv) for dv in range(n) for du in range(n)]


def _along_across(x: float, y: float, start: tuple[float, float], axis: tuple[float, float]) -> tuple[float, float]:
    """The point's distance along the axis (a unit vector) from `start`, and to its left."""
    dx, dy = x - start[0], y - start[1]
    return dx * axis[0] + dy * axis[1], dy * axis[0] - dx * axis[1]


def _in_corridor(x: float, y: float, start: tuple[float, float], axis: tuple[float, float], half: float, length: float
                 ) -> float | None:
    """The distance of the point along the corridor's axis (rounded to 1e-9, so that points level with each other tie
    whatever the yaw's float noise) when it lies in the rectangle that runs `length` from `start` along `axis` and is
    `2 * half` across it; else None. A point within 1e-9 of an edge is inside."""
    along, across = _along_across(x, y, start, axis)
    return round(along, 9) if -EPS <= along <= length + EPS and abs(across) <= half + EPS else None


def reach(shape: str, half: float, along: float, across: float, widen: float = 0.0) -> float | None:
    """How far a body whose front edge is at the origin must move straight along the axis before it first touches the
    point `along` ahead of the origin and `across` to its left, or None when it never does (the point lies more than
    `half + widen` from the axis). The body is a robot's `Body`: `half` is its half-width.
    - A `rect` sweeps its rectangle: the point is touched after `along`, whatever its `across`.
    - A `disc` of radius `half` (its centre `half` behind the origin) sweeps a disc: the point is touched when the
      disc's front arc reaches it, after `along + half - sqrt(half^2 - across^2)`; a point beside the arc, behind the
      origin, is touched after a longer move than its `along` says, and one at the disc's widest point after
      `along + half`. A point `widen` beyond the disc's side is taken at that widest point.
    A negative result is a point the body already holds (or is behind it): the callers drop it."""
    if abs(across) > half + widen + EPS:
        return None
    if shape == "disc":
        return along + half - math.sqrt(max(half * half - across * across, 0.0))
    return along


def _fine_box(start: tuple[float, float], axis: tuple[float, float], half: float, length: float
              ) -> tuple[range, range]:
    """The fine cells (u range, v range) of the box around the corridor's corners, a cell wider on every side."""
    corners = [(start[0] + t * axis[0] - s * axis[1], start[1] + t * axis[1] + s * axis[0])
               for t in (0.0, length) for s in (-half, half)]
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return (range(math.floor(min(xs) / FINE_M) - 1, math.floor(max(xs) / FINE_M) + 2),
            range(math.floor(min(ys) / FINE_M) - 1, math.floor(max(ys) / FINE_M) + 2))


def corridor(start_xy: tuple[float, float], yaw: float, width: float, length: float) -> list[tuple[int, int, float]]:
    """The fine cells whose centres lie in the rectangle that runs `length` metres from `start_xy` along `yaw` (radians,
    counter-clockwise from +x) and is `width` metres across it, centred on that axis. A centre within 1e-9 of an edge is
    inside. Each entry is (u, v, distance), the distance of the cell's centre along the axis from the start edge,
    rounded to 1e-9 so that cells level with each other tie whatever the yaw's float noise; sorted by distance, then u,
    then v. These are the cells that count as seen or unseen; what stops the body is judged finer (`path_ahead`)."""
    axis, half = (math.cos(yaw), math.sin(yaw)), width / 2  # the across direction is (-ay, ax)
    us, vs = _fine_box(start_xy, axis, half, length)
    found = [(d, u, v) for u in us for v in vs
             if (d := _in_corridor((u + 0.5) * FINE_M, (v + 0.5) * FINE_M, start_xy, axis, half, length)) is not None]
    found.sort()
    return [(u, v, d) for d, u, v in found]


# -- the free-space rule -----------------------------------------------------------------------------------------
def points(depth: np.ndarray, cam: Camera, cam_pos: Sequence[float], cam_rot: Sequence[float]) -> np.ndarray:
    """The (N, 3) map-frame points of a depth image: every pixel with depth > 0, at full resolution, unprojected at the
    pixel's centre (j + 0.5, i + 0.5) through its planar depth (`depth` is in counts of `cam.depth_unit`) and moved to
    the map by the camera's pose. A pixel of depth 0 adds no point."""
    rows, cols = np.nonzero(depth > 0)
    metres = depth[rows, cols].astype(np.float64) * cam.depth_scale
    return cam_to_map(unproject(cols + 0.5, rows + 0.5, metres, cam).T, cam_pos, cam_rot)


def band_of(climb_m: float, height_m: float, under_margin_m: float) -> tuple[float, float]:
    """`band`'s formula over a body's three sizes, for a reader that holds the sizes and not a `Robot` (the prompt's
    body sentence, which states the passing height `band` reads)."""
    return round(climb_m + DEPTH_QUANTUM_M, 9), round(height_m + under_margin_m / 2, 9)


def band(robot: Robot) -> tuple[float, float]:
    """(floor, top) of the heights, above the floor, that a point must lie in to be an obstacle to a ground robot:
    above what it drives over (its climb plus `DEPTH_QUANTUM_M`, the 3 mm a depth image cannot tell from the climb), and
    at most the body's height plus half its under-margin. A surface at exactly the passing height (the coffee table's
    0.10 m shelf, for the vacuum) lies above the top, so it is overhead for the evidence as it is for the planner,
    whatever the depth image's rounding. The vacuum's is (0.023, 0.095), the AMR's (0.013, 0.675)."""
    return band_of(robot.body.climb_m, robot.body.height_m, robot.body.under_margin_m)


class Occupied(str):
    """The state `occupied` (it equals the string "occupied") that also says where in its fine cell the obstacle stands:
    `subs` holds the 0.02 m sub-cells (a, b), [0.02a, 0.02a + 0.02) x [0.02b, 0.02b + 0.02), that hold an obstacle
    point. The mapper and the gold read the corridor's width from these, so a point outside the body's width never
    stops it however near its fine cell's centre lies. A bare "occupied" string, as a hand-built state is, stands for
    all 25 sub-cells of its cell."""

    subs: frozenset[tuple[int, int]]

    def __new__(cls, subs: Iterable[tuple[int, int]]) -> Occupied:
        obj = super().__new__(cls, "occupied")
        obj.subs = frozenset(subs)
        return obj

    def __getnewargs__(self) -> tuple:
        return (self.subs,)  # `str`'s own would rebuild it from the text alone


def sub_cells_of(c: tuple[int, int], state: str) -> Iterable[tuple[int, int]]:
    """The sub-cells an occupied state stands on in fine cell `c`: its own, or all 25 of the cell for a bare string."""
    if isinstance(state, Occupied):
        return state.subs
    n = SUB_PER_FINE
    return [(n * c[0] + da, n * c[1] + db) for db in range(n) for da in range(n)]


def observe(pts: np.ndarray, floor_z: float, band: tuple[float, float]) -> dict[tuple[int, int], str]:
    """The state of every fine cell the map-frame points of a depth image (`points`) say something about: the rule of
    `grid_observer.Observer.observe` (occupied over the band, free from floor points, else unobserved)."""
    from .grid_observer import Observer  # not at the top: it takes this module's constants

    return Observer(pts).observe(floor_z, band)


def latest(states_by_k: dict[int, dict]) -> dict[tuple[int, int], tuple[str, int]]:
    """Each fine cell's last observed state and the keyframe `k` that observed it, over `states_by_k` ({k: the cells
    `observe` gave at k}): the state at the highest k that has the cell."""
    out: dict[tuple[int, int], tuple[str, int]] = {}
    for k in sorted(states_by_k):
        out.update({c: (state, k) for c, state in states_by_k[k].items()})
    return out


class Stop(NamedTuple):
    """Where a corridor stops: `along`, how far the body can move from its start before it touches the nearest obstacle
    sub-cell (`reach`; to the sub-cell's centre), and `cell`, the fine cell that holds that sub-cell."""

    along: float
    cell: tuple[int, int]


def _behind(robot: Robot) -> float:
    """How far behind the front edge the body can touch something while it moves ahead: a disc's radius (its sides,
    beside the front arc), 0 for a box."""
    return robot.body.width_m / 2 if robot.body.shape == "disc" else 0.0


def stop(states: dict[tuple[int, int], tuple[str, int]], robot: Robot, start: tuple[float, float], heading: float,
         cap: float) -> Stop | None:
    """The corridor's stop: among the occupied fine cells, the obstacle sub-cell (`sub_cells_of`: 0.02 m) whose centre
    the body, moving from `start` (its front edge) along `heading`, touches first, within `cap`; None when it touches
    none. The body is the robot's (`robots.Body.shape`): a box sweeps its rectangle (`width_m` across), so the sub-cell
    must lie within the body's width, and the distance is its distance along the axis; a disc sweeps a disc (`reach`),
    so a sub-cell off the axis is touched later than its distance along says, by up to the radius at the disc's side,
    and one beside the front arc, behind `start`, is touched too. Only what stands within the body's width stops it (a
    door frame beside the doorway does not, however near its fine cell's centre lies). Ties go to the fine cell whose
    centre is nearer along, then the smaller u, then v, so float noise in the yaw never picks the cell."""
    axis, half, shape = (math.cos(heading), math.sin(heading)), robot.body.width_m / 2, robot.body.shape
    back = _behind(robot)
    best: tuple[float, float, int, int] | None = None
    us, vs = _fine_box((start[0] - back * axis[0], start[1] - back * axis[1]), axis, half, cap + back)
    for u in us:
        for v in vs:
            state = states.get((u, v), ("", 0))[0]
            if state != "occupied":
                continue
            level = round(_along_across((u + 0.5) * FINE_M, (v + 0.5) * FINE_M, start, axis)[0], 9)
            for a, b in sub_cells_of((u, v), state):
                along, across = _along_across((a + 0.5) * SUB_M, (b + 0.5) * SUB_M, start, axis)
                t = reach(shape, half, along, across)
                if t is None or not -EPS <= t <= cap + EPS:
                    continue
                if best is None or (round(t, 9), level, u, v) < best:
                    best = (round(t, 9), level, u, v)
    return None if best is None else Stop(best[0], (best[2], best[3]))


def front_edge(robot: Robot, base: Sequence[float], yaw: float) -> tuple[float, float]:
    """The point half the body's length from `base` (x, y) along `yaw` (radians): the body's front edge, never the
    camera, which sits behind it. With `yaw + pi` it is the rear edge."""
    return (base[0] + robot.body.length_m / 2 * math.cos(yaw), base[1] + robot.body.length_m / 2 * math.sin(yaw))


def corridor_clear(states: dict[tuple[int, int], tuple[str, int]], robot: Robot, start: tuple[float, float],
                   heading: float, cap: float, blind: float) -> tuple[dict, list[int]]:
    """How far a ground robot's body could move from `start` along `heading` (radians) before any part of it reached
    anything, as the one answer both the gold and a system's mapper give, from `states` (`latest`'s output) alone, and
    the keyframes that observed it. `path_ahead` is this from the front edge along the heading; `rear_clear` is this
    from the rear edge along the heading's opposite.

    The body moves straight and keeps the shape of the robot's `Body`: a box sweeps its rectangle (`width_m` across), a
    disc sweeps a disc (`reach`). What can stop it is an obstacle sub-cell (`observe`: 0.02 m) the body touches
    (`stop`), so only what stands within the body's width can. The answer is the distance the body moves before it
    touches the nearest such sub-cell's centre, less half a sub-cell, `cap` when there is none: the obstacle's own 2 cm,
    not its 0.1 m fine cell's near edge, which would put the answer a median 4 cm short. The cells that count as seen
    are the fine cells whose centres the body sweeps (`corridor`'s rectangle, less a disc's far corners, which it never
    sweeps), beyond `blind` (the floor the camera cannot see in front of the body) and swept before the body reaches the
    fine cell that holds the stopping sub-cell, less an occupied one: its obstacle stands outside the body's width,
    which says nothing of the corridor. When any count and fewer than `FREE_SHARE` of them are `free`, the corridor was
    not seen empty and the answer is `unknown`. The evidence is the sorted distinct keyframes that observed a counted
    cell or the stopping one. An obstacle is judged by its sub-cell's centre, so one within 1 cm of the body's side may
    fall either way (the screens of `corridor_screens` do not ask such a question)."""
    found = stop(states, robot, start, heading, cap)
    axis, half, shape = (math.cos(heading), math.sin(heading)), robot.body.width_m / 2, robot.body.shape
    value = cap if found is None else min(cap, round(max(0.0, found.along - SUB_M / 2), 3))
    limit = math.inf
    if found is not None:   # the stopping cell's centre, taken as far across as the disc's side at most
        along, across = _along_across((found.cell[0] + 0.5) * FINE_M, (found.cell[1] + 0.5) * FINE_M, start, axis)
        limit = round(reach(shape, half, along, across, widen=math.inf), 9)
    counted = []
    for u, v, d in corridor(start, heading, robot.body.width_m, cap):
        t = round(reach(shape, half, d, _along_across((u + 0.5) * FINE_M, (v + 0.5) * FINE_M, start, axis)[1]), 9)
        # an occupied cell nearer than the stopping one holds no obstacle in the body's width (else it would stop)
        if blind < d and t < limit and t <= cap + 2 * EPS and states.get((u, v), ("", 0))[0] != "occupied":
            counted.append((u, v))
    free = sum(states.get(c, ("", 0))[0] == "free" for c in counted)
    seen = {states[c][1] for c in counted if c in states}
    if found is not None:
        seen.add(states[found.cell][1])
    unseen = bool(counted) and free < FREE_SHARE * len(counted) - EPS
    return ({"type": "unknown"} if unseen else {"type": "number", "value": value}), sorted(seen)


def path_ahead(states: dict[tuple[int, int], tuple[str, int]], robot: Robot, base: Sequence[float], yaw: float,
               cap: float) -> tuple[dict, list[int]]:
    """How far a ground robot at `base` (x, y) heading `yaw` (radians) could move straight ahead: `corridor_clear` from
    the body's front edge (`front_edge`: `base` plus half the body's length along the heading, never the camera, which
    sits behind it) along the heading, the cells counted beyond `robots.blind_m`."""
    return corridor_clear(states, robot, front_edge(robot, base, yaw), yaw, cap, blind_m(robot))


def raycast(rects: Sequence[Rect], start: tuple[float, float], yaw: float, body: Body, cap: float) -> float:
    """How far a body could move from `start` along `yaw` before it touched a rect, `cap` when it touches none: the
    gold's rule (`corridor_clear`) over a map. The body moves straight and keeps `body`'s shape (a box sweeps its
    rectangle, a disc a disc: `reach`) and `start` is its front edge; a 0.02 m sub-cell is an obstacle's when its centre
    lies in a rect (a point within 1e-9 of an edge is in), and the answer is the distance the body moves before it
    touches the nearest such centre, less half a sub-cell, to the millimetre: `corridor_clear` over the cells a perfect
    sensor would make of the rects. What stands outside the body's width is not found, however near its fine cell's
    centre lies. Neutral, so a gold (the flier's wall corridor) and a map reader (`baselines_path_space`,
    `baselines_map`) both call it."""
    half, back = body.width_m / 2, body.width_m / 2 if body.shape == "disc" else 0.0
    near = [r for r in rects if math.dist((r.cx, r.cy), start) <= cap + body.width_m + math.hypot(r.sx, r.sy) / 2]
    if not near:
        return cap
    axis = (math.cos(yaw), math.sin(yaw))
    origin = (start[0] - back * axis[0], start[1] - back * axis[1])   # the disc's centre: its sides reach back
    corners = np.array([(origin[0] + t * axis[0] - s * axis[1], origin[1] + t * axis[1] + s * axis[0])
                        for t in (0.0, cap + back) for s in (-half, half)])
    lo = np.floor(corners.min(axis=0) / SUB_M).astype(int) - 1   # the sub-cells of the corridor's box, one wider
    hi = np.floor(corners.max(axis=0) / SUB_M).astype(int) + 2
    nearest = math.inf
    for r in near:   # only the sub-cells of a rect's box, within the corridor's, can lie in it: the rest are skipped
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        reach_x, reach_y = abs(c) * r.sx / 2 + abs(s) * r.sy / 2, abs(s) * r.sx / 2 + abs(c) * r.sy / 2
        a0, a1 = max(int(lo[0]), math.floor((r.cx - reach_x) / SUB_M) - 1), min(int(hi[0]), math.floor(
            (r.cx + reach_x) / SUB_M) + 2)
        b0, b1 = max(int(lo[1]), math.floor((r.cy - reach_y) / SUB_M) - 1), min(int(hi[1]), math.floor(
            (r.cy + reach_y) / SUB_M) + 2)
        if a0 >= a1 or b0 >= b1:
            continue
        a, b = (g.ravel() for g in np.meshgrid(np.arange(a0, a1), np.arange(b0, b1), indexing="ij"))
        x, y = (a + 0.5) * SUB_M, (b + 0.5) * SUB_M
        dx, dy = x - start[0], y - start[1]
        along, across = dx * axis[0] + dy * axis[1], dy * axis[0] - dx * axis[1]
        t = along + half - np.sqrt(np.maximum(half * half - across * across, 0.0)) if body.shape == "disc" else along
        stands = ((t >= -EPS) & (t <= cap + EPS) & (np.abs(across) <= half + EPS)
                  & (np.abs(c * (x - r.cx) + s * (y - r.cy)) <= r.sx / 2 + EPS)
                  & (np.abs(c * (y - r.cy) - s * (x - r.cx)) <= r.sy / 2 + EPS))
        if stands.any():
            nearest = min(nearest, float(np.round(t[stands], 9).min()))
    if nearest == math.inf:
        return cap
    return min(cap, round(max(0.0, nearest - SUB_M / 2), 3))


def place_free(states: dict[tuple[int, int], tuple[str, int]], xy: Sequence[float], radius: float,
               share: float = FREE_SHARE) -> bool:
    """Whether the place around `xy` was seen empty: of the fine cells whose centres lie within `radius` of it, at
    least `share` are `free` in `states` (`latest`'s output) and none is `occupied`. A place with no fine cell centre in
    reach is not seen empty. Cells that were not observed count against it, so a place nobody looked at is never empty
    (VISION, truth rule 2)."""
    us = range(math.floor((xy[0] - radius) / FINE_M) - 1, math.floor((xy[0] + radius) / FINE_M) + 2)
    vs = range(math.floor((xy[1] - radius) / FINE_M) - 1, math.floor((xy[1] + radius) / FINE_M) + 2)
    held = [states.get((u, v), ("", 0))[0] for u in us for v in vs
            if math.hypot((u + 0.5) * FINE_M - xy[0], (v + 0.5) * FINE_M - xy[1]) <= radius + EPS]
    return bool(held) and "occupied" not in held and held.count("free") >= share * len(held) - EPS


def _convex_hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The convex hull of the points, counter-clockwise (Andrew's monotone chain)."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def chain(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return chain(pts) + chain(reversed(pts))


def _centres(lo: tuple[float, float], hi: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fine cells whose centres can lie in the box lo..hi (a cell wider on every side): their u, v and centres."""
    us = np.arange(math.floor(lo[0] / FINE_M) - 1, math.floor(hi[0] / FINE_M) + 2)
    vs = np.arange(math.floor(lo[1] / FINE_M) - 1, math.floor(hi[1] / FINE_M) + 2)
    u, v = (a.ravel() for a in np.meshgrid(us, vs, indexing="ij"))
    return u, v, np.stack([(u + 0.5) * FINE_M, (v + 0.5) * FINE_M], axis=1)


def _cells_where(u: np.ndarray, v: np.ndarray, inside: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(u[inside].tolist(), v[inside].tolist(), strict=True))


def _disc_cells(a: tuple[float, float, float], b: tuple[float, float, float], radius: float) -> set[tuple[int, int]]:
    p, d = np.array(a[:2]), np.array(b[:2]) - np.array(a[:2])
    u, v, c = _centres((min(a[0], b[0]) - radius, min(a[1], b[1]) - radius),
                       (max(a[0], b[0]) + radius, max(a[1], b[1]) + radius))
    along = np.clip((c - p) @ d / (d @ d), 0.0, 1.0) if d @ d > 0 else np.zeros(len(c))
    return _cells_where(u, v, np.linalg.norm(c - (p + along[:, None] * d), axis=1) <= radius + EPS)


def _footprint(pose: tuple[float, float, float], length: float, width: float) -> list[tuple[float, float]]:
    """The rectangle's corners, counter-clockwise, centred on the pose's (x, y) and turned by its yaw."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * i - s * j, y + s * i + c * j)
            for i, j in ((-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2),
                         (-length / 2, width / 2))]


def _polygon_cells(poly: list[tuple[float, float]]) -> set[tuple[int, int]]:
    """The fine cells whose centres lie in the convex counter-clockwise polygon (a centre on an edge is in, to 1e-9)."""
    u, v, c = _centres((min(x for x, _ in poly), min(y for _, y in poly)),
                       (max(x for x, _ in poly), max(y for _, y in poly)))
    inside = np.ones(len(c), dtype=bool)
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True):
        inside &= (x1 - x0) * (c[:, 1] - y0) - (y1 - y0) * (c[:, 0] - x0) >= -EPS
    return _cells_where(u, v, inside)


def swath(poses: Sequence[tuple[float, float, float]], shape: str, length: float, width: float
          ) -> set[tuple[int, int]]:
    """The fine cells a footprint covers along the straight segments between consecutive (x, y, yaw) poses (yaw in
    radians; the footprint is centred on (x, y)); one pose covers its own footprint.
    - A `disc` (`length` its diameter) covers the cells whose centres lie within `length / 2` of a segment, so a turn in
      place adds nothing.
    - A `rect` (`length` along the heading, `width` across it) covers the cells inside the convex hull of its footprints
      at both ends of a segment when the yaw is unchanged, and inside either end's footprint when it changes. A turn
      under way is sampled only at its ends, so an AMR that turns is covered only roughly: M3 refines it.
    A cell whose centre lies on the boundary, to 1e-9, is covered."""
    if shape not in ("disc", "rect"):
        raise ValueError(f"a footprint is a 'disc' or a 'rect', not {shape!r}")
    out: set[tuple[int, int]] = set()
    for a, b in list(pairwise(poses)) or [(p, p) for p in poses]:  # one pose: its own footprint
        if shape == "disc":
            out |= _disc_cells(a, b, length / 2)
        elif abs(math.remainder(b[2] - a[2], math.tau)) <= EPS:
            out |= _polygon_cells(_convex_hull(_footprint(a, length, width) + _footprint(b, length, width)))
        else:
            out |= _polygon_cells(_footprint(a, length, width)) | _polygon_cells(_footprint(b, length, width))
    return out
