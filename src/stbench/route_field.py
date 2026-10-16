"""The way to a remembered thing (0.11, docs/families/way_to.md): the shortest route through floor the robot has seen,
and how much nearer one `reach_m` of driving along each heading brings it. The gold of `way_to` (`questions_way`) and
the table systems' answer (`table_way`) both call `route`, so the oracle reproduces the gold.

**Walkable floor.** The 0.1 m fine cells whose latest state is `free` (`grid.latest`, the states `clear_path` reads),
plus the cells the body swept between its delivered poses (driven space is known free: `grid.swath`, cut at a task's
end, where the robot is put elsewhere), less any cell last seen `occupied`; eroded by half the body's width, so a cell
is walkable only if every cell the body would touch standing on its centre is known too (a route is never claimed
through a gap the body cannot pass, at the price of a cell); plus the cells of the base's own centre line, where it
stood. A cell never seen is not walkable (VISION, truth rule 2: a claim that space is empty needs space seen to be
empty).

**The field.** `geo(c)` is the 8-connected Dijkstra distance in metres over walkable cells from the goal set: the
walkable cells within `goal_m` of the thing's middle. A diagonal step needs both of the cells it passes beside.

**The outcomes** (`profile_m` of the gold row), one a heading at `step_deg`: `clear_profile.profile` with the swept
cells counted seen free and the corridor `max(reach_m, room_m)` long gives the heading's run. A heading with a run under
`room_m` (the body's length plus the job's clearance: no room to move) or that ends on floor with no route to the goal
is null. Else, with `m = min(run, reach_m)`, the outcome is `reach_m * (geo(base) - geo(base + m u)) / m` metres: how
much nearer one `reach_m` of driving along it brings the robot. A run that ends against an obstacle ends where the body
touches it, which is not walkable floor (the erosion), so `geo` there is `Field.near`: the nearest walkable cell's value
and the distance to it (within half the body's width and a fine cell). A route that does not exist (the base cannot
reach the goal set through walkable floor, or it is nearer than `reach_m`), or that no heading takes a step along, is
`None`: the question is not asked.

**The reach bound** (0.11, final review S5). A drive of `m` metres cannot bring the robot nearer than the field's own
metric lets it: a path of the 8-connected field is at most `LATTICE` (sqrt(4 - 2 sqrt 2) = 1.0824, the octile
distance's worst excess over the straight line) times the straight distance between its ends, so over a straight drive
`geo(base) - geo(arrival)` is at most `LATTICE * (m + sqrt 2 FINE_M) + (LATTICE - 1) * slack`: the ends are snapped to
their cells (half a cell diagonal each) and the arrival's `near` value reaches `slack` past the walkable floor. An
outcome per `reach_m` above `max_outcome` is therefore no progress the body made: the eroded floor and the body's own
corridor disagree (a gap the erosion closed and the corridor fits, so the base's route length is the long way round it
came while the arrival is next to the goal), and the best heading of such a field is judged against a number no heading
can reach. The question is not asked there (`None`), as for any dead band. The bound is geometry and holds for every
robot and job; on the dev root (final review) it drops one of the 23 `way_to` golds, whose best outcome was 2.99 against
a bound of 1.44, the others lying between 0.94 and 1.21.

Neutral and numpy-only, like `clear_profile`: it reads no truth and no file.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from itertools import groupby, pairwise
from typing import NamedTuple

import numpy as np

from . import clear_profile, grid
from .robots import Robot

Cell = tuple[int, int]
States = dict[Cell, tuple[str, int]]
Pose = tuple[int, float, float, float]   # a delivered base pose: keyframe, x, y, yaw in radians
STEP_DEG = clear_profile.STEP_DEG
TIE = 1e-9
CENTRE_STEP_M = grid.FINE_M / 2   # the base's centre line is read at this spacing
LATTICE = math.sqrt(4 - 2 * math.sqrt(2))   # the octile distance's worst excess over the straight line: 1.0824


class Way(NamedTuple):
    """The answer of `route`: the outcome of every heading, the best heading's index, the base's route length to the
    goal set in metres (`geo` there) and the keyframes that certified the cells of the shortest route from the base."""

    profile: list[float | None]
    best: int
    distance_m: float
    ks: tuple[int, ...]


def _cell_of(x: float, y: float) -> Cell:
    return grid.fine_cell(x, y)


def runs_of(steps: Iterable[dict]) -> list[list[Pose]]:
    """The delivered base poses of `steps` (rows with `k`, `pose` and `status.task`, in order) as `route` takes them:
    one list a task, since the robot is put elsewhere between two."""
    return [[(s["k"], s["pose"]["pos"][0], s["pose"]["pos"][1], math.radians(s["pose"]["yaw_deg"])) for s in task]
            for _, task in groupby(steps, key=lambda s: s["status"]["task"])]


def swept(runs: Sequence[Sequence[Pose]], robot: Robot) -> tuple[dict[Cell, int], dict[Cell, int]]:
    """(the fine cells the body's footprint covered, the cells its base passed through), each with the first keyframe
    that did. `runs` are the delivered poses of each task in order: a straight segment joins consecutive poses of one
    run."""
    body, footprint, centre = robot.body, {}, {}
    for run in filter(None, runs):
        for n, (a, b) in enumerate(list(pairwise(run)) or [(run[0], run[0])]):
            if n and a[1:] == b[1:]:
                continue   # standing still adds nothing the previous segment did not cover
            for c in grid.swath([a[1:]] if a[1:] == b[1:] else [a[1:], b[1:]], body.shape, body.length_m,
                                body.width_m):
                footprint.setdefault(c, b[0])
            steps = max(1, math.ceil(math.dist(a[1:3], b[1:3]) / CENTRE_STEP_M))
            for i in range(steps + 1):
                t = i / steps
                centre.setdefault(_cell_of(a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])), b[0])
    return footprint, centre


def reach_cells(half: float) -> int:
    """How many fine cells out from a cell's own a body of half width `half` standing on its centre can touch."""
    return math.ceil(half / grid.FINE_M + 0.5)


def _erode(mask: np.ndarray, half: float) -> np.ndarray:
    """`mask` less every cell with a cell not in it that a body of half width `half` standing on its centre would touch
    (some part of the other cell lies within `half` of the centre), so a cell is kept only where the body fits however
    the floor under its edges is cut. The mask's border cells are False (the caller pads it), so a roll wraps in
    nothing."""
    n = reach_cells(half)
    out = mask.copy()
    for du in range(-n, n + 1):
        for dv in range(-n, n + 1):
            if math.hypot(max(abs(du) - 0.5, 0.0), max(abs(dv) - 0.5, 0.0)) * grid.FINE_M <= half + grid.EPS:
                out &= np.roll(mask, (du, dv), axis=(0, 1))
    return out


class Walk(NamedTuple):
    """The walkable fine cells: `mask[u - u0, v - v0]`."""

    u0: int
    v0: int
    mask: np.ndarray

    def has(self, c: Cell) -> bool:
        """Whether the fine cell is walkable (False outside the array)."""
        i, j = c[0] - self.u0, c[1] - self.v0
        return 0 <= i < self.mask.shape[0] and 0 <= j < self.mask.shape[1] and bool(self.mask[i, j])


def walkable(states: States, footprint: dict[Cell, int], centre: dict[Cell, int], half: float) -> Walk | None:
    """The walkable floor (the module docstring), None when nothing is known."""
    blocked = {c for c, (s, _) in states.items() if s == "occupied"}
    known = ({c for c, (s, _) in states.items() if s == "free"} | footprint.keys()) - blocked
    if not known and not centre:
        return None
    pad = reach_cells(half) + 1
    every = np.array(sorted(known | centre.keys()), dtype=np.int64)
    u0, v0 = (every.min(axis=0) - pad).tolist()
    mask = np.zeros(tuple((every.max(axis=0) + pad - (u0, v0) + 1).tolist()), dtype=bool)

    def put(cells) -> None:
        arr = np.array(sorted(cells), dtype=np.int64).reshape(-1, 2)
        mask[arr[:, 0] - u0, arr[:, 1] - v0] = True

    put(known)
    mask = _erode(mask, half)
    put(centre.keys() - blocked)
    return Walk(u0, v0, mask)


class Field(NamedTuple):
    """`geo` over the walkable cells (infinite elsewhere): `geo[u - u0, v - v0]`, metres from the goal set. `near` is
    `geo` taken out to the floor the body touches but cannot stand on: each cell's least `geo + distance` over the cells
    within the slack of `distances`, so the point where a heading's run ends, against an obstacle, has a value."""

    u0: int
    v0: int
    geo: np.ndarray
    near: np.ndarray

    def _at(self, field: np.ndarray, c: Cell) -> float:
        i, j = c[0] - self.u0, c[1] - self.v0
        return float(field[i, j]) if 0 <= i < field.shape[0] and 0 <= j < field.shape[1] else math.inf

    def at_cell(self, c: Cell) -> float:
        """`geo` at the fine cell; infinite outside the walkable floor."""
        return self._at(self.geo, c)

    def at(self, x: float, y: float) -> float:
        """`geo` at the fine cell holding the point."""
        return self.at_cell(_cell_of(x, y))

    def near_at(self, x: float, y: float) -> float:
        """`near` at the fine cell holding the point."""
        return self._at(self.near, _cell_of(x, y))


def _moves(width: int) -> list[tuple[int, float, tuple[int, int] | None]]:
    """The eight moves over a row-major array `width` wide: (flat offset, length, the two cells a diagonal passes
    by)."""
    side, diag = grid.FINE_M, grid.FINE_M * math.sqrt(2)
    out: list[tuple[int, float, tuple[int, int] | None]] = [(1, side, None), (-1, side, None), (width, side, None),
                                                          (-width, side, None)]
    out += [(a + b, diag, (a, b)) for a in (width, -width) for b in (1, -1)]
    return out


def distances(walk: Walk, goal: np.ndarray, slack: float) -> Field:
    """The Dijkstra field over `walk` from the cells of the boolean array `goal` (same shape): 8-connected, a diagonal
    only between cells whose two neighbours it passes are walkable too; `near` reaches `slack` metres past the walkable
    floor."""
    ok = np.pad(walk.mask, 1)
    width = ok.shape[1]
    flat, moves = ok.ravel().tolist(), _moves(width)
    dist = [math.inf] * len(flat)
    heap = [(0.0, i) for i in np.flatnonzero(np.pad(goal & walk.mask, 1).ravel()).tolist()]
    for _, i in heap:
        dist[i] = 0.0
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for off, step, beside in moves:
            j = i + off
            if flat[j] and d + step < dist[j] and (beside is None or (flat[i + beside[0]] and flat[i + beside[1]])):
                dist[j] = d + step
                heapq.heappush(heap, (d + step, j))
    geo = np.array(dist).reshape(ok.shape)[1:-1, 1:-1]
    n = math.floor(slack / grid.FINE_M + grid.EPS)
    wide = np.pad(geo, n, constant_values=math.inf)   # the roll wraps in nothing
    near = wide.copy()
    for du in range(-n, n + 1):
        for dv in range(-n, n + 1):
            if 0 < (d := math.hypot(du, dv) * grid.FINE_M) <= slack + grid.EPS:
                near = np.minimum(near, np.roll(wide, (du, dv), axis=(0, 1)) + d)
    return Field(walk.u0, walk.v0, geo, near[n:near.shape[0] - n, n:near.shape[1] - n])


def goal_cells(walk: Walk, xy: tuple[float, float], radius: float) -> np.ndarray:
    """The boolean array over `walk` of its cells whose centres lie within `radius` of `xy`."""
    u = (np.arange(walk.mask.shape[0]) + walk.u0 + 0.5) * grid.FINE_M
    v = (np.arange(walk.mask.shape[1]) + walk.v0 + 0.5) * grid.FINE_M
    return np.hypot(u[:, None] - xy[0], v[None, :] - xy[1]) <= radius + grid.EPS


def descend(field: Field, walk: Walk, start: Cell) -> list[Cell]:
    """The cells of the shortest route from `start` down the field to the goal set: each step to the neighbour of least
    `geo` (the first of equals in `_moves`' order), under the field's own diagonal rule."""
    out, here = [start], start
    while (best := field.at_cell(here)) > 0.0:
        at = None
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            there = (here[0] + du, here[1] + dv)
            if du and dv and not (walk.has((here[0] + du, here[1])) and walk.has((here[0], here[1] + dv))):
                continue
            if field.at_cell(there) < best:
                best, at = field.at_cell(there), there
        if at is None:
            break
        out.append(at)
        here = at
    return out


def max_outcome(reach_m: float, room_m: float, slack: float) -> float:
    """The largest outcome a straight drive can make (the module docstring's reach bound): `reach_m` times the field's
    metric excess per metre driven plus the snapping of the two ends, over the shortest drive an outcome is told over
    (`min(reach_m, room_m)`); `slack` is how far `distances` reaches past the walkable floor."""
    least = min(reach_m, room_m)
    return reach_m * (LATTICE + (LATTICE * math.sqrt(2) * grid.FINE_M + (LATTICE - 1) * slack) / least)


def best_index(profile: Sequence[float | None]) -> int | None:
    """The best heading of a profile: among those within `TIE` of the greatest outcome the one nearest ahead (the left
    one on a tie); None when no heading has one."""
    known = [(v, i) for i, v in enumerate(profile) if v is not None]
    if not known:
        return None
    top, n = max(v for v, _ in known), len(profile)
    return min((i for v, i in known if v >= top - TIE), key=lambda i: (min(i, n - i), i))


def route(states: States, runs: Sequence[Sequence[Pose]], robot: Robot, base: Sequence[float], yaw: float,
          thing: tuple[float, float], goal_m: float, reach_m: float, room_m: float, step_deg: int = STEP_DEG
          ) -> Way | None:
    """The way from `base` (x, y; heading `yaw`, radians) to the walkable floor within `goal_m` of `thing`, over
    `states` (`grid.latest`'s output) and the delivered poses `runs` (by task) the body drove, as the module docstring
    says; `reach_m` is the driving an outcome is told over and `room_m` the room a heading needs. None when there is no
    question to ask."""
    footprint, centre = swept(runs, robot)
    walk = walkable(states, footprint, centre, robot.body.width_m / 2)
    if walk is None or not (goal := goal_cells(walk, thing, goal_m) & walk.mask).any():
        return None
    slack = robot.body.width_m / 2 + grid.FINE_M
    field = distances(walk, goal, slack)
    origin = field.at(base[0], base[1])
    if not math.isfinite(origin) or origin < reach_m - grid.EPS:
        return None
    seen = {**states, **{c: ("free", k) for c, k in footprint.items() if c not in states}}
    runs_m = clear_profile.profile(seen, robot, base, yaw, max(reach_m, room_m), step_deg)
    outcomes: list[float | None] = []
    for i, run in enumerate(runs_m):
        m = None if run is None or run < room_m - grid.EPS else min(run, reach_m)
        gone = math.inf
        if m is not None:
            h = yaw + math.radians(i * step_deg)
            gone = field.near_at(base[0] + m * math.cos(h), base[1] + m * math.sin(h))
        outcomes.append(None if m is None or not math.isfinite(gone) else round(reach_m * (origin - gone) / m, 4))
    best = best_index(outcomes)
    if best is None or not 0.0 < outcomes[best] <= max_outcome(reach_m, room_m, slack) + grid.EPS:
        return None
    cells = descend(field, walk, _cell_of(base[0], base[1]))
    swept_at = {**centre, **footprint}   # a cell no depth image saw free is certified by the pose that drove it
    ks = {states[c][1] if c in states else swept_at[c] for c in cells if c in states or c in swept_at}
    return Way(outcomes, best, round(origin, 4), tuple(sorted(ks)))
