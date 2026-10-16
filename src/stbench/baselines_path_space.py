"""`path_log`'s answers to the corridor families, `path_ahead` and `rear_clear` (a module of
`baselines_path.PATH_MODULES`).

Both are one corridor, from `grid.front_edge(robot, base, heading)` along the heading: ahead of the body for
`path_ahead`, behind it for `rear_clear` (the heading plus pi, so the start is the rear edge), the body's width wide and
`cap` long. From the base pose it was last delivered:
1. **Its own path.** The fine cells its body swept between consecutive delivered base poses (`grid.swath`) count as seen
   free, and a detour is evidence. Along the corridor's axis from its start, one cell every 0.1 m at s = 0.05, 0.15,
   ...: when the first is swept and a later one is swept again after an unswept one, the robot went around something,
   which stands at the first unswept cell: the answer is s - 0.05 there (no prior, no map).
2. **The raycast.** Else the distance along the corridor to the first 0.02 m sub-cell (`grid.SUB_M`) whose centre lies
   in the corridor's rectangle and in an obstacle of the public map, less half a sub-cell (`grid.raycast`, the gold's
   `grid.corridor_clear` over the cells a perfect sensor would make), so a wall 0.31 m off the AMR's axis (half-width
   0.29) does not stop it. The obstacles are the map's wall pieces, its fixed pieces that block and stand at
   least `FLOOR_COVERING_M` tall (the whole footprint, or the legs for a robot that passes under:
   `geometry.under_obstacles`, `robots.passes_under`) and a door's leaf. A floor marking (a staging lane, a bay: a few
   millimetres of paint) is under the robot's climb and never stops it, as for the gold. The answer is the best
   response over its pose-keyed pool (`answer_keys.levels("path_log", ...)`) with the raycast as an extra candidate, the
   raycast alone when the pool is empty. On a held-out layout it holds no map: no raycast, the pool's best response
   alone, `unknown` with no pool.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import grid
from .answer_candidates import UNKNOWN
from .blind import learned_answer, pool_at
from .geometry import Rect, under_obstacles
from .mapspec import MapSpec
from .robots import Robot, passes_under

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog

# A door leaf is 0.04 m thick: `generate.DOOR_THICKNESS`, which a baseline may not import (the generator is truth side);
# `tests/test_path_log.py` ties the two.
DOOR_LEAF_M = 0.04
# A fixed piece lower than this is a floor covering (a rug, a mat, a painted lane) and obstructs nothing:
# `validate.FLOOR_COVERING_HEIGHT`, which a baseline may not import; `tests/test_answers_rear.py` ties the two.
FLOOR_COVERING_M = 0.05
# Every interior door is open throughout M1, as every script shows; an exterior door never toggles and stays as the map
# draws it (`Opening.initial_open`).
INTERIOR_DOORS_OPEN = True


def detour(swept: set[tuple[int, int]], start: tuple[float, float], yaw: float, cap: float) -> float | None:
    """How far along the corridor the robot's own detour says something stands, or None. Along the axis from `start`
    the points at s = 0.05, 0.15, ... below `cap` fall in fine cells; when the first cell is in `swept` and, after the
    first cell that is not (at s1), a later one is again, the robot went around something: `round(s1 - 0.05, 3)`."""
    along = [s for i in range(math.floor(cap / grid.FINE_M) + 1) if (s := grid.FINE_M * (i + 0.5)) < cap - grid.EPS]
    cells = [grid.fine_cell(start[0] + s * math.cos(yaw), start[1] + s * math.sin(yaw)) for s in along]
    inside = [c in swept for c in cells]
    if not inside or not inside[0] or False not in inside:
        return None
    gap = inside.index(False)
    return round(along[gap] - grid.FINE_M / 2, 3) if any(inside[gap:]) else None


def obstacles(m: MapSpec, robot: Robot) -> list[Rect]:
    """What a layout-memorising agent knows blocks the floor: the map's wall pieces, its fixed pieces that block and are
    at least `FLOOR_COVERING_M` tall (the legs only, for a robot that passes under the piece), and each door's leaf
    (`DOOR_LEAF_M` thick). Rugs, mats and painted floor markings never."""
    out = list(m.wall_pieces())
    for f in m.furniture:
        if f.mobility == "fixed" and f.blocks and f.size[2] >= FLOOR_COVERING_M:
            out += under_obstacles(f.footprint(), f.under_m) if passes_under(robot, f.under_m) else [f.footprint()]
    for door in m.doors():
        (hx, hy), (tx, ty) = m.door_leaf(door, INTERIOR_DOORS_OPEN if door.interior else door.initial_open)
        out.append(Rect((hx + tx) / 2, (hy + ty) / 2, math.hypot(tx - hx, ty - hy), DOOR_LEAF_M,
                        math.atan2(ty - hy, tx - hx)))
    return out


def corridor_answer(pl: PathLog, rec: dict, cap: float, heading: float) -> dict:
    """How far the corridor from the body's front edge along `heading` (radians) runs, from the base pose of the ask's
    step: its own detour, else the raycast with the pose pool (the module docstring)."""
    start = grid.front_edge(pl.model, rec["base"], heading)
    body = pl.model.body
    poses = [(s["pose"]["pos"][0], s["pose"]["pos"][1], math.radians(s["pose"]["yaw_deg"]))
             for _, s in sorted(pl.log.steps.items())]
    if (far := detour(grid.swath(poses, body.shape, body.length_m, body.width_m), start, heading, cap)) is not None:
        return {"type": "number", "value": far}
    pool, ray = pool_at("path_log", rec, pl.prior), None
    if pl.map is not None:   # a held-out layout has no map to cast a ray through: the pool alone
        walls = pl.cached("obstacles", lambda: obstacles(pl.map, pl.model))
        ray = {"type": "number", "value": grid.raycast(walls, start, heading, body, cap)}
    if pool:
        return learned_answer(pool, [] if ray is None else [ray])
    return dict(UNKNOWN) if ray is None else ray


def path_ahead(pl: PathLog, question: dict, rec: dict) -> dict:
    """`path_ahead` from the base pose of the ask's step, along its delivered heading."""
    return corridor_answer(pl, rec, question["slots"]["cap"], math.radians(rec["yaw"]))


def rear_clear(pl: PathLog, question: dict, rec: dict) -> dict:
    """`rear_clear` from the base pose of the ask's step: the corridor from the rear edge along the heading plus pi."""
    return corridor_answer(pl, rec, question["slots"]["cap"], math.radians(rec["yaw"]) + math.pi)


PATH_ANSWERERS: dict[str, PathAnswerer] = {"path_ahead": path_ahead, "rear_clear": rear_clear}
