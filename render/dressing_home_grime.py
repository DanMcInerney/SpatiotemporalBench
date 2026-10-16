# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's grime (M1's R.2): what a lived-in floor has on it that the
map does not draw, placed on a `Plan` (`dressing_home_plan`).

Dust lines along the foot of the baseboards, a few hairs, crumbs round the kitchen's worktops and the tables, and the
scuffs of feet at the doors. Each is a flush plate at most `FLUSH_H_M` thick, so it lies under what a robot drives over
(the helper climbs 1 cm, the vacuum 2 cm) and in the guard's flush floor band (`stbench.dressing_zones`); it is
render-only dressing, `pass_index` 0, never an entity. A piece keeps off the doors (their opening and the square the
leaf sweeps, which the guard forbids to the floor), the rugs and the mat (a plate under their pile would be hidden, and
on it would stand in the volume things rest in) and the walls. The little randomness is its own stream, seeded from the
map's name, so the pictures and clocks of `dressing_layout_home` are where they were.
"""

from __future__ import annotations

import hashlib
import math
import random

from dressing_home_parts import BASEBOARD_PROUD, CASING_W
from dressing_home_plan import Plan, Room, aabb, inside
from dressing_spec import Dressing

FLUSH_H_M = 0.01  # nothing here is thicker: the helper's climb
DUST_DEPTH, DUST_H = 0.035, 0.003
HAIR = (0.16, 0.10, 0.0012)  # depth, width, height of a hair's box: its strands span it, 1.2 mm thick
CRUMBS = (0.12, 0.12, 0.004)
VARIANTS = 4  # distinct hairs and crumb clusters (the text of their recipe: `proc:hair:2`)
SCUFF = (0.45, 0.10, 0.0015)
HAIRS = {"bathroom": 4, "bedroom": 3, "hall": 3, "living": 3, "kitchen": 2}  # clusters a room
CRUMBS_NEAR = ("counter", "sink_unit", "stove", "dining_table", "coffee_table")
CRUMB_CLUSTERS = {"kitchen": 10, "living": 3}
WALL_KEEP_M = 0.25  # scattered pieces stay this far from a wall
DOOR_KEEP_M = 0.05  # and this far beyond a door's opening and swing
Rect = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
SOFT = ("rug", "floor_mat")  # fixed pieces that stand on the floor and are no obstacle: nothing lies under or on them


def _seed(name: str) -> random.Random:
    """A generator of its own, seeded from the map's name alone (never the global one, never the decor's)."""
    return random.Random(int.from_bytes(hashlib.sha256(f"dressing/home/grime/{name}".encode()).digest()[:8], "big"))


def _grown(r: Rect, d: float) -> Rect:
    return r[0] - d, r[1] - d, r[2] + d, r[3] + d


def _hits(x0: float, y0: float, x1: float, y1: float, keep: list[Rect]) -> bool:
    return any(x0 < r[2] and r[0] < x1 and y0 < r[3] and r[1] < y1 for r in keep)


def _door_keepouts(plan: Plan) -> list[Rect]:
    """The floor a door takes: its opening, `DOOR_KEEP_M + 0.12 m` out each side of its wall, and where it enters a room
    the square its leaf sweeps (as wide as the leaf, out of the face)."""
    out = []
    for face in plan.faces:
        for g in (g for g in face.gaps if g.kind in ("door", "arch")):
            reach = (g.t1 - g.t0 if g.kind == "door" and g.enters else 0.0) + DOOR_KEEP_M
            out_to = max(reach, 0.12 + DOOR_KEEP_M)
            pts = [face.to_world(g.t0 - DOOR_KEEP_M, -0.2), face.to_world(g.t1 + DOOR_KEEP_M, out_to)]
            out.append((min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)))
    return out


def _soft(plan: Plan) -> list[Rect]:
    boxes = [f.box for room in plan.rooms for f in room.furniture if f.kind in SOFT]
    return [(b[0], b[1], b[3], b[4]) for b in boxes]


def _solids(plan: Plan) -> list[Rect]:
    """The footprints of the fixed pieces that stand on the floor whole (the floor under a table or a bed is open: dust
    and hair lie there), a hair's breadth bigger."""
    boxes = [f.box for room in plan.rooms for f in room.furniture if f.mobility == "fixed" and f.blocks
             and f.under_m == 0.0]
    return [_grown((b[0], b[1], b[3], b[4]), 0.01) for b in boxes]


def _span_cut(a: float, b: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out, cursor = [], a
    for h0, h1 in sorted(holes):
        if h1 <= cursor or h0 >= b:
            continue
        if h0 > cursor:
            out.append((cursor, h0))
        cursor = max(cursor, h1)
    return out + ([(cursor, b)] if cursor < b else [])


def dust_lines(plan: Plan, rng: random.Random, soft: list[Rect]) -> list[Dressing]:
    """A strip `DUST_DEPTH` deep along the foot of each stretch of baseboard that stands free, in runs of 0.4 to 1.4 m
    with gaps between them (dust is patchy), clear of the doors' casings and of the rugs and the mat."""
    out: list[Dressing] = []
    for face in plan.faces:
        for a, b in plan.spans(face, 0.0, DUST_H, BASEBOARD_PROUD + DUST_DEPTH, gap_margin=CASING_W + 0.03, edge=0.04,
                               keep=0.25):
            projected = [face.project((r[0], r[1], 0.0, r[2], r[3], 0.0)) for r in soft]
            under_soft = [(max(t0, 0.0), t1) for t0, t1, d0, d1 in projected
                          if d0 < BASEBOARD_PROUD + DUST_DEPTH and d1 > 0.0]
            for s0, s1 in _span_cut(a, b, under_soft):
                cursor = s0 + rng.uniform(0.0, 0.3) + 0.001  # 1 mm short of what ends the stretch: rounding
                while cursor + 0.2 < s1 - 0.001:
                    length = min(rng.uniform(0.4, 1.4), s1 - 0.001 - cursor)
                    out.append(face.item("dust_line", cursor + length / 2, length, DUST_DEPTH, 0.0, DUST_H,
                                         standoff=BASEBOARD_PROUD))
                    cursor += length + rng.uniform(0.1, 0.5)
    return out


def _scatter(room: Room, rng: random.Random, size: tuple[float, float, float], keep: list[Rect],
             near: Rect | None = None, tries: int = 60) -> Dressing | None:
    """A piece of `size` at a random place and turn on the room's floor, `WALL_KEEP_M` from every wall and clear of
    `keep` (within `near`'s grown box and outside it, if given)."""
    xs, ys = [p[0] for p in room.polygon], [p[1] for p in room.polygon]
    reach = math.hypot(size[0], size[1]) / 2
    for _ in range(tries):
        if near is None:
            x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        else:
            x, y = rng.uniform(near[0] - 0.5, near[2] + 0.5), rng.uniform(near[1] - 0.5, near[3] + 0.5)
            if near[0] - 0.02 < x < near[2] + 0.02 and near[1] - 0.02 < y < near[3] + 0.02:
                continue
        yaw = round(rng.uniform(-180.0, 180.0), 1)
        x0, y0, x1, y1 = aabb(x, y, size[0], size[1], math.radians(yaw))
        if (inside(list(room.polygon), x, y) and room.edge_distance(x, y) >= WALL_KEEP_M + reach
                and not _hits(x0, y0, x1, y1, keep)):
            return Dressing("", "", (round(x, 4), round(y, 4), 0.0), yaw, size)
    return None


def hair_and_crumbs(plan: Plan, rng: random.Random, keep: list[Rect]) -> list[Dressing]:
    """Hairs on every room's floor and clusters of crumbs by the worktops and tables. `keep` grows by each piece placed,
    so none lies on another."""
    out: list[Dressing] = []
    for room in plan.rooms:
        wanted = [("hair", HAIR, None)] * HAIRS.get(room.role, 0)
        homes = [f.box for f in room.furniture if f.kind in CRUMBS_NEAR]
        for _ in range(CRUMB_CLUSTERS.get(room.role, 0) if homes else 0):
            wanted.append(("crumbs", CRUMBS, homes[rng.randrange(len(homes))]))
        for cls, size, home in wanted:
            near = None if home is None else (home[0], home[1], home[3], home[4])
            if (d := _scatter(room, rng, size, keep, near)) is not None:
                out.append(Dressing(cls, f"proc:{cls}:{rng.randrange(VARIANTS)}", d.pos, d.yaw_deg, d.size))
                x0, y0, x1, y1 = aabb(d.pos[0], d.pos[1], size[0], size[1], math.radians(d.yaw_deg))
                keep.append(_grown((x0, y0, x1, y1), 0.02))
    return out


def scuffs(plan: Plan, rng: random.Random, keep: list[Rect]) -> list[Dressing]:
    """A mark of feet in front of each door, beyond the square its leaf sweeps where it enters the room, along the way
    through it."""
    out: list[Dressing] = []
    doors = _door_keepouts(plan)
    other = [r for r in keep if r not in doors]  # the doors' own floor is not an obstacle to the mark beyond it
    for face in plan.faces:
        room = next(r for r in plan.rooms if r.id == face.region)
        for g in (g for g in face.gaps if g.kind == "door"):
            t = (g.t0 + g.t1) / 2 + rng.uniform(-0.1, 0.1)
            standoff = 0.12 + DOOR_KEEP_M + (g.t1 - g.t0 if g.enters else 0.0)
            piece = face.item("scuff", t, SCUFF[1], SCUFF[0], 0.0, SCUFF[2], standoff=standoff)
            x0, y0, _z0, x1, y1, _z1 = piece.box
            if inside(list(room.polygon), piece.pos[0], piece.pos[1]) and not _hits(x0, y0, x1, y1, other):
                out.append(piece)
    return out


def grime(plan: Plan) -> list[Dressing]:
    """Every piece of grime of a household map, in a fixed order: dust lines, hairs and crumbs, scuffs."""
    rng = _seed(plan.name)
    soft, solids, doors = _soft(plan), _solids(plan), _door_keepouts(plan)
    dust = dust_lines(plan, rng, soft)
    taken = [*soft, *solids]  # grows with the hairs and crumbs, then the scuffs keep off them too
    floor = hair_and_crumbs(plan, rng, [*taken, *doors])
    return dust + floor + scuffs(plan, rng, [*taken, *(piece_rect(d) for d in floor)])


def piece_rect(d: Dressing) -> Rect:
    x0, y0, _z0, x1, y1, _z1 = d.box
    return _grown((x0, y0, x1, y1), 0.02)
