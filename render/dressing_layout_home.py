# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's Phase 1 dressing: what a real apartment of this layout has on its walls
and ceilings that the map does not draw. `layout(spec)` takes a home-catalog map spec (as `src/stbench/maps/<map>.json`
holds it) and returns every `Dressing` of the map in a fixed order; it depends on the map alone (the little randomness
there is, the choice of picture frames, is seeded from the map's name).

What goes where is read off the map, never off its name: a room is a kitchen if a stove stands in it, a bathroom if it
has a toilet or a bathtub, a bedroom if it has a bed, a living room if it has a sofa or a TV console, and a hall
otherwise; fixtures follow the furniture they belong to (upper cabinets over counters, a hood over a stove, a
television over a TV console, a cabinet over a toilet). `dressing_home_parts` has the trim, fittings, ceiling pieces and
window dressing and `dressing_home_plan` the geometry. Pieces are never on the open floor: they are on a wall face (at
most 0.11 m proud) or a ceiling, or above the footprint of blocking furniture (upper cabinets, hood, pendant, fan,
shower curtain), and clear of the receptacle volumes, door swings, openings and window glass that
`stbench.dressing_zones` keeps free.

Conventions (as `furniture.py`): a piece's `size` is (depth, width, height) in its own frame, x facing out of the wall
(or the way it faces), y across, z up, and `yaw_deg` is the direction it faces: into the room for a wall piece. The
recipe of a procedural piece is `proc:<class>`; an `asset:<id>` piece is the Poly Haven model fitted to that box (its
native x is the width, y the depth, z the height). A curtain's box holds its rod and rings, a shower curtain's its
rail, a roller blind's its cassette. `ceiling_lamp` stands for two things: a flush lamp centred on each map ceiling
light, whose area light at 2.45 m the render must keep below the lamp's underside (2.40 m), and a pendant over the
dining table.
"""

from __future__ import annotations

import hashlib
import math
import random

from dressing_home_grime import grime
from dressing_home_parts import (
    CASING_W,
    baseboards,
    ceiling,
    door_casings,
    outlets,
    radiators,
    switches,
    thermostat,
    window_dressing,
    window_trim,
)
from dressing_home_plan import Face, Furn, Plan, Room
from dressing_spec import Dressing

COUNTER_KINDS = ("counter", "sink_unit", "stove")
CABINET_BOTTOM, CABINET_H, CABINET_DEPTH, CABINET_MODULE = 1.45, 0.70, 0.34, 0.62
BACKSPLASH_PROUD = 0.012
TRIM_REACH = CASING_W + 0.025  # the tile stops this far from window glass: short of the casing and sill ends
FRAMES = (  # Poly Haven's real sizes (depth, width, height)
    ("hanging_picture_frame_01", (0.016, 0.594, 0.841)), ("hanging_picture_frame_02", (0.033, 0.752, 0.500)),
    ("hanging_picture_frame_03", (0.032, 0.385, 0.502)), ("fancy_picture_frame_01", (0.020, 0.603, 0.464)))
CLOCK = ("asset:wall_clock", (0.047, 0.320, 0.320))
PICTURES = {"living": 5, "kitchen": 4, "hall": 7, "bedroom": 4, "bathroom": 2}  # frames per room, at most


def _touching(face: Face, f: Furn) -> tuple[float, float] | None:
    """The stretch of the face a fixed blocking piece stands against (within 2 cm), if any."""
    a, b, d0, d1 = face.project(f.box)
    return (a, b) if f.blocks and f.mobility == "fixed" and d0 < 0.02 and d1 > 0.0 else None


def _runs(face: Face, room: Room) -> list[tuple[float, float, list[Furn]]]:
    """The counter runs on a face: merged stretches of counters, sink units and stoves standing end to end."""
    units = sorted(((*iv, f) for f in room.furniture if f.kind in COUNTER_KINDS and (iv := _touching(face, f))),
                   key=lambda u: u[0])
    runs: list[tuple[float, float, list[Furn]]] = []
    for a, b, f in units:
        if runs and a - runs[-1][1] < 0.02:
            runs[-1] = (runs[-1][0], max(runs[-1][1], b), [*runs[-1][2], f])
        else:
            runs.append((a, b, [f]))
    return runs


def _cut(a: float, b: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The stretches of a..b left when the holes are taken out."""
    out, cursor = [], a
    for h0, h1 in sorted(holes):
        if h1 <= cursor or h0 >= b:
            continue
        if h0 > cursor:
            out.append((cursor, h0))
        cursor = max(cursor, h1)
    return out + ([(cursor, b)] if cursor < b else [])


def kitchen(plan: Plan) -> list[Dressing]:
    """Backsplash behind each counter run (a short strip under a window), upper cabinets from 1.45 m over the counters
    (a cabinet over the fridge), a range hood over the stove and sockets above the worktop. Nothing stands on the
    stove top: a pot or a pan there reads as a bowl or a plate, which are tracked kinds."""
    out: list[Dressing] = []
    capped: set[str] = set()
    for room in (r for r in plan.rooms if r.role == "kitchen"):
        for face in plan.faces_of(room):
            for a, b, units in _runs(face, room):
                top = max(f.size[2] for f in units)
                windows = [(g.t0, g.t1, g.z0) for g in plan.gaps_in(face, top, CABINET_BOTTOM)]
                tile = [(w0 - TRIM_REACH, w1 + TRIM_REACH) for w0, w1, _ in windows]  # the tile stops short of the trim
                strips = [(s0, s1, CABINET_BOTTOM - top) for s0, s1 in _cut(a, b, tile)]
                strips += [(max(a, t0), min(b, t1), sill - 0.032 - top)
                           for (t0, t1), (_, _, sill) in zip(tile, windows, strict=True) if t0 < b and t1 > a]
                out += [face.item("backsplash", (s0 + s1) / 2, s1 - s0, BACKSPLASH_PROUD, top, h)
                        for s0, s1, h in strips if s1 - s0 > 0.05 and h > 0.05]
                out += _worktop_outlets(plan, face, a, b)
                stoves = [(f, _touching(face, f)) for f in units if f.kind == "stove"]
                holes = [(w0 - CASING_W - 0.1, w1 + CASING_W + 0.1) for w0, w1, _ in windows]
                for s0, s1 in _cut(a + 0.03, b - 0.03, holes + [iv for _, iv in stoves]):
                    n = max(1, math.ceil((s1 - s0) / CABINET_MODULE))
                    step = (s1 - s0) / n
                    out += [plan.claim(face.item("upper_cabinet", s0 + step * (k + 0.5), step - 0.004, CABINET_DEPTH,
                                                 CABINET_BOTTOM, CABINET_H)) for k in range(n)]
                for _f, iv in stoves:
                    out.append(plan.claim(face.item("range_hood", (iv[0] + iv[1]) / 2, iv[1] - iv[0] - 0.02, 0.50,
                                                    1.55, face.ceiling - 1.55)))
            for f in (f for f in room.furniture if f.kind == "fridge" and f.id not in capped):
                iv = _touching(face, f)
                if iv and f.size[2] + 0.5 < face.ceiling:
                    capped.add(f.id)  # a fridge in a corner stands against two faces: one cabinet
                    out.append(plan.claim(face.item("upper_cabinet", (iv[0] + iv[1]) / 2, iv[1] - iv[0] - 0.02, 0.60,
                                                    f.size[2] + 0.05, 0.45)))
    return out


def _worktop_outlets(plan: Plan, face: Face, a: float, b: float) -> list[Dressing]:
    """Sockets 1.1 m up a counter run, one every 1.2 m or so of backsplash that is clear of windows and the stove, each
    standing on the tile (its box starts at the tile's face)."""
    out = []
    for s0, s1 in plan.spans(face, 1.05, 1.165, BACKSPLASH_PROUD + 0.015, gap_margin=CASING_W + 0.1, keep=0.4):
        s0, s1 = max(s0, a + 0.1), min(s1, b - 0.1)
        n = max(1, round((s1 - s0) / 1.2)) if s1 - s0 > 0.3 else 0
        out += [face.item("outlet", s0 + (s1 - s0) * (k + 0.5) / n, 0.07, 0.015, 1.05, 0.115, standoff=BACKSPLASH_PROUD)
                for k in range(n)]  # on the tile, not in it
    return out


def television(plan: Plan) -> list[Dressing]:
    """A flat screen on the wall above each TV console, 5 cm above what may rest on it."""
    out = []
    for face in plan.faces:
        room = next(r for r in plan.rooms if r.id == face.region)
        for f in (f for f in room.furniture if f.kind == "tv_console" and _touching(face, f)):
            a, b = _touching(face, f)
            width = min(1.12, (b - a) * 0.8)
            out.append(plan.claim(face.item("wall_tv", (a + b) / 2, width, 0.05, plan.top_under(f.box) + 0.05,
                                            width * 0.58)))
    return out


def bathroom(plan: Plan) -> list[Dressing]:
    """A closed cabinet on the wall over the toilet, a roll holder beside it, and a curtain and rail along the open
    side of the tub (above its rim, inside its footprint)."""
    out = []
    for room in (r for r in plan.rooms if r.role == "bathroom"):
        for face in plan.faces_of(room):
            for f in (f for f in room.furniture if f.kind == "toilet" and _touching(face, f)):
                a, b = _touching(face, f)
                out.append(plan.claim(face.item("bathroom_cabinet", (a + b) / 2, 0.60, 0.11, 1.30, 0.70)))
                free = plan.spans(face, 0.64, 0.76, 0.11, edge=0.1, keep=0.2)
                for t in (a - 0.17, b + 0.17):
                    if any(s0 <= t - 0.075 and t + 0.075 <= s1 for s0, s1 in free):
                        out.append(face.item("toilet_roll_holder", t, 0.15, 0.11, 0.64, 0.12))
                        break
        for f in (f for f in room.furniture if f.kind == "bathtub"):
            front = f.size[0] / 2 - 0.04  # the curtain hangs just inside the rim, on the side the tub faces
            pos = (round(f.center[0] + math.cos(f.yaw) * front, 4), round(f.center[1] + math.sin(f.yaw) * front, 4),
                   f.size[2] + 0.05)
            out.append(Dressing("shower_curtain", "proc:shower_curtain", pos, round(math.degrees(f.yaw), 6),
                                (0.07, round(f.size[1] - 0.04, 4), 1.32)))
    return out


# -- pictures and clocks -------------------------------------------------------------------------------------------
def _seed(name: str) -> random.Random:
    """A generator of its own, seeded from the map's name alone (never the global one)."""
    return random.Random(int.from_bytes(hashlib.sha256(f"dressing/home/{name}".encode()).digest()[:8], "big"))


def _strip(face: Face, a: float, b: float):
    """The floor footprint, as a box, of the 12 cm of room in front of a stretch of a face (for `Plan.top_under`)."""
    (x0, y0), (x1, y1) = face.to_world(a), face.to_world(b, 0.12)
    return min(x0, x1), min(y0, y1), 0.0, max(x0, x1), max(y0, y1), 0.0


def decor(plan: Plan, rng: random.Random) -> list[Dressing]:
    """Wall clocks over the bookshelf and in the kitchen, then framed pictures on the free stretches of wall (above
    furniture first, then the longest, a pair on a long one), up to `PICTURES` a room. They keep off every earlier
    piece."""
    out: list[Dressing] = []
    for room in plan.rooms:
        for f in (f for f in room.furniture if f.kind == "bookshelf"):
            for face in (fc for fc in plan.faces_of(room) if _touching(fc, f)):
                a, b = _touching(face, f)
                out.append(plan.claim(face.item("wall_clock", (a + b) / 2, CLOCK[1][1], CLOCK[1][0], f.size[2] + 0.1,
                                                CLOCK[1][2], recipe=CLOCK[0])))
        if room.role == "kitchen":
            spans = [(b - a, face, (a + b) / 2) for face in plan.faces_of(room)
                     for a, b in plan.spans(face, 1.8, 2.12, 0.05, gap_margin=0.3, edge=0.3, keep=0.6)]
            if spans:
                _, face, t = max(spans, key=lambda s: s[0])
                out.append(plan.claim(face.item("wall_clock", t, CLOCK[1][1], CLOCK[1][0], 1.85, CLOCK[1][2],
                                                recipe=CLOCK[0])))
    order = list(FRAMES)  # the models in a seeded order, each used before any is used again
    rng.shuffle(order)
    used = 0
    for room in plan.rooms:
        cands = []
        for face in plan.faces_of(room):
            for a, b in plan.spans(face, 1.2, 2.2, 0.04, gap_margin=0.35, edge=0.3, keep=0.7):
                below = plan.top_under(_strip(face, a, b)) > 0.3
                cands.append((not below, -(b - a), len(cands), face, a, b))
        quota = PICTURES.get(room.role, 2)
        for _, _, _, face, a, b in sorted(cands, key=lambda c: c[:3]):
            n = 2 if b - a >= 1.9 else 1  # a long wall holds a pair
            for k in range(n):
                share, t = (b - a) / n, a + (b - a) * (k + 0.5) / n
                for i in range(len(order)):
                    name, (depth, width, height) = order[(used + i) % len(order)]
                    z0 = max(1.55 - height / 2, plan.top_under(_strip(face, t - width / 2, t + width / 2)) + 0.3)
                    if quota > 0 and width <= share - 0.2 and z0 + height <= face.ceiling - 0.3:
                        out.append(plan.claim(face.item("picture_frame", t, width, depth, z0, height,
                                                        recipe=f"asset:{name}")))
                        used, quota = used + i + 1, quota - 1
                        break
    return out


def layout(spec: dict) -> list[Dressing]:
    """Every Phase 1 dressing object of a household map, in a fixed order: trim, then the kitchen, television and
    bathroom, windows, radiators and thermostat, sockets and switches, the ceiling, pictures and clocks, and last the
    grime on the floor (`dressing_home_grime`)."""
    plan = Plan(spec)
    rng = _seed(plan.name)
    out: list[Dressing] = []
    for part in (baseboards, door_casings, window_trim, kitchen, television, bathroom, window_dressing, radiators,
                 thermostat, outlets, switches, ceiling):
        out += part(plan)
    return out + decor(plan, rng) + grime(plan)
