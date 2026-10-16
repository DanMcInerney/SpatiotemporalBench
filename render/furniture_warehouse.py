# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's furniture as compositions of primitives: pallet racks (with static stock), staging lanes, packing
tables, shelving, dock doors, the AMR's charger and the bays in front of its stations, and the placeholder drawings of
the two pieces of light equipment (pallet jack, roll cage).

Runs inside Blender's own Python and is never imported by `stbench`. Like `furniture.py`, each builder takes the entity
size (depth, width, height), its colour attribute and returns parts in the entity's frame (origin at the base centre,
x = front, y = left, z up). The third argument is the entity's receptacle surfaces from the map as `(cx, cy, sx, sy, z)`
in that frame, because a rack must know which of its bay-levels are free (`rack_layout`): surfaces sit at exactly the
heights the map gives, so carried items rest on them.

The rack stock (cartons on bare wooden pallets, never wrapped: clear film would read white, a tracked pallet colour) is
scenery, not an entity: each pallet takes one of a few carton layouts (mixed carton
sizes, three tints of beige), prints a white shipping label with a barcode, a black logo bar and a mark on the front of
each carton that faces the aisle, and carries corner boards and straps. It stands up to 3 cm and 2 degrees off true,
seeded by the rack's place in the scene and its bay-level, and that is clamped so that nothing leaves its bay-level: the
stock stays clear of the uprights, the back beam, the rack's front plane and the free bay-levels beside it.
"""

from __future__ import annotations

import math
import random

import bpy
from rack_layout import BAY_M, LEVEL_M, layout, stock
from shapes import Part, box, cyl

POST_X, POST_Y, BEAM_H, BEAM_D = 0.08, 0.09, 0.12, 0.14
DECK_H, DECK = 0.02, "grey*0.7"  # a wire deck is drawn as a thin plate: the items that rest at a level rest on it
CARTON = "beige*0.8"
PALLET_X, PALLET_Y = 1.0, 1.16  # across the rack and along it
LOAD_X, LOAD_Y = 0.96, 1.12  # the cartons' footprint: 2 cm in from the pallet's edge
GAP_M, DECK_Z = 0.003, 0.145  # between cartons; the pallet's height
STOCK_X = (-0.535, 0.54)  # across the rack, stock stays between its back panel and the surfaces' front edge
SURFACE_HALF = BAY_M / 2 - 0.05  # half the length of a bay-level's surface rect (2.6 m of the 2.7 m bay)
TILT_DEG, SHIFT_M = 2.0, 0.03  # the most a pallet stands off true
# carton layouts: name -> (weight, carton height, (across, along) of a plain layer; None for the mixed one)
LAYOUTS = {"2x3": (3, 0.24, (2, 3)), "2x2": (2, 0.30, (2, 2)), "2x4": (2, 0.26, (2, 4)), "3x4": (1, 0.20, (3, 4)),
           "1x2": (1, 0.34, (1, 2)), "mix": (2, 0.27, None)}


def _strut(x0: float, z0: float, x1: float, z1: float, y: float, t: float, mat: str) -> Part:
    """A square bar in the plane y = const from (x0, z0) to (x1, z1)."""
    length = math.hypot(x1 - x0, z1 - z0)
    return Part("box", ((x0 + x1) / 2, y, (z0 + z1) / 2), (t, t, length), (0, math.atan2(x1 - x0, z1 - z0), 0), mat)


def _grid(x0: float, x1: float, nx: int, ny: int) -> list[tuple[float, float, float, float]]:
    """The (x0, x1, y0, y1) of an nx by ny grid of cartons filling [x0, x1] across the load and all of its length."""
    return [(x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx, -LOAD_Y / 2 + LOAD_Y * j / ny,
             -LOAD_Y / 2 + LOAD_Y * (j + 1) / ny) for i in range(nx) for j in range(ny)]


def _layer(name: str, layer: int) -> list[tuple[float, float, float, float]]:
    """The cartons of one layer in the pallet's frame (centre at the origin, x to the rack's front). The mixed layout
    has a front column of two long cartons and a back column of three short ones, swapped on every layer."""
    plain = LAYOUTS[name][2]
    if plain:
        return _grid(-LOAD_X / 2, LOAD_X / 2, *plain)
    front, back = (2, 3) if layer % 2 == 0 else (3, 2)
    return _grid(0.0, LOAD_X / 2, 1, front) + _grid(-LOAD_X / 2, 0.0, 1, back)


def _pallet(z0: float) -> list[Part]:
    """A bare wooden pallet: three bearers under nine top boards, running across the rack, 14.5 cm high."""
    gap = (PALLET_Y - 9 * 0.1) / 8
    boards = [box(-PALLET_X / 2, PALLET_X / 2, -PALLET_Y / 2 + i * (0.1 + gap), -PALLET_Y / 2 + i * (0.1 + gap) + 0.1,
                  z0 + 0.123, z0 + DECK_Z, "oak*0.8" if i % 3 else "oak*0.72") for i in range(9)]
    return boards + [box(-PALLET_X / 2, PALLET_X / 2, dy - 0.05, dy + 0.05, z0, z0 + 0.123, "oak*0.6")
                     for dy in (-0.5, 0.0, 0.5)]


def _marks(rng: random.Random, x: float, y0: float, y1: float, za: float, zb: float) -> list[Part]:
    """What is printed on the front of a carton (its +x face at x, between y0 and y1, za and zb): a white shipping label
    with a black barcode, a black logo bar near the top and, on a wide carton, a small black mark."""
    w, h = y1 - y0 - 2 * GAP_M, zb - za - 2 * GAP_M
    lw, lh = min(0.14, 0.55 * w), min(0.09, 0.45 * h)
    yc = rng.uniform(y0 + GAP_M + lw / 2 + 0.01, y1 - GAP_M - lw / 2 - 0.01)
    zc = rng.uniform(za + GAP_M + lh / 2 + 0.015, zb - GAP_M - lh / 2 - 0.045)
    parts = [box(x, x + 0.002, yc - lw / 2, yc + lw / 2, zc - lh / 2, zc + lh / 2, "white"),
             box(x + 0.002, x + 0.0035, yc - 0.35 * lw, yc + 0.35 * lw, zc - 0.4 * lh, zc - 0.15 * lh, "black"),
             box(x, x + 0.0015, (y0 + y1) / 2 - 0.25 * w, (y0 + y1) / 2 + 0.25 * w, zb - 0.035, zb - 0.02, "black")]
    if w > 0.24:
        parts.append(box(x, x + 0.0015, y0 + 0.04, y0 + 0.075, za + 0.03, za + 0.065, "black"))
    return parts


def _bindings(top: float, z0: float) -> list[Part]:
    """Two corner boards on the front edges of the load, two horizontal straps round it and two over the top and down
    its front (all in the pallet's frame)."""
    hw, hl, cz = LOAD_X / 2, LOAD_Y / 2, z0 + DECK_Z
    parts = [box(hw - 0.03, hw + 0.008, hl - 0.03, hl + 0.008, cz, top, "beige*0.95"),
             box(hw - 0.03, hw + 0.008, -hl - 0.008, -hl + 0.03, cz, top, "beige*0.95")]
    for f in (0.33, 0.7):
        z = cz + (top - cz) * f
        parts += [box(hw, hw + 0.004, -hl - 0.004, hl + 0.004, z, z + 0.015, "dark"),
                  box(-hw, hw + 0.004, hl, hl + 0.004, z, z + 0.015, "dark"),
                  box(-hw, hw + 0.004, -hl - 0.004, -hl, z, z + 0.015, "dark")]
    for y in (-0.3, 0.3):
        parts += [box(hw, hw + 0.004, y - 0.008, y + 0.008, cz, top, "dark"),
                  box(-hw, hw + 0.004, y - 0.008, y + 0.008, top, top + 0.004, "dark")]
    return parts


def _jitter(rng: random.Random, cx: float, cy: float, y_limits: tuple[float, float]) -> tuple[float, float, float]:
    """(tilt in radians, dx, dy) of a pallet whose centre is nominally at (cx, cy): up to `TILT_DEG` and `SHIFT_M` off
    true, cut back so that the pallet stays in `STOCK_X` and `y_limits` (the load, its boards and its straps are
    all inside the pallet's footprint)."""
    tilt = math.radians(rng.uniform(-TILT_DEG, TILT_DEG))
    c, s = math.cos(tilt), math.sin(tilt)
    corners = [(x, y) for x in (-PALLET_X / 2, PALLET_X / 2) for y in (-PALLET_Y / 2, PALLET_Y / 2)]  # all else is in
    xs, ys = [c * x - s * y for x, y in corners], [s * x + c * y for x, y in corners]

    def fit(v: float, lo: float, hi: float) -> float:
        return (lo + hi) / 2 if lo > hi else min(max(v, lo), hi)

    dx = fit(rng.uniform(-SHIFT_M, SHIFT_M), STOCK_X[0] - cx - min(xs), STOCK_X[1] - cx - max(xs))
    dy = fit(rng.uniform(-SHIFT_M, SHIFT_M), y_limits[0] - cy - min(ys), y_limits[1] - cy - max(ys))
    return tilt, dx, dy


def _stack(cx: float, cy: float, z0: float, height: float, rng: random.Random, y_limits: tuple[float, float]
           ) -> list[Part]:
    """One pallet of stock on a bare wooden pallet, built of cartons in layers: its centre nominally at (cx, cy) in the
    rack's frame, its base at z0, the load `height` high. `rng` picks its layout, tints, labels and how far it stands
    off true; `y_limits` are the least and greatest y that any part of it may reach."""
    name = rng.choices(list(LAYOUTS), [v[0] for v in LAYOUTS.values()])[0]
    layers = max(2, round(height / LAYOUTS[name][1]))
    step, top = height / layers, z0 + DECK_Z + height
    parts = _pallet(z0)
    for layer in range(layers):
        za, zb = z0 + DECK_Z + layer * step, z0 + DECK_Z + (layer + 1) * step
        for x0, x1, y0, y1 in _layer(name, layer):
            roll = rng.random()
            mat = CARTON if roll < 0.5 else "beige*0.7" if roll < 0.8 else "beige*0.62" if roll < 0.92 else "beige*0.88"
            parts.append(box(x0 + GAP_M, x1 - GAP_M, y0 + GAP_M, y1 - GAP_M, za + 0.002, zb - 0.002, mat))
            if x1 > LOAD_X / 2 - 1e-9:  # a carton on the face towards the aisle
                parts += _marks(rng, x1 - GAP_M, y0, y1, za, zb)
    parts += _bindings(top, z0)
    tilt, dx, dy = _jitter(rng, cx, cy, y_limits)
    c, s = math.cos(tilt), math.sin(tilt)
    return [p._replace(loc=(cx + dx + c * p.loc[0] - s * p.loc[1], cy + dy + s * p.loc[0] + c * p.loc[1], p.loc[2]),
                       rot=(p.rot[0], p.rot[1], p.rot[2] + tilt)) for p in parts]


def _rack_number() -> int:
    """How many pallet racks the scene already holds. The builder is given no id, and the scene builds the map's racks
    in their order, so this is the rack's place in the map: it makes each rack's stock its own, the same in every
    episode."""
    return sum(1 for o in bpy.data.objects if o.name.startswith("pallet_rack"))


def pallet_rack(s, c, surfaces) -> list[Part]:
    """A single-deep pallet rack row: upright frames with bracing, orange beams at each level above the floor, a solid
    mesh back, a wire deck between the beams of every free bay-level above the floor (the floor is level 0's deck), and
    static stock (pallets of cartons) in exactly the bay-levels the map gives no surface. Front is +x."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    bays, _, free = layout(s, surfaces)
    parts: list[Part] = [box(-hx + 0.002, -hx + 0.014, -hy, hy, 0.1, sz - 0.1, "grey*0.85")]  # the back panel
    parts += [box(-hx + 0.014, -hx + 0.02, -hy, hy, z, z + 0.012, "grey*0.8")
              for z in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2)]
    for i in range(bays + 1):  # upright frames
        y = -hy + POST_Y / 2 + i * (sy - POST_Y) / bays
        for sign in (-1, 1):
            xc = sign * (hx - POST_X / 2)
            parts.append(box(xc - POST_X / 2, xc + POST_X / 2, y - POST_Y / 2, y + POST_Y / 2, 0, sz, c))
        zs = [0.25, 1.0, 1.75, 2.5, 3.25, sz - 0.05]
        x_in = hx - POST_X
        parts += [_strut(-x_in if k % 2 == 0 else x_in, a, x_in if k % 2 == 0 else -x_in, b, y, 0.03, c + "*0.8")
                  for k, (a, b) in enumerate(zip(zs, zs[1:], strict=False))]
    for level in range(1, round(sz / LEVEL_M)):  # beams, their tops at the level's height
        z = level * LEVEL_M
        for sign in (-1, 1):
            xo = sign * (hx - 0.02)
            xi = xo - sign * BEAM_D
            parts.append(box(min(xo, xi), max(xo, xi), -hy + 0.02, hy - 0.02, z - BEAM_H, z, "orange"))
    pitch = (sy - POST_Y) / bays  # between the upright frames, as above
    for bay, level in sorted(free):  # a deck from beam to beam and post to post, its top at the surface
        if level >= 1:
            y0 = -hy + POST_Y + bay * pitch
            parts.append(box(-hx + 0.02 + BEAM_D, hx - 0.02 - BEAM_D, y0, y0 + pitch - POST_Y, level * LEVEL_M - DECK_H,
                             level * LEVEL_M, DECK))
    number = _rack_number()
    for bay, level in stock(s, surfaces):
        yc = -hy + (bay + 0.5) * BAY_M
        height = 0.65 + 0.1 * ((bay * 3 + level * 5) % 4)
        # between the uprights' faces, and within the bay-level's own volume (the surface rect, 2.6 m of the bay)
        lo = max(-hy + POST_Y + bay * pitch, yc - SURFACE_HALF) + 0.005
        hi = min(-hy + (bay + 1) * pitch, yc + SURFACE_HALF) - 0.005
        for k, dy in enumerate((-0.64, 0.64)):
            rng = random.Random(f"{number}:{bay}:{level}:{k}")  # a string seed: stable across runs and machines
            parts += _stack(0.0, yc + dy, level * LEVEL_M, height - 0.05 * k, rng,
                            (lo, yc - 0.005) if k == 0 else (yc + 0.005, hi))
    return parts


def staging_lane(s, c, surfaces) -> list[Part]:
    """A floor marking: a rectangle outlined in paint (the colour attribute), 0.1 m wide, from 1 mm up to the piece's
    height (5 mm in the maps: paint is floor to a robot that climbs 1 cm, so none is drawn higher than its piece)."""
    sx, sy, sz = s
    hx, hy, w = sx / 2, sy / 2, 0.1
    return [box(-hx, hx, -hy, -hy + w, 0.001, sz, c), box(-hx, hx, hy - w, hy, 0.001, sz, c),
            box(-hx, -hx + w, -hy + w, hy - w, 0.001, sz, c), box(hx - w, hx, -hy + w, hy - w, 0.001, sz, c)]


def bay(s, c, surfaces) -> list[Part]:
    """The bay in front of a station: a floor marking, a staging lane's outline in the colour attribute, from 1 mm to
    the piece's height (5 mm: paint is floor to a robot that climbs 1 cm). It is drawn as a staging lane is, so no
    marking builder draws higher than its piece."""
    return staging_lane(s, c, surfaces)


def charger(s, c, surfaces) -> list[Part]:
    """A wall charging station: a dark housing the piece's size, solid from the floor, with its contact plate (two steel
    strips) set 4 mm proud on its front face (+x), inside the piece's footprint."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    proud = 0.004
    return [box(-hx, hx - proud, -hy, hy, 0, sz, c)] + [box(hx - proud, hx, -0.22, 0.22, z, z + 0.04, "steel")
                                                          for z in (0.10, 0.20)]


def packing_table(s, c, surfaces) -> list[Part]:
    """A steel work table: a top whose upper face is the surface height, four legs and a low shelf."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    top = surfaces[0][4] if surfaces else sz
    parts = [box(-hx, hx, -hy, hy, top - 0.04, top, c), box(-hx + 0.04, hx - 0.04, -hy + 0.04, hy - 0.04, 0.15, 0.18,
                                                              c + "*0.8")]
    parts += [box(ix * (hx - 0.05) - 0.025, ix * (hx - 0.05) + 0.025, iy * (hy - 0.05) - 0.025,
                  iy * (hy - 0.05) + 0.025, 0, top - 0.04, c + "*0.8") for ix in (-1, 1) for iy in (-1, 1)]
    return parts


def shelving(s, c, surfaces) -> list[Part]:
    """Boltless shelving: four posts and a board at each surface height, and a top board."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(ix * (hx - 0.02) - 0.02, ix * (hx - 0.02) + 0.02, iy * (hy - 0.02) - 0.02, iy * (hy - 0.02) + 0.02,
                 0, sz, c + "*0.7") for ix in (-1, 1) for iy in (-1, 1)]
    heights = sorted(z for *_, z in surfaces) or [0.5, 1.1, 1.7]
    parts += [box(-hx, hx, -hy, hy, z - 0.03, z, c) for z in heights]
    parts.append(box(-hx, hx, -hy, hy, sz - 0.03, sz, c))
    return parts


def dock_door(s, c, surfaces) -> list[Part]:
    """A closed roller shutter: a slatted panel in a dark frame, its slats on the +x face."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx - 0.03, -hy, hy, 0, sz, c + "*0.7")]
    parts += [box(hx - 0.03, hx, -hy + 0.1, hy - 0.1, z, z + 0.045, c) for z in (0.1 + 0.095 * i for i in range(30))]
    parts += [box(hx - 0.03, hx, -hy, -hy + 0.1, 0, sz, "dark*1.5"),
              box(hx - 0.03, hx, hy - 0.1, hy, 0, sz, "dark*1.5"),
              box(hx - 0.03, hx, -hy + 0.1, hy - 0.1, sz - 0.1, sz, "dark*1.5"),
              box(hx - 0.03, hx, -hy + 0.1, hy - 0.1, 0, 0.1, "yellow")]
    return parts


def pallet_jack(s, c, surfaces) -> list[Part]:
    """The placeholder of a hand pallet truck: two forks, a pump housing, a drawbar and a handle. Forks point to +x."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-0.3, hx, s0 * (hy - 0.08) - 0.08, s0 * (hy - 0.08) + 0.08, 0.02, 0.09, c) for s0 in (-1, 1)]
    parts += [cyl(-0.34, s0 * 0.14, 0, 0.2, 0.1, "dark") for s0 in (-1, 1)]
    parts += [box(-0.46, -0.2, -0.2, 0.2, 0.09, 0.32, c),
              Part("box", (-0.505, 0.0, 0.68), (0.04, 0.04, 0.84), (0, -0.431, 0), c),
              box(-hx, -hx + 0.17, -0.1, 0.1, 1.04, sz, c)]
    return parts


def roll_cage(s, c, surfaces) -> list[Part]:
    """The placeholder of a roll container: a base on casters, corner posts and three mesh sides as wires."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx, -hy, hy, 0.125, 0.15, c)]
    parts += [cyl(ix * (hx - 0.08), iy * (hy - 0.08), 0, 0.125, 0.0625, "dark") for ix in (-1, 1) for iy in (-1, 1)]
    parts += [box(ix * (hx - 0.015) - 0.015, ix * (hx - 0.015) + 0.015, iy * (hy - 0.015) - 0.015,
                  iy * (hy - 0.015) + 0.015, 0.15, sz, c) for ix in (-1, 1) for iy in (-1, 1)]
    for z in (0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, sz - 0.012):
        parts += [box(-hx, hx, iy * (hy - 0.015) - 0.006, iy * (hy - 0.015) + 0.006, z, z + 0.012, c) for iy in (-1, 1)]
        parts.append(box(-hx + 0.009, -hx + 0.021, -hy, hy, z, z + 0.012, c))
    return parts


BUILDERS = {f.__name__: f for f in (pallet_rack, staging_lane, bay, charger, packing_table, shelving, dock_door,
                                    pallet_jack, roll_cage)}
