# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's pallet as procedural geometry: a wooden pallet under a load wrapped in stretch film.

Runs inside Blender's own Python and is never imported by `stbench`. The builder is called as in `procedural_goods.py`
(bmesh, `DESIGN_BOX`, slot dict). The wooden pallet is the 1200 x 1000 ISO pallet of seven top boards, nine blocks and
three bottom boards (114 polygons, as before). The film, the slot that carries the catalog colour, is one shell over the
whole load: it bulges a little at mid-height, has stretched, tight corners with a few creases, a gathered, pleated neck
where it is pulled in under the load, spiral overlaps (a raised ridge for each lap, climbing as it goes round), the
faint seams of the cartons under it and a pillowy top. It is opaque: the colour of the film is the colour of the load.
"""

from __future__ import annotations

import math

from meshkit import solid
from partskit import Outline, loft, rings_of
from wobble import bump, smoothstep, wave

BASE = 0.145  # the top of the pallet's deck: where the load stands
TUCK = 0.03  # the film reaches this far down the deck's edge
RISE = 0.012  # the spacing of the shell's rings up the wall
SHELL = Outline(60, 50, 6)
PITCH = 0.115  # one lap of the film climbs this far
CORNER_R = 0.055


def deck(bm, box, wood: int) -> None:
    """Seven top boards, nine blocks and three bottom boards (19 slabs of 6 polygons)."""
    hx, hy = box[0] / 2, box[1] / 2
    board, w = 0.022, 0.1
    for i in range(7):  # the top deck: boards run along y
        cx = -hx + w / 2 + i * (2 * hx - w) / 6
        solid(bm, cx - w / 2, cx + w / 2, -hy, hy, BASE - board, BASE, wood)
    for cx in (-hx + w / 2, 0.0, hx - w / 2):
        solid(bm, cx - w / 2, cx + w / 2, -hy, hy, 0.0, board, wood)  # the bottom deck
        for cy in (-hy + w / 2, 0.0, hy - w / 2):
            solid(bm, cx - w / 2, cx + w / 2, cy - w / 2, cy + w / 2, board, BASE - board, wood)  # the blocks


def film(bm, hx: float, hy: float, h: float, wrap: int) -> None:
    """The load under its film, from the deck's edge to the box's top."""
    zb, rtop = BASE - TUCK, 0.05
    zw = h - rtop  # where the wall ends and the top's rounding begins
    a0, b0 = hx - 0.022, hy - 0.022  # the load stands just inside the pallet's edge, the film at its widest within it

    def inset(z: float) -> float:  # the rounding of the top edge
        return 0.0 if z <= zw else rtop - math.sqrt(max(rtop * rtop - (z - zw) ** 2, 0.0))

    def profile(z: float) -> tuple[float, float, float]:
        t = smoothstep(zb, zw, z)
        swell = 0.011 * math.sin(math.pi * t) ** 0.9 + 0.003 * math.sin(math.pi * min(1.0, 2 * t))
        neck = 0.03 * (1.0 - smoothstep(zb, zb + 0.07, z)) ** 2  # pulled in where it is gathered under the load
        s = inset(z)
        return a0 + swell - neck - s, b0 + swell - neck - s, max(CORNER_R - 0.6 * s - 0.5 * neck, 0.01)

    def push(z: float, u: float, p: tuple) -> float:
        x, y, nx, ny, edge = p
        d = 0.0
        phase = u * PITCH  # a lap climbs PITCH, so a ridge is a spiral and meets itself after one turn
        k = round((z - zb - 0.07 - phase) / PITCH)
        dz = z - (zb + 0.07 + PITCH * k + phase)
        d += (0.0042 + 0.0012 * wave(k, 0.0, 1.0)) * bump(dz, 0.016) - 0.0016 * bump(dz - 0.022, 0.014)
        for j in (1, 2, 3):  # the seams of the cartons: a groove between each course
            d -= 0.0028 * bump(z - (BASE + 0.213 * j + 0.004 * wave(u * 9, j)), 0.011)
        along = x * abs(ny) + y * abs(nx)  # the position along the face (x on a y face, y on an x face)
        for c in ((-0.2, 0.2) if abs(ny) > 0.5 else (0.0,)):
            d -= 0.0022 * bump(along - c, 0.016) * smoothstep(0.0, 0.03, edge)
        if edge < 0.14:  # the film is stretched tight over a corner: creases run out of it
            crease = math.sin(2 * math.pi * z / 0.085 + 5.0 * round(u * 4))
            d += 0.0035 * (1.0 - edge / 0.14) * crease * smoothstep(zb, zb + 0.1, z)
        fold = 0.5 + 0.5 * math.sin(2 * math.pi * u * 56.0 + 3.0 * wave(u * 7, 1.0))
        pleat = (1.0 - smoothstep(zb, zb + 0.13, z)) * fold
        d += 0.007 * pleat * (0.3 + 0.7 * smoothstep(0.0, 0.1, edge + 0.1))
        d -= 0.004 * pleat  # the folds of the neck
        d += 0.0025 * wave(u * 14, z * 3.0, 1.0)  # a little sag and slack
        return d * (1.0 - smoothstep(zw, h, z))

    stations = [zb + i * RISE for i in range(int((zw - zb) / RISE) + 1)] + [zw + rtop * math.sin(math.radians(a))
                                                                       for a in (15, 30, 45, 60, 75, 87)]
    rings = rings_of(SHELL, stations, profile, push)
    top = h - 0.0085  # the crown reaches the box's top and no further
    ring0 = rings[-1]
    cx = [sum(p[0] for p in ring0) / len(ring0), sum(p[1] for p in ring0) / len(ring0)]
    for s in (0.8, 0.6, 0.4, 0.2):  # the pillow of the top: concentric rings, higher in the middle, a little uneven
        rings.append([(cx[0] + (x - cx[0]) * s, cx[1] + (y - cx[1]) * s,
                       top + 0.008 * (1.0 - s * s) + 0.0025 * wave(x * 6, y * 6, 2.0)) for x, y, _ in ring0])
    loft(bm, rings, wrap, cap_first=True, cap_last=True)


def pallet(bm, box, slot) -> None:
    """A wooden pallet under a load wrapped in stretch film. The film is the colour attribute."""
    deck(bm, box, slot["wood"])
    film(bm, box[0] / 2, box[1] / 2, box[2], slot["wrap"])
