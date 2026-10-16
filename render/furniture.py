# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Fixed and light furniture as compositions of primitives.

Each builder takes the entity size (depth, width, height), its colour attribute and the heights of its receptacle
surfaces (from the map spec) and returns parts in the entity's local frame: origin at the base centre, x = front,
y = left, z up. Surfaces sit at exactly the heights the map gives, so carried items rest on them. The downtown's ground
pieces (sidewalks, parking lanes, loading zones, the square's paving) are in `furniture_city.py`, which registers them
here; like the warehouse's they are handed their surfaces whole, by pose.

The fixed household furniture has real detail, all of it part of the entity's one mesh
and in the entity's own palette keys: edges are bevelled 3 to 8 mm (`shapes.Part.bevel`); cabinets have door fronts
separated by 5 mm gaps over a dark carcass, with bar handles; the wardrobe has two doors; the sofa has seat and back
cushions of its own fabric; the bed has a duvet and pillows in the bed's own colour (never a loose saturated textile,
which a towel look-alike would be). The detail never rises above a receptacle surface (a cushion's top is the seat
height, a pillow's and the duvet's the bed's) and never stands in front of the sofa's seat rect, so what the map rests
on a piece still rests on it.

What the robot vacuum sees from 8 cm up is the underside of every piece that stands on legs: turned legs thicker at
the top, each on a 4 mm felt pad (`_turned_legs`); a black non-woven dust cloth (`cover`) under the sofa and the bed, in
the plane of their frames; raw board (`raw`) on the underside of every table top, shelf and drawer (the top's own
finish, a wood grain with a bump, is a smear seen from 2 cm at a grazing angle); beams under the dining table; a cable
tray under the desk. Plinths are the colour of their doors, darker. All of it is inside the piece's four leg squares
(`geometry.under_obstacles`) or at the underside the map gives (`under_m`) and above it, which
`tests/test_furniture_under_blender.py` casts rays at.
"""

from __future__ import annotations

import math

from furniture_city import BUILDERS as CITY
from furniture_house import BUILDERS as HOUSE
from shapes import Part, ball, box, cyl, ring

TOP_T = 0.04
DOOR_T, SET_BACK, GAP = 0.018, 0.02, 0.005  # a door front's thickness, how far the worktop overhangs it, the seam
EDGE = 0.004  # the rounding of a hard edge
PAD_H = 0.004  # a felt pad under a leg


def _turned_legs(hx, hy, z1, r_top, inset, mat, bottom: float = 0.62) -> list[Part]:
    """Four turned legs, their centres `inset` in from the corners, to height `z1`: radius `r_top` at the top, tapering
    to `bottom` of it at the foot, each on a felt pad 3 mm wider all round. Both stay inside a 0.15 m corner square."""
    out: list[Part] = []
    for ix in (-1, 1):
        for iy in (-1, 1):
            cx, cy, r0 = ix * (hx - inset), iy * (hy - inset), r_top * bottom
            out += [cyl(cx, cy, 0.0, PAD_H, r0 + 0.003, "dark*0.7"), cyl(cx, cy, PAD_H, z1, r0, mat, 1 / bottom)]
    return out


def _bar(x: float, y: float, z: float, length: float, vertical: bool, mat: str = "steel") -> list[Part]:
    """A bar handle on a front at x: the bar and two posts, 27 mm proud."""
    if vertical:
        h = length / 2
        return [box(x + 0.015, x + 0.027, y - 0.006, y + 0.006, z - h, z + h, mat, 0.002)] + [
            box(x, x + 0.015, y - 0.004, y + 0.004, z + d - 0.005, z + d + 0.005, mat) for d in (-h + 0.015, h - 0.015)]
    h = length / 2
    return [box(x + 0.015, x + 0.027, y - h, y + h, z - 0.006, z + 0.006, mat, 0.002)] + [
        box(x, x + 0.015, y + d - 0.005, y + d + 0.005, z - 0.004, z + 0.004, mat) for d in (-h + 0.015, h - 0.015)]


def _fronts(hx, y_lo, y_hi, plinth, z1, n, body, handle_z, *, drawers: int = 0) -> list[Part]:
    """The front of a cabinet between y_lo and y_hi: `n` door fronts, or `drawers` drawer fronts one above the other,
    between z = `plinth` and `z1` over a dark strip (the seams show it), each with a bar handle at `handle_z` (a door)
    or its middle (a drawer). The faces are at x = hx - SET_BACK."""
    xf = hx - SET_BACK
    parts = [box(xf - DOOR_T - 0.001, xf - DOOR_T, y_lo + 0.002, y_hi - 0.002, plinth, z1, "dark")]
    z0, span = plinth + 0.003, y_hi - y_lo - 0.008
    if drawers:
        step = (z1 - 0.003 - z0 + GAP) / drawers
        for k in range(drawers):
            a, b = z0 + k * step, z0 + (k + 1) * step - GAP
            parts.append(box(xf - DOOR_T, xf, y_lo + 0.004, y_hi - 0.004, a, b, body, EDGE))
            parts += _bar(xf, (y_lo + y_hi) / 2, (a + b) / 2, min(0.16, 0.6 * span), False)
        return parts
    width = (span - (n - 1) * GAP) / n
    for i in range(n):
        y0 = y_lo + 0.004 + i * (width + GAP)
        parts.append(box(xf - DOOR_T, xf, y0, y0 + width, z0, z1 - 0.003, body, EDGE))
        edge = y0 + 0.04 if i % 2 else y0 + width - 0.04  # the handle by the closing edge, alternating
        parts += _bar(xf, edge, handle_z if handle_z is not None else z1 - 0.15, 0.14, True)
    return parts


def _cabinet(s, body, top, plinth=0.07, handle_z=None, doors: int | None = None, drawers: int = 0) -> list[Part]:
    """Base cabinet: recessed plinth, carcass, worktop with a 2 cm overhang, door or drawer fronts with gaps and bar
    handles. Doors by default every 45 cm."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx - 0.04, -hy + 0.01, hy - 0.01, 0, plinth, body + "*0.75"),  # the plinth matches the doors
             box(-hx, hx - SET_BACK - DOOR_T - 0.001, -hy, hy, plinth, sz - TOP_T, body, EDGE),
             box(-hx, hx, -hy, hy, sz - TOP_T, sz, top, EDGE)]
    return parts + _fronts(hx, -hy, hy, plinth, sz - TOP_T, doors or max(1, round(sy / 0.45)), body, handle_z,
                           drawers=drawers)


def fridge(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0.05, sz, c, 0.01),
            box(-hx + 0.03, hx - 0.03, -hy + 0.02, hy - 0.02, 0, 0.05, "dark"),
            box(hx, hx + 0.003, -hy + 0.01, hy - 0.01, sz * 0.66, sz * 0.66 + 0.012, "dark"),
            box(hx, hx + 0.025, hy - 0.09, hy - 0.06, sz * 0.70, sz * 0.92, "steel", 0.004),
            box(hx, hx + 0.025, hy - 0.09, hy - 0.06, sz * 0.38, sz * 0.62, "steel", 0.004)]


def counter(s, c, zs):
    return _cabinet(s, c, "stone")


def sink_unit(s, c, zs):
    sx, sy, sz = s
    hy = sy / 2
    return _cabinet(s, c, "stone") + [
        box(-0.22, 0.0, -hy + 0.07, hy - 0.07, sz, sz + 0.004, "steel"),
        box(-0.19, -0.03, -hy + 0.10, hy - 0.10, sz, sz + 0.006, "dark"),
        cyl(-0.27, 0, sz, sz + 0.2, 0.016, "steel"),
        box(-0.27, -0.12, -0.012, 0.012, sz + 0.18, sz + 0.21, "steel", 0.003)]


def stove(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx, -hy, hy, 0, sz, c, 0.006), box(-hx, hx, -hy, hy, sz - 0.01, sz, "steel*0.5", 0.003)]
    for ix in (-0.15, 0.12):
        for iy in (-0.14, 0.14):
            parts.append(cyl(ix, iy, sz, sz + 0.006, 0.07, "dark"))
    parts += [box(hx, hx + 0.004, -hy + 0.06, hy - 0.06, 0.12, sz - 0.18, "dark*1.4", 0.004),
              box(hx, hx + 0.03, -hy + 0.06, hy - 0.06, sz - 0.24, sz - 0.21, "steel", 0.004)]
    parts += [ball(hx + 0.01, y, sz - 0.1, 0.018, 0.018, 0.018, "steel") for y in (-0.2, -0.07, 0.07, 0.2)]
    return parts


def _table(s, c, apron: float) -> list[Part]:
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    rail, z1 = c + "*0.8", sz - TOP_T
    return [box(-hx, hx, -hy, hy, z1, sz, c, 0.005),
            box(-hx + 0.01, hx - 0.01, -hy + 0.01, hy - 0.01, z1 - 0.001, z1, "raw"),  # the raw underside
            box(-0.03, 0.03, -hy + 0.1, hy - 0.1, z1 - apron, z1, rail),  # the beam under the middle
            *[box(-hx + 0.08, hx - 0.08, y - 0.02, y + 0.02, z1 - apron, z1, rail) for y in (-hy / 2, hy / 2)]
            ] + _turned_legs(hx, hy, z1, 0.032, 0.05, rail) + [
        box(-hx + 0.08, hx - 0.08, hy - 0.06, hy - 0.05, sz - TOP_T - apron, sz - TOP_T, rail),
        box(-hx + 0.08, hx - 0.08, -hy + 0.05, -hy + 0.06, sz - TOP_T - apron, sz - TOP_T, rail),
        box(hx - 0.06, hx - 0.05, -hy + 0.08, hy - 0.08, sz - TOP_T - apron, sz - TOP_T, rail),
        box(-hx + 0.05, -hx + 0.06, -hy + 0.08, hy - 0.08, sz - TOP_T - apron, sz - TOP_T, rail)]


def dining_table(s, c, zs):
    return _table(s, c, 0.07)


def coffee_table(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    shelf = [box(-hx + 0.06, hx - 0.06, -hy + 0.06, hy - 0.06, 0.1, 0.103, "raw"),  # raw board seen from below
             box(-hx + 0.06, hx - 0.06, -hy + 0.06, hy - 0.06, 0.103, 0.13, c + "*0.8", 0.003)]
    legs = _turned_legs(hx, hy, sz - TOP_T, 0.027, 0.045, c + "*0.8")
    return [box(-hx, hx, -hy, hy, sz - TOP_T, sz, c, 0.005),
            box(-hx + 0.005, hx - 0.005, -hy + 0.005, hy - 0.005, sz - TOP_T - 0.001, sz - TOP_T, "raw"), *shelf] + legs


def sofa(s, c, zs):
    """A sofa with its own cushions: a frame, two seat cushions whose tops are the seat height (the surface), two back
    cushions leaning against the back and clear of the seat's rect (which starts 5 mm behind the front of the back, so
    their faces stop 8 mm behind it), arms and feet. Every textile part is in the sofa's colour attribute, a little
    lighter or darker."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    seat_top = zs[0] if zs else 0.45
    inner = hy - 0.18
    back_x = -hx + 0.25  # the front of the back, where the seat starts
    parts = [box(-hx, hx, -hy, hy, 0.08, 0.28, c + "*0.75", 0.012),
             box(-hx, -hx + 0.12, -inner, inner, 0.28, sz, c + "*0.9", 0.02),
             box(-hx, hx, -hy, -inner, 0.08, 0.62, c + "*0.8", 0.035, 3),  # the arms
             box(-hx, hx, inner, hy, 0.08, 0.62, c + "*0.8", 0.035, 3)]
    for y0, y1 in ((-inner + 0.004, -0.004), (0.004, inner - 0.004)):
        parts.append(box(back_x, hx, y0, y1, 0.26, seat_top, c + "*1.1", 0.03, 3))  # seat cushion
        tilt, d, h = math.radians(8.0), 0.15, 0.36  # a back cushion leaning back, its front face behind back_x
        reach = (d * math.cos(tilt) + h * math.sin(tilt)) / 2
        parts.append(Part("box", (back_x - 0.008 - reach, (y0 + y1) / 2, seat_top + h / 2 * math.cos(tilt)),
                          (d, y1 - y0 - 0.01, h), (0.0, -tilt, 0.0), c + "*1.15", (), 0.035, 3))
    parts.append(box(-hx + 0.02, hx - 0.02, -hy + 0.02, hy - 0.02, 0.0795, 0.08, "cover"))  # the dust cloth
    return parts + _turned_legs(hx, hy, 0.08, 0.03, 0.06, "oak*0.45", 0.7)


def tv_console(s, c, zs):
    return _cabinet(s, c, c + "*0.8", plinth=0.05, handle_z=0.3)


def bookshelf(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx, -hy, -hy + 0.03, 0, sz, c, 0.003), box(-hx, hx, hy - 0.03, hy, 0, sz, c, 0.003),
             box(-hx, -hx + 0.012, -hy, hy, 0, sz, c + "*0.7"), box(-hx, hx, -hy, hy, 0, 0.04, c + "*0.8", 0.003),
             box(-hx, hx, -hy, hy, sz - 0.03, sz, c, 0.004)]
    parts += [box(-hx, hx, -hy + 0.03, hy - 0.03, z - 0.03, z, c, 0.003) for z in (zs or (0.4, 0.9, 1.4))]
    return parts


def rug(s, c, zs):
    """A flat slab in the rug's colour, rounded at the edge. Its pattern (a border, a lattice, a medallion, all in
    tones of the same colour) is a material layer, `wear.apply`'s: the rug is a receptacle whose surface is its top."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0, sz, c, min(0.003, sz / 3), 1)]


def hall_console(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, sz - TOP_T, sz, c, 0.005),
            box(-hx + 0.005, hx - 0.005, -hy + 0.005, hy - 0.005, sz - TOP_T - 0.001, sz - TOP_T, "raw"),
            box(-hx + 0.03, hx - 0.03, -hy + 0.06, hy - 0.06, sz - 0.16, sz - 0.158, "raw"),  # the drawer's raw bottom
            box(-hx + 0.03, hx - 0.03, -hy + 0.06, hy - 0.06, sz - 0.158, sz - TOP_T, c + "*0.85", EDGE),
            box(hx - 0.03, hx - 0.028, -hy + 0.07, hy - 0.07, sz - 0.155, sz - TOP_T - 0.005, "dark"),
            *_bar(hx - 0.03, 0.0, sz - 0.1, 0.12, False)] + _turned_legs(hx, hy, sz - TOP_T, 0.022, 0.04, c)


def shoe_rack(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, -hy + 0.025, 0, sz, c, 0.003), box(-hx, hx, hy - 0.025, hy, 0, sz, c, 0.003),
            box(-hx, hx, -hy, hy, 0.03, 0.055, c, 0.002), box(-hx, hx, -hy, hy, sz - 0.025, sz, c, 0.003),
            box(-hx, -hx + 0.01, -hy, hy, 0, sz, c + "*0.7")]


def floor_mat(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx, -hy, hy, 0, sz - 0.003, c + "*0.6")]  # the ribs are the surface, 3 mm over the base
    parts += [box(-hx + 0.06 + i * 0.18, -hx + 0.14 + i * 0.18, -hy + 0.06, hy - 0.06, 0, sz, c + "*1.3")
              for i in range(int((sx - 0.12) / 0.18))]
    return parts


def bed(s, c, zs):
    """A bed: a wooden frame and base, a mattress in the bed's colour, a duvet over the foot two thirds, and two
    pillows at the head, all in the bed's own colour (the top of the duvet and of the pillows is the surface height, and
    the duvet's soft rounded edge hangs over the mattress)."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    top = zs[0] if zs else 0.5
    duvet = c + "*0.93"
    parts = [box(-hx, hx, -hy, hy, 0.05, 0.26, "oak*0.7", 0.01),
             box(-hx + 0.02, hx - 0.02, -hy + 0.03, hy - 0.03, 0.26, 0.40, c, 0.02, 3),
             box(-hx + 0.5, hx - 0.015, -hy + 0.012, hy - 0.012, 0.30, top, duvet, 0.04, 3),
             box(hx - 0.62, hx - 0.30, -hy + 0.012, hy - 0.012, 0.30, top, c + "*0.85", 0.04, 3)]  # the fold
    for y0, y1 in ((-hy + 0.1, -0.03), (0.03, hy - 0.1)):
        parts.append(box(-hx + 0.05, -hx + 0.45, y0, y1, 0.37, top, c + "*1.04", 0.05, 3))
    parts.append(box(-hx + 0.02, hx - 0.02, -hy + 0.02, hy - 0.02, 0.0495, 0.05, "cover"))  # the base cloth
    return parts + _turned_legs(hx, hy, 0.05, 0.035, 0.05, "oak*0.45", 0.75)


def nightstand(s, c, zs):
    sx, sy, sz = s
    return _cabinet(s, c, c, plinth=0.06, drawers=2)


def desk(s, c, zs):
    """A writing desk on four legs: a top, a drawer apron under it of rails and a shallow centre drawer. The apron's
    underside, 0.60 m up, is the map's `under_m`; the legs stand in the corners and between them is knee space."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    z1 = sz - TOP_T
    z0, rail = z1 - 0.11, c + "*0.85"
    xf = hx - SET_BACK - DOOR_T  # the rails' front plane, with the drawer front
    parts = [box(-hx, hx, -hy, hy, z1, sz, c, 0.005),
             box(-hx + 0.005, hx - 0.005, -hy + 0.005, hy - 0.005, z1 - 0.001, z1, "raw")]  # the top's raw underside
    parts += _turned_legs(hx, hy, z1, 0.027, 0.04, c + "*0.8")
    parts += [box(-hx + 0.05, -hx + 0.13, -0.35, 0.35, z1 - 0.07, z1 - 0.01, "steel*0.55", 0.002),  # a cable tray
              box(-hx + 0.1, -hx + 0.11, -0.3, 0.3, z1 - 0.09, z1 - 0.07, "dark")]  # and the cables in it
    parts += [box(-hx + 0.03, -hx + 0.05, -hy + 0.06, hy - 0.06, z0, z1, rail),  # the back rail
              box(-hx + 0.06, hx - 0.06, -hy + 0.03, -hy + 0.05, z0, z1, rail),  # the side rails
              box(-hx + 0.06, hx - 0.06, hy - 0.05, hy - 0.03, z0, z1, rail),
              box(xf, xf + DOOR_T, -hy + 0.06, -0.3, z0, z1, rail),  # the front rail either side of the drawer
              box(xf, xf + DOOR_T, 0.3, hy - 0.06, z0, z1, rail),
              box(-hx + 0.05, xf - 0.001, -0.3, 0.3, z0, z1, "raw")]  # the drawer's box: raw board from below
    return parts + _fronts(hx, -0.3, 0.3, z0, z1, 1, rail, None, drawers=1)


def wardrobe(s, c, zs):
    """A wardrobe: a bevelled carcass on a dark plinth, two door fronts with a seam between them and a bar handle each
    beside the seam."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    xf = hx - 0.012  # the doors' back face
    parts = [box(-hx, xf - 0.001, -hy, hy, 0.06, sz, c, 0.008),
             box(-hx, hx - 0.03, -hy + 0.01, hy - 0.01, 0, 0.06, c + "*0.75"),
             box(xf - 0.001, xf, -hy + 0.003, hy - 0.003, 0.06, sz - 0.004, "dark")]
    for a, b, handle_y in ((-hy + 0.004, -0.003, -0.045), (0.003, hy - 0.004, 0.045)):
        parts.append(box(xf, hx, a, b, 0.065, sz - 0.01, c, EDGE))
        parts += _bar(hx, handle_y, 1.1, 0.4, True)
    return parts


def vanity(s, c, zs):
    sx, sy, sz = s
    hy = sy / 2
    return _cabinet(s, c, "stone", plinth=0.06, doors=2) + [
        box(-0.18, 0.04, -hy + 0.1, hy - 0.1, sz, sz + 0.004, "white*0.95"),
        box(-0.15, 0.01, -hy + 0.13, hy - 0.13, sz, sz + 0.006, "grey*0.5"),
        cyl(-0.2, 0, sz, sz + 0.16, 0.014, "steel"),
        box(-0.2, -0.07, -0.011, 0.011, sz + 0.14, sz + 0.17, "steel", 0.003)]


def toilet(s, c, zs):
    """A close-coupled toilet on a skirted base: a low plinth under the whole footprint (nothing of the floor shows
    under the bowl or the cistern), the cistern and a pedestal on it, the bowl with its seat ring."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0.0, 0.05, c + "*0.9", 0.012),
            box(-hx, -hx + 0.17, -hy + 0.03, hy - 0.03, 0.05, sz, c, 0.01),
            box(-hx + 0.1, 0.05, -0.1, 0.1, 0.0, 0.26, c, 0.01),
            ball(0.07, 0, 0.27, 0.23, 0.18, 0.11, c), ring(0.07, 0, 0.375, 0.15, 0.022, "z", c + "*0.85"),
            box(-hx + 0.17, -hx + 0.19, -0.05, 0.05, 0.2, 0.26, "steel", 0.003)]


def bathtub(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    t = 0.07
    return [box(-hx, hx, -hy, hy, 0, 0.12, c, 0.01), box(-hx, hx, -hy, -hy + t, 0, sz, c, 0.012),
            box(-hx, hx, hy - t, hy, 0, sz, c, 0.012), box(-hx, -hx + t, -hy, hy, 0, sz, c, 0.012),
            box(hx - t, hx, -hy, hy, 0, sz, c, 0.012),
            box(-hx + t, hx - t, -hy + t, hy - t, 0.12, 0.125, "clear*0.9"),
            cyl(0, hy - 0.18, sz, sz + 0.18, 0.016, "steel"),
            box(0.0, 0.1, hy - 0.19, hy - 0.17, sz + 0.16, sz + 0.19, "steel", 0.003)]


def bath_shelf(s, c, zs):
    return bookshelf(s, c, zs or (0.5, 1.0, 1.4))


def chair(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0.42, 0.46, c, 0.004)] + _turned_legs(hx, hy, 0.42, 0.024, 0.03, c + "*0.8") + [
        box(-hx, -hx + 0.035, -hy + 0.02, -hy + 0.07, 0.46, sz, c + "*0.8", 0.003),
        box(-hx, -hx + 0.035, hy - 0.07, hy - 0.02, 0.46, sz, c + "*0.8", 0.003),
        box(-hx, -hx + 0.03, -hy + 0.02, hy - 0.02, 0.62, 0.86, c, 0.004)]


def stool(s, c, zs):
    sx, sy, sz = s
    r = sx / 2
    parts = [cyl(0, 0, sz - 0.05, sz, r, c)]
    for ix in (-1, 1):
        for iy in (-1, 1):
            parts.append(cyl(ix * (r - 0.06), iy * (r - 0.06), 0, sz - 0.05, 0.016, c + "*0.8"))
    return parts


def laundry_basket(s, c, zs):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    t = 0.02
    return [box(-hx, hx, -hy, hy, 0, 0.03, c + "*0.8"), box(-hx, hx, -hy, -hy + t, 0, sz * 0.92, c),
            box(-hx, hx, hy - t, hy, 0, sz * 0.92, c), box(-hx, -hx + t, -hy, hy, 0, sz * 0.92, c),
            box(hx - t, hx, -hy, hy, 0, sz * 0.92, c), box(-hx, hx, -hy, hy, sz * 0.92, sz, c + "*0.7"),
            ball(-0.04, 0.03, sz - 0.07, 0.11, 0.1, 0.07, "white"),
            ball(0.07, -0.05, sz - 0.1, 0.09, 0.09, 0.07, "blue*0.9")]


def dock(s, c, zs):
    """A robot vacuum's dock: a low base plate across the footprint, a back block to the full height with a lighter cap,
    and two steel charging contacts on its front. Solid from the floor."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    front = -hx + 0.07  # the front face of the back block
    parts = [box(-hx, hx, -hy, hy, 0, 0.012, c + "*0.8", 0.004),
             box(-hx, front, -hy + 0.02, hy - 0.02, 0.012, sz - 0.003, c, 0.01, 3),
             box(-hx + 0.005, front - 0.005, -hy + 0.035, hy - 0.035, sz - 0.003, sz, c + "*1.5", 0.002)]
    parts += [box(front, front + 0.004, y - 0.012, y + 0.012, 0.03, 0.042, "steel") for y in (-0.06, 0.06)]
    parts += [box(front, front + 0.0015, -0.07, 0.07, z, z + 0.0035, "dark*0.6")  # the vent slots
              for z in (0.062, 0.07, 0.078, 0.086)]
    parts.append(box(front, front + 0.002, -0.03, 0.03, 0.048, 0.056, "white*0.95", 0.001))  # the indicator's lens
    parts += [box(front + 0.012 + 0.014 * i, front + 0.017 + 0.014 * i, -hy + 0.035, hy - 0.035, 0.012, 0.0145,
                  "dark*1.3", 0.001) for i in range(3)]  # grip ribs on the plate, 2.5 mm over it
    return parts


def _laundry_unit(hx: float, hy: float, z0: float, z1: float, c: str) -> list[Part]:
    """One appliance of the stack: a bevelled body, a control strip across the top of its front (a dark panel, a knob
    and two buttons) and a round door below it (a white bezel, a steel ring and a dome of dark glass)."""
    zc, panel_z = z0 + 0.4, z1 - 0.07  # the door's centre, the panel's
    turned = (0.0, math.pi / 2, 0.0)  # a disc whose axis points out of the front
    return [box(-hx, hx, -hy, hy, z0, z1, c, 0.008),
            box(hx - 0.002, hx + 0.003, 0.09, 0.2, z0 + 0.03, z0 + 0.075, c + "*0.9", 0.002),  # the lint filter's door
            box(hx + 0.003, hx + 0.004, 0.09, 0.2, z0 + 0.0515, z0 + 0.0535, "dark*0.7"),  # and its seam
            box(hx - 0.002, hx + 0.004, -hy + 0.03, hy - 0.03, panel_z - 0.045, panel_z + 0.045, "dark", 0.002),
            Part("cyl", (hx + 0.014, -0.18, panel_z), (0.056, 0.056, 0.02), turned, "steel", (1.0,)),
            *[box(hx + 0.004, hx + 0.01, y - 0.02, y + 0.02, panel_z - 0.012, panel_z + 0.012, "steel", 0.002)
              for y in (0.1, 0.19)],
            Part("cyl", (hx + 0.005, 0, zc), (0.44, 0.44, 0.012), turned, c + "*0.95", (1.0,)),
            ring(hx + 0.01, 0, zc, 0.2, 0.014, "x", "steel"),
            ball(hx, 0, zc, 0.03, 0.185, 0.185, "dark")]


def washer_dryer(s, c, zs):
    """A stacked washer-dryer: the washer under the dryer, a dark seam between them, each with its round door and its
    control strip, on a dark base. Solid from the floor."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    half = (sz + 0.02) / 2  # the seam's middle
    return [box(-hx + 0.02, hx - 0.02, -hy + 0.02, hy - 0.02, 0, 0.03, "dark"),
            *_laundry_unit(hx, hy, 0.02, half - 0.005, c),
            box(-hx + 0.01, hx - 0.006, -hy + 0.01, hy - 0.01, half - 0.006, half + 0.006, "dark"),
            *_laundry_unit(hx, hy, half + 0.005, sz, c)]


BUILDERS = {f.__name__: f for f in (
    fridge, counter, sink_unit, stove, dining_table, coffee_table, sofa, tv_console, bookshelf, rug, hall_console,
    shoe_rack, floor_mat, bed, nightstand, desk, wardrobe, vanity, toilet, bathtub, chair, stool, laundry_basket,
    dock, washer_dryer)}
BUILDERS["bath_shelf"] = bath_shelf
BUILDERS |= CITY
BUILDERS |= HOUSE
