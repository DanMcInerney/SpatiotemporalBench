# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's equipment as procedural geometry: a pallet jack (`procedural_jack.py`) and a roll cage (light
furniture), a forklift (`procedural_forklift.py`) and a tote robot (the movers).

Runs inside Blender's own Python and is never imported by `stbench`. Builders are called as in `procedural_goods.py`
(bmesh, `DESIGN_BOX`, slot dict) and build to the box exactly: a mover is fitted to its catalog footprint, and the
forklift's forks, rear counterweight and tyres are what reach its extents. The colour attribute is the painted body
(the jack's, the cage's frame, the forklift's and the robot's); tyres, masts and screens are neutral.
"""

from __future__ import annotations

from meshkit import rod, solid, wheel
from partskit import rounded_path, sweep
from procedural_forklift import forklift
from procedural_jack import pallet_jack


def caster(bm, cx: float, cy: float, wheels: int, frame: int, swivel: bool) -> None:
    """A 125 mm caster under the cage's base: a polyurethane wheel between two bracket plates, on a square plate."""
    r, w = 0.0625, 0.032
    wheel(bm, cx, cy, r, r, w, "y", wheels, segments=20)
    for s in (-1.0, 1.0):
        y = cy + s * (w / 2 + 0.003)
        solid(bm, cx - 0.03, cx + 0.03, y - 0.0015, y + 0.0015, 0.04, 0.125, frame)
    solid(bm, cx - 0.045, cx + 0.045, cy - 0.04, cy + 0.04, 0.115, 0.125, frame, bevel=0.003, segments=1)
    if swivel:  # the swivel's bearing
        rod(bm, (cx, cy, 0.105), (cx, cy, 0.118), 0.022, wheels, segments=12)


def roll_cage(bm, box, slot) -> None:
    """A roll container: a steel base plate with a channel frame on four casters (two fixed, two swivelling), four
    round corner posts, rails, and welded wire mesh (horizontal wires outside, vertical wires inside) on the two long
    sides and the back, stiffened by a mid rail; the front (+x) is open but for a rail across its top."""
    frame, wheels = slot["frame"], slot["wheels"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    z0, post, wire = 0.15, 0.0125, 0.003  # the top of the base, the posts' radius, a wire's half thickness
    solid(bm, -hx, hx, -hy, hy, 0.125, z0, frame, bevel=0.004, segments=1)  # the base plate
    for s in (-1.0, 1.0):  # channel rails under its long edges and ends
        solid(bm, -hx, hx, s * (hy - 0.025) - 0.025, s * (hy - 0.025) + 0.025, 0.1, 0.125, frame)
        solid(bm, s * (hx - 0.025) - 0.025, s * (hx - 0.025) + 0.025, -hy, hy, 0.1, 0.125, frame)
    solid(bm, -hx, hx, -hy, hy, z0, z0 + 0.012, frame, bevel=0.004, segments=1)  # the raised rim round the deck
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            caster(bm, sx * (hx - 0.075), sy * (hy - 0.075), wheels, frame, swivel=sx > 0)
            rod(bm, (sx * (hx - post), sy * (hy - post), z0), (sx * (hx - post), sy * (hy - post), h), post, frame,
                segments=12)  # the corner posts
    levels = [z0 + 0.1 + 0.1 * i for i in range(15)]
    for z in levels:  # horizontal wires, on the outside of each panel
        for sy in (-1.0, 1.0):
            solid(bm, -hx + post, hx - post, sy * (hy - 2 * wire) - wire, sy * (hy - 2 * wire) + wire, z, z + 2 * wire,
                  frame)
        solid(bm, -hx + wire, -hx + 3 * wire, -hy + post, hy - post, z, z + 2 * wire, frame)
    for i in range(1, 11):  # vertical wires on the long sides, inside the horizontals
        x = -hx + 2 * hx * i / 11
        for sy in (-1.0, 1.0):
            y = sy * (hy - 5 * wire)
            solid(bm, x - wire, x + wire, y - wire, y + wire, z0 + 0.012, h - 0.01, frame)
    for i in range(1, 9):  # and on the back
        y = -hy + 2 * hy * i / 9
        solid(bm, -hx + 4 * wire, -hx + 6 * wire, y - wire, y + wire, z0 + 0.012, h - 0.01, frame)
    for z in (0.9, h - post):  # a mid rail and the top rail on three sides, and across the open front at the top
        rail = [(hx - post, -hy + post, z), (-hx + post, -hy + post, z), (-hx + post, hy - post, z),
                (hx - post, hy - post, z)]
        sweep(bm, rounded_path(rail, 0.03, 4), post * 0.8, frame, segments=8)
    rod(bm, (hx - post, -hy + post, h - post), (hx - post, hy - post, h - post), post * 0.8, frame, segments=8)


def tote_robot(bm, box, slot) -> None:
    """An autonomous mobile robot for totes: a low rounded chassis with a dark bumper band and four wheels, a lidar
    puck and two safety scanners at the front, a tall mast frame at the back (two columns, cross beams, a head with a
    status screen and a light bar) and the carrying deck, a plate with raised side lips, that the totes ride on. Front
    is +x."""
    body, wheels, screen = slot["body"], slot["wheels"], slot["screen"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx + 0.006, hx - 0.006, -hy + 0.006, hy - 0.006, 0.105, 0.3, body, bevel=0.04, segments=4)  # the chassis
    solid(bm, -hx, hx, -hy, hy, 0.07, 0.105, wheels, bevel=0.015, segments=3)  # the bumper band
    solid(bm, -hx + 0.02, hx - 0.02, -hy + 0.02, hy - 0.02, 0.28, 0.31, body, bevel=0.012, segments=2)  # a cover plate
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            wheel(bm, sx * (hx - 0.12), sy * (hy - 0.03), 0.06, 0.06, 0.05, "y", wheels, segments=20)
    for s in (-1.0, 1.0):  # the safety scanners, at the front corners
        solid(bm, hx - 0.012, hx, s * 0.2 - 0.07, s * 0.2 + 0.07, 0.17, 0.25, screen, bevel=0.004, segments=1)
    for s in (-1.0, 1.0):  # the mast's two columns, joined by beams
        solid(bm, -hx, -hx + 0.25, s * (hy - 0.035) - 0.035, s * (hy - 0.035) + 0.035, 0.3, h - 0.08, body, bevel=0.012,
              segments=2)
    solid(bm, -hx, -hx + 0.25, -hy, hy, h - 0.1, h, body, bevel=0.025, segments=3)  # the head
    solid(bm, -hx, -hx + 0.25, -hy + 0.03, hy - 0.03, 0.72, 0.8, body, bevel=0.01, segments=2)  # a mid beam
    solid(bm, -hx, -hx + 0.25, -hy + 0.03, hy - 0.03, 0.42, 0.5, body, bevel=0.01, segments=2)
    solid(bm, -hx + 0.25, -hx + 0.256, -0.15, 0.15, 1.0, 1.2, screen, bevel=0.003, segments=1)  # the status screen
    solid(bm, -hx + 0.25, -hx + 0.256, -0.2, 0.2, h - 0.04, h - 0.025, screen)  # the light bar
    solid(bm, -hx + 0.25, hx - 0.05, -hy + 0.03, hy - 0.03, 0.52, 0.56, body, bevel=0.004, segments=1)  # the deck
    for s in (-1.0, 1.0):  # raised lips along it
        y = s * (hy - 0.04)
        solid(bm, -hx + 0.25, hx - 0.05, y - 0.01, y + 0.01, 0.56, 0.6, body)
    rod(bm, (hx - 0.1, 0.0, 0.31), (hx - 0.1, 0.0, 0.38), 0.05, screen, segments=24)  # the lidar puck
    rod(bm, (hx - 0.1, 0.0, 0.38), (hx - 0.1, 0.0, 0.395), 0.03, wheels, segments=24)


BUILDERS = {"pallet_jack": pallet_jack, "roll_cage": roll_cage, "forklift": forklift, "tote_robot": tote_robot}
