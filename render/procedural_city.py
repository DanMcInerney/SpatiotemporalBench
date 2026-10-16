# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's street kinds as procedural geometry: a construction skip, a front-loading dumpster, a portable toilet
and a street-vendor food cart (the vehicles are in `procedural_vehicles.py`).

Runs inside Blender's own Python and is never imported by `stbench`. Builders are called as in `procedural_goods.py`
(bmesh, `DESIGN_BOX`, slot dict) and build in the kind's local frame (origin at the base centre, x forward, y left, z
up); `procedural.build` then fits the result to the catalog box exactly. The colour attribute is the skip's and the
dumpster's painted steel, the toilet's walls and the cart's umbrella (its second set of gores a lighter tint of it).
"""

from __future__ import annotations

import math
import random

from meshkit import prism, rod, tub, wheel
from vehicle_kit import lump, solid, to_box


def skip(bm, box, S) -> None:
    """A construction skip: an open tub that widens upwards, lifting lugs, yellow reflective plates on the sloping ends,
    and a heap of rubble, planks and bricks that rises above the rim."""
    hx, hy, h = 1.5, .85, 1.0
    tub(bm, hx, hy, 0.0, h, .75, .05, .06, S["body"])
    for sx in (-1.0, 1.0):
        wall = hx * (.75 + .25 * .65)  # the end wall's outer face at the plate's height
        solid(bm, sx * wall - .015, sx * wall + .015, -.45, .45, .5, .8, S["hazard"])
        rod(bm, (sx * (hx + .02), -.25, h - .02), (sx * (hx + .02), .25, h - .02), .03, S["trim"])
    rng = random.Random(7)
    for i in range(34):
        x, y = rng.uniform(-1.15, 1.15), rng.uniform(-.55, .55)
        z = .22 + .75 * (1 - (abs(x) / 1.4) ** 2) * (1 - (abs(y) / .75) ** 2) + rng.uniform(0, .12)
        size = [rng.uniform(.25, .55), rng.uniform(.2, .45), rng.uniform(.12, .35)]
        slot = S[("debris_a", "debris_b", "debris_c", "debris_a")[i % 4]]
        lump(bm, (x, y, min(z, 1.02)), size, (rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(0, 3)), slot,
             sphere=i % 5 == 0)
    for i in range(4):  # planks lying across the heap
        lump(bm, (rng.uniform(-.7, .7), rng.uniform(-.4, .4), .98 + .03 * i), (1.1, .12, .05),
             (0, .05 * i, rng.uniform(-1.4, 1.4)), S["wood"])


def dumpster(bm, box, S) -> None:
    """A front-loading waste container: a slightly tapered steel body with two ribs, two sloping black plastic lids,
    fork pockets in the sides and four small castors."""
    prism(bm, [(-1.0, .17), (1.0, .17), (1.0, 1.10), (-1.0, 1.16)], -.72, .72, S["body"])
    for z in (.50, .85):
        solid(bm, -1.01, 1.01, -.735, .735, z, z + .05, S["body"], bevel=.01, segments=1)
    prism(bm, [(-1.04, 1.12), (.04, 1.26), (.04, 1.30), (-1.04, 1.16)], -.745, .745, S["lid"])  # rear lid, hinged up
    prism(bm, [(.0, 1.12), (1.04, 1.12), (1.04, 1.15), (.0, 1.16)], -.745, .745, S["lid"])
    solid(bm, -.55, .55, -.745, .745, .24, .34, S["trim"])  # fork pockets
    rod(bm, (1.04, -.70, 1.20), (1.04, .70, 1.20), .018, S["trim"])
    for sx in (-.85, .85):
        for sy in (-.55, .55):
            wheel(bm, sx, sy, .085, .085, .07, "y", S["tyres"])


def portable_toilet(bm, box, S) -> None:
    """A moulded plastic cabin 1.1 x 1.1 m with a grey base, ribbed walls tinted by the colour, a light translucent roof
    that slopes down to the door side (+x), a framed door with a handle and an indicator, louvres above the door and in
    the side walls, and a vent stack on the roof."""
    body, roof, base, trim, ind = S["body"], S["roof"], S["base"], S["trim"], S["indicator"]
    solid(bm, -.55, .55, -.55, .55, .10, 2.06, body, bevel=.05)
    solid(bm, -.56, .56, -.56, .56, 0.0, .14, base, bevel=.03)
    prism(bm, [(-.58, 2.08), (.58, 2.08), (.58, 2.14), (-.58, 2.20)], -.58, .58, roof)  # roof, 20 cm taller at the back
    rod(bm, (-.40, -.40, 2.18), (-.40, -.40, 2.30), .05, trim, segments=10)  # vent stack
    for y0, y1 in ((-.42, -.40), (.40, .42)):  # the door's frame on the +x face
        solid(bm, .55, .565, y0, y1, .18, 1.96, trim)
    solid(bm, .55, .565, -.42, .42, 1.94, 1.96, trim)
    solid(bm, .55, .565, -.42, .42, .18, .20, trim)
    solid(bm, .55, .58, .26, .34, .90, 1.02, trim, bevel=.01, segments=1)  # handle
    solid(bm, .55, .575, .26, .34, 1.12, 1.20, ind, bevel=.005, segments=1)  # the occupied indicator
    solid(bm, .55, .565, -.30, .30, 2.00, 2.04, trim)  # louvres over the door
    for s in (-1.0, 1.0):
        solid(bm, -.30, .30, s * .55, s * .565, 1.94, 1.98, trim)  # side vents
        for z in (.55, 1.05, 1.55):
            solid(bm, -.46, .46, s * .55, s * .568, z, z + .035, body)  # ribs of the moulding
    solid(bm, -.565, -.55, -.30, .30, 1.94, 1.98, trim)


def _canopy(bm, rx: float, ry: float, z_top: float, z_rim: float, gores: int, slots) -> None:
    """An umbrella: `gores` panels from the tip to a rim of radii (rx, ry), alternating `slots`; a short valance hangs
    from the rim. A single sheet: Cycles shades both sides."""
    tip = bm.verts.new((0.0, 0.0, z_top))
    rings = []
    for r_f, z in ((.45, z_top - .07 * (z_top - z_rim) / .3), (1.0, z_rim), (1.0, z_rim - .08)):
        rings.append([bm.verts.new((rx * r_f * math.cos(2 * math.pi * k / gores),
                                    ry * r_f * math.sin(2 * math.pi * k / gores), z)) for k in range(gores)])
    for k in range(gores):
        j = (k + 1) % gores
        slot = slots[k % 2]
        f = bm.faces.new([tip, rings[0][k], rings[0][j]])
        f.material_index = slot
        for a, b in zip(rings, rings[1:], strict=False):
            f = bm.faces.new([a[k], b[k], b[j], a[j]])
            f.material_index = slot


def food_cart(bm, box, S) -> None:
    """A street-vendor cart: a steel body on two big wheels with a counter, dark glass display panels on both sides, a
    steel pole and a large umbrella whose gores alternate the colour and a lighter tint of it. The box is 1.0 m wide, so
    the canopy is an oval 2.0 x 1.0 m."""
    steel, glass, tyres, trim, hub = S["steel"], S["glass"], S["tyres"], S["trim"], S["hub"]
    solid(bm, -.70, .70, -.36, .36, .40, .90, steel, bevel=.025)
    solid(bm, -.76, .76, -.40, .40, .90, .95, steel, bevel=.015)  # counter top
    solid(bm, -.60, .60, -.37, .37, .12, .15, trim)  # lower shelf
    for s in (-1.0, 1.0):
        solid(bm, -.52, .52, s * .36, s * .375, .50, .84, glass)  # display panels
        wheel(bm, -.15, s * .45, .34, .34, .06, "y", tyres)
        wheel(bm, -.15, s * .455, .34, .17, .07, "y", hub)
        rod(bm, (.62, s * .30, .40), (.62, s * .30, .0), .025, trim, segments=6)  # front legs
        rod(bm, (-.70, s * .30, .80), (-.98, s * .30, .72), .025, steel, segments=6)  # handle
    rod(bm, (-.15, -.45, .34), (-.15, .45, .34), .03, trim, segments=6)  # axle
    rod(bm, (-.98, -.30, .72), (-.98, .30, .72), .025, steel, segments=6)
    rod(bm, (0.0, 0.0, .95), (0.0, 0.0, 2.15), .03, steel, segments=8)  # pole
    _canopy(bm, 1.0, .5, 2.20, 1.78, 8, (S["umbrella"], S["umbrella2"]))


BUILDERS = {"skip": to_box(skip), "dumpster": to_box(dumpster), "portable_toilet": to_box(portable_toilet),
            "food_cart": to_box(food_cart)}
