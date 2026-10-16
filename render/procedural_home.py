# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's small procedural kinds: a glass, a bottle, a book, a remote, a laptop, a plant and a laundry basket.

Runs inside Blender's own Python and is never imported by `stbench`. Each builder takes the bmesh and the kind's
`DESIGN_BOX` and returns nothing: a mesh in metres with its origin at the base centre (x forward, y left, z up) and a
material slot index per `procedural.SLOTS` entry. `procedural.py` registers them and fits each to its box; every one
is built to its box to a millimetre (a test keeps it so), so the fit never squashes it. Slot numbers are the order of
`procedural.SLOTS`.
"""

from __future__ import annotations

import math

from mathutils import Matrix, Vector
from meshkit import blade, lathe, placed, rod, slab
from partskit import Outline, loft, rings_of
from wobble import bump


def glass(bm, box) -> None:
    """A tumbler: a thick base with a rounded foot, walls widening to a rolled rim."""
    r, h, t = box[0] / 2, box[2], 0.0035
    lathe(bm, [(0, 0.001), (0.7 * r, 0.001), (0.76 * r, 0.0), (0.79 * r, 0.004), (0.8 * r, 0.008), (0.93 * r, 0.7 * h),
               (0.99 * r, h - 0.012), (r, h - 0.004), (r, h), (r - t, h), (r - t - 0.0006, h - 0.006),
               (0.8 * r - t, 0.018), (0.55 * r, 0.014), (0, 0.0125)], 0)


def bottle(bm, box) -> None:
    """A wine-style bottle: a punted foot, a straight body, shoulders that curve into a long neck with a lip, a paper
    label round the body and a foil capsule at the top."""
    r, h = box[0] / 2 - 0.0004, box[2]
    shoulder = [(r - (r - 0.36 * r) * (0.5 - 0.5 * math.cos(math.pi * t)), 0.62 * h + 0.16 * h * t)
                for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    body = [(0.0, 0.007), (0.45 * r, 0.004), (0.8 * r, 0.0), (0.93 * r, 0.004), (r, 0.014), (r, 0.62 * h), *shoulder,
            (0.36 * r, 0.93 * h), (0.42 * r, 0.945 * h), (0.42 * r, 0.96 * h), (0.38 * r, 0.965 * h),
            (0.38 * r, 0.99 * h),
            (0.0, h)]
    lathe(bm, body, 0)
    lathe(bm, [(r + 0.0004, 0.24 * h), (r + 0.0004, 0.5 * h)], 1)  # the paper label: a band round the body


def book(bm, box) -> None:
    """A hardback: two boards a little larger than the page block, a rounded spine and a page block with a stepped
    fore-edge."""
    hx, hy, sz, t, spine = box[0] / 2, box[1] / 2, box[2], 0.0016, 0.006
    for z0, z1 in ((0, t), (sz - t, sz)):
        slab(bm, -hx, hx, -hy, hy, z0, z1, 0, bevel=0.0008, segments=2)
    slab(bm, -hx, hx, -hy, -hy + spine, 0, sz, 0, bevel=0.003, segments=3)
    steps = 6
    for i in range(steps):  # the pages: stepped, so that the edge catches light as a stack of leaves
        z0, z1 = t + (sz - 2 * t) * i / steps, t + (sz - 2 * t) * (i + 1) / steps
        inset = 0.003 + (0.0004 if i % 2 else 0.0)
        slab(bm, -hx + inset, hx - inset, -hy + spine, hy - inset, z0, z1, 1)


def remote(bm, box) -> None:
    """A TV remote: a rounded body, an infrared window in its end, a round power key, a ring with a centre key, a row of
    small round keys and a number pad of oblong ones; the keys stand 1.5 mm proud."""
    hx, hy, sz = box[0] / 2, box[1] / 2, box[2]
    top = sz - 0.0015
    slab(bm, -hx, hx, -hy, hy, 0, top, 0, bevel=0.005, segments=3)
    slab(bm, hx - 0.0025, hx, -0.012, 0.012, 0.004, top - 0.002, 1, bevel=0.0006, segments=1)  # the infrared window

    def key(x: float, y: float, r: float) -> None:
        rod(bm, (x, y, top - 0.0002), (x, y, sz), r, 1, segments=14)

    key(0.067, 0.0, 0.0048)
    lathe(bm, [(0.0105, top - 0.0002), (0.0105, sz), (0.0065, sz), (0.0065, top - 0.0002)], 1, segments=24, loop=True,
          offset=(0.032, 0.0))
    key(0.032, 0.0, 0.0045)
    for y in (-0.0135, 0.0, 0.0135):
        key(0.006, y, 0.0036)
    for row in range(4):
        for col in (-1, 0, 1):
            x0, y0 = -0.012 - 0.0145 * row, col * 0.0135
            slab(bm, x0 - 0.0045, x0 + 0.0045, y0 - 0.0052, y0 + 0.0052, top - 0.0002, sz, 1, bevel=0.0007, segments=1)
    for y0 in (-0.0105, 0.0105):  # and two wider keys below the pad
        slab(bm, -0.0765, -0.0685, y0 - 0.0075, y0 + 0.0075, top - 0.0002, sz, 1, bevel=0.0007, segments=1)


def laptop(bm, box) -> None:
    """Open, the way one sits on a table: a base 0.33 wide (y) x 0.23 deep (x) x 0.016 with a dark key well holding 70
    keys, a trackpad and a hinge barrel on top, and a lid 0.33 x 0.22 x 0.007 hinged along the base's back edge and
    opened 110 degrees (20 past vertical), its inner face a dark screen with a 10 mm bezel turned towards the user (+x).
    The dimensions are the model's own, so `box` is not used: `DESIGN_BOX` is this geometry's extents (a test keeps them
    equal)."""
    hx, hy, base, lid, thick, bezel = 0.115, 0.165, 0.016, 0.22, 0.007, 0.01
    slab(bm, -hx, hx, -hy, hy, 0, base, 0, bevel=0.0022)
    slab(bm, -0.105, 0.03, -0.15, 0.15, base - 0.002, base + 0.0005, 1)  # the key well, 0.5 mm proud of the deck
    pitch = 0.0195
    for row in range(5):  # keys: 14 across, five rows, and a space bar row
        x = -0.1 + 0.0215 * row
        for col in range(14):
            y = (col - 6.5) * pitch
            slab(bm, x - 0.0075, x + 0.0075, y - 0.0082, y + 0.0082, base + 0.0004, base + 0.0017, 1, bevel=0.0012,
                 segments=1)
    x = -0.1 + 0.0215 * 5
    slab(bm, x - 0.0075, x + 0.0075, -0.07, 0.07, base + 0.0004, base + 0.0017, 1, bevel=0.0012, segments=1)  # space
    slab(bm, 0.05, 0.1, -0.055, 0.055, base - 0.001, base + 0.0003, 0, bevel=0.0005, segments=1)  # the trackpad
    rod(bm, (-hx + 0.0045, -0.14, base + 0.0035), (-hx + 0.0045, 0.14, base + 0.0035), 0.0035, 1, segments=10)  # hinge
    hinge = Matrix.Translation((-hx + 0.005, 0, base)) @ Matrix.Rotation(-math.radians(110.0), 4, "Y")

    def the_lid() -> None:  # flat on the base, the hinge at x = 0 and the inner face at z = 0
        slab(bm, 0, lid, -hy, hy, 0, thick, 0, bevel=0.0015, segments=2)
        slab(bm, bezel, lid - bezel, -hy + bezel, hy - bezel, -0.0005, 0.0001, 2)  # the screen, 0.5 mm proud

    placed(bm, hinge, the_lid)


def squeeze(bm, before: set, rmax: float, top: float, z0: float) -> None:
    """Scale the vertices added since `before` about the axis through the foot, so that they reach no further than
    `rmax` from it and exactly `top` high: a leaf's tips are placed by their angles and lengths, and this puts them in
    the box."""
    new = [v for v in bm.verts if v not in before]
    reach = max(math.hypot(v.co.x, v.co.y) for v in new)
    high = max(v.co.z for v in new)
    sr, sz = (min(1.0, rmax / reach), (top - z0) / (high - z0))
    for v in new:
        v.co = Vector((v.co.x * sr, v.co.y * sr, z0 + (v.co.z - z0) * sz))


def plant(bm, box) -> None:
    """A succulent in a pot: a tapered pot with a rolled rim, a disc of soil and three whorls of pointed leaves whose
    reach stays inside the pot's rim and whose tallest tip is the box's top."""
    r, h = box[0] / 2, box[2]
    rim = 0.38 * h
    lathe(bm, [(0, 0), (0.62 * r, 0), (0.66 * r, 0.02 * rim), (0.96 * r, 0.9 * rim), (r, 0.96 * rim), (r, rim),
               (0.95 * r, rim), (0.91 * r, 0.86 * rim)], 0)
    lathe(bm, [(0, 0.88 * rim), (0.93 * r, 0.86 * rim)], 1)
    before = set(bm.verts)
    for k in range(30):  # three whorls, the inner ones taller and more upright
        whorl = 0 if k < 6 else 1 if k < 16 else 2
        f = [0.0, 0.5, 1.0][whorl]
        blade(bm, 137.508 * k, 6 + 30 * f, 5 + 14 * f, h * (0.64 - 0.28 * f), 0.03 - 0.008 * f, 0.007,
              (0.004 * math.cos(k), 0.004 * math.sin(k), 0.9 * rim), 2)
    squeeze(bm, before, r - 0.003, h, 0.9 * rim)


def laundry_basket(bm, box) -> None:
    """A wicker basket that widens to a rolled rim, its wall ribbed by the stakes of the weave and banded where the
    withies cross, on a foot ring, with a heap of cloth showing above the rim. The heap's crown is the box's top."""
    r, h = box[0] / 2, box[2]
    rim_z = 0.92 * h
    outline = Outline(0, 0, 48)  # a round ring of 192 points; a stake every four

    def radius(z: float) -> float:  # the wall before its rim, whose bead stands 9 mm proud and reaches the box
        return (r - 0.0071) * (0.85 + 0.15 * min(1.0, z / rim_z))

    def push(z: float, u: float, p: tuple) -> float:
        d = 0.0
        if 0.03 < z < rim_z - 0.03:
            d += 0.0028 * bump(((u * 48) % 1.0) - 0.5, 0.42)  # a rib for each stake
        if 0.04 < z < rim_z - 0.04:
            d += 0.0018 * bump(((z / 0.045) % 1.0) - 0.5, 0.3)  # and a band where the withies are woven across
        d += 0.009 * bump((z - (rim_z - 0.006)) / 0.02, 1.0) if z > rim_z - 0.03 else 0.0  # the rolled rim
        return d + 0.006 * bump((z - 0.012) / 0.014, 1.0)  # the foot ring

    zs = [i * 0.0115 for i in range(int(rim_z / 0.0115))] + [rim_z]
    wall = rings_of(outline, zs, lambda z: (radius(z),) * 2 + (radius(z),), push)
    floor = [rim_z - i * 0.02 for i in range(int((rim_z - 0.05) / 0.02) + 1)] + [0.05]
    inner = rings_of(outline, floor, lambda z: ((radius(z) - 0.0145),) * 2 + (radius(z) - 0.0145,))
    loft(bm, wall + inner, 0, cap_first=True, cap_last=True)
    for cx, cy, rad, top, z in ((-0.03, 0.02, 0.62, 1.0, 0.83 * h), (0.05, -0.04, 0.5, 0.9, 0.81 * h)):
        lathe(bm, [(0, z - 0.1 * h * top), (0.55 * r * rad, z - 0.07 * h * top), (0.9 * r * rad, z),
                   (0.75 * r * rad, z + 0.1 * h * top), (0.35 * r * rad, z + 0.16 * h * top), (0, z + 0.17 * h * top)],
              1, segments=24, offset=(cx * r, cy * r))  # a lump of cloth


BUILDERS = {"glass": glass, "bottle": bottle, "book": book, "remote": remote, "laptop": laptop, "plant": plant,
            "laundry_basket": laundry_basket}
