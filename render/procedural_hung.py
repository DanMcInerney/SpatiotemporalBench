# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What holds clothes that hang, as procedural clutter: a coat on a hanger (it hangs on the front of a wardrobe)
and a clothes-drying rack with washing on it (it stands on the floor).

Runs inside Blender's own Python and is never imported by `stbench`. Like `procedural_clutter.py`, a builder takes the
bmesh, its `DESIGN_BOX` (metres) and `slot` and adds geometry with its origin at the base centre; it reaches its box to
a millimetre (a test keeps it so). A thing that hangs on a piece's front (`catalog_clutter`'s `front:<piece kind>`)
stands with its back against the face: x is its depth, y its width and z its height, the face behind -x. The coat is
black, brown or beige and no tracked towel, bag or box is any of those.
"""

from __future__ import annotations

import math

from meshkit import rod, solid
from partskit import loft, sweep
from procedural_carried import curve

DESIGN_BOX = {"hanging_coat": (0.081, 0.459, 0.795), "drying_rack": (0.55, 0.9, 1.005)}
SLOTS = {"hanging_coat": ("cloth", "hanger"), "drying_rack": ("frame", "cloth")}


def hanging_coat(bm, box, slot) -> None:
    """A coat on a hanger: a wire hanger and hook at the top, sloping shoulders, a body that flares to the hem, and a
    sleeve hanging down each side, all a hand deep."""
    cloth, hanger = slot["cloth"], slot["hanger"]
    d, w, h = box[0] / 2, box[1] / 2, box[2]
    body_top = h - 0.1
    angles = [2 * math.pi * k / 20 for k in range(20)]
    rings = []
    for i in range(14):
        u = i / 13  # 0 hem .. 1 shoulder line
        z = 0.004 + (body_top - 0.004) * u
        half = w * 0.8 * curve(((0.0, 0.97), (0.45, 0.82), (0.85, 0.9), (1.0, 0.7)), u)
        depth = d / 0.9 * curve(((0.0, 0.8), (0.5, 0.9), (1.0, 0.6)), u)  # the curve peaks at 0.9
        rings.append([(depth * math.cos(a), half * math.sin(a), z) for a in angles])
    loft(bm, rings, cloth)
    for s in (1.0, -1.0):  # a sleeve down each side, the shoulder to the cuff
        sweep(bm, [(0.0, s * w * 0.55, body_top - 0.03), (0.0, s * w * 0.78, body_top - 0.12),
                   (0.0, s * w * 0.84, 0.32), (0.0, s * w * 0.82, 0.22)], 0.0, cloth, segments=12,
              radii=[0.036, 0.038, 0.036, 0.03])
    sweep(bm, [(0.0, -w * 0.5, body_top + 0.01), (0.0, 0.0, body_top + 0.07), (0.0, w * 0.5, body_top + 0.01)], 0.0,
          hanger, segments=8, radii=[0.0025, 0.0025, 0.0025])
    sweep(bm, [(0.0, 0.0, h - 0.03), (0.0, 0.02, h - 0.008), (0.0, 0.0, h - 0.0035), (0.0, -0.02, h - 0.012)], 0.0,
          hanger, segments=8, radii=[0.0025, 0.0025, 0.0025, 0.0025])


def drying_rack(bm, box, slot) -> None:
    """A folding clothes-drying rack: two end frames of splayed legs with a top bar and a lower bar each, five rails
    along its length and eight pieces of washing hung over them."""
    frame, cloth = slot["frame"], slot["cloth"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    r = 0.007
    top, low = h - r, 0.62
    tops, lows = (-0.18, 0.0, 0.18), (-0.24, 0.24)  # the rails' x, at the top and low down
    for s in (1.0, -1.0):  # an end frame at each end of the rails (y)
        y = s * (hy - r)
        for x in (-hx + r, hx - r):
            rod(bm, (x, y, 0.0), (0.66 * x, y, top), r, frame, segments=8)
        rod(bm, (-0.66 * hx, y, top), (0.66 * hx, y, top), r, frame, segments=8)
        rod(bm, (-0.84 * hx, y, 0.3), (0.84 * hx, y, 0.3), r, frame, segments=8)
    for x, z in [(x, top) for x in tops] + [(x, low) for x in lows]:
        rod(bm, (x, -hy + r, z), (x, hy - r, z), r, frame, segments=8)
    pieces = [(x, top, 0.24 + 0.02 * (i % 2), 0.32 + 0.05 * (i % 3), -hy + 0.14 + 0.19 * i) for x in tops
              for i in range(4)]
    pieces += [(x, low, 0.22, 0.28, -hy + 0.2 + 0.25 * i) for x in lows for i in range(3)]
    for x, z, width, drop, y in pieces:
        solid(bm, x - 0.011, x + 0.011, y - width / 2, y + width / 2, z + r - drop, z + r + 0.004, cloth, bevel=0.004,
              segments=1)


BUILDERS = {"hanging_coat": hanging_coat, "drying_rack": drying_rack}
