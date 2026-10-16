# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The small things of a bathroom and a living room that no question names, as procedural clutter: a toilet brush in its
holder, a bottle of shampoo and a bookshelf speaker.

Runs inside Blender's own Python and is never imported by `stbench`. Like `procedural_clutter.py`, a builder takes the
bmesh, its `DESIGN_BOX` (metres) and `slot` and adds geometry with its origin at the base centre; each reaches its box
to a millimetre (a test keeps it so). The shampoo is white, yellow, orange, pink or purple, and no tracked bottle is
any of those; the speaker is black or grey, and no tracked box is.
"""

from __future__ import annotations

from meshkit import lathe, rod, solid

DESIGN_BOX = {"toilet_brush": (0.11, 0.11, 0.38), "shampoo": (0.045, 0.065, 0.20), "speaker": (0.16, 0.16, 0.30)}
SLOTS = {"toilet_brush": ("holder", "brush"), "shampoo": ("body", "cap"), "speaker": ("cabinet", "cone", "trim")}


def toilet_brush(bm, box, slot) -> None:
    """A toilet brush: a round holder with a lip, a thin handle that stands in it and a bristle head on the end."""
    holder, brush = slot["holder"], slot["brush"]
    r, h = box[0] / 2, box[2]
    lathe(bm, [(0, 0), (0.8 * r, 0), (0.88 * r, 0.01), (r, 0.18), (r, 0.205), (0.9 * r, 0.205), (0.82 * r, 0.19),
               (0.7 * r, 0.02), (0, 0.02)], holder, segments=32)
    rod(bm, (0, 0, 0.02), (0, 0, h - 0.07), 0.006, brush, segments=10)
    lathe(bm, [(0.0, h - 0.07), (0.012, h - 0.07), (0.028, h - 0.06), (0.03, h - 0.035), (0.024, h - 0.012),
               (0.012, h - 0.003), (0, h)], brush, segments=20)


def shampoo(bm, box, slot) -> None:
    """A flat shampoo bottle: a rounded body that is wider than it is deep, a shoulder, a neck and a flip cap."""
    body, cap = slot["body"], slot["cap"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx, -hy, hy, 0.0, h - 0.045, body, bevel=0.012, segments=3)
    solid(bm, -hx * 0.55, hx * 0.55, -hy * 0.6, hy * 0.6, h - 0.047, h - 0.03, body, bevel=0.008, segments=2)
    solid(bm, -hx * 0.62, hx * 0.62, -hy * 0.62, hy * 0.62, h - 0.032, h, cap, bevel=0.006, segments=2)
    solid(bm, hx - 0.0012, hx, -hy * 0.7, hy * 0.7, 0.045, 0.125, cap)  # the label


def speaker(bm, box, slot) -> None:
    """A bookshelf speaker: a rounded cabinet with a woofer and a tweeter on its front (+x) and a bass port below."""
    cabinet, cone, trim = slot["cabinet"], slot["cone"], slot["trim"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx - 0.003, -hy, hy, 0.0, h, cabinet, bevel=0.012, segments=3)
    for z, r in ((0.085, 0.055), (0.225, 0.022)):  # a driver in a ring of trim
        rod(bm, (hx - 0.006, 0.0, z), (hx - 0.0035, 0.0, z), r + 0.006, trim, segments=28)
        rod(bm, (hx - 0.0035, 0.0, z), (hx, 0.0, z), r, cone, segments=28)
    rod(bm, (hx - 0.004, 0.0, 0.17), (hx, 0.0, 0.17), 0.012, trim, segments=16)


BUILDERS = {"toilet_brush": toilet_brush, "shampoo": shampoo, "speaker": speaker}
