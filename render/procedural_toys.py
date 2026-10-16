# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Toys for the household's `toy` kind, beside the rubber duck: a sitting teddy bear and a stack of wooden blocks.

Runs inside Blender's own Python and is never imported by `stbench`. The toy kind has one size, 0.14 x 0.10 x 0.134 m
(`catalog.KINDS["toy"]`), so each builder fills that box: the bear sits with its feet towards +x, its ears at the top
of the box and its arms at its sides; the blocks are a gate of two bricks, a lintel and a roof. Like
`procedural_clutter.py`, a builder takes the bmesh, its `DESIGN_BOX` and `slot` (slot name to material index);
`procedural.py` registers them and a test keeps each within a hundredth of its box. The colour attribute is the
bear's fur or the blocks' paint; a plush's muzzle and pads are the same colour a shade paler (the manifest's
`tint.lighten`), its eyes and nose dark.
"""

from __future__ import annotations

import math

from mathutils import Matrix
from meshkit import lathe, placed, prism, solid

DESIGN_BOX = {"teddy_bear": (0.14, 0.10, 0.134), "toy_blocks": (0.14, 0.10, 0.134)}
SLOTS = {"teddy_bear": ("fur", "muzzle", "dark"), "toy_blocks": ("block", "accent")}


def ellipsoid(bm, centre: tuple, radii: tuple, slot: int, turn: Matrix | None = None, segments: int = 20) -> None:
    """A smooth ellipsoid with semi-axes `radii` (x, y, z) at `centre`, turned by `turn` first."""
    ring = [(math.sin(math.pi * k / 10), -math.cos(math.pi * k / 10)) for k in range(11)]
    place = Matrix.Translation(centre) @ (turn if turn is not None else Matrix.Identity(4)) @ Matrix.Diagonal(
        (*radii, 1.0))
    placed(bm, place, lambda: lathe(bm, ring, slot, segments=segments))


def teddy_bear(bm, box, slot) -> None:
    """A sitting plush bear facing +x: a pear-shaped body, a round head with two ears, a muzzle with a dark nose, two
    eyes, two arms at its sides, two legs stretched out in front with pale pads on their soles, and a stub of a
    tail."""
    fur, muzzle, dark = slot["fur"], slot["muzzle"], slot["dark"]
    x0 = -0.0175
    placed(bm, Matrix.Translation((x0, 0, 0)) @ Matrix.Diagonal((1.0, 0.88, 1.0, 1.0)),
           lambda: lathe(bm, [(0, 0.004), (0.034, 0.004), (0.0445, 0.018), (0.046, 0.038), (0.04, 0.062), (0.03, 0.08),
                              (0.018, 0.09), (0, 0.093)], fur, segments=28))
    head = (x0 + 0.027, 0.0, 0.098)
    ellipsoid(bm, head, (0.034, 0.034, 0.033), fur)
    for s in (1.0, -1.0):
        ellipsoid(bm, (head[0] - 0.004, s * 0.0255, 0.1215), (0.011, 0.0125, 0.0125), fur)  # an ear
        ellipsoid(bm, (head[0] - 0.0025, s * 0.0255, 0.1215), (0.0085, 0.0095, 0.0095), muzzle)  # and its inner side
        ellipsoid(bm, (head[0] + 0.029, s * 0.0155, head[2] + 0.0105), (0.0045, 0.0045, 0.0045), dark, segments=12)
        lean = Matrix.Rotation(math.radians(-38.0), 4, "Y") @ Matrix.Rotation(math.radians(s * 8.0), 4, "X")
        ellipsoid(bm, (x0 + 0.018, s * 0.0385, 0.058), (0.011, 0.011, 0.026), fur, lean)  # an arm, hanging forward
        ellipsoid(bm, (x0 + 0.057, s * 0.021, 0.0145), (0.0285, 0.0145, 0.0145), fur)  # a leg, stretched out
        ellipsoid(bm, (x0 + 0.0835, s * 0.021, 0.0145), (0.0035, 0.0105, 0.0105), muzzle, segments=16)  # its pad
    ellipsoid(bm, (head[0] + 0.027, 0.0, head[2] - 0.0085), (0.017, 0.0165, 0.0125), muzzle)  # the muzzle
    ellipsoid(bm, (head[0] + 0.0425, 0.0, head[2] - 0.0035), (0.0058, 0.0075, 0.0052), dark, segments=12)  # the nose
    ellipsoid(bm, (x0 - 0.042, 0.0, 0.02), (0.0115, 0.0115, 0.0115), fur)  # the tail


def toy_blocks(bm, box, slot) -> None:
    """A gate of wooden blocks: two bricks on the floor a hair apart, a lintel across them and a triangular roof on the
    lintel, all with their edges eased."""
    block, accent = slot["block"], slot["accent"]
    hx, hy, brick = box[0] / 2, box[1] / 2, 0.048
    for x0, x1 in ((-hx, -0.004), (0.004, hx)):
        solid(bm, x0, x1, -hy, hy, 0.0, brick, block, bevel=0.0022, segments=2)
    solid(bm, -hx, hx, -0.025, 0.025, brick, 2 * brick, accent, bevel=0.0022, segments=2)
    prism(bm, [(-0.033, 2 * brick), (0.033, 2 * brick), (0.0, box[2])], -0.025, 0.025, block)


BUILDERS = {"teddy_bear": teddy_bear, "toy_blocks": toy_blocks}
