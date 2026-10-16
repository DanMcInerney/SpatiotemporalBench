# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's trim and small wall and ceiling fittings: baseboards, door casings,
window casings and sills, socket and switch plates, the thermostat and the smoke detector.

Runs inside Blender's own Python and is never imported by `stbench`. Each builder is `fn(d, ctx, text) -> [object]`
(`dressing.py`) and draws its piece in the Dressing's frame (`dressing_mesh`): x out of the wall (the wall at -depth/2),
y across, z up. Sizes are the real thing's: an 8 cm baseboard with a coved top, a 7 cm casing with a raised inner band,
a 7 x 11.5 cm plate with a duplex socket or a rocker, a 9 cm thermostat, an 11 cm detector. Everything stays inside the
box the layout gives it.
"""

from __future__ import annotations

import bpy
from dressing_mesh import Build

TRIM, PLASTIC, BLACK, DARK = "painted_wood_trim", "white_plastic", "black_plastic", "rubber_black"


def baseboard(d, ctx, text) -> list[bpy.types.Object]:
    """A coved 8 cm skirting: full thickness for the lower three quarters, then a quarter-round that thins to a lip."""
    t, h = d.size[0], d.size[2]
    x0, x1 = -t / 2, t / 2
    b = Build(ctx, d.cls)
    b.prism([(x0, 0.0), (x1, 0.0), (x1, 0.72 * h), (x1 - 0.29 * t, 0.78 * h), (x1 - 0.54 * t, 0.885 * h),
             (x0 + 0.38 * t, 0.96 * h), (x0 + 0.38 * t, h), (x0, h)], -d.size[1] / 2, d.size[1] / 2, TRIM)
    return [b.finish()]


def _casing(b: Build, d, material: str) -> None:
    """A 7 cm board with a raised inner band (backband and casing proper), along z for a jamb, along y for a head."""
    t, w, h = d.size
    x0, x1 = -t / 2, t / 2
    low, step = x0 + 0.62 * t, 0.15 * min(w, h)
    if h >= w:
        b.box(x0, low, -w / 2, w / 2, 0.0, h, material, bevel=0.0015)
        b.box(x0 + 0.5 * t, x1, -w / 2 + step, w / 2 - step, 0.0, h, material, bevel=0.002)
    else:
        b.box(x0, low, -w / 2, w / 2, 0.0, h, material, bevel=0.0015)
        b.box(x0 + 0.5 * t, x1, -w / 2, w / 2, step, h - step, material, bevel=0.002)


def door_casing(d, ctx, text) -> list[bpy.types.Object]:
    b = Build(ctx, d.cls)
    _casing(b, d, TRIM)
    return [b.finish()]


def window_sill(d, ctx, text) -> list[bpy.types.Object]:
    """Casing at the sides and head of a window, and under it a sill board 3 cm thick with a rounded front edge."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    if h <= 0.035:
        b.box(-t / 2, t / 2, -w / 2, w / 2, 0.0, h, TRIM, bevel=0.0055, segments=3)
    else:
        _casing(b, d, TRIM)
    return [b.finish()]


def _plate(b: Build, d, depth: float = 0.006) -> float:
    """The wall plate, bevelled, in white plastic; the x of its front face."""
    t, w, h = d.size
    b.box(-t / 2, -t / 2 + depth, -w / 2, w / 2, 0.0, h, PLASTIC, bevel=0.0012)
    return -t / 2 + depth


def outlet(d, ctx, text) -> list[bpy.types.Object]:
    """A duplex socket: the plate, a raised oval-cornered face, and in each socket the two blade slots and the round
    earth hole, in dark rubber."""
    b = Build(ctx, d.cls)
    h = d.size[2]
    xp = _plate(b, d)
    xf = xp + 0.0055
    b.box(xp - 0.001, xf, -0.0185, 0.0185, h / 2 - 0.0435, h / 2 + 0.0435, PLASTIC, bevel=0.0035, segments=3)
    for zc in (h / 2 - 0.0215, h / 2 + 0.0215):
        for y in (-0.0058, 0.0058):
            b.box(xf - 0.0001, xf + 0.0004, y - 0.0012, y + 0.0012, zc + 0.0004, zc + 0.0089, DARK)
        b.box(xf - 0.0001, xf + 0.0004, -0.0021, 0.0021, zc - 0.0125, zc - 0.0083, DARK)
    for z in (h * 0.14, h * 0.86):  # the two plate screws
        b.box(xp - 0.0001, xp + 0.0008, -0.0025, 0.0025, z - 0.0025, z + 0.0025, DARK)
    return [b.finish()]


def light_switch(d, ctx, text) -> list[bpy.types.Object]:
    """A plate with one rocker, or two on the 11.5 cm plate: each a wedge 3.2 x 6.6 cm, flush at the top and 8 mm
    proud at the foot."""
    b = Build(ctx, d.cls)
    w, h = d.size[1], d.size[2]
    xp = _plate(b, d)
    for y in ((0.0,) if w < 0.1 else (-0.0285, 0.0285)):
        b.prism([(xp - 0.001, h / 2 - 0.033), (xp + 0.0085, h / 2 - 0.033), (xp + 0.0015, h / 2 + 0.033),
                 (xp - 0.001, h / 2 + 0.033)], y - 0.016, y + 0.016, PLASTIC)
    return [b.finish()]


def thermostat(d, ctx, text) -> list[bpy.types.Object]:
    """A 9 cm square wall unit: a rounded body, a raised front, a dark display and three buttons."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0 = -t / 2
    b.box(x0, x0 + 0.022, -w / 2, w / 2, 0.0, h, PLASTIC, bevel=0.008, segments=3)
    b.box(x0 + 0.018, x0 + 0.026, -w / 2 + 0.009, w / 2 - 0.009, 0.009, h - 0.009, PLASTIC, bevel=0.005, segments=3)
    b.box(x0 + 0.0255, x0 + 0.0268, -0.026, 0.026, 0.047, 0.073, "dark_glass")
    for y in (-0.018, 0.0, 0.018):
        b.box(x0 + 0.0255, x0 + 0.0275, y - 0.006, y + 0.006, 0.019, 0.029, BLACK, bevel=0.0015)
    return [b.finish()]


def smoke_detector(d, ctx, text) -> list[bpy.types.Object]:
    """A flush ceiling disc: a shallow dome with a groove ring of vents and a round test button, top on the ceiling."""
    b = Build(ctx, d.cls)
    h = d.size[2]
    b.revolve([(0.0, 0.0), (0.0125, 0.0), (0.0135, 0.0015), (0.0175, 0.0015), (0.0185, 0.0030), (0.027, 0.0050),
               (0.0285, 0.0035), (0.0315, 0.0035), (0.0325, 0.0068), (0.040, 0.0140), (0.048, 0.0240),
               (0.0535, 0.0340), (0.0550, h), (0.0, h)], PLASTIC)
    return [b.finish()]
