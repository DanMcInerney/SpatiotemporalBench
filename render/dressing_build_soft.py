# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's window and shower dressing: curtains, roller blinds and the shower
curtain.

Runs inside Blender's own Python and is never imported by `stbench`. Each builder is `fn(d, ctx, text) -> [object]` and
draws its piece in the Dressing's frame (`dressing_mesh`): x out of the wall (the wall at -depth/2), y across, z up.
A curtain is one closed cloth solid 4 mm thick, hung from rings on a rod and folded in sine pleats whose depth grows
towards the hem; its colour is a muted linen, never a saturated textile (a towel look-alike: DECISIONS, "Realism
Phase 1, the dressing's validity guard"). A roller blind is its cassette alone, the blind drawn open.
"""

from __future__ import annotations

import math

import bpy
from dressing_mesh import Build

CHROME, STEEL, WHITE_METAL, BLACK = "chrome", "brushed_steel", "painted_metal_white", "black_plastic"
CLOTH = 0.004  # a curtain's thickness


def _hang(b: Build, d, material: str, pleat: float, depth: float, rod_r: float, ring_r: float) -> None:
    """A rod across the top of the box with finials and rings, and from the rings a cloth in folds of `pleat` metres
    reaching down to the base: sine pleats `depth` deep (a quarter more at the hem) with a slow sway."""
    t, w, h = d.size
    zr = h - 0.03
    b.rod((0.0, -w / 2 + 0.02, zr), (0.0, w / 2 - 0.02, zr), rod_r, CHROME, segments=12)
    for y in (-w / 2 + 0.016, w / 2 - 0.016):
        b.revolve([(0.0, -0.016), (0.014, -0.012), (0.02, 0.0), (0.014, 0.012), (0.0, 0.016)], CHROME,
                  at=(0.0, y, zr), axis="y", segments=14)
    n = max(1, round(w / pleat))
    for k in range(n):  # a ring where the cloth crosses the rod's plane
        b.ring((0.0, -w / 2 + w * (k + 0.5) / n, zr), ring_r, 0.0025, "y", CHROME, major=16, minor=6)
    top = zr - ring_r - 0.004

    def point(u: float, v: float) -> tuple[float, float, float]:
        a = depth * (1.0 + 0.25 * (1.0 - v))
        phase = 2 * math.pi * n * u + 0.35 * math.sin(2 * math.pi * 1.3 * v)
        return (a * math.sin(phase) + 0.003 * math.sin(2 * math.pi * u * 1.7 + 4 * v), -w / 2 + w * u,
                v * top)

    b.sheet(max(24, 12 * n), 14, point, CLOTH, material)


def curtain(d, ctx, text) -> list[bpy.types.Object]:
    """One panel of drapes: a rod with finials and rings, and linen in three or four folds across 35 cm, reaching down
    to the floor or the sill."""
    b = Build(ctx, d.cls)
    _hang(b, d, "curtain_fabric", 0.115, 0.021, 0.0125, 0.02)
    return [b.finish()]


def shower_curtain(d, ctx, text) -> list[bpy.types.Object]:
    """A shower curtain: a chrome rail across the top, rings every 14 cm and a cream vinyl sheet in shallow folds, its
    hem 5 cm above the tub's rim."""
    b = Build(ctx, d.cls)
    _hang(b, d, "shower_curtain_vinyl", 0.14, 0.016, 0.011, 0.022)
    return [b.finish()]


def roller_blind(d, ctx, text) -> list[bpy.types.Object]:
    """The cassette of a roller blind drawn open: a rounded white tube across the head of the window, dark end caps and
    a slot along its lower front where the blind would leave it."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0, x1 = -t / 2, t / 2
    b.box(x0, x1, -w / 2 + 0.004, w / 2 - 0.004, 0.0015, h, WHITE_METAL, bevel=0.028, segments=4)
    for y0, y1 in ((-w / 2, -w / 2 + 0.004), (w / 2 - 0.004, w / 2)):
        b.box(x0 + 0.004, x1 - 0.004, y0, y1, 0.004, h - 0.004, BLACK, bevel=0.0012)
    b.box(-0.007, 0.007, -w / 2 + 0.02, w / 2 - 0.02, 0.0, 0.002, BLACK)
    return [b.finish()]
