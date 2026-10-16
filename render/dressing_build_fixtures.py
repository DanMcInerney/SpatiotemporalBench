# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's wall fixtures: the panel radiator, the wall television, the
kitchen's wall cabinets, range hood and backsplash, and the bathroom's cabinet and toilet-roll holder.

Runs inside Blender's own Python and is never imported by `stbench`. Each builder is `fn(d, ctx, text) -> [object]` and
draws its piece in the Dressing's frame (`dressing_mesh`): x out of the wall (the wall at -depth/2), y across, z up.
Cabinet doors stand a little back from the box's front so that their bar handles fit inside it. The cabinet fronts are
white laminate with a 3 mm gap round each door; the radiator is ribbed with a slotted top over a dark core; the
television is a black frame round a dark glossy screen, thin at its rim and thicker behind.
"""

from __future__ import annotations

import bpy
from dressing_mesh import Build

LAMINATE, STEEL, CHROME = "white_laminate", "brushed_steel", "chrome"
WHITE_METAL, BLACK, GREY = "painted_metal_white", "black_plastic", "painted_metal_grey"
GAP = 0.003  # between a cabinet's doors and round them


def _doors(b: Build, x_front: float, thick: float, y0: float, y1: float, z0: float, z1: float, n: int) -> list[float]:
    """`n` flat doors with a gap round each between y0..y1 and z0..z1, their fronts at `x_front`; each one's centre."""
    width = (y1 - y0 - GAP * (n + 1)) / n
    centres = []
    for i in range(n):
        a = y0 + GAP + i * (width + GAP)
        b.box(x_front - thick, x_front, a, a + width, z0 + GAP, z1 - GAP, LAMINATE, bevel=0.002)
        centres.append(a + width / 2)
    return centres


def _bar_handle(b: Build, x_front: float, y: float, z0: float, length: float) -> None:
    """A vertical steel bar on two stand-offs, its far face at `x_front` + 0.012 (the stand-offs are in the door)."""
    r, off = 0.0065, 0.012
    b.rod((x_front + off - r, y, z0), (x_front + off - r, y, z0 + length), r, STEEL, segments=10)
    for z in (z0 + 0.012, z0 + length - 0.012):
        b.box(x_front - 0.001, x_front + off - r, y - 0.004, y + 0.004, z - 0.004, z + 0.004, STEEL)


def upper_cabinet(d, ctx, text) -> list[bpy.types.Object]:
    """A wall cabinet: a carcass, one door (two when over 0.62 m wide) a few millimetres back from the front, bar
    handles low on the doors' free edges."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0, front = -t / 2, t / 2 - 0.012
    b.box(x0, front - 0.016, -w / 2, w / 2, 0.0, h, LAMINATE, bevel=0.002)
    centres = _doors(b, front, 0.018, -w / 2, w / 2, 0.0, h, 2 if w > 0.62 else 1)
    for i, _ in enumerate(centres):  # at the free edge of a single door, beside the gap of a pair
        _bar_handle(b, front, w / 2 - 0.035 if len(centres) == 1 else (-0.035, 0.035)[i], 0.03, 0.14)
    return [b.finish()]


def range_hood(d, ctx, text) -> list[bpy.types.Object]:
    """A chimney hood: a steel canopy with a short vertical skirt and a grease filter and two lamps under it, three
    dials on the skirt, and a flue up to the ceiling against the wall."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0, x1 = -t / 2, t / 2
    skirt, rise = 0.05, 0.16
    b.box(x0, x1 - 0.003, -w / 2, w / 2, 0.002, skirt, STEEL, bevel=0.0015)
    b.frustum((x0, x1 - 0.003, -w / 2, w / 2), (x0, x0 + 0.26, -0.17, 0.17), skirt, skirt + rise, STEEL, bevel=0.003)
    b.box(x0 + 0.01, x0 + 0.23, -0.15, 0.15, skirt + rise - 0.01, h, STEEL, bevel=0.003)
    b.box(x0 + 0.05, x1 - 0.035, -w / 2 + 0.05, w / 2 - 0.05, 0.0, 0.007, GREY, bevel=0.001)  # the filter
    for y in (-0.16, 0.16):
        b.revolve([(0.0, 0.0), (0.015, 0.0), (0.015, 0.002), (0.0, 0.002)], "led_emitter_warm",
                  at=(x1 - 0.019, y, 0.0), segments=20)
    for y in (-0.06, 0.0, 0.06):
        b.rod((x1 - 0.003, y, 0.027), (x1, y, 0.027), 0.008, BLACK, segments=14)
    return [b.finish()]


def backsplash(d, ctx, text) -> list[bpy.types.Object]:
    """Subway tile on the wall behind the worktop: a flat bevelled slab in the library's tile (its 15 x 7.5 cm bond,
    glaze and grout are read off y and z)."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    b.box(-t / 2, t / 2, -w / 2, w / 2, 0.0, h, "ceramic_tile", bevel=0.0012)
    return [b.finish()]


def bathroom_cabinet(d, ctx, text) -> list[bpy.types.Object]:
    """A shallow wall cabinet with two plain laminate doors (no mirror: the ID pass could not show one) and a slim
    steel pull on each."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0, front = -t / 2, t / 2 - 0.007
    b.box(x0, front - 0.014, -w / 2, w / 2, 0.0, h, LAMINATE, bevel=0.002)
    for y in _doors(b, front, 0.016, -w / 2, w / 2, 0.0, h, 2):
        b.box(front - 0.001, front + 0.007, y + (0.025 if y < 0 else -0.025) - 0.006,
              y + (0.025 if y < 0 else -0.025) + 0.006, h - 0.22, h - 0.04, STEEL, bevel=0.0015)
    return [b.finish()]


def toilet_roll_holder(d, ctx, text) -> list[bpy.types.Object]:
    """A chrome two-arm holder on two wall plates with a roll between them: tissue wound on a cardboard core, the core
    showing at each end, a strip of the sheet hanging down the front. The roll lies along y, not upright."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0, zc, rr, half = -t / 2, h / 2, 0.047, 0.05
    b.revolve([(0.0185, -half), (rr, -half), (rr, half), (0.0185, half)], "paper_tissue", at=(0.0, 0.0, zc), axis="y")
    b.revolve([(0.0165, -half - 0.001), (0.0195, -half - 0.001), (0.0195, half + 0.001), (0.0165, half + 0.001)],
              "cardboard_core", at=(0.0, 0.0, zc), axis="y", segments=24, loop=True)
    b.box(rr, rr + 0.0015, -rr, rr, 0.001, zc, "paper_tissue")  # the loose end, down the front of the roll
    for s in (-1.0, 1.0):
        y = s * (half + 0.012)
        b.box(x0, x0 + 0.004, y - 0.011, y + 0.011, zc - 0.03, zc + 0.03, CHROME, bevel=0.0012)
        b.rod((x0 + 0.004, y, zc), (0.0, y, zc), 0.0045, CHROME, segments=10)
    b.rod((0.0, -half - 0.012, zc), (0.0, half + 0.012, zc), 0.0045, CHROME, segments=10)
    return [b.finish()]


def radiator(d, ctx, text) -> list[bpy.types.Object]:
    """A panel radiator on the wall, 12 cm off the floor: a front panel pressed into vertical ribs over a dark core of
    convector fins, a slotted top grille, rounded end caps and a valve stub under one end."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0, x1 = -t / 2, t / 2
    z0, z1 = 0.03, h
    b.box(x0 + 0.006, x0 + 0.014, -w / 2 + 0.02, w / 2 - 0.02, z0, z1 - 0.01, WHITE_METAL, bevel=0.004)  # back panel
    b.box(x0 + 0.012, x1 - 0.026, -w / 2 + 0.02, w / 2 - 0.02, z0 + 0.02, z1 - 0.02, BLACK)  # the dark core
    n = round((w - 0.04) / 0.064)
    pitch = (w - 0.04) / n
    for i in range(n):  # the ribs of the front panel
        y = -w / 2 + 0.02 + pitch * (i + 0.5)
        b.box(x1 - 0.028, x1, y - pitch * 0.4, y + pitch * 0.4, z0, z1 - 0.012, WHITE_METAL, bevel=0.004)
    for i in range(int((w - 0.04) / 0.014)):  # the grille bars over the top, with the core dark between them
        y = -w / 2 + 0.027 + i * 0.014
        b.box(x0 + 0.006, x1 - 0.004, y - 0.004, y + 0.004, z1 - 0.012, z1, WHITE_METAL, bevel=0.0012)
    for ya, yb in ((-w / 2, -w / 2 + 0.02), (w / 2 - 0.02, w / 2)):  # the end caps
        b.box(x0 + 0.004, x1, ya, yb, z0, z1, WHITE_METAL, bevel=0.009, segments=3)
    b.rod((0.0, w / 2 - 0.06, 0.0), (0.0, w / 2 - 0.06, z0 + 0.004), 0.011, CHROME, segments=14)
    b.rod((0.0, w / 2 - 0.06, 0.0), (0.0, w / 2 - 0.06, 0.012), 0.0145, CHROME, segments=6)
    return [b.finish()]


def wall_tv(d, ctx, text) -> list[bpy.types.Object]:
    """A flat television: a 1 cm black frame round a dark glossy screen, 1.2 cm thin at the rim, a thicker rear
    housing, and two spacers to the wall."""
    b = Build(ctx, d.cls)
    t, w, h = d.size
    x0 = -t / 2
    b.box(x0, x0 + 0.004, -w * 0.3, w * 0.3, h * 0.2, h * 0.6, BLACK, bevel=0.001)  # the mount
    b.box(x0 + 0.003, x0 + 0.024, -w * 0.39, w * 0.39, h * 0.12, h * 0.8, BLACK, bevel=0.01, segments=3)
    b.box(x0 + 0.02, x0 + 0.032, -w / 2, w / 2, 0.0, h, BLACK, bevel=0.002)
    b.box(x0 + 0.0318, x0 + 0.0325, -w / 2 + 0.009, w / 2 - 0.009, 0.009, h - 0.009, "dark_glass")
    b.box(x0 + 0.0318, x0 + 0.0335, -0.004, 0.004, 0.0035, 0.0055, "led_emitter")  # the standby light
    return [b.finish()]
