# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Procedural stand-ins for the household's Poly Haven models: a picture frame,
a wall clock, the ceiling lamps and the ceiling fan, drawn in the Dressing's box when the library does not have the
model (`dressing.draw`), so a scene builds with no download.

Runs inside Blender's own Python and is never imported by `stbench`. Each builder is `fn(d, ctx, text) -> [object]` and
draws its piece in the Dressing's frame (`dressing_mesh`): x out of the wall (or down, for a fan), y across, z up. A
frame is a moulding round a white mat and an abstract in ochre, sage and slate (no glass, no mirror: the ID pass could
not show one), thin and dark for a `hanging_` model and wide and gilt for a `fancy_` one. The clock is a black rim
round a
cream face with twelve hour marks and three hands at ten past ten: nothing a plate is (a plate lies flat, is 22 cm and
has no hands). The flush lamp is a frosted dome under a steel ring that lights for the camera only (`lighting.py` owns
the scene's light); the pendant a spun-metal dome on a cord; the fan five tilted blades under a motor and a light.
"""

from __future__ import annotations

import bpy
from dressing_mesh import Build, placed_at, turn

CHROME, WHITE_METAL, BLACK, LAMP = "chrome", "painted_metal_white", "black_plastic", "led_emitter_warm"


def picture_frame(d, ctx, text) -> list[bpy.types.Object]:
    """A mat and canvas in a four-sided moulding: a `fancy_` model gets a wide gilt moulding and the art in blocks, a
    `hanging_` one a thin black, white or walnut moulding (by the model's number) and the art in bands."""
    t, w, h = d.size
    fancy = "fancy" in d.recipe
    frame = "frame_gilt" if fancy else ("frame_black", "frame_white", "frame_walnut")[sum(map(ord, d.recipe)) % 3]
    rim = 0.04 if fancy else 0.022
    margin = 0.05 if min(w, h) > 0.4 else 0.035
    x0, x1 = -t / 2, t / 2
    b = Build(ctx, d.cls)
    b.box(x0, x0 + 0.3 * t, -w / 2, w / 2, 0.0, h, frame)  # the backing
    for ya, yb, za, zb in ((-w / 2, w / 2, 0.0, rim), (-w / 2, w / 2, h - rim, h), (-w / 2, -w / 2 + rim, rim, h - rim),
                           (w / 2 - rim, w / 2, rim, h - rim)):
        b.box(x0 + 0.1 * t, x1, ya, yb, za, zb, frame, bevel=0.0025 if fancy else 0.0015)
    b.box(x0 + 0.3 * t, x0 + 0.55 * t, -w / 2 + rim, w / 2 - rim, rim, h - rim, "art_blocks" if fancy else "art_bands")
    for ya, yb, za, zb in ((-w / 2 + rim, w / 2 - rim, rim, rim + margin),
                           (-w / 2 + rim, w / 2 - rim, h - rim - margin, h - rim),
                           (-w / 2 + rim, -w / 2 + rim + margin, rim + margin, h - rim - margin),
                           (w / 2 - rim - margin, w / 2 - rim, rim + margin, h - rim - margin)):
        b.box(x0 + 0.55 * t, x0 + 0.75 * t, ya, yb, za, zb, "mat_board")
    return [b.finish()]


def wall_clock(d, ctx, text) -> list[bpy.types.Object]:
    """A 32 cm round clock with a 1.5 cm black rim set back from a cream face: twelve marks, long at the quarters,
    and an hour, a minute and a second hand at 10:08."""
    t, w, h = d.size
    x0, x1 = -t / 2, t / 2
    r, zc = w / 2, h / 2
    rim = r - 0.014
    b = Build(ctx, d.cls)
    b.revolve([(rim, x0 + 0.004), (r, x0 + 0.004), (r, x1 - 0.004), (r - 0.0025, x1), (rim + 0.0025, x1),
               (rim, x1 - 0.008)], BLACK, at=(0.0, 0.0, zc), axis="x", segments=48, loop=True)
    face = x0 + 0.5 * t
    b.revolve([(0.0, x0 + 0.004), (rim + 0.001, x0 + 0.004), (rim + 0.001, face), (0.0, face)], "clock_face",
              at=(0.0, 0.0, zc), axis="x", segments=48)
    for k in range(12):
        long = k % 3 == 0
        length = 0.026 if long else 0.014
        half = (0.005 if long else 0.003) / 2

        def mark(length=length, half=half):
            b.box(face, face + 0.0015, -half, half, zc + rim - 0.014 - length, zc + rim - 0.014, BLACK)

        _around(b, k * 30.0, zc, mark)

    def hand(length: float, width: float, tail: float, angle: float, lift: float, material: str = BLACK) -> None:
        def draw():
            b.box(face + lift, face + lift + 0.0014, -width / 2, width / 2, zc - tail, zc + length, material)

        _around(b, angle, zc, draw)

    hand(0.075, 0.008, 0.015, (10 + 8 / 60) * 30.0, 0.0015)
    hand(0.115, 0.0055, 0.02, 8 * 6.0, 0.003)
    hand(0.12, 0.0018, 0.03, 36.0, 0.0045)
    b.revolve([(0.0, face), (0.009, face), (0.009, face + 0.007), (0.0, face + 0.007)], BLACK, at=(0.0, 0.0, zc),
              axis="x", segments=16)
    return [b.finish()]


def _around(b: Build, clockwise_deg: float, zc: float, draw) -> None:
    """Draw with `draw()` a thing built pointing up from the clock's centre, then turn it clockwise as seen from the
    front (+x looking at the wall) about the centre."""
    placed_at(b, turn(-clockwise_deg, "x", (0.0, 0.0, zc)), draw)


def _lamp_flush(b: Build, d) -> None:
    h = d.size[2]
    b.revolve([(0.0, 0.0), (0.11, 0.0), (0.16, 0.004), (0.188, 0.016), (0.197, 0.04), (0.198, h - 0.012),
               (0.0, h - 0.012)], LAMP, segments=48)
    b.revolve([(0.186, h - 0.016), (0.2, h - 0.016), (0.2, h), (0.186, h)], WHITE_METAL, segments=48, loop=True)


def _lamp_pendant(b: Build, d) -> None:
    """A spun-metal dome with a warm bulb under it, on a cord from a ceiling rose."""
    t, h = d.size[0], d.size[2]
    r = t / 2
    outer = [(r, 0.0), (r - 0.0015, 0.02), (r * 0.95, 0.06), (r * 0.83, 0.105), (r * 0.65, 0.15), (r * 0.42, 0.185),
             (r * 0.14, 0.205)]
    inner = [(rr - 0.004, z) for rr, z in outer]
    b.revolve(outer + inner[::-1], WHITE_METAL, segments=40, loop=True)
    b.revolve([(0.0, 0.0), (0.03, 0.0), (0.04, 0.03), (0.036, 0.075), (0.02, 0.1), (0.0, 0.105)], LAMP, at=(0, 0, 0.06),
              segments=18)
    b.rod((0.0, 0.0, 0.2), (0.0, 0.0, h - 0.02), 0.004, BLACK, segments=8)
    b.revolve([(0.0, h - 0.022), (0.06, h - 0.022), (0.06, h - 0.012), (0.045, h), (0.0, h)], WHITE_METAL, segments=24)


def ceiling_lamp(d, ctx, text) -> list[bpy.types.Object]:
    b = Build(ctx, d.cls)
    (_lamp_pendant if d.size[2] > 0.5 else _lamp_flush)(b, d)
    return [b.finish()]


def ceiling_fan(d, ctx, text) -> list[bpy.types.Object]:
    """A ceiling fan with five wooden blades pitched 10 degrees, on arms from a motor housing, a downrod, a canopy at
    the ceiling and a frosted light under the motor."""
    t, h = d.size[0], d.size[2]
    reach = t / 2 - 0.005
    b = Build(ctx, d.cls)
    zb = 0.155  # the blades' height above the lowest point
    b.revolve([(0.0, 0.0), (0.05, 0.0), (0.08, 0.02), (0.086, 0.05), (0.07, 0.085), (0.0, 0.09)], LAMP, segments=32)
    b.revolve([(0.0, 0.09), (0.09, 0.09), (0.105, 0.115), (0.105, 0.19), (0.09, 0.205), (0.04, 0.215), (0.0, 0.215)],
              CHROME, segments=32)
    b.rod((0.0, 0.0, 0.2), (0.0, 0.0, h - 0.04), 0.013, CHROME, segments=12)
    b.revolve([(0.0, h - 0.06), (0.04, h - 0.06), (0.075, h - 0.03), (0.075, h), (0.0, h)], CHROME, segments=32)
    for k in range(5):
        az = 72.0 * k
        length = reach - 0.2

        def blade(length=length):
            b.box(0.2, 0.2 + length, -0.085, 0.085, zb - 0.006, zb + 0.006, "fan_blade", bevel=0.004, segments=2)
            b.box(0.1, 0.205, -0.02, 0.02, zb - 0.02, zb - 0.01, CHROME, bevel=0.002)

        placed_at(b, turn(az, "z") @ turn(10.0, "x", (0.0, 0.0, zb)), blade)
    return [b.finish()]
