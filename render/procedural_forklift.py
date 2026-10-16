# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's counterbalance forklift as procedural geometry.

Runs inside Blender's own Python and is never imported by `stbench`. The builder is called as in `procedural_goods.py`
(bmesh, `DESIGN_BOX`, slot dict) and builds to the box exactly: the fork tips reach +x, the counterweight -x, the drive
tyres' outer faces +-y, the tyres and the fork blades the floor and the overhead guard's roof the box's height.
The paint (the catalog colour, slot `body`) is the rounded counterweight and hood, the chassis and cowl, the fenders,
the overhead guard and the drive wheels' rims. The mast, its chains and cylinders, the carriage and the forks are the
`mast` slot (dark steel), the tyres the `tyres` slot and the seat and steering wheel the `cab` slot.
"""

from __future__ import annotations

import math

from mathutils import Matrix, Vector
from meshkit import lathe, placed, rod, solid
from partskit import Outline, along_x, loft, sweep

DRIVE_X, DRIVE_R, DRIVE_W = 0.18, 0.33, 0.2
STEER_X, STEER_R, STEER_W = -0.92, 0.25, 0.16
GUARD_REAR, GUARD_FRONT = -0.52, 0.36  # the x of the overhead guard's posts


def tyre(bm, cx: float, cy: float, r: float, width: float, tyres: int, rim: int) -> None:
    """A solid tyre with two tread grooves and a dished rim with a ring of bolts, axle along y, on the floor."""
    w, ri = width / 2, 0.64 * r
    outline = [(ri, -w), (0.9 * r, -w), (0.97 * r, -0.86 * w), (r, -0.7 * w), (r, -0.42 * w), (0.96 * r, -0.36 * w),
               (0.96 * r, -0.24 * w), (r, -0.18 * w), (r, 0.18 * w), (0.96 * r, 0.24 * w), (0.96 * r, 0.36 * w),
               (r, 0.42 * w), (r, 0.7 * w), (0.97 * r, 0.86 * w), (0.9 * r, w), (ri, w)]
    disc = [(0.0, -0.88 * w), (0.3 * r, -0.88 * w), (0.4 * r, -0.97 * w), (ri + 0.004, -0.97 * w),
            (ri + 0.004, 0.97 * w), (0.4 * r, 0.97 * w), (0.3 * r, 0.88 * w), (0.0, 0.88 * w)]

    def build() -> None:
        lathe(bm, outline, tyres, segments=32, loop=True)
        lathe(bm, disc, rim, segments=32)
        for side in (-1.0, 1.0):
            for k in range(6):
                a = 2 * math.pi * k / 6
                rod(bm, (0.2 * r * math.cos(a), 0.2 * r * math.sin(a), side * 0.88 * w),
                    (0.2 * r * math.cos(a), 0.2 * r * math.sin(a), side * 1.0 * w * 0.97), 0.012, tyres, segments=8)

    placed(bm, Matrix.Translation((cx, cy, r)) @ Matrix.Rotation(-math.pi / 2, 4, "X"), build)


def body_shell(bm, hx: float, body: int) -> None:
    """The counterweight and the hood as one rounded loft from the rear face forward, wider and taller at the back."""
    x = (-hx, -hx + 0.03, -hx + 0.12, -1.0, -0.5, -0.3, -0.12)
    hw = (0.30, 0.40, 0.46, 0.47, 0.47, 0.46, 0.40)
    zb = (0.34, 0.30, 0.27, 0.27, 0.27, 0.27, 0.27)
    zt = (0.82, 0.93, 1.0, 1.0, 1.0, 0.98, 0.92)
    rc = (0.10, 0.12, 0.14, 0.16, 0.16, 0.14, 0.10)

    def profile(s: float) -> tuple[float, float, float, float, float]:
        i = min(max(sum(1 for v in x if v <= s) - 1, 0), len(x) - 2)
        t = (s - x[i]) / (x[i + 1] - x[i])
        here, there = (hw[i], zb[i], zt[i], rc[i]), (hw[i + 1], zb[i + 1], zt[i + 1], rc[i + 1])
        mix = [a + (b - a) * t for a, b in zip(here, there, strict=True)]
        return 0.0, (mix[1] + mix[2]) / 2, mix[0], (mix[2] - mix[1]) / 2, mix[3]

    stations = sorted({round(v, 4) for v in x} | {round(-hx + 0.05 + 0.05 * i, 4) for i in range(1, 6)})
    loft(bm, along_x(Outline(10, 8, 5), stations, profile), body, cap_first=True, cap_last=True)


def overhead_guard(bm, hy: float, h: float, body: int) -> None:
    """Four posts, a frame and parallel bars across the top (the guard's roof) so that the cab can be seen through."""
    top = h
    for x in (GUARD_REAR, GUARD_FRONT):
        for s in (-1.0, 1.0):
            z0 = 0.99 if x == GUARD_REAR else 0.72
            solid(bm, x - 0.03, x + 0.03, s * 0.45 - 0.03, s * 0.45 + 0.03, z0, top - 0.02, body, bevel=0.008,
                  segments=1)
    x0, x1 = GUARD_REAR - 0.1, GUARD_FRONT + 0.14
    for s in (-1.0, 1.0):  # the frame's long sides
        solid(bm, x0, x1, s * 0.45 - 0.03, s * 0.45 + 0.03, top - 0.04, top, body, bevel=0.006, segments=1)
    for x in (x0 + 0.02, x1 - 0.02, (x0 + x1) / 2):  # and its cross bars
        solid(bm, x - 0.02, x + 0.02, -0.45, 0.45, top - 0.035, top, body, bevel=0.005, segments=1)
    for i in range(1, 7):  # the bars of the roof, along x
        y = -0.45 + 0.9 * i / 7
        solid(bm, x0, x1, y - 0.021, y + 0.021, top - 0.034, top - 0.004, body, bevel=0.004, segments=1)


def mast(bm, mast_slot: int) -> None:
    """Outer and inner rails with cross members, the lift cylinders and tilt cylinders, and two chains over the top."""
    for s in (-1.0, 1.0):
        solid(bm, 0.48, 0.60, s * 0.2 - 0.05, s * 0.2 + 0.05, 0.06, 2.0, mast_slot, bevel=0.01,
              segments=1)  # the outer rail
        solid(bm, 0.60, 0.67, s * 0.2 - 0.04, s * 0.2 + 0.04, 0.08, 1.92, mast_slot, bevel=0.008, segments=1)  # inner
        rod(bm, (0.54, s * 0.07, 0.15), (0.54, s * 0.07, 1.78), 0.03, mast_slot, segments=12)  # lift cylinder
        rod(bm, (0.54, s * 0.07, 1.78), (0.54, s * 0.07, 1.98), 0.014, mast_slot, segments=8)  # its rod
        rod(bm, (0.40, s * 0.30, 0.74), (0.52, s * 0.27, 1.2), 0.025, mast_slot, segments=8)  # tilt cylinder
        for k in range(46):  # the chain: links alternate wide and narrow
            z = 0.95 + 0.021 * k
            w = 0.03 if k % 2 == 0 else 0.018
            solid(bm, 0.675, 0.69, s * 0.1 - w / 2, s * 0.1 + w / 2, z, z + 0.019, mast_slot)
    for z in (0.22, 1.0, 1.93):  # cross members
        solid(bm, 0.48, 0.67, -0.25, 0.25, z, z + 0.06, mast_slot, bevel=0.008, segments=1)
    rod(bm, (0.64, -0.12, 1.96), (0.64, 0.12, 1.96), 0.03, mast_slot, segments=12)  # the chain wheel at the top


def carriage(bm, hx: float, mast_slot: int) -> None:
    """The fork carriage on the inner rails, the load backrest and the two forks (an upright shank and a tapering blade,
    the tips at the box's front)."""
    x0, x1 = 0.69, 0.75
    solid(bm, x0, x1, -0.46, 0.46, 0.55, 0.62, mast_slot, bevel=0.008, segments=1)
    solid(bm, x0, x1, -0.46, 0.46, 0.10, 0.17, mast_slot, bevel=0.008, segments=1)
    for s in (-1.0, 1.0):
        solid(bm, x0, x1, s * 0.44 - 0.02, s * 0.44 + 0.02, 0.10, 0.62, mast_slot)
    for i in range(5):  # the load backrest
        y = -0.42 + 0.21 * i
        solid(bm, x1, x1 + 0.03, y - 0.015, y + 0.015, 0.62, 1.28, mast_slot)
    solid(bm, x1, x1 + 0.03, -0.44, 0.44, 1.23, 1.28, mast_slot, bevel=0.005, segments=1)
    solid(bm, x1, x1 + 0.03, -0.44, 0.44, 0.78, 0.82, mast_slot, bevel=0.005, segments=1)
    for s in (-1.0, 1.0):  # a fork: shank and blade
        yc = s * 0.3
        solid(bm, x1, x1 + 0.045, yc - 0.05, yc + 0.05, 0.03, 0.62, mast_slot, bevel=0.005, segments=1)
        tip = hx
        stations = [x1 + 0.045, x1 + 0.3, tip - 0.3, tip - 0.2, tip - 0.1, tip - 0.03, tip]
        t_ = [0.045, 0.045, 0.045, 0.036, 0.026, 0.016, 0.01]
        w_ = [0.05, 0.05, 0.05, 0.048, 0.042, 0.034, 0.028]
        rings = [[(x, yc + u, z) for u, z in ((-w, 0.0), (w, 0.0), (w, t), (-w, t))]
                 for x, t, w in zip(stations, t_, w_, strict=True)]
        loft(bm, rings, mast_slot, cap_first=True, cap_last=True, smooth=False)


def cab(bm, cab_slot: int, body: int) -> None:
    """The seat, its back and arm rests, and the steering column and wheel."""
    solid(bm, -0.46, -0.12, -0.19, 0.19, 1.0, 1.09, cab_slot, bevel=0.04, segments=3)
    solid(bm, -0.52, -0.44, -0.19, 0.19, 1.08, 1.42, cab_slot, bevel=0.035, segments=3)
    for s in (-1.0, 1.0):
        solid(bm, -0.38, -0.12, s * 0.23 - 0.015, s * 0.23 + 0.015, 1.1, 1.28, cab_slot, bevel=0.01, segments=1)
    base, top = Vector((0.2, 0.0, 0.5)), Vector((0.06, 0.0, 1.1))
    sweep(bm, [tuple(base), tuple(top)], 0.022, cab_slot, segments=12)
    d = (top - base).normalized()
    v = Vector((d.z, 0.0, -d.x)).normalized()
    side = Vector((0, 1, 0))
    ring = [tuple(top + 0.15 * (math.cos(2 * math.pi * k / 24) * side + math.sin(2 * math.pi * k / 24) * v))
            for k in range(24)]
    sweep(bm, ring, 0.014, cab_slot, segments=8, closed=True)
    for s in (-1.0, 1.0):
        rod(bm, tuple(top), tuple(top + s * 0.15 * Vector((0, 1, 0))), 0.01, cab_slot, segments=8)
    rod(bm, tuple(top), tuple(top + 0.15 * v), 0.01, cab_slot, segments=8)


def forklift(bm, box, slot) -> None:
    """A counterbalance forklift: a rounded counterweight and hood, a chassis and cowl between the wheels, an overhead
    guard on four posts, a seat and steering wheel, a mast with chains and cylinders, a carriage with a load backrest
    and two forks, two big drive tyres at the front and two steering tyres at the back. The forks are at +x."""
    body, mast_slot, tyres, cab_slot = slot["body"], slot["mast"], slot["tyres"], slot["cab"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    body_shell(bm, hx, body)
    solid(bm, -0.2, 0.5, -0.34, 0.34, 0.26, 0.46, body, bevel=0.03, segments=3)  # the chassis between the drive wheels
    solid(bm, 0.28, 0.47, -0.33, 0.33, 0.46, 0.88, body, bevel=0.04, segments=3)  # the cowl in front of the driver
    for s in (-1.0, 1.0):  # fenders over the drive tyres
        y0, y1 = (0.34, hy) if s > 0 else (-hy, -0.34)
        solid(bm, -0.14, 0.5, y0, y1, 0.69, 0.74, body, bevel=0.012, segments=2)
        edge = y1 - 0.025 if s > 0 else y0  # a lip down the fender's outer edge
        solid(bm, -0.14, 0.5, edge, edge + 0.025, 0.6, 0.74, body, bevel=0.008, segments=1)
        solid(bm, -0.14, 0.5, y0 if s > 0 else y1 - 0.03, y0 + 0.03 if s > 0 else y1, 0.46, 0.74, body)
        tyre(bm, DRIVE_X, s * (hy - DRIVE_W / 2), DRIVE_R, DRIVE_W, tyres, body)
        tyre(bm, STEER_X, s * (hy - 0.11 - STEER_W / 2), STEER_R, STEER_W, tyres, body)
    overhead_guard(bm, hy, h, body)
    cab(bm, cab_slot, body)
    mast(bm, mast_slot)
    carriage(bm, hx, mast_slot)
    for s in (-1.0, 1.0):  # a plate on each side of the hood
        solid(bm, -0.95, -0.55, s * 0.472 - 0.004, s * 0.472 + 0.004, 0.55, 0.7, mast_slot)
