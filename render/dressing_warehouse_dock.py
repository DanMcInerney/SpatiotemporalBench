# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What stands round a dock door (`dressing_layout_warehouse`): foam seal pads and a head, rubber bumpers, the flush
leveler plate in front, a dock light and the door's painted number.

Runs inside Blender's own Python and is never imported by `stbench`. Each piece is drawn in its Dressing's frame (x out
of the wall, y along it, z up); none of them reaches into the door's own footprint, so the shutter stays whole.
"""

from __future__ import annotations

from dressing_warehouse_kit import BARS, GALV, INK, LENS_LOW, PAPER, PLASTIC, RUBBER, STEEL, TREAD, Mesh


def dock_seal_pad(d, ctx, text):
    """A side pad of a dock seal: a thick black foam body with rounded edges on a galvanised backing plate."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, -hx + 0.012, -hy, hy, 0.0, sz, GALV)
    m.box(-hx + 0.012, hx, -hy + 0.004, hy - 0.004, 0.012, sz - 0.012, RUBBER, bevel=0.025)
    return [m.finish()]


def dock_seal_head(d, ctx, text):
    """The head of a dock seal over the door: a foam block with a sloping lip, on a backing plate."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, -hx + 0.012, -hy, hy, 0.0, sz, GALV)
    profile = [(-hx + 0.012, 0.0), (-hx + 0.012, sz), (0.04, sz), (hx, sz - 0.07), (hx, 0.06), (0.04, 0.0)]
    m.extrude(profile, "y", -hy + 0.004, hy - 0.004, RUBBER)
    return [m.finish()]


def dock_bumper(d, ctx, text):
    """A laminated rubber dock bumper: three bevelled layers on a steel plate, four bolt heads on its face."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, -hx + 0.012, -hy, hy, 0.0, sz, GALV)
    layer = (sz - 0.012) / 3
    for i in range(3):
        m.box(-hx + 0.012, hx - 0.008, -hy + 0.004, hy - 0.004, i * (layer + 0.006), i * (layer + 0.006) + layer,
              RUBBER, bevel=0.008)
    for y in (-0.07, 0.07):
        for z in (0.05, sz - 0.05):
            m.cylinder("x", (hx - 0.004, y, z), 0.014, 0.008, STEEL, 6)
    return [m.finish()]


def dock_leveler(d, ctx, text):
    """A flush dock leveler: a tread-plate deck with a separate lip plate across its front, a gap line between."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, hx - 0.105, -hy, hy, 0.0, sz, TREAD, bevel=0.0008)
    m.box(hx - 0.1, hx, -hy, hy, 0.0, sz, STEEL, bevel=0.0008)
    return [m.finish()]


def dock_light(d, ctx, text):
    """A dock light: a wall plate, a bent arm and an LED lamp head tipped down at the door."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx = sx / 2
    m.box(-hx, -hx + 0.01, -0.05, 0.05, 0.05, 0.15, STEEL)
    m.tube([(-hx + 0.01, 0.0, 0.10), (-0.03, 0.0, 0.15), (0.0, 0.0, 0.15)], 0.009, STEEL, 8)
    with m.turned(30, "y", (0.012, 0.0, 0.10)):
        m.box(-0.013, 0.037, -0.1, 0.1, 0.055, 0.145, PLASTIC, bevel=0.004)
        m.box(0.037, 0.040, -0.085, 0.085, 0.065, 0.135, LENS_LOW)
    return [m.finish()]


def door_number(d, ctx, text):
    """A dock door's number: a big black digit on a white plate with a black border."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.text = text
    m.box(-hx, hx - 0.002, -hy, hy, 0.0, sz, PAPER)
    b = 0.015
    for y0, y1, z0, z1 in ((-hy, hy, 0.0, b), (-hy, hy, sz - b, sz), (-hy, -hy + b, 0.0, sz), (hy - b, hy, 0.0, sz)):
        m.box(hx - 0.002, hx, y0, y1, z0, z1, BARS)
    m.print_text(text, hx - 0.0005, -hy + 0.04, hy - 0.04, 0.035, sz - 0.035, INK, bold=0.03)
    return [m.finish()]

