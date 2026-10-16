# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The fittings on the warehouse's walls (`dressing_layout_warehouse`): a 2 kg fire extinguisher on its bracket, an
alarm call point, a security camera, the electrical panel and the utility cabinet, and a swag of cables.

Runs inside Blender's own Python and is never imported by `stbench`. A piece is drawn in its Dressing's frame (x out of
the wall, y along it, z up) and fills its box. The call point, the camera, the big power panel and the cable swag are
Poly Haven models when the library has them (`asset:<id>`); these builders are the procedural stand-ins `dressing.py`
draws in their place (it calls the builder of the class's name), made to the same boxes.
"""

from __future__ import annotations

from dressing_warehouse_kit import BARS, GREY, INK, LENS_LOW, PAPER, PLASTIC, RED, RUBBER, STEEL, Mesh


def fire_extinguisher(d, ctx, text):
    """A 2 kg dry-powder extinguisher on a wall bracket: a red cylinder with a strap, a valve head with lever and
    gauge, a hose and a white label."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx = sx / 2
    r = hx - 0.006 - 0.0005  # the body's radius: against its bracket, inside the wall band
    cx = -hx + 0.006 + r
    m.box(-hx, -hx + 0.006, -0.04, 0.04, 0.12, 0.28, STEEL)  # the wall bracket
    body = [(0.0, 0.0), (0.75 * r, 0.0), (0.96 * r, 0.012), (r, 0.03), (r, 0.27), (0.92 * r, 0.30), (0.66 * r, 0.318),
            (0.0, 0.318)]
    m.revolve(body, RED, (cx, 0.0, 0.0), segments=32)
    m.band("z", (cx, 0.0, 0.2), r, r + 0.003, 0.02, STEEL, 32)  # the strap
    m.revolve([(0.0, 0.318), (0.017, 0.318), (0.017, 0.345), (0.0, 0.345)], STEEL, (cx, 0.0, 0.0), segments=16)
    m.box(cx - 0.03, cx + 0.03, -0.02, 0.02, 0.345, 0.385, STEEL, bevel=0.004)  # the valve
    m.box(cx - 0.032, cx + 0.052, -0.012, 0.012, 0.388, 0.398, RED, bevel=0.003)  # the lever
    m.box(cx - 0.032, cx + 0.045, -0.012, 0.012, 0.362, 0.372, RED, bevel=0.003)  # the carrying handle
    m.cylinder("y", (cx - 0.005, 0.0, 0.415), 0.017, 0.012, STEEL, 20)  # the pressure gauge
    m.tube([(cx + 0.03, -0.015, 0.355), (cx + 0.043, -0.043, 0.33), (cx + 0.043, -0.043, 0.21)], 0.006, RUBBER, 8)
    m.box(cx + r - 0.001, cx + r + 0.0025, -0.025, 0.025, 0.10, 0.22, PAPER)  # the label
    m.box(cx + r + 0.0025, cx + r + 0.0035, -0.018, 0.018, 0.14, 0.18, INK)
    return [m.finish()]


def fire_alarm(d, ctx, text):
    """A manual call point: a red box with a glass pane in a frame and a hinged hood over it (stand-in for the Poly
    Haven `fire_alarm`)."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, hx - 0.009, -hy, hy, 0.0, sz, RED, bevel=0.003)
    m.box(hx - 0.009, hx, -hy + 0.008, hy - 0.008, 0.012, sz - 0.012, RED, bevel=0.002)
    m.box(hx - 0.009, hx - 0.002, -0.031, 0.031, 0.04, 0.104, PAPER)  # the breakable pane
    m.box(hx - 0.002, hx, -0.012, 0.012, 0.062, 0.082, INK)
    m.box(hx - 0.009, hx, -hy + 0.008, hy - 0.008, sz - 0.03, sz - 0.012, RED, bevel=0.002)  # the hood
    return [m.finish()]


def security_camera(d, ctx, text):
    """A bullet camera on a wall arm, tipped down a little: a white housing under a sun shield with a dark lens
    (stand-in for the Poly Haven `security_camera_01`)."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx = sx / 2
    m.box(-hx, -hx + 0.01, -0.06, 0.06, sz - 0.14, sz - 0.01, GREY, bevel=0.003)
    m.tube([(-hx + 0.01, 0.0, sz - 0.075), (-hx + 0.06, 0.0, sz - 0.075), (-0.15, 0.0, sz - 0.12)], 0.012, GREY, 8)
    with m.turned(10, "y", (0.03, 0.0, sz - 0.14)):
        m.cylinder("x", (0.03, 0.0, sz - 0.14), 0.038, 0.34, PLASTIC, 24)
        m.box(-0.14, 0.2, -0.05, 0.05, sz - 0.098, sz - 0.092, PLASTIC)  # the sun shield
        m.cylinder("x", (0.2, 0.0, sz - 0.14), 0.030, 0.012, BARS, 20)
        m.cylinder("x", (0.211, 0.0, sz - 0.14), 0.020, 0.006, LENS_LOW, 20)
    return [m.finish()]


def _cabinet(d, ctx, doors: int, louvres: bool):
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, hx - 0.012, -hy, hy, 0.0, sz, GREY, bevel=0.003)
    edges = [-hy + 0.01 + k * (2 * hy - 0.02) / doors for k in range(doors + 1)]
    for a, b in zip(edges, edges[1:], strict=False):
        m.box(hx - 0.012, hx - 0.002, a + 0.002, b - 0.002, 0.01, sz - 0.01, GREY, bevel=0.003)
    for k in range(3):  # hinge barrels on the left edge
        m.cylinder("z", (hx - 0.01, -hy + 0.008, 0.1 + k * (sz - 0.2) / 2), 0.007, 0.05, STEEL, 10)
    m.box(hx - 0.002, hx, 0.5 * hy - 0.01 if doors == 1 else -0.03, 0.5 * hy + 0.01 if doors == 1 else 0.03,
          0.45 * sz, 0.55 * sz, STEEL, bevel=0.002)  # the latch
    if louvres:
        for z in (0.07 + 0.03 * k for k in range(6)):
            for a, b in zip(edges, edges[1:], strict=False):
                m.box(hx - 0.0035, hx - 0.002, a + 0.04, b - 0.04, z, z + 0.012, RUBBER)
    else:
        m.box(hx - 0.002, hx - 0.0005, -0.04, 0.04, sz - 0.1, sz - 0.04, PAPER)  # a warning label
        for z in (sz - 0.055, sz - 0.07, sz - 0.085):
            m.box(hx - 0.0005, hx, -0.032, 0.032, z, z + 0.006, INK)
    return [m.finish()]


def power_panel(d, ctx, text):
    """An electrical panel: a grey steel cabinet with one door, hinges, a latch and a warning label. Also the stand-in
    for the Poly Haven `power_box_01`."""
    return _cabinet(d, ctx, 1, False)


def utility_enclosure(d, ctx, text):
    """A utility cabinet: grey steel with two louvred doors, hinges and a latch."""
    return _cabinet(d, ctx, 2, True)
