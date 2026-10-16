# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's wall dressing (`dressing_layout_warehouse`): steel columns, insulated cladding, the concrete dado
with its hazard-striped safety band, and the conduit along the walls.

Runs inside Blender's own Python and is never imported by `stbench`. A wall piece is drawn in its Dressing's frame: x
out of the wall (its back is at the box's back face), y along the wall, z up. The band is paint, drawn as black and
yellow hazard stripes, so no wall carries a plain yellow block.
"""

from __future__ import annotations

from dressing_warehouse_kit import CLADDING, CONCRETE, GALV, GREY, RUBBER, STEEL, YELLOW, Mesh, clip_u

BAND_STRIPE_M = 0.10  # the horizontal width of one yellow or black stripe of the band, at 45 degrees


def steel_column(d, ctx, text):
    """An H column set into the wall: two flanges standing out of it and the web between them half way out, a cap
    plate and, when it stands on the slab, a base plate with anchor bolts."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy, flange = sx / 2, sy / 2, 0.012
    m.box(-hx, hx, -hy, -hy + flange, 0.0, sz, GREY)
    m.box(-hx, hx, hy - flange, hy, 0.0, sz, GREY)
    m.box(-0.004, 0.004, -hy + flange, hy - flange, 0.0, sz, GREY)
    m.box(-hx, hx, -hy, hy, sz - 0.015, sz, GREY)
    if d.pos[2] < 0.01:  # it stands on the slab; behind a rack it starts above the stock instead
        m.box(-hx, hx, -hy, hy, 0.0, 0.02, GREY, bevel=0.002)
        for x in (-hx + 0.018, hx - 0.018):
            for y in (-hy + 0.025, hy - 0.025):
                m.box(x - 0.013, x + 0.013, y - 0.013, y + 0.013, 0.02, 0.036, STEEL)
    return [m.finish()]


def wall_cladding(d, ctx, text):
    """An insulated sandwich panel: a flat face with raised ribs, a dark joint backing that shows as a shadow line
    between panels, and a galvanised flashing at its foot."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, -hx + 0.005, -hy, hy, 0.0, sz, RUBBER)
    m.box(-hx + 0.005, hx - 0.008, -hy + 0.003, hy - 0.003, 0.035, sz, CLADDING)
    m.box(-hx + 0.005, hx, -hy + 0.003, hy - 0.003, 0.0, 0.035, GALV)
    ribs = max(1, round(sy / 0.3))
    for i in range(ribs):
        c = -hy + (i + 0.5) * sy / ribs
        m.box(hx - 0.008, hx, c - 0.025, c + 0.025, 0.05, sz - 0.01, CLADDING)
    return [m.finish()]


def wall_dado(d, ctx, text):
    """A precast concrete panel: two courses with a shadow joint between them, edges slightly eased."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.box(-hx, 0.0, -hy + 0.003, hy - 0.003, 0.0, sz, CONCRETE)
    half = sz / 2
    m.box(0.0, hx, -hy + 0.003, hy - 0.003, 0.0, half - 0.002, CONCRETE, bevel=0.004)
    m.box(0.0, hx, -hy + 0.003, hy - 0.003, half + 0.002, sz, CONCRETE, bevel=0.004)
    return [m.finish()]


def safety_band(d, ctx, text):
    """The yellow safety band at the foot of the dado, as hazard paint: yellow stripes at 45 degrees on a black band."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy, w = sx / 2, sy / 2, BAND_STRIPE_M
    m.box(-hx, hx - 0.008, -hy, hy, 0.0, sz, RUBBER)
    y = -hy - sz
    while y < hy:
        stripe = clip_u([(y, 0.0), (y + w, 0.0), (y + w + sz, sz), (y + sz, sz)], -hy, hy)
        if len(stripe) >= 3:
            m.extrude(stripe, "x", hx - 0.008, hx, YELLOW)
        y += 2 * w
    return [m.finish()]


def conduit(d, ctx, text):
    """A steel conduit along the wall, horizontal (along y) or a vertical drop (along z), whichever is the piece's
    long side: saddle clamps every 0.9 m and a coupling every 3 m."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    if sz > sy:  # a vertical drop: the run is up the wall
        r = min(sx, sy) / 2 - 0.0005
        m.cylinder("z", (0.0, 0.0, sz / 2), r, sz, GALV, 16)
        for i in range(1, int(sz / 0.9)):
            z = 0.9 * i - 0.3
            m.box(-sx / 2 - 0.0005, sx / 2 + 0.0005, -sy / 2 - 0.0005, sy / 2 + 0.0005, z - 0.008, z + 0.008, STEEL)
        for i in range(int(sz / 3.0)):
            m.band("z", (0.0, 0.0, 1.4 + 3.0 * i), r, r + 0.002, 0.04, STEEL, 16)
        return [m.finish()]
    hy, r = sy / 2, min(sx, sz) / 2 - 0.0005
    m.cylinder("y", (0.0, 0.0, sz / 2), r, sy, GALV, 16)
    for i in range(1, int(sy / 0.9)):
        y = -hy + 0.9 * i - 0.3
        m.box(-sx / 2 - 0.0005, sx / 2 + 0.0005, y - 0.008, y + 0.008, -0.0005, sz + 0.0005, STEEL)
    for i in range(int(sy / 3.0)):
        m.band("y", (0.0, -hy + 1.4 + 3.0 * i, sz / 2), r, r + 0.002, 0.04, STEEL, 16)
    return [m.finish()]
