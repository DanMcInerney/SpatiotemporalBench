# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What is fixed to a pallet rack row (`dressing_layout_warehouse`): yellow upright protectors and end-of-aisle guards
(open rails and brackets, never a solid yellow block), the readable location label on every beam and the aisle sign on
a row's end.

Runs inside Blender's own Python and is never imported by `stbench`. A piece faces out of the rack face it is fixed to
(its x), y across it, z up. A label's code and a sign's lines are the recipe's text (lines apart by `|`), drawn as
geometry in dark print on a light plate and kept as the object's `stbench_text`.
"""

from __future__ import annotations

from dressing_warehouse_kit import BARS, INK, PAPER, YELLOW, Mesh

RAIL_M = 0.025  # the width of a guard's flat bar
END_RAIL_DEPTH_M = 0.03  # how far an end-of-aisle guard's rails stand out of its posts (the guard is 0.075 m deep)


def upright_protector(d, ctx, text):
    """A guard in front of an upright's foot: two flat-bar loops (low and high) open to the sides, held apart by
    two bars at the front corners, bolted to the post at the back. Open rails, so the aisle sees through it."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    for z in (0.04, sz - 0.08):  # the loops sit a millimetre inside the corner bars, so no two faces share a plane
        m.box(hx - RAIL_M - 0.001, hx - 0.001, -hy + 0.01, hy - 0.01, z, z + 0.04, YELLOW, bevel=0.002)
        for y0, y1 in ((-hy + 0.001, -hy + RAIL_M - 0.001), (hy - RAIL_M + 0.001, hy - 0.001)):
            m.box(-hx, hx - 0.01, y0, y1, z, z + 0.04, YELLOW, bevel=0.002)
    for y0, y1 in ((-hy, -hy + RAIL_M), (hy - RAIL_M, hy)):
        m.box(hx - RAIL_M, hx, y0, y1, 0.0, sz, YELLOW, bevel=0.002)
    return [m.finish()]


def end_guard(d, ctx, text):
    """A guard across the open end of a rack row: two posts and three flat rails between them, the rails thin enough
    that from above the yellow covers well under 0.6 of the guard's 0.075 m depth (0.47: it is no solid block)."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy, post = sx / 2, sy / 2, 0.06
    for y0, y1 in ((-hy, -hy + post), (hy - post, hy)):
        m.box(-hx, hx, y0, y1, 0.0, sz, YELLOW, bevel=0.003)
    x0 = hx - END_RAIL_DEPTH_M - 0.001
    for z in (0.09, 0.30, sz - 0.09):  # rails run into the posts, a millimetre short of their faces
        m.box(x0, hx - 0.001, -hy + post / 2, hy - post / 2, z, z + 0.06, YELLOW, bevel=0.003)
    return [m.finish()]


def beam_label(d, ctx, text):
    """A beam's location label (aisle-bay-level, as `A-03-02`): a white plate with the code in large dark print over a
    Code 39 barcode of it."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    m.text = text
    m.box(-hx, hx - 0.0012, -hy, hy, 0.0, sz, PAPER)
    m.print_text(text, hx - 0.0004, -hy + 0.012, hy - 0.012, 0.024, sz - 0.003, INK, wide=1.3)
    m.barcode(text, hx - 0.0012, hx - 0.0004, -hy + 0.03, hy - 0.03, 0.004, 0.018, BARS)
    return [m.finish()]


def aisle_sign(d, ctx, text):
    """An aisle sign: a white board in a dark frame, its lines of text (the aisle, then its bays) in dark print as
    large as the board takes."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy, rim = sx / 2, sy / 2, 0.02
    m.text = text
    m.box(-hx, hx - 0.01, -hy, hy, 0.0, sz, PAPER)
    for y0, y1, z0, z1 in ((-hy, hy, 0.0, rim), (-hy, hy, sz - rim, sz), (-hy, -hy + rim, 0.0, sz),
                           (hy - rim, hy, 0.0, sz)):
        m.box(-hx, hx, y0, y1, z0, z1, BARS)
    lines = text.split("|")
    share = [0.62, 0.38][:len(lines)] if len(lines) > 1 else [1.0]
    top = sz - rim - 0.03
    for line, f in zip(lines, share, strict=False):
        height = (sz - 2 * rim - 0.06) * f
        m.print_text(line, hx - 0.0095, -hy + rim + 0.04, hy - rim - 0.04, top - height + 0.012, top - 0.012, INK)
        top -= height
    return [m.finish()]
