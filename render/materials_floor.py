# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's concrete slab, worn: `slab(plan)` is the material of every region of the floor.

Runs inside Blender's own Python and is never imported by `stbench`. A pale, mottled concrete (large soft patches and a
fine grain in two greys, a faint pitted bump) with 12 mm saw-cut joints every `JOINT_M`, on object coordinates (a floor
mesh sits at the map's origin, so they are map coordinates), no file and no download. With a `floor_marks.Plan` it also
wears what a working floor wears, all as a layer of the one material (no geometry, so no depth, index or shadow of its
own): worn OSHA-yellow paint along the rack fronts, the walkway and the dock doors' aprons, rubber tyre marks along
the traffic, a polish where it drives, and a few oil stains. Without a plan (a test, a scene built without a map) the
slab has the joints and the stains. Deterministic: no noise has a seed of its own.

The paint is a flat colour on the floor, thin and worn through in patches, so it cannot be mistaken for a thing: the
staging lanes, which are furniture, are the only other yellow on the slab.
"""

from __future__ import annotations

import bpy
from floor_marks import Plan, Rect
from materials_city import lin
from shader_nodes import G, new_material, principled

JOINT_M = 6.0  # the saw-cut joints form a grid of this pitch
CONCRETE_GREYS = ((0.19, 0.19, 0.185), (0.30, 0.295, 0.285))  # linear: the darker and lighter ends of the mottling
PAINT_YELLOW = (0.90, 0.72, 0.07)  # OSHA safety yellow, sRGB
INNER = 0.10  # a hatched apron has a solid border this wide
HATCH_PITCH, HATCH_STRIPE = 0.25, 0.10


def _rect(g: G, x, y, r: Rect, shrink: float = 0.0):
    """1 inside the rectangle (shrunk by `shrink` on every side), else 0."""
    return g.math("MULTIPLY", g.between(x, r[0] + shrink, r[2] - shrink), g.between(y, r[1] + shrink, r[3] - shrink))


def _union(g: G, masks: list):
    out = masks[0] if masks else 0.0
    for m in masks[1:]:
        out = g.math("MAXIMUM", out, m)
    return out


def _rects(g: G, x, y, rects: list[Rect] | tuple[Rect, ...], shrink: float = 0.0):
    """1 inside any of the rectangles, else 0. Rectangles that share a y range (the six rack lines, the streaks of the
    aisles) test it once and their x ranges separately; the rest share an x range or stand alone. Half the nodes of one
    mask a rectangle, and a floor is shaded wherever the camera looks."""
    terms, rest = [], []
    for key in (1, 0):  # first group by the y range (the rectangle's corners 1 and 3), then the leftovers by x
        groups: dict[tuple, list[Rect]] = {}
        for r in rects if key == 1 else rest:
            groups.setdefault((r[key], r[key + 2]), []).append(r)
        rest = []
        for (lo, hi), rs in groups.items():
            if len(rs) == 1 and key == 1:
                rest += rs
                continue
            across, along = (x, y) if key == 1 else (y, x)
            slots = (0, 2) if key == 1 else (1, 3)
            span = _union(g, [g.between(across, r[slots[0]] + shrink, r[slots[1]] - shrink) for r in rs])
            terms.append(g.math("MULTIPLY", g.between(along, lo + shrink, hi - shrink), span))
    return _union(g, terms)


def _hatches(g: G, x, y, rects):
    """The aprons: a border and, inside it, diagonal stripes (45 degrees, `HATCH_STRIPE` wide every `HATCH_PITCH` along
    x)."""
    stripes = g.math("LESS_THAN", g.math("FLOORED_MODULO", g.math("SUBTRACT", x, y), HATCH_PITCH), HATCH_STRIPE)
    inside = _rects(g, x, y, rects, INNER)
    return g.math("MULTIPLY", _rects(g, x, y, rects), g.math("MAXIMUM", g.math("SUBTRACT", 1.0, inside), stripes))


def _track_mask(g: G, x, y, plan: Plan, obj):
    """Tyre marks: the tracks' rectangles times a streaky noise stretched along them. One noise serves both directions:
    where a track runs along x the two coordinates are swapped first."""
    swap = _rects(g, x, y, [r for r in plan.tracks if r[3] - r[1] <= r[2] - r[0]])  # the tracks that run along x
    u = g.math("ADD", x, g.math("MULTIPLY", swap, g.math("SUBTRACT", y, x)))
    v = g.math("ADD", y, g.math("MULTIPLY", swap, g.math("SUBTRACT", x, y)))
    streaks = g.noise(g.combine(u, g.math("MULTIPLY", v, 0.05), 0.0), 70.0, 3.0, 0.6)  # long in v
    streak = g.math("MULTIPLY_ADD", g.smooth(streaks, 0.40, 0.62), 0.6, 0.4)
    return g.math("MULTIPLY", _rects(g, x, y, plan.tracks), streak)


def slab(plan: Plan | None = None) -> bpy.types.Material:
    """A warehouse slab, worn; with `plan` its markings and tyre marks too."""
    m = new_material("shader:concrete")
    g, p = G(m.node_tree), principled(m)
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    x, y, _ = g.xyz(obj)
    grain = g.noise(obj, 90.0, 4.0)
    patches = g.noise(obj, 0.45, 4.0)
    shade = g.math("ADD", g.math("MULTIPLY", patches, 0.7), g.math("MULTIPLY", grain, 0.3))
    col = g.ramp(shade, [(0.3, CONCRETE_GREYS[0]), (0.7, CONCRETE_GREYS[1])])
    joints = g.math("MAXIMUM", g.grid(x, JOINT_M, 0.012), g.grid(y, JOINT_M, 0.012))
    col = g.mix(joints, col, (0.05, 0.05, 0.05))
    rough = 0.5
    if plan is not None:
        worn = g.smooth(g.noise(obj, 18.0, 3.0, 0.65), 0.50, 0.70)  # paint worn through in patches
        opacity = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.math("MULTIPLY", worn, 0.55)), 0.95)
        paint = g.math("MAXIMUM", _rects(g, x, y, plan.lines), _hatches(g, x, y, plan.hatches))
        yellow = g.vmath("SCALE", lin(PAINT_YELLOW), scale=g.math("MULTIPLY_ADD", patches, 0.15, 0.93))
        coverage = g.math("MULTIPLY", paint, opacity)
        col = g.mix(coverage, col, yellow)
        tyres = g.math("MULTIPLY", _track_mask(g, x, y, plan, obj), 0.45)
        col = g.mix(tyres, col, g.vmath("SCALE", col, scale=0.3))
        polish = g.math("MULTIPLY", _rects(g, x, y, plan.lanes),
                        g.math("MULTIPLY_ADD", patches, 0.6, 0.5))
        rough = g.math("SUBTRACT", g.math("SUBTRACT", rough, g.math("MULTIPLY", polish, 0.10)),
                       g.math("MULTIPLY", tyres, 0.12))
        rough = g.mix(coverage, rough, 0.42, "FLOAT")
    stains = g.math("MULTIPLY", g.smooth(patches, 0.66, 0.74), 0.9)  # the slab's lightest patches are oil-stained
    col = g.mix(g.math("MULTIPLY", stains, 0.5), col, g.vmath("SCALE", col, scale=0.35))
    rough = g.mix(stains, rough, 0.22, "FLOAT")  # an oil stain is glossy
    g.put(p, "Base Color", col)
    g.put(p, "Roughness", rough)
    g.put(p, "Normal", g.bump(g.noise(obj, 90.0, 2.0), 0.15, 0.002))  # (a bump evaluates its height three times)
    return m
