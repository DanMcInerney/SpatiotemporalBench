# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's dressing, drawn: `BUILDERS` maps a `proc:` recipe's builder name to `fn(d, ctx, text) -> [objects]`
(`dressing.build` calls it for every `Dressing` of `dressing_layout_downtown.layout`).

Runs inside Blender's own Python and is never imported by `stbench`. A builder makes its piece in the Dressing's own
frame (origin at the base centre, x its facing, y across, z up; `d.size` is its box) and returns unlinked objects with
their mesh and materials; `dressing.build` places, names and links them (`pass_index` 0). Every vertex stays inside the
box (what is not exact by construction is fitted to it). The street's own materials come from `materials_street`
(paint, iron, foliage, bark), each made once per scene and found again by name; metals and rubber come from the library
of named materials (`ctx.material`) where it has one. Seeded by the piece's own place, so a map renders the same.
"""

from __future__ import annotations

import hashlib
import math
import random

import bpy
import materials_street as ms
import street_tree as trees
from dressing_spec import Dressing
from mathutils import Matrix
from meshkit import fit, lathe, placed, rod, slab
from scene_mesh import MeshBuilder, rect
from street_marks import Mark, paint_rects

WOOD, STEEL, CONCRETE, VOID = (0.30, 0.19, 0.10), (0.05, 0.05, 0.055), (0.52, 0.51, 0.48), (0.008, 0.008, 0.009)


def _made(name: str, make):
    """The scene's material of this name, made on first use (a scene rebuild frees every material)."""
    return bpy.data.materials.get(name) or make()


def _named(ctx, name: str, fallback):
    """A material of the named library (lane M's), else the `fallback`'s."""
    try:
        return ctx.material(name)
    except KeyError:
        return fallback()


def _rng(d: Dressing, text: str | None = None) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(repr((d.cls, d.pos, text)).encode()).digest()[:8], "big"))


def _object(d: Dressing, mb: MeshBuilder, box=None) -> bpy.types.Object:
    """The unlinked object of a finished builder: its mesh fitted to `box` (the Dressing's size) when given."""
    if box is not None:
        fit(mb.bm, box)
    return bpy.data.objects.new(f"dressing_{d.cls}", mb.mesh())


def _plain(name: str, rgb, rough: float, metal: float = 0.0):
    def make():
        m = ms.plain(name, ms.lin(rgb), rough)
        ms.principled(m).inputs["Metallic"].default_value = metal
        return m
    return _made(name, make)


# -- paint and iron -------------------------------------------------------------------------------------------------
def paint(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """Road paint: `text` is `<style>:<axis>` (`street_marks.paint_rects`); flat quads at the top of the box."""
    style, axis = (text or "bar:x").split(":")
    hx, hy, h = d.size[0] / 2, d.size[1] / 2, d.size[2]
    mark = Mark(d.cls, style, (-hx, hx, -hy, hy), axis)
    mb = MeshBuilder(f"dressing_{d.cls}")
    kind = "yellow" if style == "double" else "white"
    slot = mb.slot(kind, _made(f"paint_{kind}", lambda: ms.road_paint(kind)))
    for x0, x1, y0, y1 in paint_rects(mark):
        rect(mb.bm, x0, x1, y0, y1, h, slot)
    return [_object(d, mb)]


def manhole(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A round cover: a frame ring, a recessed disc with a groove and six raised ribs."""
    r, h = d.size[0] / 2, d.size[2]
    mb = MeshBuilder(f"dressing_{d.cls}")
    iron = mb.slot("iron", _made("street_iron", lambda: ms.cast_iron("street_iron")))
    lathe(mb.bm, [(r, 0.0), (r, h * 0.85), (r - 0.04, h), (r - 0.07, h * 0.6), (r - 0.09, h * 0.62), (0.0, h * 0.62)],
          iron, segments=32)
    for k in range(6):
        placed(mb.bm, Matrix.Rotation(k * math.pi / 6, 4, "Z"),
               lambda: slab(mb.bm, -(r - 0.1), r - 0.1, -0.012, 0.012, h * 0.55, h * 0.9, iron))
    return [_object(d, mb, d.size)]


def storm_drain(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A gutter inlet: an iron frame with a row of dark slots along the kerb, between raised bars."""
    sx, sy, h = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    iron = mb.slot("iron", _made("street_iron", lambda: ms.cast_iron("street_iron")))
    void = mb.slot("void", _plain("drain_void", VOID, 0.9))
    hx, hy, w = sx / 2, sy / 2, 0.035
    slab(mb.bm, -hx, hx, -hy, hy, 0.0, h * 0.8, iron)
    for x0, x1, y0, y1 in ((-hx, hx, -hy, -hy + w), (-hx, hx, hy - w, hy), (-hx, -hx + w, -hy, hy),
                           (hx - w, hx, -hy, hy)):
        slab(mb.bm, x0, x1, y0, y1, 0.0, h, iron)
    n = 8
    pitch = (sy - 2 * w - 0.06) / n
    for i in range(n):
        y0 = -hy + w + 0.03 + i * pitch + pitch * 0.25
        rect(mb.bm, -hx + w + 0.03, hx - w - 0.03, y0, y0 + pitch * 0.5, h * 0.8 + 0.0005, void)
    return [_object(d, mb, d.size)]


# -- trees ------------------------------------------------------------------------------------------------------------
def street_tree(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """The tree pit and the stem."""
    mb = MeshBuilder(f"dressing_{d.cls}")
    bark = mb.slot("bark", _made("bark", ms.bark))
    iron = mb.slot("iron", _made("street_iron", lambda: ms.cast_iron("street_iron")))
    trees.trunk(mb.bm, d.size, bark, iron)
    return [_object(d, mb)]


def tree_crown(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """The leaf mass over the stem, its limbs and leader; `text` is the shape (0 to 5) the layout drew for it."""
    mb = MeshBuilder(f"dressing_{d.cls}")
    bark = mb.slot("bark", _made("bark", ms.bark))
    leaf = mb.slot("leaf", _made("foliage", ms.foliage))
    trees.crown(mb.bm, d.size, bark, leaf, _rng(d, text))
    return [_object(d, mb)]


# -- street furniture -------------------------------------------------------------------------------------------------
def bench(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A slatted park bench facing +x: four seat slats, three back slats and a steel frame at each end with arms."""
    mb = MeshBuilder(f"dressing_{d.cls}")
    wood = mb.slot("wood", _plain("bench_wood", WOOD, 0.7))
    steel = mb.slot("steel", _plain("bench_steel", STEEL, 0.5, 0.2))
    hy = d.size[1] / 2 - 0.02
    for i in range(4):  # the seat: slats 8 cm wide with 2 cm gaps, from the back forward
        x0 = -0.2 + i * 0.105
        slab(mb.bm, x0, x0 + 0.085, -hy, hy, 0.43, 0.46, wood, bevel=0.005, segments=1)
    for z0 in (0.52, 0.63, 0.74):  # the back, a little raked
        rake = 0.04 * (z0 - 0.52) / 0.22
        slab(mb.bm, -0.265 - rake, -0.235 - rake, -hy, hy, z0, z0 + 0.085, wood, bevel=0.005, segments=1)
    for s in (-1.0, 1.0):
        y = s * (hy - 0.025)
        for x in (-0.2, 0.2):
            slab(mb.bm, x - 0.02, x + 0.02, y - 0.02, y + 0.02, 0.0, 0.43, steel)
        slab(mb.bm, -0.275, -0.245, y - 0.02, y + 0.02, 0.43, 0.84, steel)
        slab(mb.bm, -0.25, 0.25, y - 0.02, y + 0.02, 0.60, 0.635, steel)  # arm rest
    return [_object(d, mb, d.size)]


def planter(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A concrete trough with soil and a low mass of shrub."""
    sx, sy, _ = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    concrete = mb.slot("concrete", _plain("planter_concrete", CONCRETE, 0.85))
    soil = mb.slot("soil", _plain("planter_soil", (0.10, 0.07, 0.05), 0.95))
    leaf = mb.slot("leaf", _made("foliage", ms.foliage))
    slab(mb.bm, -sx / 2, sx / 2, -sy / 2, sy / 2, 0.0, 0.5, concrete, bevel=0.02, segments=1)
    slab(mb.bm, -sx / 2 + 0.07, sx / 2 - 0.07, -sy / 2 + 0.07, sy / 2 - 0.07, 0.49, 0.505, soil)
    trees.lumps(mb.bm, leaf, _rng(d), (0.0, 0.0, 0.62), (0.16, sy * 0.3, 0.1), 10, (0.17, 0.24))
    return [_object(d, mb, d.size)]


def bike_rack(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A row of five inverted-U racks across the piece's length, standing on foot plates."""
    sx, sy, sz = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    steel = mb.slot("steel", _named(ctx, "galvanised_steel", lambda: _plain("rack_steel", (0.5, 0.5, 0.52), 0.4, 0.85)))
    r, leg, tube = sx / 2 - 0.025, sz - sx / 2 - 0.02, 0.022
    for i in range(5):
        y = (i - 2) * (sy - 0.1) / 4
        for s in (-1.0, 1.0):
            rod(mb.bm, (s * r, y, 0.0), (s * r, y, leg), tube, steel, segments=8)
            slab(mb.bm, s * r - 0.025, s * r + 0.025, y - 0.03, y + 0.03, 0.0, 0.006, steel)
        pts = [(r * math.cos(math.pi * k / 6), leg + r * math.sin(math.pi * k / 6)) for k in range(7)]
        for (x0, z0), (x1, z1) in zip(pts, pts[1:], strict=False):
            rod(mb.bm, (x0, y, z0), (x1, y, z1), tube, steel, segments=8)
    return [_object(d, mb, d.size)]


BUILDERS = {f.__name__: f for f in (paint, manhole, storm_drain, street_tree, tree_crown, bench, planter, bike_rack)}
