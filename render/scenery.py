# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The city beyond an outdoor map's edge, drawn: roads, sidewalks with kerbs, painted bays, building masses and street
furniture, from the layout of `scenery_layout.py` (which decides where everything is and why).

Runs inside Blender's own Python and is never imported by `stbench`. Everything here is static structure with
`pass_index` 0: it is never an entity, so no instance pass or oracle ever names it. The roads, sidewalks and bays are
one mesh; each building is an object of its own (its facade shader varies by the object's random value); each kind of
street furniture is built once and instanced (merging hundreds of small props into one bmesh is quadratic).
"""

from __future__ import annotations

import math
import random

import bpy
import facades
import materials_city
import materials_street
import scenery_layout as layout
import street_marks
import street_tree
from materials_street import lin, plain
from mathutils import Matrix
from meshkit import lathe, placed
from scene_mesh import MeshBuilder, cube, rect, rod
from vehicle_kit import solid

CURB_H = 0.15  # the sidewalk slabs' height, the map's own
PAINT_Z, LINE_W = 0.012, 0.1


def _walk(mb: MeshBuilder, r, top: int, side: int) -> None:
    x0, x1, y0, y1 = layout.aabb(r)
    for f in cube(mb.bm, x0, x1, y0, y1, 0.0, CURB_H, side):
        if f.calc_center_median().z > CURB_H - 1e-3:
            f.material_index = top


def _outline(mb: MeshBuilder, r, slot: int) -> None:
    """A bay: four painted lines round a rectangle."""
    x0, x1, y0, y1 = layout.aabb(r)
    w = LINE_W
    for a, b, c, d in ((x0, x1, y0, y0 + w), (x0, x1, y1 - w, y1), (x0, x0 + w, y0 + w, y1 - w),
                       (x1 - w, x1, y0 + w, y1 - w)):
        rect(mb.bm, a, b, c, d, PAINT_Z, slot)


def _ground(lay, asphalt: bpy.types.Material) -> None:
    mb = MeshBuilder("scenery_ground")
    road = mb.slot("asphalt", asphalt)
    top, side = mb.slot("sidewalk", materials_street.sidewalk()), mb.slot("kerb", materials_street.kerb())
    white = mb.slot("white", materials_street.road_paint("white"))
    for r in lay.roads:
        x0, x1, y0, y1 = layout.aabb(r)
        rect(mb.bm, x0, x1, y0, y1, 0.0, road)
    for w in lay.walks:
        _walk(mb, w.rect, top, side)
    for r in lay.bays:
        _outline(mb, r, white)
    if lay.lines:  # the centre lines and lane dashes of the continued streets
        yellow = mb.slot("yellow", materials_street.road_paint("yellow"))
        for mark in lay.lines:
            for x0, x1, y0, y1 in street_marks.paint_rects(mark):
                rect(mb.bm, x0, x1, y0, y1, PAINT_Z, yellow if mark.style == "double" else white)
    mb.finish()


def _buildings(lay, mats: facades.Materials) -> None:
    for i, b in enumerate(lay.buildings):
        along_x = abs(math.sin(b.face)) > 0.5  # a face that looks along y: the street runs along x
        w, d = (b.rect.sx, b.rect.sy) if along_x else (b.rect.sy, b.rect.sx)
        mb = MeshBuilder(f"scenery_building_{i}")
        facades.add_building(mb, mats, b.style, -w / 2, w / 2, -d / 2, d / 2, b.height)
        mb.finish(0, (b.rect.cx, b.rect.cy, 0.0), b.face + math.pi / 2)


# -- street furniture --------------------------------------------------------------------------------------------
def _lamp(mb: MeshBuilder, mats: dict) -> None:
    """A 7.5 m street lamp with an arm reaching 1.7 m towards +y."""
    pole, head = mb.slot("pole", mats["pole"]), mb.slot("lamp_head", mats["lamp_head"])
    rod(mb.bm, (0, 0, 0), (0, 0, 7.5), 0.08, pole, segments=10)
    rod(mb.bm, (0, 0, 7.3), (0, 1.7, 7.9), 0.05, pole, segments=8)
    solid(mb.bm, -0.35, 0.35, 1.35, 2.05, 7.78, 7.95, head, bevel=0.04, segments=2)


def _signal(mb: MeshBuilder, mats: dict) -> None:
    """A traffic-signal pole with a mast arm reaching 4.5 m towards +y and one head at its end, lamps facing +x."""
    pole, dark = mb.slot("pole", mats["pole"]), mb.slot("signal", mats["lamp_head"])
    lamps = [mb.slot(k, mats[k]) for k in ("sig_r", "sig_a", "sig_g")]
    rod(mb.bm, (0, 0, 0), (0, 0, 5.2), 0.1, pole, segments=10)
    rod(mb.bm, (0, 0, 5.0), (0, 4.5, 5.2), 0.06, pole, segments=8)
    solid(mb.bm, -0.18, 0.18, 4.02, 4.38, 4.2, 5.15, dark, bevel=0.02, segments=1)
    for k, slot in enumerate(lamps):
        zc = 4.95 - 0.3 * k
        solid(mb.bm, 0.18, 0.24, 4.13, 4.27, zc - 0.07, zc + 0.07, slot)


def _hydrant(mb: MeshBuilder, mats: dict) -> None:
    red = mb.slot("hydrant", mats["hydrant"])
    lathe(mb.bm, [(0, 0), (0.14, 0), (0.12, 0.6), (0.15, 0.68), (0, 0.72)], red, segments=10)


TREE_VARIANTS = 4  # distinct trees: each tree of the scenery is one of them, a little larger or smaller


def _tree(mb: MeshBuilder, mats: dict, seed: int) -> None:
    """A street tree in the posts' frame (its arm side, +y, is the road): the pit and stem and a crown over the
    sidewalk, the same pieces the dressing builds for the map's own sidewalks (`street_tree`), turned a quarter from
    their frame (x across the sidewalk) into this one."""
    bark, iron, leaf = mb.slot("bark", mats["bark"]), mb.slot("iron", mats["iron"]), mb.slot("leaf", mats["leaf"])
    turn = Matrix.Rotation(math.pi / 2, 4, "Z")
    placed(mb.bm, turn, lambda: street_tree.trunk(mb.bm, layout.TREE_TRUNK, bark, iron, snug=False))
    at = Matrix.Translation((0.0, layout.TREE_CROWN_SHIFT_M, layout.TREE_TRUNK[2])) @ turn
    placed(mb.bm, at, lambda: street_tree.crown(mb.bm, layout.TREE_CROWN, bark, leaf, random.Random(seed), snug=False))


FURNITURE = {"lamp": _lamp, "signal": _signal, "hydrant": _hydrant}


def _furniture(lay) -> None:
    mats = {"pole": plain("pole", lin((0.18, 0.20, 0.21)), 0.45),
            "lamp_head": plain("lamp_head", lin((0.10, 0.11, 0.12)), 0.4),
            "hydrant": materials_city.shader("matte_red"),
            "sig_r": materials_city.glow("sig_r", (1.0, 0.05, 0.03), 1.5),
            "sig_a": materials_city.glow("sig_a", (1.0, 0.6, 0.05), 1.5),
            "sig_g": materials_city.glow("sig_g", (0.1, 1.0, 0.3), 1.5),
            "bark": bpy.data.materials.get("bark") or materials_street.bark(),
            "iron": bpy.data.materials.get("street_iron") or materials_street.cast_iron("street_iron"),
            "leaf": bpy.data.materials.get("foliage") or materials_street.foliage()}
    protos = {}
    for name, build in FURNITURE.items():
        mb = MeshBuilder(f"scenery_proto_{name}")
        build(mb, mats)
        protos[name] = mb.mesh()
    for k in range(TREE_VARIANTS if any(p.kind == "tree" for p in lay.posts) else 0):
        mb = MeshBuilder(f"scenery_proto_tree{k}")
        _tree(mb, mats, 1000 + k)
        protos[f"tree{k}"] = mb.mesh()
    count: dict[str, int] = {}
    for p in lay.posts:
        n = count[p.kind] = count.get(p.kind, 0) + 1
        proto = f"tree{n % TREE_VARIANTS}" if p.kind == "tree" else p.kind
        ob = bpy.data.objects.new(f"scenery_{p.kind}_{n}", protos[proto])
        bpy.context.scene.collection.objects.link(ob)
        ob.location = (p.x, p.y, CURB_H)
        ob.rotation_euler = (0.0, 0.0, p.yaw - math.pi / 2)  # the prototypes' arms reach +y
        if p.kind == "tree":
            ob.scale = (0.85 + 0.1 * ((n * 7) % 5),) * 3


def build(m, asphalt: bpy.types.Material, mats: facades.Materials) -> layout.Scenery:
    """Draw the scenery of an outdoor map; returns the layout it drew."""
    lay = layout.layout(m)
    _ground(lay, asphalt)
    _buildings(lay, mats)
    _furniture(lay)
    return lay
