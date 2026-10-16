# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Building facades: a map wall with a `facade` style is drawn as that building's face (window grid or curtain wall,
a storefront band, a string course over it and a cornice and parapet at the top), in the wall's own frame. A lot deep
enough to have a roof (the scenery's buildings: the map's own walls are slabs 0.6 m thick, a face and no more) gets a
real one: a parapet round a tar roof with air-conditioning units, vents, a hatch and, on a large roof, a stair
bulkhead. No camera of the downtown sees a roof (the drone flies at 12 m, the lowest building is 18.9 m), but a higher
one would, so the roofs are drawn: seeded from the building's name, a few dozen faces a building.

Runs inside Blender's own Python and is never imported by `stbench`. Each wall is one object whose origin is its `a` end
and whose x axis runs along the wall, so the shader's bays start at the wall's end whatever way it runs; the mesh holds
the wall's slab (as `build_scene._wall_boxes` draws a plain wall: thickness `t`, `t / 2` past each end) and its trim.
Walls carry `pass_index` 0 like the rest of the structure. `box_mesh` is the same building, for any lot, so the scenery
beyond the map's edge looks like the map.
"""

from __future__ import annotations

import hashlib
import random

import bpy
import facade_shaders
from materials_street import lin, plain
from scene_mesh import MeshBuilder, cube

CORNICE_H, CORNICE_OUT, PARAPET_H = 0.9, 0.5, 0.9
PARAPET_T, ROOF_DEPTH_M = 0.3, 3.0  # a parapet wall's thickness; a lot this deep and wide has a roof to speak of
EDGE_M = 1.8  # roof equipment stands this far in from the parapet
COURSE_Z, COURSE_OUT = facade_shaders.GROUND_H, 0.1  # the string course over the storefront band


class Materials:
    """The facade materials of one scene, each built once, and the trim and roof they share."""

    def __init__(self):
        self.styles: dict[str, bpy.types.Material] = {}
        self.trim = plain("trim_stone", lin((0.74, 0.72, 0.66)), 0.7)
        self.roof = plain("roof_tar", (0.07, 0.07, 0.075), 0.9)
        self.unit = plain("roof_unit", lin((0.40, 0.41, 0.43)), 0.5)
        self.dark = plain("roof_dark", lin((0.10, 0.10, 0.11)), 0.6)

    def facade(self, style: str) -> bpy.types.Material:
        if style not in self.styles:
            self.styles[style] = facade_shaders.material(style)
        return self.styles[style]


def add_building(mb: MeshBuilder, mats: Materials, style: str, x0: float, x1: float, y0: float, y1: float,
                 height: float) -> None:
    """The box [x0, x1] x [y0, y1] x [0, height] in its object's frame, with its string course, cornice and parapet. The
    street face is the one whose normal is -y or +y (x runs along it)."""
    fa, tr, rf = mb.slot("facade", mats.facade(style)), mb.slot("trim", mats.trim), mb.slot("roof", mats.roof)
    for f in cube(mb.bm, x0, x1, y0, y1, 0.0, height, fa):
        if f.calc_center_median().z > height - 1e-3:
            f.material_index = rf
    o = COURSE_OUT
    cube(mb.bm, x0 - o, x1 + o, y0 - o, y1 + o, COURSE_Z - 0.1, COURSE_Z + 0.1, tr)
    c, z0 = CORNICE_OUT, height - CORNICE_H
    if min(x1 - x0, y1 - y0) < ROOF_DEPTH_M:  # a face: its cornice and parapet are blocks along its top
        cube(mb.bm, x0 - 0.15, x1 + 0.15, y0 - c, y1 + c, z0, height, tr)
        cube(mb.bm, x0, x1, y0, y1, height, height + PARAPET_H, tr)
        return
    t = PARAPET_T  # a roof: a cornice and a parapet wall round its edge, and what stands on it
    for a, b, c0, c1 in ((x0 - 0.15, x1 + 0.15, y0 - c, y0 + t), (x0 - 0.15, x1 + 0.15, y1 - t, y1 + c),
                         (x0 - 0.15, x0 + t, y0 + t, y1 - t), (x1 - t, x1 + 0.15, y0 + t, y1 - t)):
        cube(mb.bm, a, b, c0, c1, z0, height, tr)
    for a, b, c0, c1 in ((x0, x1, y0, y0 + t), (x0, x1, y1 - t, y1), (x0, x0 + t, y0 + t, y1 - t),
                         (x1 - t, x1, y0 + t, y1 - t)):
        cube(mb.bm, a, b, c0, c1, height, height + PARAPET_H, tr)
    _roof(mb, mats, x0, x1, y0, y1, height)


def _roof(mb: MeshBuilder, mats: Materials, x0, x1, y0, y1, height: float) -> None:
    """Units, vents, a hatch and, on a large roof, a stair bulkhead, seeded from the object's name. Each stands on the
    roof (`height`), `EDGE_M` in from the parapet and clear of what was placed before it."""
    rng = random.Random(int.from_bytes(hashlib.sha256(mb.name.encode()).digest()[:8], "big"))
    unit, dark, tr = mb.slot("unit", mats.unit), mb.slot("dark", mats.dark), mb.slot("trim", mats.trim)
    taken: list[tuple[float, float, float, float]] = []
    wide, deep = x1 - x0 - 2 * EDGE_M, y1 - y0 - 2 * EDGE_M
    if wide < 2.0 or deep < 2.0:
        return

    def put(w: float, d: float, h: float, slot: int, top: float = 0.0) -> bool:
        for _ in range(12):
            cx, cy = x0 + EDGE_M + w / 2 + rng.random() * (wide - w), y0 + EDGE_M + d / 2 + rng.random() * (deep - d)
            box = (cx - w / 2 - 0.6, cx + w / 2 + 0.6, cy - d / 2 - 0.6, cy + d / 2 + 0.6)
            if any(box[0] < t[1] and t[0] < box[1] and box[2] < t[3] and t[2] < box[3] for t in taken):
                continue
            taken.append(box)
            cube(mb.bm, cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2, height, height + h, slot)
            if top:  # a fan housing: a dark square on top
                cube(mb.bm, cx - w / 3, cx + w / 3, cy - d / 3, cy + d / 3, height + h, height + h + top, dark)
            return True
        return False

    if wide * deep > 400.0:
        put(3.5, 4.0, 3.0, tr)  # a stair or lift bulkhead
    for _ in range(rng.randint(2, 6)):
        put(rng.uniform(1.4, 2.6), rng.uniform(1.0, 1.8), rng.uniform(0.9, 1.4), unit, 0.06)  # condensers
    for _ in range(rng.randint(2, 5)):
        put(0.5, 0.5, rng.uniform(0.7, 1.4), unit)  # vent stacks
    put(1.0, 1.2, 0.45, unit)  # a roof hatch


def wall_object(w, mats: Materials) -> bpy.types.Object:
    """One wall of the map as its building's facade."""
    mb = MeshBuilder(f"facade_{w.id}")
    t = w.thickness
    add_building(mb, mats, w.facade, -t / 2, w.length + t / 2, -t / 2, t / 2, w.height)
    return mb.finish(0, (w.a[0], w.a[1], 0.0), w.yaw)


def build(m, mats: Materials) -> list[bpy.types.Object]:
    """An object for every wall of the map that names a facade style (none, for the indoor maps)."""
    return [wall_object(w, mats) for w in m.walls if w.facade]
