# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Mesh plumbing for the city's static structure (facades, scenery): a mesh builder with named material slots, flat
quads, boxes, rounded outlines and thin rods.

Runs inside Blender's own Python and is never imported by `stbench`. Map frame as in `build_scene`: metres, x east, y
north, z up. Every operator here is called on its own new geometry, never on the whole bmesh: bmesh operators that walk
the merged mesh (moving a part by transforming every vertex) are quadratic in a large mesh.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector


class MeshBuilder:
    """One bmesh with a named material slot per `slot(name, material)`; `finish` makes the object."""

    def __init__(self, name: str):
        self.name, self.bm = name, bmesh.new()
        self.slots: dict[str, int] = {}
        self.mats: list[bpy.types.Material] = []

    def slot(self, key: str, material: bpy.types.Material) -> int:
        if key not in self.slots:
            self.slots[key] = len(self.mats)
            self.mats.append(material)
        return self.slots[key]

    def mesh(self) -> bpy.types.Mesh:
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
        me = bpy.data.meshes.new(self.name)
        self.bm.to_mesh(me)
        self.bm.free()
        for m in self.mats:  # every slot exists before anything copies the mesh, or Blender resets the slot indices
            me.materials.append(m)
        return me

    def finish(self, pass_index: int = 0, location=(0.0, 0.0, 0.0), yaw: float = 0.0) -> bpy.types.Object:
        ob = bpy.data.objects.new(self.name, self.mesh())
        ob.pass_index = pass_index
        ob.location = location
        ob.rotation_euler = (0.0, 0.0, yaw)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def rect(bm, x0, x1, y0, y1, z, slot: int) -> None:
    """A flat quad facing up."""
    f = bm.faces.new([bm.verts.new(p) for p in ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))])
    f.material_index = slot


def cube(bm, x0, x1, y0, y1, z0, z1, slot: int) -> list:
    m = Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)) @ Matrix.Diagonal(
        (x1 - x0, y1 - y0, z1 - z0, 1))
    res = bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    faces = list({f for v in res["verts"] for f in v.link_faces})
    for f in faces:
        f.material_index = slot
    return faces


def rounded_rect(x0, x1, y0, y1, r: float, n: int = 5) -> list[tuple[float, float]]:
    """Counter-clockwise outline of a rectangle with every corner rounded by radius `r`."""
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90.0 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def outline_prism(bm, outline, z0: float, z1: float, top_slot: int, side_slot: int) -> None:
    """A closed solid: `outline` (counter-clockwise) extruded from z0 to z1; the top and sides take their own slots."""
    top = bm.faces.new([bm.verts.new((x, y, z1)) for x, y in outline])
    top.material_index = top_slot
    n = len(outline)
    lo = [bm.verts.new((x, y, z0)) for x, y in outline]
    hi = [bm.verts.new((x, y, z1)) for x, y in outline]
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new([lo[i], lo[j], hi[j], hi[i]]).material_index = side_slot


def rod(bm, a, b, r: float, slot: int, segments: int = 8) -> None:
    """A closed cylinder from point `a` to point `b`, made in one operator call."""
    a, b = Vector(a), Vector(b)
    d = b - a
    m = Matrix.Translation((a + b) / 2) @ Vector((0, 0, 1)).rotation_difference(d).to_matrix().to_4x4()
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=r, radius2=r, depth=d.length, matrix=m)
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = slot
