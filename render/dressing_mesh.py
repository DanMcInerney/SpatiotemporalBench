# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The kit a dressing builder makes its piece with: one `bmesh`, a material slot per look, solids with bevels of a few
millimetres, and one finished object (`dressing.py` places it).

Runs inside Blender's own Python and is never imported by `stbench`. A piece is built in its Dressing's own frame: the
origin is the centre of its base, x faces out of the wall (the way the piece looks), y runs across it and z is up, so
the box `Dressing.size = (depth, width, height)` spans x in +-depth/2, y in +-width/2 and z in 0..height; the wall is at
x = -depth/2. Every solid is closed and its edges are geometric bevels, never the Bevel shader node (it costs a 110 s
kernel compile). `finish` marks the sharp edges by angle and gives the surface smooth
shading everywhere else, so a bevel catches light as one rounded edge and a cabinet door stays flat.
"""

from __future__ import annotations

import math

import bmesh
import bpy
import meshkit
from mathutils import Matrix, Vector

SHARP_DEG = 50.0  # an edge between faces more than this far from flat is a crease (a bevel step of 45 degrees is not)
THIN_M = 1e-4  # a slab thinner than this on any side is not built (a bevel on it would collapse)
AXES = {"z": Matrix.Identity(4), "x": Matrix.Rotation(math.pi / 2, 4, "Y"), "y": Matrix.Rotation(-math.pi / 2, 4, "X")}


class Build:
    """One piece in the making. `box`, `prism`, `revolve`, `rod`, `frustum` and `sheet` add closed solids in a named
    material; `finish` returns the object (unlinked, identity transform, no pass index)."""

    def __init__(self, ctx, name: str):
        self.ctx, self.name = ctx, name
        self.bm = bmesh.new()
        self._slots: list[str] = []

    def slot(self, material: str) -> int:
        if material not in self._slots:
            self._slots.append(material)
        return self._slots.index(material)

    # -- solids ---------------------------------------------------------------------------------------------------
    def box(self, x0, x1, y0, y1, z0, z1, material: str, bevel: float = 0.0, segments: int = 2) -> None:
        """A slab, its edges rounded by `bevel` (a few millimetres)."""
        if min(x1 - x0, y1 - y0, z1 - z0) < THIN_M:
            return
        meshkit.solid(self.bm, x0, x1, y0, y1, z0, z1, self.slot(material), bevel=bevel, segments=segments)

    def prism(self, outline: list[tuple[float, float]], y0: float, y1: float, material: str) -> None:
        """The (x, z) polygon `outline` extruded along y from y0 to y1 (a moulding's profile)."""
        meshkit.prism(self.bm, outline, y0, y1, self.slot(material))

    def revolve(self, profile: list[tuple[float, float]], material: str, at=(0.0, 0.0, 0.0), axis: str = "z",
                segments: int = 32, loop: bool = False) -> None:
        """Revolve a (radius, height) profile about the axis `axis` through `at`, the profile's height running along it
        (a lamp's shade, a roll of paper, a rim). With `loop` the last point joins the first: a closed outline."""
        slot = self.slot(material)
        meshkit.placed(self.bm, Matrix.Translation(at) @ AXES[axis],
                       lambda: meshkit.lathe(self.bm, profile, slot, segments=segments, loop=loop))

    def rod(self, a, b, radius: float, material: str, segments: int = 14) -> None:
        """A closed cylinder from point `a` to point `b`."""
        meshkit.rod(self.bm, a, b, radius, self.slot(material), segments=segments)

    def ring(self, centre, big: float, small: float, axis: str, material: str, major: int = 24, minor: int = 8) -> None:
        """A torus of centre-line radius `big` and tube radius `small` about the axis `axis` through `centre` (a curtain
        ring, a hook)."""
        slot = self.slot(material)
        n0 = len(self.bm.faces)
        rows = []
        for i in range(major):
            a = 2 * math.pi * i / major
            row = []
            for j in range(minor):
                b = 2 * math.pi * j / minor
                r = big + small * math.cos(b)
                row.append(self.bm.verts.new((r * math.cos(a), r * math.sin(a), small * math.sin(b))))
            rows.append(row)
        for i in range(major):
            for j in range(minor):
                p, q = rows[i], rows[(i + 1) % major]
                self.bm.faces.new((p[j], p[(j + 1) % minor], q[(j + 1) % minor], q[j]))
        verts = [v for row in rows for v in row]
        bmesh.ops.transform(self.bm, matrix=Matrix.Translation(centre) @ AXES[axis], verts=verts)
        self.bm.faces.ensure_lookup_table()
        for f in self.bm.faces[n0:]:
            f.material_index = slot

    def frustum(self, bottom, top, z0: float, z1: float, material: str, bevel: float = 0.0) -> None:
        """A closed solid between two rectangles (x0, x1, y0, y1): `bottom` at height z0 and `top` at z1."""
        slot = self.slot(material)
        lo = [self.bm.verts.new((x, y, z0)) for x, y in ((bottom[0], bottom[2]), (bottom[1], bottom[2]),
                                                          (bottom[1], bottom[3]), (bottom[0], bottom[3]))]
        hi = [self.bm.verts.new((x, y, z1)) for x, y in ((top[0], top[2]), (top[1], top[2]), (top[1], top[3]),
                                                          (top[0], top[3]))]
        faces = [self.bm.faces.new(lo[::-1]), self.bm.faces.new(hi)]
        faces += [self.bm.faces.new((lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i])) for i in range(4)]
        for f in faces:
            f.material_index = slot
        if bevel > 0:
            edges = list({e for f in faces for e in f.edges})
            res = bmesh.ops.bevel(self.bm, geom=edges, offset=bevel, segments=2, profile=0.5, affect="EDGES")
            for f in res["faces"]:
                f.material_index, f.smooth = slot, True

    def sheet(self, nu: int, nv: int, point, thickness: float, material: str) -> None:
        """A cloth as a closed solid: the grid of `point(u, v) -> (x, y, z)` over u, v in 0..1 (nu x nv quads), with a
        second surface `thickness` further along x and the edges closed, smooth shaded."""
        slot = self.slot(material)
        front = [[self.bm.verts.new(point(i / nu, j / nv)) for j in range(nv + 1)] for i in range(nu + 1)]
        back = [[self.bm.verts.new((v.co.x + thickness, v.co.y, v.co.z)) for v in row] for row in front]
        quads = []
        for i in range(nu):
            for j in range(nv):
                quads.append((front[i][j], front[i + 1][j], front[i + 1][j + 1], front[i][j + 1]))
                quads.append((back[i][j + 1], back[i + 1][j + 1], back[i + 1][j], back[i][j]))
        for i in range(nu):
            quads.append((front[i][0], back[i][0], back[i + 1][0], front[i + 1][0]))
            quads.append((front[i][nv], front[i + 1][nv], back[i + 1][nv], back[i][nv]))
        for j in range(nv):
            quads.append((front[0][j], front[0][j + 1], back[0][j + 1], back[0][j]))
            quads.append((front[nu][j], back[nu][j], back[nu][j + 1], front[nu][j + 1]))
        for q in quads:
            f = self.bm.faces.new(q)
            f.material_index, f.smooth = slot, True

    # -- the object -----------------------------------------------------------------------------------------------
    def finish(self) -> bpy.types.Object:
        """The piece as an object with its materials (looked up from the context), unlinked."""
        bm = self.bm
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        limit = math.radians(SHARP_DEG)
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.link_faces[0].normal.angle(e.link_faces[1].normal, 0.0) > limit:
                e.smooth = False
        for f in bm.faces:
            f.smooth = True
        me = bpy.data.meshes.new(self.name)
        bm.to_mesh(me)
        bm.free()
        for key in self._slots:
            me.materials.append(self.ctx.material(key))
        return bpy.data.objects.new(self.name, me)


def placed_at(bm_build: Build, matrix: Matrix, draw) -> None:
    """Call `draw()`, which adds solids to the build, then move everything it added by `matrix` (a part set at an angle:
    a fan blade, a bracket)."""
    meshkit.placed(bm_build.bm, matrix, draw)


def turn(angle_deg: float, axis: str = "z", at=(0.0, 0.0, 0.0)) -> Matrix:
    """A rotation of `angle_deg` about the axis `axis` through the point `at`."""
    return Matrix.Translation(at) @ Matrix.Rotation(math.radians(angle_deg), 4, axis.upper()) @ Matrix.Translation(
        Vector(at) * -1.0)
