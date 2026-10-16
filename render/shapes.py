# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Primitive parts, a palette of Principled materials, and one Blender mesh per entity.

Runs inside Blender's own Python and is never imported by `stbench`. An entity is a list of `Part`s in its local frame
(origin at the base centre, x forward, y left, z up, metres). `entity_mesh` merges them into one mesh with one material
slot per distinct palette key, so the entity is one object and one `pass_index`. M1 geometry is placeholder: no
textures except on floors, no clear glass and no mirrors (the ID pass and RGB would disagree there).

A box part may be bevelled (`Part.bevel`, metres: 3 to 8 mm rounds a hard edge, a few centimetres a cushion). The bevel
only takes material off the corners, so a part keeps its bounding box and every flat face keeps its plane (a surface the
map puts items on stays at the height the map gives).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import bmesh
import bpy
from mathutils import Euler, Matrix

# -- palette -----------------------------------------------------------------------------------------------------
# sRGB colour, roughness, metallic. Keys are attribute colours or fixed names; "name*f" scales a key's brightness.
# "black" is an albedo of about 0.035 in linear light, as real black plastic and ceramic are; at 0.0035 (sRGB 0.045)
# a black item rendered as a pure-black silhouette with no visible shading (the data-cal contact sheet).
PALETTE: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "red": ((0.80, 0.07, 0.07), 0.45, 0.0), "blue": ((0.08, 0.25, 0.80), 0.45, 0.0),
    "green": ((0.10, 0.58, 0.16), 0.45, 0.0), "yellow": ((0.95, 0.80, 0.08), 0.45, 0.0),
    "white": ((0.90, 0.90, 0.88), 0.5, 0.0), "black": ((0.20, 0.20, 0.21), 0.4, 0.0),
    "orange": ((0.95, 0.42, 0.04), 0.45, 0.0), "purple": ((0.50, 0.14, 0.66), 0.45, 0.0),
    "brown": ((0.36, 0.20, 0.09), 0.55, 0.0), "grey": ((0.45, 0.45, 0.47), 0.55, 0.0),
    "pink": ((0.95, 0.48, 0.70), 0.45, 0.0), "clear": ((0.72, 0.88, 0.93), 0.15, 0.0),
    "beige": ((0.80, 0.72, 0.55), 0.7, 0.0), "oak": ((0.62, 0.43, 0.22), 0.5, 0.0),
    "steel": ((0.72, 0.72, 0.74), 0.3, 0.9), "dark": ((0.07, 0.07, 0.08), 0.5, 0.0),
    "stone": ((0.72, 0.72, 0.70), 0.35, 0.0), "foliage": ((0.10, 0.45, 0.12), 0.7, 0.0),
    "soil": ((0.12, 0.08, 0.05), 0.9, 0.0), "wall": ((0.88, 0.86, 0.81), 0.9, 0.0),
    "ceiling": ((0.95, 0.95, 0.94), 0.9, 0.0), "door": ((0.84, 0.80, 0.72), 0.5, 0.0),
    "door_entrance": ((0.30, 0.17, 0.09), 0.5, 0.0), "frame": ((0.92, 0.92, 0.90), 0.5, 0.0),
    "ground": ((0.25, 0.33, 0.18), 1.0, 0.0), "underlay": ((0.55, 0.50, 0.45), 0.8, 0.0),
    "paper": ((0.93, 0.91, 0.84), 0.9, 0.0), "fixture": ((0.97, 0.97, 0.95), 0.5, 0.0),
    # what lies under furniture, seen from 2 cm below (`wear_surfaces.raw`, `cover`): raw board, black non-woven cloth
    "raw": ((0.56, 0.47, 0.35), 0.9, 0.0), "cover": ((0.20, 0.20, 0.21), 0.95, 0.0),
}
_cache: dict[str, bpy.types.Material] = {}


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def reset() -> None:
    """Forget the cached palette materials (they are freed when a scene is rebuilt)."""
    _cache.clear()


def material(key: str, resolve=None) -> bpy.types.Material:
    """The Principled material for a palette key ("red", "oak*0.8"). `resolve` (assets mode) maps a key to a textured
    material, or to None where the plain palette material stays."""
    if resolve is not None and (textured := resolve(key)) is not None:
        return textured
    if key in _cache:
        return _cache[key]
    name, _, factor = key.partition("*")
    if name not in PALETTE:
        raise KeyError(f"unknown palette colour {name!r}; known: {sorted(PALETTE)}")
    rgb, rough, metal = PALETTE[name]
    f = float(factor) if factor else 1.0
    lin = tuple(min(1.0, srgb_to_linear(min(1.0, c * f))) for c in rgb)
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    p = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (*lin, 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    _cache[key] = m
    return m


def emission_material(key: str, strength: float) -> bpy.types.Material:
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    p = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    p.inputs["Emission Color"].default_value = (1.0, 0.96, 0.88, 1.0)
    p.inputs["Emission Strength"].default_value = strength
    p.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    return m


# -- parts -------------------------------------------------------------------------------------------------------
class Part(NamedTuple):
    shape: str  # box | cyl | sphere | torus | wedge
    loc: tuple[float, float, float]  # centre
    dims: tuple[float, float, float]  # full extents (torus: major radius, minor radius, unused)
    rot: tuple[float, float, float]  # euler XYZ, radians
    mat: str
    extra: tuple[float, ...] = ()  # cyl: (top radius / bottom radius,); wedge: (height at +x / height at -x,)
    bevel: float = 0.0  # box: the width of the rounding of its edges, metres (cut back to 45% of its thinnest side)
    bevel_segments: int = 2


def box(x0, x1, y0, y1, z0, z1, mat, bevel: float = 0.0, segments: int = 2) -> Part:
    return Part("box", ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), (0, 0, 0), mat, (),
                bevel, segments)


def cyl(cx, cy, z0, z1, r, mat, top=1.0) -> Part:
    """Upright cylinder or frustum (radius `r` at the bottom, `r * top` at the top)."""
    return Part("cyl", (cx, cy, (z0 + z1) / 2), (2 * r, 2 * r, z1 - z0), (0, 0, 0), mat, (top,))


def ball(cx, cy, cz, rx, ry, rz, mat) -> Part:
    return Part("sphere", (cx, cy, cz), (2 * rx, 2 * ry, 2 * rz), (0, 0, 0), mat)


def ring(cx, cy, cz, big, small, axis, mat) -> Part:
    """Torus of major radius `big` and tube radius `small` around the given axis ("x", "y" or "z")."""
    rot = {"z": (0, 0, 0), "y": (math.pi / 2, 0, 0), "x": (0, math.pi / 2, 0)}[axis]
    return Part("torus", (cx, cy, cz), (big, small, 0), rot, mat)


def wedge(x0, x1, y0, y1, z0, h_back, h_front, mat) -> Part:
    """Box whose top slopes linearly from height `h_back` at x0 to `h_front` at x1."""
    return Part("wedge", ((x0 + x1) / 2, (y0 + y1) / 2, z0), (x1 - x0, y1 - y0, max(h_back, h_front)), (0, 0, 0), mat,
                (h_back, h_front))


def _torus(bm, big, small, major=28, minor=10):
    rows = []
    for i in range(major):
        a = 2 * math.pi * i / major
        ring_ = []
        for j in range(minor):
            b = 2 * math.pi * j / minor
            r = big + small * math.cos(b)
            ring_.append(bm.verts.new((r * math.cos(a), r * math.sin(a), small * math.sin(b))))
        rows.append(ring_)
    for i in range(major):
        for j in range(minor):
            a, b = rows[i], rows[(i + 1) % major]
            f = bm.faces.new((a[j], a[(j + 1) % minor], b[(j + 1) % minor], b[j]))
            f.smooth = True
    return [v for r in rows for v in r]


def _wedge(bm, dims, extra):
    sx, sy, _ = dims
    h0, h1 = extra
    pts = [(-sx / 2, -sy / 2, 0), (sx / 2, -sy / 2, 0), (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, 0),
           (-sx / 2, -sy / 2, h0), (sx / 2, -sy / 2, h1), (sx / 2, sy / 2, h1), (-sx / 2, sy / 2, h0)]
    v = [bm.verts.new(p) for p in pts]
    for idx in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in idx])
    return v


def _bevel_box(bm: bmesh.types.BMesh, verts: list, width: float, segments: int, slot: int) -> None:
    """Round every edge of the box whose corners are `verts` (already placed), as `meshkit.slab` does. The box's faces
    are on `slot` already, and the faces the bevel makes take it as well."""
    edges = list({e for v in verts for e in v.link_edges})
    res = bmesh.ops.bevel(bm, geom=verts + edges, offset=width, segments=segments, profile=0.5, affect="EDGES")
    for f in res["faces"]:
        f.material_index, f.smooth = slot, True  # the strips of the rounding; the flat faces stay flat shaded


def _add_part(bm: bmesh.types.BMesh, part: Part, slot: int, exact: bool = False) -> None:
    """Add one part to `bm`, its faces on material slot `slot`. A part with no bevel is the last faces of `bm` (the
    caller adds the bevelled ones after all the others: a bevel frees faces, and the next part's faces would then fill
    the gaps); a bevelled box finds its faces from its own vertices. `exact` finds the faces of every part from its own
    vertices: a cone or a sphere frees a few faces as it is made, and the next part's faces fill those gaps, which lie
    before the last faces of `bm`, so a part after a cylinder could keep some of its faces on slot 0 (a table top's
    finish on a drawer's bottom)."""
    n0 = len(bm.faces)
    sx, sy, sz = part.dims
    if part.shape == "torus":
        verts = _torus(bm, sx, sy)
        bmesh.ops.transform(bm, matrix=Matrix.Translation(part.loc) @ Euler(part.rot).to_matrix().to_4x4(), verts=verts)
    elif part.shape == "wedge":
        verts = _wedge(bm, part.dims, part.extra)
        bmesh.ops.transform(bm, matrix=Matrix.Translation(part.loc), verts=verts)
    else:
        m = Matrix.Translation(part.loc) @ Euler(part.rot).to_matrix().to_4x4() @ Matrix.Diagonal((sx, sy, sz, 1.0))
        if part.shape == "box":
            verts = bmesh.ops.create_cube(bm, size=1.0, matrix=m)["verts"]
        elif part.shape == "cyl":
            verts = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=24, radius1=0.5,
                                          radius2=0.5 * part.extra[0], depth=1.0, matrix=m)["verts"]
        elif part.shape == "sphere":
            verts = bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=0.5, matrix=m)["verts"]
        else:
            raise ValueError(f"unknown part shape {part.shape!r}")
        if part.shape == "box" and part.bevel > 0.0:
            for f in {f for v in verts for f in v.link_faces}:
                f.material_index = slot
            _bevel_box(bm, verts, min(part.bevel, 0.45 * min(sx, sy, sz)), part.bevel_segments, slot)
            return
    if exact:
        faces = list({f for v in verts for f in v.link_faces})
    else:
        bm.faces.ensure_lookup_table()
        faces = bm.faces[n0:]
    for f in faces:
        f.material_index = slot
        if part.shape == "sphere":
            f.smooth = True


def entity_mesh(name: str, parts: list[Part], resolve=None, exact: bool = False) -> bpy.types.Mesh:
    """One mesh for all the parts, with a material slot per distinct palette key (see `material` for `resolve`; see
    `_add_part` for `exact`: the household's furniture asks for it, the look of the others is as it was)."""
    keys = list(dict.fromkeys(p.mat for p in parts))
    bm = bmesh.new()
    for p in sorted(parts, key=lambda q: q.bevel > 0.0):  # the bevelled boxes last (see `_add_part`)
        _add_part(bm, p, keys.index(p.mat), exact)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for k in keys:
        me.materials.append(material(k, resolve))
    return me


def polygon_mesh(name: str, polygon: list[tuple[float, float]], z: float, mat: bpy.types.Material) -> bpy.types.Mesh:
    bm = bmesh.new()
    bm.faces.new([bm.verts.new((x, y, z)) for x, y in polygon])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    return me


def link(name: str, me: bpy.types.Mesh, pass_index: int = 0) -> bpy.types.Object:
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.pass_index = pass_index
    return ob
