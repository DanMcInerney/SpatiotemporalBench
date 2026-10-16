# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What the warehouse's dressing builders are made with: `dressing_mesh.Build` (a mesh in a Dressing's own frame, one
material slot per named material) plus the pieces only the warehouse draws: capped tubes and extruded sections, readable
text and barcodes as real geometry, and the check that a piece stays in its box.

Runs inside Blender's own Python and is never imported by `stbench`. A builder (`dressing_build_warehouse.BUILDERS`) is
`fn(d, ctx, text) -> [object]` and draws in the Dressing's frame: origin at the centre of the base, x out of the wall or
rack face it hangs on (the size's depth), y to its left (the width), z up. `dressing.build` places the object and gives
it `pass_index` 0. `Mesh.finish` refuses geometry that leaves the Dressing's box by more than `TOLERANCE_M`, because
the validity guard (`src/stbench/dressing_zones.py`) holds the box clear of everything a question or the robot depends
on.

Materials are lane M's names (`materials_dressing.NAMES`), asked of `ctx.material`. The constants below are the ones
the warehouse uses.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from dressing_code39 import code39_bars
from dressing_mesh import Build
from dressing_spec import Dressing
from mathutils import Matrix, Vector

TOLERANCE_M = 0.005  # how far built geometry may leave its Dressing's box
GREY, WHITE, GALV, STEEL = "painted_metal_grey", "painted_metal_white", "galvanised_steel", "brushed_steel"
RED, YELLOW, RUBBER, PAPER = "painted_metal_red", "safety_yellow", "rubber_black", "label_paper"
INK, BARS, PLASTIC, CONCRETE = "rubber_black", "black_plastic", "white_plastic", "concrete_wall"
CLADDING, TREAD, PANE, LENS_HIGH, LENS_LOW = "corrugated_steel", "diamond_plate", "skylight_panel", \
    "led_emitter_cool", "led_emitter"

_GLYPHS: dict[str, tuple[list[tuple[float, float]], list[tuple[int, ...]]]] = {}


def glyphs(string: str, bold: float = 0.012) -> tuple[list[tuple[float, float]], list[tuple[int, ...]]]:
    """The flat outline faces of a string in Blender's built-in font, 1 em high, centred on the origin: (points as (u,
    v), faces as point indices). `bold` thickens every stroke by that much (in em): small print stays legible."""
    key = f"{bold}:{string}"
    if key not in _GLYPHS:
        cu = bpy.data.curves.new("stbench_text", "FONT")
        cu.body, cu.size, cu.offset = string, 1.0, bold
        cu.align_x, cu.align_y = "CENTER", "CENTER"
        ob = bpy.data.objects.new("stbench_text", cu)
        bpy.context.scene.collection.objects.link(ob)
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        _GLYPHS[key] = ([(v.co.x, v.co.y) for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        bpy.data.objects.remove(ob)
        bpy.data.curves.remove(cu)
        bpy.data.meshes.remove(me)
    return _GLYPHS[key]


def clip_u(poly: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    """A polygon of (u, v) points cut to lo <= u <= hi (Sutherland-Hodgman against the two vertical lines)."""
    for edge, keep in ((lo, lambda u: u >= lo), (hi, lambda u: u <= hi)):
        out = []
        for a, b in zip(poly, poly[1:] + poly[:1], strict=False):
            if keep(a[0]):
                out.append(a)
            if keep(a[0]) != keep(b[0]):
                out.append((edge, a[1] + (edge - a[0]) / (b[0] - a[0]) * (b[1] - a[1])))
        poly = out
        if not poly:
            break
    return poly


class Mesh(Build):
    """One dressing object's mesh in the Dressing's frame (see the module docstring)."""

    def __init__(self, d: Dressing, ctx):
        super().__init__(ctx, d.cls)
        self.d = d
        self.text: str | None = None

    def cylinder(self, axis: str, centre, r: float, length: float, material: str, segments: int = 24) -> None:
        """A capped cylinder about an axis (x, y or z) through `centre`."""
        h = length / 2
        self.revolve([(0.0, -h), (r, -h), (r, h), (0.0, h)], material, centre, axis, segments)

    def band(self, axis: str, centre, r0: float, r1: float, length: float, material: str, segments: int = 24) -> None:
        """A collar from radius r0 (flush with a pipe) out to r1, `length` along the axis (open at the inside)."""
        h = length / 2
        self.revolve([(r0, -h), (r1, -h), (r1, h), (r0, h)], material, centre, axis, segments)

    def extrude(self, outline, axis: str, t0: float, t1: float, material: str) -> None:
        """A closed solid: the polygon `outline` of (u, v) points extruded from t0 to t1 along x (u is y, v is z) or
        along y (u is x, v is z)."""
        place = (lambda u, v, t: (t, u, v)) if axis == "x" else (lambda u, v, t: (u, t, v))
        slot = self.slot(material)
        lo = [self.bm.verts.new(place(u, v, t0)) for u, v in outline]
        hi = [self.bm.verts.new(place(u, v, t1)) for u, v in outline]
        for ring in (lo[::-1], hi):
            self.bm.faces.new(ring).material_index = slot
        for i in range(len(outline)):
            j = (i + 1) % len(outline)
            self.bm.faces.new([lo[i], lo[j], hi[j], hi[i]]).material_index = slot

    def tube(self, points, r: float, material: str, segments: int = 8) -> None:
        """A capped tube of radius `r` along a polyline of points (cables, arms)."""
        pts = [Vector(p) for p in points]
        slot, rings = self.slot(material), []
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
            u = t.cross(Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))).normalized()
            v = t.cross(u)
            rings.append([self.bm.verts.new(p + r * (math.cos(a) * u + math.sin(a) * v))
                          for a in (2 * math.pi * k / segments for k in range(segments))])
        for a, b in zip(rings, rings[1:], strict=False):
            for k in range(segments):
                self.bm.faces.new([a[k], a[(k + 1) % segments], b[(k + 1) % segments], b[k]]).material_index = slot
        for ring in (rings[0][::-1], rings[-1]):
            self.bm.faces.new(ring).material_index = slot

    def turned(self, angle_deg: float, axis: str = "z", at=(0.0, 0.0, 0.0)):
        """A context manager: whatever is added inside is turned `angle_deg` about an axis through `at` afterwards."""
        return _Turned(self, Matrix.Translation(at) @ Matrix.Rotation(math.radians(angle_deg), 4, axis.upper())
                       @ Matrix.Translation(-Vector(at)))

    # -- print -----------------------------------------------------------------------------------------------------
    def print_text(self, string: str, x: float, y0: float, y1: float, z0: float, z1: float, material: str = INK,
                   bold: float = 0.012, wide: float = 1.0) -> tuple[float, float]:
        """`string` as flat geometry on the plane x, facing +x, as large as fits the rectangle y0..y1 by z0..z1
        (centred), its letters up to `wide` times broader than the font draws them where the rectangle allows (small
        print on a label reads better extended); returns the (width, height) it takes. Dark print on a light plate is
        the caller's."""
        pts, faces = glyphs(string, bold)
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        w, h = max(us) - min(us), max(vs) - min(vs)
        k = min((y1 - y0) / (w * wide), (z1 - z0) / h)
        kx = k * wide
        cy, cz = (y0 + y1) / 2 - kx * (max(us) + min(us)) / 2, (z0 + z1) / 2 - k * (max(vs) + min(vs)) / 2
        verts = [self.bm.verts.new((x, cy + kx * u, cz + k * v)) for u, v in pts]
        slot = self.slot(material)
        for f in faces:
            self.bm.faces.new([verts[i] for i in f]).material_index = slot
        return kx * w, k * h

    def barcode(self, string: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
                material: str = BARS) -> None:
        """A Code 39 barcode of `string` as thin bars between y0 and y1, z0 and z1, from x0 to x1."""
        bars = code39_bars(string)
        total = bars[-1][0] + bars[-1][1]
        for start, width in bars:
            self.box(x0, x1, y0 + (y1 - y0) * start / total, y0 + (y1 - y0) * (start + width) / total, z0, z1, material)

    # -- the object ------------------------------------------------------------------------------------------------
    def finish(self) -> bpy.types.Object:
        """The object (unlinked, identity transform; `dressing.build` places it and sets its `pass_index` 0). Raises
        ValueError when any of the mesh is outside the Dressing's box by more than `TOLERANCE_M`. The piece's class,
        recipe and printed text are custom properties of the object and of its mesh (identical pieces share a mesh)."""
        d = self.d
        lo = (-d.size[0] / 2, -d.size[1] / 2, 0.0)
        hi = (d.size[0] / 2, d.size[1] / 2, d.size[2])
        for v in self.bm.verts:
            for i in range(3):
                if not lo[i] - TOLERANCE_M <= v.co[i] <= hi[i] + TOLERANCE_M:
                    raise ValueError(f"{d.recipe}: geometry leaves the {d.cls} box in {'xyz'[i]} at {v.co[i]:.3f} m "
                                     f"(box {lo[i]:.3f} to {hi[i]:.3f})")
        ob = super().finish()
        for holder in (ob, ob.data):
            holder["stbench_class"], holder["stbench_recipe"] = d.cls, d.recipe
            if self.text is not None:
                holder["stbench_text"] = self.text
        return ob


class _Turned:
    def __init__(self, mesh: Mesh, matrix: Matrix):
        self.mesh, self.matrix = mesh, matrix

    def __enter__(self):
        self.before = set(self.mesh.bm.verts)

    def __exit__(self, *exc):
        bmesh.ops.transform(self.mesh.bm, matrix=self.matrix, verts=[v for v in self.mesh.bm.verts
                                                                     if v not in self.before])

