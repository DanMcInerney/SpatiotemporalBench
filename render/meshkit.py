# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Mesh helpers shared by the procedural builders (`procedural.py` and the warehouse modules): lathe profiles, bevelled
slabs, tapering blades, and the shapes the warehouse needs (prisms, rods, open tubs, grids).

Runs inside Blender's own Python and is never imported by `stbench`. Every helper adds faces to a `bmesh` in metres and
gives them a material slot index; none of them knows a kind. Face normals are fixed once, by `procedural.build`.
"""

from __future__ import annotations

import math

import bmesh
from mathutils import Matrix, Vector

SEGMENTS = 48


def fit(bm, box) -> None:
    """Translate to the base centre and scale each axis so the extents are exactly `box`."""
    co = [v.co for v in bm.verts]
    lo = Vector([min(c[i] for c in co) for i in range(3)])
    hi = Vector([max(c[i] for c in co) for i in range(3)])
    s = [box[i] / (hi[i] - lo[i]) for i in range(3)]
    mid = (lo + hi) / 2
    for v in bm.verts:
        v.co = Vector(((v.co.x - mid.x) * s[0], (v.co.y - mid.y) * s[1], (v.co.z - lo.z) * s[2]))


def lathe(bm, profile, slot: int, segments: int = SEGMENTS, offset=(0.0, 0.0), loop: bool = False) -> list:
    """Revolve a profile of (radius, z) points around the z axis through `offset`. A point at radius 0 is a pole. With
    `loop` the last point joins the first: a ring whose profile is a closed outline (a tyre), touching no pole."""
    ox, oy = offset
    rings = []
    for r, z in profile:
        if r <= 1e-9:
            rings.append([bm.verts.new((ox, oy, z))])
        else:
            rings.append([bm.verts.new((ox + r * math.cos(2 * math.pi * i / segments),
                                        oy + r * math.sin(2 * math.pi * i / segments), z)) for i in range(segments)])
    faces = []
    for a, b in zip(rings, rings[1:] + rings[:1] if loop else rings[1:], strict=False):
        for i in range(segments):
            j = (i + 1) % segments
            quad = [a[i % len(a)], a[j % len(a)], b[j % len(b)], b[i % len(b)]]
            quad = list(dict.fromkeys(quad))  # a pole collapses the quad to a triangle
            if len(quad) >= 3:
                faces.append(bm.faces.new(quad))
    for f in faces:
        f.material_index, f.smooth = slot, True
    return faces


def slab(bm, x0, x1, y0, y1, z0, z1, slot: int, bevel: float = 0.0, segments: int = 3) -> None:
    """An axis-aligned box, its edges rounded by `bevel` (the flat faces stay flat shaded)."""
    centre, extent = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0, 1)
    m = Matrix.Translation(centre) @ Matrix.Diagonal(extent)
    verts = bmesh.ops.create_cube(bm, size=1.0, matrix=m)["verts"]
    for f in {f for v in verts for f in v.link_faces}:
        f.material_index = slot  # a bevel's new faces inherit it
    if bevel > 0:
        edges = list({e for v in verts for e in v.link_edges})
        res = bmesh.ops.bevel(bm, geom=verts + edges, offset=min(bevel, 0.45 * min(x1 - x0, y1 - y0, z1 - z0)),
                              segments=segments, profile=0.5, affect="EDGES")
        for f in res["faces"]:
            f.smooth = True


def solid(bm, x0, x1, y0, y1, z0, z1, slot: int, bevel: float = 0.0, segments: int = 3) -> None:
    """`slab` with the bevel's faces on `slot` as well (`slab` leaves them on slot 0, which is harmless where slot 0 is
    the slot or the bevel is tiny, and which the household builders keep as they are)."""
    before = set(bm.faces)
    slab(bm, x0, x1, y0, y1, z0, z1, slot, bevel=bevel, segments=segments)
    for f in bm.faces:
        if f not in before:
            f.material_index = slot


def blade(bm, azimuth: float, tilt: float, bend: float, length: float, width: float, thick: float, base, slot: int,
          rings: int = 7) -> None:
    """A tapering leaf, a four-sided blade closed at both ends. It leaves `base` leaning `tilt` degrees from vertical
    towards `azimuth` (degrees about z) and bends `bend` degrees further outward along its `length`; it is widest near
    the base and ends in a point."""
    az = math.radians(azimuth)
    out, side = Vector((math.cos(az), math.sin(az), 0.0)), Vector((-math.sin(az), math.cos(az), 0.0))
    up = Vector((0, 0, 1))
    centre, rows = Vector(base), []
    for i in range(rings + 1):
        s, a = i / rings, math.radians(tilt + bend * i / rings)
        w, t = width * (1 - s ** 1.4) * min(1.0, 0.45 + 2.2 * s), thick * (1 - s)
        n = out * math.cos(a) - up * math.sin(a)  # the blade's upper face points this way
        corners = [centre] if i == rings else [centre + d for d in (-side * w / 2, n * t / 2, side * w / 2, -n * t / 2)]
        rows.append([bm.verts.new(c) for c in corners])
        centre = centre + (out * math.sin(a) + up * math.cos(a)) * (length / rings)
    faces = [bm.faces.new(rows[0])]
    for a, b in zip(rows, rows[1:], strict=False):
        for j in range(4):
            faces.append(bm.faces.new([a[j], a[(j + 1) % 4], *([b[(j + 1) % 4], b[j]] if len(b) == 4 else [b[0]])]))
    for f in faces:
        f.material_index, f.smooth = slot, True


def placed(bm, matrix, build) -> None:
    """Call `build()`, which adds geometry to `bm`, then move everything it added by `matrix`."""
    before = set(bm.verts)
    build()
    bmesh.ops.transform(bm, matrix=matrix, verts=[v for v in bm.verts if v not in before])


# -- shapes for the warehouse ------------------------------------------------------------------------------------
def _face(bm, verts, slot: int, smooth: bool = False):
    f = bm.faces.new(verts)
    f.material_index, f.smooth = slot, smooth
    return f


def prism(bm, outline, y0: float, y1: float, slot: int) -> None:
    """A closed solid: the polygon `outline` of (x, z) points, in order round it, extruded from y0 to y1."""
    front = [bm.verts.new((x, y0, z)) for x, z in outline]
    back = [bm.verts.new((x, y1, z)) for x, z in outline]
    _face(bm, front, slot)
    _face(bm, back[::-1], slot)
    n = len(outline)
    for i in range(n):
        j = (i + 1) % n
        _face(bm, [front[i], front[j], back[j], back[i]], slot)


def rod(bm, a, b, r: float, slot: int, segments: int = 12) -> None:
    """A closed cylinder of radius `r` from point `a` to point `b`."""
    a, b = Vector(a), Vector(b)
    d = b - a
    turn = Vector((0.0, 0.0, 1.0)).rotation_difference(d).to_matrix().to_4x4()
    placed(bm, Matrix.Translation(a) @ turn,
           lambda: lathe(bm, [(0.0, 0.0), (r, 0.0), (r, d.length), (0.0, d.length)], slot, segments=segments))


def wheel(bm, cx: float, cy: float, cz: float, r: float, width: float, axis: str, slot: int,
          segments: int = 20) -> None:
    """A tyre of radius `r` and `width` centred on (cx, cy, cz), its axle along x or y."""
    spin = Matrix.Rotation(math.pi / 2, 4, "Y") if axis == "x" else Matrix.Rotation(-math.pi / 2, 4, "X")
    h = width / 2
    placed(bm, Matrix.Translation((cx, cy, cz)) @ spin,
           lambda: lathe(bm, [(0.0, -h), (r * 0.92, -h), (r, -h * 0.7), (r, h * 0.7), (r * 0.92, h), (0.0, h)], slot,
                         segments=segments))


def _ring4(bm, hx: float, hy: float, z: float) -> list:
    return [bm.verts.new((sx * hx, sy * hy, z)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _bridge(bm, lower: list, upper: list, slot: int) -> None:
    for i in range(len(lower)):
        j = (i + 1) % len(lower)
        _face(bm, [lower[i], lower[j], upper[j], upper[i]], slot)


def tub(bm, hx: float, hy: float, z0: float, z1: float, taper: float, wall: float, floor: float, slot: int) -> None:
    """An open rectangular tub, wider at the top: outer half sizes (hx, hy) at z1 and `taper` times that at z0, walls
    `wall` thick, a floor `floor` thick, the rim flat."""
    def outer(z: float) -> tuple[float, float]:
        k = taper + (1.0 - taper) * (z - z0) / (z1 - z0)
        return hx * k, hy * k

    zf = z0 + floor
    ob, ot = _ring4(bm, *outer(z0), z0), _ring4(bm, *outer(z1), z1)
    it = _ring4(bm, outer(z1)[0] - wall, outer(z1)[1] - wall, z1)
    ib = _ring4(bm, outer(zf)[0] - wall, outer(zf)[1] - wall, zf)
    for lower, upper in ((ob, ot), (ot, it), (ib, it)):
        _bridge(bm, lower, upper, slot)
    _face(bm, ob, slot)
    _face(bm, ib[::-1], slot)


def pillow(bm, hx: float, hy: float, height: float, slot: int, nu: int = 48, nv: int = 28, flat: tuple = (3.2, 2.2),
           lumps: float = 0.05) -> None:
    """A soft sack lying on its side, closed: a flat underside and a domed top that meets it along the outline
    (x within +-hx, y within +-hy). It is `height` high at the middle; `flat` (the exponents along x and y, 2 being an
    ellipse) makes the top plateau-like and the sides steep as they grow; `lumps` is a deterministic unevenness."""
    pu, pv = flat

    def top(u: float, v: float) -> float:
        dome = max(0.0, 1.0 - abs(u) ** pu) ** 0.5 * max(0.0, 1.0 - abs(v) ** pv) ** 0.5
        return height * dome * (1.0 + lumps * math.sin(5.0 * u + 1.0) * math.cos(4.0 * v))

    grid = [[bm.verts.new((u * hx, v * hy, top(u, v))) for v in (-1.0 + 2.0 * j / nv for j in range(nv + 1))]
            for u in (-1.0 + 2.0 * i / nu for i in range(nu + 1))]
    under = [[v if (i in (0, nu) or j in (0, nv)) else bm.verts.new((v.co.x, v.co.y, 0.0))
              for j, v in enumerate(row)] for i, row in enumerate(grid)]
    for g, flip in ((grid, False), (under, True)):
        for i in range(nu):
            for j in range(nv):
                quad = [g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]]
                _face(bm, quad[::-1] if flip else quad, slot, smooth=True)
