# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Vehicle mesh helpers on top of `meshkit`: rounded-section lofts, face painting by rule, tyres, rubble.

Runs inside Blender's own Python and is never imported by `stbench`. A vehicle is built in its local frame (origin at
the base centre, x forward, y left, z up, metres) into one bmesh whose faces carry a material slot index. A loft sweeps
a rounded trapezoid section along x: bodies, cabs and hoods are lofts; boxes and beams are `solid`; tyres are `wheels`.
No helper knows a kind.
"""

from __future__ import annotations

import math

import bmesh
from mathutils import Euler, Matrix
from meshkit import fit, placed, wheel
from meshkit import solid as _solid


def to_box(builder):
    """A builder `(bmesh, box, slots)` that ends by fitting what it drew to `box` exactly. The drawings are the
    look-dev's measurements (a saloon is 4.66 m with its bumpers, a dumpster 2.1 m with its lids up) and the catalog's
    box is the truth the world uses; `procedural.build` fits to it again, so this changes nothing the library holds, but
    a builder's own extents are then the box, as the movers' test asks of every mover builder."""
    def fitted(bm, box, S) -> None:
        builder(bm, box, S)
        fit(bm, box)

    fitted.__doc__ = builder.__doc__
    return fitted


def solid(bm, x0, x1, y0, y1, z0, z1, slot, **kw) -> None:
    """`meshkit.solid` for any order of the coordinates (a mirrored side passes y0 > y1)."""
    _solid(bm, min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), min(z0, z1), max(z0, z1), slot, **kw)


def _bez(a, c, b, n):
    return [((1 - t) ** 2 * a[0] + 2 * t * (1 - t) * c[0] + t * t * b[0],
             (1 - t) ** 2 * a[1] + 2 * t * (1 - t) * c[1] + t * t * b[1]) for t in (i / (n + 1) for i in range(n + 2))]


def ring(hwb, hwt, z0, z1, r=0.12, n=3, belt=None):
    """(y, z) points of a trapezoid section: half width `hwb` at the bottom z0 and `hwt` at the top z1, each corner
    rounded by about `r` (n + 2 points). `belt` adds a vertex at that height (or heights) on each side wall. The point
    count depends only on `n` and on the belts, so sections of one loft always match."""
    cs = [(-hwb, z0), (hwb, z0), (hwt, z1), (-hwt, z1)]
    arcs, ds = [], []
    for i, c in enumerate(cs):
        a, b = cs[i - 1], cs[(i + 1) % 4]
        la, lb = math.dist(a, c), math.dist(b, c)
        d = max(1e-3, min(r, 0.45 * la, 0.45 * lb))
        ds.append(d)
        p1 = (c[0] + (a[0] - c[0]) * d / la, c[1] + (a[1] - c[1]) * d / la)
        p2 = (c[0] + (b[0] - c[0]) * d / lb, c[1] + (b[1] - c[1]) * d / lb)
        arcs.append(_bez(p1, c, p2, n))
    pts = list(arcs[0]) + arcs[1]  # the bottom-left arc ends on the bottom edge, the bottom-right on the right wall
    side = []
    if belt is not None:  # vertices on each wall, kept between the two corner arcs
        wall = math.dist(cs[1], cs[2])
        for zb in sorted((belt,) if isinstance(belt, int | float) else belt):
            f = min(max((zb - z0) / (z1 - z0), ds[1] / wall + 0.02), 1 - ds[2] / wall - 0.02)
            side.append((hwb + (hwt - hwb) * f, z0 + (z1 - z0) * f))
    pts += side + arcs[2] + arcs[3] + [(-y, z) for y, z in reversed(side)]
    return pts


def loft(bm, stations, slot: int, cap: bool = True) -> list:
    """Sweep sections along x. `stations` is [(x, [(y, z), ...]), ...] with equal point counts; returns the faces."""
    rows = [[bm.verts.new((x, y, z)) for y, z in pts] for x, pts in stations]
    faces = []
    k = len(rows[0])
    for a, b in zip(rows, rows[1:], strict=False):
        for j in range(k):
            k2 = (j + 1) % k
            faces.append(bm.faces.new([a[j], a[k2], b[k2], b[j]]))
    if cap:
        faces.append(bm.faces.new(rows[0][::-1]))
        faces.append(bm.faces.new(rows[-1]))
    for f in faces:
        f.material_index, f.smooth = slot, True
    return faces


def interp(xs, ys):
    """Piecewise-linear function through (xs, ys); flat outside."""
    def f(x):
        if x <= xs[0]:
            return ys[0]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] or 1.0)
                return ys[i - 1] + (ys[i] - ys[i - 1]) * t
        return ys[-1]
    return f


def normals(bm) -> None:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()


def paint(faces, rule) -> None:
    """Give each face the slot `rule(centre, normal)` returns (None keeps its slot). Call after `normals`."""
    for f in faces:
        s = rule(f.calc_center_median(), f.normal)
        if s is not None:
            f.material_index = s


def rim(bm, S, x: float, y: float, r: float, s: float) -> None:
    """The face of an alloy wheel: five dark spokes and a centre cap on the hub disc whose outer face is at `y` (`s`:
    the side, +1 or -1), a few millimetres proud of it."""
    for k in range(5):
        spin = Matrix.Translation((x, y, r)) @ Matrix.Rotation(2 * math.pi * k / 5, 4, "Y")
        placed(bm, spin, lambda: _solid(bm, 0.05 * r, 0.56 * r, min(0.0, s * 0.004), max(0.0, s * 0.004), -0.016, 0.016,
                                        S["trim"]))
    wheel(bm, x, y + s * 0.003, r, r * 0.14, 0.006, "y", S["trim"])


def wheels(bm, S, xs, hw, r, width, arch: bool = True, dual_rear: bool = False) -> None:
    """Tyres with hubs on both sides at each x in `xs` (outer face 1.5 cm proud of the body half width `hw`), a rim on
    each hub (spokes and a cap), and a dark arch disc behind each so the wheel reads against the paint."""
    for x in xs:
        for s in (-1.0, 1.0):
            cy = s * (hw + 0.015 - width / 2)
            if arch:
                wheel(bm, x, s * (hw + 0.002), r + 0.04, r + 0.05, 0.008, "y", S["trim"])
            wheel(bm, x, cy, r, r, width, "y", S["tyres"])
            wheel(bm, x, s * (hw + 0.016 - 0.004), r, r * 0.58, 0.012, "y", S["hub"])
            rim(bm, S, x, s * (hw + 0.018), r, s)
            if dual_rear and x == xs[-1]:
                wheel(bm, x, s * (hw - width - 0.01), r, r, width, "y", S["tyres"])


def lump(bm, centre, size, rot, slot: int, sphere: bool = False) -> None:
    """A rotated box (or an ellipsoid) of full extents `size` at `centre`: rubble, cargo, a rounded housing."""
    m = Matrix.Translation(centre) @ Euler(rot).to_matrix().to_4x4() @ Matrix.Diagonal((*size, 1.0))
    if sphere:
        res = bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=8, radius=0.5, matrix=m)
    else:
        res = bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = slot
        f.smooth = sphere
