# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Shapes for the detailed procedural kinds (`procedural_*.py`) that `meshkit` does not have: lofts through rings of
points (a rounded-rectangle shell with ribs, a fork, a hood), tubes swept along a rounded path (a tiller, a loop handle,
a wire hoop) and the deterministic wobble that wrinkles film and cloth.

Runs inside Blender's own Python and is never imported by `stbench`. Like `meshkit`, every helper adds faces to a
`bmesh` in metres and gives them a material slot index; none knows a kind. Nothing here is random: a surface's
unevenness is a sum of sines of its own coordinates (`wobble.wave`), so the same call is the same mesh in every run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from mathutils import Vector

Point = tuple[float, float, float]


# -- lofts ---------------------------------------------------------------------------------------------------------
def loft(bm, rings: Sequence[Sequence[Point]], slot: int, *, cap_first: bool = True, cap_last: bool = True,
         wrap: bool = True, close: bool = False, smooth: bool = True) -> list:
    """Quads between consecutive rings of points (every ring the same length, `wrap` joining each ring's last point to
    its first; `close` also joins the last ring to the first: a torus), and an n-gon over the first and the last ring
    when asked. Side faces are smooth shaded, caps flat. Returns the faces. A ring of repeated points (a pole) is not
    handled: end such a shape with a small cap instead."""
    verts = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(verts[0])
    faces = []
    pairs = list(zip(verts, verts[1:], strict=False)) + ([(verts[-1], verts[0])] if close else [])
    for a, b in pairs:
        for i in range(n if wrap else n - 1):
            j = (i + 1) % n
            faces.append(bm.faces.new((a[i], a[j], b[j], b[i])))
    for f in faces:
        f.material_index, f.smooth = slot, smooth
    for ring, flip, wanted in ((verts[0], False, cap_first), (verts[-1], True, cap_last)):
        if wanted:
            f = bm.faces.new(ring[::-1] if flip else ring)
            f.material_index, f.smooth = slot, False
            faces.append(f)
    return faces


class Outline:
    """The rounded rectangle of half sizes (hx, hy) and corner radius rc as points: `mx` and `my` on each side of the x
    and y faces and `k` on each corner arc, counter-clockwise from the +x face's lower end. A point is
    (x, y, nx, ny, edge): its outward normal and its distance (m) to the nearest corner arc along its side, 0 on an arc.
    The counts are fixed so that rings at different heights line up point for point: a rib at the same index is a
    straight rib, whatever the taper."""

    def __init__(self, mx: int, my: int, k: int = 5):
        self.mx, self.my, self.k = mx, my, k
        self.n = 2 * mx + 2 * my + 4 * k

    def points(self, hx: float, hy: float, rc: float) -> list[tuple[float, float, float, float, float]]:
        rc = min(rc, hx - 1e-4, hy - 1e-4)
        out = []
        # side by side: +x face (y rising), +y face (x falling), -x face (y falling), -y face (x rising), each followed
        # by the corner arc that turns into the next
        sides = ((hx, 0.0, 1.0, 0.0, hy - rc, self.my), (0.0, hy, 0.0, 1.0, hx - rc, self.mx),
                 (-hx, 0.0, -1.0, 0.0, hy - rc, self.my), (0.0, -hy, 0.0, -1.0, hx - rc, self.mx))
        for q, (px, py, nx, ny, half, m) in enumerate(sides):
            tx, ty = -ny, nx  # the direction of travel along the face
            for i in range(m):
                f = (i + 0.5) / m * 2.0 - 1.0  # -1..1 along the face's straight part
                out.append((px + tx * f * half, py + ty * f * half, nx, ny, half * (1.0 - abs(f))))
            cx, cy = [(hx - rc, hy - rc), (-hx + rc, hy - rc), (-hx + rc, -hy + rc), (hx - rc, -hy + rc)][q]
            for i in range(self.k):
                a = math.pi / 2 * (q + (i + 0.5) / self.k)
                out.append((cx + rc * math.cos(a), cy + rc * math.sin(a), math.cos(a), math.sin(a), 0.0))
        return out


def rounded_path(points: Sequence[Point], bend: float, steps: int = 6) -> list[Point]:
    """A polyline with every interior corner rounded: the corner is replaced by a quadratic curve that starts `bend`
    metres (less on short legs) before it and ends as far after it, so a tube swept along the result has no kink."""
    pts = [Vector(p) for p in points]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, p, b = pts[i - 1], pts[i], pts[i + 1]
        t = min(bend, 0.5 * (p - a).length, 0.5 * (b - p).length)
        s, e = p + (a - p).normalized() * t, p + (b - p).normalized() * t
        for j in range(steps + 1):
            u = j / steps
            out.append((1 - u) ** 2 * s + 2 * u * (1 - u) * p + u * u * e)
    out.append(pts[-1])
    return [tuple(v) for v in out]


def closed_rounded_path(points: Sequence[Point], bend: float, steps: int = 6) -> list[Point]:
    """`rounded_path` for a closed loop: every corner, the first and the last included, is rounded."""
    pts = [Vector(p) for p in points]
    n = len(pts)
    out = []
    for i in range(n):
        a, p, b = pts[i - 1], pts[i], pts[(i + 1) % n]
        t = min(bend, 0.5 * (p - a).length, 0.5 * (b - p).length)
        s, e = p + (a - p).normalized() * t, p + (b - p).normalized() * t
        for j in range(steps + 1):
            u = j / steps
            out.append(tuple((1 - u) ** 2 * s + 2 * u * (1 - u) * p + u * u * e))
    return out


def sweep(bm, path: Sequence[Point], r: float, slot: int, *, segments: int = 12, closed: bool = False,
          radii: Sequence[float] | None = None, caps: bool = True, smooth: bool = True) -> None:
    """A tube of radius `r` (or `radii`, one per path point) along `path`, a closed loop when `closed`. The ring's first
    point lies on the axis the path leaves along least, so a tube parallel to an axis reaches exactly +-r on the other
    two (`segments` a multiple of 4)."""
    pts = [Vector(p) for p in path]
    n = len(pts)
    tans = [((pts[(i + 1) % n] if closed else pts[min(i + 1, n - 1)])
             - (pts[i - 1] if closed else pts[max(i - 1, 0)])).normalized() for i in range(n)]
    ref = min((Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))), key=lambda a: abs(a.dot(tans[0])))
    normal = (ref - tans[0] * ref.dot(tans[0])).normalized()
    rings = []
    for i, p in enumerate(pts):
        normal = (normal - tans[i] * normal.dot(tans[i])).normalized()
        binormal = tans[i].cross(normal)
        rad = radii[i] if radii else r
        rings.append([tuple(p + rad * (math.cos(2 * math.pi * k / segments) * normal
                                       + math.sin(2 * math.pi * k / segments) * binormal)) for k in range(segments)])
    if closed:
        loft(bm, rings, slot, cap_first=False, cap_last=False, close=True, smooth=smooth)
    else:
        loft(bm, rings, slot, cap_first=caps, cap_last=caps, smooth=smooth)


def rings_of(outline: Outline, stations: Sequence[float], profile: Callable[[float], tuple[float, float, float]],
             push: Callable[[float, float, tuple], float] | None = None) -> list[list[Point]]:
    """Rounded-rectangle rings at the heights `stations`: `profile(z)` gives (hx, hy, corner radius) there and
    `push(z, u, point)` an extra distance along each point's outward normal (`u` runs 0..1 round the ring; `point` is
    the `Outline` tuple (x, y, nx, ny, edge))."""
    rings = []
    for z in stations:
        ring = []
        for i, p in enumerate(pts := outline.points(*profile(z))):
            x, y, nx, ny, _ = p
            d = push(z, i / len(pts), p) if push else 0.0
            ring.append((x + nx * d, y + ny * d, z))
        rings.append(ring)
    return rings


def along_x(outline: Outline, stations: Sequence[float],
            profile: Callable[[float], tuple[float, float, float, float, float]]) -> list[list[Point]]:
    """Rounded-rectangle sections in planes of constant x: `profile(x)` gives (y centre, z centre, half width, half
    height, corner radius), and each ring is (x, y, z) points, for a hood, a fork, a body that runs along x."""
    rings = []
    for x in stations:
        cy, cz, hu, hv, rc = profile(x)
        rings.append([(x, cy + u, cz + v) for u, v, *_ in outline.points(hu, hv, rc)])
    return rings
