# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""A light street tree and shrubs, as meshes: a trunk in a cast-iron tree pit, and a crown of leaf lumps.

Runs inside Blender's own Python and is never imported by `stbench`. The tree is two pieces so that the crown can sit
above what a question may put on the sidewalk (nothing tracked is taller than 3.3 m): `trunk` is the pit and the stem up
to the crown's foot, `crown` is the leaf mass above it, with a leader and the first limbs. Both are built in their own
frame (origin at the base centre, x across the sidewalk towards the kerb, y along it, z up) and fitted to the box they
are given (unless `snug` is false), so a Dressing's box holds every vertex. `lumps` is also the shrub of a planter.
Leaf lumps are noisy ellipsoids, a few hundred faces for a tree; a scene instances one mesh per distinct tree. Seeded
by the caller.
"""

from __future__ import annotations

import math
import random

from meshkit import fit, lathe, rod, slab

LUMP_SEGMENTS = (10, 6)


def _blob(bm, slot: int, rng: random.Random, centre, radii) -> None:
    """A noisy ellipsoid: `LUMP_SEGMENTS` rings of vertices pushed in or out by up to 14%, smooth shaded. Made vertex by
    vertex (a bmesh operator for each of a crown's lumps walks the whole mesh each time: quadratic)."""
    cx, cy, cz = centre
    nu, nv = LUMP_SEGMENTS
    top = bm.verts.new((cx, cy, cz + radii[2] * (1.0 + rng.uniform(-0.1, 0.1))))
    bottom = bm.verts.new((cx, cy, cz - radii[2] * (1.0 + rng.uniform(-0.1, 0.1))))
    rings = []
    for j in range(1, nv):
        phi = math.pi * j / nv
        ring = []
        for i in range(nu):
            f = 1.0 + rng.uniform(-0.14, 0.14)
            th = 2 * math.pi * i / nu
            ring.append(bm.verts.new((cx + radii[0] * f * math.sin(phi) * math.cos(th),
                                      cy + radii[1] * f * math.sin(phi) * math.sin(th),
                                      cz + radii[2] * f * math.cos(phi))))
        rings.append(ring)
    faces = [bm.faces.new((top, rings[0][(i + 1) % nu], rings[0][i])) for i in range(nu)]
    faces += [bm.faces.new((bottom, rings[-1][i], rings[-1][(i + 1) % nu])) for i in range(nu)]
    for a, b in zip(rings, rings[1:], strict=False):
        faces += [bm.faces.new((a[i], a[(i + 1) % nu], b[(i + 1) % nu], b[i])) for i in range(nu)]
    for f in faces:
        f.material_index, f.smooth = slot, True


def lumps(bm, slot: int, rng: random.Random, centre, radii, n: int, size: tuple[float, float]) -> None:
    """`n` leaf lumps scattered in the ellipsoid at `centre` with semi-axes `radii`: each a noisy ellipsoid whose
    semi-axes are `size` (lower, upper) times a random factor, a little flatter than wide."""
    cx, cy, cz = centre
    for i in range(n):
        # a spiral fill of the ellipsoid: even in angle, denser towards the middle
        a = i * 2.399963 + rng.uniform(-0.3, 0.3)
        w = 1.0 - 2.0 * (i + 0.5) / n
        r = (0.35 + 0.65 * rng.random()) ** 0.5
        s = math.sqrt(max(0.0, 1.0 - w * w))
        k = rng.uniform(*size)
        _blob(bm, slot, rng, (cx + radii[0] * r * s * math.cos(a), cy + radii[1] * r * s * math.sin(a),
                              cz + radii[2] * r * w),
              (k * rng.uniform(0.9, 1.15), k * rng.uniform(0.9, 1.15), k * rng.uniform(0.75, 0.95)))


def trunk(bm, box: tuple[float, float, float], bark: int, iron: int, snug: bool = True) -> None:
    """The tree pit and the stem: a flush cast-iron frame `box[0]` across by `box[1]` along with a round opening, a ring
    round the stem, and the stem itself from the pit to the box's top (a little thinner there)."""
    sx, sy, sz = box
    hx, hy, w, t = sx / 2, sy / 2, 0.11, 0.02
    for x0, x1, y0, y1 in ((-hx, hx, -hy, -hy + w), (-hx, hx, hy - w, hy), (-hx, -hx + w, -hy + w, hy - w),
                           (hx - w, hx, -hy + w, hy - w)):
        slab(bm, x0, x1, y0, y1, 0.0, t, iron)
    lathe(bm, [(0.0, t), (0.19, t), (0.19, t + 0.02), (0.15, t + 0.02)], iron, segments=16)  # the ring round the stem
    lathe(bm, [(0.0, 0.0), (0.15, t), (0.146, 0.14), (0.125, 1.0), (0.11, 2.2), (0.095, sz), (0.0, sz)], bark,
          segments=12)
    if snug:
        fit(bm, box)


def crown(bm, box: tuple[float, float, float], bark: int, leaf: int, rng: random.Random, snug: bool = True) -> None:
    """The leaf mass: an oval of lumps filling the upper two thirds of the box over a leader and four limbs that run
    out from the foot into the lower lumps."""
    sx, sy, sz = box
    u = min(sx, sy)
    lumps(bm, leaf, rng, (0.0, 0.0, 0.56 * sz), (0.38 * sx, 0.38 * sy, 0.34 * sz), 36, (0.15 * u, 0.23 * u))
    lumps(bm, leaf, rng, (0.0, 0.0, 0.24 * sz), (0.34 * sx, 0.34 * sy, 0.08 * sz), 9, (0.12 * u, 0.18 * u))
    rod(bm, (0.0, 0.0, 0.0), (0.0, 0.0, 0.6 * sz), 0.045, bark, segments=8)  # the leader
    for k in range(5):
        a = rng.uniform(0, 2 * math.pi) + k * 2 * math.pi / 5
        rod(bm, (0.0, 0.0, 0.03 * sz), (0.26 * sx * math.cos(a), 0.26 * sy * math.sin(a), 0.36 * sz), 0.025, bark,
            segments=6)
    if snug:
        fit(bm, box)
