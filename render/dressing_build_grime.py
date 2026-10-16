# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's grime (`dressing_home_grime` lays it out): dust along a baseboard, hairs, crumbs and a scuff.

Runs inside Blender's own Python and is never imported by `stbench`. Each builder is `fn(d, ctx, text) -> [object]`
(`dressing.py`) and draws its piece in the Dressing's frame (`dressing_mesh`): x out of the wall (or the way it faces),
y across, z up, spanning its box. A piece is flush: nothing is more than 4 mm thick, so it is under what a robot drives
over. `text` is the variant of a hair or a cluster of crumbs, the seed of its shape, so identical pieces share a mesh
and there are four of each. Dust and a scuff are films (a thin slab whose material is clear where there is none); hairs
and crumbs are solids of their own colour.
"""

from __future__ import annotations

import math
import random

import bpy
from dressing_mesh import Build

STRANDS = (2, 4)  # strands in a hair
SEGMENTS = 11  # rods in a strand
R_HAIR = 0.0006  # a strand is 1.2 mm thick (a hair is 0.07 mm: a pixel at this range, so it is drawn wider)


def dust_line(d, ctx, text) -> list[bpy.types.Object]:
    """A slab the piece's size in dust, the mask of its material (thickest at the wall) deciding what shows."""
    t, w, h = d.size
    b = Build(ctx, d.cls)
    b.box(-t / 2, t / 2, -w / 2, w / 2, 0.0, h, "dust")
    return [b.finish()]


def scuff(d, ctx, text) -> list[bpy.types.Object]:
    t, w, h = d.size
    b = Build(ctx, d.cls)
    b.box(-t / 2, t / 2, -w / 2, w / 2, 0.0, h, "scuff_mark")
    return [b.finish()]


def _strand(b: Build, rng: random.Random, a: tuple[float, float], c: tuple[float, float], lo, hi,
            material: str) -> None:
    """One strand from `a` to `c`: `SEGMENTS` rods along an arc bowed to one side, inside the box (`lo`, `hi`)."""
    dx, dy = c[0] - a[0], c[1] - a[1]
    length = math.hypot(dx, dy)
    amp = rng.uniform(-0.2, 0.2) * length
    pts = []
    for i in range(SEGMENTS + 1):
        s, bow = i / SEGMENTS, amp * math.sin(math.pi * i / SEGMENTS)
        pts.append((min(max(a[0] + dx * s - dy / length * bow, lo[0]), hi[0]),
                    min(max(a[1] + dy * s + dx / length * bow, lo[1]), hi[1])))
    for p, q in zip(pts, pts[1:], strict=False):
        b.rod((p[0], p[1], R_HAIR), (q[0], q[1], R_HAIR), R_HAIR, material, segments=6)


def hair(d, ctx, text) -> list[bpy.types.Object]:
    """Two to four strands lying across the box: the first from one corner to the opposite one (so the piece spans its
    box), the others from a place on one end to a place on the other, each bowed, 1.2 mm thick. Dark or, now and then,
    light."""
    t, w, _ = d.size
    rng = random.Random(f"hair/{text or 0}")
    b = Build(ctx, d.cls)
    material = "hair_light" if rng.random() < 0.3 else "hair_dark"
    lo, hi = (-t / 2 + R_HAIR, -w / 2 + R_HAIR), (t / 2 - R_HAIR, w / 2 - R_HAIR)
    _strand(b, rng, lo, hi, lo, hi, material)
    for _ in range(rng.randint(*STRANDS) - 1):
        _strand(b, rng, (lo[0], rng.uniform(lo[1], hi[1])), (hi[0], rng.uniform(lo[1], hi[1])), lo, hi, material)
    return [b.finish()]


def crumbs(d, ctx, text) -> list[bpy.types.Object]:
    """A cluster of 9 to 14 specks of 2 to 6 mm, bevelled, the tallest the box's height; four stand at the middle of its
    four sides so that the piece spans its box."""
    t, w, h = d.size
    rng = random.Random(f"crumbs/{text or 0}")
    b = Build(ctx, d.cls)
    spots = [(-t / 2 + 0.004, 0.0), (t / 2 - 0.004, 0.0), (0.0, -w / 2 + 0.004), (0.0, w / 2 - 0.004)]
    spots += [(rng.uniform(-t / 2 + 0.01, t / 2 - 0.01), rng.uniform(-w / 2 + 0.01, w / 2 - 0.01))
              for _ in range(rng.randint(5, 10))]
    for i, (x, y) in enumerate(spots):
        sx, sy = rng.uniform(0.002, 0.006) / 2, rng.uniform(0.002, 0.006) / 2
        height = h if i == 0 else rng.uniform(0.0015, h)
        b.box(x - sx, x + sx, y - sy, y + sy, 0.0, height, "crumb", bevel=min(0.0008, height / 3))
    return [b.finish()]
