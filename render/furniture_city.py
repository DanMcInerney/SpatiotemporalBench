# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's fixed furniture: sidewalk slabs with a kerb, parking lanes with painted bay lines, loading zones with
yellow hatching, the square's paving with its pads, and the placeholder drawings of the drone's nest and drop zones.

Runs inside Blender's own Python and is never imported by `stbench`. Like `furniture_warehouse.py`, each builder takes
the entity size (depth, width, height), its colour attribute and its receptacle surfaces from the map as `(cx, cy, sx,
sy, z)` in the entity's frame (origin at the base centre, x = front, y = left, z up), because a lane must know where its
bays are: surfaces sit at exactly the heights the map gives, so carried things rest on them. The front of a sidewalk is
its kerb side; the front of a parking lane is its road side.

The parts name `Surfaces` keys instead of palette colours (`sidewalk`, `kerb`, `paving`, `road_white`, `road_yellow`):
procedural materials that every appearance draws alike (no download), as the warehouse's concrete floor is.
"""

from __future__ import annotations

import math

import materials_street
from shapes import Part, box

LINE_M = 0.1  # the width of a painted line
PAINT_M = 0.005  # the thickness of a drop zone's marking: paint on the ground under it
ZONE_LINE_M = 0.15  # the width of a drop zone's outline and of its cross
NEST_PAINT_M = 0.0005  # the thickness of the paint on a nest's mat: the pad is solid to its height, paint and all
RING_SEGMENTS, NEST_LINE_M = 24, 0.1  # a nest's painted ring (straight pieces of paint round a circle) and its H
KERB_M = 0.2  # the kerb stone: the sidewalk slab's front edge


class Surfaces:
    """The materials of the city's ground pieces, each made once per scene."""

    KEYS = {"sidewalk": lambda: materials_street.sidewalk(edges=True), "kerb": materials_street.kerb,
            "paving": materials_street.paving, "road_white": lambda: materials_street.road_paint("white"),
            "road_yellow": lambda: materials_street.road_paint("yellow")}

    def __init__(self):
        self.made: dict[str, object] = {}

    def resolver(self, base=None):
        """`key -> Material | None`: these keys, then `base`'s (the appearance's own resolver, or none)."""
        def resolve(key: str):
            if key in self.KEYS:
                if key not in self.made:
                    self.made[key] = self.KEYS[key]()
                return self.made[key]
            return base(key) if base else None

        return resolve


def _outline(x0, x1, y0, y1, z0, z1, mat, w: float = LINE_M) -> list[Part]:
    """Four lines of paint, each `w` wide, round a rectangle."""
    return [box(x0, x1, y0, y0 + w, z0, z1, mat), box(x0, x1, y1 - w, y1, z0, z1, mat),
            box(x0, x0 + w, y0 + w, y1 - w, z0, z1, mat), box(x1 - w, x1, y0 + w, y1 - w, z0, z1, mat)]


def sidewalk(s, c, surfaces) -> list[Part]:
    """A concrete slab with a granite kerb along its front edge."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx - KERB_M, -hy, hy, 0, sz, "sidewalk"), box(hx - KERB_M, hx, -hy, hy, 0, sz, "kerb")]


def parking_lane(s, c, surfaces) -> list[Part]:
    """Painted bays: a line across the lane between each pair of bays and at both ends, and one along the road side. The
    paint is the colour attribute's (white); the road beneath is the region's floor."""
    sx, sy, sz = s
    key = f"road_{c}"
    bays = sorted(surfaces, key=lambda b: b[1])
    x0, x1 = bays[0][0] - bays[0][2] / 2, bays[0][0] + bays[0][2] / 2
    edges = [bays[0][1] - bays[0][3] / 2]
    edges += [((a[1] + a[3] / 2) + (b[1] - b[3] / 2)) / 2 for a, b in zip(bays, bays[1:], strict=False)]
    edges.append(bays[-1][1] + bays[-1][3] / 2)
    w = LINE_M
    parts = [box(x1 - w, x1, edges[0], edges[-1], 0, sz, key)]
    for i, y in enumerate(edges):
        a, b = (y, y + w) if i == 0 else (y - w, y) if i == len(edges) - 1 else (y - w / 2, y + w / 2)
        parts.append(box(x0, x1, a, b, 0, sz, key))
    return parts


def loading_zone(s, c, surfaces) -> list[Part]:
    """A yellow outline with diagonal hatching inside it (no text)."""
    sx, sy, sz = s
    cx, cy, bx, by, _ = surfaces[0]
    x0, x1, y0, y1 = cx - bx / 2, cx + bx / 2, cy - by / 2, cy + by / 2
    key = f"road_{c}"
    parts = _outline(x0, x1, y0, y1, 0, sz, key)
    inner = (x1 - x0) / 2 - LINE_M  # half the width inside the outline
    length = 2 * inner * math.sqrt(2)  # a 45 degree stripe that spans it
    y = y0 + LINE_M + inner + 0.3
    while y < y1 - LINE_M - inner:
        parts.append(Part("box", ((x0 + x1) / 2, y, sz / 2), (length, 0.14, sz), (0, 0, math.pi / 4), key))
        y += 1.5
    return parts


def plaza(s, c, surfaces) -> list[Part]:
    """Paving flags with a painted square on each pad."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx, -hy, hy, 0, sz, "paving")]
    for cx, cy, bx, by, z in surfaces:
        parts += _outline(cx - bx / 2, cx + bx / 2, cy - by / 2, cy + by / 2, z, z + 0.01, "road_white")
    return parts


def nest(s, c, surfaces) -> list[Part]:
    """The drone's landing pad: a dark mat the piece's size (the colour attribute, darkened), solid from the ground to
    its height less `NEST_PAINT_M`, with a white ring 0.25 m in from the edge and an H inside the ring painted on its
    top face up to the piece's height. From above it reads as a pad: a dark square, a ring, an H."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    z0 = sz - NEST_PAINT_M
    radius = min(hx, hy) - 0.25  # the ring's middle line
    chord = 2 * radius * math.sin(math.pi / RING_SEGMENTS) * 1.05  # a little long, so neighbours overlap
    parts = [box(-hx, hx, -hy, hy, 0, z0, f"{c}*0.4")]
    for k in range(RING_SEGMENTS):
        a = 2 * math.pi * k / RING_SEGMENTS
        parts.append(Part("box", (radius * math.cos(a), radius * math.sin(a), (z0 + sz) / 2),
                          (chord, NEST_LINE_M, NEST_PAINT_M), (0, 0, a + math.pi / 2), "road_white"))
    w, a, b = NEST_LINE_M, 0.4 * radius, 0.6 * radius  # the H: two legs along x at y = +-a, half a leg b, a bar between
    parts += [box(-b, b, y - w / 2, y + w / 2, z0, sz, "road_white") for y in (-a, a)]
    parts.append(box(-w / 2, w / 2, -a + w / 2, a - w / 2, z0, sz, "road_white"))
    return parts


def drop_zone(s, c, surfaces) -> list[Part]:
    """A drop zone: paint on the ground under it (whatever that ground's height), an outline 0.15 m wide and a cross at
    the centre, from `sz - 0.005` to `sz`. The colour attribute is the paint's (white)."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    key, z0, w = f"road_{c}", sz - PAINT_M, ZONE_LINE_M
    parts = _outline(-hx, hx, -hy, hy, z0, sz, key, w)
    parts.append(box(-hx + w, hx - w, -w / 2, w / 2, z0, sz, key))  # the cross: a bar across, then the two ends of
    parts += [box(-w / 2, w / 2, y0, y1, z0, sz, key) for y0, y1 in ((-hy + w, -w / 2), (w / 2, hy - w))]  # a bar along
    return parts


BUILDERS = {f.__name__: f for f in (sidewalk, parking_lane, loading_zone, plaza, nest, drop_zone)}
