# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The suburb's fixed furniture: houses and apartment blocks with their roofs, and the ground and boundary pieces of a
street of plots: lawns, driveways, paths, hedges, fences, garden beds and verges.

Runs inside Blender's own Python and is never imported by `stbench`. Like `furniture_city.py`, each builder takes the
entity size (depth, width, height), its colour attribute and its receptacle surfaces from the map and returns `Part`s in
the entity's frame (origin at the base centre, x = front, y = left, z up), all inside the size's box. None reads its
surfaces: a driveway's `parking` surface is as high as the slab (the map gives both the same number), so what rests on
it rests on the drawing. The downtown's `sidewalk`, `parking_lane` and `plaza`, and `nest` and `drop_zone`, serve as
they are (`furniture_city`). `Surfaces` is the pieces' materials, made once per scene (`materials_residential`).

- `house` and `block`: `procedural_house`. Blocking, solid to the ground, drawn with their roofs.
- `lawn` (0.03 m), `driveway` (0.05 m, concrete) and `path` (paving, 0.03 m) are flat ground pieces that never block.
- `hedge` (clipped, 0.9 to 1.8 m) and `fence` (close-board timber, 1.0 to 1.8 m, 0.1 m deep) block, along a boundary.
- `garden_bed` (0.3 m, an edging round soil, or round bark mulch when its colour is brown) and `verge` (a strip of
  grass 0.15 m high between the kerb and the sidewalk) block: the dressing guard keeps dressing off open ground, so
  shrubs, trees, lamps and the playground stand on these (`dressing_layout_residential`).
"""

from __future__ import annotations

import procedural_house
from materials_residential import Surfaces  # noqa: F401  (build_scene makes one a scene)
from shapes import Part, box

POST_PITCH_M, POST_W_M = 2.0, 0.1  # a fence's posts
EDGING_M = 0.12  # the stone edging of a garden bed


def house(s, c, surfaces) -> list[Part]:
    """A two-storey detached house with a pitched roof: see `procedural_house.house`."""
    return procedural_house.house(s, c)


def block(s, c, surfaces) -> list[Part]:
    """An apartment block with a flat roof and a parapet: see `procedural_house.block`."""
    return procedural_house.block(s, c)


def lawn(s, c, surfaces) -> list[Part]:
    sx, sy, sz = s
    return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0.0, sz, "lawn")]


def driveway(s, c, surfaces) -> list[Part]:
    sx, sy, sz = s
    return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0.0, sz, "driveway")]


def path(s, c, surfaces) -> list[Part]:
    sx, sy, sz = s
    return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0.0, sz, "paving")]


def hedge(s, c, surfaces) -> list[Part]:
    """A clipped hedge: one block of leaf with rounded edges."""
    sx, sy, sz = s
    return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0.0, sz, "hedge", bevel=0.12)]


def fence(s, c, surfaces) -> list[Part]:
    """A close-board fence: boards between posts every 2 m, a capping rail on top. Its long axis is y."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-0.035, 0.035, -hy, hy, 0.0, sz - 0.12, "fence"), box(-hx, hx, -hy, hy, sz - 0.12, sz, "fence")]
    posts = max(2, round(sy / POST_PITCH_M) + 1)
    for i in range(posts):
        y = -hy + POST_W_M / 2 + i * (sy - POST_W_M) / (posts - 1)
        parts.append(box(-hx, hx, y - POST_W_M / 2, y + POST_W_M / 2, 0.0, sz - 0.12, "fence"))
    return parts


def garden_bed(s, c, surfaces) -> list[Part]:
    """A raised bed: a stone edging `EDGING_M` wide round soil 5 cm under its top, or bark mulch (colour brown)."""
    sx, sy, sz = s
    hx, hy, e = sx / 2, sy / 2, EDGING_M
    fill = "mulch" if c == "brown" else "soil"
    return [box(-hx + e, hx - e, -hy + e, hy - e, 0.0, sz - 0.05, fill),
            box(-hx, hx, -hy, -hy + e, 0.0, sz, "stone"), box(-hx, hx, hy - e, hy, 0.0, sz, "stone"),
            box(-hx, -hx + e, -hy + e, hy - e, 0.0, sz, "stone"), box(hx - e, hx, -hy + e, hy - e, 0.0, sz, "stone")]


def verge(s, c, surfaces) -> list[Part]:
    sx, sy, sz = s
    return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0.0, sz, "lawn")]


BUILDERS = {f.__name__: f for f in (house, block, lawn, driveway, path, hedge, fence, garden_bed, verge)}
