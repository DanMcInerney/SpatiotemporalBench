# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What a map's dressing is: render-only detail that is never an entity.

Plain Python, no `bpy`, so a test can read a layout without Blender (as `rack_layout.py`). A map's layout module
(`dressing_layout_<catalog>.py`) has one function, `layout(spec: dict) -> list[Dressing]`: the map spec as
`src/stbench/maps/<map>.json` holds it, in, and every dressing object of the map out, in a fixed order. It depends on
the map alone (any randomness is seeded from the map's name), so every episode of a map has the same dressing.

A `Dressing` names its class (registered in `src/stbench/scenery.py`'s dressing registry, which says whether the class
can be mistaken for a tracked kind), how it is built (`recipe`: a procedural builder's name, or `asset:<id>` for an
asset listed in the manifest), where it stands (`pos`, the centre of its base in the map frame), its yaw and its size
(the extents of its box before the yaw). `box` is the axis-aligned box it occupies after the yaw, which the validity
test holds clear of every volume a question or the robot depends on. Phase 1 dressing (`phase` 1) changes only the
render; phase 2 edits the world and needs a new generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Box = tuple[float, float, float, float, float, float]  # xmin, ymin, zmin, xmax, ymax, zmax


@dataclass(frozen=True)
class Dressing:
    cls: str
    recipe: str
    pos: tuple[float, float, float]
    yaw_deg: float
    size: tuple[float, float, float]
    phase: int = 1

    @property
    def box(self) -> Box:
        """The axis-aligned box the object occupies after its yaw about its base centre."""
        a = math.radians(self.yaw_deg)
        c, s = abs(math.cos(a)), abs(math.sin(a))
        hx = (self.size[0] * c + self.size[1] * s) / 2
        hy = (self.size[0] * s + self.size[1] * c) / 2
        x, y, z = self.pos
        return (x - hx, y - hy, z, x + hx, y + hy, z + self.size[2])


def overlaps(a: Box, b: Box, eps: float = 1e-9) -> bool:
    """Whether two boxes share volume (touching faces do not count)."""
    return all(a[i] < b[i + 3] - eps and b[i] < a[i + 3] - eps for i in range(3))
