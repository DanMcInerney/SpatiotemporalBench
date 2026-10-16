# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Which bay-levels of a pallet rack are free surfaces and which hold static stock.

Plain Python (no `bpy`), so the layout can be tested without Blender. A rack is `bays` bays of `BAY_M` along its length
(local y) and `levels` levels `LEVEL_M` apart from the floor. The map file gives a surface to the bay-levels that items
may rest on and none to the others: the rack builder draws stock exactly in the others.
"""

from __future__ import annotations

BAY_M, LEVEL_M, SNAP_M = 2.7, 1.3, 0.1


def layout(size, surfaces) -> tuple[int, int, set[tuple[int, int]]]:
    """(bays, levels, free) of a rack of `size` (depth, length, height) whose surfaces are `(cx, cy, sx, sy, z)` in its
    local frame. `free` holds the (bay, level) of each surface, bays counted from the -y end and levels from the floor.
    A surface that does not sit at the middle of a bay and on a level is an error: the map and the builder disagree."""
    bays, levels = round(size[1] / BAY_M), round(size[2] / LEVEL_M)
    free: set[tuple[int, int]] = set()
    for _cx, cy, _sx, _sy, z in surfaces:
        along = cy + size[1] / 2
        bay, level = int(along // BAY_M), round(z / LEVEL_M)
        if not (0 <= bay < bays and 0 <= level < levels) or abs(along - (bay + 0.5) * BAY_M) > SNAP_M \
                or abs(z - level * LEVEL_M) > SNAP_M / 10:
            raise ValueError(f"a rack surface at y {cy}, z {z} is not a bay-level of a {bays} x {levels} rack")
        free.add((bay, level))
    return bays, levels, free


def stock(size, surfaces) -> list[tuple[int, int]]:
    """The (bay, level) pairs that hold stock: every bay-level without a surface, in bay then level order."""
    bays, levels, free = layout(size, surfaces)
    return [(b, v) for b in range(bays) for v in range(levels) if (b, v) not in free]
