# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The house's fixed furniture that the apartment has no like of: the staircase of `house_floor`.

Runs inside Blender's own Python and is never imported by `stbench`. Like the household builders (`furniture.py`, which
registers this one), `stairs` takes the entity size (depth along the flight, width, height), its colour attribute and
the heights of its receptacle surfaces (none) and returns parts in the entity's frame: origin at the base centre,
x = front (the bottom step), y = left, z up. The flight climbs towards -x; the wall is on the left (+y), the open side
on the right (-y), so the map stands the piece against a wall on its left.

The flight is a boxed one, solid to the floor (the map gives it `under_m` 0: the vacuum sees its panelled side from 8 cm
up and never drives under it): 13 oak treads, 25 cm going and 17.9 cm rise, and a top landing 15 cm deep at the
ceiling's height, where the 14th riser ends (the floor above is not modelled: a stated simplification). White risers
under the oak treads and a closed string along the wall; on the open side a newel post at the foot, a baluster to
each of the first six treads and an oak handrail that stops under the bulkhead, and above it a white guard wall that
climbs with the flight to the ceiling. The last treads run under a bulkhead 0.3 m deep, so the stairs read as
climbing through a stairwell. The panelling of the open side has a plinth and a cupboard door with its seam, which no
robot opens. Every part stays in the entity's box (`tests/test_house_floor_blender.py`).
"""

from __future__ import annotations

import math

from shapes import Part, box

RISER_M = 0.18  # about: the flight has as many risers as make this rise to the ceiling
TREAD_T, LANDING_M = 0.04, 0.15  # an oak tread's thickness; the top landing's depth
SKIN_M, PANEL_T, PLINTH_M = 0.03, 0.03, 0.07  # the open side's panelling stands this far in from the box, the guard
# wall is this thick, and a plinth this high closes the floor under the panelling
BULKHEAD_M, BULKHEAD_LEN_M = 0.30, 1.30  # the box under the ceiling over the top steps: its depth and its length
RAIL_ABOVE_M = 0.90  # the handrail over the nosing line
RAIL_END_X = 0.20  # where the handrail ends, under the bulkhead's underside, and the guard wall begins
BALUSTERS = 6  # the first treads' balusters


def stairs(s, c, zs):
    """A boxed staircase of `s` = (depth along the flight, width, height), the bottom step at +x and the wall at +y."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    n = round(sz / RISER_M)  # risers; one fewer treads, the last riser ends on the landing
    rise = sz / n
    going = (sx - LANDING_M) / (n - 1)
    slope = rise / going
    theta = math.atan(slope)
    side = -hy + SKIN_M  # the open side's panelling face

    def front(i: int) -> float:  # the front edge of tread i (1 is the bottom step)
        return hx - (i - 1) * going

    def nosing(x: float) -> float:  # the height of the line through the treads' front edges, at x
        return rise + (hx - x) * slope

    parts: list[Part] = [box(-hx, hx, -hy, side, 0.0, PLINTH_M, c + "*0.85")]  # the plinth under the open side
    for i in range(1, n):
        x1, z = front(i), i * rise
        parts += [box(x1 - going, x1, side, hy, 0.0, z - TREAD_T, c),
                  box(x1 - going, x1, side, hy, z - TREAD_T, z, "oak")]
    top = -hx + LANDING_M
    parts += [box(-hx, top, side, hy, 0.0, sz - TREAD_T, c), box(-hx, top, side, hy, sz - TREAD_T, sz, "oak")]

    # the closed string along the wall: a board on the pitch, a hand above the treads
    length, mid = 3.7, ((hx + top) / 2, (rise + sz) / 2 - 0.06)
    parts.append(Part("box", (mid[0], hy - 0.01, mid[1]), (length, 0.02, 0.26), (0.0, theta, 0.0), c))

    # the open side: the newel post at the foot, balusters on the first treads, the handrail on top of them
    post = hx - 0.055
    parts += [box(post - 0.035, post + 0.035, -hy + 0.02, -hy + 0.09, 0.0, 1.2, "oak"),
              box(post - 0.045, post + 0.045, -hy + 0.015, -hy + 0.095, 1.2, 1.23, "oak")]
    rail_y = -hy + 0.055
    for i in range(1, BALUSTERS + 1):
        x = front(i) - going / 2
        parts.append(box(x - 0.0125, x + 0.0125, rail_y - 0.0125, rail_y + 0.0125, i * rise,
                         nosing(x) + RAIL_ABOVE_M - 0.02, c))
    x_end = RAIL_END_X
    z0, z1 = nosing(post) + RAIL_ABOVE_M, nosing(x_end) + RAIL_ABOVE_M
    reach = math.hypot(post - x_end, z0 - z1)
    parts.append(Part("box", ((post + x_end) / 2, rail_y, (z0 + z1) / 2), (reach, 0.05, 0.04), (0.0, theta, 0.0),
                      "oak"))
    lower = rise * int((hx - x_end) / going + 1e-6)  # the lower of the two treads the post straddles
    parts.append(box(x_end - 0.035, x_end + 0.035, -hy + 0.02, -hy + 0.09, lower, z1 + 0.03, "oak"))

    # above the handrail the open side is a guard wall from each tread to the ceiling
    for i in range(1, n):
        a, b = front(i) - going, min(front(i), x_end)
        if a < x_end - 1e-6:
            parts.append(box(a, b, -hy + 0.012, -hy + 0.012 + PANEL_T, i * rise, sz, c))
    parts.append(box(-hx, top, -hy + 0.012, -hy + 0.012 + PANEL_T, 0.0, sz, c))
    parts.append(box(-hx, -hx + BULKHEAD_LEN_M, -hy, hy, sz - BULKHEAD_M, sz, c + "*0.97"))

    # the cupboard under the flight: a door in the panelling, a dark seam round it, a bar handle by its closing edge
    d0, d1, dz0, dz1 = -1.19, 0.09, 0.07, 1.0
    parts += [box(d0 - 0.005, d1 + 0.005, -hy + 0.024, -hy + SKIN_M, dz0 - 0.005, dz1 + 0.005, "dark"),
              box(d0, d1, -hy + 0.012, -hy + SKIN_M, dz0, dz1, c + "*0.96"),
              box(d1 - 0.06, d1 - 0.045, -hy, -hy + 0.012, 0.5, 0.74, "steel")]
    return parts


BUILDERS = {"stairs": stairs}
