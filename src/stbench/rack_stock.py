"""Where a pallet rack holds stock at floor level, read from the public map (a neutral module: a family may use it, a
system may too).

A warehouse rack row is `bays` bays of `BAY_M` along its length and `levels` levels `LEVEL_M` apart from the floor. The
map gives a surface to the bay-levels that items may rest on and none to the others, and the renderer draws static stock
(pallets of cartons, scenery and never an entity) in exactly the others. So the stock is a
fact of the map, and a question can know where the floor-level stock stands without a frame: a reader that names a
stock pallet as a movable thing on the floor has read the picture right, and `out_of_view` makes each such place a
don't-care place (docs/families/out_of_view.md).

This module restates the map contract; the renderer's own `render/rack_layout.py` (GPL, run in Blender's Python) is not
imported. `tests/test_rack_stock.py` holds the two to the same bay-levels on `warehouse_a`.

Frames: a rack's local frame has its origin at the centre, x to the front and y along the length, bays counted from the
-y end, levels from the floor. A bay holds two pallets of stock, at a quarter of `BAY_M` either side of its middle.
"""

from __future__ import annotations

from .geometry import Point, Rect, to_world
from .mapspec import MapSpec
from .rules import MATCH_M, widening

BAY_M, LEVEL_M, SNAP_M = 2.7, 1.3, 0.1  # a bay's length, the height between levels, how far a surface may sit off both
RACK_KIND = "pallet_rack"  # the map's kind of furniture that is a rack row
SLOT_M = BAY_M / 4  # a stock pallet stands this far either side of its bay's middle, along the rack
PALLET_M = (1.0, 1.16)  # a pallet of stock as drawn: across the rack and along it
STOCK_RADIUS = round(MATCH_M + widening(PALLET_M), 6)  # a place's radius: the matching radius and the pallet's widening

Surface = tuple[float, float, float, float, float]  # cx, cy, sx, sy, z in the rack's local frame
BayLevel = tuple[int, int]  # (bay, level)


def layout(size, surfaces: list[Surface]) -> tuple[int, int, set[BayLevel]]:
    """(bays, levels, free) of a rack of `size` (depth, length, height) whose receptacle surfaces are `surfaces`: `free`
    is the (bay, level) each surface gives. A surface that is not in the middle of a bay (within `SNAP_M` along it)
    and at a level's height (within a tenth of `SNAP_M`) is a ValueError: the map and the renderer would disagree."""
    length = size[1]
    bays, levels = round(length / BAY_M), round(size[2] / LEVEL_M)
    free: set[BayLevel] = set()
    for _cx, cy, _sx, _sy, z in surfaces:
        along = cy + length / 2
        bay, level = round(along / BAY_M - 0.5), round(z / LEVEL_M)
        if (not (0 <= bay < bays and 0 <= level < levels) or abs(along - (bay + 0.5) * BAY_M) > SNAP_M
                or abs(z - level * LEVEL_M) > SNAP_M / 10):
            raise ValueError(f"a rack surface at y {cy}, z {z} is not a bay-level of a {bays} x {levels} rack")
        free.add((bay, level))
    return bays, levels, free


def floor_stock(m: MapSpec) -> list[tuple[Point, float]]:
    """The places of the stock pallets standing on the floor of the map's racks, as (xy, radius), sorted: for each rack
    row and each bay without a floor-level surface, the two pallets at its slots (the bay's middle, `SLOT_M` either side
    along the rack), in the map frame to 4 decimals. The radius is `STOCK_RADIUS`. A map without racks has none."""
    places: list[tuple[Point, float]] = []
    for f in m.furniture:
        if f.kind != RACK_KIND:
            continue
        bays, _, free = layout(f.size, [(*s.rect, s.z) for s in f.surfaces])
        frame = Rect(f.center[0], f.center[1], f.size[0], f.size[1], f.yaw)
        for bay in range(bays):
            if (bay, 0) in free:
                continue
            middle = (bay + 0.5) * BAY_M - f.size[1] / 2
            for slot in (middle - SLOT_M, middle + SLOT_M):
                x, y = to_world(frame, 0.0, slot)
                places.append(((round(x, 4) + 0.0, round(y, 4) + 0.0), STOCK_RADIUS))
    return sorted(places)

