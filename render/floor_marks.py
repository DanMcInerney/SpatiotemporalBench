# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What is painted on a warehouse's floor, and where its traffic wears it.

Plain Python (no `bpy`), so a free test can read the plan: `layout(spec)` takes a warehouse map's raw spec (as
`src/stbench/maps/<map>.json` holds it) and returns a `Plan` of axis-aligned rectangles in the map frame, in metres.
`materials_floor` turns it into a material layer on the slab: no geometry, so nothing here has a depth, an index or a
shadow, and nothing can be an entity.
- `lines`: yellow lines 10 cm wide along every rack row's front, 30 cm out from the rack face, and the two edges of
  a 1.2 m pedestrian walkway across the aisle mouths;
- `hatches`: yellow diagonal hatching in front of each dock door, past its staging lane (the lane whose span along
  the wall overlaps the door's, wherever it stands in front of it): the keep-clear apron of the door;
- `tracks`: tyre marks, a pair of 14 cm streaks 1.1 m apart down each aisle, in from each dock door and across the dock;
- `lanes`: the polished bands the traffic wears, 2.2 m wide along the aisles and across the dock.
Everything is read off the map's furniture and regions, a region by its kind (`dock`, `aisle`), never off its name. The
yellow is a flat paint on the floor: a line is thinner than any tracked thing, and the floor's own yellow lane outlines
(the staging lanes) are furniture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1
LINE_W = 0.10
RACK_LINE_OFFSET_M = 0.30  # from the rack's face to the line's centre
WALKWAY_W, WALKWAY_GAP = 1.2, 0.20  # a walkway's width, and how far it stands from the aisle mouths
WALKWAY_INSET_M = 0.6  # from the dock's end walls
CROSS_GAP_M = 1.6  # the cross-dock traffic runs this far south of the walkway
HATCH_DEPTH, HATCH_GAP = 0.70, 0.20  # the apron's depth, and its distance from the end of the staging lane
TRACK_W, TRACK_GAP = 0.14, 1.10  # a tyre streak, and the distance between the two of a pair
DOOR_TRACK_FROM, DOOR_TRACK_LEN = 0.8, 4.0  # a door's streaks start this far from its wall and run this far
AISLE_TRACK_LEAD, AISLE_END_GAP = 3.0, 0.5  # an aisle's streaks start this far before its mouth, stop short of its end
LANE_W = 2.2
DOCK, AISLE = "dock", "aisle"  # the warehouse's region kinds (`look_warehouse.KINDS`)


@dataclass(frozen=True)
class Plan:
    lines: tuple[Rect, ...]
    hatches: tuple[Rect, ...]
    tracks: tuple[Rect, ...]
    lanes: tuple[Rect, ...]


def _strip(along: str, c: float, lo: float, hi: float, width: float) -> Rect:
    """A strip `width` wide centred on the line `c` (a y for a strip along x, an x for one along y), `lo` to `hi`."""
    h = width / 2
    box = (lo, c - h, hi, c + h) if along == "x" else (c - h, lo, c + h, hi)
    return tuple(round(v, 3) for v in box)


def _facing(f: dict) -> tuple[float, float]:
    return math.cos(f["yaw"]), math.sin(f["yaw"])


def _runs_along(face: tuple[float, float]) -> str:
    """The axis a piece that faces `face` stretches along: a piece that faces x stretches along y."""
    return "y" if abs(face[0]) > 0.5 else "x"


def _rack_lines(spec: dict) -> list[Rect]:
    """One line per rack row, parallel to it, in front of its face."""
    out = []
    for f in spec["furniture"]:
        if f["kind"] != "pallet_rack":
            continue
        depth, length, _ = f["size"]
        face = _facing(f)
        along = _runs_along(face)
        cx, cy = (f["center"][0] + face[0] * (depth / 2 + RACK_LINE_OFFSET_M),
                  f["center"][1] + face[1] * (depth / 2 + RACK_LINE_OFFSET_M))
        c, mid = (cx, f["center"][1]) if along == "y" else (cy, f["center"][0])
        out.append(_strip(along, c, mid - length / 2, mid + length / 2, LINE_W))
    return out


def _regions(spec: dict, kind: str) -> list[Rect]:
    """The bounding boxes of the regions of a kind, west to east. A region's kind is its `kind`, else its name (the rule
    of `look.kind_of`, which this module does not import: a kind is no part of what the asset library is built from)."""
    boxes = []
    for r in spec["regions"]:
        if str(r.get("kind", r["name"])) == kind:
            xs, ys = [p[0] for p in r["polygon"]], [p[1] for p in r["polygon"]]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return sorted(boxes)


def _extent(f: dict) -> Rect:
    """The axis-aligned footprint (x0, y0, x1, y1) of a furniture piece (its size is depth along its facing, width
    across)."""
    c, s = abs(math.cos(f["yaw"])), abs(math.sin(f["yaw"]))
    hx, hy = (f["size"][0] * c + f["size"][1] * s) / 2, (f["size"][0] * s + f["size"][1] * c) / 2
    return f["center"][0] - hx, f["center"][1] - hy, f["center"][0] + hx, f["center"][1] + hy


def _lane_end(spec: dict, door: dict, face: tuple[float, float]) -> float:
    """How far from the wall (along the door's facing) the staging lane in front of the door reaches; 0.6 m of floor if
    there is none. A lane is the door's when its span along the wall overlaps the door's, whether it stands on the
    door's axis or off it (a lane may be moved aside from its door, as `warehouse_a`'s first and last are); of those
    the farthest counts."""
    reach, along = 0.6, "xy".index(_runs_along(face))  # along: the axis the door stretches along
    span = _extent(door)
    for s in spec["furniture"]:
        if s["kind"] != "staging_lane":
            continue
        box = _extent(s)
        if min(box[along + 2], span[along + 2]) - max(box[along], span[along]) > 1e-6:
            dx, dy = s["center"][0] - door["center"][0], s["center"][1] - door["center"][1]
            half = (box[3 - along] - box[1 - along]) / 2  # the lane's half length along the door's facing
            reach = max(reach, dx * face[0] + dy * face[1] + half)
    return reach


def _door_marks(spec: dict) -> tuple[list[Rect], list[Rect]]:
    """(the aprons, the streaks) of the dock doors, in the order of their positions."""
    hatches, tracks = [], []
    for door in sorted((f for f in spec["furniture"] if f["kind"] == "dock_door"), key=lambda f: f["center"]):
        face, (cx, cy), width = _facing(door), door["center"], door["size"][1]
        near = _lane_end(spec, door, face) + HATCH_GAP
        mid = (cx + face[0] * (near + HATCH_DEPTH / 2), cy + face[1] * (near + HATCH_DEPTH / 2))
        hx, hy = (width / 2, HATCH_DEPTH / 2) if _runs_along(face) == "x" else (HATCH_DEPTH / 2, width / 2)
        hatches.append(tuple(round(v, 3) for v in (mid[0] - hx, mid[1] - hy, mid[0] + hx, mid[1] + hy)))
        a, b = DOOR_TRACK_FROM, DOOR_TRACK_FROM + DOOR_TRACK_LEN
        for dc in (-TRACK_GAP / 2, TRACK_GAP / 2):  # the fork truck's way in from the door
            if _runs_along(face) == "x":  # the door faces y: the streaks run along y, either side of its axis
                tracks.append(_strip("y", cx + dc, *sorted((cy + face[1] * a, cy + face[1] * b)), TRACK_W))
            else:
                tracks.append(_strip("x", cy + dc, *sorted((cx + face[0] * a, cx + face[0] * b)), TRACK_W))
    return hatches, tracks


def layout(spec: dict) -> Plan | None:
    """The floor plan of a warehouse map; None for a map of another catalog (it has no markings)."""
    if spec.get("catalog") != "warehouse":
        return None
    lines = _rack_lines(spec)
    aisles, docks = _regions(spec, AISLE), _regions(spec, DOCK)
    hatches, tracks = _door_marks(spec)
    lanes: list[Rect] = []
    if docks and aisles:
        dock, mouth = docks[0], min(a[1] for a in aisles)
        y1 = mouth - WALKWAY_GAP
        x0, x1 = dock[0] + WALKWAY_INSET_M, dock[2] - WALKWAY_INSET_M
        lines += [_strip("x", y, x0, x1, LINE_W) for y in (y1 - WALKWAY_W, y1)]
        cross = y1 - WALKWAY_W - CROSS_GAP_M  # the cross-dock traffic runs south of the walkway
        pair = (-TRACK_GAP / 2, TRACK_GAP / 2)
        tracks += [_strip("x", cross + dc, dock[0] + 1.5, dock[2] - 2.0, TRACK_W) for dc in pair]
        lanes.append(_strip("x", cross, dock[0] + 1.0, dock[2] - 1.5, LANE_W))
    for x0, y0, x1, y1 in aisles:
        mid, start = (x0 + x1) / 2, y0 - AISLE_TRACK_LEAD
        tracks += [_strip("y", mid + dc, start, y1 - AISLE_END_GAP, TRACK_W) for dc in (-TRACK_GAP / 2, TRACK_GAP / 2)]
        lanes.append(_strip("y", mid, start, y1 - 0.3, LANE_W))
    return Plan(tuple(lines), tuple(hatches), tuple(tracks), tuple(lanes))
