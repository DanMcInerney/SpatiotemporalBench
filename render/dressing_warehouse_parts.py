# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What `dressing_layout_warehouse.layout` is built from: the hall's four wall faces, what stands against them, and the
pallet racks' frames.

Plain Python (no `bpy`), like `dressing_spec` and `rack_layout`. A wall face is addressed by a coordinate along it, and
a `mount` puts a box at a distance range from the face (0 is the face, the cladding stands `CLAD_T` proud, a wall piece
may be 0.12 proud at most: `src/stbench/dressing_zones.py`). `blockers` and `runs` say which stretches of a face
furniture stands against and how high it reaches there, so wall pieces start above it (no piece meets a receptacle
volume). A `Rack` knows its frame (x = front, y = left, z up, as `furniture_warehouse.pallet_rack` draws it) and puts
boxes at the front, the end and the beams in the map frame.

A Dressing's box is (depth along its facing, width across, height) and its yaw is the direction it faces: away from the
wall it hangs on, or out of the rack face it is fixed to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import rack_layout
from dressing_spec import Dressing

# What a tracked thing may stand as high as on each kind of receptacle, so what is built against it starts above (the
# catalog's tallest kind that fits the surface: `dressing_zones.tallest_resting`, which tests/test_dressing_layout_
# warehouse.py checks these against): a pallet on a rack or a staging lane, a tote on a shelf, a drum on a work table.
HEADROOM_M = {"rack": 1.0, "staging lane": 1.0, "shelf": 0.32, "work table": 0.88}
DEFAULT_HEADROOM_M = 1.0
RECEPTACLE_MARGIN_M = 0.05  # a receptacle's volume reaches this far above the tallest kind
STACK_MARGIN_M = 0.05  # a wall piece starts this far above the volume of what stands against its face
WALL_NEAR_M = 0.15  # furniture this close to a face is in the way of what stands on it
EDGE_M = 0.05  # a blocker's stretch is widened by this on both sides

# the rack frame, as `furniture_warehouse.pallet_rack` builds it (tests/test_dressing_layout_warehouse.py reads these
# from that file): upright posts, the beams' front face set in from the rack's front, their height
POST_Y, BEAM_SET_IN, BEAM_H = 0.09, 0.02, 0.12

AABB = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


@dataclass(frozen=True)
class Hall:
    """The inside of the building: the faces of its four walls, and its height."""

    x0: float
    y0: float
    x1: float
    y1: float
    h: float

    @staticmethod
    def of(spec: dict) -> Hall:
        b, t = spec["bounds"], spec["wall_thickness"] / 2
        return Hall(b[0] + t, b[1] + t, b[2] - t, b[3] - t, spec["wall_height"])

    @property
    def faces(self) -> dict[str, Face]:
        return {"S": Face("S", 0, self.y0, 1, self.x0, self.x1), "N": Face("N", 0, self.y1, -1, self.x0, self.x1),
                "W": Face("W", 1, self.x0, 1, self.y0, self.y1), "E": Face("E", 1, self.x1, -1, self.y0, self.y1)}

    def inside(self, x: float, y: float) -> bool:
        return self.x0 < x < self.x1 and self.y0 < y < self.y1


@dataclass(frozen=True)
class Face:
    """A wall face. `axis` 0 runs along x (the south and north walls), 1 along y; `pos` is its coordinate across; `sign`
    is +1 when the room lies on its + side; `lo`..`hi` is its extent along the axis."""

    name: str
    axis: int
    pos: float
    sign: int
    lo: float
    hi: float

    @property
    def yaw_deg(self) -> float:
        return {(0, 1): 90.0, (0, -1): 270.0, (1, 1): 0.0, (1, -1): 180.0}[(self.axis, self.sign)]

    @property
    def normal(self) -> tuple[float, float]:
        return (0.0, float(self.sign)) if self.axis == 0 else (float(self.sign), 0.0)

    def point(self, a: float) -> tuple[float, float]:
        return (a, self.pos) if self.axis == 0 else (self.pos, a)

    def gap(self, box: AABB) -> float:
        """How far the nearest side of the box is from the face (negative when it crosses it)."""
        lo, hi = (box[1], box[3]) if self.axis == 0 else (box[0], box[2])
        return (lo - self.pos) if self.sign > 0 else (self.pos - hi)

    def stretch(self, box: AABB) -> tuple[float, float]:
        return (box[0], box[2]) if self.axis == 0 else (box[1], box[3])


def footprint(f: dict) -> AABB:
    """The axis-aligned box of a furniture piece's footprint (its size is depth along its facing, width across)."""
    c, s = abs(math.cos(f["yaw"])), abs(math.sin(f["yaw"]))
    hx = (f["size"][0] * c + f["size"][1] * s) / 2
    hy = (f["size"][0] * s + f["size"][1] * c) / 2
    return (f["center"][0] - hx, f["center"][1] - hy, f["center"][0] + hx, f["center"][1] + hy)


def reach(f: dict) -> float:
    """The height up to which a furniture piece, or a thing resting on it, is in the way of a wall piece."""
    top = f["size"][2]
    for s in f["surfaces"]:
        top = max(top, s["z"] + HEADROOM_M.get(s["kind"], DEFAULT_HEADROOM_M) + RECEPTACLE_MARGIN_M)
    return top + STACK_MARGIN_M


def blockers(face: Face, furniture: list[dict]) -> list[tuple[float, float, float]]:
    """(from, to, height) for each furniture piece within `WALL_NEAR_M` of the face: where it stands along the face and
    how high what is built against it must start."""
    out = []
    for f in furniture:
        box = footprint(f)
        if face.gap(box) <= WALL_NEAR_M:
            a, b = face.stretch(box)
            out.append((a - EDGE_M, b + EDGE_M, reach(f)))
    return out


def runs(face: Face, held: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """The face cut into stretches of one height: (from, to, the height wall pieces start at), 0 where nothing stands
    against it. Neighbouring stretches of the same height are one."""
    cuts = sorted({face.lo, face.hi, *(min(max(v, face.lo), face.hi) for b in held for v in b[:2])})
    out: list[list[float]] = []
    for a, b in zip(cuts, cuts[1:], strict=False):
        if b - a < 1e-6:
            continue
        top = max((h for lo, hi, h in held if lo <= (a + b) / 2 <= hi), default=0.0)
        if out and abs(out[-1][2] - top) < 1e-9:
            out[-1][1] = b
        else:
            out.append([a, b, top])
    return [(a, b, h) for a, b, h in out]


def clear(face: Face, a0: float, a1: float, z0: float, held: list[tuple[float, float, float]]) -> bool:
    """Whether nothing stands against the face between a0 and a1 up to z0."""
    return not any(lo < a1 and a0 < hi and h > z0 for lo, hi, h in held)


def mount(face: Face, a0: float, a1: float, z0: float, z1: float, lo: float, hi: float, cls: str, recipe: str
          ) -> Dressing:
    """A box on a face between a0 and a1 along it and z0 and z1 up, from `lo` to `hi` out from it (lo may be negative:
    it sinks into the wall). It faces into the room."""
    d, (x, y), (nx, ny) = (lo + hi) / 2, face.point((a0 + a1) / 2), face.normal
    return Dressing(cls, recipe, (x + nx * d, y + ny * d, z0), face.yaw_deg, (hi - lo, a1 - a0, z1 - z0))


def slab(cls: str, recipe: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Dressing:
    """An overhead box between the given extents, its length along x."""
    return Dressing(cls, recipe, ((x0 + x1) / 2, (y0 + y1) / 2, z0), 0.0, (x1 - x0, y1 - y0, z1 - z0))


class Rack:
    """One pallet rack row of the map, in its own frame: x to its front, y to its left, z up, the origin at the base
    centre (`furniture_warehouse.pallet_rack`). `item` puts a box given in that frame into the map frame."""

    def __init__(self, f: dict, hall: Hall):
        self.id, (self.cx, self.cy), self.yaw = f["id"], f["center"], f["yaw"]
        self.sx, self.sy, self.sz = f["size"]
        self.hx, self.hy = self.sx / 2, self.sy / 2
        self.bays, self.levels, self.free = rack_layout.layout(f["size"], [(*s["rect"], s["z"]) for s in f["surfaces"]])
        self.pitch = (self.sy - POST_Y) / self.bays  # between the upright frames, as the builder draws them
        self.open_end = next((e for e in (-1, 1) if hall.inside(*self.world(0.0, e * (self.hy + 0.3)))), 0)

    def world(self, lx: float, ly: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.cx + c * lx - s * ly, self.cy + s * lx + c * ly)

    def post_y(self, i: int) -> float:
        """The y of upright frame i, 0 .. bays."""
        return -self.hy + POST_Y / 2 + i * self.pitch

    def bay_y(self, bay: int) -> float:
        return -self.hy + POST_Y / 2 + (bay + 0.5) * self.pitch

    def bay_number(self, bay: int) -> int:
        """1 for the bay at the open end of the row (the bay nearest the way in), counting inwards."""
        return self.bays - bay if self.open_end > 0 else bay + 1

    def item(self, cls: str, recipe: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
             turn: int = 0) -> Dressing:
        """The box [x0, x1] x [y0, y1] x [z0, z1] of the frame, facing out of the rack's front (`turn` 0), out of its
        -y end (-1) or its +y end (+1)."""
        pos = (*self.world((x0 + x1) / 2, (y0 + y1) / 2), z0)
        size = (x1 - x0, y1 - y0, z1 - z0) if turn == 0 else (y1 - y0, x1 - x0, z1 - z0)
        return Dressing(cls, recipe, pos, math.degrees(self.yaw) + 90.0 * turn, size)
