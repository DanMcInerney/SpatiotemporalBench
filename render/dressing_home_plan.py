# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household map as `dressing_layout_home` needs it: wall faces with their openings, obstacles, rooms.

Plain Python over the map spec as the JSON file holds it (no `bpy`, no `stbench`: this runs in Blender's own Python, and
a test loads `render/` by path). A `Face` is one wall face that looks into a room, in its own frame: `t` runs along it
from the room's corner, `d` is the distance out of the wall (`n`, into the room), `z` is height. `Plan.spans` and
`Plan.depth_free` answer where a slab of a given height and depth fits on a face between the openings and the
furniture, so a layout places trim and fittings where nothing stands. The obstacles are what a piece of dressing may
not cut: the bodies of blocking fixed furniture and the volume above each receptacle surface (`ITEM_CLEAR_M`, more than
the tallest kind that rests on one plus the margin the validity zones keep). Light furniture is no obstacle: it stands,
and is moved to, open floor, and dressing on a wall never reaches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import look
from dressing_spec import Box, Dressing

ITEM_CLEAR_M = 0.50  # above a receptacle surface: the tallest kind that rests on one, a desk lamp's 0.45 m, plus 0.05 m
EPS = 1e-9
FACE_GAP_M = 0.0005  # every wall piece stands this far off its face, so that a coordinate that lands exactly on the
# room's edge (a head over a door, under the ceiling band) is never a sliver outside it by rounding


def aabb(cx: float, cy: float, sx: float, sy: float, yaw: float) -> tuple[float, float, float, float]:
    """xmin, ymin, xmax, ymax of a rectangle (extents sx along its facing `yaw`, sy across) about its centre."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    hx, hy = (sx * c + sy * s) / 2, (sx * s + sy * c) / 2
    return cx - hx, cy - hy, cx + hx, cy + hy


def inside(poly: list[tuple[float, float]], x: float, y: float) -> bool:
    n, hit = len(poly), False
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


@dataclass(frozen=True)
class Furn:
    id: str
    name: str
    kind: str
    mobility: str
    blocks: bool
    center: tuple[float, float]
    size: tuple[float, float, float]
    yaw: float
    surfaces: tuple[tuple[str, tuple[float, float, float, float], float], ...]  # (id, local rect cx cy sx sy, z)
    under_m: float = 0.0  # the open height under it (a table's), 0 for a piece that stands on the floor whole

    @property
    def box(self) -> Box:
        x0, y0, x1, y1 = aabb(*self.center, self.size[0], self.size[1], self.yaw)
        return x0, y0, 0.0, x1, y1, self.size[2]


@dataclass(frozen=True)
class Gap:
    """An opening in a face: `t0..t1` along it, `z0..z1` up it. For a door, where its hinge and latch jambs are and
    whether this face is the room it swings into."""

    t0: float
    t1: float
    z0: float
    z1: float
    kind: str  # "door" | "arch" | "window"
    ref: str
    hinge_t: float | None = None
    latch_t: float | None = None
    enters: bool = False


@dataclass(frozen=True)
class Face:
    region: str
    wall: str
    p: tuple[float, float]  # the start of the face (a corner of the room)
    e: tuple[float, float]  # unit vector along it
    n: tuple[float, float]  # unit vector into the room
    length: float
    gaps: tuple[Gap, ...]
    ceiling: float

    @property
    def yaw_deg(self) -> float:
        return round(math.degrees(math.atan2(self.n[1] + 0.0, self.n[0] + 0.0)), 6)

    def to_world(self, t: float, d: float = 0.0) -> tuple[float, float]:
        return self.p[0] + self.e[0] * t + self.n[0] * d, self.p[1] + self.e[1] * t + self.n[1] * d

    def project(self, box: Box) -> tuple[float, float, float, float]:
        """(t0, t1, d0, d1) of a box's footprint in this face's frame."""
        pts = [(box[0], box[1]), (box[3], box[1]), (box[3], box[4]), (box[0], box[4])]
        ts = [(x - self.p[0]) * self.e[0] + (y - self.p[1]) * self.e[1] for x, y in pts]
        ds = [(x - self.p[0]) * self.n[0] + (y - self.p[1]) * self.n[1] for x, y in pts]
        return min(ts), max(ts), min(ds), max(ds)

    def item(self, cls: str, t: float, width: float, depth: float, z0: float, height: float, standoff: float = 0.0,
             recipe: str | None = None) -> Dressing:
        """A piece on this face centred at `t`, `width` along it, `depth` out of it (from `standoff`), `height` up from
        `z0`: size is (depth, width, height) and the yaw is the way it faces, into the room. The recipe is
        `proc:<class>` unless one is given."""
        x, y = self.to_world(t, standoff + FACE_GAP_M + depth / 2)
        return Dressing(cls, recipe or f"proc:{cls}", (round(x, 4), round(y, 4), round(z0, 4)), self.yaw_deg,
                        (round(depth, 4), round(width, 4), round(height, 4)))


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    role: str  # kitchen | bathroom | bedroom | living | hall
    polygon: tuple[tuple[float, float], ...]
    z1: float
    lights: tuple[tuple[float, float, float], ...]
    furniture: tuple[Furn, ...]

    @property
    def centroid(self) -> tuple[float, float]:
        n = len(self.polygon)
        return sum(p[0] for p in self.polygon) / n, sum(p[1] for p in self.polygon) / n

    def edge_distance(self, x: float, y: float) -> float:
        """Distance from a point of the room to the nearest wall face."""
        best = math.inf
        for a, b in zip(self.polygon, self.polygon[1:] + self.polygon[:1], strict=True):
            dx, dy = b[0] - a[0], b[1] - a[1]
            u = max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy)))
            best = min(best, math.hypot(x - a[0] - u * dx, y - a[1] - u * dy))
        return best


@dataclass
class Plan:
    """The map spec read for placing dressing; `claims` are the boxes of the pieces placed so far that later pieces
    keep clear of (see `claim`)."""

    spec: dict
    name: str = ""
    rooms: list[Room] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    obstacles: list[Box] = field(default_factory=list)
    claims: list[Box] = field(default_factory=list)

    def __post_init__(self) -> None:
        spec = self.spec
        self.name = spec["name"]
        furn = [Furn(f["id"], f["name"], f["kind"], f["mobility"], bool(f.get("blocks", True)),
                     (float(f["center"][0]), float(f["center"][1])), tuple(float(v) for v in f["size"]),
                     float(f["yaw"]), tuple((s["id"], tuple(float(v) for v in s["rect"]), float(s["z"]))
                                            for s in f["surfaces"]), float(f.get("under_m", 0.0)))
                for f in spec["furniture"]]
        for r in spec["regions"]:
            poly = tuple((float(x), float(y)) for x, y in r["polygon"])
            held = tuple(f for f in furn if inside(list(poly), *f.center))
            role = look.get("home", "ROLE", look.kind_of(r))  # what the room is for, by its kind, not by its furniture
            lights = tuple((float(c["pos"][0]), float(c["pos"][1]), float(c["pos"][2]))
                           for c in spec["ceiling_lights"] if c["region"] == r["id"])
            self.rooms.append(Room(r["id"], r["name"], role, poly, float(r["z1"]), lights, held))
        for f in furn:
            if f.mobility != "fixed":
                continue
            if f.blocks:
                self.obstacles.append(f.box)
            for _sid, (lcx, lcy, sx, sy), z in f.surfaces:
                c, s = math.cos(f.yaw), math.sin(f.yaw)
                x0, y0, x1, y1 = aabb(f.center[0] + c * lcx - s * lcy, f.center[1] + s * lcx + c * lcy, sx, sy, f.yaw)
                self.obstacles.append((x0, y0, z, x1, y1, z + ITEM_CLEAR_M))
        self.faces = [fc for room in self.rooms for fc in self._faces(room)]

    # -- faces -------------------------------------------------------------------------------------------------
    def _region_at(self, x: float, y: float) -> str | None:
        return next((r.id for r in self.rooms if inside(list(r.polygon), x, y)), None)

    def _faces(self, room: Room) -> list[Face]:
        """The wall faces of a room: each edge of its polygon that lies on a wall's face."""
        out = []
        poly = list(room.polygon)
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1], strict=True))
        for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length < 1e-6:
                continue
            e = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
            n = (-e[1], e[0]) if area > 0 else (e[1], -e[0])
            for w in self.spec["walls"]:
                wa, wb = w["a"], w["b"]
                wl = math.hypot(wb[0] - wa[0], wb[1] - wa[1])
                u = ((wb[0] - wa[0]) / wl, (wb[1] - wa[1]) / wl)
                nw = (-u[1], u[0])
                half = float(w["thickness"]) / 2
                d = (a[0] - wa[0]) * nw[0] + (a[1] - wa[1]) * nw[1]
                sa = (a[0] - wa[0]) * u[0] + (a[1] - wa[1]) * u[1]
                sb = (b[0] - wa[0]) * u[0] + (b[1] - wa[1]) * u[1]
                parallel = abs(e[0] * u[1] - e[1] * u[0]) < 1e-6
                on_face = abs(abs(d) - half) < 2e-3 and d * (n[0] * nw[0] + n[1] * nw[1]) > 0
                along = min(sa, sb) > -half - 2e-3 and max(sa, sb) < wl + half + 2e-3
                if parallel and on_face and along:
                    gaps = self._gaps(w, u, nw, sa, e[0] * u[0] + e[1] * u[1], length, room.id)
                    out.append(Face(room.id, w["id"], a, e, n, length, gaps, room.z1))
                    break
        return out

    def _gaps(self, w: dict, u, nw, s0: float, direction: float, length: float, region: str) -> tuple[Gap, ...]:
        """The openings of wall `w` that fall on a face starting at wall coordinate `s0` and running `direction`
        (+1 or -1) along it, clipped to the face. Door and window ids are in `ref`."""
        thickness = float(self.spec["wall_thickness"])

        def t_of(s: float) -> float:
            return (s - s0) / direction

        gaps = []
        for o in self.spec["openings"]:
            if o["wall"] != w["id"]:
                continue
            lo, hi = o["at"] - o["width"] / 2, o["at"] + o["width"] / 2
            t0, t1 = sorted((t_of(lo), t_of(hi)))
            if t1 <= 0 or t0 >= length:
                continue
            hinge = latch = None
            enters = False
            if o["type"] == "door":
                hinge_s, latch_s = (lo, hi) if o.get("hinge", "a") == "a" else (hi, lo)
                hinge, latch = t_of(hinge_s), t_of(latch_s)
                sign = 1.0 if o.get("swing", "left") == "left" else -1.0
                cx = w["a"][0] + u[0] * o["at"] + sign * nw[0] * thickness
                cy = w["a"][1] + u[1] * o["at"] + sign * nw[1] * thickness
                enters = self._region_at(cx, cy) == region
            gaps.append(Gap(max(t0, 0.0), min(t1, length), 0.0, float(o["height"]), o["type"], o["id"], hinge, latch,
                            enters))
        for x in self.spec["windows"]:
            if x["wall"] != w["id"]:
                continue
            t0, t1 = sorted((t_of(x["at"] - x["width"] / 2), t_of(x["at"] + x["width"] / 2)))
            if t1 > 0 and t0 < length:
                gaps.append(Gap(max(t0, 0.0), min(t1, length), float(x["sill"]), float(x["sill"] + x["height"]),
                                "window", x["id"]))
        return tuple(sorted(gaps, key=lambda g: g.t0))

    def faces_of(self, room: Room) -> list[Face]:
        return [f for f in self.faces if f.region == room.id]

    # -- what stands where -------------------------------------------------------------------------------------
    def claim(self, d: Dressing) -> Dressing:
        """Keep later pieces clear of `d` (pictures keep off the television, a radiator off the thermostat)."""
        self.claims.append(d.box)
        return d

    def _blocking(self, face: Face, z0: float, z1: float, t0: float | None = None, t1: float | None = None):
        """(t0, t1, d0, d1) of every obstacle and claim in front of the face that overlaps z0..z1 (and t0..t1)."""
        lo, hi = (0.0, face.length) if t0 is None else (max(t0, 0.0), min(t1, face.length))
        for box in (*self.obstacles, *self.claims):
            if not (box[2] < z1 - EPS and box[5] > z0 + EPS):
                continue
            a, b, d0, d1 = face.project(box)
            if d1 <= EPS or not (a < hi - EPS and b > lo + EPS):  # behind the wall, or beside this face
                continue
            yield max(a, 0.0), min(b, face.length), d0, d1

    def depth_free(self, face: Face, t0: float, t1: float, z0: float, z1: float, cap: float) -> float:
        """How far a slab over t0..t1 and z0..z1 can stand out of the face before it meets an obstacle (at most
        `cap`)."""
        limit = cap
        for _a, _b, d0, _d1 in self._blocking(face, z0, z1, t0, t1):
            limit = min(limit, max(d0, 0.0))
        return limit

    def spans(self, face: Face, z0: float, z1: float, depth: float, gap_margin: float = 0.0, edge: float = 0.0,
              keep: float = 0.0) -> list[tuple[float, float]]:
        """The stretches of the face, at least `keep` long, where a slab `depth` deep over z0..z1 meets no opening
        (kept `gap_margin` clear), obstacle or claim, and stays `edge` from the ends."""
        blocked = [(g.t0 - gap_margin, g.t1 + gap_margin) for g in self.gaps_in(face, z0, z1)]
        blocked += [(a, b) for a, b, d0, _d1 in self._blocking(face, z0, z1) if d0 < depth - EPS]
        out, cursor = [], edge
        for a, b in sorted(blocked):
            if a - cursor >= keep - EPS and a - cursor > EPS:
                out.append((cursor, a))
            cursor = max(cursor, b)
        if face.length - edge - cursor >= keep - EPS and face.length - edge - cursor > EPS:
            out.append((cursor, face.length - edge))
        return out

    @staticmethod
    def gaps_in(face: Face, z0: float, z1: float) -> list[Gap]:
        return [g for g in face.gaps if g.z0 < z1 - EPS and g.z1 > z0 + EPS]

    def top_under(self, box: Box) -> float:
        """The highest top of an obstacle under the footprint of `box` (0 when the floor is free)."""
        return max((o[5] for o in self.obstacles if o[0] < box[3] - EPS and o[3] > box[0] + EPS
                    and o[1] < box[4] - EPS and o[4] > box[1] + EPS), default=0.0)

    # -- the ceiling -------------------------------------------------------------------------------------------
    def ceiling_spot(self, room: Room, radius: float, avoid: list[tuple[float, float, float]], inset: float = 0.5,
                     within: tuple[float, float, float, float] | None = None,
                     near: tuple[float, float] | None = None, gap: float = 0.35) -> tuple[float, float]:
        """The point of the room's ceiling nearest `near` (the middle of the room) that is `inset` from every wall face,
        inside the rectangle `within` if one is given, and `gap` clear of each circle in `avoid` (x, y, radius), on a
        5 cm grid; ties go to the smaller x, then y."""
        xs, ys = [p[0] for p in room.polygon], [p[1] for p in room.polygon]
        x0, y0, x1, y1 = within or (min(xs), min(ys), max(xs), max(ys))
        near = near or room.centroid
        best = None
        for ix in range(math.ceil(x0 * 20 - 1e-9), int(x1 * 20 + 1e-9) + 1):
            for iy in range(math.ceil(y0 * 20 - 1e-9), int(y1 * 20 + 1e-9) + 1):
                x, y = ix / 20, iy / 20
                if not inside(list(room.polygon), x, y) or room.edge_distance(x, y) < inset:
                    continue
                if any(math.hypot(x - ax, y - ay) < r + radius + gap for ax, ay, r in avoid):
                    continue
                key = (round(math.hypot(x - near[0], y - near[1]), 3), x, y)
                if best is None or key < best:
                    best = key
        if best is None:
            raise ValueError(f"no ceiling spot of radius {radius} in {room.name}")
        return best[1], best[2]
