"""The engine-neutral map spec: loader, derived geometry and a self-consistency check.

The file format is documented in `src/stbench/maps/README.md`. Everything here is plain data and 2D geometry, so the
Blender builder, the planner and the validator read one description. A map may be outdoors (`outdoor`: a sun and a sky,
no ceilings) and may name a `depth_unit` (the unit of its depth images in the early releases; the robot's
`DepthSpec` owns it now, and `check_map` still fits the map's diagonal in it); both keys are optional, so the indoor
maps keep their bytes. So are a piece's `under_m` (the height of its open underside) and a region's `kind`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import DEPTH_STEPS, DEPTH_UNITS
from .catalog import CATALOGS
from .geometry import (
    Point,
    Rect,
    corners,
    dist_to_rect,
    point_in_polygon,
    point_in_rect,
    rect_inside_rect,
    rects_overlap,
    to_world,
)
from .map_files import bundled_maps, is_map_path, map_bytes  # noqa: F401  (bundled_maps is re-exported)

INTERIOR_TOL = 1e-4  # shapes are pulled in by this much before an "inside the room" test
DEFAULT_CATALOG = "home"  # the vocabulary of a map that names none (`catalog.CATALOGS`)
DEFAULT_DEPTH_UNIT = "mm"  # the unit of the depth images of a map that names none (`camera.DEPTH_UNITS`)
# The units a map may name: a metre step would round a depth by up to 0.5 m, beyond `rules.DEPTH_TOL_M` (0.1 m), so a
# visible point would read as occluded. `camera.DEPTH_UNITS` also holds "m", for reading a camera file.
MAP_DEPTH_UNITS = ("mm", "cm")
DEFAULT_LIGHT_WATTS = 45.0  # a ceiling light that gives no `watts` (the apartment's fixtures, at 2.45 m)
# The clear floor a robot dock keeps, the one owner of the apron (the docking path and the clutter keep-out import it):
# in front of a `dock` piece's front face, and on each side of it. iRobot's home-base guidance is 4 ft (1.2 m) in front
# and 1.5 ft (0.5 m) at each side; Roborock asks 1.5 m in front.
DOCK_APRON_FRONT_M = 1.2
DOCK_APRON_SIDE_M = 0.5


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    polygon: tuple[Point, ...]
    anchor: Point  # a reference standing point; planner viewpoints start near it
    level: int
    z0: float
    z1: float
    kind: str = ""  # the room's kind word: the map's optional `kind`, else its name


@dataclass(frozen=True)
class Wall:
    id: str
    a: Point
    b: Point
    thickness: float
    height: float
    exterior: bool
    facade: str | None = None  # the style the renderer draws the wall in (a building's face); None: the plain wall

    @property
    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    @property
    def direction(self) -> Point:
        n = self.length
        return ((self.b[0] - self.a[0]) / n, (self.b[1] - self.a[1]) / n)

    @property
    def normal(self) -> Point:
        """Left of the wall direction (rotated +90 degrees)."""
        ux, uy = self.direction
        return (-uy, ux)

    @property
    def yaw(self) -> float:
        ux, uy = self.direction
        return math.atan2(uy, ux)

    def point_at(self, s: float) -> Point:
        ux, uy = self.direction
        return (self.a[0] + ux * s, self.a[1] + uy * s)


@dataclass(frozen=True)
class Opening:
    """A door (with a leaf and an `open` state) or an arch (a doorless opening) cut through a wall."""

    id: str
    type: str  # "door" | "arch"
    wall: str
    at: float  # distance of the opening centre from the wall's `a` end
    width: float
    height: float
    hinge: str  # "a" | "b": which jamb carries the hinge (doors only)
    swing: str  # "left" | "right": side of the wall the open leaf swings to, looking from a to b (doors only)
    interior: bool  # exterior doors (the entrance) never toggle
    initial_open: bool
    between: tuple[str, str]  # region ids, or "outside"
    center: Point
    yaw: float  # of the wall
    swings_into: str | None  # region id on the swing side

    def rect(self, thickness: float) -> Rect:
        return Rect(self.center[0], self.center[1], self.width, thickness, self.yaw)


@dataclass(frozen=True)
class Window:
    id: str
    wall: str
    at: float
    width: float
    sill: float
    height: float
    center: Point
    yaw: float


@dataclass(frozen=True)
class Surface:
    """A rectangle in the furniture's local frame (origin at its centre) at absolute height z."""

    id: str
    kind: str
    rect: tuple[float, float, float, float]  # cx, cy, sx, sy (local)
    z: float


@dataclass(frozen=True)
class Furniture:
    """size = (depth along the facing direction, width across, height); yaw is the direction its front faces."""

    id: str
    name: str
    kind: str
    mobility: str  # "fixed" | "light"
    center: Point
    size: tuple[float, float, float]
    yaw: float
    color: str
    blocks: bool  # False for rugs and mats: they never obstruct the ego or other furniture
    surfaces: tuple[Surface, ...]
    under_m: float = 0.0  # the height of its open underside; 0 is solid to the floor (`geometry.under_obstacles`)

    def footprint(self) -> Rect:
        return Rect(self.center[0], self.center[1], self.size[0], self.size[1], self.yaw)


@dataclass(frozen=True)
class CeilingLight:
    region: str
    pos: tuple[float, float, float]
    watts: float = DEFAULT_LIGHT_WATTS  # of the point light the renderer puts at `pos`


@dataclass(frozen=True)
class Outdoor:
    """The light of an outdoor map: the sun's elevation above the horizon and its rotation about the vertical, in
    degrees, and the strengths of the sun and of the sky."""

    sun_elevation_deg: float
    sun_rotation_deg: float
    sun_strength: float
    sky_strength: float


@dataclass(frozen=True)
class MapSpec:
    name: str
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    wall_thickness: float
    wall_height: float
    regions: tuple[Region, ...]
    walls: tuple[Wall, ...]
    openings: tuple[Opening, ...]
    windows: tuple[Window, ...]
    furniture: tuple[Furniture, ...]
    ceiling_lights: tuple[CeilingLight, ...]
    catalog: str = DEFAULT_CATALOG  # which `catalog.CATALOGS` entry the map's items and movers are drawn from
    depth_unit: str = DEFAULT_DEPTH_UNIT  # kept for the older releases' maps; the robot's `DepthSpec` owns the unit
    outdoor: Outdoor | None = None  # None: indoors (ceilings and lights); else the sun and sky, and no ceilings

    # -- lookups ---------------------------------------------------------------------------------------------
    def region(self, region_id: str) -> Region:
        return next(r for r in self.regions if r.id == region_id)

    def wall(self, wall_id: str) -> Wall:
        return next(w for w in self.walls if w.id == wall_id)

    def doors(self) -> tuple[Opening, ...]:
        return tuple(o for o in self.openings if o.type == "door")

    def toggling_doors(self) -> tuple[Opening, ...]:
        return tuple(o for o in self.doors() if o.interior)

    def region_of(self, xy: Point) -> str | None:
        """The room containing the point, or None in a doorway, in a wall or outside."""
        for r in self.regions:
            if point_in_polygon(xy[0], xy[1], list(r.polygon)):
                return r.id
        return None

    def contains(self, xy: Point) -> bool:
        """Inside a room, or in a door or arch opening (the thin gap between two rooms)."""
        if self.region_of(xy) is not None:
            return True
        return any(point_in_rect(o.rect(self.wall_thickness), xy[0], xy[1], INTERIOR_TOL) for o in self.openings)

    def rect_inside(self, rect: Rect) -> bool:
        """The rectangle (pulled in slightly) lies wholly inside the room holding its centre.

        Rooms are convex (check_map verifies it), so all four corners being inside is enough; this also rejects a
        piece that straddles a wall into the next room.
        """
        room = self.region_of((rect.cx, rect.cy))
        if room is None:
            return False
        poly = list(self.region(room).polygon)
        return all(point_in_polygon(x, y, poly) for x, y in corners(rect, INTERIOR_TOL))

    # -- derived geometry ------------------------------------------------------------------------------------
    def wall_pieces(self) -> list[Rect]:
        """Solid wall boxes with door and arch openings cut out; outer ends extend by half a thickness at corners."""
        pieces: list[Rect] = []
        for w in self.walls:
            cuts = sorted(
                (o.at - o.width / 2, o.at + o.width / 2) for o in self.openings if o.wall == w.id
            )
            spans: list[tuple[float, float]] = []
            cursor = -w.thickness / 2
            for lo, hi in cuts:
                if lo > cursor:
                    spans.append((cursor, lo))
                cursor = max(cursor, hi)
            end = w.length + w.thickness / 2
            if end > cursor:
                spans.append((cursor, end))
            for lo, hi in spans:
                cx, cy = w.point_at((lo + hi) / 2)
                pieces.append(Rect(cx, cy, hi - lo, w.thickness, w.yaw))
        return pieces

    def door_line(self, opening: Opening) -> tuple[Point, Point]:
        """The segment across the opening (jamb to jamb); an ego path crossing it passes through the door."""
        w = self.wall(opening.wall)
        return (w.point_at(opening.at - opening.width / 2), w.point_at(opening.at + opening.width / 2))

    def door_leaf(self, opening: Opening, is_open: bool) -> tuple[Point, Point]:
        """Hinge point and leaf tip. Closed: the leaf spans the opening. Open: it stands perpendicular to the wall."""
        w = self.wall(opening.wall)
        lo, hi = self.door_line(opening)
        hinge, other = (lo, hi) if opening.hinge == "a" else (hi, lo)
        if not is_open:
            return hinge, other
        nx, ny = w.normal
        sign = 1.0 if opening.swing == "left" else -1.0
        return hinge, (hinge[0] + sign * nx * opening.width, hinge[1] + sign * ny * opening.width)

    def door_keepout(self, opening: Opening) -> tuple[Point, float]:
        """Furniture stays at least this far from the centre of an interior opening, so passages never close."""
        return opening.center, opening.width / 2 + 0.3

    def furniture_by_id(self) -> dict[str, Furniture]:
        return {f.id: f for f in self.furniture}


def surface_world_rect(furniture_center: Point, furniture_yaw: float, surface: Surface) -> Rect:
    """The surface rectangle in the map frame for a furniture piece at the given pose."""
    frame = Rect(furniture_center[0], furniture_center[1], 0.0, 0.0, furniture_yaw)
    wx, wy = to_world(frame, surface.rect[0], surface.rect[1])
    return Rect(wx, wy, surface.rect[2], surface.rect[3], furniture_yaw)


# -- loading -----------------------------------------------------------------------------------------------------
def _pt(v: list[float]) -> Point:
    return (float(v[0]), float(v[1]))


def load_map(name_or_path: str | Path) -> MapSpec:
    """Load a map by name (e.g. "apartment_a": a private map in a `STBENCH_MAPS` folder, or a bundled one, as
    `map_files.map_bytes` finds it) or a map spec file by path. A map loaded by name must carry that name."""
    data = map_bytes(name_or_path)
    m = parse_map(json.loads(data.decode("utf-8")))
    if not is_map_path(name_or_path) and m.name != str(name_or_path):
        raise ValueError(f"the map file {name_or_path}.json is named {m.name!r}: a map is found by its own name")
    return m


def parse_map(raw: dict) -> MapSpec:
    """The map a dict describes. It reads only the keys it names: an earlier map file may carry a `schema`."""
    regions = tuple(
        Region(r["id"], r["name"], tuple(_pt(p) for p in r["polygon"]), _pt(r["anchor"]), int(r["level"]),
               float(r["z0"]), float(r["z1"]), str(r.get("kind", r["name"])))
        for r in raw["regions"]
    )
    walls = tuple(
        Wall(w["id"], _pt(w["a"]), _pt(w["b"]), float(w["thickness"]), float(w["height"]), bool(w["exterior"]),
             w.get("facade"))
        for w in raw["walls"]
    )
    by_wall = {w.id: w for w in walls}
    thickness = float(raw["wall_thickness"])

    def region_at(p: Point) -> str | None:
        return next((r.id for r in regions if point_in_polygon(p[0], p[1], list(r.polygon))), None)

    openings = []
    for o in raw["openings"]:
        w = by_wall[o["wall"]]
        center = w.point_at(o["at"])
        swing = o.get("swing", "left")
        into = None
        if o["type"] == "door":
            sign = 1.0 if swing == "left" else -1.0
            nx, ny = w.normal
            into = region_at((center[0] + sign * nx * thickness, center[1] + sign * ny * thickness))
        openings.append(
            Opening(
                o["id"], o["type"], o["wall"], float(o["at"]), float(o["width"]), float(o["height"]),
                o.get("hinge", "a"), swing, bool(o.get("interior", True)), bool(o.get("initial_open", False)),
                (o["between"][0], o["between"][1]), center, w.yaw, into,
            )
        )
    windows = []
    for x in raw["windows"]:
        w = by_wall[x["wall"]]
        windows.append(
            Window(x["id"], x["wall"], float(x["at"]), float(x["width"]), float(x["sill"]), float(x["height"]),
                   w.point_at(x["at"]), w.yaw)
        )
    furniture = tuple(
        Furniture(
            f["id"], f["name"], f["kind"], f["mobility"], _pt(f["center"]),
            (float(f["size"][0]), float(f["size"][1]), float(f["size"][2])), float(f["yaw"]), f["color"],
            bool(f.get("blocks", True)),
            tuple(Surface(s["id"], s["kind"], tuple(float(v) for v in s["rect"]), float(s["z"]))
                  for s in f["surfaces"]),
            float(f.get("under_m", 0.0)),
        )
        for f in raw["furniture"]
    )
    lights = tuple(CeilingLight(c["region"], (float(c["pos"][0]), float(c["pos"][1]), float(c["pos"][2])),
                                float(c.get("watts", DEFAULT_LIGHT_WATTS)))
                   for c in raw["ceiling_lights"])
    outdoor = raw.get("outdoor")
    return MapSpec(
        raw["name"], tuple(float(v) for v in raw["bounds"]), thickness, float(raw["wall_height"]), regions,
        walls, tuple(openings), tuple(windows), furniture, lights, str(raw.get("catalog", DEFAULT_CATALOG)),
        str(raw.get("depth_unit", DEFAULT_DEPTH_UNIT)),
        None if outdoor is None else Outdoor(float(outdoor["sun_elevation_deg"]), float(outdoor["sun_rotation_deg"]),
                                             float(outdoor["sun_strength"]), float(outdoor["sky_strength"])),
    )


# -- self-check --------------------------------------------------------------------------------------------------
def check_map(m: MapSpec) -> list[str]:
    """Return a list of problems with the spec itself (empty when consistent). Free-space reachability is in route."""
    problems: list[str] = []
    if m.catalog not in CATALOGS:
        problems.append(f"unknown catalog {m.catalog!r}; known: {', '.join(CATALOGS)}")
    if m.depth_unit not in MAP_DEPTH_UNITS:
        problems.append(f"unsupported depth unit {m.depth_unit!r}; a map names one of: {', '.join(MAP_DEPTH_UNITS)}")
    elif (reach := _diagonal(m)) > (limit := DEPTH_STEPS * DEPTH_UNITS[m.depth_unit]):
        problems.append(f"the map's 3D diagonal {reach:.1f} m is past the {limit:.3f} m that a depth image in "
                        f"{m.depth_unit} holds")
    ids = [r.id for r in m.regions] + [w.id for w in m.walls] + [o.id for o in m.openings] + \
        [x.id for x in m.windows] + [f.id for f in m.furniture]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate id {dup}")
    names = [r.name for r in m.regions]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"duplicate region name {dup}")
    for r in m.regions:
        if not point_in_polygon(r.anchor[0], r.anchor[1], list(r.polygon)):
            problems.append(f"anchor of {r.id} lies outside the region")
        if not _convex(r.polygon):
            problems.append(f"region {r.id} is not convex")
    for o in m.openings:
        w = m.wall(o.wall)
        if o.at - o.width / 2 < 0 or o.at + o.width / 2 > w.length:
            problems.append(f"opening {o.id} does not fit in wall {w.id}")
        for rid in o.between:
            if rid != "outside" and all(r.id != rid for r in m.regions):
                problems.append(f"opening {o.id} joins unknown region {rid}")
        if o.type == "door" and o.swings_into is None:
            problems.append(f"door {o.id} swings into no region")
        if o.type == "door" and not 0.8 <= o.width <= 0.95:
            problems.append(f"door {o.id} width {o.width} outside 0.8-0.95 m")
    for x in m.windows:
        w = m.wall(x.wall)
        if not w.exterior:
            problems.append(f"window {x.id} is not on an exterior wall")
        if x.at - x.width / 2 < 0 or x.at + x.width / 2 > w.length or x.sill + x.height > w.height:
            problems.append(f"window {x.id} does not fit in wall {w.id}")
        for o in m.openings:
            if o.wall == x.wall and abs(o.at - x.at) < (o.width + x.width) / 2:
                problems.append(f"window {x.id} overlaps opening {o.id}")
    for light in m.ceiling_lights:
        if all(r.id != light.region for r in m.regions):
            problems.append(f"ceiling light in unknown region {light.region}")
    if m.outdoor is not None and m.ceiling_lights:
        problems.append("an outdoor map has no ceilings, so it has no ceiling lights")
    for f in m.furniture:
        fp = f.footprint()
        if not m.rect_inside(fp):
            problems.append(f"furniture {f.id} ({f.name}) leaves the rooms")
        if not 0 <= f.under_m < f.size[2]:
            problems.append(f"furniture {f.id} ({f.name}) has under_m {f.under_m}; it must be at least 0 and below "
                            f"its height {f.size[2]}")
        for s in f.surfaces:
            sr = surface_world_rect(f.center, f.yaw, s)
            if not rect_inside_rect(sr, fp, 1e-6):
                problems.append(f"surface {s.id} of {f.id} leaves the furniture footprint")
            if s.z > f.size[2] + 1e-6:
                problems.append(f"surface {s.id} of {f.id} is above the furniture")
        if f.mobility == "fixed" and f.blocks:
            for o in (o for o in m.openings if o.interior):
                c, radius = m.door_keepout(o)
                if _rect_near_point(fp, c, radius):
                    problems.append(f"furniture {f.id} blocks door {o.id}")
    blockers = [f for f in m.furniture if f.blocks]
    for i, a in enumerate(blockers):
        for b in blockers[i + 1:]:
            if rects_overlap(a.footprint(), b.footprint(), margin=0.0):
                problems.append(f"furniture {a.id} and {b.id} overlap")
    return problems


def _diagonal(m: MapSpec) -> float:
    """The longest straight line inside the map's box (its bounds, from the floor to the wall height), metres: the
    farthest anything in the map can be from a camera in it."""
    x0, y0, x1, y1 = m.bounds
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + m.wall_height**2)


def _rect_near_point(rect: Rect, p: Point, radius: float) -> bool:
    return float(dist_to_rect(np.array([p[0]]), np.array([p[1]]), rect)[0]) < radius


def _convex(poly: tuple[Point, ...]) -> bool:
    n = len(poly)
    signs = set()
    for i in range(n):
        (x1, y1), (x2, y2), (x3, y3) = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if abs(cross) > 1e-12:
            signs.add(cross > 0)
    return len(signs) <= 1
