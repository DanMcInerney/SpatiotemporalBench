"""Where Phase 1 dressing may not stand and where it may: the free proof that a
render-only object cannot touch the timeline.

Dressing is geometry the render adds that is no entity (`scenery.DRESSING`). It leaves the worlds, the routes and every
gold alone only if it stays out of every volume that something tracked, or a robot, can occupy, and so only ever
occludes what lies behind it. This is plain Python over the map spec (the dict that a
`render/dressing_layout_<catalog>.py` module's `layout(spec)` takes, or a parsed `MapSpec`) and the catalog, so a
test can call it without Blender. Boxes are `(xmin, ymin, zmin, xmax, ymax, zmax)` in the map frame, as
`render/dressing_spec.Dressing.box`, and `dressing_boxes` has the arithmetic on them.

`forbidden` lists the volumes a dressing box may not meet, each with a reason, and where each comes from:
- **receptacles**: every surface rect of a fixed furniture piece, from its height up to the tallest kind of the catalog
  that fits on it, tracked or clutter (a clutter kind whose places name the surface's kind), plus
  `RECEPTACLE_MARGIN_M`. `placement._try_slot` rejects a kind whose footprint plus `EDGE_MARGIN` on each side exceeds
  the rect, and any surface takes any kind (`placement.find_spot`: 70% typical, 30% anywhere);
- **keep-outs** (`dressing_robots.keep_out_zones`): the places a robot's manoeuvre uses, by piece kind (a bay's
  footprint and a charger's apron, up to `zones.TOP_M`), where a render-only thing would show the place blocked in
  every case;
- **robots** (`dressing_robots`): for a catalog with robots planned on its maps, the swept body of each of them and the
  open underside of each fixed piece one passes under. A catalog with none planned has instead **open floor the camera
  moves over**: every point of a room at least `WALL_PROUD_MAX_M` from a wall face or a blocking fixed footprint, from
  `FLUSH_M` above the floor to the highest camera height of the retired profiles that fits under the map's ceiling,
  plus a margin (`RETIRED_CAMERA_HEIGHTS_M`: 1.5 m; the city's 12 m drone flies over no indoor map), where a camera
  kept `changes.WORLD_CLEARANCE_M` = 0.25 m from every wall piece and blocking footprint leaves a piece at most 0.12 m
  proud 0.13 m (`EGO_BUFFER_M`);
- **open floor light furniture and movers stand and drive on**: the same floor up to the tallest light piece, mover or
  floor clutter kind plus `HEIGHT_MARGIN_M`. `changes._shift_valid` lets a light piece stand anywhere in its room clear
  of blocking footprints, a mover drives the free grid of `route.FreeSpace` at its own clearance (their motion model is
  retired until M2's movers lane), and floor clutter lies anywhere on the open floor;
- **door swings**: the square the leaf sweeps from closed to open on its swing side, from the hinge
  (`MapSpec.door_leaf`), as high as the door (the generator keeps the entrance shut and every interior door open).
  The render's leaf is a 4 cm slab on that line, flush against the hinge jamb when open and 2 cm behind the wall's
  centre line when shut (`render/build_scene.Door`), so it stays inside the square and the wall;
- **openings** (doors and arches): the cut through the wall and `WALL_PROUD_MAX_M` before it on both sides;
- **window glass**: the window rectangle through the wall and `WALL_PROUD_MAX_M` before it on both sides.

`allowed_phase1` lists the only places a Phase 1 box may be, and `covered` asks whether a box lies wholly inside them:
- `wall_bands`: at most `WALL_PROUD_MAX_M` proud of a wall face that looks into a room;
- `ceiling_bands`: the slab under a room's ceiling, `CEILING_BAND_M` deep;
- `structure_band`: building structure above `STRUCTURE_FROM_M` (a tall hall's roof and services; none in the house);
- `footprint_bands`: the column over a blocking fixed piece's footprint, from the floor to the ceiling, and at most
  `WALL_PROUD_MAX_M` proud of its sides. Nothing tracked can be there (`route.blocking_rects` keeps everything out
  of it) except in its receptacle volumes, which stay forbidden, and, under a piece a planned robot passes under, in
  the open underside `dressing_robots` forbids: so the stove top, the fridge top, a cabinet run over the counters and
  a guard on a rack upright are in, and a thing on a shelf or a box in the knee space of a desk is not;
- `flush_floor_bands`: a plate or strip no thicker than `FLUSH_M` on a room's floor.
`DressingGuard` bundles both lists, for a layout test. It lets trim (at most `dressing_robots.TRIM_PROUD_M` proud of a
wall) and a radiator high enough (`dressing_robots.EXEMPT_ABOVE_M`) stand in a swept body.

Where an axis-aligned box would be too permissive (an allowed band, or a footprint that leaves open floor), a wall,
region or footprint that is not turned a quarter turn is an error: both maps are rectilinear, and a map that is not
needs this module extended. Where it would be too strict (a forbidden volume) the box of the turned shape stands in.
"""

from __future__ import annotations

import math

import numpy as np

from .catalog import CATALOGS, Catalog, Kind, for_map, place_of
from .changes import WORLD_CLEARANCE_M
from .dressing_boxes import EPS, FLUSH_M, WALL_PROUD_MAX_M, Box, Zone, aabb, covered, open_floor, rooms, square
from .dressing_robots import (
    EXEMPT_ABOVE_M,
    TRIM_PROUD_M,
    TRIM_SLACK_M,
    keep_out_zones,
    planned_robots,
    swept_zones,
    under_zones,
)
from .geometry import Rect
from .mapspec import MapSpec, parse_map, surface_world_rect
from .placement import EDGE_MARGIN

EGO_BUFFER_M = round(WORLD_CLEARANCE_M - WALL_PROUD_MAX_M, 6)  # what is left of the camera's clearance: 0.13 m
# The camera heights of the profiles `robots` replaced (human 1.5 m, floor 0.12 m, drone 12 m), kept for the catalogs
# that have no robot planned (`dressing_robots.PLANNED_ROBOTS`) until their lanes add theirs.
RETIRED_CAMERA_HEIGHTS_M = (1.5, 0.12, 12.0)
RECEPTACLE_MARGIN_M = 0.05  # a receptacle's volume reaches this far above the tallest kind that fits
WALL_SINK_M = 0.02  # a wall piece may sink into its wall, and a ceiling piece into the ceiling, by this much
CEILING_BAND_M = 0.5  # a ceiling piece hangs down at most this far
STRUCTURE_FROM_M = 5.0  # building structure (a tall hall's roof, services and lights) stands above this height
HEIGHT_MARGIN_M = 0.05  # added to the tallest light piece, mover or floor clutter kind
ANGLE_STEPS = 901  # of the quarter turn scanned to see whether a kind fits a surface at some angle


def _map(spec: dict | MapSpec) -> MapSpec:
    return spec if isinstance(spec, MapSpec) else parse_map(spec)


def _catalog(m: MapSpec, catalog: Catalog | str | None) -> Catalog:
    return for_map(m) if catalog is None else CATALOGS[catalog] if isinstance(catalog, str) else catalog


# -- receptacle volumes --------------------------------------------------------------------------------------------
def fits(kind: Kind, sx: float, sy: float) -> bool:
    """Whether `placement._try_slot` can put a thing of the kind on a surface of that size (along the surface's own
    axes): at some angle, or at a quarter turn for a `square` kind, with both extents and `EDGE_MARGIN` on each side
    inside it."""
    w, d = kind.size[0], kind.size[1]
    a = np.array([0.0, math.pi / 2]) if getattr(kind, "square", False) else np.linspace(0.0, math.pi / 2, ANGLE_STEPS)
    ex, ey = w * np.cos(a) + d * np.sin(a), w * np.sin(a) + d * np.cos(a)
    return bool(np.any((ex <= sx - 2 * EDGE_MARGIN + 1e-9) & (ey <= sy - 2 * EDGE_MARGIN + 1e-9)))


def tallest_resting(catalog: Catalog, sx: float, sy: float, surface_kind: str | None = None) -> float:
    """The height of the tallest kind of the catalog that can rest on a surface of that size (0 when none can): a
    tracked kind, or a clutter kind whose places name the surface's kind (`surface_kind`, None for no clutter)."""
    tall = max((k.size[2] for k in catalog.kinds.values() if fits(k, sx, sy)), default=0.0)
    clutter = (k.size[2] for k in catalog.clutter if surface_kind in {place_of(p)[0] for p in k.places}
               and fits(k, sx, sy))
    return max([tall, *clutter])


def receptacle_volumes(m: MapSpec, catalog: Catalog) -> list[tuple[str, str, Box]]:
    """(furniture id, surface id, volume) per surface of each fixed furniture piece (`placement.build_slots`: light
    furniture offers none): its rect from its height up to the tallest kind that fits there plus `RECEPTACLE_MARGIN_M`.
    A surface nothing fits on has no volume."""
    out = []
    for f in m.furniture:
        for s in (s for s in f.surfaces if f.mobility == "fixed"):
            rect = surface_world_rect(f.center, f.yaw, s)
            top = tallest_resting(catalog, rect.sx, rect.sy, s.kind)
            if top:
                x0, y0, x1, y1 = aabb(rect)
                out.append((f.id, s.id, (x0, y0, s.z, x1, y1, s.z + top + RECEPTACLE_MARGIN_M)))
    return out


def receptacle_zones(m: MapSpec, catalog: Catalog) -> list[Zone]:
    return [(f"receptacle {fid}/{sid} (up to {v[5]:.2f} m)", v) for fid, sid, v in receptacle_volumes(m, catalog)]


# -- open floor ----------------------------------------------------------------------------------------------------
def _floor_zones(m: MapSpec, catalog: Catalog) -> list[Zone]:
    floor = open_floor(m)
    heavy = [f.size[2] for f in m.furniture if f.mobility == "light"]
    heavy += [k.size[2] for k in catalog.mover_kinds.values()]
    heavy += [k.size[2] for k in catalog.clutter if k.on_floor]
    top_heavy = max(heavy, default=0.0) + HEIGHT_MARGIN_M
    out: list[Zone] = []
    if not planned_robots(catalog.name):
        # every camera height that fits under the map's ceiling (a 12 m drone flies over no apartment or warehouse)
        top_ego = max(h for h in RETIRED_CAMERA_HEIGHTS_M if h < m.wall_height) + EGO_BUFFER_M
        out += [(f"open floor the camera moves over (up to {top_ego:.2f} m)",
                 (r[0], r[1], FLUSH_M, r[2], r[3], top_ego)) for r in floor]
    return out + [(f"open floor light furniture and movers stand and drive on (up to {top_heavy:.2f} m)",
                   (r[0], r[1], FLUSH_M, r[2], r[3], top_heavy)) for r in floor]


# -- doors and windows ---------------------------------------------------------------------------------------------
def _through_wall(m: MapSpec, wall: str, centre: tuple[float, float], width: float, yaw: float, z0: float, z1: float
                  ) -> Box:
    """The box of a cut `width` wide through a wall at `centre`, from `WALL_PROUD_MAX_M` in front to the same behind."""
    half = m.wall(wall).thickness / 2 + WALL_PROUD_MAX_M
    x0, y0, x1, y1 = aabb(Rect(centre[0], centre[1], width, 2 * half, yaw))
    return (x0, y0, z0, x1, y1, z1)


def door_zones(m: MapSpec) -> list[Zone]:
    out = []
    for o in m.openings:
        out.append((f"{o.type} opening {o.id}", _through_wall(m, o.wall, o.center, o.width, o.yaw, 0.0, o.height)))
        if o.type != "door":
            continue
        hinge, tip = m.door_leaf(o, True)
        _, closed = m.door_leaf(o, False)
        pts = [hinge, closed, tip, (closed[0] + tip[0] - hinge[0], closed[1] + tip[1] - hinge[1])]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append((f"swing of door {o.id}", (min(xs), min(ys), 0.0, max(xs), max(ys), o.height)))
    return out


def window_zones(m: MapSpec) -> list[Zone]:
    return [(f"glass of window {w.id}", _through_wall(m, w.wall, w.center, w.width, w.yaw, w.sill, w.sill + w.height))
            for w in m.windows]


def forbidden(spec: dict | MapSpec, catalog: Catalog | str | None = None) -> list[Zone]:
    """Every volume a Phase 1 dressing box may not meet, each with the reason: receptacles, the keep-outs of bays and
    chargers, the planned robots' swept bodies and the undersides they pass under, open floor (the camera part, for a
    catalog with no robot planned, then the light furniture, mover and clutter part), door swings and openings, and
    window glass. The catalog defaults to the map's."""
    m = _map(spec)
    cat = _catalog(m, catalog)
    return [*receptacle_zones(m, cat), *keep_out_zones(m), *swept_zones(m, cat.name), *under_zones(m, cat.name),
            *_floor_zones(m, cat), *door_zones(m), *window_zones(m)]


# -- where dressing may stand ----------------------------------------------------------------------------------------
def _faces(m: MapSpec, piece: Rect, z0: float, proud: float, what: str) -> list[Zone]:
    """The strips along the faces of a wall piece that look into a room, from `z0` to the ceiling."""
    x0, y0, x1, y1 = square(piece, what)
    along_x = abs(math.cos(piece.yaw)) > 0.5
    out = []
    for face, sign in ((y0, -1), (y1, 1)) if along_x else ((x0, -1), (x1, 1)):
        probe = (piece.cx, face + sign * 0.02) if along_x else (face + sign * 0.02, piece.cy)
        if m.region_of(probe) is None:  # the outside of the building, or the end of another wall
            continue
        a, b = sorted((face - sign * WALL_SINK_M, face + sign * proud))
        top = m.wall_height + WALL_SINK_M
        box = (x0, a, z0, x1, b, top) if along_x else (a, y0, z0, b, y1, top)
        out.append((f"{what} band at {proud:.2f} m along the face at {face:.2f} m", box))
    return out


def wall_bands(spec: dict | MapSpec, proud: float = WALL_PROUD_MAX_M) -> list[Zone]:
    """Per face of each wall piece that looks into a room, the strip from the floor to the ceiling at most `proud`
    from the face (and `WALL_SINK_M` into the wall); and over each door or arch, the wall above its head. The cut of
    the opening itself is no wall."""
    m = _map(spec)
    out = []
    for piece in m.wall_pieces():
        out += _faces(m, piece, 0.0, proud, "wall")
    for o in m.openings:
        if o.height < m.wall_height:
            head = Rect(o.center[0], o.center[1], o.width, m.wall(o.wall).thickness, o.yaw)
            out += _faces(m, head, o.height, proud, f"wall over {o.id}")
    return out


def ceiling_bands(spec: dict | MapSpec, depth: float = CEILING_BAND_M) -> list[Zone]:
    """Per room, the slab under the ceiling `depth` deep (and `WALL_SINK_M` into the ceiling)."""
    m = _map(spec)
    return [(f"ceiling band over {r.id}", (x0, y0, m.wall_height - depth, x1, y1, m.wall_height + WALL_SINK_M))
            for r, (x0, y0, x1, y1) in zip(m.regions, rooms(m), strict=True)]


def structure_band(spec: dict | MapSpec, from_z: float = STRUCTURE_FROM_M) -> list[Zone]:
    """Per room of a hall taller than `from_z`, the space from there to the roof (none in a house)."""
    m = _map(spec)
    if m.wall_height <= from_z:
        return []
    return [(f"structure above {from_z:.1f} m over {r.id}", (x0, y0, from_z, x1, y1, m.wall_height + WALL_SINK_M))
            for r, (x0, y0, x1, y1) in zip(m.regions, rooms(m), strict=True)]


def footprint_bands(spec: dict | MapSpec) -> list[Zone]:
    """Per blocking fixed furniture piece, the column over its footprint (and `WALL_PROUD_MAX_M` around it) from the
    floor to the ceiling. Receptacle volumes and open undersides inside it are forbidden by `forbidden`, not left out
    here."""
    m = _map(spec)
    out = []
    for f in m.furniture:
        if f.mobility == "fixed" and f.blocks:
            x0, y0, x1, y1 = square(f.footprint(), f"furniture {f.id}")
            d = WALL_PROUD_MAX_M
            out.append((f"on or over {f.name} {f.id}", (x0 - d, y0 - d, 0.0, x1 + d, y1 + d, m.wall_height)))
    return out


def flush_floor_bands(spec: dict | MapSpec) -> list[Zone]:
    """Per room, the floor's first `FLUSH_M`: a flush plate or strip (a dock leveler) needs no clearance."""
    m = _map(spec)
    return [(f"flush floor of {r.id}", (x0, y0, 0.0, x1, y1, FLUSH_M))
            for r, (x0, y0, x1, y1) in zip(m.regions, rooms(m), strict=True)]


def allowed_phase1(spec: dict | MapSpec) -> list[Zone]:
    """Every place a Phase 1 box may be: wall, ceiling, structure, footprint and flush floor bands."""
    m = _map(spec)
    return [*wall_bands(m), *ceiling_bands(m), *structure_band(m), *footprint_bands(m), *flush_floor_bands(m)]


class DressingGuard:
    """A map's forbidden volumes and allowed places, computed once, to test boxes against: `problems(box)` is empty
    for a box that lies in an allowed place and meets no forbidden volume. A layout's own test can build one from the
    spec it lays out: `DressingGuard(spec, "home")`."""

    def __init__(self, spec: dict | MapSpec, catalog: Catalog | str | None = None):
        m = _map(spec)
        self.forbidden = forbidden(m, catalog)
        self.allowed = allowed_phase1(m)
        self._trim = [z[1] for z in wall_bands(m, TRIM_PROUD_M + TRIM_SLACK_M)]
        self._no = np.array([z[1] for z in self.forbidden], dtype=float).reshape(-1, 6)

    def problems(self, box: Box, cls: str | None = None) -> list[str]:
        """Why the box may not stand there, each reason once: it is not inside an allowed place, and every kind of
        forbidden volume it meets. An empty list means it may. Trim (a box inside the bands at most `TRIM_PROUD_M`
        proud of a wall) and a dressing class of `EXEMPT_ABOVE_M` from its height up (`cls`) meet no swept body."""
        out = []
        if not covered(box, (z[1] for z in self.allowed)):
            out.append("not wholly inside a wall, ceiling, structure, footprint or flush floor band")
        b = np.array(box)
        hit = np.all((b[:3] < self._no[:, 3:] - EPS) & (self._no[:, :3] < b[3:] - EPS), axis=1)
        reasons = dict.fromkeys(f"meets {self.forbidden[i][0]}" for i in np.flatnonzero(hit))  # a floor is many boxes
        if covered(box, self._trim) or (cls in EXEMPT_ABOVE_M and box[2] >= EXEMPT_ABOVE_M[cls] - EPS):
            reasons = {r: None for r in reasons if not r.startswith("meets swept body")}
        return out + list(reasons)
