"""`amr/1`'s constants, its map contract and its plan: what a script of the tote AMR is before any world exists.

The job reads a map by piece and region kind only, never by name or id (`docs/job-amr.md`, "The map"): exactly one
`charger`; stations, `shelving` (where totes are loaded) and `packing_table` (where they are delivered), each with one
`bay` touching its front face; rack aisles, regions of kind `aisle` lined by `pallet_rack` pieces whose floor-level
`rack` surfaces are the pick faces. A map without one of them refuses the job (`check_map`, ValueError naming the map
and the kind). `plan_tasks` and `gaps` draw from the map and the seed alone, so both are testable without a world.

A task is `replenish` (a shelving piece, then one or two packing tables), `collect` (the same stations in the other
order: the tables reversed, then the shelving) or `pick` (one to two pick faces of one aisle, then one packing table).
Every task draws the same numbers in the same order whatever its kind decides, so the swap pair's stations are one set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import job_gaps
from .geometry import Point, to_world
from .mapspec import Furniture, MapSpec, Region
from .seeding import make_rng
from .timeline_ego import TaskPlan

JOB_ID = "amr/1"
REPLENISH, COLLECT, PICK = "replenish", "collect", "pick"
KIND_P = {REPLENISH: 0.35, COLLECT: 0.25, PICK: 0.4}
TABLES_PER_TASK = (1, 2)  # packing tables a replenish or collect task serves
PICKS_PER_TASK = (1, 2)  # pick faces of a pick task, in one aisle
T_FACE_M = 1.9  # T, where the robot turns to face a station: its base this far from the face (1.8-2.4; docs/job-amr.md)
DOCK_GAP_M = 0.12  # the docked body's front edge from a station's face (its clearance is 0.10)
SIDE_M = 2.0  # the arrival waypoint: this far from T along the face (then 1.5, 1.0)
SIDE_FALLBACK_M = (SIDE_M, 1.5, 1.0)
PICK_GAP_M = 0.35  # a pick stop: the body's side this far from the rack's face, centred on the bay
CHARGER_RUN_M = 1.0  # forward onto the charger at a task's end, reversed off it at its start
REAR_CAP_M = 5.0  # the reverse corridor's length: `rear_clear`'s cap for the AMR (`questions_words.cap_of`)
EXTENSION_M = 0.3  # the corridor's extension starts this far behind T's rear edge: the rear drop's nearest place
BAY_STANDOFF_M = 0.7  # a bay is a keep-out up to this far short of T (the planner keeps 0.612 m from one: T stays free)
CROSSING_STANDOFF_M = 0.5  # a before_crossing: the base this far before the way crosses an aisle's boundary
WAIT_S = 10.0  # the hold on the charger at a task's start and end
STATION_S = (15.0, 40.0)  # at_station: a dock's transfer, or a pick face's wait for the picker
DRAIN_PCT_PER_MIN = 0.15  # MiR250: about 13 h on a charge (its specifications)
TASK_MAX_S = 900.0
BAY_COIN_P = 0.5  # a station visit's bay is blocked with this probability, a world-independent coin (`job_amr_drops`)
REAR_DROP_P = 0.6  # a pallet or a drum is left in a visit's reverse corridor with this probability
BEFORE_STATION, BEFORE_UNDOCK, BEFORE_CROSSING = "before_station", "before_undock", "before_crossing"

CHARGER, SHELVING, TABLE, BAY, RACK, LANE = ("charger", "shelving", "packing_table", "bay", "pallet_rack",
                                              "staging_lane")
AISLE = "aisle"  # a region's kind
RACK_SURFACE = "rack"  # a surface's kind
PICK_Z_MAX_M = 0.01  # a rack surface at most this high is at floor level: a pick face
MOUTH_PROBE_M = 0.2  # a boundary edge of an aisle: the point this far outside its middle lies in another kind's region
FACE_PROBE_M = 0.5  # a rack's face looks into an aisle when the point this far in front of it lies in the aisle
TOUCH_M = 0.01  # a bay touches a station's face when its near edge is within this of it


@dataclass(frozen=True)
class Face:
    """A pick face: a floor-level `rack` surface of a rack that looks into the aisle."""

    rack: str  # the pallet_rack piece's map id
    surface: str  # its surface's id
    point: Point  # the surface's middle along the rack's front face, on the face
    out: Point  # the unit vector out of the face, into the aisle
    depth_m: float  # from the aisle's mouth along it: faces are ordered by it


@dataclass(frozen=True)
class Aisle:
    region: str  # the aisle's region id
    kind: str  # the region's kind ("aisle")
    inward: Point  # the unit vector from the mouth into the aisle: the heading along it
    mouth: Point  # the middle of the longest edge shared with a region of another kind
    edges: tuple[tuple[Point, Point, Point], ...]  # the edges shared with other kinds: (a, b, unit outward normal)
    faces: tuple[Face, ...]  # ordered from the mouth


def pieces(m: MapSpec, kind: str) -> list[Furniture]:
    """The fixed pieces of the kind, in map order."""
    return [f for f in m.furniture if f.kind == kind and f.mobility == "fixed"]


def charger(m: MapSpec) -> Furniture:
    found = pieces(m, CHARGER)
    if len(found) != 1:
        raise ValueError(f"{m.name} must hold exactly one {CHARGER!r} piece for {JOB_ID}, not {len(found)}")
    return found[0]


def front_of(f: Furniture) -> tuple[Point, Point, Point]:
    """(the middle of the piece's front face, the unit vector out of it, the unit vector along it): the piece's yaw is
    the direction its front faces."""
    d = (math.cos(f.yaw), math.sin(f.yaw))
    return (f.center[0] + d[0] * f.size[0] / 2, f.center[1] + d[1] * f.size[0] / 2), d, (-d[1], d[0])


def bay_of(m: MapSpec, station: Furniture) -> Furniture:
    """The one `bay` piece touching the station's front face: its near edge on the face and its middle within the face's
    width (the zone rule insets a zone by 0.05 m, so a station touching its bay never blocks it)."""
    (fx, fy), d, s = front_of(station)
    found = []
    for b in pieces(m, BAY):
        along_d = (b.center[0] - fx) * d[0] + (b.center[1] - fy) * d[1]
        along_s = (b.center[0] - fx) * s[0] + (b.center[1] - fy) * s[1]
        if abs(along_d - b.size[0] / 2) <= TOUCH_M and abs(along_s) <= station.size[1] / 2:
            found.append(b)
    if len(found) != 1:
        raise ValueError(f"{m.name}: the {station.kind} {station.id} must have exactly one {BAY!r} piece touching its "
                         f"front face, not {len(found)}")
    return found[0]


def _edges(m: MapSpec, region: Region) -> list[tuple[Point, Point, Point]]:
    """The edges of the region's polygon that another kind's region lies across: (a, b, the unit outward normal)."""
    poly = list(region.polygon)
    area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1], strict=True))
    if area < 0:
        poly.reverse()  # counter-clockwise: the outward normal of a-b is (dy, -dx)
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = ((b[1] - a[1]) / length, -(b[0] - a[0]) / length)
        probe = ((a[0] + b[0]) / 2 + n[0] * MOUTH_PROBE_M, (a[1] + b[1]) / 2 + n[1] * MOUTH_PROBE_M)
        other = m.region_of(probe)
        if other is not None and m.region(other).kind != region.kind:
            out.append((a, b, n))
    return out


def aisles(m: MapSpec) -> list[Aisle]:
    """The regions of kind `aisle`, in map order, each with its mouth, the way along it and its pick faces from the
    mouth in. An aisle with no edge shared with another kind has no mouth, and one with no pick face none: ValueError.
    """
    found = []
    for region in (r for r in m.regions if r.kind == AISLE):
        edges = _edges(m, region)
        if not edges:
            raise ValueError(f"{m.name}: the {AISLE} {region.id} shares no edge with a region of another kind")
        a, b, n = max(edges, key=lambda e: math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]))
        mouth, inward = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), (-n[0], -n[1])
        faces = []
        for rack in (f for f in pieces(m, RACK) if m.region_of(f.center) == region.id):
            _, d, _ = front_of(rack)
            for surface in rack.surfaces:
                if surface.kind != RACK_SURFACE or surface.z > PICK_Z_MAX_M:
                    continue
                point = to_world(rack.footprint(), rack.size[0] / 2, surface.rect[1])
                ahead = (point[0] + d[0] * FACE_PROBE_M, point[1] + d[1] * FACE_PROBE_M)
                if m.region_of(ahead) == region.id:
                    depth = (point[0] - mouth[0]) * inward[0] + (point[1] - mouth[1]) * inward[1]
                    faces.append(Face(rack.id, surface.id, point, d, depth))
        if not faces:
            raise ValueError(f"{m.name}: the {AISLE} {region.id} has no floor-level {RACK_SURFACE!r} surface to pick")
        found.append(Aisle(region.id, region.kind, inward, mouth, tuple(edges),
                           tuple(sorted(faces, key=lambda f: (f.depth_m, f.rack, f.surface)))))
    return found


def check_map(m: MapSpec) -> None:
    """ValueError naming the map and what is missing: no charger (or several), no shelving or packing table, a station
    without exactly one bay on its face, no aisle with a pick face."""
    charger(m)
    for kind in (SHELVING, TABLE):
        stations = pieces(m, kind)
        if not stations:
            raise ValueError(f"{m.name} has no {kind!r} piece for {JOB_ID}")
        for station in stations:
            bay_of(m, station)
    if not aisles(m):
        raise ValueError(f"{m.name} has no region of kind {AISLE!r} for {JOB_ID}")


def _kind(u: float) -> str:
    """The task kind a uniform draw names: `KIND_P` in its order."""
    edge = 0.0
    for kind, p in KIND_P.items():
        edge += p
        if u < edge:
            return kind
    return next(reversed(KIND_P))


def region_kind(m: MapSpec, f: Furniture) -> str:
    """The kind of the region the piece stands in (ValueError when it stands in none)."""
    region = m.region_of(f.center)
    if region is None:
        raise ValueError(f"{m.name}: the {f.kind} {f.id} stands in no region")
    return m.region(region).kind


def plan_tasks(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    """The script's assignments, from `make_rng(m.name, seed, JOB_ID, "job")` and the map alone:
    - each task's kind by `KIND_P`;
    - replenish: one shelving piece, then one or two packing tables, drawn without replacement; collect: those tables
      in the other order, then the shelving (one station set, so the swap pair's tasks share it);
    - pick: one aisle, one or two of its pick faces (kept in order from its mouth), then one packing table.
    `params` hold `stations` (the map ids the robot docks at, in order), for a pick also `aisle` (a region id) and
    `faces` (`[rack id, surface id]` pairs in order), and `regions` (the kinds of the regions the task works in, for the
    relevance). They are the job's, never delivered: no assignment names one. A task draws the same numbers in the same
    order whatever kind it is."""
    if n_tasks < 1:
        raise ValueError(f"a script has at least one task, not {n_tasks}")
    check_map(m)
    shelves, tables, rows = pieces(m, SHELVING), pieces(m, TABLE), aisles(m)
    reach, home = max(len(a.faces) for a in rows), region_kind(m, charger(m))
    rng = make_rng(m.name, seed, JOB_ID, "job")
    plans: list[TaskPlan] = []
    for _ in range(n_tasks):
        u, shelf = float(rng.random()), shelves[int(rng.integers(len(shelves)))]
        n_tables = min(int(rng.integers(TABLES_PER_TASK[0], TABLES_PER_TASK[1] + 1)), len(tables))
        order = [tables[int(i)] for i in rng.permutation(len(tables))]
        row, n_picks = rows[int(rng.integers(len(rows)))], int(rng.integers(PICKS_PER_TASK[0], PICKS_PER_TASK[1] + 1))
        luck = rng.random(reach)
        kind, served = _kind(u), order[:n_tables]
        if kind == REPLENISH:
            stops, params = [shelf, *served], {}
        elif kind == COLLECT:
            stops, params = [*reversed(served), shelf], {}
        else:
            ranked = sorted(range(len(row.faces)), key=lambda i: luck[i])[:min(n_picks, len(row.faces))]
            stops = [order[0]]
            params = {"aisle": row.region, "faces": [[row.faces[i].rack, row.faces[i].surface] for i in sorted(ranked)]}
        regions = {home, *(region_kind(m, f) for f in stops), *([row.kind] if kind == PICK else [])}
        plans.append(TaskPlan(kind, {"stations": [f.id for f in stops], **params, "regions": sorted(regions)}))
    return plans


def targets_of(plan: TaskPlan) -> tuple[str, ...]:
    """What a task acts on (`Task.targets`, private): its stations' map ids and, for a pick, its aisle's region id."""
    return (*map(str, plan.params["stations"]), *([str(plan.params["aisle"])] if plan.kind == PICK else []))


def gaps(map_name: str, seed: int, n_tasks: int) -> list[float]:
    """The n - 1 unobserved gaps between a script's tasks, in seconds: `job_gaps.gaps` with this job's id and longest
    task, from the seed alone."""
    return job_gaps.gaps(map_name, seed, JOB_ID, n_tasks, TASK_MAX_S)
