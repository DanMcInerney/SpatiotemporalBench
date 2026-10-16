"""`amr/1`'s path: the charger, a station visit, a pick run and where a way crosses an aisle's boundary.

The primitives of one task, over the free space the script planned (`route.robot_free_space` with the keep-outs of
`keep_outs`: every staging lane, every bay up to short of T, the reverse corridors' extensions): docked head-on on the
charger (a `waiting` hold), reversed off it `CHARGER_RUN_M` (`undocking`), then the task's stops, then transit to the
point `CHARGER_RUN_M` out, a turn to face the charger, a forward `approach` onto it and a `waiting` hold.

A **station** (a shelving piece or a packing table) is visited from where the robot is: transit to the arrival waypoint
W, `SIDE_M` from **T** along the face (T is the base `T_FACE_M` from the face, centred on the station's bay), on the
side nearer the robot; a straight leg along the face to T; a turn to face the station; a `before_station` decision; a
straight `approach` to the docked pose (the front edge `DOCK_GAP_M` from the face); an `at_station` hold; a
`before_undock` decision on the hold's last frame; `undocking`, reversing along the same line to T with the heading
kept. The side arrival leaves the reverse corridor (the body's width, 5 m behind the docked rear edge) crossed only at
T, and the keep-outs leave no other way across it, so what is behind a station is seen on the way in and never swept.
A visit whose bay holds a drum (the script's `skips`, from `job_amr_drops.block_bays`) is **skipped**: after the
`before_station` the robot holds at T in `waiting` for the visit's seconds and drives on from T; it never approaches,
docks or reverses, so it has no `at_station` hold, no `before_undock` and no corridor it sweeps.

A **pick run** drives to the aisle (a `before_crossing` where the way crosses into the aisle's region), to each pick
pose (the heading along the aisle, the body's side `PICK_GAP_M` from the rack's face, centred on the pick face, an
`at_station` hold), and out to a packing table (a `before_crossing` where the way crosses out). A crossing is the job's
own rule: the point where the way first crosses an edge that an `aisle` region shares with a region of another kind
(`crossing`), never an opening."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Point, Rect, distance, q, segments_cross, wrap_angle
from .job_amr_plan import (
    BAY,
    BAY_STANDOFF_M,
    BEFORE_CROSSING,
    BEFORE_STATION,
    BEFORE_UNDOCK,
    CHARGER_RUN_M,
    CROSSING_STANDOFF_M,
    DOCK_GAP_M,
    EXTENSION_M,
    LANE,
    PICK,
    PICK_GAP_M,
    REAR_CAP_M,
    SHELVING,
    SIDE_FALLBACK_M,
    T_FACE_M,
    TABLE,
    WAIT_S,
    Aisle,
    Face,
    aisles,
    bay_of,
    charger,
    front_of,
    pieces,
)
from .job_vacuum_path import Driver
from .keyframes import Hold, Mark, Move
from .mapspec import Furniture, MapSpec
from .robots import Robot
from .route import FreeSpace
from .timeline_ego import TaskPlan

RESERVE_REAR = True  # plan round the reverse corridors' extensions (see `keep_outs`)


@dataclass(frozen=True)
class Dock:
    """Where the robot docks at a station: all on the line through the middle of the station's bay."""

    station: str  # the station's map id
    bay: str  # its bay's map id
    heading: float  # facing the station, radians
    docked: Point  # the base with its front edge `DOCK_GAP_M` from the face
    t: Point  # the base `T_FACE_M` from the face
    out: Point  # the unit vector out of the face
    along: Point  # the unit vector along the face


@dataclass
class Route:
    """The primitives of one task and what the script needs of them: the docks it visits in order (a skipped visit's
    too), and for each `at_station` hold in order whether the robot carries a load once it has ended."""

    prims: list = field(default_factory=list)
    docks: list[Dock] = field(default_factory=list)
    loaded_after: list[bool] = field(default_factory=list)


def _pt(p: Point) -> Point:
    return (q(p[0]), q(p[1]))


def charger_pose(m: MapSpec, robot: Robot) -> tuple[Point, float, Point]:
    """(the docked base, its heading, the point `CHARGER_RUN_M` out): head-on, the body's front edge on the charger's
    face (an anchor may be touched), facing the charger; the point is on the same line in front of it."""
    face, d, _ = front_of(charger(m))
    half = robot.body.length_m / 2
    docked = _pt((face[0] + d[0] * half, face[1] + d[1] * half))
    return docked, q(wrap_angle(math.atan2(-d[1], -d[0]))), _pt((docked[0] + d[0] * CHARGER_RUN_M,
                                                                  docked[1] + d[1] * CHARGER_RUN_M))


def dock_of(m: MapSpec, station: Furniture, robot: Robot) -> Dock:
    """The dock of a station: on the line through the middle of its bay, facing it."""
    bay = bay_of(m, station)
    face, d, s = front_of(station)
    off = (bay.center[0] - face[0]) * s[0] + (bay.center[1] - face[1]) * s[1]

    def base(away: float) -> Point:
        return _pt((face[0] + d[0] * away + s[0] * off, face[1] + d[1] * away + s[1] * off))

    return Dock(station.id, bay.id, q(wrap_angle(station.yaw + math.pi)), base(DOCK_GAP_M + robot.body.length_m / 2),
                base(T_FACE_M), d, s)


def extension(dock: Dock, robot: Robot) -> Rect:
    """The reverse corridor's extension behind a docked visit: from T's rear edge plus `EXTENSION_M` to `REAR_CAP_M`
    behind the docked rear edge, the body's width across, along the dock's line."""
    half = robot.body.length_m / 2
    near = (dock.t[0] + dock.out[0] * (half + EXTENSION_M), dock.t[1] + dock.out[1] * (half + EXTENSION_M))
    far = (dock.docked[0] + dock.out[0] * (half + REAR_CAP_M), dock.docked[1] + dock.out[1] * (half + REAR_CAP_M))
    return Rect((near[0] + far[0]) / 2, (near[1] + far[1]) / 2, math.hypot(far[0] - near[0], far[1] - near[1]),
                robot.body.width_m, math.atan2(dock.out[1], dock.out[0]))


def bay_keep_out(bay: Furniture) -> Rect:
    """The part of a bay the planner keeps out of: from the station's face to `BAY_STANDOFF_M` short of T (the whole bay
    when T is far enough). The planner keeps 0.612 m from it, so T and the leg along the face stay free whatever stands
    in the bay: a drum in a bay stands wholly inside it (`job_amr_drops.block_bays`)."""
    depth = min(bay.size[0], T_FACE_M - BAY_STANDOFF_M)
    (dx, dy), cut = (math.cos(bay.yaw), math.sin(bay.yaw)), bay.size[0] - depth  # the cut comes off the outer end
    return Rect(bay.center[0] - dx * cut / 2, bay.center[1] - dy * cut / 2, depth, bay.size[1], bay.yaw)


def keep_outs(m: MapSpec, robot: Robot) -> list[Rect]:
    """The places every task is planned round whatever they hold: each staging lane's footprint (a lane's load must not
    change a path), each bay's footprint from the station's face up to `BAY_STANDOFF_M` short of T (so no way crosses a
    bay, whatever a drum in it says, and T and the leg along the face stay free) and, with `RESERVE_REAR`, the reverse
    corridor's extension behind each station (nothing sweeps what a rear drop may stand in)."""
    out = [f.footprint() for f in pieces(m, LANE)]
    if RESERVE_REAR:
        out += [extension(dock_of(m, f, robot), robot) for f in [*pieces(m, SHELVING), *pieces(m, TABLE)]]
    return out + [bay_keep_out(bay) for bay in pieces(m, BAY)]


def pick_pose(face: Face, aisle: Aisle, robot: Robot) -> tuple[Point, float]:
    """The base at a pick face: its side `PICK_GAP_M` from the rack's face, centred on the face, heading along the aisle
    (away from its mouth)."""
    gap = PICK_GAP_M + robot.body.width_m / 2
    return (_pt((face.point[0] + face.out[0] * gap, face.point[1] + face.out[1] * gap)),
            q(math.atan2(aisle.inward[1], aisle.inward[0])))


def _path(free: FreeSpace, a: Point, b: Point) -> list[Point] | None:
    """The A* polyline from a to b with repeated points dropped; None when there is none."""
    found = free.path(a, b, strict=True)
    return None if found is None else [p for i, p in enumerate(found) if i == 0 or distance(p, found[i - 1]) > 1e-9]


def way(free: FreeSpace, a: Point, b: Point, what: str) -> list[Point]:
    """`_path`, or ValueError naming what the way was for."""
    found = _path(free, a, b)
    if found is None:
        raise ValueError(f"no way {what} on the AMR's free space: {a} to {b}")
    return found


def station_poly(free: FreeSpace, dock: Dock, at: Point) -> list[Point]:
    """The way from `at` to W and on along the face to T: W is `SIDE_M` from T along the face on the side nearer `at`;
    the other side, then 1.5 m and 1.0 m, when it is not free, not reachable or the leg to T is not clear."""
    near = 1.0 if (at[0] - dock.t[0]) * dock.along[0] + (at[1] - dock.t[1]) * dock.along[1] >= 0 else -1.0
    for side in SIDE_FALLBACK_M:
        for sign in (near, -near):
            w = (dock.t[0] + dock.along[0] * sign * side, dock.t[1] + dock.along[1] * sign * side)
            if free.is_free(w) and free.line_clear(w, dock.t) and (found := _path(free, at, w)) is not None:
                return [*found, dock.t]
    raise ValueError(f"no arrival to the station {dock.station} from {at}: T {dock.t} has no free leg along its face")


def _hit(a: Point, b: Point, p: Point, r: Point) -> float:
    """Metres from a along a-b to where it crosses the line p-r (the segments are known to cross)."""
    (dx, dy), (ex, ey) = (b[0] - a[0], b[1] - a[1]), (r[0] - p[0], r[1] - p[1])
    return ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / (dx * ey - dy * ex) * distance(a, b)


def crossing(m: MapSpec, poly: list[Point], into: bool, standoff: float = CROSSING_STANDOFF_M) -> tuple[int, float]:
    """Where a decision before the way crosses an aisle's boundary sits on the polyline: (segment index, metres from
    that segment's start), `standoff` before the first crossing along its segment, or at the segment's start when that
    is closer. The boundary is any edge that an `aisle` region shares with a region of another kind; `into` takes the
    first crossing into an aisle, else the first out of one. ValueError when the way crosses none."""
    hits = []
    for i, (a, b) in enumerate(zip(poly, poly[1:], strict=False)):
        for aisle in aisles(m):
            for p, r, n in aisle.edges:
                if segments_cross(a, b, p, r) and (((b[0] - a[0]) * n[0] + (b[1] - a[1]) * n[1] < 0) == into):
                    hits.append((i, _hit(a, b, p, r)))
    if not hits:
        raise ValueError(f"the way crosses no {'entry to' if into else 'exit from'} an aisle")
    i, s = min(hits)
    return i, max(0.0, s - standoff)


def ahead_of_boundary(m: MapSpec, xy: Point, yaw: float) -> float | None:
    """The distance along the heading to the nearest aisle boundary edge (any crossing), None when it meets none."""
    d = (math.cos(yaw), math.sin(yaw))
    best = None
    for aisle in aisles(m):
        for p, r, _ in aisle.edges:
            (ex, ey), denom = (r[0] - p[0], r[1] - p[1]), d[0] * (r[1] - p[1]) - d[1] * (r[0] - p[0])
            if abs(denom) < 1e-12:
                continue
            t = ((p[0] - xy[0]) * ey - (p[1] - xy[1]) * ex) / denom
            u = ((p[0] - xy[0]) * d[1] - (p[1] - xy[1]) * d[0]) / denom
            if t >= -1e-9 and -1e-9 <= u <= 1 + 1e-9 and (best is None or t < best):
                best = max(t, 0.0)
    return best


def visit(m: MapSpec, drive: Driver, route: Route, free: FreeSpace, dock: Dock, at: Point, speed: dict[str, float],
          hold_s: float, leaving_aisle: bool = False, skip_s: float | None = None) -> Point:
    """One station visit (the module docstring) from `at`; the point the robot is left at, T. With `leaving_aisle` the
    way there is out of an aisle, and a `before_crossing` stands before it crosses the aisle's boundary. With `skip_s`
    the bay is blocked: the robot holds at T for `skip_s` seconds in `waiting` after the `before_station` and leaves
    from T (`hold_s`, the dock's transfer, is not used)."""
    poly = station_poly(free, dock, at)
    mark = (*crossing(m, poly, into=False), BEFORE_CROSSING) if leaving_aisle else None
    drive.go(poly, speed["transit"], "transit", mark)
    drive.turn(dock.t, dock.heading, "transit")
    if skip_s is not None:
        drive.prims += [Mark(BEFORE_STATION), Hold(dock.t, dock.heading, skip_s, "waiting")]
    else:
        drive.prims += [Mark(BEFORE_STATION), Move(dock.t, dock.docked, dock.heading, speed["approach"], "approach"),
                        Hold(dock.docked, dock.heading, hold_s, "at_station"), Mark(BEFORE_UNDOCK),
                        Move(dock.docked, dock.t, dock.heading, speed["undocking"], "undocking")]  # reversing
    route.docks.append(dock)
    return dock.t


def stops(plan: TaskPlan) -> int:
    """How many `at_station` holds a task has: its stations and, for a pick, its pick faces."""
    return len(plan.params["stations"]) + (len(plan.params["faces"]) if plan.kind == PICK else 0)


def task_prims(m: MapSpec, free: FreeSpace, robot: Robot, plan: TaskPlan, speed: dict[str, float],
               holds: list[float], skips: dict[int, float] | None = None) -> Route:
    """The primitives of one task: docked on the charger, undocking, transit to the task's stops (a replenish or
    collect: its stations in order; a pick: the aisle, the pick faces and a packing table), transit back, a forward
    approach onto the charger and a hold. `holds` are the seconds of each `at_station` hold in order (`stops(plan)` of
    them: a pick face's wait, a dock's transfer); a skipped visit takes its entry and does not use it, so the holds of
    the visits after it are what they would be. `skips` maps the index of each station visit the robot skips (its bay
    holds a drum) to the seconds it waits at T. A pick's crossings are `before_crossing` decisions on the way in and
    the way out. The robot carries a load from the end of the first hold made until the end of the last planned stop's,
    if that stop docked: the first stop loads it (the shelving of a replenish, the first table of a collect, the first
    pick face) and the last unloads it. A skipped last stop leaves it loaded to the task's end, and a task that skips
    every stop never loads (`Route.loaded_after`, one value for each hold made)."""
    home, heading, out = charger_pose(m, robot)
    drive, route = Driver(heading, robot.turn_dps, exact=True), Route()
    drive.prims += [Hold(home, heading, WAIT_S, "waiting"), Move(home, out, heading, speed["undocking"], "undocking")]
    by_id = m.furniture_by_id()
    left, at = iter(holds), out
    if plan.kind == PICK:
        aisle = next(a for a in aisles(m) if a.region == plan.params["aisle"])
        faces = {(f.rack, f.surface): f for f in aisle.faces}
        for j, (rack, surface) in enumerate(plan.params["faces"]):
            base, yaw = pick_pose(faces[rack, surface], aisle, robot)
            poly = way(free, at, base, f"to the pick face {rack}/{surface}")
            mark = (*crossing(m, poly, into=True), BEFORE_CROSSING) if j == 0 else None
            drive.go(poly, speed["transit"], "transit", mark)
            drive.turn(base, yaw, "transit")
            drive.prims.append(Hold(base, yaw, next(left), "at_station"))
            at = base
    stations, skips = plan.params["stations"], skips or {}
    route.loaded_after = [True] * (len(plan.params["faces"]) if plan.kind == PICK else 0)
    for j, station in enumerate(stations):
        at = visit(m, drive, route, free, dock_of(m, by_id[station], robot), at, speed, next(left),
                   leaving_aisle=plan.kind == PICK, skip_s=skips.get(j))
        if j not in skips:
            route.loaded_after.append(j != len(stations) - 1)  # the last planned stop's hold unloads
    back = way(free, at, out, "back to the charger")
    drive.go(back, speed["transit"], "transit")
    drive.turn(out, heading, "transit")
    drive.prims += [Move(out, home, heading, speed["approach"], "approach"), Hold(home, heading, WAIT_S, "waiting")]
    route.prims = drive.prims
    return route
