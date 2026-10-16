"""The drone's flight: the way over the streets (from the map alone), the primitives of one flight, and the tether.

The way is A* on a free grid of the map's own (`FLIGHT_GRID_M`), the walls grown by the drone's body and 3 m clearance,
and every fixed piece lower than the transit height less that clearance flown over (`robots.flies_over`): it reads
nothing the world moves, so no flight's path tells of the world. The flight is the primitives of `keyframes`:
preflight on the pad, the gimbal to -45 and a vertical takeoff, the approach at transit height, the gimbal to -90 and
an inspection hover, a vertical descent to the lowering height, the lowering, the release, the tether reeled in and a
vertical climb out, the gimbal to -45 and the way back, the gimbal to -90 and a vertical landing. An inspection flies
the same primitives with the hover at the lowering height named `inspect` (the swap pair's one difference, with the
tether), and a `Mark()` at every boundary of that hover, so both kinds frame at the same instants. A delivery the drone
aborts at `before_lower` (its drop zone holds an item) flies the inspection's primitives, the package kept: the
decision's response is the job's, `flight_prims` is given the kind that was flown.
"""

from __future__ import annotations

import math

from . import robots
from .geometry import Point, distance, point_in_polygon, wrap_angle
from .job_drone_plan import (
    ALT_M,
    BEFORE_DESCENT,
    BEFORE_LOWER,
    BEFORE_RETURN,
    DELIVER,
    FLIGHT_GRID_M,
    GIMBAL_DPS,
    LEG_M,
    LOWER_M,
    PITCH_DEG,
    PREFLIGHT_S,
    RELEASE_S,
)
from .keyframes import Frame, Hold, Mark, Move, Prim, Tilt, Turn
from .mapspec import Furniture, MapSpec
from .route import FreeSpace, blocking_rects
from .zones import ground_z

MIN_TURN_RAD = math.radians(1.0)  # a vertex that turns the heading less than this is not turned in place
SAMPLE_M = 1.0  # the way is sampled this often for the regions it crosses
PITCH = {name: math.radians(deg) for name, deg in PITCH_DEG.items()}  # radians, up positive


def pad_z(m: MapSpec, nest: Furniture) -> float:
    """The base height of a drone on the nest: the nest's top above its ground, plus the landing gear."""
    return ground_z(m, nest) + nest.size[2] + LEG_M


def flight_free(m: MapSpec, robot: robots.Robot) -> FreeSpace:
    """The free grid the drone flies on: its base keeps its disc's radius and `body.clearance_m` from every wall (at any
    height) and from every fixed piece it does not fly over at transit height, grown by the cell's half-diagonal so
    every point of a free cell keeps it."""
    body = robot.body
    clearance = body.length_m / 2 + body.clearance_m + FLIGHT_GRID_M * math.sqrt(2) / 2
    return FreeSpace(m, blocking_rects(m, over=robots.over_m(robot, ALT_M)), clearance, FLIGHT_GRID_M)


def flight_way(free: FreeSpace, a: Point, b: Point, what: str) -> list[Point]:
    """The polyline from `a` to `b` over the free grid, smoothed to chords whose every point is in a free cell."""
    way = free.path(a, b, strict=True)
    if way is None:
        raise ValueError(f"no way {what} on the drone's free space: {a} to {b}")
    return [p for i, p in enumerate(way) if i == 0 or distance(p, way[i - 1]) > 1e-9]


def regions_crossed(m: MapSpec, way: list[Point]) -> frozenset[str]:
    """The ids of the regions the way passes through, sampled every `SAMPLE_M`, with its ends'."""
    found = set()
    for a, b in zip(way, way[1:], strict=False):
        n = max(math.ceil(distance(a, b) / SAMPLE_M), 1)
        for i in range(n + 1):
            x, y = a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n
            found |= {r.id for r in m.regions if point_in_polygon(x, y, list(r.polygon))}
    return frozenset(found)


def _heading(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _fly(way: list[Point], yaw: float, z: float, pitch: float, speed: float, turn_dps: float, phase: str
         ) -> tuple[list[Prim], float]:
    """A level flight along the polyline: at each vertex a turn in place to the next segment's heading when it differs
    by more than `MIN_TURN_RAD`, then a straight move. Returns the primitives and the heading it ends on."""
    prims: list[Prim] = []
    for a, b in zip(way, way[1:], strict=False):
        heading = _heading(a, b)
        if abs(wrap_angle(heading - yaw)) > MIN_TURN_RAD:
            prims.append(Turn(a, yaw, heading, turn_dps, phase, z=z, pitch=pitch))
        yaw = heading
        prims.append(Move(a, b, heading, speed, phase, z=(z, z), pitch=pitch))
    return prims, yaw


def hover_phases(kind: str) -> tuple[str, str, str]:
    """The phases of the hover at the lowering height: lowering, release and the tether reeled in. A delivery's are
    `lower`, `release` and `climb_out`; an inspection hovers the same time, all of it `inspect`. A delivery the drone
    aborts (its zone holds an item) flies the inspection's: the caller asks for `INSPECT`'s."""
    return ("lower", "release", "climb_out") if kind == DELIVER else ("inspect",) * 3


def flight_prims(robot: robots.Robot, kind: str, way: list[Point], z_pad: float, z_low: float,
                 speed: dict[str, float], inspect_s: float) -> list[Prim]:
    """The primitives of one flight along `way` (the nest first, the drop zone's centre last), the pad's base height
    `z_pad`, the lowering height `z_low` and the speeds of `job_drone_plan.SPEED_PHASES`."""
    nest, zone, dps = way[0], way[-1], robot.turn_dps
    preflight, transit, down = PITCH["preflight"], PITCH["transit"], PITCH["down"]
    lower_s = LOWER_M / speed["lower"]  # the tether pays out, and is reeled in, at the lowering speed
    lowering, release, reel = hover_phases(kind)
    yaw0 = _heading(way[0], way[1])
    out: list[Prim] = [Hold(nest, yaw0, PREFLIGHT_S, "preflight", z=z_pad, pitch=preflight),
                       Tilt(nest, yaw0, z_pad, preflight, transit, GIMBAL_DPS, "takeoff"),
                       Move(nest, nest, yaw0, speed["climb"], "takeoff", z=(z_pad, ALT_M), pitch=transit)]
    approach, yaw = _fly(way, yaw0, ALT_M, transit, speed["approach"], dps, "approach")
    out += approach
    out += [Tilt(zone, yaw, ALT_M, transit, down, GIMBAL_DPS, "inspect"),
            Hold(zone, yaw, inspect_s, "inspect", z=ALT_M, pitch=down), Mark(BEFORE_DESCENT),
            Move(zone, zone, yaw, speed["descend"], "descend", z=(ALT_M, z_low), pitch=down), Mark(BEFORE_LOWER),
            Hold(zone, yaw, lower_s, lowering, z=z_low, pitch=down), Mark(),
            Hold(zone, yaw, RELEASE_S, release, z=z_low, pitch=down), Mark(),
            Hold(zone, yaw, lower_s, reel, z=z_low, pitch=down), Mark(),
            Move(zone, zone, yaw, speed["climb"], "climb_out", z=(z_low, ALT_M), pitch=down), Mark(BEFORE_RETURN),
            Tilt(zone, yaw, ALT_M, down, transit, GIMBAL_DPS, "return")]
    back, yaw = _fly(way[::-1], yaw, ALT_M, transit, speed["approach"], dps, "return")
    out += back
    out += [Tilt(nest, yaw, ALT_M, transit, down, GIMBAL_DPS, "land"),
            Move(nest, nest, yaw, speed["descend"], "land", z=(ALT_M, z_pad), pitch=down)]
    return out


def tethers(steps: list[Frame], lower_speed: float, z_low: float, lowers: bool) -> list[float]:
    """The tether paid out (metres) at each frame of a flight that lowers its package (`lowers`: a delivery that is not
    aborted): 0 until the `before_lower` frame, then rising at the lowering speed to `LOWER_M` over `lower`, held over
    `release`, falling at the same speed over the reel-in (`climb_out` at the lowering height) and 0 from the climb on.
    An inspection, and a delivery aborted at `before_lower`, pay out none."""
    if not lowers:
        return [0.0] * len(steps)
    start = next(f.t for f in steps if f.decision == BEFORE_LOWER)
    reeling = max((f.t for f in steps if f.phase == "release"), default=start)
    out = []
    for f in steps:
        if f.phase == "lower":
            out.append(min(LOWER_M, lower_speed * (f.t - start)))
        elif f.phase == "release":
            out.append(LOWER_M)
        elif f.phase == "climb_out" and f.z <= z_low + 1e-6:
            out.append(max(0.0, LOWER_M - lower_speed * (f.t - reeling)))
        else:
            out.append(0.0)
    return out
