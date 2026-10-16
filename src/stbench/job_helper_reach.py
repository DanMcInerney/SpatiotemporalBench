"""Where the home helper may work: the surfaces it can reach, and the poses it reaches them from (`helper/1`).

A helper that carries a thing from one surface to another needs floor to stand on within arm's reach of each. A reach
pose is a cell of the robot's free space whose centre lies `REACH_M` to `REACH_MAX_M` (front edge to the piece's
footprint) from a fixed surface piece, in the piece's own room and reachable from the park pose, the robot facing the
nearest point of the footprint. Which pose of those a task uses is the one nearest the ideal: on the piece's front axis,
`REACH_M` from its face (the nearest free cell is a slide along the face, a back-off, or a corner, in that order of
distance).

`apartment_a` was drawn for a 0.17 m vacuum, and the helper keeps 0.355 m from everything (`route.robot_free_space`: its
circumscribed disc, its 0.10 m clearance and a cell's half diagonal): the fridge stands in a corner closed by counters,
the washer-dryer in a pocket behind a dining chair, the nightstands, the desk and the bed in channels under 0.7 m wide.
So the sources and targets are not named by kind: `qualified(m)` finds them on the map, by a rule that reads the map and
the robot alone. A piece *counts* when its reach cells on the map's own floor (fixed and light pieces at their map
places, the interior doors open, no clutter) cover at least `MIN_REACH_AREA_M2`; a room *qualifies* when at least
`MIN_ROOM_PIECES` of its pieces count, so that a world which clutters one piece's reach floor leaves another of the room
(`reach_pose` tries a task's drawn piece first, then the room's others in id order). Measured on seeds 101-130 (docs/
job-helper.md) a qualified room has a reach pose in every world.
"""

from __future__ import annotations

import math
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .geometry import Point, dist_to_rect, nearest_on_rect, points_in_polygon, q
from .mapspec import Furniture, MapSpec
from .robots import ROBOTS, Robot
from .route import ROBOT_GRID_RES, FreeSpace, robot_free_space
from .world_processes import doors_on_ways

if TYPE_CHECKING:
    from .changes import World

REACH_M, REACH_MAX_M = 0.3, 0.6  # the body's front edge this far from the piece's footprint: the arm's working range
MIN_REACH_AREA_M2 = 0.15  # a piece counts when this much map floor is within its reach
MIN_ROOM_PIECES = 2  # a room qualifies when this many pieces count
BAND_TOL_M = 1e-9  # a cell exactly on the band's edge is in it, whatever the last bit of a distance is
ROBOT_ID = "helper"
Pose = tuple[Point, float]  # base position, heading


def surface_pieces(m: MapSpec) -> list[Furniture]:
    """The pieces a helper may take a thing from or put it on: the fixed, blocking pieces that have a surface (a
    counter, a table, a shelf; not a rug), in id order."""
    return sorted((f for f in m.furniture if f.mobility == "fixed" and f.blocks and f.surfaces), key=lambda f: f.id)


def room_of(m: MapSpec, piece: Furniture) -> str:
    room = m.region_of(piece.center)
    if room is None:
        raise ValueError(f"the {piece.name} ({piece.id}) of {m.name} stands in no region")
    return room


def reach_band(robot: Robot) -> tuple[float, float]:
    """The distances from a piece's footprint to the base's centre that put the front edge `REACH_M` to `REACH_MAX_M`
    out."""
    half = robot.body.length_m / 2
    return REACH_M + half, REACH_MAX_M + half


def ideal_pose(piece: Furniture, robot: Robot) -> Point:
    """The base position the reach prefers: on the piece's front axis, the front edge `REACH_M` from its face."""
    reach = piece.size[0] / 2 + REACH_M + robot.body.length_m / 2
    return (piece.center[0] + reach * math.cos(piece.yaw), piece.center[1] + reach * math.sin(piece.yaw))


def reach_cells(m: MapSpec, free: FreeSpace, mask: np.ndarray, piece: Furniture, robot: Robot
                ) -> tuple[np.ndarray, np.ndarray]:
    """The centres (xs, ys) of the cells of `mask` (a mask over `free`'s grid) within the piece's reach, in its room."""
    lo, hi = reach_band(robot)
    xs, ys = free.cells_xy(mask)
    near = dist_to_rect(xs, ys, piece.footprint())
    polygon = [list(p) for p in m.region(room_of(m, piece)).polygon]
    keep = (near >= lo - BAND_TOL_M) & (near <= hi + BAND_TOL_M) & points_in_polygon(xs, ys, polygon)
    return xs[keep], ys[keep]


def reach_pose(m: MapSpec, free: FreeSpace, mask: np.ndarray, piece: Furniture, robot: Robot) -> Pose | None:
    """The reach pose at the piece: the cell of `reach_cells` nearest `ideal_pose` (the first in row order on a tie),
    facing the nearest point of the footprint; None when no cell is in reach."""
    xs, ys = reach_cells(m, free, mask, piece, robot)
    if not len(xs):
        return None
    ix, iy = ideal_pose(piece, robot)
    # to the micrometre: two cells the same distance from the ideal (one each side of a face's axis) must tie on every
    # platform, whatever the last bit of `hypot` is, and the first in row order then wins
    i = int(np.argmin(np.round(np.hypot(xs - ix, ys - iy), 6)))
    x, y = float(xs[i]), float(ys[i])
    (nx, ny), _ = nearest_on_rect(piece.footprint(), x, y)
    return (q(x), q(y)), q(math.atan2(ny - y, nx - x))


def bare_world(m: MapSpec) -> World:
    """The map's own world: every piece at its map place, the interior doors open, nothing carried and no clutter."""
    from .generate import build_world  # noqa: PLC0415  (generate imports the job registry, which imports the jobs)

    w = build_world(m, 0)
    w.carried.clear()
    return w


def free_space_for(m: MapSpec, w: World, robot: Robot, rooms: frozenset[str], t: float = 0.0) -> FreeSpace:
    """The robot's free space in `w` at `t` with every interior door off the ways between `rooms` shut in both leaf
    positions (the planner of a task that works in those rooms)."""
    on = doors_on_ways(m, rooms)
    return robot_free_space(m, w, robot, t, shut=sorted(o.id for o in m.toggling_doors() if o.id not in on))


@cache
def qualified(m: MapSpec, park: Point) -> MappingProxyType[str, tuple[str, ...]]:
    """The rooms and pieces a helper working from `park` may use on the map (see the module docstring): region id -> the
    ids of the room's counted pieces, in id order. ValueError when `park` is not free floor on the map."""
    robot, w = ROBOTS[ROBOT_ID], bare_world(m)
    park_room = m.region_of(park)
    if park_room is None:
        raise ValueError(f"the park pose {park} of {m.name} stands in no region")
    counted: dict[str, list[str]] = {}
    for room in sorted({room_of(m, f) for f in surface_pieces(m)}):
        free = free_space_for(m, w, robot, frozenset({park_room, room}))
        if not free.is_free(park):
            raise ValueError(f"the park pose {park} is not free floor on {m.name} for the {robot.id}")
        mask = free.reachable_from(park)
        for piece in (f for f in surface_pieces(m) if room_of(m, f) == room):
            if len(reach_cells(m, free, mask, piece, robot)[0]) * ROBOT_GRID_RES**2 >= MIN_REACH_AREA_M2:
                counted.setdefault(room, []).append(piece.id)
    return MappingProxyType({room: tuple(ids) for room, ids in counted.items() if len(ids) >= MIN_ROOM_PIECES})
