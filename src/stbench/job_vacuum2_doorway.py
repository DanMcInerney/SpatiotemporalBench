"""`vacuum/2`'s doorway coin: the doorway ahead of the robot is blocked or clear by a coin, the world is made to say it,
and a blocked one is answered by skipping the room (docs/jobs.md, "The doorway coin").

**The coin.** For room `j` of task `i`, when its `before_room` decision is reached, `make_rng(m.name, seed, JOB_ID,
"doorway", i, j)`'s one draw is under `DOORWAY_COIN_P` for a blocked doorway. It reads no world. The opening is the one
`zones.zone_of` finds for the base at the decision (`opening_ahead`): the place a `zone_clear` question names, which
must be one of the room's.

**Clear before planning.** `clear_ways` runs after the gap's own changes and before the task is planned: every carried
thing and floor clutter piece standing in the way of any interior opening (`zone_shapes.doorway_way`, taller than
the zone's `height_m`) moves out by the world's own law (`world_processes_home.move_item`, `clutter.move_floor`) at
`t_edge`, the task's start less `changes.EDGE_BUFFER`. That restores the floor the last task's blockers closed, and it
is why a clear coin finds nothing there. A thing that finds no place is an error.

**Blocking after planning.** When a decision's coin says blocked, the task has been planned up to the decision, and
`Coins.close` moves one thing into the way (two for an opening a thing cannot close alone: `SINGLE_MAX_M`) at the same
`t_edge`, so no change is ever seen happening: a carried thing of the home's floor kinds (bag, box, shoes, toy), else a
floor clutter piece taller than the robot's climb. It skips `FloorGuard`, as the AMR's drum does: the next task's
clearing opens the doorway again. The things are drawn from `make_rng(.., "doorway_at", i, j)` and stand
- with their centres in the way and within `PLANE_M` of the wall's plane (near enough that the planner cannot go round
  them), centred across the gap within `OFFSET_M`, so that the widest free width left across the opening
  (`across_gaps`) is under `gap_limit`, the body's width plus twice its clearance: the planner cannot pass, and a
  `zone_clear` gold reads the thing;
- at least `keep_from_body` (the body's radius, its clearance and `KEEP_M`) from the path so far, the decision pose
  included (the robot turns round on the spot there), and clear of each other, of the other things, of the walls and the
  fixed pieces and of the open door leaf.
The path up to the decision was planned before any of this, whichever way the coin falls. A coin that finds no thing
or no spot, or whose doorway the planner still crosses, is a `ValueError`: the seed is skipped, never a doorway
that says blocked and is not. Light furniture is not moved: one that stood in a way would fail the job's check
(`job_vacuum2_checks`), as a person would not be counted (none stood in a way at a decision on seeds 101-130).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import clutter, zone_shapes, zones
from .changes import FURNITURE_MARGIN, World
from .geometry import Point, Rect, corners, dist_to_rect, distance, rects_overlap, to_local, to_world
from .job_vacuum import DOORWAY_STANDOFF_M
from .job_vacuum2_plan import DOORWAY_COIN_P, JOB_ID, ZONE
from .job_vacuum_path import crossing
from .keyframes import Hold, Move, Turn
from .mapspec import MapSpec, Opening, Region
from .placement import ITEM_MARGIN, Spot, free_of, occupied_entry
from .robots import Robot
from .route import FreeSpace, blocking_rects
from .seeding import make_rng
from .timeline import Timeline
from .validate import collision_class, door_leaf_rect, non_blocking
from .world_processes_home import floor_clutter, move_item

TRIES = 60  # arrangements a coin tries from each pool of things before it gives up
SINGLE_MAX_M = 1.0  # an opening this wide or more takes two things (the 1.2 m arch), a narrower one takes one
OFFSET_M = 0.1  # the things, taken together, are centred across the gap within this
PLANE_M = 0.12  # a thing's centre lies this near the wall's plane at most: close enough that the planner cannot pass it
PAIR_GAP_M = (0.05, 0.3)  # the space between the two things that close a wide opening
TURN_JITTER = 0.15  # radians a thing strays from lying along or across the wall
KEEP_M = 0.05  # a blocker keeps the body's clearance and this much more from the path so far
SAMPLE_M = 0.05  # the path is sampled this far apart when a blocker is checked against it
SOLID_MARGIN_M = 0.02  # clear of a wall, a fixed piece or the open leaf of the door


def doorway_blocked(m: MapSpec, seed: int, i: int, j: int) -> bool:
    """Whether the coin says the doorway into room `j` of task `i` is blocked: one draw from a stream of its own, no
    world."""
    return float(make_rng(m.name, seed, JOB_ID, "doorway", i, j).random()) < DOORWAY_COIN_P


def decision_mark(m: MapSpec, room: Region, poly: list[Point], last: bool) -> tuple[int, float]:
    """Where a doorway decision stands on a path: (segment index, metres from that segment's start),
    `DOORWAY_STANDOFF_M` of path before the room's doorway crossing (the last crossing when coming in, else the first),
    walking back over as many segments as it takes, or at the path's start when that is nearer. (`crossing` stops at
    the start of the segment that holds the crossing, which on a path that turns at the door stood 0.19 m from it: no
    blocker can lie in the way and stay clear of the body.)"""
    k, at = crossing(m, room, poly, 0.0, last)
    left = DOORWAY_STANDOFF_M
    while left > at and k > 0:
        left -= at
        k -= 1
        at = distance(poly[k], poly[k + 1])
    return k, max(at - left, 0.0)


def opening_ahead(m: MapSpec, xy: Point) -> Opening:
    """The interior opening whose way is nearest the base at `xy`: the doorway a `zone_clear` question at a decision
    there names (`zones.zone_of`, the one rule the gold reads)."""
    zone = zones.zone_of(m, ZONE, xy)
    if zone is None:
        raise ValueError(f"{m.name} has no interior opening")
    return next(o for o in m.openings if o.id == zone.piece)


def stands(way: Rect, rect: Rect, z0: float, z1: float) -> bool:
    """Whether a thing with this footprint and this span of heights stands in the way: taller than the zone's height
    and overlapping it (the rule a `zone_clear` gold reads, over the truth)."""
    return z1 > ZONE.height_m and z0 < ZONE.top_m and rects_overlap(rect, way)


def standing(w: World, way: Rect) -> list[str]:
    """The ids of the world's carried things and floor clutter standing in the way, sorted."""
    out = [eid for eid, c in w.carried.items() if stands(way, *occupied_entry(c.spec, c.spot.pos, c.spot.yaw))]
    out += [eid for eid, (rect, z0, z1) in w.clutter.items() if stands(way, rect, z0, z1)]
    return sorted(out)


def standing_in(tl: Timeline, m: MapSpec, o: Opening, t: float) -> list[str]:
    """The ids of the entities standing in the opening's way at time `t` in a script: what a robot's camera would find
    there, from the timeline alone (a door leaf, a floor covering and a person are not counted)."""
    way, out = zone_shapes.doorway_way(m, o), []
    for e in tl.entities:
        if collision_class(e) not in ("fixed", "light", "carried") or non_blocking(e):
            continue
        p = tl.placement_of(e.id, t)
        if p is not None and stands(way, Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw), p.pos[2],
                                    p.pos[2] + e.size[2]):
            out.append(e.id)
    return sorted(out)


def gap_limit(robot: Robot) -> float:
    """The narrowest gap the body goes through: its width plus twice its clearance (a rectangle as its circumscribed
    disc, as the planner reads it)."""
    body = robot.body
    width = body.length_m if body.shape == "disc" else math.hypot(body.length_m, body.width_m)
    return width + 2 * body.clearance_m


def keep_from_body(robot: Robot) -> float:
    """How far from the path the body's centre a blocker stands: its radius, its clearance and `KEEP_M`."""
    body = robot.body
    radius = body.length_m / 2 if body.shape == "disc" else math.hypot(body.length_m, body.width_m) / 2
    return radius + body.clearance_m + KEEP_M


def across_gaps(o: Opening, rects: list[Rect]) -> list[float]:
    """The free widths across the opening, jamb to jamb, between the shadows the rectangles cast on the wall's axis
    (left to right: before the first, between each two, after the last)."""
    frame = Rect(o.center[0], o.center[1], 0.0, 0.0, o.yaw)
    spans = []
    for r in rects:
        turn = r.yaw - o.yaw
        half = (abs(math.cos(turn)) * r.sx + abs(math.sin(turn)) * r.sy) / 2
        x = to_local(frame, r.cx, r.cy)[0]
        spans.append((x - half, x + half))
    edge, out = -o.width / 2, []
    for lo, hi in sorted(spans):
        out.append(max(lo - edge, 0.0))
        edge = max(edge, hi)
    return [*out, max(o.width / 2 - edge, 0.0)]


# -- before planning: the doorways are clear --------------------------------------------------------------------
def clear_ways(w: World, m: MapSpec, seed: int, i: int, t: float) -> None:
    """Move out of every interior opening's way each carried thing and floor clutter piece standing in it, at `t`: the
    module docstring says why. ValueError names a thing that finds no place."""
    rng = make_rng(m.name, seed, JOB_ID, "doorway_clear", i)
    ways = [zone_shapes.doorway_way(m, o) for o in m.openings if o.interior]
    for way in ways:
        for eid in standing(w, way):
            for _ in range(TRIES):
                move = move_item if eid in w.carried else clutter.move_floor
                moved = move(w, m, rng, eid, t)
                if moved and not any(eid in standing(w, other) for other in ways):
                    break
            else:
                raise ValueError(f"{eid} stands in the way of an opening and finds no other place")


# -- after planning: a blocked doorway -------------------------------------------------------------------------
def trail_of(prims: list) -> list[tuple[Point, Point]]:
    """The straight runs of the base over the primitives so far, as (from, to): a move is one, a turn or a hold on the
    spot is a run of no length."""
    return [(p.a, p.b) if isinstance(p, Move) else (p.at, p.at) for p in prims if isinstance(p, Move | Turn | Hold)]


def sampled(trail: list[tuple[Point, Point]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [np.empty(0)], [np.empty(0)]
    for a, b in trail:
        s = np.linspace(0.0, 1.0, max(math.ceil(math.dist(a, b) / SAMPLE_M), 1) + 1)
        xs.append(a[0] + (b[0] - a[0]) * s)
        ys.append(a[1] + (b[1] - a[1]) * s)
    return np.concatenate(xs), np.concatenate(ys)


def _sizes(w: World) -> dict[str, tuple[float, float, float]]:
    out = {eid: c.spec.size for eid, c in w.carried.items()}
    return out | {e.id: e.size for e in w.entities if e.id in w.clutter}


def _pools(w: World, robot: Robot, taken: set[str]) -> list[list[str]]:
    """The things that may close a doorway, in the order they are tried: the carried things of the home's floor kinds,
    then the floor clutter taller than the robot's climb (and the zone's height). Not those already moved this task."""
    floor = {kind for kind, _ in w.catalog.floor_kinds}
    carried = [eid for eid in sorted(w.carried) if w.carried[eid].spec.kind in floor and eid not in taken]
    sizes, low = _sizes(w), max(robot.body.climb_m, ZONE.height_m)
    pieces = [eid for eid in sorted(floor_clutter(w)) if sizes[eid][2] > low and eid not in taken]
    return [carried, pieces]


def _arrange(rng: np.random.Generator, o: Opening, way: Rect, sizes: list[tuple[float, float, float]]
             ) -> list[Rect] | None:
    """Footprints for the things in the way, side by side across the opening, centred on its gap within `OFFSET_M`,
    each with its centre in the way and within `PLANE_M` of the wall's plane and its width across inside the way's;
    None when a draw does not fit."""
    frame = Rect(o.center[0], o.center[1], 0.0, 0.0, o.yaw)  # x along the wall, y across it
    centre = to_local(frame, way.cx, way.cy)[1]
    turns = [float(rng.integers(2)) * math.pi / 2 + float(rng.uniform(-TURN_JITTER, TURN_JITTER)) for _ in sizes]
    extent = [(abs(math.cos(t)) * s[0] + abs(math.sin(t)) * s[1], abs(math.sin(t)) * s[0] + abs(math.cos(t)) * s[1])
              for t, s in zip(turns, sizes, strict=True)]
    shift, between = float(rng.uniform(-OFFSET_M, OFFSET_M)), float(rng.uniform(*PAIR_GAP_M))
    total = sum(e[0] for e in extent) + between * (len(sizes) - 1)
    at, xs = shift - total / 2, []
    for ex, _ in extent:
        xs.append(at + ex / 2)
        at += ex + between
    out = []
    for x, (ex, _), turn, size in zip(xs, extent, turns, sizes, strict=True):
        lo, hi = max(centre - way.sy / 2, -PLANE_M), min(centre + way.sy / 2, PLANE_M)
        if abs(x) + ex / 2 > way.sx / 2 or lo > hi:
            return None
        y = float(rng.uniform(lo, hi))
        cx, cy = to_world(frame, x, y)
        out.append(Rect(cx, cy, size[0], size[1], o.yaw + turn))
    return out


def _fits(w: World, m: MapSpec, o: Opening, ids: list[str], rects: list[Rect], heights: list[float]) -> bool:
    """Whether the things stand clear of each other, of the other things and light pieces of the world, of the walls and
    fixed pieces and of the open leaf of the door (each thing taken out of the world it came from)."""
    rest = w
    for eid in ids:
        rest = rest.without(eid)
    if any(rects_overlap(a, b, ITEM_MARGIN) for k, a in enumerate(rects) for b in rects[k + 1:]):
        return False
    solid = [*m.wall_pieces(), *blocking_rects(m)]
    if o.type == "door":
        thickness = next(e.size[1] for e in w.entities if e.id == w.door_entity[o.id])
        solid.append(door_leaf_rect(m, o, True, thickness))
    for rect, high in zip(rects, heights, strict=True):
        if not free_of(rect, 0.0, high, rest.occupied(), ITEM_MARGIN):
            return False
        if any(rects_overlap(rect, r, FURNITURE_MARGIN) for r in rest.light_rects.values()):
            return False
        if any(rects_overlap(rect, r, SOLID_MARGIN_M) for r in solid):
            return False
    return True


def _put(w: World, m: MapSpec, eid: str, rect: Rect, t: float) -> None:
    """The thing on the floor at the rectangle from `t` on: its ledger placement and its record in the world."""
    w.ledger.place(eid, t, (rect.cx, rect.cy, 0.0), rect.yaw, None)
    if eid in w.carried:
        w.carried[eid].spot = Spot(clutter.floor_slot(m, rect.cx, rect.cy), (rect.cx, rect.cy, 0.0), rect.yaw)
    else:
        _, z0, z1 = w.clutter[eid]
        w.clutter[eid] = (rect, 0.0, z1 - z0)


def block(w: World, m: MapSpec, robot: Robot, seed: int, i: int, j: int, o: Opening,
          trail: list[tuple[Point, Point]], t: float, taken: set[str]) -> list[str]:
    """Move the thing (or two) that close the opening into its way at `t`, as the module docstring says; the ids, added
    to `taken`. ValueError when no draw of any pool fits."""
    rng = make_rng(m.name, seed, JOB_ID, "doorway_at", i, j)
    way, limit, keep = zone_shapes.doorway_way(m, o), gap_limit(robot), keep_from_body(robot)
    n, sizes = (1 if o.width < SINGLE_MAX_M else 2), _sizes(w)
    px, py = sampled(trail)
    for pool in _pools(w, robot, taken):
        for _ in range(TRIES if len(pool) >= n else 0):
            ids = [pool[k] for k in rng.permutation(len(pool))[:n]]
            rects = _arrange(rng, o, way, [sizes[eid] for eid in ids])
            if rects is None or max(across_gaps(o, rects)) >= limit - 1e-6:
                continue
            if not all(rects_overlap(r, way) and dist_to_rect(px, py, r).min(initial=math.inf) >= keep
                       for r in rects):
                continue
            if _fits(w, m, o, ids, rects, [sizes[eid][2] for eid in ids]):
                for eid, rect in zip(ids, rects, strict=True):
                    _put(w, m, eid, rect, t)
                taken.update(ids)
                return ids
    raise ValueError(f"no floor thing closes the {o.id} (room {j})")


def sealed(free: FreeSpace, m: MapSpec, o: Opening) -> bool:
    """Whether no cell of the planner's free space lies in the opening's gap: nothing can cross it."""
    gap = free.polygon_mask(corners(o.rect(m.wall_thickness)))
    return not bool((free.free & gap).any())


@dataclass
class Coins:
    """One task's doorway coins as `_task_prims` meets them: the world, the instant the world changes at, the planner's
    free space as the world stands, and the things moved so far."""

    w: World
    m: MapSpec
    robot: Robot
    seed: int
    i: int
    t: float
    free: Callable[[], FreeSpace]
    taken: set[str] = field(default_factory=set)

    def blocked(self, j: int) -> bool:
        return doorway_blocked(self.m, self.seed, self.i, j)

    def close(self, j: int, room: str, xy: Point, trail: list[tuple[Point, Point]]) -> FreeSpace:
        """Make the doorway ahead of the base at `xy` blocked (as the module docstring says) and return the planner's
        free space with the blockers in it; ValueError when the doorway is no opening of the region `room`, or can
        still be crossed."""
        o = opening_ahead(self.m, xy)
        if room not in o.between:
            raise ValueError(f"the doorway ahead of {xy} is the {o.id}, which is no opening of the region {room}")
        block(self.w, self.m, self.robot, self.seed, self.i, j, o, trail, self.t, self.taken)
        free = self.free()
        if not sealed(free, self.m, o):
            raise ValueError(f"the planner still crosses the {o.id} with its blockers in it")
        return free
