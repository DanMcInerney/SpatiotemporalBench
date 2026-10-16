"""World building for the script generator: `build_world(map, seed)` and the assembly of a `Timeline` from a world.

`build_world` draws the initial layout from a PCG64 generator seeded by `(map, seed, SEED_KEY)`: the map's furniture and
doors, then the sampled carried items (identical sets first, each on one surface). The map's catalog
(`catalog.for_map`) decides what the world holds: how many items of which kinds. SEED_KEY is frozen, so nothing
redraws an item. Interior doors are open from time 0 and the entrance is shut; each door
entity carries its opening's `hinge` and `swing`, so the open leaf follows from its placement and state without the
map. A piece's `under_m` (the height of its open underside) goes into its entity.

`generate_script` is a script: that world, then `clutter.place`, then the light furniture's own starts (`light_starts`,
whose one-piece law `light_start` the gaps draw from again), then the path, status and decision points of a robot's job
over it (`job_registry`), which calls back
`world_processes.gaps` once before each task for the world's changes in the gaps, all under `to_timeline`, and last
the people overlay (`people.add`) on the finished timeline. Each step draws from its own key (`SEED_KEY`, "clutter",
"light_start" and "people"; the job's own; the gaps' per call), so changing one redraws no other.
`to_timeline` is the one assembly the script generator, the tests' hand-made scripts and `tests/synth.py` share: the
world's regions (with their kind), entities (carried and light ones named for their asset model) and the ledger's
placements and states, under an ego the caller built. The clock of a script or a case is drawn by
`case.start_clock`, which owns the distribution; `EpisodeInfo` carries the script's.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from . import SEED_KEY, asset_manifest, case, clutter, job_registry, people, world_processes
from .catalog import FLOOR_ITEM_Z
from .changes import FURNITURE_MARGIN, Carried, World
from .clutter_reach import FloorGuard
from .dressing_boxes import WALL_PROUD_MAX_M
from .geometry import Rect, corners, dist_to_rect, point_in_polygon, rects_overlap, to_world, under_obstacles
from .ledger import Ledger
from .mapspec import DOCK_APRON_FRONT_M, DOCK_APRON_SIDE_M, INTERIOR_TOL, MapSpec, load_map
from .placement import ITEM_MARGIN, ItemSpec, Spot, build_slots, find_spot, occupied_entry, place_set, sample_plan
from .route import blocking_rects
from .seeding import make_rng
from .timeline import Entity, EpisodeInfo, Region, Timeline
from .timeline_ego import Ego
from .validate import door_leaf_rect

DOOR_THICKNESS = 0.04
CHAIR_KIND = "chair"
CHAIR_PULL_M, CHAIR_YAW_DEG, SHIFT_M = (0.0, 0.6), 15.0, (0.3, 1.0)  # a chair's pull-out and turn; others' shift
LIGHT_TRIES = 60  # draws a piece of light furniture gets to find a valid start before it keeps its map place
ARCH_CLEAR_M = 0.3  # the floor a start keeps clear in front of an arch's gap (a door's keep-out is `mapspec`'s)


def build_world(m: MapSpec, seed: int) -> World:
    """Entities and initial placements: furniture and doors from the map, then the sampled carried items: each
    identical set together first (`place_set`), then every other item, one by one."""
    ledger = Ledger()
    w = World(m, ledger, [], [])
    entity_of: dict[str, str] = {}
    for f in sorted(m.furniture, key=lambda f: f.mobility != "fixed"):  # fixed first, then light, in map order
        eid = w.new_entity_id()
        entity_of[f.id] = eid
        attrs: dict[str, object] = {"name": f.name, "color": f.color}
        if f.surfaces:
            attrs["receptacles"] = sorted({s.kind for s in f.surfaces})
        w.entities.append(Entity(eid, f.kind, attrs, f.mobility, f.size, under_m=f.under_m))
        ledger.place(eid, 0.0, (f.center[0], f.center[1], 0.0), f.yaw, None)
        if f.mobility == "light":
            w.light_rects[eid] = f.footprint()
    names = {r.id: r.name for r in m.regions} | {"outside": "outside"}
    for d in m.doors():
        eid = w.new_entity_id()
        w.door_entity[d.id] = eid
        attrs = {"between": [names[d.between[0]], names[d.between[1]]], "hinge": d.hinge, "swing": d.swing}
        w.entities.append(Entity(eid, "door", attrs, "fixed", (d.width, DOOR_THICKNESS, d.height)))
        ledger.place(eid, 0.0, (d.center[0], d.center[1], 0.0), d.yaw, None)
        ledger.set_state(eid, "open", 0.0, bool(d.interior))  # interior doors stay open throughout; the entrance shut
    w.slots = build_slots(m, entity_of)

    rng = make_rng(m.name, seed, SEED_KEY, "items")
    occupied: list = []
    plan = sample_plan(rng, catalog=w.catalog)
    for specs in plan.sets:
        for spec, spot in zip(specs, place_set(rng, specs, w.slots, occupied, w.catalog), strict=True):
            add_item(w, spec, spot, occupied)
    for spec in plan.others:
        add_item(w, spec, find_spot(rng, spec, w.slots, occupied, catalog=w.catalog), occupied)
    return w


def add_item(w: World, spec: ItemSpec, spot: Spot | None, occupied: list) -> None:
    """A carried item at its starting spot (an item that found no room is left out)."""
    if spot is None:
        return
    eid = w.new_entity_id()
    w.entities.append(Entity(eid, spec.kind, dict(spec.attributes), "carried", spec.size))
    w.ledger.place(eid, 0.0, spot.pos, spot.yaw, spot.slot.furniture_entity)
    w.carried[eid] = Carried(eid, ItemSpec(spec.kind, dict(spec.attributes)), spot)
    w.used_keys.add(spec.key)
    occupied.append(occupied_entry(spec, spot.pos, spot.yaw))


def with_asset(e: Entity) -> Entity:
    """Carried and light entities name the manifest model of their kind in their colour (so an item added later does
    too, and a description always has the same model); furniture, doors and structure have none."""
    if e.mobility not in ("carried", "light"):
        return e
    return replace(e, asset=asset_manifest.model_for(e.kind, e.attributes.get("color")).id)


def to_timeline(m: MapSpec, w: World, ego: Ego, info: EpisodeInfo) -> Timeline:
    """The world as it stands (its regions, entities and the ledger's placements and states) under `ego`."""
    placements, states = w.ledger.finish()
    return Timeline(
        episode=info,
        regions=[Region(r.id, r.name, r.kind, r.polygon, r.level, r.z0, r.z1) for r in m.regions],
        entities=[with_asset(e) for e in w.entities],
        placements=placements,
        states=states,
        ego=ego,
    )


# -- a script: the world, clutter, light-furniture starts and a robot's job over them ------------------------------
def generate_script(map_name: str, seed: int, robot_id: str, job_id: str, n_tasks: int) -> Timeline:
    """The script of `n_tasks` tasks of job `job_id` for robot `robot_id` on a map, from its seed: the built world, its
    clutter, the light furniture's starts, then the job's plan and path (with the world's gap callback), then the people
    over the finished script. A job of another robot is refused."""
    m = load_map(map_name)
    job = job_registry.get(job_id)
    if job.robot != robot_id:
        raise ValueError(f"job {job_id} is {job.robot}'s, not {robot_id}'s")
    w = build_world(m, seed)
    clutter.place(w, m, make_rng(m.name, seed, SEED_KEY, "clutter"))
    light_starts(w, m, make_rng(m.name, seed, SEED_KEY, "light_start"), job.anchors)
    plans = job.plan(m, seed, n_tasks)
    ego = job.script(m, w, plans, seed, between=world_processes.gaps(w, m, seed))
    start = case.start_clock(make_rng(m.name, seed, robot_id, job_id, "start_clock"))
    info = EpisodeInfo(m.name, seed, robot_id, job_id, start)
    return people.add(to_timeline(m, w, ego, info), m, make_rng(m.name, seed, SEED_KEY, "people"))


def light_starts(w: World, m: MapSpec, rng: np.random.Generator, anchors: frozenset[str] = frozenset()) -> None:
    """Each piece of light furniture starts the script at a place of its own (`light_start`, at t = 0), in the map's
    order, from one stream."""
    for eid in list(w.light_rects):
        light_start(w, m, rng, eid, anchors)


def map_place(w: World, m: MapSpec, eid: str) -> Rect:
    """Where the map puts a piece of light furniture, wherever a start or a gap has since put it: `build_world` makes
    the light entities in the map's order."""
    pieces = [e.id for e in w.entities if e.mobility == "light"]
    return [f for f in m.furniture if f.mobility == "light"][pieces.index(eid)].footprint()


def light_start(w: World, m: MapSpec, rng: np.random.Generator, eid: str, anchors: frozenset[str], t: float = 0.0
                ) -> bool:
    """A new place for one piece of light furniture from `t` on, drawn from its map place (`map_place`), never from
    where it stands: a chair is pulled out `CHAIR_PULL_M` against its facing direction and turned up to `CHAIR_YAW_DEG`
    either way; every other piece moves `SHIFT_M` in a uniform direction to a uniform yaw. A place is valid when
    `start_valid` holds (the piece's corners lie in its own room and it is clear of blocking fixed pieces
    (`FURNITURE_MARGIN`), other light pieces, the open leaf of every interior door (`FURNITURE_MARGIN`: a door opens
    and closes between tasks, and its leaf must not strike a piece), the walls (`WALL_PROUD_MAX_M`, where the dressing
    guard assumes furniture stays), the keep-out of every interior door (`m.door_keepout`) and the gap of every arch
    (`ARCH_CLEAR_M` each side: an arch's keep-out circle would freeze the dining chair beside the kitchen's arch at its
    map place), the apron of every piece of one of the `anchors` kinds (a robot's dock) and floor items and clutter
    (only its legs, for a piece on legs: a toy may lie under a seat, as the world puts them)); after `t` = 0 it
    must also split no planned robot's free floor (`FloorGuard.allows_piece`, the doors in their state at `t`, the piece
    itself not in the world). After `LIGHT_TRIES` draws without one the piece stays where it is and this is False.
    The new place is written at `t` and into `w.light_rects`."""
    entity = next(e for e in w.entities if e.id == eid)
    old, guard = map_place(w, m, eid), None
    for _ in range(LIGHT_TRIES):
        if entity.kind == CHAIR_KIND:
            pull, turn = float(rng.uniform(*CHAIR_PULL_M)), math.radians(CHAIR_YAW_DEG)
            new = Rect(old.cx - pull * math.cos(old.yaw), old.cy - pull * math.sin(old.yaw), old.sx, old.sy,
                       old.yaw + float(rng.uniform(-turn, turn)))
        else:
            angle, dist = float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(*SHIFT_M))
            new = Rect(old.cx + dist * math.cos(angle), old.cy + dist * math.sin(angle), old.sx, old.sy,
                       float(rng.uniform(-math.pi, math.pi)))
        if not start_valid(w, m, eid, old, new, anchors, entity.under_m):
            continue
        if t > 0.0:
            guard = guard or FloorGuard(w.without(eid), t)
            if not guard.allows_piece(new, entity.size[2], entity.under_m):
                continue
        w.ledger.place(eid, t, (new.cx, new.cy, 0.0), new.yaw, None)
        w.light_rects[eid] = new
        return True
    return False


def start_valid(w: World, m: MapSpec, eid: str, old: Rect, new: Rect, anchors: frozenset[str],
                under_m: float = 0.0) -> bool:
    """Whether `new` is a valid place for piece `eid`, whose own place is `old` (its room): see `light_start`."""
    room = next((r for r in m.regions if point_in_polygon(old.cx, old.cy, list(r.polygon))), None)
    if room is None or not all(point_in_polygon(x, y, list(room.polygon)) for x, y in corners(new, INTERIOR_TOL)):
        return False
    aprons = [Rect(*to_world(f.footprint(), DOCK_APRON_FRONT_M / 2, 0.0), f.size[0] + DOCK_APRON_FRONT_M,
                   f.size[1] + 2 * DOCK_APRON_SIDE_M, f.yaw) for f in m.furniture if f.kind in anchors]
    others = [r for i, r in w.light_rects.items() if i != eid]
    leaves = [door_leaf_rect(m, o, True, DOOR_THICKNESS) for o in m.toggling_doors()]
    stands = under_obstacles(new, under_m)  # what touches the floor: the legs of a piece with an open underside
    floor = [r for r, z0, _ in w.occupied() if z0 < FLOOR_ITEM_Z]
    if (any(rects_overlap(new, r, FURNITURE_MARGIN) for r in blocking_rects(m) + others + leaves)
            or any(rects_overlap(new, r, WALL_PROUD_MAX_M) for r in m.wall_pieces())
            or any(rects_overlap(leg, r, ITEM_MARGIN) for leg in stands for r in floor)
            or any(rects_overlap(new, a) for a in aprons)):
        return False
    for o in (o for o in m.openings if o.interior):
        if o.type == "door":  # its leaf swings and it is approached: the keep-out round its centre
            c, radius = m.door_keepout(o)
            if float(dist_to_rect(np.array([c[0]]), np.array([c[1]]), new)[0]) < radius:
                return False
        elif rects_overlap(new, o.rect(m.wall_thickness + 2 * ARCH_CLEAR_M)):  # an arch: its gap, and the way to it
            return False
    return True
