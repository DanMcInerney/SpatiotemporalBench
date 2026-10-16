"""What the vacuum's job tests share: stand-in clutter (until R.2 fills `clutter.place`), generated scripts cached per
worker process, and the room and seed a script is drawn for.

`stand_in_clutter` has `clutter.place`'s signature. It drops 15-25 pieces of mobility `clutter` on open floor, so the
planner meets floor obstacles from the start: boxes 0.10-0.30 m across and 0.05-0.25 m high (the vacuum cannot drive
over them) and flat items 0.01 m high (it can). Each is an entity with a placement at t = 0 and an entry in
`world.clutter` as `placement.occupied_entry` makes it. They lie in a room, clear of fixed and light pieces, of floor
items, of the keep-out around every opening and the swing of its door, of the dock's apron and of each other. At R.2's
merge the coordinator re-runs the path tests with the real clutter.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import replace
from functools import cache
from types import SimpleNamespace
from unittest import mock

import numpy as np

from stbench import build, clutter, job_vacuum, job_vacuum2, job_vacuum2_backout, job_vacuum2_doorway
from stbench.catalog import FLOOR_ITEM_Z
from stbench.generate import generate_script
from stbench.geometry import Rect, corners, dist_to_rect, point_in_polygon, rects_overlap, to_world
from stbench.job_registry import get
from stbench.mapspec import DOCK_APRON_FRONT_M, DOCK_APRON_SIDE_M, INTERIOR_TOL, load_map
from stbench.placement import occupied_entry
from stbench.robots import ROBOTS
from stbench.route import blocking_rects, robot_free_space
from stbench.timeline import Entity

MAP, ROBOT, JOB = "apartment_a", "vacuum", "vacuum/1"
ROOMS = ("kitchen", "living room", "bedroom", "bathroom")  # the rooms `vacuum/1` draws a task's room from
GAP_M = 0.05  # clear floor between a stand-in piece and what is near it
OUTSIDE_THE_DOCK = (5.44, 5.05)  # in the hall, 0.5 m in front of the docked vacuum: where every task's path leaves from


def stand_in_clutter(world, m, rng: np.random.Generator) -> None:
    """15-25 pieces of clutter in `world` (the signature of `clutter.place`), drawn from `rng`."""
    floor_items = [r for r, z0, _ in world.occupied() if z0 < FLOOR_ITEM_Z]
    blocked = blocking_rects(m) + list(world.light_rects.values()) + floor_items
    aprons = [Rect(*to_world(f.footprint(), DOCK_APRON_FRONT_M / 2, 0.0), f.size[0] + DOCK_APRON_FRONT_M,
                   f.size[1] + 2 * DOCK_APRON_SIDE_M, f.yaw) for f in m.furniture if f.kind == "dock"]
    placed: list[Rect] = []
    for i in range(int(rng.integers(15, 26))):
        flat = i % 3 == 2
        high = 0.01 if flat else float(rng.uniform(0.05, 0.25))
        size = (float(rng.uniform(0.10, 0.30)), float(rng.uniform(0.10, 0.30)), high)
        for _ in range(200):
            room = m.regions[int(rng.integers(len(m.regions)))]
            xs, ys = [p[0] for p in room.polygon], [p[1] for p in room.polygon]
            rect = Rect(float(rng.uniform(min(xs), max(xs))), float(rng.uniform(min(ys), max(ys))), size[0], size[1],
                        float(rng.uniform(-math.pi, math.pi)))
            if (all(point_in_polygon(x, y, list(room.polygon)) for x, y in corners(rect, INTERIOR_TOL))
                    and not any(rects_overlap(rect, r, GAP_M) for r in blocked + aprons + placed)
                    and not any(_near_opening(m, o, rect) for o in m.openings)):
                break
        else:
            continue
        placed.append(rect)
        eid = world.new_entity_id()
        world.entities.append(Entity(eid, "box", {"color": "brown"}, "clutter", size))
        world.ledger.place(eid, 0.0, (rect.cx, rect.cy, 0.0), rect.yaw, None)
        world.clutter[eid] = occupied_entry(SimpleNamespace(size=size), (rect.cx, rect.cy, 0.0), rect.yaw)


def _near_opening(m, opening, rect: Rect) -> bool:
    """Within the band of an opening and, for a door, the quarter disc its leaf sweeps (a radius of its width from the
    hinge, which lies at most 1.5 widths from the centre)."""
    reach = opening.width * 1.5 + GAP_M
    return float(dist_to_rect(np.array([opening.center[0]]), np.array([opening.center[1]]), rect)[0]) < reach


@cache
def apartment():
    return load_map(MAP)


def robot():
    return ROBOTS[ROBOT]


@cache
def generated(seed: int):
    """The seed's script and the world it was planned in, with stand-in clutter, cached per worker process: neither is
    mutated by a test."""
    caught = {}
    real = job_vacuum.SPEC

    def script(m, w, plans, seed_, **kwargs):
        caught["world"] = w
        return real.script(m, w, plans, seed_, **kwargs)

    with mock.patch.object(clutter, "place", stand_in_clutter), mock.patch.object(
            job_vacuum, "SPEC", replace(real, script=script)):
        tl = generate_script(MAP, seed, ROBOT, JOB, 1)
    return caught["world"], tl


def script(seed: int, clutter_on: bool = True):
    """The one-task script of the seed on `apartment_a`, with stand-in clutter or with none."""
    return generated(seed)[1] if clutter_on else generate_script(MAP, seed, ROBOT, JOB, 1)


def free_space(world):
    """The vacuum's free space in a generated world."""
    return robot_free_space(apartment(), world, robot())


@cache
def room_of(seed: int) -> str:
    """The room `vacuum/1` draws the seed's task for."""
    return str(get(JOB).plan(load_map(MAP), seed, 1)[0].params["room"])


@cache
def seeds_for(room: str, n: int = 3) -> tuple[int, ...]:
    """The first `n` seeds (from 1) whose task is to clean `room`."""
    found, seed = [], 0
    while len(found) < n:
        seed += 1
        if room_of(seed) == room:
            found.append(seed)
    return tuple(found)


# -- vacuum/2: scripts of 4-6 tasks with the real clutter ----------------------------------------------------------
JOB2, TASKS2 = "vacuum/2", (4, 6)


@cache
def generated2(seed: int, sites: tuple[str, ...] = (), coins=None) -> tuple[list, object]:
    """The seed's `vacuum/2` script on `apartment_a` with the real clutter (`clutter.place`), its task count the seed's
    own in 4-6 (`build.tasks_for`), and a copy of the world as each task left it once planned (after the gap callback,
    the clearing of the doorways and the blockers moved in, which a skip's path was planned with), so a test can ask
    `robot_obstacles` of the world a task saw. `sites` are the piece kinds the vacuum backs out of
    (`job_vacuum2_backout.SITE_KINDS`, none as shipped),
    and `coins` says which doorway coins are blocked, `(map, seed, task, room) -> bool` (None: the seed's own; the
    back-out tests ask for none blocked, so every room is cleaned). Cached per worker process; a test mutates
    neither."""
    worlds: list = []
    held: list = []
    real_clear, real_frames = job_vacuum2_doorway.clear_ways, job_vacuum2.frames

    def clearing(w, *args):
        held[:] = [w]
        real_clear(w, *args)

    def framing(*args, **kwargs):  # called once a task, right after it is planned
        worlds.append(copy.deepcopy(held[0]))
        return real_frames(*args, **kwargs)

    n = build.tasks_for(TASKS2, MAP, seed, ROBOT, JOB2)
    with mock.patch.object(job_vacuum2_doorway, "clear_ways", clearing), mock.patch.object(
            job_vacuum2, "frames", framing), mock.patch.object(job_vacuum2_backout, "SITE_KINDS", sites), forced(coins):
        tl = generate_script(MAP, seed, ROBOT, JOB2, n)
    return worlds, tl


def never(_m, _seed, _i, _j) -> bool:
    """A doorway coin that is never blocked: every room of every task is entered and cleaned."""
    return False


def forced(coins):
    """A context in which the doorway coins are `coins(map, seed, task, room) -> bool`, or the seed's own when None."""
    if coins is None:
        return contextlib.nullcontext()
    return mock.patch.object(job_vacuum2_doorway, "doorway_blocked", coins)
