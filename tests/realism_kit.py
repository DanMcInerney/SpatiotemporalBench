"""The helpers of the realism tests (lane R.2): the apartment with its clutter placed, the timeline of that world with a
docked vacuum, and the plain geometry the tests check the clutter against, written out here from the map and the design
(the apron's 1.2 m and 0.5 m, a door's band of 0.5 m, the swept quarter disc of a leaf) and not read from the code under
test. Everything is cached per worker process, so a test never builds a world twice and never changes one."""

from __future__ import annotations

import math
from functools import cache

import numpy as np

from builders import hand_job, keyframe, start_status
from stbench import SEED_KEY, job_registry
from stbench.clutter import place
from stbench.generate import build_world, light_starts, to_timeline
from stbench.geometry import Rect, point_in_rect, to_local, to_world
from stbench.mapspec import MapSpec, load_map
from stbench.robots import ROBOTS
from stbench.seeding import make_rng
from stbench.timeline import Ego, EpisodeInfo, Task, TaskPlan, Timeline
from world_kit import world_timeline

MAP = "apartment_a"
DOCKED_BASE = (4.94, 5.05, 0.0, 0.0)  # the vacuum on its dock: the pose of the look procedure's "docked"
APRON_FRONT_M, APRON_SIDE_M, DOOR_BAND_M = 1.2, 0.5, 0.5  # iRobot's home-base guidance; the plan's band before a door


@cache
def apartment() -> MapSpec:
    return load_map(MAP)


def clutter_rng(seed: int) -> np.random.Generator:
    """The stream lane A hands `place`: the map's name, the seed and the generator's frozen key, then "clutter"."""
    return make_rng(MAP, seed, SEED_KEY, "clutter")


@cache
def cluttered(seed: int):
    """The map's world for `seed` with its clutter placed (`build_world`, then `place`): shared, so never edited."""
    m = apartment()
    w = build_world(m, seed)
    place(w, m, clutter_rng(seed))
    return w


@cache
def generated_world(seed: int):
    """The world `generate_script` makes of a seed before it plans a path: built, cluttered, and the light furniture
    started (the vacuum's dock the anchor): the world of a look-dev script that a probe can rebuild."""
    m = apartment()
    w = build_world(m, seed)
    place(w, m, clutter_rng(seed))
    light_starts(w, m, make_rng(MAP, seed, SEED_KEY, "light_start"), job_registry.get("vacuum/1").anchors)
    return w


@cache
def clutter_timeline(seed: int) -> Timeline:
    """`cluttered(seed)` as a timeline with no ego."""
    return world_timeline(apartment(), cluttered(seed), seed)


def script(seed: int, robot: str, poses: list[tuple[float, float, float]], room: str = "hall",
           world=None) -> Timeline:
    """The cluttered world (or `world`, e.g. `generated_world(seed)`) under a one-task script of `robot` with its base
    at each pose (x, y, yaw in degrees), one second apart, each in the robot's start set, stamped with
    `builders.hand_job`; the task is the vacuum's first kind, or `hand` for another robot."""
    m, r = apartment(), ROBOTS[robot]
    kfs = [keyframe(robot, k, float(k), (x, y, 0.0), math.radians(yaw), task=0, phase=r.status.start["phase"],
                    status=start_status(robot)) for k, (x, y, yaw) in enumerate(poses)]
    job_id = hand_job(robot)
    try:
        plan = TaskPlan(job_registry.get(job_id).task_kinds[0], {"room": room})
    except KeyError:
        plan = TaskPlan("hand", {"room": room})
    info = EpisodeInfo(m.name, seed, robot, job_id, "2026-03-02T08:00:00")
    return to_timeline(m, world or cluttered(seed), Ego(kfs, [Task(0, 0, len(kfs), plan, ())], []), info)


@cache
def docked_script(seed: int) -> Timeline:
    """The cluttered world under a one-keyframe script: the vacuum docked, one `clean_room` task of the hall."""
    x, y, _, yaw = DOCKED_BASE
    return script(seed, "vacuum", [(x, y, math.degrees(yaw))])


def clutter_ids(tl: Timeline) -> list[str]:
    return [e.id for e in tl.entities if e.mobility == "clutter"]


def footprint(tl: Timeline, eid: str) -> Rect:
    """An entity's footprint at t 0 (the clutter is static)."""
    p, e = tl.placement_of(eid, 0.0), tl.entity(eid)
    return Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)


def on_floor(tl: Timeline, eid: str) -> bool:
    """Whether the entity lies on the floor (no support, base at 0)."""
    p = tl.placement_of(eid, 0.0)
    return p.on is None and p.pos[2] < 1e-9


def samples(rect: Rect, n: int = 5) -> list[tuple[float, float]]:
    """An n x n grid of points over a footprint, its edges and corners included."""
    return [to_world(rect, lx, ly) for lx in np.linspace(-rect.sx / 2, rect.sx / 2, n)
            for ly in np.linspace(-rect.sy / 2, rect.sy / 2, n)]


def keepout_hits(m: MapSpec, rect: Rect) -> list[str]:
    """What a footprint touches of the places nothing may stand: a doorway or the band 0.5 m out from both faces of its
    wall across its width, the quarter disc a door leaf sweeps from closed to open, and the dock's apron (1.2 m in front
    of the dock, 0.5 m each side), tested at `samples` of the footprint."""
    pts, hits = samples(rect), []
    for o in m.openings:
        band = Rect(*o.center, o.width, m.wall(o.wall).thickness + 2 * DOOR_BAND_M, o.yaw)
        if any(point_in_rect(band, x, y) for x, y in pts):
            hits.append(f"the band of {o.id}")
        if o.type == "door":
            hinge, closed = m.door_leaf(o, False)
            _, tip = m.door_leaf(o, True)
            for x, y in pts:
                inside = math.hypot(x - hinge[0], y - hinge[1]) <= o.width
                for end in (closed, tip):  # the leaves are at a right angle: the sweep is a quadrant
                    inside &= (x - hinge[0]) * (end[0] - hinge[0]) + (y - hinge[1]) * (end[1] - hinge[1]) >= 0
                if inside:
                    hits.append(f"the swing of {o.id}")
                    break
    for dock in (f for f in m.furniture if f.kind == "dock"):
        fp = dock.footprint()
        for x, y in pts:
            lx, ly = to_local(fp, x, y)
            if -fp.sx / 2 <= lx <= fp.sx / 2 + APRON_FRONT_M and abs(ly) <= fp.sy / 2 + APRON_SIDE_M:
                hits.append("the dock's apron")
                break
    return hits
