"""What the people tests share: the apartment's generated `vacuum/1` scripts, and a multi-task script with gap changes
that stands in for `vacuum/2` until lane A2 merges. Everything is cached per worker process; a test never edits one.

`multi_task` is the robot's real job (`vacuum/1`'s `script`, planned on the world as each gap leaves it, as A2 plans)
three times in a row, 30 minutes apart, with `changes`' own processes inside each gap (light furniture shifted, carried
items moved, a delivery or a removal) and the bathroom's door shut for 10 minutes of it. It holds no resident:
`people.add` is the test's call. `obstacles_at` is the standing world at one instant, written out here from the
timeline and not read from `people`, for the tests that check a resident against it.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cache
from unittest.mock import patch

import numpy as np

from conftest import script_on_map
from stbench import SEED_KEY, clutter, job_registry, people, people_downtown, people_warehouse
from stbench.catalog import FLOOR_ITEM_Z
from stbench.changes import add_or_remove, move_carried, shift_light_furniture
from stbench.generate import build_world, generate_script, light_starts, to_timeline
from stbench.geometry import Rect, q
from stbench.mapspec import MapSpec, load_map
from stbench.robots import ROBOTS
from stbench.route import blocking_rects
from stbench.seeding import make_rng
from stbench.timeline import Ego, EpisodeInfo, TaskPlan, Timeline
from stbench.validate import door_entities, door_leaf_rect

MAP = "apartment_a"
GAP_S = 1800.0  # between a task's last keyframe and the next one's first
ROOMS = ("living room", "kitchen", "bedroom")  # the three tasks of `multi_task`
DOOR_SHUT_S = (300.0, 900.0)  # the bathroom's door is shut from here to there after the gap's start


@cache
def apartment() -> MapSpec:
    return load_map(MAP)


@cache
def generated(seed: int) -> Timeline:
    """The one-task `vacuum/1` script of the seed as `generate_script` makes it, a resident home or not."""
    return generate_script(MAP, seed, "vacuum", "vacuum/1", 1)


def people_rng(seed: int, key: str = "people") -> np.random.Generator:
    return make_rng(MAP, seed, SEED_KEY, key)


@cache
def multi_task(seed: int) -> Timeline:
    """Three `vacuum/1` tasks (`ROOMS`) over one world that changes in the gaps between them; no resident."""
    m, job = apartment(), job_registry.get("vacuum/1")
    w = build_world(m, seed)
    clutter.place(w, m, make_rng(MAP, seed, SEED_KEY, "clutter"))
    light_starts(w, m, make_rng(MAP, seed, SEED_KEY, "light_start"), job.anchors)
    bathroom_door = w.door_entity["D3"]
    keyframes, tasks, decisions = [], [], []
    start = 0.0
    for i, room in enumerate(ROOMS):
        if i:
            lo, hi, rng = start, start + GAP_S, make_rng(MAP, seed, "people kit", i)
            shift_light_furniture(w, rng, lo, hi, np.empty((0, 2)))
            occupied = w.occupied()
            add_or_remove(w, rng, lo, hi, move_carried(w, rng, lo, hi, occupied), occupied)
            w.ledger.set_state(bathroom_door, "open", lo + DOOR_SHUT_S[0], False)
            w.ledger.set_state(bathroom_door, "open", lo + DOOR_SHUT_S[1], True)
            start = hi
        task_ego = job.script(m, w, [TaskPlan("clean_room", {"room": room})], seed * 10 + i)
        k0 = len(keyframes)
        keyframes += [replace(kf, k=k0 + kf.k, t=q(start + kf.t), task=i) for kf in task_ego.keyframes]
        tasks.append(replace(task_ego.tasks[0], index=i, k0=k0, k1=len(keyframes)))
        decisions += [replace(d, k=k0 + d.k) for d in task_ego.decisions]
        start = keyframes[-1].t
    info = EpisodeInfo(MAP, seed, "vacuum", "vacuum/1", "2026-03-02T08:00:00")
    return to_timeline(m, w, Ego(keyframes, tasks, decisions), info)


def obstacles_at(tl: Timeline, m: MapSpec, t: float) -> list[Rect]:
    """What stands on the floor at time `t`, as rectangles: blocking fixed furniture, light furniture, things lying on
    the floor and the door leaves in the state they have at `t`, read from the timeline."""
    rects = blocking_rects(m)
    for e in tl.entities:
        p = tl.placement_of(e.id, t)
        if p is None or e.mobility not in ("light", "carried", "clutter"):
            continue
        if e.mobility == "light" or p.pos[2] < FLOOR_ITEM_Z:
            rects.append(Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw))
    for door_id, eid in door_entities(tl, m).items():
        opening = next(o for o in m.doors() if o.id == door_id)
        rects.append(door_leaf_rect(m, opening, bool(tl.state_of(eid, t, "open")), tl.entity(eid).size[1]))
    return rects


def gaps(tl: Timeline) -> list[tuple[float, float]]:
    """(last keyframe of a task, first keyframe of the next) for each gap of the script."""
    tasks = tl.ego.tasks
    return [(tl.keyframe(a.k1 - 1).t, tl.keyframe(b.k0).t) for a, b in zip(tasks, tasks[1:], strict=False)]


def resident(tl: Timeline) -> str | None:
    """The id of the script's person entity, None when it has none."""
    return next((e.id for e in tl.entities if e.kind == "person"), None)


def person_ids(tl: Timeline) -> list[str]:
    """The ids of every person entity of the script."""
    return [e.id for e in tl.entities if e.kind == "person"]


def footprints(tl: Timeline, eid: str) -> list[tuple[float, float, Rect, float]]:
    """(from, until, footprint, height) of each placement of the entity, as the placements say."""
    e = tl.entity(eid)
    return [(p.t0, math.inf if p.t1 is None else p.t1, Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw), p.pos[2])
            for p in tl.placements_of(eid)]


def drone_hover(n: int, z: float = 7.0, at: tuple[float, float] = (50.0, -8.0), bare: bool = False) -> Timeline:
    """A hand script on `downtown_a`: the drone hovers `n` keyframes, a second apart, `z` m over `at` (the plaza, clear
    of every drop zone), facing east. It has no job's task, so no job check applies. `bare` sets the world's own things
    (items and light pieces: the pads' street furniture, the cars) aside, for the walks that pin one rule at its
    edge."""
    tl = script_on_map("downtown_a", 1, "drone", [(at[0], at[1], z, 0.0)] * n)
    if bare:
        keep = {e.id for e in tl.entities if e.mobility not in ("carried", "light", "clutter")}
        tl = replace(tl, entities=[e for e in tl.entities if e.id in keep],
                     placements=[p for p in tl.placements if p.entity in keep])
    return tl


def amr_drive(n: int) -> Timeline:
    """A hand script on `warehouse_a`: the AMR drives east along the dock at y = 6.5 from x = 3, 0.2 m a keyframe."""
    return script_on_map("warehouse_a", 1, "amr", [(3.0 + 0.2 * k, 6.5, 0.0, 0.0) for k in range(n)])


@cache
def with_people(catalog: str) -> tuple[MapSpec, Timeline]:
    """The catalog's hand script (`amr_drive`, `drone_hover`) with its people overlaid, present for every task: the
    warehouse's three pickers, the downtown's pedestrians cut to two (a walk costs a million-cell grid)."""
    mod, name, tl, count = {"warehouse": (people_warehouse, "warehouse_a", amr_drive(60), 3),
                            "downtown": (people_downtown, "downtown_a", drone_hover(60), 2)}[catalog]
    m = load_map(name)
    with patch.object(mod, "P_HOME", 1.0), patch.object(mod, "COUNT", count):
        return m, people.add(tl, m, make_rng(name, 7, SEED_KEY, "people"))


def perimeter(rect: Rect, step: float = 0.01) -> np.ndarray:
    """Points every `step` metres along the rectangle's outline, corners included, as an (n, 2) array."""
    hx, hy = rect.sx / 2, rect.sy / 2
    corners_ = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy), (hx, hy)]
    local = []
    for (x0, y0), (x1, y1) in zip(corners_, corners_[1:], strict=False):
        n = max(int(math.hypot(x1 - x0, y1 - y0) / step), 1)
        local += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    return np.array([(rect.cx + c * x - s * y, rect.cy + s * x + c * y) for x, y in local])


def segment_distance(points: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> float:
    """The least distance from any of the points to the closed segment a-b."""
    ab = np.array(b) - np.array(a)
    length2 = float(ab @ ab)
    t = np.zeros(len(points)) if length2 == 0.0 else np.clip((points - np.array(a)) @ ab / length2, 0.0, 1.0)
    return float(np.hypot(*(points - (np.array(a) + t[:, None] * ab)).T).min())


def robot_gaps(tl: Timeline, eid: str | None = None) -> list[float]:
    """For each keyframe at which the person (the first, unless `eid` names one) exists and is within 2 m of the robot,
    the gap from its footprint to the robot's body at that keyframe and to the body swept to the keyframe before it and
    after it (within the task): the least distance from the outline to the segment, less the body's radius (the
    circumscribed disc for a rectangle). Written out here from the placements and the robot spec."""
    eid, body = eid or resident(tl), ROBOTS[tl.episode.robot].body
    radius = body.length_m / 2 if body.shape == "disc" else math.hypot(body.length_m, body.width_m) / 2
    out = []
    for task in tl.ego.tasks:
        bases = [tl.keyframe(k).base[:2] for k in range(task.k0, task.k1)]
        for j, base in enumerate(bases):
            p = tl.placement_of(eid, tl.keyframe(task.k0 + j).t)
            if p is None or math.dist(p.pos[:2], base) > 2.0:
                continue
            e = tl.entity(eid)
            outline = perimeter(Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw))
            segments = [(base, base)] + ([(bases[j - 1], base)] if j else []) + (
                [(base, bases[j + 1])] if j + 1 < len(bases) else [])
            out.append(min(segment_distance(outline, a, b) for a, b in segments) - radius)
    return out
