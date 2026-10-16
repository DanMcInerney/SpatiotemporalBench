"""What the AMR job's tests share: `warehouse_a` with the AMR planned on it (`PLANNED_ROBOTS["warehouse"]` lands with
M3-RW; until it merges every test and the gate patch it in), generated scripts cached per worker process, and the rules
a script is read by (the true body along every segment, the zone a bay is, the changes of a world).

`generated(seed, n)` is a script of `n` tasks of `amr/1` from `generate_script` with everything on (world processes, the
bay coin, the rear drops); `quiet(seed, n, ...)` is the same with the gaps' processes off and, when asked, the coin or
the drops, for the tests that compare two scripts that must differ in one thing only. Nothing here mutates a cached
script."""

from __future__ import annotations

import math
from functools import cache
from unittest import mock

import numpy as np
import pytest

from stbench import dressing_robots, job_amr, job_amr_drops, world_processes
from stbench.catalog import FLOOR_ITEM_Z
from stbench.generate import generate_script
from stbench.geometry import Rect, dist_to_rect, rect_distance, rects_overlap
from stbench.job_amr_path import dock_of, keep_outs
from stbench.job_amr_plan import BAY, LANE, SHELVING, TABLE, pieces
from stbench.job_registry import get
from stbench.mapspec import load_map
from stbench.robots import ROBOTS
from stbench.timeline import Timeline
from stbench.validate import collision_class, non_blocking

MAP, ROBOT, JOB = "warehouse_a", "amr", "amr/1"
SEEDS = (101, 102, 103)  # first 3 tasks: replenish x2, collect; pick x2, replenish; collect, pick, collect
M = load_map(MAP)
AMR = ROBOTS[ROBOT]


def planned():
    """`warehouse_a`'s planned robots as M3-RW will have them: the AMR (a no-op patch once that merges)."""
    return mock.patch.dict(dressing_robots.PLANNED_ROBOTS, {"warehouse": ("amr",)})


@pytest.fixture
def amr_checks_off(monkeypatch):
    """`amr/1`'s own checks off, for the tests that exercise `validate`'s generic rules on a hand-made AMR script: such
    a script is not a whole task of the job (it starts anywhere and has no plan of stations), which the job's checks
    rightly refuse."""
    from dataclasses import replace  # noqa: PLC0415

    monkeypatch.setattr(job_amr, "SPEC", replace(job_amr.SPEC, checks=lambda tl, m: []))


@cache
def generated(seed: int, n: int) -> Timeline:
    """The seed's `amr/1` script of `n` tasks with everything on, cached per worker process."""
    with planned():
        return generate_script(MAP, seed, ROBOT, JOB, n)


def quiet(seed: int, n: int, *, coin: bool | None = None, coin_of=None, rear: bool = True, processes: bool = False,
          build=None) -> Timeline:
    """The seed's script of `n` tasks with the gaps' processes off (unless `processes`), every bay's coin forced to
    `coin` (None: the real one) or decided by `coin_of(map, seed, i, j)`, the rear drops off unless `rear`, and
    `build(w)` run on the world right after it is built (to add or remove things). Not cached."""
    from stbench import generate  # noqa: PLC0415

    patches = [planned()]
    if not processes:
        patches.append(mock.patch.object(world_processes, "gaps", lambda w, m, seed: lambda lo, hi, keep_open: None))
    if coin is not None:
        patches.append(mock.patch.object(job_amr_drops, "bay_blocked", lambda m, seed, i, j: coin))
    if coin_of is not None:
        patches.append(mock.patch.object(job_amr_drops, "bay_blocked", coin_of))
    if not rear:
        patches.append(mock.patch.object(job_amr_drops, "rear_dropped", lambda m, seed, i, j: False))
    if build is not None:
        real = generate.build_world

        def built(m, seed):
            w = real(m, seed)
            build(w)
            return w

        patches.append(mock.patch.object(generate, "build_world", built))
    for p in patches:
        p.start()
    try:
        return generate_script(MAP, seed, ROBOT, JOB, n)
    finally:
        for p in reversed(patches):
            p.stop()


def plans_of(seed: int, n: int = 5):
    return get(JOB).plan(M, seed, n)


def stations(kind: str) -> dict[str, object]:
    return {f.id: f for f in pieces(M, kind)}


SHELVES, TABLES = stations(SHELVING), stations(TABLE)
ALL_STATIONS = {**SHELVES, **TABLES}


def docks() -> dict[str, object]:
    """Each station's dock, by the station's map id."""
    return {sid: dock_of(M, f, AMR) for sid, f in ALL_STATIONS.items()}


def visits(tl: Timeline, task) -> list[tuple[int, object, str, list]]:
    """(j, the dock, the response, the keyframes from the visit's `before_station` to the end of its undocking at T, or
    of its wait at T when the response is "skip"), for every station visit of the task in plan order."""
    kfs = tl.ego.keyframes[task.k0:task.k1]
    here = docks()
    out = []
    for j, d in enumerate(d for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_station"):
        dock = here[task.plan.params["stations"][j]]
        i = d.k - task.k0
        if d.response == "skip":
            end = next(x for x in range(i + 1, len(kfs)) if kfs[x].phase != "waiting") - 1
        else:
            end = next(x for x in range(i, len(kfs)) if kfs[x].phase == "undocking" and kfs[x].base[:2] == dock.t)
        out.append((j, dock, d.response, kfs[i:end + 1]))
    return out


def asked(tl: Timeline, i: int, j: int):
    """The `before_station` decision of visit j of task i."""
    task = tl.ego.tasks[i]
    return [d for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_station"][j]


def docked_visits(tl: Timeline, task) -> list[tuple[int, object, list]]:
    """(j, the dock, the keyframes from its `before_station` to the end of its undocking) of the visits that docked."""
    return [(j, dock, kfs) for j, dock, response, kfs in visits(tl, task) if response == "go"]


def skipped_visits(tl: Timeline, task) -> list[tuple[int, object, list]]:
    """(j, the dock, the keyframes from its `before_station` to the end of its wait at T) of the visits it skipped."""
    return [(j, dock, kfs) for j, dock, response, kfs in visits(tl, task) if response == "skip"]


@cache
def forced(seed: int, n: int, i: int, j: int, blocked: bool, **kw) -> Timeline:
    """`quiet(seed, n)` with the real coins except visit j of task i, whose coin says `blocked`, cached per worker."""
    real = job_amr_drops.bay_blocked
    return quiet(seed, n, **kw, coin_of=lambda m, s, a, b: blocked if (a, b) == (i, j) else real(m, s, a, b))


# -- the body along a path -----------------------------------------------------------------------------------------
def obstacles_at(tl: Timeline, t: float) -> list[Rect]:
    """What the AMR's body keeps its clearance from at time `t`, besides the walls: the fixed pieces and the light
    pieces (whole footprints), and the carried things and clutter lying on the floor taller than its climb. The charger
    (an anchor that may be touched), the staging lanes and the bays (markings, not obstacles) are not in it."""
    out = []
    for e in tl.entities:
        cls = collision_class(e)
        if cls not in ("fixed", "light", "carried") or non_blocking(e) or e.kind == "charger":
            continue
        p = tl.placement_of(e.id, t)
        if p is None or (cls == "carried" and (p.pos[2] >= FLOOR_ITEM_Z or e.size[2] <= AMR.body.climb_m)):
            continue
        out.append(tl.footprint(e.id, t))
    return out


def path_gap(tl: Timeline, task, rects_at) -> float:
    """The least distance from the AMR's true rectangle (0.8 x 0.58 m at its heading) to the rectangles `rects_at(t)`
    gives, over every point of every straight segment between consecutive keyframes of the task, 0.05 m apart (`t` the
    segment's first keyframe's time); a turn in place sweeps the circumscribed disc."""
    kfs = tl.ego.keyframes[task.k0:task.k1]
    radius = math.hypot(AMR.body.length_m, AMR.body.width_m) / 2
    best = math.inf
    for a, b in zip(kfs, kfs[1:], strict=False):
        rects = rects_at(a.t)
        run = math.dist(a.base[:2], b.base[:2])
        if run < 1e-9:
            if abs(a.yaw - b.yaw) > 1e-9:
                xs, ys = np.array([a.base[0]]), np.array([a.base[1]])
                best = min(best, min(float(dist_to_rect(xs, ys, r)[0]) for r in rects) - radius)
            continue
        steps = max(math.ceil(run / 0.05), 1)
        for s in range(steps + 1):
            f = s / steps
            x, y = a.base[0] + (b.base[0] - a.base[0]) * f, a.base[1] + (b.base[1] - a.base[1]) * f
            body = Rect(x, y, AMR.body.length_m, AMR.body.width_m, a.yaw)
            best = min([best, *(rect_distance(body, r) for r in rects if _near(body, r))])
    return best


def least_clearance(tl: Timeline, task) -> float:
    """`path_gap` to the walls and to what `obstacles_at` lists: the docking and undocking segments are included, the
    body's rectangle, not the planner's disc, being what stays 0.10 m off a station's face."""
    return path_gap(tl, task, lambda t: obstacles_at(tl, t) + M.wall_pieces())


def _near(body: Rect, r: Rect) -> bool:
    reach = math.hypot(body.sx, body.sy) / 2 + math.hypot(r.sx, r.sy) / 2 + 1.0
    return math.hypot(body.cx - r.cx, body.cy - r.cy) <= reach


# -- what the plan and the map say ---------------------------------------------------------------------------------
def bay_zone(dock) -> Rect:
    """A station's bay as the zone rule reads it: the bay piece's footprint inset by 0.05 m on every side."""
    bay = next(b for b in pieces(M, BAY) if b.id == dock.bay)
    return Rect(bay.center[0], bay.center[1], bay.size[0] - 0.1, bay.size[1] - 0.1, bay.yaw)


def lane_rects() -> list[Rect]:
    return [f.footprint() for f in pieces(M, LANE)]


def keep_out_rects() -> list[Rect]:
    return keep_outs(M, AMR)


def overlapping(rect: Rect, others: list[Rect], margin: float = 0.0) -> bool:
    return any(rects_overlap(rect, r, margin) for r in others)
