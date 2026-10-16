"""`vacuum/2`'s constants, its plan and its gaps: what a script of several tasks is before any world exists.

Split from `job_vacuum2.py` (which owns the job and its path) so the draws stay readable by themselves: `plan_tasks`
is world-independent (the map and the seed alone), and so is `gaps`, so both are testable without a world. A task is
`clean_rooms` (one room, or two with `TWO_ROOMS_P`, drawn without replacement) or `resume` (the room the task before
it was sent home in); a task that is not the last stops with `STOP_P` unless its last room is the dock's.

`eligible` says which rooms of a plan are entered through a **coin decision**, `before_room` (docs/jobs.md, "The
doorway coin"): the rooms of a `clean_rooms` task, but not the dock's room (no doorway is crossed to enter it) and not
the room a stopping task stops in (the resume sweeps what that task's brush left, so it must reach the room). A
`resume` has none. `ZONE` is the doorway question's place, the way through the opening ahead.
"""

from __future__ import annotations

from . import job_gaps
from .job_vacuum import _dock, _region_of_kind
from .mapspec import MapSpec
from .robots import ROBOTS
from .seeding import make_rng
from .timeline_ego import TaskPlan
from .zones import DOORWAY, ZoneRule

JOB_ID = "vacuum/2"
CLEAN, RESUME = "clean_rooms", "resume"
TWO_ROOMS_P = 0.25  # a clean_rooms task cleans two rooms, else one
STOP_P = 0.3  # a clean_rooms task that is not the script's last is sent home before its last room is done
# The share of the room's reachable fine cells the brush has swept before the stop (by area, not by path length: the
# edge pass is a third of a living room's path and most of its floor, so a stop by length left a hole too small to
# resume).
STOP_SHARE = (0.3, 0.7)
GAP_S = (600.0, 10800.0)  # each gap log-uniform from 10 minutes to 3 hours, before the span budget
SPAN_S = 7 * 3600.0  # a script spans at most this, first keyframe to last (every case clock then ends by 17:00)
TASK_MAX_S = 20 * 60.0  # the most a task may take (a check); the span budget counts it for every task
BEFORE_DOORWAY, ROOM_DONE, CLEANING_STOPPED, BEFORE_BACKOUT = ("before_doorway", "room_done", "cleaning_stopped",
                                                              "before_backout")
BEFORE_ROOM = "before_room"  # the doorway into a room to clean, where the coin falls: it asks `zone_clear`
GO, SKIP = "go", "skip"  # the responses of a `before_room` decision
DOORWAY_COIN_P = 0.5  # the share of eligible doorways the coin blocks
# "Is the doorway ahead of you clear of anything taller than ...": what the robot cannot drive over, its climb
ZONE = ZoneRule("the doorway ahead of you", DOORWAY, height_m=ROBOTS["vacuum"].body.climb_m)


def dock_room_id(m: MapSpec) -> str:
    """The region the dock stands in (the hall on `apartment_a`): the room a task starts and ends in."""
    room = m.region_of(_dock(m).center)
    if room is None:
        raise ValueError(f"the dock of {m.name} stands in no region")
    return room


def phrase(kinds: list[str]) -> str:
    """The room words of a brief: "kitchen" or "kitchen, then the hall" (the assignment says "clean the ...")."""
    return ", then the ".join(kinds)


def plan_tasks(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    """The script's assignments, from `make_rng(m.name, seed, JOB_ID, "job")` and the map alone:
    - task 0 is `clean_rooms`; after a task that stops, the next is `resume` over the room it stopped in (its last
      room); every other task is `clean_rooms`, with one room or, with `TWO_ROOMS_P`, two, drawn uniformly without
      replacement;
    - a task's first room is never the dock's (a one-room task of the hall alone crosses no doorway and asks nothing
      during the job); the hall comes back only as a second room;
    - a `clean_rooms` task that is not the last stops with `STOP_P` at a share uniform in `STOP_SHARE` of its last
      room's reachable floor swept, unless that room is the dock's (a resume of the hall alone would cross no doorway);
      a `resume` never stops.
    `params` hold `rooms` (the brief's words, the only key a brief reads), `kinds` (the room kinds in order) and, for a
    task that stops, `stop` (the share): the last two are the job's, never delivered. Every task draws the same
    numbers in the same order whatever they decide, so no draw depends on which tasks may stop."""
    if n_tasks < 1:
        raise ValueError(f"a script has at least one task, not {n_tasks}")
    dock = dock_room_id(m)
    rng = make_rng(m.name, seed, JOB_ID, "job")
    away = [r.kind for r in m.regions if r.id != dock]
    plans: list[TaskPlan] = []
    stopped: str | None = None  # the room kind the task before was sent home in
    for i in range(n_tasks):
        two, first, other = float(rng.random()), int(rng.integers(len(away))), int(rng.integers(len(m.regions) - 1))
        stop_u, share = float(rng.random()), float(rng.uniform(*STOP_SHARE))
        if stopped is not None:
            plans.append(TaskPlan(RESUME, {"rooms": phrase([stopped]), "kinds": [stopped]}))
            stopped = None
            continue
        rooms = [away[first]]
        if two < TWO_ROOMS_P:
            rooms.append([r.kind for r in m.regions if r.kind != rooms[0]][other])
        params: dict[str, object] = {"rooms": phrase(rooms), "kinds": rooms}
        last = next(r.id for r in m.regions if r.kind == rooms[-1])
        if i < n_tasks - 1 and last != dock and stop_u < STOP_P:
            params["stop"] = share
            stopped = rooms[-1]
        plans.append(TaskPlan(CLEAN, params))
    return plans


def gaps(map_name: str, seed: int, n_tasks: int) -> list[float]:
    """The n - 1 unobserved gaps between a script's tasks, in seconds: `job_gaps.gaps` (the law every job shares) with
    this job's id, longest task and constants, from `make_rng(map_name, seed, JOB_ID, "gaps")` alone (the seed's, never
    the world's): each log-uniform in `GAP_S`, shrunk to the span budget `SPAN_S - n * TASK_MAX_S` when their sum passes
    it, so the script spans at most `SPAN_S` with every task at its longest."""
    return job_gaps.gaps(map_name, seed, JOB_ID, n_tasks, TASK_MAX_S, GAP_S, SPAN_S)


def eligible(m: MapSpec, plan: TaskPlan, j: int) -> bool:
    """Whether room `j` of the plan is entered through a `before_room` decision (the module docstring says which rooms):
    a `clean_rooms` task's room that is not the dock's and not the room a stopping task stops in. From the map and the
    plan alone."""
    kinds = plan.params["kinds"]
    stops = plan.params.get("stop") is not None and j == len(kinds) - 1
    return plan.kind == CLEAN and not stops and _region_of_kind(m, str(kinds[j])).id != dock_room_id(m)
