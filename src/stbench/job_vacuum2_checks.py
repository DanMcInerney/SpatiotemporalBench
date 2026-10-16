"""`vacuum/2`'s own `validate` rules, run through `JobSpec.checks` (codes `job_*`), on top of `validate`'s generic ones.

Per task: the phases run docked, undocking, transit, then cleaning and transit alternating once for each room the robot
cleans, docking, docked; the brush is on exactly in `cleaning`; the first and last keyframe stand at the docked pose
(`job_vacuum`'s tolerance); the decisions are exactly the sequence its plan and its responses give
(`expected_decisions`), each in its phase, with a `room_done` or `cleaning_stopped` on the last keyframe of its room's
cleaning; the task takes at most `TASK_MAX_S`.

The doorway coin (`job_vacuum2_doorway`), over the world the script holds: a `before_room` is answered "go" or "skip"
and no other decision is answered; it is answered "skip" exactly when something stands in the way of the doorway ahead
at its time; the things that close it leave no free width across the opening as wide as the body plus twice its
clearance and keep the body's clearance and 0.05 m from the path up to the decision; and at a task's start nothing
stands in the way of any doorway but the doorways its own skips are answered at.

Per script: the gaps lie in `GAP_S`, the span is at most `SPAN_S`, and a `resume` follows a task whose plan stops and
names exactly the room that task stopped in (and a task that stops is followed by one).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .geometry import dist_to_rect, distance, wrap_angle
from .job_vacuum import DOCKED_TOL_M, DOCKED_TOL_RAD, _region_of_kind, docked_pose
from .job_vacuum2_backout import site_of
from .job_vacuum2_doorway import across_gaps, gap_limit, keep_from_body, opening_ahead, sampled, standing_in
from .job_vacuum2_plan import (
    BEFORE_BACKOUT,
    BEFORE_DOORWAY,
    BEFORE_ROOM,
    CLEAN,
    CLEANING_STOPPED,
    GAP_S,
    GO,
    RESUME,
    ROOM_DONE,
    SKIP,
    SPAN_S,
    TASK_MAX_S,
    dock_room_id,
    eligible,
)
from .mapspec import MapSpec
from .robots import ROBOTS
from .timeline import Timeline
from .timeline_ego import DecisionPoint, Task, TaskPlan
from .validate import Problem

TOL_S = 0.01  # the times a file keeps are 6 decimals: a gap or a span may miss its bound by this much
TOL_M = 0.001  # the metres a blocker may fall short of its clearance by (the file keeps 6 decimals)


def phase_order(n_rooms: int) -> list[str]:
    """The phases a task that cleans `n_rooms` rooms runs, each run once: docked, undocking, transit, cleaning and
    transit once a room, docking, docked."""
    return ["docked", "undocking", "transit", *["cleaning", "transit"] * n_rooms, "docking", "docked"]


def expected_decisions(tl: Timeline, m: MapSpec, plan: TaskPlan, found: Sequence[DecisionPoint]
                       ) -> list[tuple[str, str]]:
    """The plan's decision points in order, as (kind, phase), given the task's decisions `found` (their responses say
    which rooms the robot skipped): for each room a `before_room` (transit) where `eligible`, else a `before_doorway`
    (transit), unless it is the dock's room or the robot stood in it already, having crossed it to the room it skipped
    before (its place at that decision); then, for a room not skipped, a before_backout (transit) in a site room of a
    `clean_rooms` task and a room_done (cleaning), or cleaning_stopped for the last room of a task that stops; after
    the last room, when it was cleaned, a before_doorway (transit) unless it is the dock's room. ValueError when a room
    is not a unique kind."""
    dock, kinds = dock_room_id(m), [str(k) for k in plan.params["kinds"]]
    asked = iter([d for d in found if d.kind == BEFORE_ROOM])
    out: list[tuple[str, str]] = []
    cleaned, left_at = False, None  # whether the last room was cleaned; the keyframe a room was skipped at
    for j, kind in enumerate(kinds):
        room = _region_of_kind(m, kind)
        inside = left_at is not None and tl.region_of(tl.keyframe(left_at).base[:2]) == room.id
        left_at = None
        if room.id != dock and not inside:
            if eligible(m, plan, j):
                out.append((BEFORE_ROOM, "transit"))
                d = next(asked, None)
                if d is not None and d.response == SKIP:
                    cleaned, left_at = False, d.k
                    continue
            else:
                out.append((BEFORE_DOORWAY, "transit"))
        if plan.kind == CLEAN and site_of(m, room, ROBOTS["vacuum"]) is not None:
            out.append((BEFORE_BACKOUT, "transit"))
        stops = j == len(kinds) - 1 and plan.params.get("stop") is not None
        out.append((CLEANING_STOPPED if stops else ROOM_DONE, "cleaning"))
        cleaned = True
    if cleaned and _region_of_kind(m, kinds[-1]).id != dock:
        out.append((BEFORE_DOORWAY, "transit"))
    return out


def checks(tl: Timeline, m: MapSpec) -> list[Problem]:
    """Every problem of the job's own rules over the script (none when the script has no task). A map without a dock
    cannot hold the job (`job_dock`)."""
    if not tl.ego.tasks:
        return []
    try:
        dock = docked_pose(m)
    except ValueError as e:
        return [Problem("job_dock", str(e))]
    out = _script_problems(tl)
    for task in tl.ego.tasks:
        out += _task_problems(tl, m, task, dock)
    return out


def _script_problems(tl: Timeline) -> list[Problem]:
    tasks, out = tl.ego.tasks, []
    if tl.n_keyframes and (span := tl.keyframe(tl.n_keyframes - 1).t - tl.keyframe(0).t) > SPAN_S + TOL_S:
        out.append(Problem("job_span", f"the script spans {span:.0f} s; vacuum/2 spans at most {SPAN_S:.0f} s"))
    for i, task in enumerate(tasks):
        if i and 0 < task.k0 < tl.n_keyframes:
            gap = tl.keyframe(task.k0).t - tl.keyframe(task.k0 - 1).t
            if not GAP_S[0] - TOL_S <= gap <= GAP_S[1] + TOL_S:
                out.append(Problem("job_gap", f"the gap before task {i} is {gap:.1f} s; vacuum/2's are {GAP_S[0]:.0f} "
                                              f"to {GAP_S[1]:.0f} s"))
        stops = task.plan.params.get("stop") is not None
        before = tasks[i - 1].plan if i else None
        if task.plan.kind == RESUME:
            ok = before is not None and before.params.get("stop") is not None and (
                task.plan.params.get("kinds") == [before.params["kinds"][-1]])
            if not ok:
                out.append(Problem("job_resume", f"task {i} resumes, but the task before it does not stop in its room"))
        if stops and (i + 1 == len(tasks) or tasks[i + 1].plan.kind != RESUME):
            out.append(Problem("job_resume", f"task {i} is sent home, and no resume follows it"))
    return out


def _task_problems(tl: Timeline, m: MapSpec, task: Task, dock: tuple[float, float, float]) -> list[Problem]:
    kfs = [tl.keyframe(k) for k in range(task.k0, min(task.k1, tl.n_keyframes))]
    if not kfs:
        return []  # `validate` reports the task's layout
    out: list[Problem] = []
    decisions = [d for d in tl.decisions_in(task.k0, task.k1) if d.k < tl.n_keyframes]
    try:
        rooms = [_region_of_kind(m, str(k)) for k in task.plan.params["kinds"]]
        want = expected_decisions(tl, m, task.plan, decisions)
    except (KeyError, ValueError) as e:
        return [Problem("job_targets", f"task {task.index}'s plan names no rooms of the map: {e}")]
    cleaned = sum(kind in (ROOM_DONE, CLEANING_STOPPED) for kind, _ in want)
    if task.targets != tuple(r.id for r in rooms):
        out.append(Problem("job_targets", f"task {task.index} targets {list(task.targets)}, its plan's rooms are "
                                          f"{[r.id for r in rooms]}"))
    phases = [kf.phase for i, kf in enumerate(kfs) if i == 0 or kf.phase != kfs[i - 1].phase]
    if phases != phase_order(cleaned):
        out.append(Problem("job_phase_order", f"task {task.index} runs {phases}; vacuum/2 runs {phase_order(cleaned)}"))
    out += [Problem("job_brush", f"keyframe {kf.k}: the brush is {kf.status.get('brush')} in {kf.phase!r}")
            for kf in kfs if kf.status.get("brush") != (kf.phase == "cleaning")]
    for kf, which in ((kfs[0], "first"), (kfs[-1], "last")):
        off, turn = distance(kf.base[:2], dock[:2]), abs(wrap_angle(kf.yaw - dock[2]))
        if off > DOCKED_TOL_M or turn > DOCKED_TOL_RAD:
            out.append(Problem("job_dock", f"task {task.index}'s {which} keyframe {kf.k} is {off:.3f} m and {turn:.4f} "
                                           f"rad from the docked pose {dock}"))
    if kfs[-1].t - kfs[0].t > TASK_MAX_S + TOL_S:
        out.append(Problem("job_task_time", f"task {task.index} takes {kfs[-1].t - kfs[0].t:.0f} s; vacuum/2's tasks "
                                            f"take at most {TASK_MAX_S:.0f} s"))
    found = [(d.kind, tl.keyframe(d.k).phase) for d in decisions]
    if found != want:
        out.append(Problem("job_decisions", f"task {task.index} holds decisions {found}; its plan gives {want}"))
        return out + _doorway_problems(tl, m, task, decisions)
    ends = [kf.k for i, kf in enumerate(kfs) if kf.phase == "cleaning" and (i + 1 == len(kfs)
                                                                          or kfs[i + 1].phase != "cleaning")]
    done = [d.k for d in decisions if d.kind in (ROOM_DONE, CLEANING_STOPPED)]
    if done != ends:
        out.append(Problem("job_decisions", f"task {task.index}'s {ROOM_DONE} and {CLEANING_STOPPED} decisions at "
                                            f"{done} are not on the last cleaning keyframe of each room: {ends}"))
    return out + _doorway_problems(tl, m, task, decisions)


def _doorway_problems(tl: Timeline, m: MapSpec, task: Task, decisions: list[DecisionPoint]) -> list[Problem]:
    """The coin's rules (the module docstring), over the world the script holds."""
    out: list[Problem] = []
    skipped: set[str] = set()  # the openings the task's skips are answered at
    robot = ROBOTS["vacuum"]
    for d in decisions:
        if d.response is not None and d.kind != BEFORE_ROOM or d.kind == BEFORE_ROOM and d.response not in (GO, SKIP):
            out.append(Problem("job_response", f"task {task.index}'s {d.kind} at keyframe {d.k} is answered "
                                               f"{d.response!r}; a {BEFORE_ROOM} is answered {GO!r} or {SKIP!r}, "
                                               f"no other decision is answered"))
            continue
        if d.kind != BEFORE_ROOM:
            continue
        kf = tl.keyframe(d.k)
        o = opening_ahead(m, kf.base[:2])
        things = standing_in(tl, m, o, kf.t)
        if (d.response == SKIP) != bool(things):
            out.append(Problem("job_doorway", f"task {task.index}'s {BEFORE_ROOM} at keyframe {d.k} is answered "
                                              f"{d.response!r}, but {things or 'nothing'} stands in the {o.id}'s way"))
        if d.response == SKIP and things:
            skipped.add(o.id)
            rects = [tl.footprint(eid, kf.t) for eid in things]
            if (gap := max(across_gaps(o, rects))) >= gap_limit(robot):
                out.append(Problem("job_doorway_gap", f"the things in the {o.id} at keyframe {d.k} leave {gap:.3f} m "
                                                      f"free across it; the body needs {gap_limit(robot):.3f} m"))
            near, keep = _nearest(tl, task, d.k, rects), keep_from_body(robot)
            if near < keep - TOL_M:
                out.append(Problem("job_doorway_near", f"the things in the {o.id} come within {near:.3f} m of the "
                                                       f"base's path up to keyframe {d.k}; they keep {keep:.3f} m"))
    start = tl.keyframe(task.k0).t
    for o in m.openings:
        if o.interior and o.id not in skipped and (things := standing_in(tl, m, o, start)):
            out.append(Problem("job_doorway", f"task {task.index} starts with {things} in the way of the {o.id}, which "
                                              f"none of its skips is answered at"))
    return out


def _nearest(tl: Timeline, task: Task, k: int, rects: list) -> float:
    """The least distance from the base's path from the task's start to keyframe `k` (the straight lines between
    consecutive keyframes) to the rectangles, 0.05 m apart."""
    kfs = [tl.keyframe(i).base[:2] for i in range(task.k0, k + 1)]
    px, py = sampled(list(zip(kfs, kfs[1:], strict=False)) or [(kfs[0], kfs[0])])
    return min(float(dist_to_rect(px, py, r).min()) for r in rects) if rects else math.inf
