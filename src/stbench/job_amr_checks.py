"""`amr/1`'s own `validate` rules, run through `JobSpec.checks` (codes `job_*`), on top of `validate`'s generic ones.

Per task, from the plan, the map and the decisions' responses:
- `job_response`: every `before_station` carries the response "go" or "skip" and no other decision carries one; the
  response is "skip" exactly when a carried or light thing taller than the zone's height stands on the floor over the
  bay's zone at the decision's time (the bay holds a drum), "go" otherwise: the robot answers the world;
- `job_phase_order`: the phases run `waiting`, `undocking`, then `transit`/`at_station` for each pick face of a pick and
  `transit`, `approach`, `at_station`, `undocking` for each docked station or `transit`, `waiting` for each skipped one,
  then `transit`, `approach`, `waiting`;
- `job_charger`: the first and last keyframes stand at the charger's docked pose, head-on (1 cm, 1e-3 rad);
- `job_loaded`: `loaded` is false at the start and changes only from the last keyframe of an `at_station` hold to the
  next: to true at the end of the first hold made, and to false at the end of the last planned stop's hold if that stop
  docked (a skipped last stop leaves it true to the task's end; a task that skips every stop never loads);
- `job_decisions`: exactly the plan's decisions, in order and phase (a `before_undock` only for a docked visit). Every
  `before_station` stands at its station's T, facing it, and the bay nearest its base (the zone rule's nearest piece)
  is that station's bay; every `before_undock` is the last `at_station` keyframe of a docked visit, at the docked pose;
  every `before_crossing` lies in transit at most `CROSSING_STANDOFF_M` before an aisle's boundary, the first outside
  every aisle and the second inside one;
- `job_targets`: the task targets its stations (and a pick's aisle); `job_task_time`: at most `TASK_MAX_S`.
Per script: `job_gap` (each gap in `job_gaps.GAP_S`) and `job_span` (at most `job_gaps.SPAN_S`).
"""

from __future__ import annotations

from . import job_gaps
from .geometry import distance, rects_overlap, wrap_angle
from .job_amr_drops import ZONE_HEIGHT_M, zone_rect
from .job_amr_path import ahead_of_boundary, charger_pose, dock_of
from .job_amr_plan import (
    AISLE,
    BAY,
    BEFORE_CROSSING,
    BEFORE_STATION,
    BEFORE_UNDOCK,
    CROSSING_STANDOFF_M,
    PICK,
    TASK_MAX_S,
    pieces,
    targets_of,
)
from .mapspec import MapSpec
from .robots import ROBOTS
from .timeline import Timeline
from .timeline_ego import Task
from .validate import Problem

TOL_S = 0.01  # the times a file keeps are 6 decimals: a gap or a span may miss its bound by this much
DOCKED_TOL_M, DOCKED_TOL_RAD = 0.01, 1e-3  # how near a task's first and last pose must be the charger's docked pose
AT_T_TOL_M, AT_T_TOL_RAD = 1e-4, 1e-4  # a before_station stands at T, facing the station, to this much
GO, SKIP = "go", "skip"  # a `before_station`'s responses
BLOCKER_MOBILITIES = ("carried", "light", "clutter")  # what can stand in a bay and be taller than its zone allows


def phase_order(task: Task, skipped: frozenset[int] | set[int] = frozenset()) -> list[str]:
    """The runs of phases a task of this plan makes, in order, when the visits of `skipped` (indices into the plan's
    stations) wait at T instead of docking."""
    plan = task.plan
    order = ["waiting", "undocking"]
    if plan.kind == PICK:
        order += ["transit", "at_station"] * len(plan.params["faces"])
    for j in range(len(plan.params["stations"])):
        order += ["transit", "waiting"] if j in skipped else ["transit", "approach", "at_station", "undocking"]
    return order + ["transit", "approach", "waiting"]


def expected_decisions(task: Task, skipped: frozenset[int] | set[int] = frozenset()) -> list[tuple[str, str]]:
    """The plan's decision points in order, as (kind, phase): a pick's two crossings first, then for each station a
    before_station (transit, at T) and, for a docked one, a before_undock (at_station, the last frame of the hold)."""
    out = [(BEFORE_CROSSING, "transit")] * 2 if task.plan.kind == PICK else []
    for j in range(len(task.plan.params["stations"])):
        out += [(BEFORE_STATION, "transit")] + ([] if j in skipped else [(BEFORE_UNDOCK, "at_station")])
    return out


def expected_loaded(task: Task, skipped: frozenset[int] | set[int] = frozenset()) -> list[bool]:
    """`loaded` after each `at_station` hold the task makes, in order (a pick's faces, then its docked stations): true
    from the first, false after the last planned stop's hold when that stop docked."""
    n = len(task.plan.params["stations"])
    docked = [j for j in range(n) if j not in skipped]
    after = [True] * ((len(task.plan.params["faces"]) if task.plan.kind == PICK else 0) + len(docked))
    if docked and docked[-1] == n - 1:
        after[-1] = False
    return after


def checks(tl: Timeline, m: MapSpec) -> list[Problem]:
    """Every problem of the job's own rules over the script (none when the script has no task)."""
    if not tl.ego.tasks:
        return []
    try:
        home, heading, _ = charger_pose(m, ROBOTS["amr"])
    except ValueError as e:
        return [Problem("job_charger", str(e))]
    out = _script_problems(tl)
    for task in tl.ego.tasks:
        out += _task_problems(tl, m, task, home, heading)
    return out


def _script_problems(tl: Timeline) -> list[Problem]:
    out = []
    if tl.n_keyframes and (span := tl.keyframe(tl.n_keyframes - 1).t - tl.keyframe(0).t) > job_gaps.SPAN_S + TOL_S:
        out.append(Problem("job_span", f"the script spans {span:.0f} s; amr/1 spans at most {job_gaps.SPAN_S:.0f} s"))
    for task in tl.ego.tasks[1:]:
        if 0 < task.k0 < tl.n_keyframes:
            gap = tl.keyframe(task.k0).t - tl.keyframe(task.k0 - 1).t
            lo, hi = job_gaps.GAP_S
            if not lo - TOL_S <= gap <= hi + TOL_S:
                out.append(Problem("job_gap", f"the gap before task {task.index} is {gap:.1f} s; amr/1's are {lo:.0f} "
                                              f"to {hi:.0f} s"))
    return out


def _runs(kfs, phase: str) -> list[list]:
    """The runs of consecutive keyframes in `phase`."""
    runs: list[list] = []
    for i, kf in enumerate(kfs):
        if kf.phase == phase:
            runs.append([]) if i == 0 or kfs[i - 1].phase != phase else None
            runs[-1].append(kf)
    return runs


def _holds_blocker(tl: Timeline, zone, t: float) -> bool:
    """Whether a thing taller than the zone rule's height stands on the floor over the zone at time `t`."""
    for e in tl.entities:
        if e.mobility not in BLOCKER_MOBILITIES or e.size[2] <= ZONE_HEIGHT_M:
            continue
        p = tl.placement_of(e.id, t)
        if p is not None and p.on is None and rects_overlap(tl.footprint(e.id, t), zone):
            return True
    return False


def _response_problems(tl: Timeline, m: MapSpec, task: Task, name: str, decisions) -> list[Problem]:
    """`job_response`: the responses are the job's alone and follow the bay's state at each `before_station`."""
    out, asked = [], 0
    stations = task.plan.params["stations"]
    for d in decisions:
        if d.kind != BEFORE_STATION:
            if d.response is not None:
                out.append(Problem("job_response", f"{name}'s {d.kind} at keyframe {d.k} carries the response "
                                                   f"{d.response!r}; only a before_station answers one"))
            continue
        asked += 1
        if d.response not in (GO, SKIP):
            out.append(Problem("job_response", f"{name}'s before_station at keyframe {d.k} carries the response "
                                               f"{d.response!r}, not {GO!r} or {SKIP!r}"))
        elif asked <= len(stations):
            bay = next(b for b in pieces(m, BAY) if b.id == dock_of(m, m.furniture_by_id()[stations[asked - 1]],
                                                                    ROBOTS["amr"]).bay)
            held = _holds_blocker(tl, zone_rect(bay), tl.keyframe(d.k).t)
            if held != (d.response == SKIP):
                out.append(Problem("job_response", f"{name}'s before_station {asked - 1} at keyframe {d.k} answers "
                                                   f"{d.response!r} with the bay {'blocked' if held else 'clear'}"))
    return out


def _task_problems(tl: Timeline, m: MapSpec, task: Task, home, heading: float) -> list[Problem]:
    kfs = [tl.keyframe(k) for k in range(task.k0, min(task.k1, tl.n_keyframes))]
    if not kfs:
        return []  # `validate` reports the task's layout
    name, out = f"task {task.index}", []
    stations = list(task.plan.params["stations"])
    if task.targets != targets_of(task.plan):
        want = list(targets_of(task.plan))
        out.append(Problem("job_targets", f"{name} targets {list(task.targets)}, its plan's are {want}"))
    decisions = [d for d in tl.decisions_in(task.k0, task.k1) if d.k < tl.n_keyframes]
    out += _response_problems(tl, m, task, name, decisions)
    skipped = {n for n, d in enumerate(d for d in decisions if d.kind == BEFORE_STATION) if d.response == SKIP}
    phases = [kf.phase for i, kf in enumerate(kfs) if i == 0 or kf.phase != kfs[i - 1].phase]
    if phases != phase_order(task, skipped):
        out.append(Problem("job_phase_order", f"{name} runs {phases}; its plan gives {phase_order(task, skipped)}"))
        return out
    for kf, which in ((kfs[0], "first"), (kfs[-1], "last")):
        off, turn = distance(kf.base[:2], home), abs(wrap_angle(kf.yaw - heading))
        if off > DOCKED_TOL_M or turn > DOCKED_TOL_RAD:
            out.append(Problem("job_charger", f"{name}'s {which} keyframe {kf.k} is {off:.3f} m and {turn:.4f} rad "
                                              f"from the charger's docked pose {home}, {heading:.4f}"))
    if kfs[-1].t - kfs[0].t > TASK_MAX_S + TOL_S:
        out.append(Problem("job_task_time", f"{name} takes {kfs[-1].t - kfs[0].t:.0f} s; amr/1's tasks take at most "
                                            f"{TASK_MAX_S:.0f} s"))
    out += _loaded_problems(kfs, name, task, skipped)
    found = [(d.kind, tl.keyframe(d.k).phase) for d in decisions]
    if found != expected_decisions(task, skipped):
        want = expected_decisions(task, skipped)
        out.append(Problem("job_decisions", f"{name} holds decisions {found}; its plan gives {want}"))
        return out
    holds, bays = _runs(kfs, "at_station"), pieces(m, BAY)
    ours = [j for j in range(len(stations)) if j not in skipped]  # the plan's docked visits, in order
    docked = holds[len(holds) - len(ours):]
    visit = {BEFORE_STATION: 0, BEFORE_UNDOCK: 0, BEFORE_CROSSING: 0}
    for d in decisions:
        kf, n = tl.keyframe(d.k), visit[d.kind]
        visit[d.kind] += 1
        if d.kind == BEFORE_CROSSING:
            outside = (m.region_of(kf.base[:2]) is None) or m.region(m.region_of(kf.base[:2])).kind != AISLE
            ahead = ahead_of_boundary(m, kf.base[:2], kf.yaw)
            if ahead is None or not 0.0 < ahead <= CROSSING_STANDOFF_M + 1e-6 or outside != (n == 0):
                out.append(Problem("job_decisions", f"{name}'s crossing {n} at keyframe {kf.k} is not within "
                                                    f"{CROSSING_STANDOFF_M} m of an aisle's boundary, "
                                                    f"{'entering' if n == 0 else 'leaving'} it"))
            continue
        j = n if d.kind == BEFORE_STATION else ours[n]  # the plan's station this decision is about
        dock = dock_of(m, m.furniture_by_id()[stations[j]], ROBOTS["amr"])
        if d.kind == BEFORE_STATION:
            near = min(bays, key=lambda b: (distance(kf.base[:2], b.center), b.id))
            if (distance(kf.base[:2], dock.t) > AT_T_TOL_M or abs(wrap_angle(kf.yaw - dock.heading)) > AT_T_TOL_RAD
                    or near.id != dock.bay):
                out.append(Problem("job_decisions", f"{name}'s before_station {n} at keyframe {kf.k} is not at T "
                                                    f"{dock.t} facing station {stations[j]}, whose bay is {dock.bay}; "
                                                    f"the nearest bay is {near.id}"))
        elif kf.k != docked[n][-1].k or distance(kf.base[:2], dock.docked) > AT_T_TOL_M:
            out.append(Problem("job_decisions", f"{name}'s before_undock {n} at keyframe {kf.k} is not the last "
                                                f"at_station keyframe of the docked visit at {dock.docked}"))
    return out


def _loaded_problems(kfs, name: str, task: Task, skipped: set[int]) -> list[Problem]:
    """`loaded` is false at the start and flips only from the last keyframe of an `at_station` hold to the next, as
    `expected_loaded` says: true after the first hold made, false after the last planned stop's if it docked."""
    holds = _runs(kfs, "at_station")
    flips = [(a.k, b.status["loaded"]) for a, b in zip(kfs, kfs[1:], strict=False)
             if a.status["loaded"] != b.status["loaded"]]
    want, held = [], False
    for hold, after in zip(holds, expected_loaded(task, skipped), strict=False):
        if after != held:
            want.append((hold[-1].k, after))
        held = after
    if flips != want or kfs[0].status["loaded"] or kfs[-1].status["loaded"] != held:
        return [Problem("job_loaded", f"{name}'s loaded changes at {flips}, ends {kfs[-1].status['loaded']}; it "
                                      f"changes at {want} and ends {held}, after the holds it makes")]
    return []
