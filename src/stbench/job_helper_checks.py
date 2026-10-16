"""`helper/1`'s own `validate` rules, run through `JobSpec.checks` (codes `job_*`), on top of `validate`'s generic ones.

Per task: the phases run waiting, navigating, grasping, carrying, placing, navigating, waiting (`job_phase_order`); the
first and last keyframe stand at the park pose (`job_park`); the decisions are exactly the sequence its plan gives
(`expected_decisions`), each in its phase, with the `before_grasp` on the last navigating keyframe of the first run and
the `before_place` on the last carrying keyframe, and at each the base within arm's reach of a counted piece of the room
the plan names, facing the nearest point of it (`job_decisions`, `job_reach`); the task takes at most `TASK_MAX_S`
(`job_task_time`). The carry (`job_helper_carry`): the targets are the plan's two rooms, a carried thing and the two
pieces (`job_targets`); the thing stood on the source piece until the last grasping keyframe and stands from the last
placing keyframe on a counted piece of the target room, on the planned piece unless the response sent it elsewhere, and
outside the window then, with no other placement of it in the task (`job_thing`); the camera is at -45 degrees through
the grasp and the place and at its default otherwise (`job_tilt`); the `before_place` response is the window's state at
that moment (`job_spot`). Per script: each gap lies in `job_gaps.GAP_S` (`job_gap`), the span is at most
`job_gaps.SPAN_S` (`job_span`), and the world holds at most one person (`job_one_person`): the home's one resident
(`JobSpec.one_person`; `person_mem`, which would name them as "the person", is asked by no job now).
"""

from __future__ import annotations

import math

from . import job_gaps
from .geometry import Rect, distance, nearest_on_rect, q, rects_overlap, wrap_angle
from .job_helper_plan import (
    BEFORE_DOORWAY,
    BEFORE_GRASP,
    BEFORE_PLACE,
    FETCH,
    PHASES,
    TASK_MAX_S,
    TIDY,
    TILT_DEG,
    park_pose,
    park_room,
)
from .job_helper_reach import ROBOT_ID, qualified, reach_band
from .job_helper_spot import BAND, ELSEWHERE, GO, SLIVER_M, grow
from .mapspec import MapSpec
from .person_seen import PERSON_KIND
from .robots import ROBOTS
from .timeline import Timeline
from .timeline_ego import Keyframe, Task, TaskPlan
from .validate import Problem
from .zone_shapes import surface_window

TOL_S = 0.01  # the times a file keeps are 6 decimals: a gap or a span may miss its bound by this much
PARK_TOL_M, PARK_TOL_RAD = 0.01, 1e-3  # how near a task's first and last pose must be the park pose
FACE_TOL_RAD = 1e-3  # how exactly the base faces the piece at a grasp or a place
BAND_TOL_M = 1e-3  # how far outside the reach band a pose may lie (the file keeps 6 decimals)
OVERLAP_TOL_M = 1e-3  # a thing put beside the window may reach this far into it (the file keeps 6 decimals)
PITCH_TOL_RAD = 1e-5  # the file keeps the pitch to 6 decimals
TIME_TOL_S = 1e-3  # and the times too


def expected_decisions(m: MapSpec, plan: TaskPlan) -> list[tuple[str, str]]:
    """The plan's decision points in order, as (kind, phase): a before_doorway (navigating) when the source is in
    another room than the park room, the before_grasp (navigating), a before_doorway (carrying) when the target is in
    another room than the source (a fetch's target is the park room), the before_place (carrying), and for a tidy a
    before_doorway (navigating) when the target is not in the park room."""
    home, (src, dst) = park_room(m), plan.params["regions"]
    out = [(BEFORE_DOORWAY, "navigating")] if src != home else []
    out.append((BEFORE_GRASP, "navigating"))
    out += [(BEFORE_DOORWAY, "carrying")] if dst != src else []
    out.append((BEFORE_PLACE, "carrying"))
    return out + ([(BEFORE_DOORWAY, "navigating")] if plan.kind == TIDY and dst != home else [])


def checks(tl: Timeline, m: MapSpec) -> list[Problem]:
    """Every problem of the job's own rules over the script (none when the script has no task). A map the job cannot
    run on (no dock, no outside door) is one `job_park` problem."""
    if not tl.ego.tasks:
        return []
    try:
        park = park_pose(m)
        counted = qualified(m, park[0])
    except ValueError as e:
        return [Problem("job_park", str(e))]
    out = _script_problems(tl)
    for task in tl.ego.tasks:
        out += _task_problems(tl, m, task, park, counted)
    return out


def _script_problems(tl: Timeline) -> list[Problem]:
    out: list[Problem] = []
    if tl.n_keyframes and (span := tl.keyframe(tl.n_keyframes - 1).t - tl.keyframe(0).t) > job_gaps.SPAN_S + TOL_S:
        out.append(Problem("job_span", f"the script spans {span:.0f} s; helper/1 spans at most "
                                       f"{job_gaps.SPAN_S:.0f} s"))
    if len(people := [e.id for e in tl.entities if e.kind == PERSON_KIND]) > 1:
        info = tl.episode
        out.append(Problem("job_one_person", f"the script {info.map} seed {info.seed} holds {len(people)} people "
                                             f"({', '.join(people)}); helper/1's home holds at most one resident"))
    lo, hi = job_gaps.GAP_S
    for task in tl.ego.tasks[1:]:
        if 0 < task.k0 < tl.n_keyframes:
            gap = tl.keyframe(task.k0).t - tl.keyframe(task.k0 - 1).t
            if not lo - TOL_S <= gap <= hi + TOL_S:
                out.append(Problem("job_gap", f"the gap before task {task.index} is {gap:.1f} s; helper/1's are "
                                              f"{lo:.0f} to {hi:.0f} s"))
    return out


def _task_problems(tl: Timeline, m: MapSpec, task: Task, park, counted) -> list[Problem]:
    kfs = [tl.keyframe(k) for k in range(task.k0, min(task.k1, tl.n_keyframes))]
    plan = task.plan
    if not kfs or plan.kind not in (TIDY, FETCH):
        return []  # `validate` reports the task's layout and its kind
    out: list[Problem] = []
    regions = tuple(plan.params["regions"])
    if len(task.targets) != 5 or task.targets[:2] != regions:
        out.append(Problem("job_targets", f"task {task.index} targets {list(task.targets)}; helper/1's are its plan's "
                                          f"rooms {list(regions)}, the thing and the source and target pieces"))
    phases = [kf.phase for i, kf in enumerate(kfs) if i == 0 or kf.phase != kfs[i - 1].phase]
    if tuple(phases) != PHASES:
        out.append(Problem("job_phase_order", f"task {task.index} ({plan.kind}) runs {phases}; helper/1 runs "
                                              f"{list(PHASES)}"))
    for kf, which in ((kfs[0], "first"), (kfs[-1], "last")):
        off, turn = distance(kf.base[:2], park[0]), abs(wrap_angle(kf.yaw - park[1]))
        if off > PARK_TOL_M or turn > PARK_TOL_RAD:
            out.append(Problem("job_park", f"task {task.index}'s {which} keyframe {kf.k} is {off:.3f} m and {turn:.4f} "
                                           f"rad from the park pose {park}"))
    if kfs[-1].t - kfs[0].t > TASK_MAX_S + TOL_S:
        out.append(Problem("job_task_time", f"task {task.index} takes {kfs[-1].t - kfs[0].t:.0f} s; helper/1's tasks "
                                            f"take at most {TASK_MAX_S:.0f} s"))
    decisions = [d for d in tl.decisions_in(task.k0, task.k1) if d.k < tl.n_keyframes]
    found = [(d.kind, tl.keyframe(d.k).phase) for d in decisions]
    want = expected_decisions(m, plan)
    if found != want:
        out.append(Problem("job_decisions", f"task {task.index} holds decisions {found}; its plan gives {want}"))
        return out
    at = {d.kind: d.k for d in decisions if d.kind != BEFORE_DOORWAY}
    out += _reach_problems(m, task, kfs, at, counted)
    if len(task.targets) == 5 and not out:
        answer = next(d.response for d in decisions if d.kind == BEFORE_PLACE)
        out += _tilt_problems(tl, task, kfs, at) + _spot_problems(tl, m, task, kfs[at[BEFORE_PLACE] - task.k0], answer)
        out += _thing_problems(tl, m, task, kfs, kfs[at[BEFORE_PLACE] - task.k0], answer, counted)
    return out


def _reach_problems(m: MapSpec, task: Task, kfs: list[Keyframe], at: dict[str, int], counted) -> list[Problem]:
    """`before_grasp` on the last navigating keyframe of the first run, `before_place` on the last carrying keyframe,
    the base at each within reach of a counted piece of the plan's room and facing it."""
    src, dst = task.plan.params["regions"]
    first_run = next(i for i, kf in enumerate(kfs) if kf.phase == "navigating")
    last_nav = next(kf.k for i, kf in enumerate(kfs)
                    if i >= first_run and (i + 1 == len(kfs) or kfs[i + 1].phase != "navigating"))
    last_carry = max(kf.k for kf in kfs if kf.phase == "carrying")
    out = []
    if at[BEFORE_GRASP] != last_nav:
        out.append(Problem("job_decisions", f"task {task.index}'s {BEFORE_GRASP} is not on the last navigating "
                                            "keyframe of its first run"))
    if at[BEFORE_PLACE] != last_carry:
        out.append(Problem("job_decisions", f"task {task.index}'s {BEFORE_PLACE} is not on the last carrying "
                                            "keyframe"))
    for decision, room in ((BEFORE_GRASP, src), (BEFORE_PLACE, dst)):
        kf = kfs[at[decision] - task.k0]
        if not any(_faces(m, kf, pid) for pid in counted[room]):
            out.append(Problem("job_reach", f"task {task.index}'s {decision} keyframe {kf.k} is not within reach of, "
                                            f"and facing, a counted piece of the {m.region(room).name}"))
    return out


def _tilt_problems(tl: Timeline, task: Task, kfs: list[Keyframe], at: dict[str, int]) -> list[Problem]:
    """Every keyframe's camera is at the robot's default pitch or the grasp's and place's; it is at the second from the
    `before_grasp` keyframe through the last grasping one, on the `before_place` keyframe and on the last placing one,
    and at the first while the robot waits."""
    robot = ROBOTS[ROBOT_ID]
    default, low = q(math.radians(robot.camera.pitch_deg)), q(math.radians(TILT_DEG))
    down = {*range(at[BEFORE_GRASP], max(kf.k for kf in kfs if kf.phase == "grasping") + 1), at[BEFORE_PLACE],
            max(kf.k for kf in kfs if kf.phase == "placing")}
    out = []
    for kf in kfs:
        if kf.k in down:
            want = {low}
        elif kf.phase == "waiting":
            want = {default}
        else:
            want = {default, low}
        if not any(abs(kf.cam_pitch - v) <= PITCH_TOL_RAD for v in want):
            out.append(Problem("job_tilt", f"task {task.index}'s keyframe {kf.k} ({kf.phase}) has the camera at "
                                           f"{math.degrees(kf.cam_pitch):.2f} degrees; helper/1 holds "
                                           f"{sorted(round(math.degrees(v), 2) for v in want)} there"))
    return out


def _window_of(m: MapSpec, kf: Keyframe) -> tuple[Rect, float] | None:
    """The window the base faces at the keyframe and its surface's height."""
    found = surface_window(m, kf.base[:2], kf.yaw)
    return None if found is None else (found[2], found[1].z)


def _spot_problems(tl: Timeline, m: MapSpec, task: Task, kf: Keyframe, answer: str | None) -> list[Problem]:
    """The `before_place` response is the window's state at that moment: `elsewhere` when a carried item or a clutter
    piece reaches into the window by more than `SLIVER_M`, else `go`."""
    found = _window_of(m, kf)
    if found is None:
        return [Problem("job_spot", f"task {task.index}'s before_place keyframe {kf.k} faces no surface")]
    window, ground = found
    inner, low, high = grow(window, -SLIVER_M), ground + BAND[0], ground + BAND[1]
    blocked = []
    for e in tl.entities_at(kf.t):
        if e.mobility not in ("carried", "clutter"):
            continue
        p = tl.placement_of(e.id, kf.t)
        box = Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)
        if p.pos[2] + e.size[2] > low and p.pos[2] < high and rects_overlap(box, inner):
            blocked.append(e.id)
    want = ELSEWHERE if blocked else GO
    if answer != want:
        return [Problem("job_spot", f"task {task.index}'s before_place response is {answer!r}; the window holds "
                                    f"{blocked or 'nothing'}, which gives {want!r}")]
    return []


def _piece_at(tl: Timeline, m: MapSpec, eid: str) -> str | None:
    """The id on the map of the piece an entity stands for: the furniture of its kind whose centre it stands at."""
    p, e = tl.placement_of(eid, 0.0), tl.entity(eid)
    if p is None:
        return None
    return next((f.id for f in m.furniture if f.kind == e.kind and distance(f.center, p.pos[:2]) < 1e-3), None)


def _thing_problems(tl: Timeline, m: MapSpec, task: Task, kfs: list[Keyframe], place: Keyframe, answer: str | None,
                    counted) -> list[Problem]:
    """The thing's placements (the module docstring): on the source piece until the last grasping keyframe, on a counted
    piece of the target room from the last placing keyframe, none between, and no other change of it in the task."""
    eid, source, target = task.targets[2:5]
    entity, dst = tl.entity(eid), task.plan.params["regions"][1]
    t_grasp = max(kf.t for kf in kfs if kf.phase == "grasping")
    t_place = max(kf.t for kf in kfs if kf.phase == "placing")
    t0, t1 = kfs[0].t, kfs[-1].t
    out: list[Problem] = []
    if entity.mobility != "carried":
        return [Problem("job_thing", f"task {task.index}'s thing {eid} is a {entity.mobility} entity, not a "
                                     "carried one")]
    before = tl.placement_of(eid, t_grasp - 2 * TIME_TOL_S)
    after = tl.placement_of(eid, t_place)
    inside = [p for p in tl.placements_of(eid) if t0 < p.t0 < t1 or t0 < (p.t1 or math.inf) < t1]
    if before is None or before.on != source or before.t1 is None or abs(before.t1 - t_grasp) > TIME_TOL_S:
        out.append(Problem("job_thing", f"task {task.index}'s thing {eid} does not stand on its source {source} until "
                                        f"the last grasping keyframe at {t_grasp}"))
    earlier = tl.placement_of(eid, t_place - 2 * TIME_TOL_S)
    if after is None or abs(after.t0 - t_place) > TIME_TOL_S or earlier is not None:
        out.append(Problem("job_thing", f"task {task.index}'s thing {eid} does not appear at the last placing keyframe "
                                        f"at {t_place}, and not before"))
    elif len(inside) != 2:
        out.append(Problem("job_thing", f"task {task.index}'s thing {eid} has {len(inside)} placements changing in the "
                                        "task; helper/1's has two (it leaves, it appears)"))
    else:
        piece = _piece_at(tl, m, after.on) if after.on else None
        if piece not in counted[dst]:
            out.append(Problem("job_thing", f"task {task.index}'s thing {eid} is put on {after.on}, which is no "
                                            f"counted piece of the {m.region(dst).name}"))
        elif after.on != target and answer != ELSEWHERE:
            out.append(Problem("job_thing", f"task {task.index}'s thing {eid} is put on {after.on}, not its target "
                                            f"{target}, though the response is {answer!r}"))
        elif after.on == target and answer == ELSEWHERE:
            window = _window_of(m, place)
            box = Rect(after.pos[0], after.pos[1], entity.size[0], entity.size[1], after.yaw)
            if window is not None and rects_overlap(box, window[0], -OVERLAP_TOL_M):
                out.append(Problem("job_thing", f"task {task.index}'s thing {eid} is put in the occupied window"))
    return out


def _faces(m: MapSpec, kf: Keyframe, piece_id: str) -> bool:
    """Whether the base is within arm's reach of the piece's footprint (the front edge `REACH_M` to `REACH_MAX_M` out)
    and faces its nearest point."""
    lo, hi = reach_band(ROBOTS[ROBOT_ID])
    (nx, ny), d = nearest_on_rect(m.furniture_by_id()[piece_id].footprint(), kf.base[0], kf.base[1])
    toward = math.atan2(ny - kf.base[1], nx - kf.base[0])
    return lo - BAND_TOL_M <= d <= hi + BAND_TOL_M and abs(wrap_angle(kf.yaw - toward)) <= FACE_TOL_RAD
