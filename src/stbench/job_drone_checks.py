"""`drone/1`'s own `validate` rules, run through `JobSpec.checks` (codes `job_*`), on top of `validate`'s generic ones.

Per flight: the phases run in the flight's order (`phase_order`); the first and last keyframe stand on the pad at its
top plus `LEG_M` (`job_nest`); transit and inspection are at `ALT_M`, the lowering at `LOWER_M` above the zone's
ground, no keyframe above the transit height or below the pad (`job_heights`); the camera's pitch is the phase's, within
the gimbal's limits and turned no faster than `GIMBAL_DPS` (`job_gimbal`); the tether is paid out only while lowering,
held over the release and reeled in over the hover's last part, none in an inspection or in a delivery that was aborted
(`job_tether`); the decisions are the three the flight makes, each on the last keyframe of its phase (`job_decisions`);
the response at `before_lower` is `abort` exactly when the flight is a delivery and its zone holds a standing item at
the flight's first keyframe, else `go`, and the other decisions carry none (`job_response`); the battery never rises;
the flight takes at most `TASK_MAX_S`. Per script: the gaps lie in the shared law's range, the span is at most its
budget, and every flight's task names its zone's region and centre. The plan and the map are what a check reads, and
the placements of the carried items for the one thing the response follows: whether the zone holds an item at the
start. A delivery that holds one is aborted: it runs the inspection's phases (`job_phase_order`, `job_heights`) and
pays out no tether.
"""

from __future__ import annotations

import math

from .geometry import Rect, distance, rects_overlap
from .job_drone_path import pad_z
from .job_drone_plan import (
    ABORT,
    ALT_M,
    BEFORE_DESCENT,
    BEFORE_LOWER,
    BEFORE_RETURN,
    DELIVER,
    GIMBAL_DPS,
    GO,
    LOWER_M,
    PITCH_DEG,
    TASK_MAX_S,
    drop_zones,
    nest_of,
)
from .job_drone_zone import STAND_M, zone_rect
from .job_gaps import GAP_S, SPAN_S
from .mapspec import Furniture, MapSpec
from .robots import ROBOTS, Robot
from .timeline import Timeline
from .timeline_ego import Keyframe, Task
from .validate import Problem
from .zones import ground_z

TOL_M = 0.01  # a height or a place may miss its mark by this much
TOL_S = 0.01  # the times a file keeps are 6 decimals: a gap or a span may miss its bound by this much
TOL_DEG = 1e-3  # the file keeps the pitch to 6 decimals of a radian
TETHER_TOL_M = 1e-4
PHASE_PITCH = {"preflight": "preflight", "takeoff": "transit", "approach": "transit", "return": "transit",
               "inspect": "down", "descend": "down", "lower": "down", "release": "down", "climb_out": "down",
               "land": "down"}
DECISIONS = [(BEFORE_DESCENT, "inspect"), (BEFORE_LOWER, "descend"), (BEFORE_RETURN, "climb_out")]


def phase_order(lowers: bool) -> list[str]:
    """The phases a flight runs, each run once: a flight that lowers its package (a delivery that was not aborted)
    lowers and releases, any other hovers (`inspect`) through them (so `inspect` runs twice)."""
    hover = ["lower", "release"] if lowers else ["inspect"]
    return ["preflight", "takeoff", "approach", "inspect", "descend", *hover, "climb_out", "return", "land"]


def holds_item(tl: Timeline, m: MapSpec, zone: Furniture, t: float) -> bool:
    """Whether the zone holds a standing item at time `t`, from the timeline's placements alone: some carried thing
    taller than `STAND_M` above the zone's ground has a footprint that overlaps the zone (the piece's footprint less
    `EDGE_M`)."""
    rect, ground = zone_rect(zone), ground_z(m, zone)
    for e in tl.entities:
        p = tl.placement_of(e.id, t) if e.mobility == "carried" else None
        if p is not None and p.pos[2] + e.size[2] - ground > STAND_M and rects_overlap(
                Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw), rect):
            return True
    return False


def checks(tl: Timeline, m: MapSpec) -> list[Problem]:
    """Every problem of the job's own rules over the script (none when the script has no flight). A map without exactly
    one nest, or without a drop zone, cannot hold the job (`job_nest`)."""
    if not tl.ego.tasks:
        return []
    try:
        nest, zones = nest_of(m), {z.id: z for z in drop_zones(m)}
    except ValueError as e:
        return [Problem("job_nest", str(e))]
    out = _script_problems(tl)
    for task in tl.ego.tasks:
        zone = zones.get(str(task.plan.params.get("zone")))
        if zone is None:
            out.append(Problem("job_targets", f"flight {task.index}'s plan names the zone "
                                              f"{task.plan.params.get('zone')!r}, which the map lacks"))
        else:
            out += _flight_problems(tl, m, task, nest, zone)
    return out


def _script_problems(tl: Timeline) -> list[Problem]:
    out = []
    if tl.n_keyframes and (span := tl.keyframe(tl.n_keyframes - 1).t - tl.keyframe(0).t) > SPAN_S + TOL_S:
        out.append(Problem("job_span", f"the script spans {span:.0f} s; drone/1 spans at most {SPAN_S:.0f} s"))
    for i, task in enumerate(tl.ego.tasks):
        if i and 0 < task.k0 < tl.n_keyframes:
            gap = tl.keyframe(task.k0).t - tl.keyframe(task.k0 - 1).t
            if not GAP_S[0] - TOL_S <= gap <= GAP_S[1] + TOL_S:
                out.append(Problem("job_gap", f"the gap before flight {i} is {gap:.1f} s; the job's are {GAP_S[0]:.0f} "
                                              f"to {GAP_S[1]:.0f} s"))
    return out


def _flight_problems(tl: Timeline, m: MapSpec, task: Task, nest: Furniture, zone: Furniture) -> list[Problem]:
    kfs = [tl.keyframe(k) for k in range(task.k0, min(task.k1, tl.n_keyframes))]
    if not kfs:
        return []  # `validate` reports the task's layout
    i, robot = task.index, ROBOTS["drone"]
    z_pad, z_low = pad_z(m, nest), ground_z(m, zone) + LOWER_M
    held = holds_item(tl, m, zone, kfs[0].t)
    aborted = task.plan.kind == DELIVER and held
    lowers = task.plan.kind == DELIVER and not held
    out: list[Problem] = _response_problems(tl, task, held, aborted)
    if task.targets != (m.region_of(zone.center),):
        out.append(Problem("job_targets", f"flight {i} targets {list(task.targets)}, its zone lies in "
                                          f"{m.region_of(zone.center)}"))
    centre = (task.plan.params.get("zone_x"), task.plan.params.get("zone_y"))
    if None in centre or distance(centre, zone.center) > TOL_M:
        out.append(Problem("job_targets", f"flight {i}'s plan puts its zone at {centre}, it lies at "
                                          f"{tuple(zone.center)}"))
    runs = [kf.phase for j, kf in enumerate(kfs) if j == 0 or kf.phase != kfs[j - 1].phase]
    if runs != phase_order(lowers):
        what = "a delivery over a zone that holds an item (aborted)" if aborted else f"a {task.plan.kind}"
        out.append(Problem("job_phase_order", f"flight {i} runs {runs}; {what} runs {phase_order(lowers)}"))
    elif (found := [(d.kind, tl.keyframe(d.k).phase) for d in tl.decisions_in(task.k0, task.k1)]) != DECISIONS:
        out.append(Problem("job_decisions", f"flight {i} holds decisions {found}; drone/1 holds {DECISIONS}"))
    else:
        out += _decision_problems(tl, task, kfs)
    if kfs[-1].t - kfs[0].t > TASK_MAX_S + TOL_S:
        out.append(Problem("job_task_time", f"flight {i} takes {kfs[-1].t - kfs[0].t:.0f} s; drone/1's flights take "
                                            f"at most {TASK_MAX_S:.0f} s"))
    if any(b.status["battery_pct"] > a.status["battery_pct"] for a, b in zip(kfs, kfs[1:], strict=False)):
        out.append(Problem("job_battery", f"flight {i}'s battery rises"))
    return (out + _nest_problems(kfs, i, nest, z_pad) + _height_problems(kfs, i, z_pad, z_low, lowers)
            + _gimbal_problems(kfs, i, robot) + _tether_problems(kfs, i, z_low, lowers, robot))


def _response_problems(tl: Timeline, task: Task, held: bool, aborted: bool) -> list[Problem]:
    """The response of each decision of the flight: `before_lower`'s is `abort` for a delivery whose zone held an item
    at the flight's start (`aborted`), else `go`; the other decisions carry none."""
    out = []
    for d in tl.decisions_in(task.k0, task.k1):
        want = (ABORT if aborted else GO) if d.kind == BEFORE_LOWER else None
        if d.response != want:
            out.append(Problem("job_response", f"flight {task.index}'s {d.kind} (keyframe {d.k}) responds "
                                               f"{d.response!r}; a {task.plan.kind} over a zone that "
                                               f"{'holds' if held else 'holds no'} item responds {want!r}"))
    return out


def _decision_problems(tl: Timeline, task: Task, kfs: list[Keyframe]) -> list[Problem]:
    """Each decision stands on the last keyframe of the first run of its phase."""
    ends: dict[str, list[int]] = {}
    for j, kf in enumerate(kfs):
        if j + 1 == len(kfs) or kfs[j + 1].phase != kf.phase:
            ends.setdefault(kf.phase, []).append(kf.k)
    return [Problem("job_decisions", f"flight {task.index}'s {d.kind} (keyframe {d.k}) is not on the last keyframe of "
                                     f"its {phase!r} run ({ends[phase][0]})")
            for d, (_, phase) in zip(tl.decisions_in(task.k0, task.k1), DECISIONS, strict=True)
            if d.k != ends[phase][0]]


def _nest_problems(kfs: list[Keyframe], i: int, nest: Furniture, z_pad: float) -> list[Problem]:
    out = []
    for kf, which in ((kfs[0], "first"), (kfs[-1], "last")):
        off, high = distance(kf.base[:2], nest.center), abs(kf.base[2] - z_pad)
        if off > TOL_M or high > TOL_M:
            out.append(Problem("job_nest", f"flight {i}'s {which} keyframe {kf.k} is {off:.3f} m from the nest and "
                                           f"{high:.3f} m from the pad's height {z_pad:.3f}"))
    return out


def _height_problems(kfs: list[Keyframe], i: int, z_pad: float, z_low: float, lowers: bool) -> list[Problem]:
    """Between the pad and `ALT_M` always; at `ALT_M` in `approach` and `return`; at the lowering height over the hover:
    the `descend` run's last keyframe, `lower` and `release` of a flight that lowers, the second run of `inspect` of
    one that does not."""
    descended = [kf.k for kf in kfs if kf.phase == "descend"]
    hovering = set(descended[-1:])  # the last keyframe of the descent is the hover's first
    hovering |= {kf.k for kf in kfs if kf.phase in ("lower", "release")}
    if not lowers:
        hovering |= {kf.k for kf in kfs if kf.phase == "inspect" and descended and kf.k > descended[-1]}
    out = []
    for kf in kfs:
        z, why = kf.base[2], None
        if not z_pad - TOL_M <= z <= ALT_M + TOL_M:
            why = f"outside the pad's {z_pad:.3f} m to {ALT_M} m"
        elif kf.phase in ("approach", "return") and abs(z - ALT_M) > TOL_M:
            why = f"not at the transit height {ALT_M} m"
        elif kf.k in hovering and abs(z - z_low) > TOL_M:
            why = f"not at the lowering height {z_low:.3f} m"
        if why:
            out.append(Problem("job_heights", f"flight {i}'s keyframe {kf.k} ({kf.phase!r}) is at {z:.3f} m, {why}"))
    return out


def _gimbal_problems(kfs: list[Keyframe], i: int, robot: Robot) -> list[Problem]:
    out = []
    lo, hi = robot.camera.pitch_min_deg, robot.camera.pitch_max_deg
    for kf in kfs:
        pitch, want = math.degrees(kf.cam_pitch), PITCH_DEG[PHASE_PITCH[kf.phase]]
        if abs(pitch - want) > TOL_DEG:
            out.append(Problem("job_gimbal", f"flight {i}'s keyframe {kf.k} looks {pitch:.2f} degrees in {kf.phase!r}; "
                                             f"the phase's is {want}"))
        elif not lo - TOL_DEG <= pitch <= hi + TOL_DEG:
            out.append(Problem("job_gimbal", f"flight {i}'s keyframe {kf.k} looks {pitch:.2f} degrees, outside the "
                                             f"gimbal's {lo} to {hi}"))
    for a, b in zip(kfs, kfs[1:], strict=False):
        turned = abs(math.degrees(b.cam_pitch - a.cam_pitch))
        if turned > GIMBAL_DPS * (b.t - a.t) + TOL_DEG:
            out.append(Problem("job_gimbal", f"flight {i}'s gimbal turns {turned:.1f} degrees between keyframes {a.k} "
                                             f"and {b.k}, {b.t - a.t:.2f} s apart; it turns {GIMBAL_DPS:.0f} a second"))
    return out


def _tether_problems(kfs: list[Keyframe], i: int, z_low: float, lowers: bool, robot: Robot) -> list[Problem]:
    """The tether of a flight that lowers its package (a delivery that was not aborted): over `lower` it rises to
    `LOWER_M` (at the lowering speed at most, and by its last keyframe); over `release` it is `LOWER_M`; over the
    reel-in (`climb_out` at the lowering height) it falls to 0 by the reel-in's last keyframe; 0 elsewhere. An
    inspection's, and an aborted delivery's, is 0 throughout."""
    top, out = robot.speeds_mps["lower"][1], []
    tether = [float(kf.status["tether_m"]) for kf in kfs]
    for a, b, ta, tb in zip(kfs, kfs[1:], tether, tether[1:], strict=False):
        if abs(tb - ta) > top * (b.t - a.t) + TETHER_TOL_M:
            out.append(Problem("job_tether", f"flight {i}'s tether changes faster than {top} m/s between keyframes "
                                             f"{a.k} and {b.k}"))
    reel = [j for j, kf in enumerate(kfs) if kf.phase == "climb_out" and kf.base[2] <= z_low + TOL_M]
    lower = [j for j, kf in enumerate(kfs) if kf.phase == "lower"]
    for j, kf in enumerate(kfs):
        want = LOWER_M if kf.phase == "release" else None if kf.phase == "lower" or j in reel else 0.0
        if not lowers:
            want = 0.0
        if not -TETHER_TOL_M <= tether[j] <= LOWER_M + TETHER_TOL_M or (
                want is not None and abs(tether[j] - want) > TETHER_TOL_M):
            out.append(Problem("job_tether", f"flight {i}'s keyframe {kf.k} holds {tether[j]:.3f} m of tether in "
                                             f"{kf.phase!r}"))
    if lowers and lower and reel:
        rising = all(tether[a] <= tether[b] + TETHER_TOL_M for a, b in zip(lower, lower[1:], strict=False))
        falling = all(tether[a] >= tether[b] - TETHER_TOL_M for a, b in zip(reel, reel[1:], strict=False))
        if not rising or not falling or abs(tether[lower[-1]] - LOWER_M) > TETHER_TOL_M or tether[reel[-1]] > (
                TETHER_TOL_M):
            out.append(Problem("job_tether", f"flight {i}'s tether does not pay out to {LOWER_M} m over the lowering "
                                             f"and reel in to 0 over the hover"))
    return out
