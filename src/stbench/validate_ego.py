"""The ego rules of `validate`: a robot's keyframes, tasks, decision points, status, camera and body.

Everything here reads the robot from `robots.ROBOTS` (its spec decides the phases, the camera and the body) and the
job, when registered, from `job_registry` (its task kinds, decision vocabulary, anchors and its own `checks`); no
robot's phases or kinds are named. A job that is not registered (its module is not written yet) is checked only for
belonging to the script's robot: its decisions, anchors and checks are not read. The body rule keeps the robot's
footprint at its base `body.clearance_m` (less `BODY_TOL_M`) from what obstructs it, in the state each thing has at that
keyframe's time:
- the map's wall pieces (always, a flier included);
- every fixed and light piece, by `geometry.under_obstacles`: its four leg squares when it has an open underside the
  robot passes under (`robots.passes_under`), else the whole footprint; floor coverings (rugs, mats) never;
- every door leaf, open or shut (a shut leaf closes its opening);
- carried items and clutter lying on the floor and taller than `climb_m`;
and a flier passes over what `robots.flies_over(robots.over_m(robot, base_z), top)` allows. A piece of one of the job's
`anchors` kinds (the vacuum's dock) may be touched, never overlapped. Only the fixed, light and carried classes are
read (clutter collides as carried): the `mover` and `structure` rows of the mobility table obstruct "always", which is
applied when movers return (M2).
"""

from __future__ import annotations

import math

from . import job_registry, people, robots
from .catalog import FLOOR_ITEM_Z, mover_kind
from .geometry import Rect, nearest_on_rect, rect_distance, rects_overlap, under_obstacles
from .job_registry import JobSpec
from .mapspec import MapSpec
from .timeline import Timeline
from .validate import QUAT_TOL, Problem, collision_class, door_entities, door_leaf_rect, non_blocking

BODY_TOL_M = 0.001  # the clearance a keyframe may fall short of by, and the overlap an anchor may be touched with
SPACING_TOL_S = 1e-6  # the spacing a keyframe may fall short of `min_dt_s` by (the file keeps 6 decimals)
CAMERA_TOL = 1e-6  # the camera's pose may differ from the one derived from the base by this much
PITCH_TOL_DEG = 1e-4  # the file keeps the pitch to 6 decimals of a radian
MOVER_CAMERA_TOL = 1e-6
Piece = tuple[Rect, str, float, bool]  # footprint part, what it is, its top, may be touched (a job's anchor)


def ego_problems(tl: Timeline, m: MapSpec) -> list[Problem]:
    """Every problem of the ego, then the job's own: layout, tasks, decisions, then (for a known robot) status, camera
    and body, and the movers' distance from the camera."""
    robot = robots.ROBOTS.get(tl.episode.robot)
    job = _job(tl.episode.job)
    out = _keyframes(tl, m) + _tasks(tl) + _job_rules(tl, job) + _decisions(tl, job)
    if robot is None:
        out.append(Problem("unknown_robot", f"robot {tl.episode.robot!r} is not one of {sorted(robots.ROBOTS)}"))
    else:
        out += _spacing(tl, robot) + _status(tl, robot) + _camera(tl, robot)
        out += _body(tl, m, robot, job)
    out += _mover_camera(tl, robot)
    if job is not None:
        out += job.checks(tl, m)
    return out


def _job(job_id: str) -> JobSpec | None:
    try:
        return job_registry.get(job_id)
    except KeyError:
        return None


def _keyframes(tl: Timeline, m: MapSpec) -> list[Problem]:
    out: list[Problem] = []
    kfs = tl.ego.keyframes
    for i, kf in enumerate(kfs):
        if kf.k != i:
            out.append(Problem("keyframe_index", f"keyframe at index {i} has k={kf.k}"))
        if i and kf.t < kfs[i - 1].t:
            out.append(Problem("keyframe_order", f"keyframe {kf.k} at t={kf.t} precedes t={kfs[i - 1].t}"))
        if not m.contains((kf.base[0], kf.base[1])):
            out.append(Problem("outside_map", f"keyframe {kf.k}'s base at {kf.base} is outside the map"))
        if abs(sum(c * c for c in kf.rot) - 1.0) > QUAT_TOL:
            out.append(Problem("bad_rotation", f"keyframe {kf.k} has non-unit rotation {kf.rot}"))
    return out


def _tasks(tl: Timeline) -> list[Problem]:
    out: list[Problem] = []
    tasks, kfs = tl.ego.tasks, tl.ego.keyframes
    for i, t in enumerate(tasks):
        if t.index != i:
            out.append(Problem("task_layout", f"the task at position {i} has index {t.index}"))
        if t.k1 <= t.k0:
            out.append(Problem("task_layout", f"task {i} is empty: [{t.k0}, {t.k1})"))
        elif t.k0 != (tasks[i - 1].k1 if i else 0):
            out.append(Problem("task_layout", f"task {i} starts at keyframe {t.k0}, not where the last one ended"))
    if kfs and (not tasks or tasks[-1].k1 != len(kfs)):
        end = tasks[-1].k1 if tasks else 0
        out.append(Problem("task_layout", f"the tasks end at keyframe {end}, but there are {len(kfs)} keyframes"))
    for kf in kfs:
        holder = [t.index for t in tasks if t.k0 <= kf.k < t.k1]
        if not holder:
            out.append(Problem("keyframe_outside_task", f"keyframe {kf.k} lies in no task"))
        elif kf.task not in holder:
            out.append(Problem("keyframe_outside_task", f"keyframe {kf.k} says task {kf.task}, in task {holder[0]}"))
    return out


def _job_rules(tl: Timeline, job: JobSpec | None) -> list[Problem]:
    """job_robot: the job belongs to another robot than the script's: the registered job's `robot`, or for a job that is
    not registered the robot its id names (the part before "/": a helper's script whose job is `vacuum/1`, a mistyped
    `vaccum/1`). task_kind: a task's plan is of a kind outside the registered job's. A job that is not registered gets
    no more than the first check."""
    ep = tl.episode
    owner = ep.job.partition("/")[0] if job is None else job.robot
    out: list[Problem] = []
    if owner != ep.robot:
        out.append(Problem("job_robot", f"job {ep.job!r} belongs to robot {owner!r}, not to the script's {ep.robot!r}"))
    if job is not None:
        out += [Problem("task_kind", f"task {t.index}'s plan kind {t.plan.kind!r} is not one of {job.id}'s: "
                                     f"{', '.join(job.task_kinds)}")
                for t in tl.ego.tasks if t.plan.kind not in job.task_kinds]
    return out


def _decisions(tl: Timeline, job: JobSpec | None) -> list[Problem]:
    out: list[Problem] = []
    last = -1
    for d in tl.ego.decisions:
        if d.k <= last:
            out.append(Problem("decision_order", f"decision at keyframe {d.k} does not follow the one at {last}"))
        last = d.k
        if not any(t.k0 <= d.k < t.k1 for t in tl.ego.tasks):
            out.append(Problem("decision_order", f"decision at keyframe {d.k} lies in no task"))
        if job is not None and d.kind not in job.decisions:
            out.append(Problem("decision_kind", f"decision {d.kind!r} at keyframe {d.k} is not one of {job.id}'s: "
                                                f"{', '.join(job.decisions)}"))
    return out


def _spacing(tl: Timeline, robot: robots.Robot) -> list[Problem]:
    """Two keyframes of one task are at least the robot's `min_dt_s` apart, so every clock of a task is unique at whole
    seconds."""
    kfs = tl.ego.keyframes
    return [Problem("frame_spacing", f"keyframes {a.k} and {b.k} of task {a.task} are {b.t - a.t:.3f} s apart; "
                                     f"{robot.id} keeps {robot.frames.min_dt_s}")
            for a, b in zip(kfs, kfs[1:], strict=False)
            if a.task == b.task and b.t - a.t < robot.frames.min_dt_s - SPACING_TOL_S]


def _status(tl: Timeline, robot: robots.Robot) -> list[Problem]:
    """Phase and status against the robot's vocabulary; each task's first keyframe also against the start set."""
    out: list[Problem] = []
    firsts = {t.k0 for t in tl.ego.tasks}
    for kf in tl.ego.keyframes:
        status = {"phase": kf.phase, **kf.status}
        found = robots.check_status(robot, status, first=False)
        for text in found:
            out.append(Problem("unknown_phase" if text.startswith("unknown phase") else "bad_status",
                               f"keyframe {kf.k}: {text}"))
        if kf.k in firsts:
            out += [Problem("start_status", f"keyframe {kf.k}: {text}")
                    for text in robots.check_status(robot, status, first=True) if text not in found]
    return out


def _camera(tl: Timeline, robot: robots.Robot) -> list[Problem]:
    """The camera is derived from the base (`robots.camera_pose`) and tilts only within the robot's limits."""
    out: list[Problem] = []
    cam = robot.camera
    for kf in tl.ego.keyframes:
        pos, rot = robots.camera_pose(robot, kf.base, kf.yaw, kf.cam_pitch)
        off = max(abs(a - b) for a, b in zip((*pos, *rot), (*kf.pos, *kf.rot), strict=True))
        if off > CAMERA_TOL:
            out.append(Problem("camera_mismatch", f"keyframe {kf.k}'s camera is {off:.4f} off the pose its base gives"))
        pitch = math.degrees(kf.cam_pitch)
        if not cam.pitch_min_deg - PITCH_TOL_DEG <= pitch <= cam.pitch_max_deg + PITCH_TOL_DEG:
            out.append(Problem("pitch_limit", f"keyframe {kf.k}'s camera pitch {pitch:.2f} deg is outside "
                                              f"{cam.pitch_min_deg} to {cam.pitch_max_deg}"))
    return out


# -- the body rule -----------------------------------------------------------------------------------------------
def _parts(rect: Rect, under_m: float, robot: robots.Robot) -> list[Rect]:
    """What of a piece obstructs: the legs when it has an open underside the robot passes under, else all of it."""
    return under_obstacles(rect, under_m) if under_m > 0 and robots.passes_under(robot, under_m) else [rect]


def _gap(robot: robots.Robot, base: tuple[float, float, float], yaw: float, rect: Rect, touch: bool) -> float:
    """How far the robot's footprint at its base is from a rectangle: a disc's centre distance less its radius; a
    rectangle's `geometry.rect_distance` at its heading. A piece that may be touched (`touch`) is only a problem when
    overlapped deeper than `BODY_TOL_M`, which a rectangle reports as -1 (its distance is 0 for any overlap)."""
    body = robot.body
    if body.shape == "disc":
        return nearest_on_rect(rect, base[0], base[1])[1] - body.length_m / 2
    foot = Rect(base[0], base[1], body.length_m, body.width_m, yaw)
    if touch:
        return -1.0 if rects_overlap(foot, rect, -BODY_TOL_M) else 0.0
    return rect_distance(foot, rect)


def _body(tl: Timeline, m: MapSpec, robot: robots.Robot, job: JobSpec | None) -> list[Problem]:
    anchors = job.anchors if job is not None else frozenset()
    body = robot.body
    reach = math.hypot(body.length_m, body.width_m) / 2  # the footprint lies within this of the base
    static: list[Piece] = [(r, "a wall", math.inf, False) for r in m.wall_pieces()]
    moving = []  # the entities whose rectangle depends on the time
    for e in tl.entities:
        cls = collision_class(e)
        if cls not in ("fixed", "light", "carried", "mover") or non_blocking(e):  # not structure: see the docstring
            continue
        if cls == "fixed":
            r = tl.footprint(e.id, 0.0)
            if r is not None:
                static += [(part, e.id, e.size[2], e.kind in anchors) for part in _parts(r, e.under_m, robot)]
        else:
            moving.append(e)
    doors = [(eid, tl.entity(eid), next(o for o in m.doors() if o.id == door_id))
             for door_id, eid in door_entities(tl, m).items()]
    out: list[Problem] = []
    for kf in tl.ego.keyframes:
        over = robots.over_m(robot, kf.base[2])
        # a flier passes over what is lower than it clears, but never over a wall
        pieces = [p for p in static if p[1] == "a wall" or not robots.flies_over(over, p[2])]
        for e in moving:
            p = tl.placement_of(e.id, kf.t)
            if p is None or robots.flies_over(over, p.pos[2] + e.size[2]):
                continue
            cls = collision_class(e)
            if cls == "carried" and (p.pos[2] >= FLOOR_ITEM_Z or e.size[2] <= body.climb_m):
                continue  # not on the floor, or low enough to drive over
            r = Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)
            parts = [r] if cls in ("carried", "mover") else _parts(r, e.under_m, robot)
            pieces += [(part, e.id, p.pos[2] + e.size[2], e.kind in anchors) for part in parts]
        for eid, entity, opening in doors:
            if not robots.flies_over(over, entity.size[2]):
                leaf = door_leaf_rect(m, opening, bool(tl.state_of(eid, kf.t, "open")), entity.size[1])
                pieces.append((leaf, f"the leaf of door {eid}", entity.size[2], False))
        for rect, what, _, touch in pieces:
            need = -BODY_TOL_M if touch else body.clearance_m - BODY_TOL_M
            if math.hypot(rect.cx - kf.base[0], rect.cy - kf.base[1]) - math.hypot(rect.sx, rect.sy) / 2 - reach > need:
                continue  # too far for the footprints to come within `need`
            d = _gap(robot, kf.base, kf.yaw, rect, touch)
            if d < need:
                how = f"overlaps {what}, an anchor, which may be touched" if touch else (
                    f"is {d:.3f} m from {what}; it must keep {need:.3f} m")
                out.append(Problem("body_clearance", f"keyframe {kf.k}: the {robot.id}'s body {how}"))
    return out


# -- movers ------------------------------------------------------------------------------------------------------
def _kind_of(name: str):
    """A mover's kind record: a catalog's mover kind, else a person kind (`people`, a truth module: the catalog, which
    the neutral modules and the baselines read, never imports it)."""
    try:
        return mover_kind(name)
    except KeyError:
        return people.person_kind(name)


def _mover_camera(tl: Timeline, robot: robots.Robot | None) -> list[Problem]:
    """mover_camera_clearance: at every keyframe each mover's centre is at least its kind's `camera_m` from the camera
    (xy), unless the robot flies over the mover. unknown_mover_kind: a mover of a kind no catalog has and no person kind
    is (its clearance, box and model are unknown)."""
    out: list[Problem] = []
    keep: dict[str, tuple[float, float]] = {}
    for e in tl.entities:
        if collision_class(e) != "mover":
            continue
        try:
            kind = _kind_of(e.kind)
        except KeyError:
            out.append(Problem("unknown_mover_kind", f"mover {e.id} is a {e.kind!r}, a kind of no catalog"))
            continue
        keep[e.id] = (kind.camera_m, e.size[2])  # the entity's own height (a person's box is its model's)
    for kf in tl.ego.keyframes:
        over = None if robot is None else robots.over_m(robot, kf.base[2])
        for eid, (camera_m, top) in keep.items():
            p = tl.placement_of(eid, kf.t)
            if p is None or robots.flies_over(over, p.pos[2] + top):  # over a mover lower than it clears, on its floor
                continue
            d = math.hypot(p.pos[0] - kf.pos[0], p.pos[1] - kf.pos[1])
            if d < camera_m - MOVER_CAMERA_TOL:
                msg = f"{eid} is {d:.3f} m from the camera at keyframe {kf.k}; its kind keeps {camera_m}"
                out.append(Problem("mover_camera_clearance", msg))
    return out
