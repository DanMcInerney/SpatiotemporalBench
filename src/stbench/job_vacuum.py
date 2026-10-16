"""The vacuum's job, `vacuum/1`: clean a room, then go back to the dock.

The declarative part is the robot, the domain, the task kinds, the decision points, the events, what is task-critical,
the dock as the one thing the robot may touch, and the assignment's words. `_plan` draws the task's room, `_script`
writes the path (docked, undocking, transit, cleaning, transit, docking, docked), the status and the decision points
over a built world, and `_checks` holds the job's own `validate` rules. The path is planned on `route.robot_free_space`,
covered by `coverage` (an edge pass, then strips) and sampled into keyframes by `keyframes` under the robot's frame
policy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import coverage
from .events import EventRule
from .geometry import Point, distance, q, wrap_angle
from .job_registry import JobSpec, Relevance
from .job_vacuum_path import Driver, crossing
from .keyframes import Hold, Mark, Move, frames
from .mapspec import Furniture, MapSpec, Region
from .robots import ROBOTS, Robot, camera_pose
from .route import FreeSpace, robot_free_space
from .seeding import make_rng
from .timeline_ego import DecisionPoint, Ego, Keyframe, Task, TaskPlan
from .validate import Problem

if TYPE_CHECKING:
    from .changes import World
    from .timeline import Timeline
    from .world_processes import Between

JOB_ID = "vacuum/1"
DOCK_KIND = "dock"  # the job's anchor: the map's one piece of this kind is the dock
UNDOCK_M, DOCK_REVERSE_M = 0.5, 0.4  # forward off the dock; back onto it, from the point this far in front of it
DOORWAY_STANDOFF_M = 0.5  # a before_doorway keyframe: the base this far before the door line, along the path
STRIP_M = 0.20  # strip spacing, along the room's longer axis (the brush is 0.25 m across)
DRAIN_PCT_PER_MIN = 1.1  # battery: a Roomba's 90-minute run
# A task's room is drawn from the regions other than the dock's own (the hall): a hall task crosses no doorway and asks
# nothing during the job (DECISIONS, "M1's calls before the lanes start"). M2's multi-room tasks bring the hall back.
EXCLUDE_DOCK_ROOM = True
PHASE_ORDER = ("docked", "undocking", "transit", "cleaning", "transit", "docking", "docked")  # a task's phases
SPEED_PHASES = ("undocking", "transit", "cleaning", "docking")  # speeds are drawn once a task, in this order
BEFORE_DOORWAY, ROOM_DONE = "before_doorway", "room_done"  # the job's decision kinds
DOCKED_TOL_M, DOCKED_TOL_RAD = 0.01, 1e-3  # how near a task's first and last pose must be the docked pose

# What the vacuum meets on a floor: the floor-level things, and the light furniture. Floor clutter is not here: no M1
# question names it and `critical` reads asked subjects only (DECISIONS, "M1's calls before the lanes start").
FLOOR_KINDS = frozenset({"shoes", "bag", "toy", "box", "chair", "stool", "laundry_basket"})
PHASES = frozenset({"cleaning", "transit", "docking"})


def _dock(m: MapSpec) -> Furniture:
    docks = [f for f in m.furniture if f.kind == DOCK_KIND]
    if len(docks) != 1:
        raise ValueError(f"{m.name} must hold exactly one {DOCK_KIND!r} piece for {JOB_ID}, not {len(docks)}")
    return docks[0]


def docked_pose(m: MapSpec) -> tuple[float, float, float]:
    """(x, y, yaw) of the docked vacuum: its disc touching the dock's front face, facing out (the dock's own yaw)."""
    dock = _dock(m)
    reach = dock.size[0] / 2 + ROBOTS["vacuum"].body.length_m / 2
    return (q(dock.center[0] + math.cos(dock.yaw) * reach), q(dock.center[1] + math.sin(dock.yaw) * reach),
            q(wrap_angle(dock.yaw)))


def _region_of_kind(m: MapSpec, kind: str) -> Region:
    rooms = [r for r in m.regions if r.kind == kind]
    if len(rooms) != 1:
        raise ValueError(f"{m.name} has {len(rooms)} regions of kind {kind!r}; a task names its room by a unique kind")
    return rooms[0]


def _plan(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    if n_tasks != 1:
        raise ValueError(f"{JOB_ID} writes one task a script in M1, not {n_tasks}: multi-task scripts come with M2's "
                         f"gaps and world processes")
    dock_room = m.region_of(_dock(m).center)
    rooms = [r for r in m.regions if not (EXCLUDE_DOCK_ROOM and r.id == dock_room)]
    rng = make_rng(m.name, seed, JOB_ID, "job")
    return [TaskPlan("clean_room", {"room": rooms[int(rng.integers(len(rooms)))].kind})]


# -- the path ----------------------------------------------------------------------------------------------------
def _way(free: FreeSpace, a: Point, b: Point, what: str) -> list[Point]:
    way = free.path(a, b, strict=True)
    if way is None:
        raise ValueError(f"no way {what} on the vacuum's free space: {a} to {b} (is the dock's apron blocked?)")
    return [p for i, p in enumerate(way) if i == 0 or distance(p, way[i - 1]) > 1e-9]


def _entry(m: MapSpec, free: FreeSpace, room: Region, start: Point) -> Point:
    """Where the vacuum comes into the room: the free floor of the room nearest to one of its openings, whichever the
    shortest way from `start` reaches."""
    xs, ys = free.cells_xy(coverage.room_floor(free, room))
    found = []
    for o in sorted((o for o in m.openings if room.id in o.between), key=lambda o: o.id):
        i = int(((xs - o.center[0]) ** 2 + (ys - o.center[1]) ** 2).argmin())
        cell = (float(xs[i]), float(ys[i]))
        way = free.path(start, cell, strict=True)
        if way is not None:
            found.append((sum(distance(a, b) for a, b in zip(way, way[1:], strict=False)), cell))
    if not found:
        raise ValueError(f"no way into the {room.name} from {start}")
    return min(found)[1]


def _task_prims(m: MapSpec, free: FreeSpace, robot: Robot, room: Region, speed: dict[str, float]) -> list:
    """The primitives of one task: docked, undocking, transit in (a before_doorway decision), cleaning (an edge pass,
    then strips; a room_done decision), transit out (a before_doorway decision), docking, docked."""
    x, y, yaw = docked_pose(m)
    home, out = (x, y), (x + UNDOCK_M * math.cos(yaw), y + UNDOCK_M * math.sin(yaw))
    front = (x + DOCK_REVERSE_M * math.cos(yaw), y + DOCK_REVERSE_M * math.sin(yaw))
    entry = _entry(m, free, room, out)
    loop = coverage.perimeter(free, room, entry)
    cleaning = loop + coverage.strips(free, room, entry, STRIP_M, loop[-1])[1:]
    way_in, way_out = _way(free, out, loop[0], "to the room"), _way(free, cleaning[-1], front, "back to the dock")
    drive = Driver(yaw, robot.turn_dps)
    at, s = crossing(m, room, way_in, DOORWAY_STANDOFF_M, last=True)
    drive.go(way_in, speed["transit"], "transit", (at, s, BEFORE_DOORWAY))
    drive.go(cleaning, speed["cleaning"], "cleaning")
    drive.prims.append(Mark(ROOM_DONE))
    at, s = crossing(m, room, way_out, DOORWAY_STANDOFF_M, last=False)
    drive.go(way_out, speed["transit"], "transit", (at, s, BEFORE_DOORWAY))
    drive.turn(front, yaw, "docking")
    drive.prims.append(Move(front, home, yaw, speed["docking"], "docking"))  # reversing: heading kept, motion opposite
    return [Hold(home, yaw, 0.0, "docked"), Move(home, out, yaw, speed["undocking"], "undocking"), *drive.prims,
            Hold(home, yaw, 0.0, "docked")]


def _script(m: MapSpec, w: World, plans: list[TaskPlan], seed: int, between: Between | None = None) -> Ego:
    # `between` (the world's gap callback) is accepted and never called: `vacuum/1` writes one task, so it has no gap.
    if len(plans) != 1:
        raise ValueError(f"{JOB_ID} writes one task a script in M1, not {len(plans)}")
    robot = ROBOTS["vacuum"]
    room = _region_of_kind(m, str(plans[0].params["room"]))
    rng = make_rng(m.name, seed, JOB_ID, "job", 0)
    low, high = robot.status.start["battery_pct"]
    battery = int(rng.integers(low, high + 1))
    speed = {phase: float(rng.uniform(*robot.speeds_mps[phase])) for phase in SPEED_PHASES}
    steps = frames(_task_prims(m, robot_free_space(m, w, robot), robot, room, speed), robot.frames)
    pitch = q(math.radians(robot.camera.pitch_deg))
    kfs = []
    for k, f in enumerate(steps):
        base = (f.xy[0], f.xy[1], 0.0)
        pos, rot = camera_pose(robot, base, f.yaw, pitch)
        drained = int(DRAIN_PCT_PER_MIN * (f.t - steps[0].t) / 60)
        kfs.append(Keyframe(k, f.t, pos, rot, base, f.yaw, pitch, 0, f.phase,
                            {"battery_pct": battery - drained, "brush": f.phase == "cleaning"}))
    decisions = [DecisionPoint(k, f.decision) for k, f in enumerate(steps) if f.decision]
    return Ego(kfs, [Task(0, 0, len(kfs), plans[0], (room.id,))], decisions)


# -- the job's own checks ----------------------------------------------------------------------------------------
def _checks(tl: Timeline, m: MapSpec) -> list[Problem]:
    """The job's rules on top of `validate`'s, for each task (none when the script has none): the phases of
    `PHASE_ORDER`, each present, in order; the brush on exactly in `cleaning`; the first and last keyframe at the docked
    pose; and exactly a before_doorway in transit, room_done on the last cleaning keyframe and a before_doorway in
    transit. A map without a dock cannot hold the job (`job_dock`)."""
    if not tl.ego.tasks:
        return []
    try:
        dock = docked_pose(m)
    except ValueError as e:
        return [Problem("job_dock", str(e))]
    return [p for task in tl.ego.tasks for p in _task_problems(tl, task, dock)]


def _task_problems(tl: Timeline, task: Task, dock: tuple[float, float, float]) -> list[Problem]:
    kfs = [tl.keyframe(k) for k in range(task.k0, min(task.k1, tl.n_keyframes))]
    if not kfs:
        return []  # `validate` reports the task's layout
    out = []
    phases = [kf.phase for i, kf in enumerate(kfs) if i == 0 or kf.phase != kfs[i - 1].phase]
    if tuple(phases) != PHASE_ORDER:
        out.append(Problem("job_phase_order", f"task {task.index} runs {phases}; {JOB_ID} runs {list(PHASE_ORDER)}"))
    out += [Problem("job_brush", f"keyframe {kf.k}: the brush is {kf.status.get('brush')} in {kf.phase!r}")
            for kf in kfs if kf.status.get("brush") != (kf.phase == "cleaning")]
    for kf, which in ((kfs[0], "first"), (kfs[-1], "last")):
        off, turn = distance(kf.base[:2], dock[:2]), abs(wrap_angle(kf.yaw - dock[2]))
        if off > DOCKED_TOL_M or turn > DOCKED_TOL_RAD:
            out.append(Problem("job_dock", f"task {task.index}'s {which} keyframe {kf.k} is {off:.3f} m and {turn:.4f} "
                                           f"rad from the docked pose {dock}"))
    decisions = [d for d in tl.decisions_in(task.k0, task.k1) if d.k < tl.n_keyframes]
    found = [(d.kind, tl.keyframe(d.k).phase) for d in decisions]
    if found != [(BEFORE_DOORWAY, "transit"), (ROOM_DONE, "cleaning"), (BEFORE_DOORWAY, "transit")]:
        want = f"{BEFORE_DOORWAY} in transit, {ROOM_DONE} in cleaning, {BEFORE_DOORWAY} in transit"
        out.append(Problem("job_decisions", f"task {task.index} holds decisions {found}; {JOB_ID} holds {want}"))
    elif decisions[1].k != max(kf.k for kf in kfs if kf.phase == "cleaning"):
        out.append(Problem("job_decisions", f"task {task.index}'s {ROOM_DONE} is not on its last cleaning keyframe"))
    return out


def _relevance(plan: TaskPlan) -> Relevance:
    return Relevance(FLOOR_KINDS, frozenset({str(plan.params["room"]), "hall"}), PHASES)


SPEC = JobSpec(
    id=JOB_ID,
    robot="vacuum",
    domain="home",
    task_kinds=("clean_room",),
    # `path_ahead` left `before_doorway` in M1 (DECISIONS, "M1's review"): the decision points stay, so the job keeps
    # its id, its scripts and its renders, and the family returns with M2's sites.
    decisions={BEFORE_DOORWAY: (), ROOM_DONE: ()},
    events={
        "when you left the dock": EventRule("last", "undocking"),
        "when you started cleaning": EventRule("first", "cleaning"),
        "when you finished cleaning": EventRule("last", "cleaning"),  # the `room_done` decision sits on it
        "when you got back to the dock": EventRule("first", "docked", after_phase="docking"),
    },
    plan=_plan,
    script=_script,
    relevance=_relevance,
    checks=_checks,
    anchors=frozenset({DOCK_KIND}),
    assignments={"clean_room": "Its current assignment is to clean the {room}."},
    swaps=(),
    # 0.7.0: the vacuum's numbers (FD-0.7.0-review "values to declare", as `vacuum/2`'s): C 0.10 m, R 1.0 m. This job
    # asks no look point question, so nothing reads C; its default end family, `object_where`, reads R.
    clearance=0.10,
    reacquire=1.0,
)
