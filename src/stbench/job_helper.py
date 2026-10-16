"""The home helper's job, `helper/1`: scripts of one to six tasks in which a Stretch-class helper, parked on the clear
apron of the home's charging corner, carries a thing from one surface to another or brings one to a surface of its
room, then parks again, with unobserved gaps (and their world changes, and the resident) between tasks
(docs/job-helper.md).

The declarative part is here: the robot, the task kinds, the decisions and the families they ask, the four events, what
is task-critical, the dock as the one anchor and the assignments' words. `job_helper_plan` draws the plan and the gaps
from the map and the seed alone, `job_helper_reach` finds the surfaces the helper can reach and the poses it reaches
them from, `job_helper_carry` makes the thing it carries a world entity (with `job_helper_spot`, the window it puts it
down in and the coin that sets it), `job_helper_subgoal` states what it is doing, `job_helper_checks` holds the job's
own `validate` rules.

The path of a task is: waiting at the park pose; navigating to a pose within reach of the source surface (a
`before_doorway` 0.5 m before the doorway into the source's room, when that is another room than the park room); a turn
to face the surface, the camera tilting down from -15 to -45 degrees, `before_grasp`, grasping, the camera tilting back
up as carrying begins; carrying to the target surface (a `before_doorway` into its room), the turn, the tilt down,
`before_place`, placing (on an `elsewhere` that the window's surface cannot meet, the helper first drives to the room's
other counted piece, tilting up and down again, all of it `placing`), the tilt back up as navigating begins,
navigating back to the park pose (a `before_doorway` into the park room, when the target was in another), the turn to
the park heading and waiting. A fetch's target is the park room's piece nearest the park pose. A task is planned on
`route.robot_free_space` with every interior door off the ways between its rooms shut in both leaf positions, so no
door's state can change a path. The helper's body and arm are not drawn; the thing it carries is a world entity that
leaves its source at the last grasping frame and appears at its target at the last placing frame, and is hidden in
between.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .catalog import HOME
from .events import EventRule
from .geometry import Point, distance, q, wrap_angle
from .job_helper_carry import commit, edge_time, prepare, targets
from .job_helper_checks import checks as _checks
from .job_helper_plan import (  # noqa: F401  (the constants are this module's public names)
    BEFORE_DOORWAY,
    BEFORE_GRASP,
    BEFORE_PLACE,
    DOCK_KIND,
    DOORWAY_STANDOFF_M,
    DRAIN_PCT_PER_MIN,
    FETCH,
    GRASP_S,
    JOB_ID,
    PARK_FROM_DOCK_M,
    PARK_HOLD_S,
    PLACE_S,
    SPOT_COIN_P,
    TASK_MAX_S,
    TIDY,
    TIDY_P,
    TILT_DEG,
    TILT_DPS,
    gaps,
    park_pose,
    park_room,
    plan_tasks,
    spot_blocked,
)
from .job_helper_reach import ROBOT_ID, Pose, free_space_for, qualified, reach_pose
from .job_helper_spot import ZONE
from .job_helper_subgoal import subgoals
from .job_registry import JobSpec, Relevance
from .job_vacuum_path import Driver, crossing
from .keyframes import Hold, Mark, Tilt, Turn, frames
from .mapspec import MapSpec
from .robots import ROBOTS, Robot, camera_pose
from .route import FreeSpace
from .seeding import make_rng
from .task_predicate import TaskThings
from .timeline_ego import DecisionPoint, Ego, Keyframe, Task, TaskPlan

if TYPE_CHECKING:
    import numpy as np

    from .changes import World
    from .world_processes import Between

SPEED_PHASES = ("navigating", "carrying")  # speeds are drawn once a task, in this order
PHASES = frozenset({"navigating", "carrying"})  # what is task-critical to see: the ground the helper covers
DRIVING = frozenset({"navigating", "carrying"})  # where the helper drives: `clear_path` and `way_to` are asked here


def _plan(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    return plan_tasks(m, seed, n_tasks)


# -- the path ----------------------------------------------------------------------------------------------------
def _way(free: FreeSpace, a: Point, b: Point, what: str) -> list[Point]:
    way = free.path(a, b, strict=True)
    if way is None:
        raise ValueError(f"no way {what} on the helper's free space: {a} to {b}")
    return [p for i, p in enumerate(way) if i == 0 or distance(p, way[i - 1]) > 1e-9]


def _face(drive: Driver, at: Point, yaw: float, phase: str) -> None:
    """Turn in place to exactly `yaw`, however small the turn (`Driver.turn` leaves one under a degree to the next
    segment): a grasp, a place and the park heading are faced to within a micro-radian."""
    if wrap_angle(yaw - drive.yaw) != 0.0:
        drive.prims.append(Turn(at, drive.yaw, yaw, drive.turn_dps, phase))
    drive.yaw = yaw


def _spot(m: MapSpec, free: FreeSpace, mask: np.ndarray, robot: Robot, region: str, preferred: str, what: str
          ) -> Pose:
    """The reach pose at the room's surface the task prefers, else at the room's other counted pieces in id order;
    ValueError naming the pieces tried when none has one in this world."""
    ids = [preferred, *(p for p in qualified(m, park_pose(m)[0])[region] if p != preferred)]
    pieces = m.furniture_by_id()
    for pid in ids:
        if (pose := reach_pose(m, free, mask, pieces[pid], robot)) is not None:
            return pose
    raise ValueError(f"no reach pose at the {what} ({m.region(region).name}: pieces {', '.join(ids)}) in this world")


def _leg(m: MapSpec, free: FreeSpace, drive: Driver, a: Point, b: Point, origin: str, dest: str, speed: float,
         phase: str, what: str) -> None:
    """Drive from `a` to `b`: a `before_doorway` 0.5 m before the doorway into `dest` when it is another room than
    `origin` (the last crossing into it, so a detour round a shut door cannot add or move a decision)."""
    way = _way(free, a, b, what)
    mark = None
    if dest != origin:
        i, s = crossing(m, m.region(dest), way, DOORWAY_STANDOFF_M, last=True)
        mark = (i, s, BEFORE_DOORWAY)
    drive.go(way, speed, phase, mark)


def _aim(at: Point, yaw: float, down: bool, phase: str, robot: Robot) -> Tilt:
    """The camera tilting in place from the default pitch to the grasp's and place's, or back (`down`: to it)."""
    default, low = _pitches(robot)
    return Tilt(at, yaw, 0.0, default if down else low, low if down else default, TILT_DPS, phase)


def _pitches(robot: Robot) -> tuple[float, float]:
    """(the default pitch, the pitch at a grasp and a place), radians."""
    return q(math.radians(robot.camera.pitch_deg)), q(math.radians(TILT_DEG))


def _task_prims(m: MapSpec, free: FreeSpace, robot: Robot, plan: TaskPlan, speed: dict[str, float], grasp_s: float,
                place_s: float, source: Pose, target: Pose, alt: Pose | None) -> list:
    """The primitives of one task (the module docstring). Everything up to the `before_place` mark is the plan's and the
    world's alone; only after it does the response show: `alt`, the pose at the room's other piece, when the thing goes
    there."""
    park, park_yaw = park_pose(m)
    home, (src, dst) = park_room(m), plan.params["regions"]
    low = _pitches(robot)[1]
    drive = Driver(park_yaw, robot.turn_dps, exact=True)
    drive.prims.append(Hold(park, park_yaw, PARK_HOLD_S, "waiting"))
    _leg(m, free, drive, park, source[0], home, src, speed["navigating"], "navigating", "to the source")
    _face(drive, source[0], source[1], "navigating")
    drive.prims += [_aim(*source, True, "navigating", robot), Mark(BEFORE_GRASP),
                    Hold(source[0], source[1], grasp_s, "grasping", pitch=low),
                    _aim(*source, False, "carrying", robot)]
    _leg(m, free, drive, source[0], target[0], src, dst, speed["carrying"], "carrying", "to the target")
    _face(drive, target[0], target[1], "carrying")
    drive.prims += [_aim(*target, True, "carrying", robot), Mark(BEFORE_PLACE)]
    final = target
    if alt is not None:
        drive.prims.append(_aim(*target, False, "placing", robot))
        _leg(m, free, drive, target[0], alt[0], dst, dst, speed["carrying"], "placing", "to the other piece")
        _face(drive, alt[0], alt[1], "placing")
        drive.prims.append(_aim(*alt, True, "placing", robot))
        final = alt
    drive.prims += [Hold(final[0], final[1], place_s, "placing", pitch=low), _aim(*final, False, "navigating", robot)]
    _leg(m, free, drive, final[0], park, dst, home, speed["navigating"], "navigating", "back to the park pose")
    _face(drive, park, park_yaw, "navigating")
    drive.prims.append(Hold(park, park_yaw, PARK_HOLD_S, "waiting"))
    return drive.prims


def _script(m: MapSpec, w: World, plans: list[TaskPlan], seed: int, between: Between | None = None) -> Ego:
    """The script of `plans` over the world `w`. Before each task `between(lo, hi, keep_open)` lets the world change in
    the gap (`lo` None before the first task): `keep_open` is the park room and the task's two, and the task is planned
    on the world after the call, every interior door off the ways between them shut in both leaf positions. The gaps are
    drawn first (`gaps`: the seed's alone), so the world cannot move a task's time. With the reach poses known, the
    thing is chosen and the place spot's coin enforced on the world `EDGE_BUFFER` before the task (`job_helper_carry`),
    on surfaces only, so neither the floor nor the path changes; and last the thing's placements are written."""
    if not plans:
        raise ValueError(f"{JOB_ID} writes at least one task")
    robot, (park, _) = ROBOTS[ROBOT_ID], park_pose(m)
    home, pause = park_room(m), gaps(m.name, seed, len(plans))
    pitch = _pitches(robot)[0]
    low, high = robot.status.start["battery_pct"]
    kfs: list[Keyframe] = []
    tasks: list[Task] = []
    decisions: list[DecisionPoint] = []
    t_end = 0.0
    for i, plan in enumerate(plans):
        src, dst = plan.params["regions"]
        keep_open = frozenset({home, src, dst})
        start = 0.0 if i == 0 else t_end + pause[i - 1]
        if between is not None:
            between(None if i == 0 else t_end, start, keep_open)
        free = free_space_for(m, w, robot, keep_open, start)
        rng = make_rng(m.name, seed, JOB_ID, "job", i)
        battery = int(rng.integers(low, high + 1))
        speed = {phase: float(rng.uniform(*robot.speeds_mps[phase])) for phase in SPEED_PHASES}
        grasp_s, place_s = float(rng.uniform(*GRASP_S)), float(rng.uniform(*PLACE_S))
        try:
            mask = free.reachable_from(park)
            source = _spot(m, free, mask, robot, src, str(plan.params["source"]), "source")
            target = _spot(m, free, mask, robot, dst, str(plan.params["target"]), "target")
            carry = prepare(m, w, free, mask, plan, seed, i, source, target, edge_time(start, i))
            prims = _task_prims(m, free, robot, plan, speed, grasp_s, place_s, source, target, carry.alt)
        except ValueError as e:
            raise ValueError(f"{JOB_ID} on {m.name}, seed {seed}, task {i} ({plan.kind} from the "
                             f"{plan.params['from_room']}): {e}") from e
        steps = frames(prims, robot.frames, t0=start)
        commit(w, carry, steps)
        k0 = len(kfs)
        for j, f in enumerate(steps):
            base = (f.xy[0], f.xy[1], 0.0)
            cam = pitch if f.pitch is None else f.pitch
            pos, rot = camera_pose(robot, base, f.yaw, cam)
            drained = int(DRAIN_PCT_PER_MIN * (f.t - steps[0].t) / 60)
            status = {"battery_pct": battery - drained}
            kfs.append(Keyframe(k0 + j, f.t, pos, rot, base, f.yaw, cam, i, f.phase, status))
            if f.decision:
                answer = carry.response if f.decision == BEFORE_PLACE else None
                decisions.append(DecisionPoint(k0 + j, f.decision, answer))
        tasks.append(Task(i, k0, len(kfs), plan, targets(plan, carry)))
        t_end = steps[-1].t
    return Ego(kfs, tasks, decisions)


def _relevance(plan: TaskPlan) -> Relevance:
    rooms = {str(plan.params[k]) for k in ("from_room", "to_room", "park_room") if k in plan.params}
    return Relevance(frozenset(HOME.kinds), frozenset(rooms), PHASES)


SPEC = JobSpec(
    id=JOB_ID,
    robot=ROBOT_ID,
    domain="home",
    task_kinds=(TIDY, FETCH),
    # `zone_clear` is asked once a task, at `before_place`, about the window of the surface the helper faces: a coin
    # that reads no world sets it occupied or free before the task, and the helper then puts the thing down on the
    # window or beside it (FD-0.5.0-control-impl §4). The earlier during families fell for other reasons
    # (`where_was_i` to `other_case` +0.24; `out_of_view` BROKEN learned shortcuts at the release gate check), so no
    # other kind asks
    # 0.6.0: `zone_clear` is retired, so no decision kind asks a family; the coin and the `elsewhere` response stay
    # (the window's blocker is told: `World.told`), and the questions come at look points (`looks`) and at the end
    decisions={BEFORE_DOORWAY: (), BEFORE_GRASP: (), BEFORE_PLACE: ()},
    events={
        "when you left your spot": EventRule("first", "navigating"),
        "when you started carrying": EventRule("first", "carrying"),
        "when you finished carrying": EventRule("last", "carrying"),  # the `before_place` decision sits on it
        "when you got back to your spot": EventRule("first", "waiting", after_phase="navigating"),
    },
    plan=_plan,
    script=_script,
    relevance=_relevance,
    checks=_checks,
    anchors=frozenset({DOCK_KIND}),
    assignments={TIDY: "Its current assignment is to carry something from the {from_room} to the {to_room}.",
                 FETCH: "Its current assignment is to bring something from the {from_room} to its spot."},
    swaps=(),
    continues=frozenset(),
    # the end question is where the helper last saw a thing of the rooms it works in (`object_where`, critical things
    # that did not move); `distance_between` leaves every job, and the helper's carried thing is never asked (the
    # robot's own log tells about it)
    # 0.6.0: `object_was`, `same_place` and `task_things` join
    # 0.6.0 (DECISIONS, "Before the binding read", 1): `object_was` and `same_place` leave again (0 a medium case here)
    ends=("object_where", "task_things"),
    zones={BEFORE_PLACE: ZONE},
    subgoals=subgoals,
    ladder=(0.25, 0.5, 1.0, 2.0),  # the home's scale: rooms of 3-5 m, things seen from 1-3 m
    # 0.9.0 (FD-decision-questions §6, impl §0): `clear_path` is asked in the four moves of `clear_directions`;
    # `open_heading` names no subject, so a slot can ask it beside the others.
    looks=("clear_path", "closest_approach", "ego_mem", "way_to", "ego_now", "open_heading", "approach_mem"),
    # 0.10.0 (FD-questions-0.10 §3; lane J): three slots a point, harm first. Per drawn slot the lanes measured
    # `clear_path` 0.293, `closest_approach` 0.063, `ego_mem` 0.582, `person_mem` 0.076, `ego_now` 0.359 and
    # `open_heading` 0.467. Slot 1 is the harm questions, whole (100), split evenly: `closest_approach` has the lowest
    # fill of the pair, and half the slot is the least that gives it 20 rows (`abilities.MIN_ABILITY_N`) over the 36
    # dev cases (800 points at 0.5 x 0.063 = 25), while the corridor, at the same half, has the rows to spare. Slot 2 is
    # what was seen before: `person_mem` at 40 (the same count on the same arithmetic, 800 x 0.4 x 0.076 = 24) and
    # `ego_mem` at 25, its fill being so high that a quarter of the draws are enough. Slot 3 is the rest, `ego_now` at
    # 20 and `open_heading` (a diagnostic) at 15, sized so that the three slots give about 0.5 questions a point. The
    # compile gave the fills 0.275, 0.046, 0.512, 0.061, 0.342 and 0.509 and 1 in 12.5 frames on short and medium
    # cases, over the 1 in 15 floor; the helper has the supply to go higher, and the harm share (39% of its look
    # questions) is what raising the first slot would lift. No score enters it.
    # 0.10.0, after the check (DECISIONS 2026-10-14, "The helper keeps clear_path as fixed structure; person_mem leaves
    # it"): `person_mem` failed `separation` (the table +0.368 over blind, 0.40 needed, 19 rows of which 7 answerable),
    # so it is dropped by the evidence rule and its 40 go to `ego_mem` (fill 0.512), which slot 2 now holds whole at 65.
    # `clear_path`'s `memory_needed` evidence cells failed (answerable +0.136 [0.022, 0.283]; long +0.177) and were
    # first read as the helper's recurring routes in one home; the final review found the cause in the
    # decision-frames-first draw instead (`look_decisions` below), and that cause is removed.
    # 0.11 (FD-0.11-design §3.4.2): `way_to`, which way to set off for a thing the helper saw ("bring me my glasses"),
    # takes the 35 that slot 2 drew nothing on, so `ego_mem` keeps its 65 and its supply; it is asked in `look_phases`.
    # 0.11.0 (FD-0.11-hard §3.2): `approach_mem`, "how close would you come to the person you saw last, where you last
    # saw them" (the resident out of view behind the helper as it reverses), takes 30 of the 65 that slot 3 draws
    # nothing on, so `ego_now` and `open_heading` keep their shares and their supply and the helper loses no rows.
    # Expected 15 to 35 rows on the dev root (about 400 points x 0.3 x the evidence rule's fill), which join the large
    # pooled apartment clearance cell: a thin supply stays at the cell weight n / 20, and is reported.
    look_shares=((("clear_path", 50), ("closest_approach", 50)), (("ego_mem", 65), ("way_to", 35)),
                 (("ego_now", 20), ("open_heading", 15), ("approach_mem", 30))),
    # 0.10.0 (FD-questions-0.10 §3): the frames of the grasp and the doorway come first in the look draw (the decisions
    # the fetch turns on). `before_place` is not among them (DECISIONS 2026-10-14, "The final review's repair pass";
    # FD-0.10-final-review B1): a decision frame puts a structure question at the same plan pose, by the same
    # furniture, in every script, and 44 of the helper's 106 `clear_path` rows sat at decision frames, 8 of the carrier
    # gold's 10 at `before_place`. Split by the review: decision frames alone +0.211 (answerable +0.405 [0.181, 0.672]),
    # other frames +0.000, and without `before_place` every gate passes (+0.010 [-0.072, 0.111]). So the draw no
    # longer puts that frame first, and `clear_path` is no longer a fixed-structure exception on this job.
    look_decisions=(BEFORE_GRASP, BEFORE_DOORWAY),
    # The corridor and the heading are asked while the helper drives (`navigating`, `carrying`); in `waiting`,
    # `grasping` and `placing` its status answers them (review §2.5: back waiting +1.00). `clear_path`'s `unknown`
    # golds are kept at 35% (a turned corridor never looked at is `unknown`; brief §6). The helper's home holds one
    # resident (14 shifts) or none (2), which `job_helper_checks` guards (`person_mem`, which names "the person", is
    # no longer asked: 0.10.0).
    clear_phases=DRIVING,
    look_phases={"way_to": DRIVING},
    look_unknown_keep={"clear_path": 35},
    clear_directions=("ahead", "back", "left", "right"),  # the helper turns in place and has a rear (brief §4.1)
    one_person=True,
    # 0.7.0 (FD-0.7.0-difficulty §2.3, FD-0.7.0-review "values to declare"): the clearance C is 0.15 m, the distance
    # from the target within which OWMM-Agent counts a pick (its looser option is 0.8 m); a compliant hand absorbs
    # 2-5 cm of offset, which fits the C/2 rung.
    # R is 1.0 m, the radius within which the robot finds a thing again with it in view: HomeRobot's real-world find
    # rule, and Habitat's ObjectNav success radius (R/2 = 0.5 m is the matching radius of `task_things`).
    clearance=0.15,
    reacquire=1.0,
    task_things=TaskThings("on the floor within 1.2 m of the way you went to fetch and put down the thing", "path",
                           frozenset({"navigating", "grasping", "carrying", "placing"}), until="placing",
                           reach_m=1.2, band_m=0.4, floor=True,
                           empty_keep_percent=26),  # Phase A read: p 0.659
)
