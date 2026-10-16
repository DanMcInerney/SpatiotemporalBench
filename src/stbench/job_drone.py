"""The delivery drone's job, `drone/1`: scripts of one to six flights on `downtown_a` in which a Wing-class tether drone
leaves its nest in the square, flies the streets at 15 m to a drop point, inspects it from above, descends to 7 m,
lowers the package on its tether, releases it, climbs out and flies home (an inspection flight is the same flight, the
lowering not made), with unobserved gaps between flights through which the downtown's own processes change it
(docs/job-drone.md, design §4).

The declarative part is the robot, the domain, the task kinds, the decision points, the events, what is task-critical,
the nest as the one thing the robot may touch, and the assignment's words. `plan_flights`, `gaps` and `coin_blocked`
(`job_drone_plan`) draw from the map and the seed alone; `_script` writes each flight over the world (the way and the
primitives of `job_drone_path`, the zone coin of `job_drone_zone`), `_checks` (`job_drone_checks`) holds the job's own
`validate` rules and `subgoals` (`job_drone_subgoal`) the text of what the drone is doing.

The drone's way depends on the map alone, so no flight's path tells of the world. A drop zone's state is made by the
world before each flight: a coin that reads no world says blocked or clear, and an item is moved in or every item out
(`job_drone_zone.enforce`), so a constant answer earns nothing. The drone asks itself at `before_lower` whether the zone
is clear (`zone_clear`) and then acts on what the zone holds: over a zone that holds an item a delivery is aborted (the
hover is the inspection's, no tether is paid out, the package comes home), over a clear one it goes on. The response is
the world's after the ask, so the path, the frames and the text up to `before_lower` are the same whichever way the coin
fell. The drone's body, tether and package are not drawn (a stated simplification: the camera looks down, so a real
drone's frames would show only the package, which no question reads): an abort changes no pixel, only the status, the
phases and the subgoal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .catalog import DOWNTOWN_KINDS
from .changes import EDGE_BUFFER
from .events import EventRule
from .geometry import q
from .job_drone_checks import checks as _checks
from .job_drone_path import flight_free, flight_prims, flight_way, pad_z, regions_crossed, tethers
from .job_drone_plan import (  # noqa: F401  (the constants are this module's public names)
    ABORT,
    ALT_M,
    BEFORE_DESCENT,
    BEFORE_LOWER,
    BEFORE_RETURN,
    DELIVER,
    DRAIN_PCT_PER_MIN,
    GIMBAL_DPS,
    GO,
    INSPECT,
    INSPECT_P,
    INSPECT_S,
    JOB_ID,
    LEG_M,
    LOWER_M,
    NEST_KIND,
    SPEED_PHASES,
    TASK_MAX_S,
    ZONE_COIN_P,
    coin_blocked,
    drop_zones,
    gaps,
    nest_of,
    plan_flights,
)
from .job_drone_subgoal import subgoals
from .job_drone_zone import enforce, standing, zone_rect
from .job_registry import JobSpec, Relevance
from .keyframes import frames
from .mapspec import MapSpec
from .robots import ROBOTS, camera_pose
from .seeding import make_rng
from .timeline_ego import DecisionPoint, Ego, Keyframe, Task, TaskPlan
from .zones import ZoneRule, ground_z

if TYPE_CHECKING:
    from .changes import World
    from .world_processes import Between

PHASES = frozenset({"approach", "inspect", "descend", "lower", "release", "return"})  # where the task's things matter
ZONE = ZoneRule("the drop zone", "drop_zone")
NEAR_M = 12.0  # a thing is critical within this of the drop zone's centre: the ground half-width the camera sees from
# `ALT_M` (15 m x tan 40 degrees, half the 80 degree horizontal field, is 12.6 m, rounded down)
LADDER = (1.0, 2.0, 4.0, 8.0)  # the distance credit's rungs for the downtown: things are seen from 7 to 15 m up


def _plan(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    return plan_flights(m, seed, n_tasks)


def _script(m: MapSpec, w: World, plans: list[TaskPlan], seed: int, between: Between | None = None) -> Ego:
    """The script of `plans` over the world `w`. A flight's way is the map's and its times and poses are the map's and
    the seed's alone; `between(lo, hi, keep_open)` lets the world change in the gap before it (`lo` None before the
    first: `keep_open` is the regions the way crosses, the downtown having no doors), then the zone coin is enforced on
    the world as the gap left it, `EDGE_BUFFER` before the flight starts (at 0 for the first), and last the flight is
    framed: a delivery whose zone then holds a standing item is aborted at `before_lower` (it flies the inspection's
    phases, pays out no tether, and the decision's response is `abort`; else `go`), which moves no frame's time or pose.
    The gaps are drawn first (`gaps`: the seed's alone), so the world cannot move a flight's time."""
    if not plans:
        raise ValueError(f"{JOB_ID} writes at least one flight")
    robot, nest = ROBOTS["drone"], nest_of(m)
    zones = {z.id: z for z in drop_zones(m)}
    pad, pause, free = pad_z(m, nest), gaps(m.name, seed, len(plans)), flight_free(m, robot)
    low, high = robot.status.start["battery_pct"]
    kfs: list[Keyframe] = []
    tasks: list[Task] = []
    decisions: list[DecisionPoint] = []
    t_end = 0.0
    for i, plan in enumerate(plans):
        zone = zones[str(plan.params["zone"])]
        way = flight_way(free, nest.center, zone.center, f"from the nest to {zone.name}")
        ground = ground_z(m, zone)
        z_low = ground + LOWER_M
        rng = make_rng(m.name, seed, JOB_ID, "job", i)
        battery = int(rng.integers(low, high + 1))
        speed = {phase: float(rng.uniform(*robot.speeds_mps[phase])) for phase in SPEED_PHASES}
        inspect_s = float(rng.uniform(*INSPECT_S))
        start = 0.0 if i == 0 else t_end + pause[i - 1]
        if between is not None:
            between(None if i == 0 else t_end, start, regions_crossed(m, way))
        enforce(w, m, make_rng(m.name, seed, JOB_ID, "drop", i), zone, coin_blocked(m.name, seed, i),
                0.0 if i == 0 else start - EDGE_BUFFER)
        aborted = plan.kind == DELIVER and bool(standing(w, zone_rect(zone), ground))
        flown = INSPECT if aborted else plan.kind
        steps = frames(flight_prims(robot, flown, way, pad, z_low, speed, inspect_s), robot.frames, t0=start)
        tether = tethers(steps, speed["lower"], z_low, flown == DELIVER)
        k0 = len(kfs)
        for j, f in enumerate(steps):
            base = (f.xy[0], f.xy[1], f.z)
            pos, rot = camera_pose(robot, base, f.yaw, f.pitch)
            drained = int(DRAIN_PCT_PER_MIN * (f.t - steps[0].t) / 60)
            kfs.append(Keyframe(k0 + j, f.t, pos, rot, base, f.yaw, f.pitch, i, f.phase,
                                {"battery_pct": battery - drained, "tether_m": q(tether[j])}))
            if f.decision:
                response = (ABORT if aborted else GO) if f.decision == BEFORE_LOWER else None
                decisions.append(DecisionPoint(k0 + j, f.decision, response))
        tasks.append(Task(i, k0, len(kfs), plan, (str(m.region_of(zone.center)),)))
        t_end = steps[-1].t
    return Ego(kfs, tasks, decisions)


def _relevance(plan: TaskPlan) -> Relevance:
    """The things of the downtown's kinds in the drop zone's region and the nest's, and within `NEAR_M` of the drop
    zone's centre: what stands where the drone looked."""
    return Relevance(frozenset(DOWNTOWN_KINDS),
                     frozenset({str(plan.params["region"]), str(plan.params["nest_region"])}), PHASES,
                     near=((float(plan.params["zone_x"]), float(plan.params["zone_y"])),), near_m=NEAR_M)


SPEC = JobSpec(
    id=JOB_ID,
    robot="drone",
    domain="delivery",
    task_kinds=(DELIVER, INSPECT),
    # `before_departure` is no decision of this job: at the nest it falls in the task's first two look windows, where
    # nothing is asked (`questionnaire.GAP_STEPS`). `zone_clear` is asked at `before_lower` alone (DECISIONS, "Camera
    # only, a subgoal in every frame, and the controller's questions"): the drone is at least 16 steps into its flight
    # there in every flight and the zone is in view, and the drone then acts on the answer (an abort, below), so the
    # coin is a decision and not a leftover. `before_descent` is 16 steps into its flight in under half the flights and
    # asks nothing, and so does `before_return`. `clearance_then` is not asked: after the response the drone's own path
    # and subgoal tell what the zone held. `changed` and `out_of_view` are never asked: both read the robot band's free
    # space and the floor at the robot's base, which is in the air. The zone's state is the world's, set by the coin
    # before the flight; the response (`abort` for a delivery over a zone that holds an item, else `go`) follows it.
    # 0.6.0: `zone_clear` is retired, so no decision kind asks a family; the coin and the abort stay (the zone's
    # blocker is told: `World.told`), and the questions come at look points (`looks`) and at the end
    decisions={BEFORE_DESCENT: (), BEFORE_LOWER: (), BEFORE_RETURN: ()},
    events={
        "when you took off": EventRule("first", "takeoff"),
        "when you started lowering the package": EventRule("first", "lower"),
        "when you released the package": EventRule("first", "release"),
        "when you landed": EventRule("last", "land"),
    },
    plan=_plan,
    script=_script,
    relevance=_relevance,
    checks=_checks,
    anchors=frozenset({NEST_KIND}),
    assignments={DELIVER: "Its current assignment is to deliver a package to the drop point at {address}.",
                 INSPECT: "Its current assignment is to inspect the drop point at {address}."},
    swaps=((DELIVER, INSPECT),),
    continues=frozenset(),
    # `when_seen` is dropped after its time-shortcut repair (a memoryless learner of the lag behind the
    # window's end scores +0.30 over the floor; DECISIONS, "`when_seen` dropped from the helper and the drone"), and
    # `distance_between` leaves every job (no controller asks how far apart two arbitrary things are). `object_where`
    # asks what stands within `NEAR_M` of the drop point (`_relevance`).
    # 0.6.0: `object_was` and `same_place` join. No `task_things` (DECISIONS, "0.6.0's Phase A review calls", 8): the
    # map's floor-surface centres near the drop point score +0.25 to +0.40 against its gold, because vehicles stand in
    # parking lanes and loading zones, and a vehicle made don't-care leaves 0.25 items a task
    # 0.6.0 (DECISIONS, "Before the binding read", 1): `object_was` and `same_place` leave again, below §9's drop line
    # (0.33 a medium case here)
    # 0.8.0 (review §4.3, §4.4): the drone asks no end question. `object_where` retires from it (a delivery drone
    # re-finds nothing on the ground; walls are what it can hit) and its end ask funds the corridor's look asks.
    ends=(),
    zones={BEFORE_LOWER: ZONE},
    subgoals=subgoals,
    ladder=LADDER,
    # 0.8.0 (review §4.4, §9): `clear_path` replaces `ego_now` (nothing it names can be hit from 15 m; the corridor
    # takes its slot). The corridor is asked in every phase of the status (the floor below decides), and an `unknown`
    # corridor gold is kept at 75% (so about a fifth of the rows).
    looks=("clear_path", "ego_mem"),
    # 0.10.0 (FD-questions-0.10 §3; lane J): two slots a point, harm first. The lanes measured, per drawn slot,
    # `clear_path` 0.585 (after the unknown keep) and `ego_mem` 0.153. Slot 1 is the corridor, the drone's harm question
    # with supply, at 60: it has the fill to give a question at more than half the points it is drawn at, so a whole
    # slot would make four of five questions the same corridor, and 60 lands the drone over the 1 in 15 floor. Slot 2
    # is `ego_mem` at 70, what it saw before, whose fill is capped by the distinct things a flight holds. No score
    # enters it.
    # Final review S2 (DECISIONS 2026-10-14, "The final review's repair pass"): `drop_zone_people`, which held a third
    # slot whole, leaves the drone's looks. It drew 2 rows over the 36 dev cases (413 of its 593 draws fell outside its
    # three phases and 174 more found nobody within 4 m, of which 1% are kept), under `abilities.MIN_ABILITY_N`, so no
    # gate reads it, and the 2 rows sat inside the drone's counted clearance `now` cell with no per-family diagnostic.
    # Its code and page stay for a dataset with more supply (`look_phases` goes with it: a job names no phase for a
    # family it does not ask). Nothing else is rebalanced.
    look_shares=((("clear_path", 60),), (("ego_mem", 70),)),
    # 0.10.0 (FD-questions-0.10 §3): the frames of the lowering and of the descent come first in the look draw (the
    # decisions the delivery turns on).
    look_decisions=(BEFORE_LOWER, BEFORE_DESCENT),
    look_unknown_keep={"clear_path": 75},
    clear_phases=frozenset(ROBOTS["drone"].status.phases),
    clear_directions=("ahead", "back"),  # a flier's two (0.8.0's draw; sideways is a constant, +0.15, brief A2)
    # 0.8.0 (review §2.4, §9; final review S2): the corridor is asked only with the base at least 8.35 m up: the tallest
    # thing fixed in the place that is not a wall, plus the body's half height (0.15 m) plus 0.25 m. That is the street
    # lamp's head, 7.95 m (render/scenery_layout.py `POST_HEIGHT_M["lamp"]`, drawn by render/scenery.py `_lamp`: pole to
    # 7.5 m, an arm to 7.9 m reaching 1.7 m over the street, a head at 7.78-7.95 m): `scenery_layout._strip_posts`
    # stands one every 28 m in the kerb strip of every sidewalk of the map. They are scenery, not in the map, so the
    # gold's walls-only corridor ignores them and the floor must clear them. The street-tree crowns (6.90 m: render/
    # dressing_layout_downtown.py, trunk to 3.40 m above the 0.15 m sidewalk, crown 3.5 m tall) and the signal poles
    # (5.2 m) are lower. At 7.01 m (descend, inspect) the body reaches down to 6.86 m, and landed on the nest (0.27 m) a
    # straight flight meets benches and people. The gold code never imports `render/`: this is its declared sizes, read
    # once.
    fly_floor_m=8.35,
    # 0.7.0 (FD-0.7.0-difficulty §2.3, FD-0.7.0-review "values to declare"): the clearance C is 2.4 m, the smaller of
    # the distances Wing's patent US12399508B2 keeps a delivery point from obstacles (at least 10 ft from a building,
    # 8 ft from trees; things over about 3 ft tall count as obstacles); Amazon and Flytrex advise clear areas of the
    # same order. Zipline's two feet is how precisely it places the package, not a clearance. R is 2.0 m, the match
    # radius: Wing's US12221211B2 prefers never to deliver more than about 2 m from the intended point. `reacquire` is
    # not set.
    clearance=2.4,
)
