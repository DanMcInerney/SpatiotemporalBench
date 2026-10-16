"""The tote AMR's job, `amr/1`: scripts of one to six tasks on a warehouse in which a MiR250-class tote robot leaves its
charger, docks at the tote shelves and the packing tables (or runs an order-picking loop into a rack aisle and on to a
packing table), reverses out of every station it docks at, and returns to its charger, with unobserved gaps between the
tasks through which the warehouse's own processes change it (docs/job-amr.md).

The declarative part is here (`SPEC`); `job_amr_plan` holds the constants, the map contract and the plan, `job_amr_path`
a task's primitives, `job_amr_drops` the balance from the world (the bay coin and the rear drop) and `job_amr_checks`
the job's own `validate` rules. A task is planned on `route.robot_free_space` with every staging lane a keep-out, loaded
or not (a lane's load must not change a path: with the lanes' pallets the only obstacles, A* would cut across empty
lanes and round loaded ones, and the robot's own path log would read each lane's load, the answer a `slots_empty`
question would ask for; the job asks none, but a path that told the lanes would still leak).

Before each task: `between(lo, hi, keep_open)` lets the world change in the gap (the warehouse has no doors, so
`keep_open` is the regions the task uses); then, at `t_start - changes.EDGE_BUFFER`, every bay the task visits is
cleared and the free space is planned; the bays the coins block get a drum each across the docking line (the free space
is already computed, and the drum stands inside a keep-out anyway, so the path to a `before_station` never depends on
it); the route is built, a visit whose bay holds a drum being skipped after its `before_station` (that decision's
`response` is "skip", else "go"); and some docked visits get a pallet or a drum in their reverse corridor
(`job_amr_drops`), so the path never depends on those."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import job_amr_drops as drops
from .changes import EDGE_BUFFER
from .events import EventRule
from .geometry import q
from .job_amr_checks import checks as _checks
from .job_amr_path import dock_of, keep_outs, stops, task_prims
from .job_amr_plan import (
    BEFORE_CROSSING,
    BEFORE_STATION,
    BEFORE_UNDOCK,
    COLLECT,
    DRAIN_PCT_PER_MIN,
    JOB_ID,
    PICK,
    REPLENISH,
    STATION_S,
    bay_of,
    charger,
    check_map,
    gaps,
    plan_tasks,
    targets_of,
)
from .job_amr_subgoal import subgoals
from .job_registry import JobSpec, Relevance
from .keyframes import frames
from .mapspec import MapSpec
from .robots import ROBOTS, camera_pose
from .route import robot_free_space
from .seeding import make_rng
from .task_predicate import TaskThings
from .timeline_ego import DecisionPoint, Ego, Keyframe, Task, TaskPlan
from .zones import ZoneRule

if TYPE_CHECKING:
    from .changes import World
    from .world_processes import Between

RELEVANT_KINDS = frozenset({"tote", "bin", "pallet", "drum", "sack"})  # not `box`: the racks' stock cartons read as one
PHASES = frozenset({"transit", "approach", "at_station"})
LADDER = (0.5, 1.0, 2.0, 4.0)  # the warehouse's sight lines are 5 to 10 m, so the unit is 0.5 m (impl §7)
SPEED_PHASES = ("transit", "approach", "undocking")  # speeds are drawn once a task, in this order


def _plan(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    return plan_tasks(m, seed, n_tasks)


def _relevance(plan: TaskPlan) -> Relevance:
    return Relevance(RELEVANT_KINDS, frozenset(map(str, plan.params["regions"])), PHASES, floor=True)


def _keep_open(m: MapSpec, plan: TaskPlan) -> frozenset[str]:
    """The region ids the task uses: the charger's, each station's and a pick's aisle (the callback's `keep_open`)."""
    by_id = m.furniture_by_id()
    rooms = {m.region_of(charger(m).center), *(m.region_of(by_id[s].center) for s in plan.params["stations"])}
    aisle = [str(plan.params["aisle"])] if plan.kind == PICK else []
    return frozenset({*(r for r in rooms if r is not None), *aisle})


def _script(m: MapSpec, w: World, plans: list[TaskPlan], seed: int, between: Between | None = None) -> Ego:
    """The script of `plans` over the world `w`. Before each task `between(lo, hi, keep_open)` lets the world change in
    the gap (`lo` None before the first task); the gaps are drawn first (`gaps`: the seed's alone), so the world cannot
    move a task's time. A visit that does not fit in a world raises ValueError naming the seed, the task and the piece:
    the world moves where a visit falls, never whether it happens."""
    if not plans:
        raise ValueError(f"{JOB_ID} writes at least one task")
    check_map(m)
    robot, pause = ROBOTS["amr"], gaps(m.name, seed, len(plans))
    pitch, by_id = q(math.radians(robot.camera.pitch_deg)), m.furniture_by_id()
    low, high = robot.status.start["battery_pct"]
    keep_out = keep_outs(m, robot)
    kfs: list[Keyframe] = []
    tasks: list[Task] = []
    decisions: list[DecisionPoint] = []
    t_end = 0.0
    for i, plan in enumerate(plans):
        start = 0.0 if i == 0 else t_end + pause[i - 1]
        if between is not None:
            between(None if i == 0 else t_end, start, _keep_open(m, plan))
        edge = 0.0 if i == 0 else float(math.floor(start - EDGE_BUFFER))  # a whole second, after the gap's changes
        bays = [bay_of(m, by_id[s]) for s in plan.params["stations"]]
        drops.clear_bays(w, m, make_rng(m.name, seed, JOB_ID, "clear", i), edge, bays)
        free = robot_free_space(m, w, robot, start, keep_out=keep_out)
        docks, taken = [dock_of(m, by_id[s], robot) for s in plan.params["stations"]], set()
        coins = [(j, by_id[d.bay], d.docked) for j, d in enumerate(docks) if drops.bay_blocked(m, seed, i, j)]
        skips = {j: drops.skip_wait_s(m, seed, i, j) for j in drops.block_bays(w, m, seed, i, edge, coins, taken)}
        rng = make_rng(m.name, seed, JOB_ID, "job", i)
        battery = int(rng.integers(low, high + 1))
        speed = {phase: float(rng.uniform(*robot.speeds_mps[phase])) for phase in SPEED_PHASES}
        holds = [float(rng.uniform(*STATION_S)) for _ in range(stops(plan))]
        try:
            route = task_prims(m, free, robot, plan, speed, holds, skips)
        except ValueError as e:
            raise ValueError(f"{JOB_ID} on {m.name}, seed {seed}, task {i} ({plan.kind}): {e}") from e
        steps = frames(route.prims, robot.frames, t0=start)
        k0, loaded, done, asked = len(kfs), False, 0, 0
        for j, f in enumerate(steps):
            base = (f.xy[0], f.xy[1], 0.0)
            pos, rot = camera_pose(robot, base, f.yaw, pitch)
            drained = int(DRAIN_PCT_PER_MIN * (f.t - steps[0].t) / 60)
            kfs.append(Keyframe(k0 + j, f.t, pos, rot, base, f.yaw, pitch, i, f.phase,
                                {"battery_pct": battery - drained, "loaded": loaded}))
            if f.decision:
                response = None
                if f.decision == BEFORE_STATION:  # the response follows the bay's state, and comes after the ask
                    response, asked = ("skip" if asked in skips else "go"), asked + 1
                decisions.append(DecisionPoint(k0 + j, f.decision, response))
            if f.phase == "at_station" and (j + 1 == len(steps) or steps[j + 1].phase != "at_station"):
                loaded, done = route.loaded_after[done], done + 1  # a hold's last keyframe still has the old value
        tasks.append(Task(i, k0, len(kfs), plan, targets_of(plan)))
        t_end = steps[-1].t
        docked = [None if j in skips else d for j, d in enumerate(route.docks)]
        drops.rear_drops(w, m, seed, i, edge, docked, drops.swept_bodies(kfs[k0:], robot), robot, taken)
    return Ego(kfs, tasks, decisions)


SPEC = JobSpec(
    id=JOB_ID,
    robot="amr",
    domain="warehouse",
    task_kinds=(REPLENISH, COLLECT, PICK),
    # no `path_ahead` (the station's face is 0.12 m ahead of every undock) and no `slots_empty` (the lanes hold few
    # things, so a constant list beats `all_unknown` by more than a shortcut may), so the lanes are no zone rule of the
    # job; no `rear_clear` (the rear drop is clear by construction and no decision rides on it; docs/job-amr.md)
    # The release gate check on seeds 2001-2006 (DECISIONS, "The AMR's release gate check") dropped `zone_clear`
    # (`blind` +0.67: its memory rows were clear bays) and `rear_clear` (memory_needed +0.083 on the in-pool twin, n 12)
    # 0.5.0 (DECISIONS, "0.5.0 declared"): `where_was_i` is dropped (a lookup of the frame header's own pose) and
    # `out_of_view` stays; the relevance has `floor` set (a tote robot works the floor, so a critical thing stood on it)
    # 0.5.0-control: `zone_clear` returns at `before_station` as a now question (its gold reads the decision's look
    # window alone, so it has no memory rows), the zone being the bay, which a coin leaves blocked or clear and the
    # robot answers by skipping a blocked visit (`job_amr_drops`); `out_of_view` stays at the memory decisions
    # 0.6.0 (DECISIONS, "0.6.0 declared before lanes", call 1): `zone_clear` and `out_of_view` are retired, so no
    # decision kind asks a family; the coin, the skip and the bay stay (the bay's blocker is told: `World.told`), and
    # the questions come at look points (`looks`) and at the end
    decisions={BEFORE_STATION: (), BEFORE_UNDOCK: (), BEFORE_CROSSING: ()},
    # A skipped visit holds at T in `waiting`, so the closing hold is the task's last `waiting` step, not the first one
    # after a `transit`; a "stop" is a place the robot worked at (a skipped station is not one)
    events={
        "when you left the charger": EventRule("first", "transit"),
        "when you reached your first stop": EventRule("first", "at_station"),
        "when you finished at your last stop": EventRule("last", "at_station"),
        "when you got back to the charger": EventRule("last", "waiting"),
    },
    plan=_plan,
    script=_script,
    relevance=_relevance,
    checks=_checks,
    anchors=frozenset({"charger"}),
    assignments={
        REPLENISH: "Its current assignment is to bring totes from the tote shelves to the packing tables.",
        COLLECT: "Its current assignment is to collect the empty totes from the packing tables and return them to the "
                 "tote shelves.",
        PICK: "Its current assignment is an order-picking run: collect the picks from the racks, then bring them to a "
              "packing table.",
    },
    swaps=((REPLENISH, COLLECT),),
    continues=frozenset(),
    # The same check took `changed` (2 questions in 18 cases; its oracle 0.60) and `order` (4; `other_case` +0.25) out
    # of `ends`. The release's binding gate check, over the short and medium cases that ship, took `when_seen` out
    # (DECISIONS, "when_seen is dropped from the AMR and the vacuum"): `questionnaire_blind` BROKEN at 0.103 under
    # abstaining
    # 0.5.0 (DECISIONS, "0.5.0's code review"): `same_place` is dropped (its only job: `latest_table` answers yes to
    # every question, so it scores 2y - 1 on the yes share y, and the compile's 9 yes to 5 no was 0.286 over the margin)
    # 0.5.0-control (FD-0.5.0-control-impl §6): `distance_between` leaves every job; `object_where` asks "where did you
    # last see the X" of the critical, unmoved things; `object_was` keeps the moved ones, with their anchors
    # 0.6.0: `adjacent` is retired; `same_place` and `task_things` join. The task's predicate drops `floor`
    # (DECISIONS, "0.6.0's Phase A review calls", 7): "within 2 m of the way you drove to your stations" is 1.06 items
    # a task and 35% empty, and the stock the racks hold off the floor stays don't-care
    # 0.6.0 (DECISIONS, "Before the binding read", 1): `object_was` and `same_place` leave, below §9's drop line (0.42 a
    # medium case here)
    # 0.8.0 (review §4.3, §4.4; DECISIONS, "0.8.0 declared"): `closest_person` takes `object_where`'s end slot (a tote
    # robot's safety question is how close it came to a person; `object_where` asks a loose re-find radius and was
    # +0.80 to +1.00 for sol), one a case, so the end holds the budget (2.88 a medium case in 0.7.0)
    ends=("task_things", "closest_person"),
    zones={BEFORE_STATION: ZoneRule("the bay ahead of you", "bay")},
    subgoals=subgoals,
    ladder=LADDER,
    looks=("closest_approach", "ego_mem", "ego_now", "approach_mem"),
    # 0.10.0 (FD-questions-0.10 §3; lane J): three slots a point, harm first. The AMR asks no `clear_path` (review §2.5:
    # in transit with the stopper screen 62% of its golds are the 5 m cap, a held-out constant scores +0.33 and the map
    # reader +0.85), so it declares no `clear_phases`; its harm question is the person nearest it, `closest_approach`
    # (the end keeps `closest_person`). The lanes measured, per drawn slot, `closest_approach` 0.108 (every frame
    # tried), `ego_mem` 0.110 and `ego_now` 0.112 (after the floor filter), 0.33 questions a point even if all three
    # held, 1 in 18 frames. The compile at whole slots gave 0.027, 0.026 and 0.101: 883 of the 1,043 draws of
    # `closest_approach` find no one in range (and its "no one" is kept at 2%), 56 more are lost to the subject
    # `closest_person` reserves for its task, and 401 of `ego_mem`'s find only things an earlier look named (the floor
    # holds few). The supply, not the share, is the limit (1 in 37 frames on short and medium cases), and a share under
    # 100 only loses questions, so every slot is whole. Harm first is then the order alone: the person (slot 1), what
    # was seen before (slot 2), what is in view (slot 3). No score enters it.
    # 0.11.0 (FD-0.11-hard §3.2): slot 3 gives half to `approach_mem`, "how close would you come to the person you saw
    # last, where you last saw them": `closest_approach` for the person now out of view, the AMR's blind aisle end
    # (a picker seen 20 s ago, hidden by the rack now), which a planner keeps as an obstacle. `closest_approach` finds
    # someone in view at about 10% of the frames; a person within 10 m who was seen recently and is out of view is
    # 41% of them. The half comes from `ego_now`, the most-asked during family of the warehouse (astra +0.71, sol
    # +0.59), whose mean a halving does not move (105 rows to about 52); `approach_mem` is a harm row and needs memory.
    # Slots 1 and 2 stay whole: the supply, not the share, is their limit. Declared before the compile, to be read
    # there (rows, the "no one" share, the gates), and dropped from the job by the evidence rule if a gate fails.
    look_shares=((("closest_approach", 100),), (("ego_mem", 100),), (("ego_now", 50), ("approach_mem", 50))),
    # 0.10.0 (FD-questions-0.10 §3): the frames of the crossing and of the station come first in the look draw (the
    # decisions the drive turns on; `before_undock` has no supply there).
    look_decisions=(BEFORE_CROSSING, BEFORE_STATION),
    # 0.7.0 (FD-0.7.0-difficulty §2.3, FD-0.7.0-review "The values to declare"): the clearance C is 0.45 m, the room the
    # MiR250 asks beside its path (580 mm wide, 1,450 mm corridor by default: 0.435 m a side; its side protective
    # fields are 210 mm plus a 65 mm tolerance). ISO 3691-4 Annex A.2.1 (0.5 m beside the path) is a layout rule and
    # corroborates the order of size only; docking is fiducial-guided at millimetres and is not C. R is the matching
    # radius, 1.0 m, a convention: no warehouse source gives one, so `reacquire` is not set.
    clearance=0.45,
    # 0.6.0 lane R3: the path is the whole drive (every phase, no `until`): a final stop that is blocked and skipped is
    # still driven to (its subgoal says so, and 13 of the 16 false positives on the cut leg were real drums and
    # pallets). The keep is set again for the new gold: 22 of the 41 tasks of release-c's short and medium AMR cases are
    # empty after `told` and `RELAX` (p 0.537), so 100 (1 - p) / (2 p) = 43
    task_things=TaskThings("within 2 m of the way you drove",
                           "path", frozenset({"transit", "approach", "at_station", "undocking", "waiting"}),
                           reach_m=2.0, band_m=0.3, empty_keep_percent=43),
)
