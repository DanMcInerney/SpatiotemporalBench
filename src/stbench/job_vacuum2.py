"""The vacuum's full job, `vacuum/2`: scripts of several tasks, each cleaning one or two rooms, some sent home before
they finish and resumed later, with unobserved gaps between them.

`vacuum/1` stays frozen beside it; this module reuses its helpers by import. The declarative part is `vacuum/1`'s (the
robot, the dock, the four events, the relevance) with two task kinds and five decision kinds. `plan_tasks` and `gaps`
(`job_vacuum2_plan`) draw from the map and the seed alone; `_script` writes each task's path over the world as the gap
before it left it, `_checks` (`job_vacuum2_checks`) holds the job's own `validate` rules and `subgoals`
(`job_vacuum2_subgoal`) the text of each keyframe.

A task is planned on `route.robot_free_space` with every interior door off its ways shut in both leaf positions
(`world_processes.doors_on_ways`: the gap callback keeps the doors on the ways open), so no door's state can change a
path, and the robot's own path log learns none. The path of a task is: docked, undocking, transit to the first room;
for each room its edge pass and strips (a resumed room: only the strips that cross floor the stopped task's brush left
unswept), with a `room_done` decision on the last cleaning frame, or a `cleaning_stopped` where a stopping task cuts its
last room's cleaning once the brush has swept its share of the room's floor; transit to the next room, or back to the
dock and docking.

**The doorway coin** (`job_vacuum2_doorway`). A decision stands before the doorway into each room but the dock's. In an
eligible room (`job_vacuum2_plan.eligible`) it is a `before_room`: a coin, drawn from the seed alone, says the doorway
ahead is blocked or clear, and `zone_clear` is asked there. The world is made to say it (before the task is planned
every doorway is cleared; when the coin says blocked, things are moved into the way once the task is planned up to the
decision) and the robot then responds: "go" through a clear doorway, and for a blocked one "skip": the way in is cut at
the decision, the room is left uncleaned and the path goes on, planned with the blockers in the world, to the next
room's doorway or home, the brush off. Any other room is entered at a `before_doorway`, which asks nothing, and so is
the doorway out of the last room when it was cleaned. A robot that stands in the next room already, having crossed it to
reach a skipped one, has no doorway ahead and no decision.
"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from . import coverage, grid, hazard_rule
from . import job_vacuum2_backout as backout
from . import job_vacuum2_doorway as doorway
from .changes import EDGE_BUFFER
from .events import EventRule
from .geometry import Point, distance, q
from .job_registry import JobSpec, Relevance
from .job_vacuum import (
    DOCK_KIND,
    DOCK_REVERSE_M,
    DRAIN_PCT_PER_MIN,
    FLOOR_KINDS,
    PHASES,
    SPEED_PHASES,
    STRIP_M,
    UNDOCK_M,
    _entry,
    _region_of_kind,
    _way,
    docked_pose,
)
from .job_vacuum2_checks import checks as _checks
from .job_vacuum2_plan import (  # noqa: F401  (the constants are this module's public names)
    BEFORE_BACKOUT,
    BEFORE_DOORWAY,
    BEFORE_ROOM,
    CLEAN,
    CLEANING_STOPPED,
    GAP_S,
    GO,
    JOB_ID,
    RESUME,
    ROOM_DONE,
    SKIP,
    SPAN_S,
    STOP_P,
    STOP_SHARE,
    TASK_MAX_S,
    TWO_ROOMS_P,
    ZONE,
    dock_room_id,
    eligible,
    gaps,
    plan_tasks,
)
from .job_vacuum2_subgoal import subgoals
from .job_vacuum_path import Driver
from .keyframes import Hold, Mark, Move, frames
from .mapspec import MapSpec, Region
from .robots import ROBOTS, Robot, Tool, camera_pose
from .route import FreeSpace, robot_free_space
from .seeding import make_rng
from .task_predicate import TaskThings
from .timeline_ego import DecisionPoint, Ego, Keyframe, Task, TaskPlan
from .world_processes import doors_on_ways

if TYPE_CHECKING:
    from .changes import World
    from .world_processes import Between


def _plan(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    return plan_tasks(m, seed, n_tasks)


# -- the path ----------------------------------------------------------------------------------------------------
def _brush(robot: Robot) -> Tool:
    return next(t for t in robot.tools if t.name == "brush")


def _tool_pose(tool: Tool, xy: Point, yaw: float) -> tuple[float, float, float]:
    """Where the tool stands when the base is at `xy` heading `yaw`: its offset turned by the heading."""
    (ox, oy), c, s = tool.offset_m, math.cos(yaw), math.sin(yaw)
    return (xy[0] + c * ox - s * oy, xy[1] + s * ox + c * oy, yaw)


def fine_cells(free: FreeSpace, mask: np.ndarray) -> set[tuple[int, int]]:
    """The fine cells (`grid`) that hold the centre of a cell of `mask`, a mask over `free`'s grid."""
    xs, ys = free.cells_xy(mask)
    return {grid.fine_cell(float(x), float(y)) for x, y in zip(xs, ys, strict=True)}


def cut(poly: list[Point], floor: set[tuple[int, int]], share: float, brush: Tool) -> list[Point]:
    """The polyline up to where the brush has swept `share` of `floor` (the room's reachable fine cells): the place a
    stopped task had come to when the resident sent the robot home, a point on a segment (found by bisection). The
    polyline whole when it never sweeps that much."""
    want, swept = share * len(floor), set()

    def reach(a: Point, b: Point) -> set[tuple[int, int]]:
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        poses = [_tool_pose(brush, a, yaw), _tool_pose(brush, b, yaw)]
        return grid.swath(poses, brush.shape, brush.size_m[0], brush.size_m[1]) & floor

    out = [poly[0]]
    for a, b in zip(poly, poly[1:], strict=False):
        if len(swept | reach(a, b)) >= want:
            lo, hi = 0.0, 1.0
            for _ in range(16):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if len(swept | reach(a, _lerp(a, b, mid))) >= want else (mid, hi)
            end = _lerp(a, b, hi)
            return out + ([end] if distance(end, out[-1]) > 1e-9 else [])
        swept |= reach(a, b)
        out.append(b)
    return out


def _lerp(a: Point, b: Point, f: float) -> Point:
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def brush_cells(robot: Robot, kfs: list[Keyframe]) -> set[tuple[int, int]]:
    """The fine cells (`grid`) the brush swept in the last run of consecutive brush-on keyframes of `kfs`: the
    stopped task's last room, which a resume finishes. The brush is the robot's tool of that name, at its offset."""
    tool = _brush(robot)
    runs: list[list[Keyframe]] = []
    for i, kf in enumerate(kfs):
        if not kf.status["brush"]:
            continue
        if i == 0 or not kfs[i - 1].status["brush"]:
            runs.append([])
        runs[-1].append(kf)
    poses = [_tool_pose(tool, kf.base[:2], kf.yaw) for kf in (runs[-1] if runs else [])]
    return grid.swath(poses, tool.shape, tool.size_m[0], tool.size_m[1]) if poses else set()


def unswept(free: FreeSpace, floor: np.ndarray, swept: set[tuple[int, int]], reach: float = 0.0) -> np.ndarray:
    """`floor` (a mask over `free`'s grid) less the cells whose fine cell is in `swept`, grown by `reach` metres (a
    square round each cell): what a strip within `reach` of it still has to sweep."""
    u = np.floor((free.x0 + (np.arange(free.nx) + 0.5) * free.res) / grid.FINE_M + grid.EPS).astype(np.int64)
    v = np.floor((free.y0 + (np.arange(free.ny) + 0.5) * free.res) / grid.FINE_M + grid.EPS).astype(np.int64)
    span = 1 << 20
    code = (u[None, :] + span) * (2 * span) + (v[:, None] + span)
    done = np.array([(a + span) * (2 * span) + (b + span) for a, b in swept], dtype=np.int64)
    left = floor & ~np.isin(code, done)
    n = round(reach / free.res)
    pad, out = np.pad(left, n), left.copy()
    for dy in range(2 * n + 1):
        for dx in range(2 * n + 1):
            out |= pad[dy:dy + left.shape[0], dx:dx + left.shape[1]]
    return out


def _skip(drive: Driver, n0: int, way_in: list[Point], mark: tuple[int, float, str], coins: doorway.Coins, j: int,
          room: Region, lead: tuple[Point, Point]) -> tuple[Point, FreeSpace]:
    """The robot's response to a blocked doorway: the way in (primitives from `n0` on) is cut at the decision's `Mark`,
    where the base stands, the blockers are moved in (clear of the path so far: the undocking run `lead` and the
    primitives), and that place is returned with the planner's free space with them. ValueError when the doorway the
    base faces is not one of the room's."""
    i, s, _ = mark
    a, b = way_in[i], way_in[i + 1]
    f = s / distance(a, b)
    at = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
    cut = next(k for k in range(n0, len(drive.prims))
               if isinstance(drive.prims[k], Mark) and drive.prims[k].decision == BEFORE_ROOM)
    del drive.prims[cut + 1:]
    drive.yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    return at, coins.close(j, room.id, at, [lead, *doorway.trail_of(drive.prims)])


def _task_prims(m: MapSpec, free: FreeSpace, robot: Robot, plan: TaskPlan, rooms: list[Region],
                speed: dict[str, float], swept: set[tuple[int, int]], depths: list[float],
                coins: doorway.Coins) -> tuple[list, list[str]]:
    """The primitives of one task, and the response ("go" or "skip") of each of its `before_room` decisions in order:
    docked, undocking, then for each room transit in (a decision unless it is the dock's room or the robot stands in it
    already: a `before_room` where `eligible`, whose coin says clear ("go") or blocked, else a `before_doorway`; in a
    site room of a `clean_rooms` task, a back-out with its before_backout decision, drive-in `depths[j]`) and
    cleaning (edge pass and strips, a room_done decision on the last frame; a resume has no edge pass and only the
    strips over what the stopped task's brush left; a stopping task is cut at its share and marks cleaning_stopped); a
    blocked doorway ends the way in at its decision ("skip": no cleaning, the rest planned on the free space with the
    blockers), then transit back (a before_doorway decision when the last room was cleaned and is not the dock's),
    docking, docked."""
    x, y, yaw = docked_pose(m)
    home, out = (x, y), (x + UNDOCK_M * math.cos(yaw), y + UNDOCK_M * math.sin(yaw))
    front = (x + DOCK_REVERSE_M * math.cos(yaw), y + DOCK_REVERSE_M * math.sin(yaw))
    dock_room = dock_room_id(m)
    stop = plan.params.get("stop")
    drive = Driver(yaw, robot.turn_dps, exact=True)
    at, answers, cleaned = out, [], False
    for j, room in enumerate(rooms):
        stops = stop is not None and j == len(rooms) - 1
        entry = _entry(m, free, room, at)
        site = None if plan.kind == RESUME else backout.site_of(m, room, robot)
        out_to = None
        if site is not None:
            out_to = backout.plan(free, site, at, depths[j])
            if out_to is None:
                raise ValueError(f"no valid approach line to back out of the {site.name} in the {room.name}")
            entry = out_to.approach
        if plan.kind == RESUME:
            reach = _brush(robot).size_m[0] / 2
            left = unswept(free, coverage.reachable_floor(free, room, entry), swept, reach)
            cleaning = coverage.strips(free, room, entry, STRIP_M, entry, left)
        else:
            loop = coverage.perimeter(free, room, entry)
            cleaning = loop + coverage.strips(free, room, entry, STRIP_M, loop[-1])[1:]
        way_in = _way(free, at, entry if out_to is not None else cleaning[0], f"to the {room.name}")
        mark = None
        if room.id != dock_room and m.region_of(at) != room.id:
            i, s = doorway.decision_mark(m, room, way_in, last=True)
            mark = (i, s, BEFORE_ROOM if eligible(m, plan, j) else BEFORE_DOORWAY)
        n0 = len(drive.prims)
        drive.go(way_in, speed["transit"], "transit", mark)
        if mark is not None and mark[2] == BEFORE_ROOM:
            blocked = coins.blocked(j)
            answers.append(SKIP if blocked else GO)
            if blocked:
                at, free = _skip(drive, n0, way_in, mark, coins, j, room, (home, out))
                cleaned = False
                continue
        if out_to is not None:
            run = distance(out_to.approach, out_to.far)
            drive.go([out_to.approach, out_to.far], speed["transit"], "transit", (0, run, BEFORE_BACKOUT))
            drive.prims.append(Move(out_to.far, out_to.approach, out_to.yaw, speed["docking"], "transit"))  # reversing
            drive.go(_way(free, out_to.approach, cleaning[0], f"to the {room.name}'s edge"), speed["transit"],
                     "transit")
        drive.begin(cleaning[0], "cleaning")
        if stops:
            floor = fine_cells(free, coverage.reachable_floor(free, room, entry))
            cleaning = cut(cleaning, floor, float(stop), _brush(robot))
        drive.go(cleaning, speed["cleaning"], "cleaning")
        drive.prims.append(Mark(CLEANING_STOPPED if stops else ROOM_DONE))
        at, cleaned = cleaning[-1], True
    way_out = _way(free, at, front, "back to the dock")
    mark = None
    if cleaned and rooms[-1].id != dock_room:
        i, s = doorway.decision_mark(m, rooms[-1], way_out, last=False)
        mark = (i, s, BEFORE_DOORWAY)
    drive.go(way_out, speed["transit"], "transit", mark)
    drive.turn(front, yaw, "docking")
    drive.prims.append(Move(front, home, yaw, speed["docking"], "docking"))  # reversing: heading kept, motion opposite
    return [Hold(home, yaw, 0.0, "docked"), Move(home, out, yaw, speed["undocking"], "undocking"), *drive.prims,
            Hold(home, yaw, 0.0, "docked")], answers


def _script(m: MapSpec, w: World, plans: list[TaskPlan], seed: int, between: Between | None = None) -> Ego:
    """The script of `plans` over the world `w`. Before each task `between(lo, hi, keep_open)` lets the world change
    in the gap (`lo` None before the first task): `keep_open` is the dock's region and the task's rooms, and the task
    is planned on the world after the call, with every interior door off the ways between them shut in both leaf
    positions. The gaps are drawn first (`gaps`: the seed's alone), so the world cannot move a task's time. Then, at
    `t_edge` (the task's start less `changes.EDGE_BUFFER`, 0 for task 0), every doorway is cleared
    (`doorway.clear_ways`), and the doorways that coins block get their blockers while the task is planned."""
    if not plans:
        raise ValueError(f"{JOB_ID} writes at least one task")
    robot = ROBOTS["vacuum"]
    dock_room, pause = dock_room_id(m), gaps(m.name, seed, len(plans))
    pitch = q(math.radians(robot.camera.pitch_deg))
    low, high = robot.status.start["battery_pct"]
    kfs: list[Keyframe] = []
    tasks: list[Task] = []
    decisions: list[DecisionPoint] = []
    t_end = 0.0
    for i, plan in enumerate(plans):
        rooms = [_region_of_kind(m, str(kind)) for kind in plan.params["kinds"]]
        keep_open = frozenset({dock_room, *(r.id for r in rooms)})
        start = 0.0 if i == 0 else t_end + pause[i - 1]
        if between is not None:
            between(None if i == 0 else t_end, start, keep_open)
        edge = 0.0 if i == 0 else float(math.floor(start - EDGE_BUFFER))  # a whole second, after the gap's changes
        doorway.clear_ways(w, m, seed, i, edge)
        on_ways = doors_on_ways(m, keep_open)
        planner = partial(robot_free_space, m, w, robot, start,
                          shut=sorted(o.id for o in m.toggling_doors() if o.id not in on_ways))
        rng = make_rng(m.name, seed, JOB_ID, "job", i)
        battery = int(rng.integers(low, high + 1))
        speed = {phase: float(rng.uniform(*robot.speeds_mps[phase])) for phase in SPEED_PHASES}
        swept = brush_cells(robot, kfs[tasks[-1].k0:]) if plan.kind == RESUME else set()
        brng = make_rng(m.name, seed, JOB_ID, "backout", i)
        depths = [float(brng.random()) for _ in rooms]
        try:
            prims, answers = _task_prims(m, planner(), robot, plan, rooms, speed, swept, depths,
                                         doorway.Coins(w, m, robot, seed, i, edge, planner))
        except ValueError as e:
            raise ValueError(f"{JOB_ID} on {m.name}, seed {seed}, task {i} ({plan.params['rooms']}): {e}") from e
        steps = frames(prims, robot.frames, t0=start)
        k0, responses = len(kfs), iter(answers)
        for j, f in enumerate(steps):
            base = (f.xy[0], f.xy[1], 0.0)
            pos, rot = camera_pose(robot, base, f.yaw, pitch)
            drained = int(DRAIN_PCT_PER_MIN * (f.t - steps[0].t) / 60)
            kfs.append(Keyframe(k0 + j, f.t, pos, rot, base, f.yaw, pitch, i, f.phase,
                                {"battery_pct": battery - drained, "brush": f.phase == "cleaning"}))
            if f.decision:
                answer = next(responses) if f.decision == BEFORE_ROOM else None
                decisions.append(DecisionPoint(k0 + j, f.decision, answer))
        tasks.append(Task(i, k0, len(kfs), plan, tuple(r.id for r in rooms)))
        t_end = steps[-1].t
    return Ego(kfs, tasks, decisions)


def _relevance(plan: TaskPlan) -> Relevance:
    return Relevance(FLOOR_KINDS, frozenset({*map(str, plan.params["kinds"]), "hall"}), PHASES, floor=True)


SPEC = JobSpec(
    id=JOB_ID,
    robot="vacuum",
    domain="home",
    task_kinds=(CLEAN, RESUME),
    # which families a job asks is the compiler's definition, not the job's path (DECISIONS, "M1's review")
    # M2's mapping (the coordinator, after the family pairs merged; DECISIONS, "M2's calls"): the back-out families
    # (`rear_clear`, `path_ahead`) are not asked, since the back-out gate failed and no plan carries a back-out
    # M2's fast gate check on look-dev seeds 141-145 (DECISIONS, "M2's fast gate check"): `out_of_view` and
    # `uncovered` are dropped (their golds empty after their declared fallbacks: 93% at 0.1 m; 100% where a task stops),
    # and `order` and `distance_between` are out of `ends` (under one question a case)
    # M2's gate check on the dev render (DECISIONS, "M2's gate check"): `changed` is dropped (it failed memory_needed,
    # shortcuts and separation: 77% of its golds are the empty set, which the current view and the form guess)
    # The release's binding gate check, over the short and medium cases that ship (DECISIONS, "when_seen is dropped from
    # the AMR and the vacuum"): `when_seen` is dropped (memory_needed and shortcuts +0.27; `blind`, `path_log` BROKEN)
    # 0.5.0 (DECISIONS, "0.5.0 declared"): `where_was_i` is dropped (a lookup of the frame header's own pose); its
    # relevance has `floor` set (it works the floor, so a critical thing stood on the floor)
    # 0.5.0-control (DECISIONS, "Camera only, a subgoal in every frame, and the controller's questions"): the vacuum
    # asks `zone_clear` (now) at `before_room`, the doorway into a room it is about to clean, which a coin has
    # blocked or left clear and a blocked one is answered by skipping the room (`job_vacuum2_doorway`); every other
    # decision, the doorway out of the last room included, asks nothing, and no `uncovered` is asked (the 8 cm
    # camera never certifies a cell free and unswept)
    # 0.6.0: `zone_clear` is retired, so no decision kind asks a family; the doorway coin and the skip stay (the
    # blocker is told: `World.told`), and the questions come at look points (`looks`) and at the end
    decisions={BEFORE_ROOM: (), BEFORE_DOORWAY: (), ROOM_DONE: (), CLEANING_STOPPED: (), BEFORE_BACKOUT: ()},
    events={
        "when you left the dock": EventRule("last", "undocking"),
        "when you started cleaning": EventRule("first", "cleaning"),
        "when you finished cleaning": EventRule("last", "cleaning"),  # the `room_done` or `cleaning_stopped` frame
        "when you got back to the dock": EventRule("first", "docked", after_phase="docking"),
    },
    plan=_plan,
    script=_script,
    relevance=_relevance,
    checks=_checks,
    anchors=frozenset({DOCK_KIND}),
    assignments={CLEAN: "Its current assignment is to clean the {rooms}.",
                 RESUME: "Its current assignment is to finish cleaning the {rooms}."},
    swaps=(),
    continues=frozenset({RESUME}),
    # 0.6.0: `adjacent` is retired; `object_was`, `same_place` and `task_things` join
    # 0.6.0 (DECISIONS, "Before the binding read", 1): `object_was` and `same_place` leave again (0.10 a medium case
    # here)
    # 0.9.0 (the final review, B1; DECISIONS, "the evidence verdict ships", 2): `task_things` was dropped from the
    # vacuum. It failed the binding evidence verdict: `path_log_in_pool` scored +0.258 over `all_unknown` on the long
    # cases (+0.135 on the medium), a layout prior of the home's recurring item spots that no sensor reads (90% of its
    # gold items had another script's item within 0.5 m, and "the floor of the rooms you cleaned" reached every such
    # spot).
    # 0.11.0 (FD-0.11-design §3.4.3): it returns as the list of what the brush could catch, which the page of 0.9.0
    # anticipated ("a list limited to things off the layout's recurring spots"): the items are the cords and cloth of
    # `hazard_rule.CATCH_KINDS`, and the kinds that recur at one spot from script to script (`SPOT_KINDS`: bath mat,
    # pedestal mat, clothes pile) are don't-care, as `tangle_ahead`'s are. The gates decide: if `path_log_in_pool` or
    # the balance rule fails again, the vacuum drops it again by rule (the design's drop rule, never a threshold moved)
    ends=("object_where", "task_things"),
    zones={BEFORE_ROOM: ZONE},
    subgoals=subgoals,
    ladder=(0.25, 0.5, 1.0, 2.0),  # a home: rooms of 3-5 m, things seen at 1-3 m (FD-0.5.0-control-impl §7)
    # 0.9.0 (FD-decision-questions §6, impl §0): `clear_path` asks ahead, back and left (`clear_directions`; right is a
    # constant, +0.28, brief A2). The corridor and the heading are asked while the vacuum cleans (`cleaning`); docked,
    # its status answers them (review §2.5: +1.00), and in `transit` it leaves its dock on a fixed route, the same poses
    # and golds in different scripts, so a held-out constant per phase scored +0.103 over `all_unknown` on those rows,
    # past the 0.10 form margin (final review S3).
    looks=("clear_path", "tangle_ahead", "ego_mem", "open_heading"),
    # 0.10.0 (FD-questions-0.10 §3; lane J): three slots a point, harm first. Per drawn slot the lanes measured
    # `clear_path` 0.538, `tangle_ahead` 0.247, `ego_mem` 0.143 and `open_heading` 0.514; the first compile at slots
    # 50 / 50, 50 and 10 gave 0.434, 0.171, 0.092 and 0.447 (the stretches of nearby points and the things a case holds
    # take draws that the every-frame measure did not), 1 in 16.1 frames on short and medium cases, under the 1 in 15
    # floor. Slot 1 is the two harm questions, whole (100) and split evenly: each has the rows at half a slot (the
    # compile gave 133 cable rows and 315 corridor rows), so neither is cut to feed the other. Slot 2 is `ego_mem`, what
    # was seen before, whole: its fill is capped by the distinct things a case holds, so a whole slot costs nothing
    # that another family could use. Slot 3 is the rest, `open_heading` (a diagnostic), at 20, the share that brings the
    # job over the floor: the second compile gave 0.429, 0.169, 0.055 and 0.457 and 1 in 14.0 frames. No score enters
    # it.
    look_shares=((("clear_path", 50), ("tangle_ahead", 50)), (("ego_mem", 100),), (("open_heading", 20),)),
    # `tangle_ahead` is asked while the brush runs or the vacuum drives to its next strip
    look_phases={"tangle_ahead": frozenset({"cleaning", "transit"})},
    # 0.10.0 (FD-questions-0.10 §3): the frames of the doorway and of the room done come first in the look draw (where
    # the run changes room; `before_backout` never occurs: it has no site kinds).
    look_decisions=(BEFORE_DOORWAY, ROOM_DONE),
    clear_phases=frozenset({"cleaning"}),
    clear_directions=("ahead", "back", "left"),
    # 0.7.0 (FD-0.7.0-difficulty §2.3, FD-0.7.0-review "values to declare"): the clearance C is 0.10 m, the loose end
    # of what camera vacuums aim to pass beside an obstacle (Roborock says the S7 MaxV gets close to shoes; a secondary
    # guide calls a 15 cm berth wasteful and a pass within about 3 cm of a cable better). The smallest objects they
    # detect (15 mm, 5 x 3 cm) are no clearance and are not the source.
    # R is 1.0 m, the radius within which the robot finds a thing again with it in view: HomeRobot's real-world find
    # rule, and Habitat's ObjectNav success radius (R/2 = 0.5 m is the matching radius of `task_things`).
    clearance=0.10,
    reacquire=1.0,
    # The keep is the declared rule, 100 (1 - p) / (2 p), from the first compile's empty share p: 28 of the 39 tasks of
    # the 32 short and medium vacuum cases asked are empty after `told` and `RELAX` (p 0.718, 11 with items), so 20
    # 0.11.0 (final review S4): the phrase names the kinds (`hazard_rule.catch_words`), as `tangle_ahead` does
    task_things=TaskThings("on the floor of the rooms you cleaned that your brush could catch "
                           + hazard_rule.catch_words(), "rooms", frozenset({"cleaning"}), floor=True,
                           item_kinds=frozenset(hazard_rule.CATCH_KINDS),
                           dont_care_kinds=frozenset(hazard_rule.SPOT_KINDS), empty_keep_percent=20),
)
