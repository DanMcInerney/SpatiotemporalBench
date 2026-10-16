"""`vacuum/2`'s plan, gaps, script and checks, over `apartment_a` with the real clutter: scripts of 4-6 tasks of seeds
101-104, built once for the module (`job_kit.generated2`, one xdist group). Coordinates are the map's
(`maps/apartment_a.json`): the dock at (4.71, 5.05), 0.12 m deep, facing east, the vacuum a 0.17 m disc keeping 0.02 m,
so its docked pose is (4.94, 5.05) heading 0."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pytest

import job_kit
from stbench import brief, case, job_vacuum2, job_vacuum2_doorway, job_vacuum_path, world_processes
from stbench import timeline as T
from stbench.episode import pose_rows
from stbench.events import resolve
from stbench.generate import generate_script
from stbench.geometry import dist_to_rect
from stbench.job_registry import get
from stbench.keyframes import Mark, Move
from stbench.mapspec import load_map
from stbench.robots import ROBOTS, robot_doc
from stbench.route import robot_obstacles
from stbench.seeding import make_rng
from stbench.timeline_ego import TaskPlan
from stbench.validate import validate

pytestmark = pytest.mark.xdist_group(name=__name__)  # one worker builds the scripts once and every test reads them

SEEDS = (101, 102, 103, 104)
DOCKED = (4.94, 5.05, 0.0)
CLEAN, RESUME = "clean_rooms", "resume"
M = load_map("apartment_a")
JOB = get("vacuum/2")


@pytest.fixture(scope="module")
def scripts():
    """{seed: (the world each task was planned on, the script)}: 4-6 tasks, the real clutter."""
    return {seed: job_kit.generated2(seed) for seed in SEEDS}


def plans_of(seed: int, n: int = 5) -> list[TaskPlan]:
    return JOB.plan(M, seed, n)


def digest(seed: int, n: int) -> str:
    return hashlib.sha256(T.dumps(generate_script("apartment_a", seed, "vacuum", "vacuum/2", n)).encode()).hexdigest()


def test_scripts_are_deterministic():
    """Contract: a 4-task `vacuum/2` script is byte-identical (`timeline.dumps`) in this process and in another with
    another `PYTHONHASHSEED`, and another seed gives another one. Regression caught: an unseeded draw, or a set's or
    dict's iteration order in the plan, the gaps, the doors kept open, the back-outs or the path."""
    out = subprocess.run([sys.executable, __file__, "103"], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout.strip()
    assert digest(103, 4) == out != digest(104, 4)


def clearance_along(world, tl, task, t: float) -> tuple[float, float]:
    """The least distance from the base to the walls and what the planner reads (`robot_obstacles` of the world the task
    was planned on, at `t`) and to the dock (which it touches), over every point of every segment between consecutive
    keyframes of the task (the straight lines that are the path), 0.05 m apart."""
    kfs = tl.ego.keyframes[task.k0:task.k1]
    xs, ys = [], []
    for a, b in zip(kfs, kfs[1:], strict=False):
        n = max(math.ceil(math.dist(a.base[:2], b.base[:2]) / 0.05), 1)
        s = np.linspace(0.0, 1.0, n + 1)
        xs.append(a.base[0] + (b.base[0] - a.base[0]) * s)
        ys.append(a.base[1] + (b.base[1] - a.base[1]) * s)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    dock = next(r for r in robot_obstacles(M, world, ROBOTS["vacuum"], t) if r.sx == 0.12 and r.sy == 0.28)
    others = [r for r in M.wall_pieces() + robot_obstacles(M, world, ROBOTS["vacuum"], t) if r != dock]
    return (min(float(dist_to_rect(xs, ys, r).min()) for r in others), float(dist_to_rect(xs, ys, dock).min()))


def test_every_task_runs_validly(scripts):
    """Contract: on every script `validate` (the generic rules and the job's own) finds nothing, and the base keeps
    the body's clearance (0.17 m radius + 0.02 m = 0.19 m, less a millimetre) from the walls and everything the
    planner reads (`robot_obstacles` of the world as the task's gap left it) at every point of every segment between
    keyframes, not only at keyframes, and never goes into the dock it touches (0.17 m). Regression caught: a path
    through furniture, a leg, a door leaf or a moved piece, or a status outside the vocabulary."""
    for seed, (worlds, tl) in scripts.items():
        assert validate(tl, M) == [], seed
        assert len(worlds) == len(tl.ego.tasks) >= 4
        for task in tl.ego.tasks:
            wall, dock = clearance_along(worlds[task.index], tl, task, tl.keyframe(task.k0).t)
            assert wall >= 0.19 - 0.001, (seed, task.index)
            assert dock >= 0.17 - 0.001, (seed, task.index)


def flipping(m):
    """A `gaps` stand-in that, through the ledger before each task, opens every interior door on the task's ways (as
    `world_processes.gaps` guarantees) and toggles every one off them."""
    def gaps(w, _m, _seed):
        def between(lo, hi, keep_open):
            on = world_processes.doors_on_ways(m, keep_open)
            states = w.ledger.finish()[1]
            for door in m.toggling_doors():
                eid = w.door_entity[door.id]
                now = next(bool(s.value) for s in states if s.entity == eid and s.key == "open" and s.covers(hi))
                w.ledger.set_state(eid, "open", hi, True if door.id in on else not now)
        return between
    return gaps


def test_paths_ignore_door_states():
    """Contract (design §8.1, "paths leak"): a task is planned with every door off its ways an obstacle in both leaf
    positions, so no door's state changes a path (the robot's own path log would read it). On seed 101 the keyframes
    are byte for byte those of the same script with every door off each task's ways toggled before it (open to shut,
    shut to open, alternating from task to task, set through the ledger). Regression caught: a detour round a shut door,
    which the path log would read as the door's state."""
    quiet = lambda w, m, seed: lambda lo, hi, keep_open: None  # noqa: E731
    with mock.patch.object(world_processes, "gaps", quiet):
        base = generate_script("apartment_a", 101, "vacuum", "vacuum/2", 5)
    with mock.patch.object(world_processes, "gaps", flipping(M)):
        flipped = generate_script("apartment_a", 101, "vacuum", "vacuum/2", 5)
    states = {(s.entity, s.value) for s in flipped.states}
    assert len({v for _, v in states}) == 2  # both an open and a shut state exist: the doors really were toggled
    assert T.digest(base) != T.digest(flipped)  # the worlds differ (the door states) ...
    assert base.ego_hash == flipped.ego_hash  # ... and the paths do not


def parity(_m, _seed, i, j) -> bool:
    """A doorway coin that reads neither the world nor the seed: blocked for the rooms whose task and room indices add
    up to an even number."""
    return (i + j) % 2 == 0


def test_decisions_come_from_the_plan_and_the_coins_alone(scripts):
    """Contract (design §8.5, 0.5.0-control): a fixed plan scripted over two different worlds (seeds 101 and 102, `plan`
    replaced by seed 101's) with the same doorway coins (`parity`, which reads no seed) gives the same decision kinds,
    task by task, in the same order. Regression caught: a decision point the world triggered (a clutter piece, a chair,
    a shut door), or a coin that is read from the world."""
    _, first = scripts[101]
    n = len(first.ego.tasks)
    swapped = replace(JOB, plan=lambda m, seed, n_tasks: JOB.plan(m, 101, n_tasks))
    with mock.patch.object(job_vacuum2_doorway, "doorway_blocked", parity):
        home = generate_script("apartment_a", 101, "vacuum", "vacuum/2", n)
        with mock.patch.object(job_vacuum2, "SPEC", swapped):
            other = generate_script("apartment_a", 102, "vacuum", "vacuum/2", n)
    assert any(d.response == "skip" for d in home.ego.decisions)  # the coins did something
    assert other.world_hash != home.world_hash  # another world
    kinds = lambda tl: [(tl.task_of(d.k).index, d.kind) for d in tl.ego.decisions]  # noqa: E731
    assert kinds(other) == kinds(home)
    assert [t.plan for t in other.ego.tasks] == [t.plan for t in home.ego.tasks]


def test_each_task_starts_docked_in_the_start_set(scripts):
    """Contract: every task's first and last keyframe stand at the docked pose (4.94, 5.05) heading 0 and are `docked`,
    its battery starts in 90-100 and never rises, its brush is off at both ends, and the task index counts from 0 in
    order: nothing carried over from an earlier task shows where a window sits. Regression caught: state carried over
    from the task before, or a battery that keeps draining across a gap."""
    for _, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            for kf in (kfs[0], kfs[-1]):
                assert (kf.phase, kf.base, kf.yaw, kf.task) == ("docked", (DOCKED[0], DOCKED[1], 0.0), 0.0, task.index)
                assert kf.status["brush"] is False
            assert 90 <= kfs[0].status["battery_pct"] <= 100
            batteries = [kf.status["battery_pct"] for kf in kfs]
            assert batteries == sorted(batteries, reverse=True) and batteries[0] - batteries[-1] <= 20


def test_the_span_ends_by_17(scripts):
    """Contract (DECISIONS: gaps and the 7-hour span): over 1,000 seeds and 4-6 tasks the gap draws are each in 10
    minutes to 3 hours and, with every task at its longest (`TASK_MAX_S`), a script spans at most 7 hours (`SPAN_S`),
    the budget binding for some; a case clock starts 07:00-09:59, so it ends by 16:59, never at night under the
    daytime light. On the scripts every task takes at most `TASK_MAX_S`, the longest at least 20% under it, and the
    span is within `SPAN_S`. Regression caught: a case clock after 17:00, or a task or a script longer than the budget
    counts."""
    tight = 0
    for seed in range(1, 1001):
        n = 4 + seed % 3
        gaps = job_vacuum2.gaps("apartment_a", seed, n)
        assert len(gaps) == n - 1 and all(600.0 - 1e-6 <= g <= 10800.0 + 1e-6 for g in gaps), seed
        total = sum(gaps) + n * job_vacuum2.TASK_MAX_S
        assert total <= job_vacuum2.SPAN_S + 1e-6, seed
        tight += total > job_vacuum2.SPAN_S - 1e-6
    assert tight > 20  # the budget binds for 6 tasks: the shrink really ran
    assert (job_vacuum2.TASK_MAX_S, job_vacuum2.SPAN_S) == (1200.0, 25200.0)
    latest = max(datetime.fromisoformat(case.start_clock(make_rng("clock", i))) for i in range(2000))
    assert (latest + timedelta(seconds=job_vacuum2.SPAN_S)).time() <= datetime(2026, 1, 1, 17).time()
    longest = 0.0
    for _, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            longest = max(longest, kfs[-1].t - kfs[0].t)
        assert tl.ego.keyframes[-1].t - tl.ego.keyframes[0].t <= job_vacuum2.SPAN_S
    assert longest <= 0.8 * job_vacuum2.TASK_MAX_S


def test_gaps_come_from_the_seed_alone():
    """Contract: `gaps(map, seed, n)` needs no world: the same call twice gives the same list, another seed another one,
    and a script's gaps (the time from a task's last keyframe to the next one's first) are exactly those. Regression
    caught: gaps drawn from the world's stream, so that a chair's move would shift a clock."""
    assert job_vacuum2.gaps("apartment_a", 7, 5) == job_vacuum2.gaps("apartment_a", 7, 5)
    assert job_vacuum2.gaps("apartment_a", 7, 5) != job_vacuum2.gaps("apartment_a", 8, 5)
    assert job_vacuum2.gaps("apartment_a", 7, 1) == []
    _, tl = job_kit.generated2(101)
    want = job_vacuum2.gaps("apartment_a", 101, len(tl.ego.tasks))
    got = [tl.keyframe(b.k0).t - tl.keyframe(a.k1 - 1).t for a, b in zip(tl.ego.tasks, tl.ego.tasks[1:], strict=False)]
    assert got == pytest.approx(want, abs=1e-3)


def test_plans_follow_their_laws():
    """Contract (plan §5.1; DECISIONS, the plan's laws): over seeds 1-1000 of five tasks, task 0 is `clean_rooms`; no
    task's first room is the hall and the hall comes only as a second room; the share of two-room `clean_rooms` tasks
    is 0.25 and the share of the tasks allowed to stop (not the last, last room not the hall) that stop is 0.3, each
    within 4 standard errors; a `resume` follows exactly the tasks that stop and names the room they stopped in, never
    the hall; no plan stops its last task; a stop's share is in 0.3-0.7. Regression caught: a resume with nothing
    before it, a stop no task resumes, or a hall-only task that asks nothing."""
    clean = two = allowed = stops = 0
    for seed in range(1, 1001):
        plans = plans_of(seed)
        assert plans[0].kind == CLEAN and len(plans) == 5
        for i, plan in enumerate(plans):
            kinds = plan.params["kinds"]
            stop = plan.params.get("stop")
            if plan.kind == RESUME:
                before = plans[i - 1]
                assert i and before.params.get("stop") is not None and kinds == [before.params["kinds"][-1]]
                assert kinds != ["hall"] and stop is None
                continue
            assert plan.kind == CLEAN and kinds[0] != "hall" and "hall" not in kinds[:1]
            assert 1 <= len(kinds) <= 2 and len(set(kinds)) == len(kinds)
            clean, two = clean + 1, two + (len(kinds) == 2)
            if i < 4 and kinds[-1] != "hall":
                allowed, stops = allowed + 1, stops + (stop is not None)
                assert (stop is None) or (0.3 <= stop <= 0.7 and plans[i + 1].kind == RESUME)
            else:
                assert stop is None
            assert plan.params["rooms"] == ", then the ".join(kinds)
    for count, total, p in ((two, clean, 0.25), (stops, allowed, 0.3)):
        assert abs(count / total - p) <= 4 * math.sqrt(p * (1 - p) / total), (count, total)
    with pytest.raises(ValueError, match="at least one task"):
        plans_of(1, 0)


def test_assignments_read_as_briefs():
    """Contract: for every plan of seeds 1-200, `brief.compose` (variant "correct") renders each task's text with the
    room words only: "clean the kitchen, then the hall" for a `clean_rooms` task, "finish cleaning the kitchen" for a
    `resume`, no key of the job's (`kinds`, `stop`) in it; by hand, kitchen then hall gives "Its current assignment is
    to clean the kitchen, then the hall." and a resumed bedroom "Its current assignment is to finish cleaning the
    bedroom." Regression caught: a template that names a param the plan lacks, or a stop's share leaking into a brief.
    """
    robot = robot_doc(ROBOTS["vacuum"])
    hand = [TaskPlan(CLEAN, {"rooms": "kitchen, then the hall", "kinds": ["kitchen", "hall"]}),
            TaskPlan(RESUME, {"rooms": "bedroom", "kinds": ["bedroom"]})]
    texts = brief.compose(hand, job=JOB, robot=robot, domain="home", channel="rgb_pose")
    assert "Its current assignment is to clean the kitchen, then the hall. You will receive" in texts[0]
    assert texts[1] == "Its current assignment is to finish cleaning the bedroom."
    for seed in range(1, 201):
        plans = plans_of(seed)
        texts = brief.compose(plans, job=JOB, robot=robot, domain="home", channel="rgb_pose", variant="correct")
        assert len(texts) == len(plans)
        for plan, text in zip(plans, texts, strict=True):
            lead = "finish cleaning" if plan.kind == RESUME else "clean"
            assert f"Its current assignment is to {lead} the {plan.params['rooms']}." in text
            assert "stop" not in text and "kinds" not in text and not any(ch.isdigit() for ch in text.split(".")[0])


def test_the_keyframe_swath_is_the_paths():
    """Contract (plan §5.1; M2's `uncovered` reads the brush's swath from keyframes): for every task, `grid.swath` (the
    brush disc, 0.25 m) over each run of consecutive brush-on keyframes, taken together, equals cell for cell
    `grid.swath` over the primitives' cleaning moves, so no corner is cut between keyframes: every brush-on vertex is
    a keyframe (the run's first point included, which `Driver.begin` frames, and a vertex that turns under 1 degree,
    which `Driver(exact=True)` frames). Regression caught: corner-cutting between keyframes, which makes `uncovered`'s
    gold wrong."""
    from stbench import grid

    caught = []
    real = job_vacuum2.frames

    def spy(prims, policy, t0=0.0):
        caught.append(prims)
        return real(prims, policy, t0=t0)

    with mock.patch.object(job_vacuum2, "frames", spy):
        tl = generate_script("apartment_a", 101, "vacuum", "vacuum/2", 5)
    brush = ROBOTS["vacuum"].tools[0]
    assert len(caught) == len(tl.ego.tasks) == 5
    cleaned = 0
    for task, prims in zip(tl.ego.tasks, caught, strict=True):
        want: set = set()
        for p in prims:
            if isinstance(p, Move) and p.phase == "cleaning":
                want |= grid.swath([(*p.a, p.yaw), (*p.b, p.yaw)], brush.shape, *brush.size_m)
        kfs = tl.ego.keyframes[task.k0:task.k1]
        got: set = set()
        for run in _runs([kf for kf in kfs], lambda kf: kf.status["brush"]):
            got |= grid.swath([(kf.base[0], kf.base[1], kf.yaw) for kf in run], brush.shape, *brush.size_m)
        assert got == want, task.index
        cleaned += bool(want)
    assert cleaned >= 3  # tasks that skip every room clean nothing and sweep nothing


def _runs(items, on):
    """The maximal runs of consecutive items for which `on` holds."""
    out: list[list] = []
    for i, item in enumerate(items):
        if on(item):
            if i == 0 or not on(items[i - 1]):
                out.append([])
            out[-1].append(item)
    return out


def test_a_vertex_that_turns_under_a_degree_is_a_frame_when_the_driver_is_exact():
    """Contract: `Driver(exact=True)` puts a `Mark` at a vertex whose turn is under `MIN_TURN_RAD` (1 degree), where no
    turn is made, so the straight lines between keyframes are the polyline; the default driver does not (`vacuum/1`'s
    primitives stay as they were); `begin` puts a frame at the start of a run, and a decision at the start of a polyline
    after no turn gets a frame of its own phase. Hand cases: the polyline (0,0), (1,0), (2,0.0087) turns atan(0.0087) =
    0.5 degrees at (1,0); a polyline (0,0), (1,0) after a heading of 0 turns nothing. Regression caught: a corner cut by
    up to a centimetre that flips a swath cell, or a decision that carries the phase of the run before."""
    poly = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0087)]
    plain, exact = job_vacuum_path.Driver(0.0, 45.0), job_vacuum_path.Driver(0.0, 45.0, exact=True)
    plain.go(poly, 0.3, "cleaning")
    exact.go(poly, 0.3, "cleaning")
    assert [type(p) for p in plain.prims] == [Move, Move]
    assert [type(p) for p in exact.prims] == [Move, Mark, Move]
    again = job_vacuum_path.Driver(0.0, 45.0, exact=True)
    again.go([(0.0, 0.0), (1.0, 0.0)], 0.3, "transit", (0, 0.0, "before_doorway"))
    assert [type(p).__name__ for p in again.prims] == ["Hold", "Mark", "Move"] and again.prims[0].phase == "transit"
    shy = job_vacuum_path.Driver(0.0, 45.0)
    shy.go([(0.0, 0.0), (1.0, 0.0)], 0.3, "transit", (0, 0.0, "before_doorway"))
    assert [type(p).__name__ for p in shy.prims] == ["Mark", "Move"]


def test_every_event_resolves_in_every_task(scripts):
    """Contract (plan §2.2): each of the four events resolves, through `events.resolve` over a task's delivered pose
    rows (their `k` and status), to the keyframe the plan names in that task: the last undocking one, the first
    cleaning one, the last cleaning one (the `room_done` or `cleaning_stopped` decision's, the stop in a stopped task)
    and the first docked one after the docking ones; a task that skipped every room has no cleaning, and its two
    cleaning events resolve to nothing. Regression caught: an event no system can resolve in a later task."""
    cleaned = 0
    for _, (_, tl) in scripts.items():
        rows = pose_rows(tl)
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            steps = [{"k": r["k"], "status": r["status"]} for r in rows[task.k0:task.k1]]
            cleans = [kf.k for kf in kfs if kf.phase == "cleaning"]
            want = {"when you left the dock": max(kf.k for kf in kfs if kf.phase == "undocking"),
                    "when you started cleaning": min(cleans, default=None),
                    "when you finished cleaning": max(cleans, default=None),
                    "when you got back to the dock": task.k1 - 1}
            assert {w: resolve(rule, steps) for w, rule in JOB.events.items()} == want
            ends = [d for d in tl.decisions_in(task.k0, task.k1) if d.kind in ("room_done", "cleaning_stopped")]
            assert bool(ends) == bool(cleans)
            if cleans:
                cleaned += 1
                assert ends[-1].k == max(cleans)
                assert (ends[-1].kind == "cleaning_stopped") == (task.plan.params.get("stop") is not None)
    assert cleaned >= 8


def with_task_times(tl, first: int, shift: float):
    """`tl` with every keyframe from keyframe `first` on `shift` seconds later."""
    kfs = [replace(kf, t=kf.t + shift) if kf.k >= first else kf for kf in tl.ego.keyframes]
    return replace(tl, ego=replace(tl.ego, keyframes=kfs))


@pytest.mark.parametrize(("edit", "code"), [
    ("brush in transit", "job_brush"),
    ("first keyframe off the dock", "job_dock"),
    ("last keyframe facing away", "job_dock"),
    ("a decision dropped", "job_decisions"),
    ("cleaning_stopped off the last cleaning keyframe", "job_decisions"),
    ("a phase skipped", "job_phase_order"),
    ("a gap of a minute", "job_gap"),
    ("a script of 9 hours", "job_span"),
    ("a task of 30 minutes", "job_task_time"),
    ("a resume with nothing before it", "job_resume"),
    ("a stop no resume follows", "job_resume"),
    ("a resume of another room", "job_resume"),
    ("targets that are not the plan's rooms", "job_targets"),
])
def test_the_jobs_own_checks_have_teeth(edit, code):
    """Contract: `checks` flags a script that is not a `vacuum/2` one: the brush on outside `cleaning`, a task that
    does not start or end at the docked pose, a decision missing or off its keyframe, a phase missing, a gap under 10
    minutes, a script of over 7 hours, a task of over 20 minutes, a resume that follows no stopped task or names
    another room, a stop no resume follows, or targets that are not the plan's rooms. Seed 101's task 0 stops and task
    1 resumes it. Regression caught: a rule of the job that no script can break."""
    _, tl = job_kit.generated2(101)
    t0, t1 = tl.ego.tasks[0], tl.ego.tasks[1]
    assert (t0.plan.params["stop"] is not None, t1.plan.kind) == (True, RESUME)
    kfs, decisions, tasks = list(tl.ego.keyframes), list(tl.ego.decisions), list(tl.ego.tasks)
    transit = next(kf for kf in kfs if kf.phase == "transit")
    broken = tl
    if edit == "brush in transit":
        kfs[transit.k] = replace(transit, status={**transit.status, "brush": True})
    elif edit == "first keyframe off the dock":
        kfs[0] = replace(kfs[0], base=(4.96, 5.05, 0.0))
    elif edit == "last keyframe facing away":
        kfs[t0.k1 - 1] = replace(kfs[t0.k1 - 1], yaw=math.pi / 2)
    elif edit == "a decision dropped":
        decisions.pop(0)
    elif edit == "cleaning_stopped off the last cleaning keyframe":
        i = next(i for i, d in enumerate(decisions) if d.kind == "cleaning_stopped")
        decisions[i] = replace(decisions[i], k=decisions[i].k - 1)
    elif edit == "a phase skipped":
        kfs = [replace(kf, phase="docking") if kf.phase == "undocking" and kf.task == 0 else kf for kf in kfs]
    elif edit == "a gap of a minute":
        broken = with_task_times(tl, t1.k0, -(tl.keyframe(t1.k0).t - tl.keyframe(t0.k1 - 1).t) + 60.0)
    elif edit == "a script of 9 hours":
        broken = with_task_times(tl, tasks[-1].k0, 32400.0)
    elif edit == "a task of 30 minutes":
        broken = with_task_times(tl, t0.k0 + 100, 1800.0)
    elif edit == "a resume with nothing before it":
        tasks[0] = replace(t0, plan=replace(t1.plan))
    elif edit == "a stop no resume follows":
        tasks[1] = replace(t1, plan=TaskPlan(CLEAN, {"rooms": "bedroom", "kinds": ["bedroom"]}))
    elif edit == "a resume of another room":
        tasks[1] = replace(t1, plan=TaskPlan(RESUME, {"rooms": "bedroom", "kinds": ["bedroom"]}))
    else:
        tasks[0] = replace(t0, targets=("R9",))
    broken = replace(broken, ego=replace(broken.ego, keyframes=kfs if broken is tl else broken.ego.keyframes,
                                         decisions=decisions, tasks=tasks))
    assert code in {p.code for p in job_vacuum2._checks(broken, M)}
    assert job_vacuum2._checks(tl, M) == []


def test_the_doors_off_a_tasks_ways_are_shut_in_the_planner(scripts):
    """Contract (design §8.1): a task is planned with every interior door off the ways between the dock's room and its
    rooms shut in both leaf positions, so a robot sent to some rooms keeps out of the others: on every task of the
    scripts, the regions its keyframes stand in (a doorway is none) are the hall, its own rooms and the ends of the
    doors on its ways (a way from the living room to the bathroom may cross the kitchen by its arch). Regression
    caught: a planner that lets a task wander into rooms it was not sent to, or one that leaves an off-way door's
    state free to change its path."""
    for _, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            visited = {tl.region_of(kf.base[:2]) for kf in tl.ego.keyframes[task.k0:task.k1]} - {None}
            on = world_processes.doors_on_ways(M, {"R1", *task.targets})
            ends = {r for o in M.openings if o.id in on for r in o.between}
            assert visited <= {"R1", *task.targets} | ends, (task.index, visited)


if __name__ == "__main__":  # the second process of `test_scripts_are_deterministic`
    print(digest(int(sys.argv[1]), 4))
