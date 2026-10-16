"""`helper/1`'s declarative part, plan, park pose, reachable surfaces, briefs, events and checks, over `apartment_a`
(`tests/helper_kit.py`; the scripts of seeds 101 and 102 are built once for both helper modules, one xdist group).
The home is the frozen one: its dock at (4.71, 5.05) facing east, the front door at the south end of the hall, so the
park pose is (5.40, 5.05) heading -90 degrees. The helper's reachable surfaces on it, measured (docs/job-helper.md):
the hall's console and shoe rack, the living room's sofa, coffee table and TV console."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace

import pytest

import helper_kit
from stbench import brief, job_gaps, job_helper, job_helper_plan, job_helper_reach
from stbench import timeline as T
from stbench.catalog import HOME
from stbench.episode import pose_rows
from stbench.events import resolve
from stbench.generate import generate_script
from stbench.job_registry import get
from stbench.mapspec import load_map
from stbench.robots import ROBOTS, robot_doc
from stbench.timeline_ego import TaskPlan
from stbench.validate import validate

pytestmark = pytest.mark.xdist_group(name="helper")  # one worker builds the scripts once; both helper modules read them

SEEDS = (101, 102)
M = load_map("apartment_a")
JOB = get("helper/1")
TIDY, FETCH = "tidy", "fetch"


@pytest.fixture(scope="module")
def scripts():
    """{seed: (the world each task was planned on, the script)}: 4-6 tasks, the real clutter and the resident."""
    from stbench import build

    return {seed: helper_kit.generated(seed, build.tasks_for((4, 6), "apartment_a", seed, "helper", "helper/1"))
            for seed in SEEDS}


def digest(seed: int, n: int) -> str:
    return hashlib.sha256(T.dumps(generate_script("apartment_a", seed, "helper", "helper/1", n)).encode()).hexdigest()


def test_the_helpers_declarative_part():
    """Contract (M3 brief H; FD-0.5.0-control-impl §4): `helper/1` is the helper's job in the home with the task kinds
    `tidy` and `fetch`, three decision kinds (which families it asks is `test_jobs_ask_060`'s; the coin at
    `before_place` is about the window of the surface the helper faces), the zone rule for that window (a spot, 0.01 m
    to 0.5 m above its surface), the home's ladder, a subgoal, four events over its own phases, the dock as its anchor,
    no swap pair and no continued kind, a brief text for each kind, and a relevance that holds every carried kind of
    the home, the task's rooms (the park room's kind included) and the phases navigating and carrying. Regression
    caught: a zone rule no coin reads, an event naming another robot's phase, a window that may not start at a task,
    or a kind with no assignment text."""
    assert (JOB.robot, JOB.domain, JOB.task_kinds) == ("helper", "home", (TIDY, FETCH))
    assert set(JOB.decisions) == {"before_doorway", "before_grasp", "before_place"}
    assert JOB.anchors == frozenset({"dock"}) and JOB.swaps == () and JOB.continues == frozenset()
    assert set(JOB.events) == {"when you left your spot", "when you started carrying",
                               "when you finished carrying", "when you got back to your spot"}
    assert JOB.assignments == {
        TIDY: "Its current assignment is to carry something from the {from_room} to the {to_room}.",
        FETCH: "Its current assignment is to bring something from the {from_room} to its spot."}
    assert set(JOB.zones) == {"before_place"} and JOB.slot_zones == ()
    zone = JOB.zones["before_place"]
    assert (zone.piece, zone.height_m, zone.top_m) == ("spot", 0.01, 0.5)
    assert JOB.ladder == (0.25, 0.5, 1.0, 2.0) and JOB.subgoals is not None
    plan = TaskPlan(TIDY, {"from_room": "hall", "to_room": "living room", "park_room": "hall"})
    rel = JOB.relevance(plan)
    assert rel.regions == frozenset({"hall", "living room"}) and rel.phases == frozenset({"navigating", "carrying"})
    assert rel.kinds == frozenset(HOME.kinds)
    fetch = TaskPlan(FETCH, {"from_room": "living room", "park_room": "hall"})
    assert JOB.relevance(fetch).regions == frozenset({"living room", "hall"})


def test_scripts_are_deterministic():
    """Contract: a 1-task `helper/1` script is byte-identical (`timeline.dumps`) in this process and in another with
    another `PYTHONHASHSEED`, and another seed gives another one. Regression caught: an unseeded draw, or a set's or
    dict's iteration order in the qualified rooms, the doors kept open, the reach poses or the path."""
    out = subprocess.run([sys.executable, __file__, "103"], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout.strip()
    assert digest(103, 1) == out != digest(104, 1)


def test_the_park_pose_is_the_docks_apron():
    """Contract: the park pose is on the dock's axis, 0.63 m in front of its front face, facing the centre of the
    outside door: by hand (4.71 + 0.12 / 2 + 0.63, 5.05) = (5.40, 5.05), and the door at x 5.4 on the south wall is
    straight ahead, heading -90 degrees; the pose lies inside the dock's apron (1.2 m in front of the face, 0.5 m
    beyond each side), which clutter and light furniture keep clear. A map with no dock refuses the job, naming the
    map. Regression caught: a pose off the apron, which the world's clutter blocks in half its worlds, or a heading
    that does not face the door."""
    (x, y), yaw = job_helper_plan.park_pose(M)
    assert (x, y, yaw) == (5.4, 5.05, -1.570796)
    assert 4.77 <= x <= 4.77 + 1.2 and 5.05 - 0.14 - 0.5 <= y <= 5.05 + 0.14 + 0.5
    no_dock = replace(M, furniture=tuple(f for f in M.furniture if f.kind != "dock"))
    with pytest.raises(ValueError, match="apartment_a.*dock"):
        job_helper_plan.park_pose(no_dock)
    shut_in = replace(M, openings=tuple(o for o in M.openings if "outside" not in o.between))
    with pytest.raises(ValueError, match="apartment_a.*outside"):
        job_helper_plan.park_pose(shut_in)


def test_the_reachable_surfaces_are_those_the_map_gives():
    """Contract (DECISIONS, the helper's surfaces): a room qualifies when at least two of its surface pieces count, a
    piece counting when 0.15 m2 of the map's own floor (its pieces at their map places, the interior doors open, no
    clutter) lies within the arm's reach of it, 0.3-0.6 m from the front edge, in its room and reachable from the park
    pose. On `apartment_a`, by the measure in docs/job-helper.md: the hall (console F20, shoe rack F21) and the living
    room (sofa F10, coffee table F11, TV console F12), and no other: the bookshelf (0.11 m2), the dining table
    (0.06 m2) and the bath shelf (0.06 m2) do not count, and the kitchen counters, the nightstands, the desk, the bed
    reach nothing; the bathroom has one counted piece, the vanity. A park pose that is not free floor is refused.
    Regression caught: a rule that names a piece, one that counts a piece the helper cannot stand at, or a room with one
    piece whose clutter would leave a task nowhere to go."""
    assert dict(job_helper_reach.qualified(M, (5.4, 5.05))) == {helper_kit.HALL: ("F20", "F21"),
                                                               helper_kit.LIVING: ("F10", "F11", "F12")}
    with pytest.raises(ValueError, match="not free floor"):
        job_helper_reach.qualified(M, (0.4, 0.4))  # inside the fridge
    assert [f.id for f in job_helper_reach.surface_pieces(M)] == [
        "F02", "F03", "F04", "F06", "F07", "F10", "F11", "F12", "F13", "F20", "F21", "F30", "F31", "F32", "F33", "F40",
        "F43"]  # the fixed, blocking pieces with a surface, in id order: the rugs and the fixtures with none are not


def test_plans_follow_their_laws():
    """Contract (the plan's laws): over seeds 1-1000 of five tasks, the tidy share is 0.5 within 4 standard errors; a
    tidy goes between two different qualified rooms and a fetch from a qualified room other than the hall (the park
    room) to the hall; every piece named is a counted piece of its room; `from_room` and `to_room` are the rooms' kind
    words and `regions` their ids; a fetch has no `to_room`, and its target is the hall console F20 in every plan (by
    hand: the park room's counted piece nearest the park pose, 0.95 m from it; the shoe rack F21 is 4.55 m away),
    drawn from nothing. Regression caught: a task with nowhere to go, a fetch from the park room (which crosses no
    doorway), a fetch with no surface to put the thing on, or a law that is not the declared one."""
    surfaces, kinds = helper_kit.SURFACES, {"R1": "hall", "R2": "living room"}
    tidy = total = 0
    for seed in range(1, 1001):
        plans = JOB.plan(M, seed, 5)
        assert len(plans) == 5
        for plan in plans:
            p = plan.params
            src, dst = p["regions"]
            total += 1
            assert p["from_room"] == kinds[src] and p["park_room"] == "hall" and p["source"] in surfaces[src]
            if plan.kind == TIDY:
                tidy += 1
                assert src != dst and p["to_room"] == kinds[dst] and p["target"] in surfaces[dst]
            else:
                assert plan.kind == FETCH and src == "R2" and dst == "R1" and "to_room" not in p
                assert p["target"] == "F20"
    assert abs(tidy / total - 0.5) <= 4 * math.sqrt(0.25 / total), (tidy, total)
    with pytest.raises(ValueError, match="at least one task"):
        JOB.plan(M, 1, 0)


def test_assignments_read_as_briefs():
    """Contract: for every plan of seeds 1-200, `brief.compose` (variant "correct") renders each task's text with the
    room words only: by hand a tidy from the hall to the living room is "Its current assignment is to carry something
    from the hall to the living room." and a fetch "...to bring something from the living room to its spot.", no piece
    id and no region id in it. Regression caught: a template that names a param the plan lacks, or a piece id in a
    brief."""
    robot = robot_doc(ROBOTS["helper"])
    hand = [TaskPlan(TIDY, {"from_room": "hall", "to_room": "living room", "source": "F20", "target": "F11"}),
            TaskPlan(FETCH, {"from_room": "living room", "source": "F11"})]
    texts = brief.compose(hand, job=JOB, robot=robot, domain="home", channel="rgb_pose")
    assert "Its current assignment is to carry something from the hall to the living room. You will receive" in texts[0]
    assert texts[1] == "Its current assignment is to bring something from the living room to its spot."
    for seed in range(1, 201):
        plans = JOB.plan(M, seed, 5)
        for plan, text in zip(plans, brief.compose(plans, job=JOB, robot=robot, domain="home", channel="rgb_pose"),
                              strict=True):
            sentence = text.split(".")[0] if text.startswith("Its") else text.split("Its current assignment is ")[1]
            assert sentence.startswith("to ") or text.startswith("Its")
            assert f"from the {plan.params['from_room']}" in text
            assert not any(token in text for token in ("F1", "F2", "R1", "R2", "regions", "source"))


def test_every_event_resolves_in_every_task(scripts):
    """Contract (plan §2.2): each of the four events resolves, through `events.resolve` over a task's delivered pose
    rows (their `k` and status), to the keyframe the plan names: the first navigating keyframe, the first carrying one,
    the last carrying one (the `before_place` decision's) and the first waiting one after the navigating ones (the
    task's last waiting hold; a fetch has one navigating run, a tidy two). Regression caught: an event no system can
    resolve in a later task."""
    for _, (_, tl) in scripts.items():
        rows = pose_rows(tl)
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            steps = [{"k": r["k"], "status": r["status"]} for r in rows[task.k0:task.k1]]
            waiting_end = next(kf.k for kf in kfs if kf.phase == "waiting" and kf.k > task.k0 + 1)
            want = {"when you left your spot": min(kf.k for kf in kfs if kf.phase == "navigating"),
                    "when you started carrying": min(kf.k for kf in kfs if kf.phase == "carrying"),
                    "when you finished carrying": max(kf.k for kf in kfs if kf.phase == "carrying"),
                    "when you got back to your spot": waiting_end}
            assert {w: resolve(rule, steps) for w, rule in JOB.events.items()} == want
            place = next(d for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
            assert place.k == want["when you finished carrying"]


def test_the_span_ends_by_17(scripts):
    """Contract (the gap law, `job_gaps`): a script of the helper spans at most 7 hours (every gap 10 minutes to 3
    hours, shrunk to the budget `SPAN_S - n * TASK_MAX_S`), so a case clock that starts 07:00-09:59 ends by 17:00 under
    the daytime light; on the scripts every task takes at most `TASK_MAX_S` (600 s), the longest at least 20% under it,
    and the script's gaps are exactly `job_gaps.gaps` with the helper's id. Regression caught: a case clock after
    17:00, or a task longer than the span budget counts."""
    assert job_helper.TASK_MAX_S == 600.0 and job_gaps.SPAN_S == 25200.0
    longest = 0.0
    for seed, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            longest = max(longest, kfs[-1].t - kfs[0].t)
        assert tl.ego.keyframes[-1].t - tl.ego.keyframes[0].t <= job_gaps.SPAN_S
        want = job_gaps.gaps("apartment_a", seed, "helper/1", len(tl.ego.tasks), job_helper.TASK_MAX_S)
        tasks = tl.ego.tasks
        got = [tl.keyframe(b.k0).t - tl.keyframe(a.k1 - 1).t for a, b in zip(tasks, tasks[1:], strict=False)]
        assert got == pytest.approx(want, abs=1e-3)
    assert 0.0 < longest <= 0.8 * job_helper.TASK_MAX_S


def with_task_times(tl, first: int, shift: float):
    """`tl` with every keyframe from keyframe `first` on `shift` seconds later."""
    kfs = [replace(kf, t=kf.t + shift) if kf.k >= first else kf for kf in tl.ego.keyframes]
    return replace(tl, ego=replace(tl.ego, keyframes=kfs))


@pytest.mark.parametrize(("edit", "code"), [
    ("first keyframe off the park pose", "job_park"),
    ("last keyframe facing away", "job_park"),
    ("a decision dropped", "job_decisions"),
    ("before_place off the last carrying keyframe", "job_decisions"),
    ("a phase skipped", "job_phase_order"),
    ("a base that does not face its piece at the grasp", "job_reach"),
    ("a gap of a minute", "job_gap"),
    ("a script of 9 hours", "job_span"),
    ("a task of 20 minutes", "job_task_time"),
    ("targets that are not the plan's rooms", "job_targets"),
])
def test_the_jobs_own_checks_have_teeth(edit, code, scripts):
    """Contract: `checks` flags a script that is not a `helper/1` one: a task that does not start or end at the park
    pose, a decision missing or off its keyframe, a phase missing, a grasp from a pose that does not face a counted
    piece of the room, a gap under 10 minutes, a script of over 7 hours, a task of over 10 minutes, or targets that are
    not the plan's rooms. Regression caught: a rule of the job that no script can break."""
    _, tl = scripts[101]
    t0, t1 = tl.ego.tasks[0], tl.ego.tasks[1]
    kfs, decisions, tasks = list(tl.ego.keyframes), list(tl.ego.decisions), list(tl.ego.tasks)
    grasp = next(d for d in decisions if d.kind == "before_grasp")
    broken = tl
    if edit == "first keyframe off the park pose":
        kfs[0] = replace(kfs[0], base=(5.2, 5.05, 0.0))
    elif edit == "last keyframe facing away":
        kfs[t0.k1 - 1] = replace(kfs[t0.k1 - 1], yaw=math.pi / 2)
    elif edit == "a decision dropped":
        decisions.pop(0)
    elif edit == "before_place off the last carrying keyframe":
        i = next(i for i, d in enumerate(decisions) if d.kind == "before_place")
        decisions[i] = replace(decisions[i], k=decisions[i].k - 1)
    elif edit == "a phase skipped":
        kfs = [replace(kf, phase="placing") if kf.phase == "grasping" and kf.task == 0 else kf for kf in kfs]
    elif edit == "a base that does not face its piece at the grasp":
        kfs[grasp.k] = replace(kfs[grasp.k], yaw=kfs[grasp.k].yaw + 0.01)
    elif edit == "a gap of a minute":
        broken = with_task_times(tl, t1.k0, -(tl.keyframe(t1.k0).t - tl.keyframe(t0.k1 - 1).t) + 60.0)
    elif edit == "a script of 9 hours":
        broken = with_task_times(tl, tasks[-1].k0, 32400.0)
    elif edit == "a task of 20 minutes":
        broken = with_task_times(tl, t0.k0 + 20, 1200.0)
    else:
        tasks[0] = replace(t0, targets=("R9", "R9"))
    broken = replace(broken, ego=replace(broken.ego, keyframes=kfs if broken is tl else broken.ego.keyframes,
                                         decisions=decisions, tasks=tasks))
    assert code in {p.code for p in job_helper._checks(broken, M)}
    assert job_helper._checks(tl, M) == []
    assert validate(tl, M) == []


if __name__ == "__main__":  # the second process of `test_scripts_are_deterministic`
    print(digest(int(sys.argv[1]), 1))
