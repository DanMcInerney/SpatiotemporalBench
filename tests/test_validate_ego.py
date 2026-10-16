"""validate()'s ego rules: keyframes, tasks, frame spacing, the camera derived from the base, the
robot, status, decision points and the job. Each test breaks one thing in the hand-built timeline
(tests/builders.py)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from builders import MUG, keyframe, minimal_timeline
from conftest import script_on_map
from stbench import job_vacuum
from stbench.mapspec import load_map
from stbench.timeline import DecisionPoint, Task, TaskPlan
from stbench.validate import Problem, validate
from validate_kit import codes, generic_rules_only, with_keyframes  # noqa: F401  (a fixture)

# these tests hand-make vacuum scripts, which the vacuum's own checks (`test_job_vacuum.py`) would refuse
pytestmark = pytest.mark.usefixtures("generic_rules_only")

POSES = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, 0.0), (3.2, 3.4, 0.0, 0.0)]  # the living room's open floor


# -- the ego: keyframes, tasks, frames, decisions -------------------------------------------------------------------
def test_keyframe_outside_the_map(apt):
    """Contract: a keyframe's base lies inside the map (`outside_map`). Hand case: a base at x = -3.0, 3 m outside the
    west wall and clear of everything, so only the map rule fires. Regression caught: a robot driven out of the
    apartment, whose camera then sees the void."""
    tl = minimal_timeline()
    kfs = list(tl.ego.keyframes)
    kfs[2] = keyframe("helper", 2, 3.0, (-3.0, 1.0, 0.0), 0.5, task=0, phase="navigating")
    assert codes(with_keyframes(tl, kfs), apt) == {"outside_map"}


def test_keyframe_times_never_decrease(apt):
    """Contract: time runs forward across tasks too: a keyframe at 2.0 s after one at 3.0 s is `keyframe_order` (the
    task change keeps the spacing rule out of it). Regression caught: a script whose clock runs backwards, which
    breaks every "before" and "after" in the questions."""
    tl = minimal_timeline()
    tl.placements = [replace(p, t1=None) if p.entity == MUG and p.t0 == 0.0 else p for p in tl.placements
                     if not (p.entity == MUG and p.t0 > 0.0)]  # the mug stays: task 1, widened, would hold its move
    kfs = list(tl.ego.keyframes)
    kfs[3] = replace(kfs[3], t=2.0)  # the first keyframe of task 1, after keyframe 2 at 3.0 s
    assert codes(with_keyframes(tl, kfs), apt) == {"keyframe_order"}


def test_keyframe_index_must_match_position(apt):
    """Contract: a keyframe's k is its list index (`keyframe_index`). Regression caught: a gap or repeat in k, which
    misaligns the pose rows and the rendered frames."""
    tl = minimal_timeline()
    kfs = list(tl.ego.keyframes)
    kfs[2] = replace(kfs[2], k=7)
    assert "keyframe_index" in codes(with_keyframes(tl, kfs), apt)


def test_tasks_cover_every_keyframe(apt):
    """Contract: the tasks are contiguous from keyframe 0 to n, each non-empty and numbered by position; a gap, an
    overlap and an empty task are each flagged `task_layout`, and a keyframe in no task `keyframe_outside_task`.
    Regression caught: a keyframe in no task, which delivery and the compiler cannot place."""
    plan = TaskPlan("fetch", {})
    gap = minimal_timeline()
    gap.ego = replace(gap.ego, tasks=[Task(0, 0, 2, plan, ()), Task(1, 3, 6, plan, ())])
    assert {"task_layout", "keyframe_outside_task"} <= codes(gap, apt)
    overlap = minimal_timeline()
    overlap.ego = replace(overlap.ego, tasks=[Task(0, 0, 4, plan, ()), Task(1, 3, 6, plan, ())])
    assert "task_layout" in codes(overlap, apt)
    empty = minimal_timeline()
    empty.ego = replace(empty.ego, tasks=[Task(0, 0, 3, plan, ()), Task(1, 3, 3, plan, ()), Task(2, 3, 6, plan, ())])
    assert "task_layout" in codes(empty, apt)
    short = minimal_timeline()
    short.ego = replace(short.ego, tasks=[Task(0, 0, 3, plan, ()), Task(1, 3, 5, plan, ())])  # keyframe 5 uncovered
    assert {"task_layout", "keyframe_outside_task"} <= codes(short, apt)
    assert validate(minimal_timeline(), apt) == []  # the same plan, laid out right, is fine


def test_a_keyframe_names_the_task_it_lies_in(apt):
    """Contract: a keyframe's `task` is the index of the task whose range holds it. Regression caught: a keyframe that
    claims another task's status, which a window's rebasing would then number wrongly."""
    tl = minimal_timeline()
    kfs = list(tl.ego.keyframes)
    kfs[1] = replace(kfs[1], task=1)
    assert codes(with_keyframes(tl, kfs), apt) == {"keyframe_outside_task"}


def test_frames_respect_min_dt(apt):
    """Contract: two keyframes of one task are at least the robot's `min_dt_s` (1.0 s) apart; 0.9 s is flagged and
    exactly 1.0 s is not. Regression caught: clocks that collide at whole seconds (map X-H3)."""
    tl = minimal_timeline()
    kfs = list(tl.ego.keyframes)
    kfs[1] = replace(kfs[1], t=0.9)  # keyframe 0 is at 0.0 s and keyframe 2 at 3.0 s
    assert codes(with_keyframes(tl, kfs), apt) == {"frame_spacing"}
    kfs[1] = replace(kfs[1], t=1.0)
    assert validate(with_keyframes(minimal_timeline(), kfs), apt) == []


def test_camera_is_derived_from_the_base(apt):
    """Contract: a keyframe's camera is exactly the pose `robots.camera_pose` gives for its base, heading and pitch; a
    camera 1 cm off is flagged. Regression caught: a camera drifting from the robot that carries it."""
    tl = minimal_timeline()
    kfs = list(tl.ego.keyframes)
    kfs[1] = replace(kfs[1], pos=(kfs[1].pos[0], kfs[1].pos[1] + 0.01, kfs[1].pos[2]))
    assert codes(with_keyframes(tl, kfs), apt) == {"camera_mismatch"}


def test_the_camera_tilts_only_within_the_robots_limits(apt):
    """Contract: the helper's gimbal runs from -60 to +20 degrees; -61 degrees is flagged `pitch_limit` and -60 is not.
    Regression caught: a camera tilted past what the robot can do."""
    tl = minimal_timeline()
    kfs = list(tl.ego.keyframes)
    for deg, expect in ((-61.0, {"pitch_limit"}), (-60.0, set()), (20.0, set()), (21.0, {"pitch_limit"})):
        kfs[1] = keyframe("helper", 1, 1.5, (3.8, 1.0, 0.0), 0.5, task=0, phase="navigating", pitch=math.radians(deg))
        assert codes(with_keyframes(minimal_timeline(), kfs), apt) == expect, deg


def test_an_unknown_robot_cannot_be_checked(apt):
    """Contract: the robot decides the phases, camera and body, so a script of a robot nobody has is `unknown_robot`,
    reported and not skipped (its job is renamed with it, so only this one code fires). Regression caught: a
    `ROBOTS.get` that returns None and skips every robot rule, which validates a script of a robot that does not
    exist."""
    tl = minimal_timeline()
    tl.episode = replace(tl.episode, robot="hovercraft", job="hovercraft/1")
    assert codes(tl, apt) == {"unknown_robot"}


def test_status_is_the_robots_vocabulary_and_each_task_starts_in_its_start_set(apt):
    """Contract: a phase must be one of the robot's, the status keys exactly its own with their types, and a task's
    first keyframe in the start set (the vacuum's: docked, battery 90-100, brush off). Regression caught: a status that
    carries world facts, or one carried over from an earlier task (which reveals a window's place, map X-H4)."""
    tl = script_on_map("apartment_a", 1, "vacuum", POSES)
    assert validate(tl, load_map("apartment_a")) == []

    def broken(**edit):
        kfs = list(tl.ego.keyframes)
        kfs[1] = replace(kfs[1], **edit)
        return codes(with_keyframes(replace(tl), kfs), load_map("apartment_a"))

    assert broken(phase="flying") == {"unknown_phase"}
    assert broken(status={"battery_pct": 100}) == {"bad_status"}  # the brush key is missing
    assert broken(status={"battery_pct": 100, "brush": False, "room": "hall"}) == {"bad_status"}
    assert broken(status={"battery_pct": 100.5, "brush": False}) == {"bad_status"}  # a float where an int is wanted
    first = replace(tl.ego.keyframes[0], status={"battery_pct": 50, "brush": False})
    assert codes(with_keyframes(replace(tl), [first, *tl.ego.keyframes[1:]]), load_map("apartment_a")) == {
        "start_status"}
    started = replace(tl.ego.keyframes[0], phase="cleaning")
    assert codes(with_keyframes(replace(tl), [started, *tl.ego.keyframes[1:]]), load_map("apartment_a")) == {
        "start_status"}
    mid = replace(tl.ego.keyframes[1], status={"battery_pct": 50, "brush": False})  # not a first keyframe: any battery
    assert codes(with_keyframes(replace(tl), [tl.ego.keyframes[0], mid, tl.ego.keyframes[2]]),
                 load_map("apartment_a")) == set()


def test_decisions_are_sorted_inside_a_task_and_in_the_jobs_vocabulary(apt):
    """Contract: decision points have strictly increasing `k`, each inside a task, each kind one of the registered job's
    (`vacuum/1`: before_doorway, room_done). Regression caught: a decision point the compiler cannot place, or a kind
    no family reads."""
    tl = script_on_map("apartment_a", 1, "vacuum", POSES)
    m = load_map("apartment_a")

    def with_decisions(*ds):
        return codes(replace(tl, ego=replace(tl.ego, decisions=[DecisionPoint(k, kind) for k, kind in ds])), m)

    assert with_decisions((0, "before_doorway"), (2, "room_done")) == set()
    assert with_decisions((2, "before_doorway"), (0, "room_done")) == {"decision_order"}
    assert with_decisions((1, "before_doorway"), (1, "room_done")) == {"decision_order"}  # distinct k
    assert with_decisions((9, "room_done")) == {"decision_order"}  # in no task
    assert with_decisions((1, "lap")) == {"decision_kind"}


# -- the job: whose it is, which task kinds, its own checks ----------------------------------------------------------

def test_the_job_is_the_scripts_own_robots(apt, monkeypatch):
    """Contract (plan §2.2): `job_robot` when the job id's robot (the part before "/") is not the script's robot, or
    when the registered job is another robot's; a job nobody has written is read no further (its decisions and task
    kinds are not checked). Hand cases: the helper's script stamped `vacuum/1` (its plans made a vacuum's, so only this
    code fires), the typo `vaccum/1` on the vacuum's script with a decision of a made-up kind, and the vacuum's script
    under a registered job that says it is the helper's. Regression caught: `validate` failing open on the job (a
    helper script judged by the vacuum's dock and decisions, or a typo that switches every job rule off)."""
    stamped = minimal_timeline()
    stamped.ego = replace(stamped.ego, tasks=[replace(t, plan=TaskPlan("clean_room", {})) for t in stamped.ego.tasks])
    stamped.episode = replace(stamped.episode, job="vacuum/1")
    assert codes(stamped, apt) == {"job_robot"}

    m = load_map("apartment_a")
    typo = script_on_map("apartment_a", 1, "vacuum", POSES)
    typo.episode = replace(typo.episode, job="vaccum/1")
    typo.ego = replace(typo.ego, decisions=[DecisionPoint(1, "made_up_kind")])
    assert codes(typo, m) == {"job_robot"}

    monkeypatch.setattr(job_vacuum, "SPEC", replace(job_vacuum.SPEC, robot="helper"))
    assert codes(script_on_map("apartment_a", 1, "vacuum", POSES), m) == {"job_robot"}


def test_a_plan_is_of_one_of_the_jobs_task_kinds(apt):
    """Contract (plan §2.2): `task_kind` for a task whose plan kind is not one of the registered job's `task_kinds`
    (`vacuum/1`: clean_room), read for every task; a job nobody has written has no kinds to read. Hand case: the
    vacuum's script with its one task planned as "mop_room", then as "clean_room"; the helper's `fetch` plans on
    `helper/0`, a job with no module, are fine. Regression caught: a task of a kind the job does not know, which the
    harness's brief composer meets as a KeyError at run time."""
    m = load_map("apartment_a")
    assert codes(script_on_map("apartment_a", 1, "vacuum", POSES, plan=TaskPlan("clean_room", {"room": "kitchen"})),
                 m) == set()
    mop = script_on_map("apartment_a", 1, "vacuum", POSES, plan=TaskPlan("mop_room", {"room": "kitchen"}))
    found = validate(mop, m)
    assert [p.code for p in found] == ["task_kind"] and "mop_room" in found[0].message
    assert validate(minimal_timeline(), apt) == []  # `fetch` on `helper/0`


def test_the_jobs_own_checks_run_and_their_problems_are_reported(apt, monkeypatch):
    """Contract (plan §2.2): `validate` runs the registered job's `checks(timeline, map)` and reports the problems it
    returns, so the job's own rules reach `stbench validate`; a job nobody has written has none to run. Hand case:
    `vacuum/1`'s `checks` replaced by one that returns a single problem of its own code. Regression caught: the hook
    dropped from `validate`, which leaves lane A's checks passing their unit tests and never running on a script."""
    m = load_map("apartment_a")
    tl = script_on_map("apartment_a", 1, "vacuum", POSES)
    assert validate(tl, m) == []
    seen = []

    def checks(timeline, map_):
        seen.append((timeline, map_))
        return [Problem("job_check", "the job's own rule")]

    monkeypatch.setattr(job_vacuum, "SPEC", replace(job_vacuum.SPEC, checks=checks))
    assert validate(tl, m) == [Problem("job_check", "the job's own rule")]
    assert len(seen) == 1 and seen[0][0] is tl and seen[0][1] is m
    assert validate(minimal_timeline(), apt) == []  # `helper/0` has no job to run
