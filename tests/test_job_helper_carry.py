"""`helper/1`'s real carry over `apartment_a` with the real clutter, gaps and resident: the scripts of seeds 101 and 102
are built once for the helper's modules (`tests/helper_kit.py`, one xdist group). The thing a task moves is a world
entity that leaves its source at the last grasping keyframe and appears at its target at the last placing keyframe; the
camera tilts down from -15 to -45 degrees around the grasp and the place; the place spot's coin and response
(`job_helper_spot`) show only after the `before_place` decision. Hand values are the map's: the park pose (5.40, 5.05),
the hall console F20 at (5.4, 6.0) (0.95 m from it), the shoe rack F21 at (4.8, 0.5), the living room's sofa F10, coffee
table F11 and TV console F12."""

from __future__ import annotations

import math
from dataclasses import replace
from unittest import mock

import pytest

import helper_kit
from stbench import build, job_helper_carry, job_helper_checks, job_helper_reach, questionnaire
from stbench.generate import generate_script
from stbench.geometry import Rect, corners, distance, point_in_rect, rects_overlap
from stbench.job_helper_spot import ELSEWHERE, GO
from stbench.job_registry import get
from stbench.mapspec import load_map, surface_world_rect
from stbench.validate import validate
from stbench.zone_shapes import surface_window

pytestmark = pytest.mark.xdist_group(name="helper")  # one worker builds the scripts once; every module reads them

SEEDS = (101, 102)
M = load_map("apartment_a")
JOB = get("helper/1")
COUNTED = job_helper_reach.qualified(M, helper_kit.PARK)  # region -> the counted pieces' ids
DOWN, DEFAULT = math.radians(-45.0), math.radians(-15.0)


@pytest.fixture(scope="module")
def scripts():
    return {seed: helper_kit.generated(seed, build.tasks_for((4, 6), "apartment_a", seed, "helper", "helper/1"))
            for seed in SEEDS}


def piece_id(tl, eid: str) -> str:
    """The map's id of the furniture piece an entity stands for: the piece of its kind it stands at the centre of."""
    e, p = tl.entity(eid), tl.placement_of(eid, 0.0)
    return next(f.id for f in M.furniture if f.kind == e.kind and distance(f.center, p.pos[:2]) < 1e-3)


def frames_of(tl, task):
    """(the task's keyframes, the time of its last grasping keyframe, the time of its last placing one)."""
    kfs = tl.ego.keyframes[task.k0:task.k1]
    return (kfs, max(kf.t for kf in kfs if kf.phase == "grasping"), max(kf.t for kf in kfs if kf.phase == "placing"))


def test_the_thing_leaves_at_the_last_grasping_keyframe_and_appears_at_the_last_placing_one(scripts):
    """Contract (FD-0.5.0-control-impl §3): each task's thing is a carried entity whose placement on the source piece
    ends at the time of the task's last `grasping` keyframe, which has no placement from then until the time of the
    task's last `placing` keyframe, where a new placement starts, on a counted piece of the target room (the planned
    piece unless the response sent it elsewhere); `before_place` is the last carrying keyframe, and the placement's
    `on` is the piece's entity. Regression caught: a thing that never moves (the review's sixth defect), one that
    vanishes at the wrong frame (the grasp's frames would show it still on the source, the place's would miss it), or
    one that lands on a piece of another room."""
    for seed, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            kfs, t_grasp, t_place = frames_of(tl, task)
            eid, source, target = task.targets[2:5]
            assert tl.entity(eid).mobility == "carried"
            old, new = tl.placement_of(eid, t_grasp - 0.5), tl.placement_of(eid, t_place)
            assert old.on == source and old.t1 == pytest.approx(t_grasp) and new.t0 == pytest.approx(t_place)
            assert tl.placement_of(eid, t_grasp) is None and tl.placement_of(eid, (t_grasp + t_place) / 2) is None
            assert tl.placement_of(eid, t_place - 0.5) is None and tl.placement_of(eid, t_place + 0.5) == new
            dst = task.plan.params["regions"][1]
            assert piece_id(tl, new.on) in COUNTED[dst]
            assert piece_id(tl, source) in COUNTED[task.plan.params["regions"][0]]
            answer = next(d.response for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
            assert (new.on == target) or answer == ELSEWHERE, (seed, task.index)
            assert tl.placement_of(eid, t_place + 1.0).on == new.on
            assert [p for p in tl.placements_of(eid) if kfs[0].t < p.t0 < kfs[-1].t] == [new]


def test_targets_hold_the_thing_and_its_pieces(scripts):
    """Contract: a task's `targets` are (source room, target room, the thing, the source piece's entity, the target
    piece's entity): the rooms are the plan's, the pieces are counted pieces of those rooms, the planned ones unless the
    world moved the helper to another piece of the room (never another room), and the thing is a carried entity.
    Regression caught: targets that name regions only (the in-task rule would then refuse the thing's placements, and
    the end questions would ask about the thing the robot's log names)."""
    for _, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            src, dst, eid, source, target = task.targets
            assert (src, dst) == tuple(task.plan.params["regions"]) and tl.entity(eid).mobility == "carried"
            assert piece_id(tl, source) in COUNTED[src] and piece_id(tl, target) in COUNTED[dst]
            assert tl.entity(source).mobility == tl.entity(target).mobility == "fixed"


def test_a_fetch_puts_the_thing_on_the_park_rooms_nearest_piece(scripts):
    """Contract: a fetch brings the thing to a surface of the park room, the counted piece nearest the park pose (by
    hand: the hall console F20, 0.95 m from the pose; the shoe rack F21 is 4.55 m away), then drives back to the park
    pose: it runs the same phases as a tidy, its `before_place` is at a pose that faces a counted piece of the hall, and
    it ends at the park pose waiting. Regression caught: a fetch that still puts the thing down at the park pose itself
    (no surface, nothing to render), or one that goes to a piece far from its spot."""
    fetches = 0
    for _, (_, tl) in scripts.items():
        for task in (t for t in tl.ego.tasks if t.plan.kind == "fetch"):
            fetches += 1
            assert task.plan.params["target"] == "F20"
            kfs, _, _ = frames_of(tl, task)
            phases = [kf.phase for i, kf in enumerate(kfs) if i == 0 or kf.phase != kfs[i - 1].phase]
            assert phases == ["waiting", "navigating", "grasping", "carrying", "placing", "navigating", "waiting"]
            place = next(d for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
            kf = tl.keyframe(place.k)
            assert distance(kf.base[:2], helper_kit.PARK) > 0.1 and piece_id(tl, task.targets[4]) == "F20"
            assert distance(kfs[-1].base[:2], helper_kit.PARK) < 0.01
    assert fetches >= 1


def test_the_camera_tilts_down_for_the_grasp_and_the_place(scripts):
    """Contract: the camera is at -45 degrees from the `before_grasp` keyframe through the last grasping one and on the
    `before_place` keyframe and the last placing one, at -15 degrees (the robot's default) while it waits, every
    keyframe's pitch is one of the two (the tilts end on a keyframe) and within the robot's limits (-60 to +20 degrees),
    and the camera's pose is the base's with that pitch (`validate`'s `camera_mismatch`). Regression caught: a grasp
    seen from -15 degrees (a low surface is out of the picture), a pitch written but not applied to the camera's pose,
    or a tilt that runs on past a keyframe."""
    for _, (_, tl) in scripts.items():
        assert validate(tl, M) == []
        for task in tl.ego.tasks:
            kfs, _, _ = frames_of(tl, task)
            at = {d.kind: d.k for d in tl.decisions_in(task.k0, task.k1) if d.kind != "before_doorway"}
            last_grasp = max(kf.k for kf in kfs if kf.phase == "grasping")
            last_place = max(kf.k for kf in kfs if kf.phase == "placing")
            for kf in kfs:
                assert min(abs(kf.cam_pitch - DOWN), abs(kf.cam_pitch - DEFAULT)) < 1e-5
                assert -60 <= math.degrees(kf.cam_pitch) <= 20
                if at["before_grasp"] <= kf.k <= last_grasp or kf.k in (at["before_place"], last_place):
                    assert kf.cam_pitch == pytest.approx(DOWN, abs=1e-5), kf.k
                if kf.phase == "waiting":
                    assert kf.cam_pitch == pytest.approx(DEFAULT, abs=1e-5), kf.k


def test_each_task_asks_the_place_spot_once_after_its_first_sixteen_steps(scripts):
    """Contract: every task holds exactly one `before_place` decision, with a response (`go` or `elsewhere`), at least
    16 steps (`questionnaire.GAP_STEPS`) into its task, so it can be asked; the other decision kinds carry none. The
    response is the window's state at that keyframe (the check `job_spot`). Regression caught: a decision a task does
    not reach in time (it would never be asked), or a response missing from the ego (the subgoal and the checks read
    it)."""
    for _, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            decisions = tl.decisions_in(task.k0, task.k1)
            places = [d for d in decisions if d.kind == "before_place"]
            assert len(places) == 1 and places[0].response in (GO, ELSEWHERE)
            assert places[0].k - task.k0 >= questionnaire.GAP_STEPS
            assert all(d.response is None for d in decisions if d.kind != "before_place")


def test_coins_forced_either_way_leave_the_frames_up_to_the_decision_identical():
    """Contract (FD-0.5.0-control-impl §4): the first task's keyframes (poses, times, camera, phases, statuses), its
    thing, its subgoals and its decisions up to and including `before_place` are the same whether the place spot's coin
    is forced blocked or clear; only the response and what follows differ. The world's coin is enforced after the path
    is planned, on surfaces. Regression caught: a coin that moves the path (the planner would read the clutter), the
    camera or the thing, which would let the answer be read off the frames before the question."""
    runs = {}
    for blocked in (True, False):
        with mock.patch.object(job_helper_carry, "spot_blocked", lambda *_, blocked=blocked: blocked):
            runs[blocked] = generate_script("apartment_a", 103, "helper", "helper/1", 2)
    a, b = runs[True], runs[False]
    task = a.ego.tasks[0]
    place = next(d for d in a.decisions_in(task.k0, task.k1) if d.kind == "before_place")
    assert place.k == next(d for d in b.decisions_in(task.k0, task.k1) if d.kind == "before_place").k
    assert a.ego.keyframes[:place.k + 1] == b.ego.keyframes[:place.k + 1]
    assert a.ego.tasks[0].targets == b.ego.tasks[0].targets
    assert a.ego.tasks[0].plan == b.ego.tasks[0].plan
    responses = {blocked: next(d.response for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
                 for blocked, tl in runs.items()}
    assert responses[True] == ELSEWHERE and responses[False] == GO
    assert validate(a, M) == [] and validate(b, M) == []


def test_the_response_free_spot_lies_outside_the_window_and_on_the_surface(scripts):
    """Contract: when the response is `elsewhere` and the thing is put on the planned piece, its footprint lies wholly
    on that piece's surface and does not meet the window the helper faced (nor the other things there: `validate`'s
    `carried_collision`); when the response is `go` the thing's footprint centre is within 0.2 m of the window's centre
    (the free spot nearest it). Regression caught: a thing put down in the occupied window it just answered
    `elsewhere` about, or half off the table."""
    seen = {GO: 0, ELSEWHERE: 0}
    for _, (_, tl) in scripts.items():
        for task in tl.ego.tasks:
            kfs, _, t_place = frames_of(tl, task)
            place = next(d for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
            kf = tl.keyframe(place.k)
            piece, surface, window = surface_window(M, kf.base[:2], kf.yaw)
            new = tl.placement_of(task.targets[2], t_place)
            seen[place.response] += 1
            if new.on != task.targets[4]:
                continue  # the room's other piece: not the window's surface
            e = tl.entity(task.targets[2])
            box = Rect(new.pos[0], new.pos[1], e.size[0], e.size[1], new.yaw)
            on_surface = surface_world_rect(piece.center, piece.yaw, surface)
            assert all(point_in_rect(on_surface, x, y, 0.011) for x, y in corners(box))
            if place.response == ELSEWHERE:
                assert not rects_overlap(box, window)
            else:
                assert distance((box.cx, box.cy), (window.cx, window.cy)) < 0.2
    assert seen[GO] + seen[ELSEWHERE] >= 8


def pick(scripts, wanted):
    """The first (timeline, task) over the scripts of which `wanted(timeline, task)` holds."""
    return next((tl, task) for _, (_, tl) in scripts.items() for task in tl.ego.tasks if wanted(tl, task))


def ended_on_target_after(answer):
    def wanted(tl, task):
        kfs, _, t_place = frames_of(tl, task)
        place = next(d for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
        return place.response == answer and tl.placement_of(task.targets[2], t_place).on == task.targets[4]

    return wanted


@pytest.mark.parametrize("edit", ["thing never put down", "thing leaves early", "thing put on another room's piece",
                                  "no pieces in the targets", "tilt missing at the grasp", "response flipped",
                                  "thing put in the occupied window"])
def test_the_carry_checks_have_teeth(edit, scripts):
    """Contract: `job_helper_checks` flags a script whose carry is not helper/1's: a thing with no placement after its
    task's last placing keyframe or leaving before its last grasping keyframe (`job_thing`), one put on a piece of
    another room (`job_thing`), targets that name no pieces (`job_targets`), a grasp seen at the default pitch
    (`job_tilt`), a `before_place` response that the window does not bear out (`job_spot`), or an `elsewhere` thing put
    in the window (`job_thing`). Regression caught: a rule of the job that no script can break."""
    tl, task = pick(scripts, ended_on_target_after(ELSEWHERE if edit == "thing put in the occupied window" else GO)
                    if edit == "thing put in the occupied window" else lambda tl, task: True)
    eid = task.targets[2]
    kfs, t_grasp, t_place = frames_of(tl, task)
    old = next(p for p in tl.placements_of(eid) if p.t1 == pytest.approx(t_grasp))
    new = tl.placement_of(eid, t_place)
    placements, tasks = list(tl.placements), list(tl.ego.tasks)
    keyframes, decisions = list(tl.ego.keyframes), list(tl.ego.decisions)
    place_i = next(i for i, d in enumerate(decisions) if d.kind == "before_place" and task.k0 <= d.k < task.k1)
    code = "job_thing"
    if edit == "thing never put down":
        placements.remove(new)
    elif edit == "thing leaves early":
        placements[placements.index(old)] = replace(old, t1=old.t1 - 3.0)
    elif edit == "thing put on another room's piece":
        here = piece_id(tl, new.on)
        room = COUNTED[task.plan.params["regions"][1]]
        other = next(e.id for e in tl.entities
                     if e.mobility == "fixed" and e.kind in ("sofa", "coffee_table", "tv_console")
                     and piece_id(tl, e.id) != here and piece_id(tl, e.id) not in room)
        placements[placements.index(new)] = replace(new, on=other)
    elif edit == "no pieces in the targets":
        code = "job_targets"
        tasks[0 if task.index == 0 else task.index] = replace(task, targets=task.targets[:3])
    elif edit == "tilt missing at the grasp":
        code = "job_tilt"
        k = next(d.k for d in decisions if d.kind == "before_grasp" and task.k0 <= d.k < task.k1)
        keyframes[k] = replace(keyframes[k], cam_pitch=DEFAULT)
    elif edit == "response flipped":
        code = "job_spot"
        flipped = GO if decisions[place_i].response == ELSEWHERE else ELSEWHERE
        decisions[place_i] = replace(decisions[place_i], response=flipped)
    else:  # put in the occupied window
        window = surface_window(M, tl.keyframe(decisions[place_i].k).base[:2], tl.keyframe(decisions[place_i].k).yaw)[2]
        placements[placements.index(new)] = replace(new, pos=(window.cx, window.cy, new.pos[2]))
    ego = replace(tl.ego, keyframes=keyframes, tasks=tasks, decisions=decisions)
    edited = replace(tl, placements=placements, ego=ego)
    found = {p.code for p in job_helper_checks.checks(edited, M)}
    assert code in found, (edit, found)
    assert job_helper_checks.checks(tl, M) == []


def test_the_skip_rate_stays_under_a_tenth():
    """Contract (FD-0.5.0-control-impl §12): a seed whose script cannot be made (no thing for a source window, no place
    for it on the target) raises ValueError and is skipped by the build's rule, and few do: of seeds 101-112 on
    `apartment_a`, at most one raises, and the ones that do name the task and the reason. Regression caught: a thing or
    place rule so strict that crowded surfaces skip a fifth of the seeds (the first version of this lane did)."""
    skipped = []
    for seed in range(101, 113):
        try:
            tl = generate_script("apartment_a", seed, "helper", "helper/1",
                                 build.tasks_for((4, 6), "apartment_a", seed, "helper", "helper/1"))
        except ValueError as e:
            assert f"seed {seed}" in str(e) and "task" in str(e)
            skipped.append(seed)
        else:
            assert validate(tl, M) == []
    assert len(skipped) <= 1, skipped
