"""`helper/1`'s paths over `apartment_a` with the real clutter, gaps and resident: the scripts of seeds 101 and 102 are
built once for both helper modules (`tests/helper_kit.py`, one xdist group). Coordinates are the map's: the park pose
(5.40, 5.05) heading -90 degrees in the hall, the TV console F12 at (0.25, 4.2) facing east (its face at x 0.45), the
hall console F20 at (5.4, 6.0) facing south (its face at y 5.85). The helper is a 0.34 x 0.33 m rectangle keeping
0.10 m, so its planner keeps 0.2371 + 0.10 = 0.3371 m from every obstacle and a free cell's centre a cell's half
diagonal more, on a 0.025 m grid whose cell centres are at 0.0125 + 0.025 k."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import helper_kit
from stbench import build, job_helper, world_processes
from stbench import timeline as T
from stbench.generate import generate_script
from stbench.geometry import dist_to_rect
from stbench.job_helper_reach import free_space_for, reach_pose
from stbench.job_registry import get
from stbench.mapspec import load_map
from stbench.placement import occupied_entry
from stbench.route import robot_obstacles
from stbench.validate import validate

pytestmark = pytest.mark.xdist_group(name="helper")  # one worker builds the scripts once; both helper modules read them

SEEDS = (101, 102)
M = load_map("apartment_a")
JOB = get("helper/1")
REACH = math.hypot(0.34, 0.33) / 2 + 0.10  # how near the planner lets the base's centre come to an obstacle


@pytest.fixture(scope="module")
def scripts():
    """{seed: (the world each task was planned on, the script)}: 4-6 tasks, the real clutter and the resident."""
    return {seed: helper_kit.generated(seed, build.tasks_for((4, 6), "apartment_a", seed, "helper", "helper/1"))
            for seed in SEEDS}


def clearance_along(world, tl, task, t: float) -> float:
    """The least distance from the base to the walls and what the planner reads (`robot_obstacles` of the world the task
    was planned on, at `t`), over every point of every segment between consecutive keyframes of the task (the straight
    lines that are the path), 0.05 m apart."""
    kfs = tl.ego.keyframes[task.k0:task.k1]
    xs, ys = [], []
    for a, b in zip(kfs, kfs[1:], strict=False):
        n = max(math.ceil(math.dist(a.base[:2], b.base[:2]) / 0.05), 1)
        s = np.linspace(0.0, 1.0, n + 1)
        xs.append(a.base[0] + (b.base[0] - a.base[0]) * s)
        ys.append(a.base[1] + (b.base[1] - a.base[1]) * s)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    return min(float(dist_to_rect(xs, ys, r).min())
               for r in M.wall_pieces() + robot_obstacles(M, world, helper_kit.robot(), t))


def test_every_task_runs_validly(scripts):
    """Contract: on every script `validate` (the generic rules, the job's own, the resident's body rule) finds nothing,
    and the base keeps the planner's 0.3371 m (less a millimetre) from the walls and everything it reads
    (`robot_obstacles` of the world as the task's gap left it, door leaves in their state) at every point of every
    segment between keyframes, not only at keyframes. Regression caught: a path through furniture, a door leaf, a moved
    chair or a clutter piece, or a status outside the vocabulary."""
    for seed, (worlds, tl) in scripts.items():
        assert validate(tl, M) == [], seed
        assert len(worlds) == len(tl.ego.tasks) >= 4
        for task in tl.ego.tasks:
            near = clearance_along(worlds[task.index], tl, task, tl.keyframe(task.k0).t)
            assert near >= REACH - 0.001, (seed, task.index)


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
    positions, so no door's state changes a path (the robot's own path log would read it). On seed 101 the keyframes of
    three tasks are byte for byte those of the same script with every door off each task's ways toggled before it (the
    bathroom's and the bedroom's, set through the ledger, alternating). Regression caught: a detour round a shut door,
    which the path log would read as the door's state."""
    quiet = lambda w, m, seed: lambda lo, hi, keep_open: None  # noqa: E731
    with mock.patch.object(world_processes, "gaps", quiet):
        base = generate_script("apartment_a", 101, "helper", "helper/1", 3)
    with mock.patch.object(world_processes, "gaps", flipping(M)):
        flipped = generate_script("apartment_a", 101, "helper", "helper/1", 3)
    assert len({s.value for s in flipped.states}) == 2  # an open and a shut state both exist: the doors were toggled
    assert T.digest(base) != T.digest(flipped)  # the worlds differ (the door states) ...
    assert base.ego_hash == flipped.ego_hash  # ... and the paths do not


def test_decisions_come_from_the_plan_alone(scripts):
    """Contract (design §8.5): a fixed plan scripted over two different worlds (seeds 101 and 102, `plan` replaced by
    seed 101's) gives the same decision kinds, task by task, in the same order, and the same plans. Regression caught: a
    decision point the world triggered (a clutter piece, a chair, a shut door, a detour through the kitchen)."""
    _, home = scripts[101]
    swapped = replace(JOB, plan=lambda m, seed, n_tasks: JOB.plan(m, 101, n_tasks))
    with mock.patch.object(job_helper, "SPEC", swapped):
        other = generate_script("apartment_a", 102, "helper", "helper/1", 3)
    assert other.world_hash != home.world_hash  # another world
    first = lambda tl: [(tl.task_of(d.k).index, d.kind) for d in tl.ego.decisions if tl.task_of(d.k).index < 3]  # noqa: E731
    # four decisions a task: two doorways, a grasp and a place
    assert first(other) == first(home) and len(first(home)) == 12
    assert [t.plan for t in other.ego.tasks] == [t.plan for t in home.ego.tasks[:3]]


def test_each_task_starts_and_ends_parked(scripts):
    """Contract: every task's first and last keyframe stand at the park pose (5.40, 5.05) heading -90 degrees and are
    `waiting`, its battery starts in 80-100 and never rises, and the task index counts from 0 in order: nothing carried
    over from an earlier task shows where a window sits. In every world a task was planned on (the start, and after up
    to five gaps' light furniture moves, floor clutter and carried things) nothing the planner reads lies within its
    0.3371 m of the park pose. Regression caught: state carried over from the task before, a battery that keeps
    draining across a gap, or a park pose the world blocks."""
    for _, (worlds, tl) in scripts.items():
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            for kf in (kfs[0], kfs[-1]):
                assert (kf.phase, kf.base, kf.yaw, kf.task) == ("waiting", (5.4, 5.05, 0.0), -1.570796, task.index)
            assert 80 <= kfs[0].status["battery_pct"] <= 100
            batteries = [kf.status["battery_pct"] for kf in kfs]
            assert batteries == sorted(batteries, reverse=True) and batteries[0] - batteries[-1] <= 5
            near = [float(dist_to_rect(np.array([5.4]), np.array([5.05]), r)[0])
                    for r in robot_obstacles(M, worlds[task.index], helper_kit.robot(), kfs[0].t)]
            assert min(near) >= REACH, (task.index, min(near))


def test_a_reach_pose_faces_its_piece():
    """Contract: on the map's own floor the reach pose at the TV console (face at x 0.45, 4.2 m up the west wall) is the
    free cell nearest the ideal pose, on its front axis with the front edge 0.3 m from the face, x 0.45 + 0.3 + 0.17 =
    0.92, y 4.2: the first cell centre at or beyond x 0.92 is 0.9375, and the cells 0.0125 below and above y 4.2 tie,
    the first in row order (4.1875) winning, facing the face (heading 180 degrees); at the hall console (face at y
    5.85) it is y 5.3625 (5.85 - 5.3625 = 0.4875 is the first cell at 0.47 or more) and x 5.3875 (of 5.3875 and 5.4125),
    heading +90. With a 0.3 m box put on the TV console's ideal pose, the pose slides along the face to the first cell
    the planner keeps 0.3371 + 0.0177 m from the box: 0.15 + 0.3548 = 0.5048, so the cell 0.5125 up or down (down, in
    row order), the distance unchanged from the face (x 0.9375), still facing it. Regression caught: a robot that
    grasps from across the room, from a corner it does not face, a pose the world's clutter silently overlaps, or a
    tie that the last bit of a float breaks differently on another platform."""
    w, free = helper_kit.bare_free_space()
    mask, robot = helper_kit.reachable_mask(free), helper_kit.robot()
    tv, console = M.furniture_by_id()["F12"], M.furniture_by_id()["F20"]
    (x, y), yaw = reach_pose(M, free, mask, tv, robot)
    assert (x, y, yaw) == (0.9375, 4.1875, 3.141593)  # the cells 4.1875 and 4.2125 tie: the first in row order
    (x, y), yaw = reach_pose(M, free, mask, console, robot)
    assert (x, y, yaw) == (5.3875, 5.3625, 1.570796)  # x 5.3875 and 5.4125 tie: the first in row order
    w.clutter["box"] = occupied_entry(SimpleNamespace(size=(0.3, 0.3, 0.2)), (0.92, 4.2, 0.0), 0.0)
    free = free_space_for(M, w, robot, frozenset({"R1", "R2"}))
    (x, y), yaw = reach_pose(M, free, free.reachable_from(helper_kit.PARK), tv, robot)
    assert (x, y, yaw) == (0.9375, 3.6875, 3.141593)  # up and down tie at 0.5125: the first in row order


def test_the_world_moves_where_a_reach_falls_never_whether_the_task_exists():
    """Contract (M2-A2's rule): when the piece a task prefers has no reach pose in the world, the helper reaches the
    room's next counted piece in id order (never another room); when none has one, the script raises naming the job,
    the map, the seed, the task and the room's pieces. Regression caught: a task the world silently drops or moves to
    another room, or an error that does not say which task and piece cannot be reached."""
    no_pose = lambda m, free, mask, piece, robot: None if piece.id == "F12" else ((1.0, 2.0), 0.5)  # noqa: E731
    with mock.patch.object(job_helper, "reach_pose", no_pose):
        assert job_helper._spot(M, None, None, helper_kit.robot(), "R2", "F12", "source") == ((1.0, 2.0), 0.5)
    blocked = lambda m, free, mask, piece, robot: None  # noqa: E731
    with mock.patch.object(job_helper, "reach_pose", blocked), pytest.raises(
            ValueError, match=r"helper/1 on apartment_a, seed 101, task 0 \((tidy|fetch) from the (hall|living room)\)"
                              r".*no reach pose at the source"):
        generate_script("apartment_a", 101, "helper", "helper/1", 1)
    with mock.patch.object(job_helper, "reach_pose", blocked), pytest.raises(ValueError, match="F12, F10, F11"):
        job_helper._spot(M, None, None, helper_kit.robot(), "R2", "F12", "source")
