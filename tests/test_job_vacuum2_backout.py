"""`vacuum/2`'s back-outs (`job_vacuum2_backout`), switched on for these tests by `SITE_KINDS`: the feasibility gate
failed on `apartment_a` (docs/jobs.md, "Back-outs"), so no shipped plan carries one, and what is tested is the
mechanism the brief specifies: the approach lines by hand on the coffee table, a back-out in a script of the plan, the
reverse along the way in, the corridor behind, none in a resumed room and the error when no line fits. Seeds 101, 104
and 105."""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest

import job_kit
from stbench import grid, job_vacuum2, job_vacuum2_doorway
from stbench import job_vacuum2_backout as backout
from stbench.generate import build_world
from stbench.geometry import Rect, dist_to_rect
from stbench.mapspec import load_map
from stbench.robots import ROBOTS
from stbench.route import robot_free_space, robot_obstacles
from stbench.timeline_ego import TaskPlan
from stbench.validate import validate

pytestmark = pytest.mark.xdist_group(name="test_job_vacuum2")  # shares the worker, and the cache, of the job tests

SITES = ("dining_table", "coffee_table")
SEEDS = (101, 104, 105)
ROBOT = ROBOTS["vacuum"]
M = load_map("apartment_a")


@pytest.fixture(autouse=True)
def sites():
    """The back-outs on, as the brief builds them, for the test (and for `validate`, which reads the same table)."""
    with mock.patch.object(backout, "SITE_KINDS", SITES):
        yield


def bare_world():
    """Seed 1's world with nothing on its floor: no carried item (a bag or a shoe lies in the coffee table's way in
    some) and no clutter, the light furniture at its map places."""
    w = build_world(M, 1)
    w.carried.clear()
    return w


def backouts(tl):
    return [d for d in tl.ego.decisions if d.kind == "before_backout"]


def on_line(kf, far, u) -> bool:
    """Whether the keyframe's base lies on the line through the far point along the unit vector `u`."""
    return abs((kf.base[0] - far.base[0]) * u[1] - (kf.base[1] - far.base[1]) * u[0]) < 1e-6


def test_the_first_approach_line_is_the_coffee_tables_short_axis_from_the_south():
    """Contract (brief §3, the manoeuvre): the approach lines are tried in the order the piece's axes, both ways, the
    offsets 0, +-0.1, +-0.2 across, each at the approach distances; the approach point stands that far outside the
    footprint's edge and the far point is `BACKOUT_MIN_M` to the footprint's middle in. Hand case on the coffee table
    (centre (2.4, 4.4), 0.6 x 1.1 turned a quarter, so 0.6 m deep north to south, its edge at y 4.1): the first line
    runs north through the centre, the approach point 0.25 m outside the edge at (2.4, 3.85); a drive-in of depth 0
    ends 0.3 m on at (2.4, 4.15), of depth 1 at the middle, 0.55 m on, (2.4, 4.4), and of depth 0.5 at 0.425 m; there
    are 20 lines, the second offset 0.1 to the side. Regression caught: a line that starts anywhere, a far point past
    the middle or short of the least, or a different order of lines."""
    w = bare_world()
    piece = next(f for f in M.furniture if f.kind == "coffee_table")
    free = robot_free_space(M, w, ROBOT, 0.0, shut=["D3", "D4"])  # the living room: D2, or D1 and its arch
    lines = backout.lines(piece)
    assert len(lines) == 20 and lines[0][1:] == (math.pi / 2, 0.3) and lines[0][0] == pytest.approx((2.4, 4.4))
    assert lines[1][0] == pytest.approx((2.3, 4.4)) and lines[2][0] == pytest.approx((2.5, 4.4))
    for depth, far in ((0.0, 4.15), (0.5, 4.275), (1.0, 4.4)):
        got = backout.plan(free, piece, (5.44, 5.05), depth)
        assert got.yaw == pytest.approx(math.pi / 2)
        assert got.approach == pytest.approx((2.4, 3.85)) and got.far == pytest.approx((2.4, far))


def test_backouts_follow_the_plan():
    """Contract (brief §3): a `clean_rooms` task's cleaning of a site room (here the kitchen's dining table and the
    living room's coffee table) begins with exactly one back-out, a `before_backout` decision in transit between the
    doorway decision and the room's `room_done` or `cleaning_stopped`, and a script of them is valid under `validate`
    (the job's check reads the same table of sites). Regression caught: a back-out the world removes, two in a room,
    or one in a hall or bedroom."""
    found = 0
    for seed in SEEDS:
        _, tl = job_kit.generated2(seed, SITES, job_kit.never)
        assert validate(tl, M) == [], seed
        for task in tl.ego.tasks:
            kinds = [d.kind for d in tl.decisions_in(task.k0, task.k1)]
            want = sum(kind in ("kitchen", "living room") for kind in task.plan.params["kinds"])
            assert kinds.count("before_backout") == (want if task.plan.kind == "clean_rooms" else 0), (seed, task.index)
            for d in tl.decisions_in(task.k0, task.k1):
                assert (tl.keyframe(d.k).phase == "transit") == (d.kind.startswith("before_")), d
            for i, kind in enumerate(kinds):
                if kind == "before_backout":
                    assert kinds[i - 1] in ("before_room", "before_doorway")
                    assert kinds[i + 1] in ("room_done", "cleaning_stopped")
            found += kinds.count("before_backout")
    assert found >= 5


def test_backouts_reverse_along_their_way_in():
    """Contract (brief §3): the robot drives in along the approach line, faces in at the far point (the decision), and
    reverses to where it turned onto the line with its heading kept, motion opposite, as docking reverses: the
    keyframes after the decision lie on the line, behind the far point, on the same heading, ending at the approach
    point; the drive-in is at least 0.3 m and ends no farther than the footprint's middle. Hand check: the line is the
    one through the far point along its heading. Regression caught: a reverse that swerves, turns amid the legs, or
    ends short of the way in."""
    n = 0
    for seed in SEEDS:
        _, tl = job_kit.generated2(seed, SITES, job_kit.never)
        for d in backouts(tl):
            far, kfs = tl.keyframe(d.k), tl.ego.keyframes
            u = (math.cos(far.yaw), math.sin(far.yaw))
            start = kfs[d.k - 1]  # the drive-in's start: the keyframe a turn onto the line ended with
            assert on_line(start, far, u) and abs(start.yaw - far.yaw) < 1e-6 and start.phase == "transit"
            approach = start.base[:2]
            assert math.dist(approach, far.base[:2]) >= 0.3 - 1e-6
            for kf in kfs[d.k + 1:]:  # the reverse: on the line, behind the far point, the heading kept ...
                if math.dist(kf.base[:2], approach) < 1e-6:
                    break  # ... until it is back at the approach point, where the turn away from the line is made
                assert on_line(kf, far, u) and abs(kf.yaw - far.yaw) < 1e-6 and kf.phase == "transit"
                assert (kf.base[0] - far.base[0]) * u[0] + (kf.base[1] - far.base[1]) * u[1] < 0
            else:
                pytest.fail(f"seed {seed}: the reverse never reaches the approach point {approach}")
            piece = next(f for f in M.furniture if f.kind in SITES and math.dist(f.center, far.base[:2]) < 1.0)
            assert (piece.center[0] - far.base[0]) * u[0] + (piece.center[1] - far.base[1]) * u[1] >= -1e-6
            n += 1
    assert n >= 5


def corridor_beyond_the_drive_in(worlds, tl, d) -> float | None:
    """The share of the cells of the corridor behind the far point (the body's width, 3 m, from its rear edge) that
    lie before the first thing in it (a wall, what the planner reads) and beyond the reach of the drive-in (farther
    back than the approach point's body), which the base swept before the decision; None when there are none."""
    task, far = tl.task_of(d.k), tl.keyframe(d.k)
    drive = math.dist(tl.keyframe(d.k - 1).base[:2], far.base[:2])
    back = far.yaw + math.pi
    rear = (far.base[0] + math.cos(back) * 0.17, far.base[1] + math.sin(back) * 0.17)
    things = M.wall_pieces() + robot_obstacles(M, worlds[task.index], ROBOT, far.t)
    cells = grid.corridor(rear, back, ROBOT.body.width_m, 3.0)
    centres = np.array([((u + 0.5) * 0.1, (v + 0.5) * 0.1) for u, v, _ in cells])
    inside = np.zeros(len(cells), dtype=bool)
    for r in things:
        inside |= dist_to_rect(centres[:, 0], centres[:, 1], r) <= 0
    stops = [dist for (_, _, dist), hit in zip(cells, inside, strict=True) if hit]
    beyond = [(u, v) for u, v, dist in cells if drive + 0.17 < dist < min(stops, default=math.inf)]
    kfs = tl.ego.keyframes[task.k0:d.k + 1]
    poses = [(kf.base[0], kf.base[1], kf.yaw) for kf in kfs]
    swept = grid.swath(poses, "disc", ROBOT.body.length_m, ROBOT.body.width_m)
    return sum(c in swept for c in beyond) / len(beyond) if beyond else None


def test_the_reverse_corridor_beyond_the_drive_in_is_not_swept():
    """Contract (brief §3): the floor behind a back-out is not a stretch the robot has just swept, or its own path log
    would answer `rear_clear`: over the back-outs of the scripts that have corridor cells before the first thing in it
    and beyond the reach of the drive-in (the cells `rear_clear` counts, less the drive-in's own 0.3 to 0.5 m), the
    median share the base swept before the decision is under 0.1 (it was 0 of 44 back-outs on seeds 101-119 but a few
    where the arrival crossed the corridor). The corridors are short, about 0.6 m, so the drive-in is most of them: the
    gate's whole-corridor share failed (docs/jobs.md). Regression caught: an approach that drives down its own
    corridor."""
    shares = []
    for seed in SEEDS:
        worlds, tl = job_kit.generated2(seed, SITES, job_kit.never)
        shares += [x for d in backouts(tl) if (x := corridor_beyond_the_drive_in(worlds, tl, d)) is not None]
    assert len(shares) >= 3 and float(np.median(shares)) < 0.1


def test_no_backout_in_a_resumed_room():
    """Contract (brief §3): the stopped task has swept the corridor, so a resumed room has no back-out while the stopped
    task, which began its room with one, has. Hand check on seed 104: task 0 cleans the kitchen (a site room) and is
    sent home, task 1 resumes it; the first holds a `before_backout`, the second none. Regression caught: a resumed
    room that backs out again, into a corridor its own earlier sweep answers."""
    _, tl = job_kit.generated2(104, SITES, job_kit.never)
    first, second = tl.ego.tasks[0], tl.ego.tasks[1]
    assert (first.plan.kind, first.plan.params["kinds"], second.plan.kind) == ("clean_rooms", ["kitchen"], "resume")
    assert [d.kind for d in tl.decisions_in(first.k0, first.k1)] == [
        "before_doorway", "before_backout", "cleaning_stopped", "before_doorway"]
    assert [d.kind for d in tl.decisions_in(second.k0, second.k1)] == ["before_doorway", "room_done", "before_doorway"]


def test_a_backout_that_cannot_fit_raises():
    """Contract (brief §3): the world moves where a back-out falls, never whether it exists, so a room whose every
    approach point is blocked is an error naming the seed, the task and the room, never a task without its back-out.
    Hand case: with one line each way (offset 0, one approach distance) the coffee table's four approach points are
    (2.4, 3.85) south of it, (1.6, 4.4) and (3.2, 4.4) at its ends and (2.4, 4.95) north; a 0.4 m box 0.3 m tall on each
    leaves none, and with the north one left clear a back-out exists (the two at the ends are no way in anyway: the legs
    of a short end leave 0.3 m, under the 0.38 m the body needs). Regression caught: a task written without its
    back-out, or an error that does not say where."""
    plans = [TaskPlan("clean_rooms", {"rooms": "living room", "kinds": ["living room"]})]
    points = [(2.4, 3.85), (1.6, 4.4), (3.2, 4.4), (2.4, 4.95)]  # south, west, east, north
    with mock.patch.object(backout, "OFFSETS_M", (0.0,)), mock.patch.object(backout, "APPROACHES_M", (0.25,)):
        for blocked, raises in ((points, True), (points[:3], False)):
            w = bare_world()
            w.clutter |= {f"B{i}": (Rect(x, y, 0.4, 0.4, 0.0), 0.0, 0.3) for i, (x, y) in enumerate(blocked)}
            if raises:
                with pytest.raises(ValueError, match=r"seed 7, task 0 \(living room\).*coffee table.*living room"):
                    job_vacuum2._script(M, w, plans, 7)
            else:
                with mock.patch.object(job_vacuum2_doorway, "doorway_blocked", job_kit.never):
                    tl_ego = job_vacuum2._script(M, w, plans, 7)
                assert [d.kind for d in tl_ego.decisions].count("before_backout") == 1
