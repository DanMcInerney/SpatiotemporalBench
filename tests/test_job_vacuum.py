"""`vacuum/1`'s plan, script and checks, over `apartment_a` with stand-in clutter on the floor (`job_kit`). Coordinates
are the map's (`maps/apartment_a.json`): the dock `F23` at (4.71, 5.05), 0.12 m deep, facing east; the vacuum a 0.17 m
disc keeping 0.02 m, so its docked pose is x = 4.71 + 0.06 + 0.17 = 4.94."""

from __future__ import annotations

import math
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

import job_kit
from stbench import generate, job_vacuum, job_vacuum_path
from stbench.catalog import CATALOGS
from stbench.episode import pose_rows
from stbench.events import resolve
from stbench.generate import generate_script
from stbench.geometry import Rect, dist_to_rect, nearest_on_rect, segments_cross, under_obstacles, wrap_angle
from stbench.job_registry import get
from stbench.keyframes import Mark, Move, Turn
from stbench.route import robot_obstacles
from stbench.validate import validate

DOCKED = (4.94, 5.05, 0.0)  # (x, y, yaw), by hand
OUT, FRONT = (5.44, 5.05), (5.34, 5.05)  # 0.5 m off the dock; the point 0.4 m in front of the docked pose
PHASES = ["docked", "undocking", "transit", "cleaning", "transit", "docking", "docked"]
SEEDS = [(room, seed) for room in job_kit.ROOMS for seed in job_kit.seeds_for(room)]


def phase_runs(tl) -> list[str]:
    kfs = tl.ego.keyframes
    return [kf.phase for i, kf in enumerate(kfs) if i == 0 or kf.phase != kfs[i - 1].phase]


def clearance_along(tl, pieces: list[Rect], step: float = 0.05) -> float:
    """The least distance from the base to `pieces`, over every point of every segment between consecutive keyframes
    (the straight lines that are the path), sampled every `step` m."""
    least = math.inf
    for a, b in zip(tl.ego.keyframes, tl.ego.keyframes[1:], strict=False):
        n = max(math.ceil(math.dist(a.base[:2], b.base[:2]) / step), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        xs, ys = a.base[0] + (b.base[0] - a.base[0]) * t, a.base[1] + (b.base[1] - a.base[1]) * t
        least = min(least, *(float(dist_to_rect(xs, ys, r).min()) for r in pieces))
    return least


def test_the_docked_pose_is_the_discs_touching_the_dock_facing_out():
    """Contract (plan §4.3): the docked vacuum's disc touches the dock's front face, facing out. Hand case: the dock's
    front face is at x = 4.71 + 0.06 = 4.77, the disc's radius 0.17, so the base stands at (4.94, 5.05), heading 0
    (east, the dock's own facing). Regression caught: a docked pose inside the dock or short of it."""
    assert job_vacuum.docked_pose(job_kit.apartment()) == DOCKED


def test_the_plan_draws_a_room_other_than_the_docks_and_one_task_only():
    """Contract (DECISIONS, "M1's calls"): a task cleans a room drawn from the regions other than the dock's own (the
    hall); its plan is `clean_room` with the room's kind, its target the region's id; M1 writes one task a script.
    Regression caught: a hall task, which crosses no doorway and asks nothing during the job, or a plan of two tasks
    that M2's gaps do not yet support."""
    m, job = job_kit.apartment(), get("vacuum/1")
    rooms = {job.plan(m, seed, 1)[0].params["room"] for seed in range(1, 60)}
    assert rooms == {"kitchen", "living room", "bedroom", "bathroom"}
    plan = job.plan(m, 1, 1)[0]
    assert (plan.kind, set(plan.params)) == ("clean_room", {"room"})
    ids = {"living room": "R2", "kitchen": "R3", "bedroom": "R4", "bathroom": "R5"}  # the map's regions
    assert job_kit.script(1).ego.tasks[0].targets == (ids[plan.params["room"]],)
    with pytest.raises(ValueError, match="M2"):
        job.plan(m, 1, 2)


@pytest.mark.parametrize(("room", "seed"), SEEDS)
def test_every_room_cleans_validly(room, seed):
    """Contract: each target room's script, with clutter on the floor, is valid under `validate` (the generic rules and
    the job's own), runs the phases docked, undocking, transit, cleaning, transit, docking, docked, and its base keeps
    the body's clearance (0.17 m and 0.02 m) from the walls and everything the planner reads (`robot_obstacles`) at
    every point of every segment between keyframes, not only at keyframes, the dock aside (the robot leaves and reaches
    it). Regression caught: a path through furniture, a leg or a piece of clutter, or a status outside the
    vocabulary."""
    world, tl = job_kit.generated(seed)
    m = job_kit.apartment()
    assert validate(tl, m) == []
    assert phase_runs(tl) == PHASES
    dock = Rect(4.71, 5.05, 0.12, 0.28, 0.0)
    pieces = [r for r in robot_obstacles(m, world, job_kit.robot()) if r != dock] + m.wall_pieces()
    assert clearance_along(tl, pieces) >= 0.19 - 0.001
    assert clearance_along(tl, [dock]) >= 0.17 - 0.001  # it touches the dock and never goes into it


def test_paths_pass_under_tables_not_through_legs():
    """Contract (plan §2.1, §4.3): the vacuum cleans under the tables it fits under (an underside of 0.10 m or more) and
    never through a leg, nor under the sofa (0.08 m) or the bed (0.05 m). Hand cases: some keyframe's base is inside the
    dining table's footprint (x 1.4 to 2.8, y 1.5 to 2.3) and outside its four leg squares in every kitchen script, and
    inside the desk's (x 8.35 to 8.95, y 3.25 to 4.45) likewise in a bedroom script whose chair is pulled out 0.5 m (a
    chair at the desk closes its opening to the vacuum, as in a home; a 0.6 m pull would put it within 0.05 m of D4's
    open leaf, which a start refuses) and whose knee space is open: with no clutter on the floor (the real clutter,
    almost always, leaves something above the 0.02 m the vacuum climbs in the desk's footprint: a power strip, cables,
    a heater, a fan, dumbbells, a heap of clothes, a drying rack, so the vacuum cannot get in, as in a home) two of the
    first six bedroom seeds leave it open. At every keyframe of the first six bedroom
    seeds with the chair pulled out, and of the kitchen's and living room's, the disc (0.17 m) touches no leg square of
    either, the sofa's footprint (x 1.4 to 3.4, y 5.25 to 6.15) or the bed's (x 6.7 to 8.3, y 4.15 to 6.15). Regression
    caught: the under-furniture rule ignored (a path through a leg, or a sofa the vacuum cleans under)."""
    m = job_kit.apartment()
    table, desk = m.furniture_by_id()["F07"], m.furniture_by_id()["F33"]
    sofa, bed = m.furniture_by_id()["F10"].footprint(), m.furniture_by_id()["F30"].footprint()
    legs = [leg for f in (table, desk) for leg in under_obstacles(f.footprint(), f.under_m)]
    assert len(legs) == 8

    def inside(rect: Rect, base) -> bool:
        return float(dist_to_rect(np.array([base[0]]), np.array([base[1]]), rect)[0]) == 0.0

    def under(piece, tl) -> bool:
        return any(inside(piece.footprint(), kf.base) and not any(inside(leg, kf.base) for leg in legs)
                   for kf in tl.ego.keyframes)

    assert all(under(table, job_kit.script(seed)) for seed in job_kit.seeds_for("kitchen"))
    home = CATALOGS["home"]
    bare = replace(home, clutter=tuple(k for k in home.clutter if not k.on_floor))  # nothing on the floor
    with mock.patch.object(generate, "CHAIR_PULL_M", (0.5, 0.5)), mock.patch.object(generate, "CHAIR_YAW_DEG", 0.0), \
            mock.patch.dict(CATALOGS, {"home": bare}):
        pulled = [generate_script(job_kit.MAP, seed, job_kit.ROBOT, job_kit.JOB, 1)
                  for seed in job_kit.seeds_for("bedroom", 6)]
    assert sum(under(desk, tl) for tl in pulled) == 2
    others = [job_kit.script(seed) for room in ("kitchen", "living room") for seed in job_kit.seeds_for(room)]
    for tl in [*pulled, *others]:
        for kf in tl.ego.keyframes:
            assert all(nearest_on_rect(r, kf.base[0], kf.base[1])[1] >= 0.17 - 0.001 for r in [*legs, sofa, bed]), kf.k
            assert not any(inside(r, kf.base) for r in (sofa, bed)), kf.k


@pytest.mark.parametrize(("room", "seed"), SEEDS[::3])
def test_decisions_follow_the_phase_plan(room, seed):
    """Contract (design §8.5): a task holds a before_doorway in transit, room_done on its last cleaning keyframe and a
    before_doorway in transit, in that order, from the phase plan alone: stand-in clutter leaves kinds, order and phases
    unchanged. Each before_doorway keyframe faces the room's doorway, within 0.5 m of it. Regression caught: a decision
    triggered by the world, or one that does not look at the door it is about."""
    with_clutter, without = job_kit.script(seed), job_kit.script(seed, clutter_on=False)
    for tl in (with_clutter, without):
        kfs, decisions = tl.ego.keyframes, tl.ego.decisions
        assert [d.kind for d in decisions] == ["before_doorway", "room_done", "before_doorway"]
        assert [kfs[d.k].phase for d in decisions] == ["transit", "cleaning", "transit"]
        assert decisions[1].k == max(kf.k for kf in kfs if kf.phase == "cleaning")
        m = job_kit.apartment()
        room_id = tl.ego.tasks[0].targets[0]
        doors = [m.door_line(o) for o in m.openings if room_id in o.between]
        for d in (decisions[0], decisions[2]):
            kf = kfs[d.k]
            ahead = (kf.base[0] + 0.55 * math.cos(kf.yaw), kf.base[1] + 0.55 * math.sin(kf.yaw))
            assert any(segments_cross(kf.base[:2], ahead, a, b) for a, b in doors), (room, d)
    assert [d.kind for d in with_clutter.ego.decisions] == [d.kind for d in without.ego.decisions]


def test_a_decision_sits_half_a_metre_before_the_doorway_or_at_the_start_of_its_segment():
    """Contract (plan §4.3): `before_doorway` is on the segment that crosses the room's opening, `DOORWAY_STANDOFF_M`
    (0.5 m) before the crossing along it, or at the segment's start (after its turn) when that is closer; on the way
    in it is the last crossing into the room and on the way out the first crossing out of it. Hand cases on the
    kitchen's door D1, whose line is x = 4.6 from y 1.475 to 2.325: the path (5.4, 1.9) to (3.9, 1.9) crosses at 0.8 m,
    so the mark is 0.3 m along; from (4.9, 1.9) it crosses at 0.3 m, closer than 0.5, so the mark is at the start; a
    path in and out again, (5.4, 1.9) to (3.9, 1.9) to (5.4, 1.9), crosses at 0.8 m on segment 0 and 0.7 m on segment
    1: the last is (1, 0.2), the first (0, 0.3). The driver puts the move, the mark and the rest of the move, and a
    turn first when the mark is at a start. Regression caught: a mark at the wrong end of the crossing, or before the
    turn it should follow."""
    m = job_kit.apartment()
    kitchen = m.region("R3")
    cross = job_vacuum_path.crossing
    assert cross(m, kitchen, [(5.4, 1.9), (3.9, 1.9)], 0.5, last=True) == pytest.approx((0, 0.3))
    assert cross(m, kitchen, [(4.9, 1.9), (3.9, 1.9)], 0.5, last=True) == (0, 0.0)
    both = [(5.4, 1.9), (3.9, 1.9), (5.4, 1.9)]
    assert cross(m, kitchen, both, 0.5, last=True) == pytest.approx((1, 0.2))
    assert cross(m, kitchen, both, 0.5, last=False) == pytest.approx((0, 0.3))
    with pytest.raises(ValueError, match="kitchen"):
        cross(m, kitchen, [(5.4, 4.0), (5.4, 5.0)], 0.5, last=True)
    drive = job_vacuum_path.Driver(0.0, 45.0)
    drive.go([(0.0, 0.0), (2.0, 0.0)], 0.5, "p", (0, 0.5, "d"))
    assert drive.prims == [Move((0, 0), (0.5, 0.0), 0.0, 0.5, "p"), Mark("d"),
                           Move((0.5, 0.0), (2.0, 0), 0.0, 0.5, "p")]
    drive = job_vacuum_path.Driver(0.0, 45.0)
    drive.go([(0.0, 0.0), (0.0, 1.0)], 0.5, "p", (0, 0.0, "d"))
    assert drive.prims == [Turn((0, 0), 0.0, math.pi / 2, 45.0, "p"), Mark("d"),
                           Move((0, 0), (0, 1.0), math.pi / 2, 0.5, "p")]


@pytest.mark.parametrize(("room", "seed"), SEEDS)
def test_each_task_starts_docked_in_the_start_set(room, seed):
    """Contract: a task's first keyframe is `docked` at the docked pose (4.94, 5.05) heading 0, battery in 90-100, brush
    off, and its last is docked there too: nothing carried over from an earlier task shows where a window sits.
    Regression caught: a script whose state starts anywhere, which a window's first step would reveal."""
    kfs = job_kit.script(seed).ego.keyframes
    for kf in (kfs[0], kfs[-1]):
        assert (kf.phase, kf.base, kf.yaw) == ("docked", (DOCKED[0], DOCKED[1], 0.0), 0.0)
    assert 90 <= kfs[0].status["battery_pct"] <= 100 and kfs[0].status["brush"] is False


def test_the_battery_drains_a_roombas_pace_and_the_brush_is_on_only_while_cleaning():
    """Contract (plan §4.3): the battery falls 1.1% a minute (a Roomba's 90-minute run), a whole number; the brush is on
    exactly in `cleaning`. Hand cases: no drain before 55 s (1.1 x 55 / 60 = 1.008 is the first whole one: the last
    keyframe before 54.5 s holds the start value, the first from 55 s one less) and two by 110 s. Regression caught: a
    battery that does not drain, or drains at another pace, or a brush running in transit."""
    kfs = job_kit.script(job_kit.seeds_for("kitchen")[0]).ego.keyframes
    start = kfs[0].status["battery_pct"]
    assert all(isinstance(kf.status["battery_pct"], int) for kf in kfs)
    assert {kf.status["battery_pct"] for kf in kfs if kf.t < 54.5} == {start}
    assert {kf.status["battery_pct"] for kf in kfs if 55 <= kf.t < 109} == {start - 1}
    assert {kf.status["battery_pct"] for kf in kfs if 110 <= kf.t < 163} == {start - 2}
    assert [kf.status["brush"] for kf in kfs] == [kf.phase == "cleaning" for kf in kfs]


def test_each_phase_runs_at_one_speed_drawn_from_the_robots_range():
    """Contract (plan §4.3): a task draws one speed a phase, within the robot's range (the vacuum's table: cleaning 0.28
    to 0.31 m/s, transit 0.30 to 0.35); the keyframes are on straight runs at that speed. Hand case: two keyframes with
    the base moving straight between them (no turn) in cleaning are 0.28 to 0.31 m apart a second, all of one task at
    one speed to a hundredth. Regression caught: a speed outside the robot's range or one that wanders in a phase."""
    kfs = job_kit.script(job_kit.seeds_for("kitchen")[0]).ego.keyframes
    speeds = {"cleaning": set(), "transit": set()}
    for a, b in zip(kfs, kfs[1:], strict=False):
        d = math.dist(a.base[:2], b.base[:2])
        if a.phase == b.phase in speeds and d > 0.5 and abs(wrap_angle(b.yaw - a.yaw)) < 1e-6:
            speeds[a.phase].add(round(d / (b.t - a.t), 2))
    assert speeds["cleaning"] and speeds["transit"]
    assert all(0.28 <= v <= 0.31 for v in speeds["cleaning"]) and all(0.30 <= v <= 0.35 for v in speeds["transit"])
    assert max(speeds["cleaning"]) - min(speeds["cleaning"]) <= 0.01


@pytest.mark.parametrize(("room", "seed"), SEEDS)
def test_frames_follow_the_policy(room, seed):
    """Contract (plan §4.3, map X-H3): a task's keyframes are at least 1.0 s apart (so no two share a clock second), no
    gap between two is more than 1.0 m, 45 degrees or 5 s, and the end of each phase is a keyframe: the base ends
    undocking 0.5 m off the dock at (5.44, 5.05), and transit at (5.34, 5.05), 0.4 m in front of it. Regression caught:
    clocks that collide at whole seconds, or stretches of the path with no frame."""
    tl = job_kit.script(seed)
    kfs = tl.ego.keyframes
    for a, b in zip(kfs, kfs[1:], strict=False):
        assert b.t - a.t >= 1.0 - 1e-6
        assert math.dist(a.base[:2], b.base[:2]) <= 1.0 + 3e-6
        assert abs(wrap_angle(b.yaw - a.yaw)) <= math.radians(45) + 3e-6
        assert b.t - a.t <= 5.0 + 3e-6
    assert len({row["clock"] for row in pose_rows(tl)}) == len(kfs)
    assert [kf.base[:2] for kf in kfs if kf.phase == "undocking"][-1] == OUT
    ends_transit = [kf.base[:2] for i, kf in enumerate(kfs) if kf.phase == "transit" and kfs[i + 1].phase == "docking"]
    assert ends_transit == [FRONT]


@pytest.mark.parametrize(("room", "seed"), SEEDS[::2])
def test_every_event_resolves(room, seed):
    """Contract (plan §2.2, M1's question 2): each of the vacuum's four events resolves, through `events.resolve` over
    the pose rows a system is sent (their `k` and status), to the keyframe the plan names: the last undocking one (0.5 m
    off the dock), the first cleaning one, the room_done decision's (the last cleaning one) and the final docked one,
    the first docked keyframe after the docking ones. Regression caught: an event no system can resolve from what it is
    sent."""
    tl = job_kit.script(seed)
    kfs = tl.ego.keyframes
    steps = [{"k": row["k"], "status": row["status"]} for row in pose_rows(tl)]
    want = {
        "when you left the dock": max(kf.k for kf in kfs if kf.phase == "undocking"),
        "when you started cleaning": min(kf.k for kf in kfs if kf.phase == "cleaning"),
        "when you finished cleaning": tl.ego.decisions[1].k,
        "when you got back to the dock": len(kfs) - 1,
    }
    got = {wording: resolve(rule, steps) for wording, rule in get("vacuum/1").events.items()}
    assert got == want


def test_a_one_room_task_has_a_frame_count_near_the_designs():
    """Contract (design §10: about 120 images a short case; plan §4.3 asks 60 to 200 a room): a one-room task is neither
    a handful of frames nor a flood. Measured over the first six seeds of each room with stand-in clutter: bedrooms 57
    to 105, bathrooms 61 to 107, kitchens 151 to 200, living rooms 213 to 349 (the room is 15 m2: its 6 m2 of free
    floor is 30 m of strips, 17 m of edge pass and a U-turn of four frames every strip), so each lies in 50 to 360 and
    the mean is 80 to 200. Regression caught: a frame policy far from the design's (a frame a decimetre, or one a
    minute)."""
    counts = [len(job_kit.script(seed).ego.keyframes) for _, seed in SEEDS]
    assert all(50 <= n <= 360 for n in counts), counts
    assert 80 <= sum(counts) / len(counts) <= 200


@pytest.mark.parametrize(("edit", "code"), [
    ("brush in transit", "job_brush"),
    ("first keyframe off the dock", "job_dock"),
    ("first keyframe facing away from the dock", "job_dock"),
    ("last keyframe facing away from the dock", "job_dock"),
    ("a decision dropped", "job_decisions"),
    ("room_done off the last cleaning keyframe", "job_decisions"),
    ("a phase skipped", "job_phase_order"),
])
def test_the_jobs_own_checks_have_teeth(edit, code):
    """Contract (plan §4.3): `checks` flags a script that is not a vacuum/1 task: the brush on out of `cleaning`, a task
    that does not start at the dock or does not start or end facing out of it (the same position, a quarter turn off:
    the docked pose is a heading as much as a place), a decision missing or off its keyframe, a phase missing.
    Regression caught: a rule of the job that no script can break, such as a docked check of the position alone."""
    tl = job_kit.script(job_kit.seeds_for("bathroom")[0])
    kfs, decisions = list(tl.ego.keyframes), list(tl.ego.decisions)
    transit = next(kf for kf in kfs if kf.phase == "transit")
    if edit == "brush in transit":
        kfs[transit.k] = replace(transit, status={**transit.status, "brush": True})
    elif edit == "first keyframe off the dock":
        kfs[0] = replace(kfs[0], base=(4.96, 5.05, 0.0))
    elif edit == "first keyframe facing away from the dock":
        kfs[0] = replace(kfs[0], yaw=math.pi / 2)
    elif edit == "last keyframe facing away from the dock":
        kfs[-1] = replace(kfs[-1], yaw=math.pi / 2)
    elif edit == "a decision dropped":
        decisions.pop(0)
    elif edit == "room_done off the last cleaning keyframe":
        decisions[1] = replace(decisions[1], k=decisions[1].k - 1)
    else:
        kfs = [replace(kf, phase="docking") if kf.phase == "undocking" else kf for kf in kfs]
    broken = replace(tl, ego=replace(tl.ego, keyframes=kfs, decisions=decisions))
    assert code in {p.code for p in job_vacuum._checks(broken, job_kit.apartment())}


def test_a_script_of_two_tasks_is_refused_until_m2():
    """Contract: M1's `script` writes one task. Regression caught: a two-task script with no gap between them."""
    m, w = job_kit.apartment(), job_kit.generated(1)[0]
    plans = get("vacuum/1").plan(m, 1, 1) * 2
    with pytest.raises(ValueError, match="one task"):
        job_vacuum._script(m, w, plans, 1)
