"""`vacuum/2`'s doorway coin (`job_vacuum2_doorway`, docs/jobs.md, "The doorway coin"): which rooms get a coin, what a
blocked coin does to the world and to the robot's path, and that the answer the robot gives follows the world.
Hand cases on `apartment_a` (doorways: D2 0.9 m wide on the wall x = 4.6, which runs north, and the arch O1 1.2 m wide
at (3.6, 2.8) along x) for the units; seeds 101-104 for the scripts, built once for the module (`job_kit.generated2`,
with the coins forced where a contract needs them)."""

from __future__ import annotations

import math
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

import job_kit
from stbench import job_vacuum2, job_vacuum2_doorway, zone_shapes
from stbench.generate import build_world
from stbench.geometry import Rect, corners, dist_to_rect, nearest_on_rect, rect_distance, to_local
from stbench.job_registry import get
from stbench.job_vacuum2_checks import checks
from stbench.mapspec import load_map
from stbench.robots import ROBOTS
from stbench.route import robot_free_space
from stbench.timeline_ego import DecisionPoint, TaskPlan
from stbench.validate import validate
from stbench.world_processes import doors_on_ways

pytestmark = pytest.mark.xdist_group(name="test_job_vacuum2")  # shares the worker, and the cache, of the job tests

M = load_map("apartment_a")
ROBOT = ROBOTS["vacuum"]
DOORWAYS = {o.id: o for o in M.openings if o.interior}
SEEDS = (101, 102, 103, 104)
NEEDS = 0.34 + 2 * 0.02  # the vacuum's disc and twice its clearance: the narrowest gap it goes through
KEEP = 0.34 / 2 + 0.02 + 0.05  # a blocker's least distance from the base's path: radius, clearance and 0.05 m


def all_blocked(_m, _seed, _i, _j) -> bool:
    return True


def all_clear(_m, _seed, _i, _j) -> bool:
    return False


def even_tasks(_m, _seed, i, _j) -> bool:
    """Blocked in tasks 0, 2, 4, clear in tasks 1, 3, 5."""
    return i % 2 == 0


def things_in(tl, o, t: float) -> list[str]:
    """What stands in the opening's way at time `t`, read from the placements and sizes alone (the test's own reading,
    not the job's): every carried thing, clutter piece or light piece with a part over 0.02 m above the floor whose
    footprint meets the way."""
    way = zone_shapes.doorway_way(M, o)
    out = []
    for e in tl.entities:
        p = tl.placement_of(e.id, t) if e.mobility in ("carried", "clutter", "light") else None
        if p is not None and p.pos[2] + e.size[2] > 0.02 and rect_distance(
                Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw), way) == 0.0:
            out.append(e.id)
    return out


def opening_at(tl, d):
    """The doorway the zone question at decision `d` names: the opening whose way is nearest the base."""
    xy = tl.keyframe(d.k).base[:2]

    def far(o) -> float:
        way = zone_shapes.doorway_way(M, o)
        return math.dist((way.cx, way.cy), xy)

    return min(DOORWAYS.values(), key=far)


def before_rooms(tl):
    return [d for d in tl.ego.decisions if d.kind == "before_room"]


def path_samples(points):
    """The straight lines through the points as arrays of samples 0.05 m apart."""
    xs, ys = [np.empty(0)], [np.empty(0)]
    for a, b in zip(points, points[1:], strict=False):
        s = np.linspace(0.0, 1.0, max(math.ceil(math.dist(a, b) / 0.05), 1) + 1)
        xs.append(a[0] + (b[0] - a[0]) * s)
        ys.append(a[1] + (b[1] - a[1]) * s)
    return np.concatenate(xs), np.concatenate(ys)


# -- hand cases --------------------------------------------------------------------------------------------------
def test_the_free_widths_across_a_doorway_by_hand():
    """Contract: `across_gaps` is the free width between the jambs and the shadows the things cast on the wall's axis,
    left to right. Hand cases. D2 (0.9 m, jambs at -0.45 and +0.45 along the wall, which runs north for it): a 0.4 m
    by 0.3 m box lying along the wall at 0.05 m north of the centre covers -0.15 to 0.25, so the widths are 0.30 and
    0.20; turned a quarter it covers 0.3 m, -0.10 to 0.20, so 0.35 and 0.25. The arch O1 (1.2 m, jambs at +-0.6, the
    wall along x): a 0.2 m thing at -0.25 and a 0.3 m one at +0.2 cover -0.35 to -0.15 and 0.05 to 0.35, so 0.25, 0.20
    and 0.25. Nothing in the way leaves the whole width. Regression caught: a gap measured on the wrong axis (the
    thing's depth for its width), or a rotated thing's shadow taken as its width."""
    d2, o1 = DOORWAYS["D2"], DOORWAYS["O1"]
    along = Rect(4.6, 3.95, 0.4, 0.3, math.pi / 2)  # its long side runs along the wall (D2's wall runs north)
    assert job_vacuum2_doorway.across_gaps(d2, [along]) == pytest.approx([0.30, 0.20])
    assert job_vacuum2_doorway.across_gaps(d2, [along._replace(yaw=0.0)]) == pytest.approx([0.35, 0.25])
    pair = [Rect(3.35, 2.8, 0.2, 0.2, 0.0), Rect(3.8, 2.8, 0.3, 0.2, 0.0)]
    assert job_vacuum2_doorway.across_gaps(o1, pair) == pytest.approx([0.25, 0.20, 0.25])
    assert job_vacuum2_doorway.across_gaps(d2, []) == pytest.approx([0.9])
    assert job_vacuum2_doorway.gap_limit(ROBOT) == pytest.approx(NEEDS)
    assert job_vacuum2_doorway.keep_from_body(ROBOT) == pytest.approx(KEEP)


def test_a_thing_stands_in_the_way_when_it_is_over_the_zones_height_and_meets_it():
    """Contract: a thing stands in a doorway's way (what a `zone_clear` gold reads) when it rises over the zone's
    height, 0.02 m (the vacuum's climb), and its footprint meets the way. Hand cases on D2's way (x 4.6 to 4.95, y 3.55
    to 4.25): a 0.2 m box 0.03 m tall at (4.8, 3.9) stands; the same 0.02 m tall does not (the robot drives over it);
    one beside the way, at (5.3, 3.9), does not. Regression caught: a mat that closes a doorway, or a blocker the gold
    would not read."""
    way = zone_shapes.doorway_way(M, DOORWAYS["D2"])
    assert (way.cx, way.sx, way.sy) == pytest.approx((4.775, 0.7, 0.35))
    on = Rect(4.8, 3.9, 0.2, 0.2, 0.0)
    assert job_vacuum2_doorway.stands(way, on, 0.0, 0.03)
    assert not job_vacuum2_doorway.stands(way, on, 0.0, 0.02)
    assert not job_vacuum2_doorway.stands(way, on._replace(cx=5.3), 0.0, 0.5)


def test_a_decision_stands_half_a_metre_of_path_before_the_doorway():
    """Contract: a doorway decision stands `DOORWAY_STANDOFF_M` (0.5 m) of path before the crossing, walking back over
    segments, so that a path that turns at the door still leaves room in front of it. Hand cases on D2's line (x 4.6)
    for the living room: down the hall and west through it, (5.4, 3.9) to (4.0, 3.9), crosses at 0.8 m, so the
    decision is 0.3 m along; a path that turns 0.3 m before the door, (5.4, 4.9) to (4.9, 3.9) to (4.0, 3.9), crosses
    0.3 m into its second segment, so the other 0.2 m are walked back on the first, whose length is sqrt(1.25) =
    1.118034 m: 0.918034 m along it. Regression caught: a decision 0.3 m from the door, where no blocker can lie in
    the way and stay clear of the body."""
    living = M.regions[1]
    assert living.kind == "living room"
    straight = job_vacuum2_doorway.decision_mark(M, living, [(5.4, 3.9), (4.0, 3.9)], last=True)
    assert straight[0] == 0 and straight[1] == pytest.approx(0.3)
    turning = job_vacuum2_doorway.decision_mark(M, living, [(5.4, 4.9), (4.9, 3.9), (4.0, 3.9)], last=True)
    assert turning[0] == 0 and turning[1] == pytest.approx(math.sqrt(1.25) - 0.2)


def test_the_zone_question_names_the_doorway_the_job_blocks():
    """Contract: `opening_ahead` is the doorway the zone rule finds for the base (`zones.zone_of`: the nearest way's
    centre), so the doorway a decision blocks is the one its question names. Hand cases: a base 0.5 m east of D2's line,
    in the hall, names D2; one 0.5 m west of O1's line in the living room (the arch's wall runs along x, so 0.5 m south
    of it) names O1; a base in the kitchen 0.8 m north of the arch, nearer it than D1, names O1. Regression caught: a
    blocker in a doorway the question is not about."""
    assert job_vacuum2_doorway.opening_ahead(M, (5.1, 3.9)).id == "D2"
    assert job_vacuum2_doorway.opening_ahead(M, (3.6, 3.3)).id == "O1"
    assert job_vacuum2_doorway.opening_ahead(M, (3.6, 2.0)).id == "O1"


# -- the coin and the rooms ---------------------------------------------------------------------------------------
def test_only_eligible_rooms_get_a_coin():
    """Contract: a room is entered through a `before_room` (and its coin drawn) when its task is `clean_rooms`, it is
    not the dock's room (the hall: no doorway is crossed) and not the room a stopping task stops in; a resume has none.
    Hand cases: a kitchen-then-hall task has one (the kitchen), a task that stops in the bedroom has none, a task of the
    living room then the kitchen that stops has one (the living room), a resume has none. On the scripts of seeds
    101-104 with their own coins, every coin drawn is for an eligible room, each `before_room` has exactly one coin, and
    every `before_room` stands in a `clean_rooms` task. Regression caught: a coin for the room a resume finishes, which
    would leave the stopped brush's sweep short, or a question about a doorway nobody ever blocks."""
    kitchen_hall = TaskPlan("clean_rooms", {"rooms": "kitchen, then the hall", "kinds": ["kitchen", "hall"]})
    stops = TaskPlan("clean_rooms", {"rooms": "bedroom", "kinds": ["bedroom"], "stop": 0.5})
    stops_two = TaskPlan("clean_rooms", {"rooms": "living room, then the kitchen", "kinds": ["living room", "kitchen"],
                                         "stop": 0.4})
    resume = TaskPlan("resume", {"rooms": "bedroom", "kinds": ["bedroom"]})
    assert [job_vacuum2.eligible(M, kitchen_hall, j) for j in (0, 1)] == [True, False]
    assert job_vacuum2.eligible(M, stops, 0) is False
    assert [job_vacuum2.eligible(M, stops_two, j) for j in (0, 1)] == [True, False]
    assert job_vacuum2.eligible(M, resume, 0) is False
    drawn = []
    real = job_vacuum2_doorway.doorway_blocked

    def spy(m, seed, i, j):
        drawn.append((i, j))
        return real(m, seed, i, j)

    for seed in SEEDS:
        drawn.clear()
        with mock.patch.object(job_vacuum2_doorway, "doorway_blocked", spy):
            _, tl = job_kit.generated2.__wrapped__(seed)
        for i, j in drawn:
            assert job_vacuum2.eligible(M, tl.ego.tasks[i].plan, j), (seed, i, j)
        for d in before_rooms(tl):
            assert tl.task_of(d.k).plan.kind == "clean_rooms", seed
        assert len(drawn) == len(before_rooms(tl)) >= 1, seed


def test_a_blocked_coin_closes_the_doorway_and_a_clear_one_leaves_it_open():
    """Contract: with every coin blocked, every `before_room` is answered "skip" and the things in its doorway (one in a
    door, two in the arch) leave no free width across it as wide as the body plus twice its clearance (0.38 m), are
    centred on the gap within 0.1 m, keep 0.24 m (the radius, the clearance and 0.05 m) from the base's path up to the
    decision, and the planner on the world as the task left it cannot cross: from the base to the point straight across
    the doorway there is no way, or one at least 1.5 m longer than the straight line (a detour the other way round).
    With every coin clear every one is answered "go" and nothing stands in any doorway. Doors and the arch are both
    met. Regression caught: a blocker the robot could squeeze past (the gold says blocked, the planner says go), one
    on top of the robot, or a clear doorway with something in it."""
    seen = set()
    for seed in SEEDS:
        worlds, tl = job_kit.generated2(seed, (), all_blocked)
        for d in before_rooms(tl):
            task, kf, o = tl.task_of(d.k), tl.keyframe(d.k), opening_at(tl, d)
            assert d.response == "skip", (seed, d)
            seen.add(o.id)
            things = things_in(tl, o, kf.t)
            rects = [tl.footprint(e, kf.t) for e in things]
            assert len(things) == (2 if o.id == "O1" else 1), (seed, o.id, things)
            assert max(job_vacuum2_doorway.across_gaps(o, rects)) < NEEDS
            frame = Rect(o.center[0], o.center[1], 0.0, 0.0, o.yaw)
            along = [to_local(frame, *c)[0] for r in rects for c in corners(r)]
            assert abs((min(along) + max(along)) / 2) <= 0.1 + 1e-6, (seed, o.id)
            path = path_samples([tl.keyframe(k).base[:2] for k in range(task.k0, d.k + 1)])
            assert min(float(dist_to_rect(*path, r).min()) for r in rects) >= KEEP - 1e-3, (seed, o.id)
            on = doors_on_ways(M, {"R1", *task.targets})
            shut = sorted(x.id for x in M.toggling_doors() if x.id not in on)
            free = robot_free_space(M, worlds[task.index], ROBOT, kf.t, shut=shut)
            base = kf.base[:2]
            ahead = (2 * o.center[0] - base[0], 2 * o.center[1] - base[1])  # as far beyond the doorway, straight across
            way = free.path(base, ahead, strict=True)
            if way is not None:
                length = sum(math.dist(a, b) for a, b in zip(way, way[1:], strict=False))
                assert length >= math.dist(base, ahead) + 1.5, (seed, o.id, length)
        assert validate(tl, M) == []
        _, clear = job_kit.generated2(seed, (), all_clear)
        assert before_rooms(clear) and all(d.response == "go" for d in before_rooms(clear))
        for d in before_rooms(clear):
            assert things_in(clear, opening_at(clear, d), clear.keyframe(d.k).t) == [], (seed, d)
    assert "O1" in seen and len(seen & {"D1", "D2", "D3", "D4"}) >= 2


def test_a_skip_happens_exactly_when_the_doorway_holds_a_blocker():
    """Contract: over the scripts of seeds 101-104 with their own coins, a `before_room` is answered "skip" if and only
    if something stands in the way of the doorway its question names at the decision's time (read here from the
    placements and sizes), about half are skips, no other decision is answered, and a skip is followed by no cleaning of
    that room: the rooms cleaned are the plan's less the skipped. The job's own check agrees. Regression caught: a
    robot that drives on into a doorway it was told is blocked, or one that skips a clear one."""
    skips = total = 0
    for seed in SEEDS:
        _, tl = job_kit.generated2(seed)
        assert checks(tl, M) == []
        for d in tl.ego.decisions:
            assert (d.response is not None) == (d.kind == "before_room"), (seed, d)
            if d.kind != "before_room":
                continue
            held = things_in(tl, opening_at(tl, d), tl.keyframe(d.k).t)
            assert (d.response == "skip") == bool(held), (seed, d, held)
            skips, total = skips + (d.response == "skip"), total + 1
        for task in tl.ego.tasks:
            decisions = tl.decisions_in(task.k0, task.k1)
            cleaned = sum(d.kind in ("room_done", "cleaning_stopped") for d in decisions)
            assert cleaned == len(task.targets) - sum(d.response == "skip" for d in decisions), (seed, task.index)
    assert total >= 8 and 0.2 <= skips / total <= 0.8


def test_the_next_task_starts_with_every_doorway_clear():
    """Contract: before a task is planned every doorway is cleared, so the blockers of the task before are gone and a
    clear coin finds nothing there. With the coins blocked in even tasks and clear in odd ones, at the first keyframe of
    every odd task no doorway holds anything (the blockers of the even task before it stood the whole task and were
    moved out in the gap), while at the start of an even task exactly the doorways its skips are answered at do.
    Regression caught: a blocker left in a doorway for the next task, which would say blocked to a coin that said
    clear."""
    held = emptied = 0
    for seed in SEEDS:
        _, tl = job_kit.generated2(seed, (), even_tasks)
        for task in tl.ego.tasks:
            t0 = tl.keyframe(task.k0).t
            standing = {o.id for o in DOORWAYS.values() if things_in(tl, o, t0)}
            if task.index % 2:
                assert standing == set(), (seed, task.index, standing)
                before = tl.ego.tasks[task.index - 1]
                emptied += any(d.response == "skip" for d in tl.decisions_in(before.k0, before.k1))
            else:
                skipped = {opening_at(tl, d).id for d in tl.decisions_in(task.k0, task.k1)
                           if d.kind == "before_room" and d.response == "skip"}
                assert standing == skipped, (seed, task.index)
                held += bool(skipped)
    assert held >= 3 and emptied >= 2


def test_a_coin_is_forced_without_moving_anything_up_to_its_decision():
    """Contract (FD-0.5.0-control-impl §4): the keyframes up to and including a coin's decision, and the decisions
    before it, are the same whichever way the coin falls: on seed 103, with every other coin clear, the first
    `before_room` forced blocked and forced clear give equal keyframes through that decision and the same decision
    kind, answered "skip" in one and "go" in the other, while the world differs (a thing moved into the doorway) and the
    keyframes after it differ (the skip leaves the room). Regression caught: a path planned with the blockers in the
    world before the decision, which tells the coin to everything that reads the frames."""
    def coin(i, j, blocked):
        return lambda _m, _seed, task, room: blocked if (task, room) == (i, j) else False

    _, base = job_kit.generated2(103, (), all_clear)
    first = before_rooms(base)[0]
    task = base.task_of(first.k)
    room = next(j for j in range(len(task.targets)) if job_vacuum2.eligible(M, task.plan, j))
    _, a = job_kit.generated2(103, (), coin(task.index, room, True))
    _, b = job_kit.generated2(103, (), coin(task.index, room, False))
    da, db = (next(d for d in before_rooms(t) if t.task_of(d.k).index == task.index) for t in (a, b))
    assert (da.k, da.kind, da.response, db.response) == (db.k, db.kind, "skip", "go")
    assert a.ego.keyframes[:da.k + 1] == b.ego.keyframes[:da.k + 1]
    assert a.ego.decisions[:a.ego.decisions.index(da)] == b.ego.decisions[:b.ego.decisions.index(db)]
    assert a.ego.tasks[:task.index] == b.ego.tasks[:task.index]
    assert a.world_hash != b.world_hash
    assert a.ego.keyframes[da.k + 1:da.k + 40] != b.ego.keyframes[da.k + 1:da.k + 40]


def test_a_coin_that_finds_no_thing_raises_and_names_where():
    """Contract: when no carried thing of a floor kind and no floor clutter piece can close the doorway, the script
    is not written: `_script` raises a `ValueError` naming the seed, the task and the doorway, never a doorway that
    says blocked and is not (the seed is skipped by the declared rule). Hand case: seed 1's world with nothing on its
    floor and the coin blocked, for the bedroom behind D4. Regression caught: a blocked coin that is silently left
    clear."""
    w = build_world(M, 1)
    w.carried.clear()
    w.clutter.clear()
    plans = [TaskPlan("clean_rooms", {"rooms": "bedroom", "kinds": ["bedroom"]})]
    with mock.patch.object(job_vacuum2_doorway, "doorway_blocked", all_blocked), pytest.raises(
            ValueError, match=r"seed 7, task 0 \(bedroom\).*no floor thing closes the D4"):
        job_vacuum2._script(M, w, plans, 7)


# -- the job's own rules have teeth -----------------------------------------------------------------------------------
def single_thing(tl, later: bool):
    """The first `before_room` of the script closed by one thing (in a task with another after it when `later`), with
    its keyframe, doorway and thing."""
    for d in before_rooms(tl):
        kf, o = tl.keyframe(d.k), opening_at(tl, d)
        if len(things := things_in(tl, o, kf.t)) == 1 and d.response == "skip" and (
                not later or tl.task_of(d.k).index + 1 < len(tl.ego.tasks)):
            return d, kf, o, things[0]
    raise AssertionError("no decision closed by one thing")


def moved(tl, eid, t, dx, dy):
    """`tl` with the entity's placement that holds time `t` shifted by (dx, dy) metres."""
    return tl.with_placements({eid: [replace(p, pos=(p.pos[0] + dx, p.pos[1] + dy, p.pos[2])) if p.covers(t) else p
                                     for p in tl.placements_of(eid)]})


@pytest.mark.parametrize(("edit", "code"), [
    ("a skip with nothing in the doorway", "job_doorway"),
    ("a go with a blocker in the doorway", "job_doorway"),
    ("a response on a decision that is not asked", "job_response"),
    ("a response that is no response", "job_response"),
    ("a blocker moved 0.3 m along the wall", "job_doorway_gap"),
    ("a base 0.1 m from a blocker", "job_doorway_near"),
    ("a blocker left in a doorway for the next task", "job_doorway"),
])
def test_the_doorway_checks_have_teeth(edit, code):
    """Contract: `checks` flags a script whose doorway coin breaks a rule: a `before_room` answered "skip" with nothing
    in its doorway or answered "go" with a blocker there (`job_doorway`), a decision that is not a `before_room`
    carrying a response or a response that is neither "go" nor "skip" (`job_response`), a blocker pushed 0.3 m along
    the wall so that the robot could pass (`job_doorway_gap`), a base 0.1 m from a blocker at the decision
    (`job_doorway_near`) and a blocker standing in a doorway at a task's start that none of its skips is answered at
    (`job_doorway`). Seed 103 with every coin blocked is the script edited (with the coins blocked in even tasks only,
    for the last edit); its own checks find nothing. Regression caught: a rule of the coin that no script can break."""
    _, tl = job_kit.generated2(103, (), even_tasks if "next task" in edit else all_blocked)
    assert checks(tl, M) == []
    d, kf, o, eid = single_thing(tl, later="next task" in edit)
    decisions, kfs = list(tl.ego.decisions), list(tl.ego.keyframes)
    at, broken = decisions.index(d), tl
    if edit == "a skip with nothing in the doorway":
        broken = moved(tl, eid, kf.t, 3.0, 3.0)
    elif edit == "a go with a blocker in the doorway":
        decisions[at] = DecisionPoint(d.k, d.kind, "go")
    elif edit == "a response on a decision that is not asked":
        i = next(i for i, x in enumerate(decisions) if x.kind == "room_done")
        decisions[i] = DecisionPoint(decisions[i].k, "room_done", "go")
    elif edit == "a response that is no response":
        decisions[at] = DecisionPoint(d.k, d.kind, "abort")
    elif edit == "a blocker moved 0.3 m along the wall":
        broken = moved(tl, eid, kf.t, math.cos(o.yaw) * 0.3, math.sin(o.yaw) * 0.3)
    elif edit == "a base 0.1 m from a blocker":
        rect = tl.footprint(eid, kf.t)
        (nx, ny), dist = nearest_on_rect(rect, kf.base[0], kf.base[1])
        step = (dist - 0.1) / dist
        base = (kf.base[0] + (nx - kf.base[0]) * step, kf.base[1] + (ny - kf.base[1]) * step, kf.base[2])
        kfs[d.k] = replace(kf, base=base)
    else:  # the blocker stays where it is for good: no later placement moves it out, and the next task has no skip
        stays = next(p for p in tl.placements_of(eid) if p.covers(kf.t))
        broken = tl.with_placements({eid: [replace(p, t1=None) if p is stays else p
                                           for p in tl.placements_of(eid) if p.t0 <= stays.t0]})
        assert tl.task_of(d.k).index % 2 == 0
    broken = replace(broken, ego=replace(broken.ego, decisions=decisions, keyframes=kfs))
    assert code in {p.code for p in checks(broken, M)}


def test_the_job_names_the_doorway_as_the_zone_of_its_coin():
    """Contract: `vacuum/2` has a zone for `before_room` alone, the doorway rule (the way through the opening ahead, the
    vacuum's climb as its height), a ladder of 0.25 m rungs and a subgoal; `vacuum/1` is unchanged. Regression caught:
    the coin placed at the exit doorway, which is never blocked, or a blocker with no zone (`World.told` keys on the
    job's zones, so its blocker would be asked about)."""
    job = get("vacuum/2")
    assert set(job.zones) == {"before_room"}
    assert job.zones["before_room"].piece == "doorway" and job.zones["before_room"].height_m == ROBOT.body.climb_m
    assert job.ladder == (0.25, 0.5, 1.0, 2.0) and job.subgoals is not None
    assert get("vacuum/1").subgoals is None and "before_room" not in get("vacuum/1").decisions
