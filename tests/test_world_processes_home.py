"""The home's processes between tasks (`world_processes_home`): who a gap draws again, where the doors stand, that a
gap's changes are one instant, that a re-draw follows the law the world started with, and that the world stays valid
through many gaps. The worlds are the apartment's seeds 101-103 (`process_kit`); rates are written out here from the
declared half-lives (carried things 5 h, chairs 6 h, the stool, the basket and floor clutter 12 h, doors 2 h, doors open
70%), the doors as `maps/apartment_a.json` has them (D1 kitchen, D2 living room, D3 bathroom, D4 bedroom from the hall
R1; D0 the entrance) and the chairs' map places from the map: (x, y, facing)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from builders import keyframe, start_status
from process_kit import HOUR, SEEDS, copy_world, fresh, guard_off
from realism_kit import DOCKED_BASE, apartment
from stbench import clutter
from stbench import world_processes_home as home
from stbench.catalog import FLOOR_ITEM_Z
from stbench.generate import light_start, to_timeline
from stbench.geometry import wrap_angle
from stbench.ledger import Ledger
from stbench.robots import ROBOTS
from stbench.route import robot_free_space
from stbench.seeding import make_rng
from stbench.timeline import Ego, Task, TaskPlan
from stbench.validate import validate
from stbench.world_processes import door_states, doors_on_ways, gaps, half_life_p
from validate_kit import generic_rules_only  # noqa: F401  (a fixture)
from world_kit import info

# One xdist group (`--dist loadgroup`): the worlds and the driven scripts are built once a worker.
pytestmark = pytest.mark.xdist_group(name=__name__)
CHAIRS = ((1.114, 1.9, 0.0), (3.0865, 1.9, math.pi), (3.6, 4.9, math.pi), (7.9585, 3.85, 0.0))  # L01-L04: map places
INTERIOR = ("D1", "D2", "D3", "D4")
WORLD_CODES = {"light_collision", "carried_collision", "placement_overlap"}
HELPER_CLIMB_M = 0.01  # the lowest height a planned robot (the helper) must go round


def within(hits: int, n: int, p: float) -> bool:
    """Whether `hits` of `n` independent draws lies within 4 standard errors of the chance `p`."""
    return abs(hits / n - p) <= 4 * math.sqrt(p * (1 - p) / n)


def test_half_life_law():
    """Contract (M2-A2W §2): a thing of half-life H is drawn again over a gap g with chance 1 - 2^(-g/H): by hand 0 for
    a gap of 0, 1/2 for g = H, 3/4 for 2H, and 34% for 3 h at H = 5 h; and each process's selection, alone and over
    1,000 draws of a 10-minute, a 1-hour and a 3-hour gap, picks that share of its things within 4 standard errors:
    the four chairs (H = 6 h), the stool and the basket (12 h), the 30 carried things (5 h), the floor clutter (12 h),
    and the three interior doors not forced open for the last task (2 h; the forced one is always drawn again, and the
    entrance never is). Regression caught: a rate that ignores the gap's length (V0's Poisson count a gap did), a
    process on another half-life than declared, or a door the draw can never reach."""
    h = 3600.0
    assert half_life_p(0.0, h) == 0.0 and half_life_p(h, h) == 0.5 and half_life_p(2 * h, h) == 0.75
    assert half_life_p(3 * h, 5 * h) == pytest.approx(0.3402, abs=1e-4)
    w, m = fresh(101), apartment()
    kind = {e.id: e.kind for e in w.entities}
    chairs = [i for i in w.light_rects if kind[i] == "chair"]
    others = [i for i in w.light_rects if kind[i] != "chair"]
    floor = home.floor_clutter(w)
    assert (len(chairs), len(others), len(w.carried)) == (4, 2, 30) and len(floor) > 15
    rng, forced = make_rng("test_half_life_law"), frozenset({"D1"})
    for gap_h in (1 / 6, 1.0, 3.0):
        got = Counter()
        for _ in range(1000):
            light = home.light_movers(rng, w, gap_h * HOUR)
            got["chairs"] += sum(1 for i in light if i in chairs)
            got["others"] += sum(1 for i in light if i in others)
            got["items"] += len(home.item_movers(rng, w, gap_h * HOUR))
            got["floor"] += len(home.clutter_movers(rng, w, gap_h * HOUR))
            doors = home.door_redraws(rng, m, gap_h * HOUR, forced)
            assert "D1" in doors and "D0" not in doors
            got["doors"] += len(doors) - 1
        for name, n_things, half_h in (("chairs", len(chairs), 6), ("others", len(others), 12), ("items", 30, 5),
                                       ("floor", len(floor), 12), ("doors", 3, 2)):
            assert within(got[name], 1000 * n_things, 1 - 2 ** (-gap_h / half_h)), (gap_h, name, got[name])


def test_the_doors_on_the_ways_are_open_at_hi():
    """Contract (M2-A2W §1, design §8.1): after the doors' draw every door of `doors_on_ways(m, keep_open)` is open, so
    no door's state changes the next task's path; over 200 draws with random gaps, rooms and forced sets every other
    interior door is seen both open and shut; a door forced open for the last task, drawn again for a task that does
    not need it, is open with the start's 70% (within 4 standard errors, over 600 draws) and no more; the entrance is
    never written again; and `start` draws each interior door open 70% of the time with the ways' doors open (hand
    case: for {hall, bedroom} D4 is always open and D1, D2, D3 each open 70% within 4 standard errors over 200
    starts). Regression caught: a task sent to a room behind a shut door, a door that never moves, a forced door that
    stays open and so marks the window's place, or the entrance opened."""
    m, w, rng = apartment(), fresh(101), make_rng("test_doors")
    rooms, seen, t = ["R1", "R2", "R3", "R4", "R5"], defaultdict(set), 100.0
    for _ in range(200):
        keep = frozenset(str(r) for r in rng.choice(rooms, size=int(rng.integers(1, 4)), replace=False))
        forced = frozenset(str(d) for d in rng.choice(INTERIOR, size=int(rng.integers(0, 3)), replace=False))
        t += 1.0
        home.redraw_doors(w, m, rng, t, home.door_redraws(rng, m, float(rng.uniform(60.0, 3 * HOUR)), forced), keep)
        now, on = door_states(w), doors_on_ways(m, sorted(keep))
        assert all(now[d] for d in on), (sorted(keep), now)
        for d in INTERIOR:
            if d not in on:
                seen[d].add(now[d])
    assert all(seen[d] == {True, False} for d in INTERIOR), dict(seen)
    forced_open = 0
    for _ in range(300):
        t += 1.0
        home.redraw_doors(w, m, rng, t, home.door_redraws(rng, m, 600.0, frozenset({"D1", "D2"})), frozenset({"R1"}))
        now = door_states(w)
        forced_open += now["D1"] + now["D2"]
    assert within(forced_open, 600, 0.7), forced_open
    assert [(s.t0, s.value) for s in w.ledger.finish()[1] if s.entity == w.door_entity["D0"]] == [(0.0, False)]
    opened, wb = Counter(), fresh(102)
    for _ in range(200):
        home.start(wb, m, rng, frozenset({"R1", "R4"}))
        now = door_states(wb)
        assert now["D4"] and not now["D0"]
        opened.update(d for d in ("D1", "D2", "D3") if now[d])
    assert all(within(opened[d], 200, 0.7) for d in ("D1", "D2", "D3")), opened


@pytest.fixture(scope="module")
def driven():
    """Each of the three worlds driven through `gaps`: the start call, then five one-hour gaps cut from the stretches
    between its tasks (seed 101: two tasks, the others three), each call's new placements and states, and what each
    planned robot's reachable floor (from the hall level with the dock, the doors as they stand at the end of the call,
    the world just before it and just after) lost over each call, summed: the vacuum on every world, the helper on the
    first (a free space is 25 ms)."""
    m = apartment()
    (dock,) = [f for f in m.furniture if f.kind == "dock"]
    front = (5.4, dock.center[1])

    def reach(w, robot, doors):
        ledger = Ledger()
        for door, eid in w.door_entity.items():
            ledger.set_state(eid, "open", 0.0, doors[door])
        fs = robot_free_space(m, replace(w, ledger=ledger), ROBOTS[robot])
        assert fs.is_free(front)
        return fs, fs.reachable_from(front)

    runs = {}
    for seed, tasks in TASKS.items():
        w = fresh(seed)
        height = {e.id: e.size[2] for e in w.entities}
        between, calls, cut = gaps(w, m, seed), [], Counter()
        before_start = w.ledger.finish()
        between(None, 0.0, KEEPS[0])
        after_start = w.ledger.finish()
        for k, (lo, hi) in enumerate(slices(tasks)):
            old_p, old_s = w.ledger.finish()
            then = copy_world(w)
            between(lo, hi, KEEPS[k])
            new_p, new_s = w.ledger.finish()
            seen_p, seen_s = {(p.entity, p.t0) for p in old_p}, {(s.entity, s.key, s.t0) for s in old_s}
            calls.append(SimpleNamespace(lo=lo, hi=hi, placements=[p for p in new_p if (p.entity, p.t0) not in seen_p],
                                         states=[s for s in new_s if (s.entity, s.key, s.t0) not in seen_s]))
            doors = door_states(w)
            if not any(p.pos[2] < FLOOR_ITEM_Z and height[p.entity] > HELPER_CLIMB_M for p in calls[-1].placements):
                continue  # nothing a robot must go round came to rest on the floor, so no floor can be cut
            for robot in ("vacuum", "helper") if seed == SEEDS[0] else ("vacuum",):
                (_, before), (fs, after) = reach(then, robot, doors), reach(w, robot, doors)
                cut[robot] += int((before & fs.free & ~after).sum())
        runs[seed] = SimpleNamespace(world=w, tasks=tasks, calls=calls, cut=dict(cut),
                                     start=(before_start, after_start))
    return runs


TASKS = {101: (0.0, 18000.0), 102: (0.0, 7200.0, 18000.0), 103: (0.0, 10800.0, 18000.0)}  # task times, seconds
KEEPS = [frozenset(k) for k in ({"R1", "R3"}, {"R1", "R4"}, {"R1", "R5"}, {"R1", "R2"}, {"R1"})]  # rooms kept open


def slices(tasks: tuple[float, ...]) -> list[tuple[float, float]]:
    """The unobserved stretches between consecutive tasks (a task is two keyframes 1 s apart), cut at whole hours."""
    out = []
    for a, b in zip(tasks, tasks[1:], strict=False):
        edges = [a + 1.0] + [h * HOUR for h in range(1, int(b / HOUR)) if a + 1.0 < h * HOUR < b] + [b]
        out += list(zip(edges, edges[1:], strict=False))
    return out


def test_one_gap_is_one_instant(driven):
    """Contract (M2-A2W §2, "one instant a gap"): on each of the three worlds, every placement and every state a gap
    call adds starts at one and the same whole second, at least 30 s from both ends of the gap, so the robot never
    sees one happen and no change lands where another thing has not yet left; the start call adds no placement and
    leaves every state at t = 0. Regression caught: a change with a time of its own (a chair taking another's place
    before it moved), a change in the 30 s next to a task, or a start that writes into the first task's time."""
    changes = 0
    for seed, run in driven.items():
        (old_p, _), (new_p, new_s) = run.start
        assert new_p == old_p and {s.t0 for s in new_s} == {0.0}, seed
        for call in run.calls:
            times = {p.t0 for p in call.placements} | {s.t0 for s in call.states}
            assert len(times) <= 1, (seed, call.lo, sorted(times))
            for t in times:
                assert t == int(t) and call.lo + 30 <= t <= call.hi - 30, (seed, call.lo, call.hi, t)
            changes += len(call.placements) + len(call.states)
    assert changes >= 30  # the gaps changed something: six things a gap, as the declared rates give


def docked_ego(times: tuple[float, ...]) -> Ego:
    """A vacuum docked for two keyframes (1 s apart) at each task time: every task is the same clean_room, which needs
    no path, for the timeline's tasks to bound the gaps."""
    kfs, tasks, (x, y, _, yaw), robot = [], [], DOCKED_BASE, ROBOTS["vacuum"]
    for i, t in enumerate(times):
        for dt in (0.0, 1.0):
            kfs.append(keyframe("vacuum", len(kfs), t + dt, (x, y, 0.0), yaw, task=i, phase=robot.status.start["phase"],
                                status=start_status("vacuum")))
        tasks.append(Task(i, 2 * i, 2 * i + 2, TaskPlan("clean_room", {"room": "kitchen"}), ()))
    return Ego(kfs, tasks, [])


def test_moves_stay_valid(driven, generic_rules_only):  # noqa: F811
    """Contract (M2-A2W §2): after five gap calls between hand-made tasks (two tasks on seed 101, three on 102 and 103,
    the vacuum docked at each) the timeline of each world has no `light_collision`, `carried_collision` or
    `placement_overlap`, and `validate` finds nothing else; and no floor a planned robot (the vacuum, the helper) could
    reach from the hall before a gap call is cut off by it (`FloorGuard`, the doors as they stand after the call: 0
    cells; the helper on seed 101 only). Regression caught: a chair in a doorway or on a thing, a thing taking
    another's place, a room a gap cuts off."""
    m, moved = apartment(), 0
    for seed, run in driven.items():
        tl = to_timeline(m, run.world, docked_ego(run.tasks), info(m, seed))
        assert [p for p in validate(tl, m) if p.code in WORLD_CODES] == [], seed
        assert validate(tl, m) == [], seed
        assert "vacuum" in run.cut and not any(run.cut.values()), (seed, run.cut)
        moved += sum(len(c.placements) for c in run.calls)
    assert moved >= 20


def test_moves_redraw_the_start_law(monkeypatch):
    """Contract (M2-A2W §2, "re-draws by the start's law"): a gap's draw is the start's, so the world at a later task is
    distributed as the world at the first. Over 25 re-draws of each of the four chairs on each world, from wherever
    each stands, the new place is the chair's pull-out from its map place and not from its last place: 0 to 0.6 m
    against its facing, on that line, turned at most 15 degrees (a draw from where it stands would walk off past 0.6 m
    in a few gaps); over 60 rounds a carried thing of a floor kind (shoes, bag, toy, box) is asked a floor spot, at
    its kind's anchor pieces, in 50% of its draws (within 4 standard errors) and any other thing never, being asked a
    receptacle instead; and over 300 draws of a 30-day gap each interior door is open 70% (within 4 standard errors)
    and the entrance is never drawn. Regression caught: a drift that makes later tasks look unlike the first (chairs
    far from their tables, everything on the floor), a drop share that is not the start's, a door that stays shut."""
    guard_off(monkeypatch)
    m, rng, t = apartment(), make_rng("test_start_law"), 100.0
    moved = 0
    for seed in SEEDS:
        w = fresh(seed)
        chairs = [e.id for e in w.entities if e.kind == "chair"]
        for _ in range(25):
            t += 1.0
            for eid, (x0, y0, yaw0) in zip(chairs, CHAIRS, strict=True):
                if not light_start(w, m, rng, eid, home.ANCHORS, t):
                    continue
                moved += 1
                now = w.light_rects[eid]
                dx, dy = now.cx - x0, now.cy - y0
                along, across = -(dx * math.cos(yaw0) + dy * math.sin(yaw0)), -dx * math.sin(yaw0) + dy * math.cos(yaw0)
                assert -1e-4 <= along <= 0.6 + 1e-4 and abs(across) < 1e-4, (seed, eid, along, across)
                assert abs(wrap_angle(now.yaw - yaw0)) <= math.radians(15) + 1e-4, (seed, eid)
    assert moved >= 150  # of 300 re-draws: a chair hemmed in keeps its place (the start's rule)
    asked = Counter()

    def floor_spot(world, m, rng, size, **kw):
        asked["floor", kw["anchors"]] += 1

    def receptacle(rng, spec, slots, occupied, **kw):
        asked["receptacle", spec.kind] += 1

    monkeypatch.setattr(clutter, "floor_spot", floor_spot)
    monkeypatch.setattr(home, "find_spot", receptacle)
    tracked = Counter()
    for seed in SEEDS:
        w = fresh(seed)
        floor_kinds = dict(w.catalog.floor_kinds)
        for eid, c in w.carried.items():
            for _ in range(60):
                assert not home.move_item(w, m, rng, eid, t)  # no spot is found: it stays
            tracked[c.spec.kind in floor_kinds] += 60
            if c.spec.kind in floor_kinds:
                assert asked["floor", floor_kinds[c.spec.kind]] > 0
    on_floor = sum(n for (what, _), n in asked.items() if what == "floor")
    assert tracked[True] >= 900 and within(on_floor, tracked[True], 0.5), (on_floor, tracked)
    assert sum(n for (what, _), n in asked.items() if what == "receptacle") == sum(tracked.values()) - on_floor
    open_, w = Counter(), fresh(101)
    for _ in range(300):
        t += 1.0
        drawn = home.door_redraws(rng, m, 30 * 24 * HOUR, frozenset())
        assert drawn == list(INTERIOR)
        home.redraw_doors(w, m, rng, t, drawn, frozenset({"R1"}))
        open_.update(d for d, state in door_states(w).items() if state and d != "D0")
    assert all(within(open_[d], 300, 0.7) for d in INTERIOR), open_


def test_rates_cite_their_sources():
    """Contract (VISION rule 10, "every rate has a source or says it is a modelling choice"): every rate constant of
    `world_processes_home` (the half-lives `H_*_S` and `P_OPEN`) has a non-empty `SOURCES` entry, and no entry names
    anything else. Regression caught: a rate with no stated basis, or a stale entry for a rate that is gone."""
    rates = {n for n in vars(home) if n.startswith(("H_", "P_")) and n.isupper()}
    assert rates == {"H_ITEM_S", "H_CHAIR_S", "H_LIGHT_S", "H_FLOOR_CLUTTER_S", "H_DOOR_S", "P_OPEN"}
    assert set(home.SOURCES) == rates and all(len(home.SOURCES[n]) > 50 for n in rates)
