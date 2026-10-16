"""The world's processes (A2W's `world_processes`, the doors of `doors_on_ways`) inside real `vacuum/2` scripts of
seeds 101-104, with the real clutter and the resident (`job_kit.generated2`, one xdist group with the job tests):
every change a non-mover entity makes after t = 0 falls inside a gap, nothing changes in view, the doors a task needs
are open through it, and every case clock a window can carry ends by 17:00."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import replace
from datetime import datetime, time

import pytest

import job_kit
from stbench import build, case
from stbench import episode as ep
from stbench.changes import EDGE_BUFFER
from stbench.validate import door_entities
from stbench.world_processes import doors_on_ways

pytestmark = pytest.mark.xdist_group(name="test_job_vacuum2")  # the scripts of `test_job_vacuum2` are built once

SEEDS = (101, 102, 103, 104)
M = job_kit.apartment()


@pytest.fixture(scope="module")
def scripts():
    return {seed: job_kit.generated2(seed)[1] for seed in SEEDS}


def gaps_of(tl) -> list[tuple[float, float]]:
    """The unobserved gaps: from the last keyframe of a task to the first of the next."""
    tasks = tl.ego.tasks
    return [(tl.keyframe(a.k1 - 1).t, tl.keyframe(b.k0).t) for a, b in zip(tasks, tasks[1:], strict=False)]


def change_times(tl) -> list[tuple[str, float]]:
    """(entity, time) of every start or end of a placement or a state after t = 0 of an entity that is not a mover (a
    resident walks in view): a thing moved, appearing or removed, a door opened or shut."""
    movers = {e.id for e in tl.entities if e.mobility == "mover"}
    out = []
    for p in tl.placements:
        if p.entity not in movers:
            out += [(p.entity, t) for t in (p.t0, p.t1) if t is not None and t > 0.0]
    for s in tl.states:
        if s.entity not in movers:
            out += [(s.entity, t) for t in (s.t0, s.t1) if t is not None and t > 0.0]
    return out


def outside_the_gaps(tl) -> list[tuple[str, float]]:
    """The changes that do not fall at a whole second inside a gap, 30 s from both its ends."""
    gaps = gaps_of(tl)
    return [(eid, t) for eid, t in change_times(tl)
            if t != int(t) or not any(lo + EDGE_BUFFER <= t <= hi - EDGE_BUFFER for lo, hi in gaps)]


def in_view(tl) -> list[tuple[str, float]]:
    """The changes that fall between two keyframes of one task, or before the first keyframe."""
    times = [kf.t for kf in tl.ego.keyframes]
    out = []
    for eid, t in change_times(tl):
        i = bisect_left(times, t)
        if not 0 < i < len(times) or tl.keyframe(i - 1).task == tl.keyframe(i).task:
            out.append((eid, t))
    return out


def test_gaps_hold_every_change(scripts):
    """Contract (plan §5.1, "gaps hold every change"): every placement or state of a non-mover entity that starts or
    ends after t = 0 does so at a whole second inside a gap, at least 30 s (`changes.EDGE_BUFFER`) from both its ends,
    and one gap holds its changes at one instant (the world's processes make one call a gap), or at two when a doorway
    coin (`job_vacuum2_doorway`) moves things as well, the second being `t_edge`, the whole second 30 s before the next
    task starts. The world really moves: over the four scripts more than 100 changes fall in the gaps, more than 10 of
    them doors. Regression caught: a change the robot could watch happen, or one that falls in a task's own time, or a
    doorway change at any other instant."""
    moved, doors = 0, 0
    for seed, tl in scripts.items():
        gaps, door = gaps_of(tl), set(door_entities(tl, M).values())
        assert outside_the_gaps(tl) == [], seed
        by_gap: dict[int, set[float]] = {}
        for eid, t in change_times(tl):
            by_gap.setdefault(next(i for i, (lo, hi) in enumerate(gaps) if lo < t < hi), set()).add(t)
            doors += eid in door
        edge = {i: float(math.floor(hi - EDGE_BUFFER)) for i, (_, hi) in enumerate(gaps)}
        assert all(len(times) == 1 or (len(times) == 2 and max(times) == edge[i]) for i, times in by_gap.items()), (
            seed, by_gap)
        moved += len(change_times(tl))
    assert moved >= 100 and doors >= 10


def test_nothing_changes_in_view(scripts):
    """Contract (plan §5.1, "nothing changes in view"): no non-mover entity's placement or state differs between two
    keyframes of one task. Every change falls between the last keyframe of one task and the first of the next, and each
    entity that ever changes (moved, appeared, removed, a door toggled) is the same at every keyframe of every task, by
    `placement_of` and `state_of`; the two checks have teeth: a placement moved to the middle of a task is found by
    both. Regression caught: a change the camera could watch happen."""
    for seed, tl in scripts.items():
        assert in_view(tl) == [], seed
        changing = {eid for eid, _ in change_times(tl)}
        assert len(changing) >= 8
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            for eid in sorted(changing):
                assert len({tl.placement_of(eid, kf.t) for kf in kfs}) == 1, (seed, task.index, eid)
                assert len({tuple(sorted(tl.state_of(eid, kf.t).items())) for kf in kfs}) == 1, (seed, task.index, eid)
    tl = scripts[101]
    eid, t = next((e, t) for e, t in change_times(tl) if t > tl.keyframe(tl.ego.tasks[1].k0).t)
    mid = tl.keyframe(tl.ego.tasks[2].k0 + 5).t
    moved = [replace(p, t0=mid) if p.entity == eid and p.t0 == t else p for p in tl.placements]
    broken = replace(tl, placements=moved)
    assert (eid, mid) in in_view(broken) and (eid, mid) in outside_the_gaps(broken)


def test_the_doors_a_task_needs_are_open_through_it_and_the_others_are_not_all_open(scripts):
    """Contract (design §8.1): every door on a way between the dock's room and a task's rooms (`doors_on_ways`) is open
    at the task's first keyframe and at its last, and the doors off them are left to the world's processes: over the
    four scripts some task has a door off its ways shut and some has all of them open, so the paths that ignore their
    state (`test_paths_ignore_door_states`) have both to ignore. Regression caught: a task that meets a door shut on its
    own way, or doors that never vary."""
    shut_off_way = open_off_way = 0
    for seed, tl in scripts.items():
        doors = door_entities(tl, M)
        for task in tl.ego.tasks:
            on = doors_on_ways(M, {"R1", *task.targets})
            first, last = tl.keyframe(task.k0).t, tl.keyframe(task.k1 - 1).t
            for door, eid in doors.items():
                if door in on:
                    assert tl.state_of(eid, first, "open") and tl.state_of(eid, last, "open"), (seed, task.index, door)
                elif M.toggling_doors() and door in {o.id for o in M.toggling_doors()}:
                    shut_off_way += not tl.state_of(eid, first, "open")
                    open_off_way += bool(tl.state_of(eid, first, "open"))
    assert shut_off_way >= 3 and open_off_way >= 3


def test_every_case_clock_ends_by_17(scripts, tmp_path):
    """Contract (plan §5.1, "the last task ends by 17:00 on the script's clock"; DECISIONS: the 7-hour span): each case
    of the four scripts (three a script, from `build.write_cases`) starts its own clock between 07:00 and 09:59 and its
    last step reads before 17:00, as does each script's own clock at its last keyframe. Regression caught: a delivered
    clock that reads night under the fixed daytime light."""
    sids = {ep.write_script(tmp_path, tl).id: tl for tl in scripts.values()}
    cases = build.write_cases(tmp_path)
    assert len(cases) == 3 * len(SEEDS)
    for tl in scripts.values():
        last = datetime.fromisoformat(tl.clock(tl.keyframe(tl.n_keyframes - 1).t))
        assert time(7) <= datetime.fromisoformat(tl.episode.start_clock).time() < time(10) and last.time() < time(17)
    for cid in cases:
        paths = case.CasePaths.from_id(tmp_path, cid)
        rec = case.read_case(paths)
        tl = sids[rec.script]
        first = datetime.fromisoformat(rec.start_clock)
        last = datetime.fromisoformat(case.case_clock(rec, tl.keyframe(rec.k1 - 1).t))
        assert time(7) <= first.time() < time(10) and last.date() == first.date() and last.time() < time(17), cid
