"""`amr/1`'s response to a blocked bay (0.5.0-control): the robot reaches T, is asked about the bay, finds a drum across
its docking line, waits and drives on. The scripts are `amr_kit.generated` (one xdist group of its own); the forced
ones are `amr_kit.quiet` and `amr_kit.forced` (the gaps' processes off, the bay coins decided by the test). By hand
from `maps/warehouse_a.json`: the world of seed 101 holds two drums off the staging lanes, so a task of that script
blocks at most two of its bays; the packing table F09's T is (14.8, 6.5), the robot facing east (heading 0)."""

from __future__ import annotations

from unittest import mock

import pytest

import amr_kit
from stbench import job_amr_drops as drops
from stbench.geometry import Rect, wrap_angle
from stbench.seeding import make_rng

pytestmark = pytest.mark.xdist_group(name="test_job_amr_skip")

M, AMR = amr_kit.M, amr_kit.AMR


@pytest.fixture(scope="module")
def scripts():
    return {seed: amr_kit.generated(seed, 3) for seed in amr_kit.SEEDS}


def test_a_skipped_visit_waits_at_t_and_drives_on(scripts):
    """Contract (0.5.0-control): a visit whose bay holds a drum runs `transit` to T and, after the `before_station`
    (the last `transit` keyframe), `waiting` at T facing the station for the seconds of its own stream (map, seed,
    "amr/1", "hold", task, visit), uniform 10 to 20, the last keyframe of the wait coming when the stream says or up
    to `min_dt_s` after it (the frame policy holds the robot until two keyframes are that far apart); the next keyframe
    is at T again in
    `transit` (the next leg starts there); no keyframe of the task stands at the docked pose; and a `before_undock`
    follows only the visits that docked. Regression caught: a skip that still approaches, a wait that is not the
    stream's, a leg that starts elsewhere, or an undock decision for a visit that never docked."""
    skips = 0
    for seed, tl in scripts.items():
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            for j, dock, hold in amr_kit.skipped_visits(tl, task):
                phases = [kf.phase for kf in hold]
                assert phases[0] == "transit" and len(phases) > 1 and set(phases[1:]) == {"waiting"}
                assert all(kf.base[:2] == dock.t and abs(wrap_angle(kf.yaw - dock.heading)) < 1e-6 for kf in hold)
                seconds = float(make_rng("warehouse_a", seed, "amr/1", "hold", task.index, j).uniform(10.0, 20.0))
                assert 10.0 <= seconds <= 20.0
                waited = hold[-1].t - hold[0].t
                assert seconds - 0.01 <= waited <= seconds + AMR.frames.min_dt_s + 0.01, (seed, task.index, j)
                after = tl.keyframe(hold[-1].k + 1)
                assert after.base[:2] == dock.t and after.phase == "transit"
                assert all(kf.base[:2] != dock.docked for kf in kfs), (seed, task.index, j)
                skips += 1
            go = sum(response == "go" for _, _, response, _ in amr_kit.visits(tl, task))
            assert sum(d.kind == "before_undock" for d in tl.decisions_in(task.k0, task.k1)) == go
    assert skips >= 3


def test_the_hold_comes_from_its_own_stream():
    """Contract (0.5.0-control, plan §14): the seconds a skipped visit waits are drawn from a stream of their own, so
    no other draw moves with them: the same script with every wait 7 seconds longer has the same world (every drum, drop
    and placement of the first task; the people, walking by the clock, differ) and the same path (the same phases,
    poses and statuses in the same order, the same decisions and responses); only the times from the first wait on
    differ. Regression caught: a wait drawn from the job's stream (every later speed and hold would move), or a path
    that depends on how long the robot waited."""
    real = amr_kit.quiet(101, 1)
    with mock.patch.object(drops, "skip_wait_s", lambda m, seed, i, j: drops.SKIP_WAIT_S[1] + 7.0):
        longer = amr_kit.quiet(101, 1)

    def route(tl):
        out = []
        for kf in tl.ego.keyframes:
            row = (kf.phase, kf.base, kf.yaw, kf.status["loaded"])
            if not out or out[-1] != row:
                out.append(row)
        return out

    def things(tl):
        return sorted((p.entity, p.t0, p.pos, p.on) for p in tl.placements if tl.entity(p.entity).mobility != "mover")

    assert things(real) == things(longer) != []
    assert route(real) == route(longer) and len(real.ego.keyframes) < len(longer.ego.keyframes)
    kinds = lambda tl: [(tl.keyframe(d.k).phase, d.kind, d.response) for d in tl.ego.decisions]  # noqa: E731
    assert kinds(real) == kinds(longer)
    assert real.ego.keyframes[-1].t < longer.ego.keyframes[-1].t


def test_loaded_follows_the_holds_the_robot_makes():
    """Contract (0.5.0-control): `loaded` turns true at the end of the first hold made and false at the end of the last
    planned stop's hold if that stop docked; a skipped last stop leaves it true to the task's end, and a visit skipped
    first moves the first hold to the next station. Forced on seed 101's three tasks (3, 3 and 2 stations): task 0
    skips only its first station, task 1 only its last, task 2 both; and, on a second script, task 0 skips its first
    two stations and docks the last (the only hold made is the last planned stop's: loaded never turns true), task 1
    skips the middle one only. By hand: with the first skipped, two holds are made and `loaded` flips after the first
    and the second; with the last skipped, one flip after the first hold, and the task ends loaded; with the first two
    skipped, no flip; with all skipped, no hold and no flip. Regression caught: a load that appears at a stop the robot
    never made, or one dropped for a stop it skipped."""
    plans = amr_kit.plans_of(101, 3)
    assert [len(p.params["stations"]) for p in plans] == [3, 3, 2]

    def run(skip: dict[int, set[int]]):
        tl = amr_kit.quiet(101, 3, coin_of=lambda m, seed, i, j: j in skip[i], rear=False)
        out = []
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            loaded = [kf.status["loaded"] for kf in kfs]
            flips = [i for i in range(1, len(kfs)) if loaded[i] != loaded[i - 1]]
            holds = [i for i in range(len(kfs) - 1)
                     if kfs[i].phase == "at_station" and kfs[i + 1].phase != "at_station"]
            out.append((flips, holds, loaded[-1]))
        return out

    (f0, h0, e0), (f1, h1, e1), (f2, h2, e2) = run({0: {0}, 1: {2}, 2: {0, 1}})
    assert len(h0) == 2 and f0 == [h0[0] + 1, h0[1] + 1] and not e0  # the first skipped: loads at the second station
    assert len(h1) == 2 and f1 == [h1[0] + 1] and e1  # the last skipped: still loaded at the end
    assert h2 == [] and f2 == [] and not e2  # everything skipped: nothing is ever loaded
    (f0, h0, e0), (f1, h1, e1), _ = run({0: {0, 1}, 1: {1}, 2: set()})
    assert len(h0) == 1 and f0 == [] and not e0  # the one hold made is the last planned stop's
    assert len(h1) == 2 and f1 == [h1[0] + 1, h1[1] + 1] and not e1


def test_rear_drops_serve_docked_visits_only():
    """Contract (0.5.0-control): `rear_drops` takes each visit's dock at the visit's index and None for a skipped visit,
    which has no corridor to reverse along and whose coin is not read: with every rear coin saying drop and nothing
    swept, a pallet or a drum is moved behind each docked visit and none behind the skipped ones, the call counts no
    miss, and the stream indices of the docked visits are theirs (a drop for visit 3 comes from `rear_at` task, 3).
    Regression caught: a drop behind a visit that never reversed, or docked visits whose indices shifted when an
    earlier one was skipped (every later draw would move)."""
    from process_kit import copy_world  # noqa: PLC0415
    from stbench import SEED_KEY  # noqa: PLC0415
    from stbench.generate import build_world, light_starts  # noqa: PLC0415
    from stbench.geometry import rects_overlap  # noqa: PLC0415

    with amr_kit.planned():
        base = build_world(M, 101)
        light_starts(base, M, make_rng(M.name, 101, SEED_KEY, "light_start"), frozenset({"charger"}))
        here = amr_kit.docks()
        docks = [None, here["F10"], None, here["F12"]]
        w = copy_world(base)
        with mock.patch.object(drops, "rear_dropped", lambda m, seed, i, j: True):
            assert drops.rear_drops(w, M, 101, 0, 500.0, docks, [], AMR, set()) == 0
        moved = [eid for eid, c in w.carried.items() if c.spot != base.carried[eid].spot]
        assert len(moved) == 2
        for dock in (here["F09"], here["F11"]):  # the corridors of the skipped visits' stations: nothing moved there
            strip = drops.extension(dock, AMR)
            assert not any(rects_overlap(Rect(w.carried[e].spot.pos[0], w.carried[e].spot.pos[1],
                                                    w.carried[e].spec.size[0], w.carried[e].spec.size[1],
                                                    w.carried[e].spot.yaw), strip) for e in moved)
