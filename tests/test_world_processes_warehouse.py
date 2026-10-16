"""The warehouse's processes between tasks (`world_processes_warehouse`): who a gap draws again and at what rates, that
a gap's changes are one instant, that nothing is added or removed, that light equipment keeps off the charger's apron
and the lanes, and that every change in a real `amr/1` script falls inside a gap and never in view. The worlds are
`warehouse_a`'s seeds 101-103 as `generate_script` builds one before it plans a path; rates are written out here from
the declared half-lives (pallets 2 h; totes, bins and boxes 3 h; drums and sacks 8 h; pallet jacks and the roll cage 1
h)."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter

import pytest

import amr_kit
from process_kit import HOUR, copy_world
from stbench import SEED_KEY
from stbench import world_processes_warehouse as wh
from stbench.changes import EDGE_BUFFER, when_in_gap
from stbench.generate import build_world, light_starts
from stbench.geometry import Rect, rects_overlap, to_world
from stbench.seeding import make_rng
from stbench.world_processes import gaps, half_life_p

pytestmark = pytest.mark.xdist_group(name="test_job_amr")  # one worker builds the scripts once, every test reads them

M = amr_kit.M


def within(hits: int, n: int, p: float) -> bool:
    """Whether `hits` of `n` independent draws lies within 4 standard errors of the chance `p`."""
    return abs(hits / n - p) <= 4 * math.sqrt(p * (1 - p) / n)


def base_world(seed: int):
    """The seed's world as `generate_script` has it before the job plans: built, the light equipment started (the AMR
    planned on the warehouse, as M3-RW will have it)."""
    with amr_kit.planned():
        w = build_world(M, seed)
        light_starts(w, M, make_rng(M.name, seed, SEED_KEY, "light_start"), frozenset({"charger"}))
    return w


def test_half_life_law():
    """Contract (M3-W §5): a thing of half-life H is drawn again over a gap g with chance 1 - 2^(-g/H); by hand a pallet
    (2 h) over 1 h is 1 - 2^(-1/2) = 0.2929, a tote, bin or box (3 h) over 3 h is 0.5, a drum or sack (8 h) over 4 h is
    0.2929 and a pallet jack or the roll cage (1 h) over 1 h is 0.5; and each process's selection, over 1,000 draws of a
    1-hour, a 3-hour and an 8-hour gap, picks that share of its things within 4 standard errors, kind by kind. The
    constants are the declared ones and each has a note of its source. Regression caught: a rate that ignores the gap's
    length, a kind on another half-life than declared, or a thing the draw can never reach."""
    assert (wh.H_PALLET_S, wh.H_TOTE_S, wh.H_DRUM_S, wh.H_LIGHT_S) == (2 * HOUR, 3 * HOUR, 8 * HOUR, HOUR)
    assert wh.HALF_LIFE_S == {"pallet": 2 * HOUR, "tote": 3 * HOUR, "bin": 3 * HOUR, "box": 3 * HOUR, "drum": 8 * HOUR,
                              "sack": 8 * HOUR}
    assert half_life_p(HOUR, 2 * HOUR) == pytest.approx(0.2929, abs=1e-4)
    assert half_life_p(3 * HOUR, 3 * HOUR) == 0.5 and half_life_p(4 * HOUR, 8 * HOUR) == pytest.approx(0.2929, abs=1e-4)
    assert half_life_p(HOUR, HOUR) == 0.5
    assert set(wh.SOURCES) == {"H_PALLET_S", "H_TOTE_S", "H_DRUM_S", "H_LIGHT_S"}
    assert all("modelling choice" in note for note in wh.SOURCES.values())
    w = base_world(101)
    kinds = Counter(c.spec.kind for c in w.carried.values())
    assert sum(kinds.values()) >= 22 and len(w.light_rects) == 3 and set(kinds) <= set(wh.HALF_LIFE_S)
    rng = make_rng("test_half_life_law")
    for gap_h in (1.0, 3.0, 8.0):
        got, light = Counter(), 0
        for _ in range(1000):
            for eid in wh.item_movers(rng, w, gap_h * HOUR):
                got[w.carried[eid].spec.kind] += 1
            light += len(wh.light_movers(rng, w, gap_h * HOUR))
        for kind, n in kinds.items():
            p = 1 - 2 ** (-gap_h * HOUR / wh.HALF_LIFE_S[kind])
            assert within(got[kind], 1000 * n, p), (gap_h, kind, got[kind])
        assert within(light, 3000, 1 - 2 ** (-gap_h)), (gap_h, light)


def test_a_gap_is_one_instant_and_adds_and_removes_nothing():
    """Contract (M3-W §5): `start` writes nothing (the hall has no doors); one `gap` call makes all its changes at one
    whole second drawn by `changes.when_in_gap` from its key, at least 30 s from both ends of the gap, adds and removes
    no entity, and moves only light equipment and carried things; over 8 two-hour gaps from one world something moves
    in most of them. Regression caught: a change in view, a change on two instants, or a thing added or removed."""
    w = copy_world(base_world(102))
    before = w.ledger.finish()
    wh.start(w, M, make_rng("start"), frozenset({"R1"}))
    assert w.ledger.finish() == before
    names = {e.id for e in w.entities}
    moved_gaps = 0
    with amr_kit.planned():
        for i in range(8):
            lo, hi = 1000.0 + i * 8000.0, 1000.0 + i * 8000.0 + 7200.0
            done = wh.gap(w, M, make_rng("gap", i), lo, hi, frozenset({"R1"}))
            t = when_in_gap(make_rng("gap", i), lo, hi)
            assert set(done) == {"light", "items"} and lo + EDGE_BUFFER <= t <= hi - EDGE_BUFFER
            starts = {p.t0 for p in w.ledger.finish()[0] if lo < p.t0 < hi}
            assert starts <= {t}, (i, starts, t)
            moved_gaps += bool(starts)
    assert {e.id for e in w.entities} == names and {p.entity for p in w.ledger.finish()[0]} == names
    assert moved_gaps >= 6


def test_light_equipment_keeps_off_the_charger_and_the_lanes():
    """Contract (M3-W §5, §1): after 12 one-hour gaps pallet jacks and the roll cage have been drawn from their map
    places again and again, and none ever overlaps a staging lane or the charger's apron (1.2 m in front of the
    charger's face and 0.5 m to each side, so x 16.18-17.50 and y 6.30-7.90 on this map). Regression caught: a light
    piece parked where the AMR cannot back off its charger, or on a lane the processes alone fill."""
    w = copy_world(base_world(103))
    apron = Rect(*to_world(Rect(17.44, 7.10, 0.12, 0.60, math.pi), 0.6, 0.0), 0.12 + 1.2, 0.60 + 1.0, math.pi)
    lanes, moved = amr_kit.lane_rects(), 0
    with amr_kit.planned():
        for i in range(12):
            lo = 100.0 + i * 4000.0
            moved += wh.gap(w, M, make_rng("light", i), lo, lo + HOUR, frozenset({"R1"}))["light"][1]
            for rect in w.light_rects.values():
                assert not rects_overlap(rect, apron) and not amr_kit.overlapping(rect, lanes), (i, rect)
    assert moved >= 8  # the pieces really moved


def test_the_warehouse_module_is_the_one_gaps_loads():
    """Contract (`world_processes.gaps`): a map of the `warehouse` catalog loads `world_processes_warehouse`: its
    callback writes nothing at the first call and, for a gap, changes things at one instant. Regression caught: a
    catalog whose module is never found (the processes silently off)."""
    w = copy_world(base_world(101))
    between = gaps(w, M, 101)
    before = w.ledger.finish()
    between(None, 0.0, frozenset({"R1"}))
    assert w.ledger.finish() == before
    with amr_kit.planned():
        between(500.0, 9500.0, frozenset({"R1", "R3"}))
    changed = {p.t0 for p in w.ledger.finish()[0] if p.t0 > 0.0}
    assert len(changed) == 1 and 530.0 <= changed.pop() <= 9470.0


def gaps_of(tl) -> list[tuple[float, float]]:
    """The unobserved gaps: from the last keyframe of a task to the first of the next."""
    tasks = tl.ego.tasks
    return [(tl.keyframe(a.k1 - 1).t, tl.keyframe(b.k0).t) for a, b in zip(tasks, tasks[1:], strict=False)]


def change_times(tl) -> list[tuple[str, float]]:
    """(entity, time) of every start or end of a placement or a state after t = 0 of an entity that is not a mover."""
    movers = {e.id for e in tl.entities if e.mobility == "mover"}
    out = []
    for p in tl.placements:
        if p.entity not in movers:
            out += [(p.entity, t) for t in (p.t0, p.t1) if t is not None and t > 0.0]
    for s in tl.states:
        if s.entity not in movers:
            out += [(s.entity, t) for t in (s.t0, s.t1) if t is not None and t > 0.0]
    return out


def test_gaps_hold_every_change():
    """Contract (plan §5.2, "gaps hold every change"): in real `amr/1` scripts of seeds 101-103 (3 tasks, the bay coin,
    the rear drops and the processes on) every placement of a non-mover that starts or ends after t = 0 does so at a
    whole second inside a gap, at least 30 s (`changes.EDGE_BUFFER`) from both its ends, and no such change falls
    between two keyframes of one task or before the first keyframe. The world really moves: more than 20 changes fall in
    the six gaps. Regression caught: a change the robot could watch happen, or one that falls in a task's own time."""
    moved = 0
    for seed in amr_kit.SEEDS:
        tl = amr_kit.generated(seed, 3)
        gaps_ = gaps_of(tl)
        times = [kf.t for kf in tl.ego.keyframes]
        changes = change_times(tl)
        moved += len(changes)
        for eid, t in changes:
            assert t == int(t) and any(lo + EDGE_BUFFER <= t <= hi - EDGE_BUFFER for lo, hi in gaps_), (seed, eid, t)
            i = bisect_left(times, t)
            assert 0 < i < len(times) and tl.keyframe(i - 1).task != tl.keyframe(i).task, (seed, eid, t)
    assert moved > 20
