"""`world_processes.gaps`, the engine behind the world's changes between tasks: which call draws from which key, what a
catalog's module is handed, and what a catalog with no module does. The apartment's doors are as `maps/apartment_a.json`
has them (D1 kitchen, D2 living room, D3 bathroom, D4 bedroom, each from the hall R1; D0 the entrance); the doors on a
way are worked by hand in `test_doors_on_ways.py`."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from realism_kit import apartment
from stbench import SEED_KEY, world_processes
from stbench.generate import build_world
from stbench.seeding import make_rng
from stbench.world_processes import gaps


def test_each_call_draws_from_its_own_key_and_the_module_is_handed_the_previous_tasks_doors(monkeypatch):
    """Contract (M2-A2W §1): the i-th call of the callback draws from `make_rng(map, seed, SEED_KEY, "world", i)`; the
    first call (lo None) is the catalog module's `start`, each later one its `gap` with `lo`, `hi` and `keep_open` as
    given and `forced`, the doors forced open for the previous task, which the callback keeps: those of
    `doors_on_ways` of the previous call's `keep_open` (hand cases: {hall, kitchen} gives D1 and D2, {hall, bedroom}
    D4). A gap too short for an instant 30 s from both ends (under 60 s) is refused. Regression caught: two calls on one
    stream (a gap that repeats the last one's draws), a key by task and not by call, a forced set taken from the
    current call's rooms (a door that stays open however often it is drawn again), or a one-second gap whose change
    the robot could watch."""
    calls = []
    module = SimpleNamespace(
        start=lambda w, m, rng, keep_open: calls.append(("start", float(rng.random()), keep_open)),
        gap=lambda w, m, rng, lo, hi, keep_open, forced: calls.append(("gap", float(rng.random()), keep_open, lo, hi,
                                                                       forced)))
    monkeypatch.setattr(world_processes, "_home", lambda catalog: module)
    m = apartment()
    between = gaps(build_world(m, 7), m, 7)
    kitchen, bedroom, hall = frozenset({"R1", "R3"}), frozenset({"R1", "R4"}), frozenset({"R1"})
    between(None, 0.0, kitchen)
    between(100.0, 4000.0, bedroom)
    between(4100.0, 9000.0, hall)
    first = [float(make_rng("apartment_a", 7, SEED_KEY, "world", i).random()) for i in range(3)]
    assert calls == [("start", first[0], kitchen),
                     ("gap", first[1], bedroom, 100.0, 4000.0, frozenset({"D1", "D2"})),
                     ("gap", first[2], hall, 4100.0, 9000.0, frozenset({"D4"}))]
    with pytest.raises(ValueError, match="60"):
        between(9100.0, 9159.0, hall)
    assert len(calls) == 3


def test_a_catalog_with_no_module_only_opens_the_doors_on_the_ways():
    """Contract (M2-A2W §1): a catalog without a `world_processes_<catalog>` module changes nothing but its doors: every
    door of `doors_on_ways(m, keep_open)` that is shut is opened, at 0 for the first call and at the call's own
    instant (drawn from its key) for a gap, and no other door is touched. Hand case on the apartment's doors (the map
    given a catalog that has no module): with D1 shut from 100 s and D3 shut from 0, the first call for
    {hall, bedroom} opens nothing at 0 (D4 is open), and the gap (200, 5000) for {hall, kitchen} opens D1 at the whole
    second its key draws in [230, 4970], leaves D2 alone (open, no new state) and D3 shut (off the ways). Regression
    caught: a door on the next task's way left shut (a detour that leaks it), a door off the ways opened, or an open
    door rewritten."""
    m = replace(apartment(), catalog="nowhere")
    w = build_world(apartment(), 1)
    d = w.door_entity
    w.ledger.set_state(d["D1"], "open", 100.0, False)
    w.ledger.set_state(d["D3"], "open", 0.0, False)

    def states(door: str) -> list[tuple]:
        return [(s.t0, s.t1, s.value) for s in w.ledger.finish()[1] if s.entity == d[door]]

    between = gaps(w, m, 1)
    between(None, 0.0, frozenset({"R1", "R4"}))
    assert states("D4") == [(0.0, None, True)] and states("D3") == [(0.0, None, False)]
    t = float(make_rng("apartment_a", 1, SEED_KEY, "world", 1).integers(230, 4971))
    between(200.0, 5000.0, frozenset({"R1", "R3"}))
    assert states("D1") == [(0.0, 100.0, True), (100.0, t, False), (t, None, True)] and 230 <= t <= 4970
    assert states("D2") == [(0.0, None, True)] and states("D3") == [(0.0, None, False)]
    assert states("D0") == [(0.0, None, False)]
