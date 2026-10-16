"""The downtown's processes between flights (`world_processes_downtown`): who a gap draws again and at what rate, that a
gap's changes are one instant inside it, that nothing is added or removed and that the world stays valid through many
gaps. The worlds are `downtown_a`'s seeds 101-103 as `generate.build_world` makes them (the downtown has no clutter and
no light furniture to place). Rates are written out here from the declared half-lives: cars, vans, pickups and box
trucks 1 h, food carts 6 h, skips, dumpsters and portable toilets 48 h."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from process_kit import copy_world
from stbench import world_processes_downtown as downtown
from stbench.catalog import DOWNTOWN_KINDS
from stbench.generate import build_world, to_timeline
from stbench.mapspec import load_map
from stbench.seeding import make_rng
from stbench.timeline import EpisodeInfo
from stbench.timeline_ego import Ego
from stbench.validate import validate
from stbench.world_processes import gaps, half_life_p

pytestmark = pytest.mark.xdist_group(name=__name__)
M = load_map("downtown_a")
HOUR = 3600.0
HALF_LIFE_H = {"car": 1, "van": 1, "pickup": 1, "box_truck": 1, "food_cart": 6, "skip": 48, "dumpster": 48,
               "portable_toilet": 48}


def fresh(seed: int):
    """A world that a test may change."""
    return copy_world(build_world(M, seed))


def within(hits: int, n: int, p: float) -> bool:
    """Whether `hits` of `n` independent draws lies within 4 standard errors of the chance `p`."""
    return abs(hits / n - p) <= 4 * math.sqrt(p * (1 - p) / n)


def test_every_kind_has_the_declared_half_life():
    """Contract (M3-Y §5): the downtown's eight kinds each have a half-life, the ones written above, and each constant
    names its source or says "modelling choice". Regression caught: a kind a catalog edit adds that no process draws
    again (it would stand where it is for ever), or a rate with no source."""
    assert set(downtown.HALF_LIFE_S) == set(DOWNTOWN_KINDS) == set(HALF_LIFE_H)
    assert {k: v / HOUR for k, v in downtown.HALF_LIFE_S.items()} == {k: float(v) for k, v in HALF_LIFE_H.items()}
    assert set(downtown.SOURCES) == {"H_VEHICLE_S", "H_CART_S", "H_FIXTURE_S"}
    assert all("modelling choice" in text and "PRIOR-ART" in text for text in downtown.SOURCES.values())


def test_a_gap_draws_each_thing_at_its_kinds_rate():
    """Contract: a thing of half-life H is drawn again over a gap g with chance 1 - 2^(-g/H) (by hand 1/2 for a vehicle
    over an hour, 87.5% over three, 11% for a cart over an hour, 1.4% for a skip), whatever else the world holds: over
    1,000 draws of a 10-minute, a 1-hour and a 3-hour gap on world 101, each kind's things are picked that share
    within 4 standard errors. Regression caught: a rate that ignores the gap's length, or a kind on another half-life
    than declared."""
    assert half_life_p(HOUR, HOUR) == 0.5 and half_life_p(3 * HOUR, HOUR) == 0.875
    w = fresh(101)
    kind = {eid: c.spec.kind for eid, c in w.carried.items()}
    held = Counter(kind.values())
    assert len(w.carried) >= 26 and all(held[k] >= 1 for k in ("car", "van", "box_truck", "portable_toilet")), held
    rng = make_rng("test_a_gap_draws_each_thing_at_its_kinds_rate")
    for hours in (1 / 6, 1.0, 3.0):
        got = Counter()
        for _ in range(1000):
            got.update(kind[eid] for eid in downtown.movers(rng, w, hours * HOUR))
        for name, count in held.items():
            assert within(got[name], 1000 * count, 1 - 2 ** (-hours / HALF_LIFE_H[name])), (hours, name, got[name])


def test_one_gap_is_one_instant_and_nothing_is_added_or_removed():
    """Contract (M2-A2W §2, "one instant a gap"): `start` changes nothing (no placement, no state); over four gaps of 10
    minutes to 3 hours every placement a gap call adds starts at one and the same whole second at least 30 s from both
    ends of the gap, so no flight sees one happen; no thing is added or removed (the same carried things, each placed
    at the end). A gap of under a minute is refused. Regression caught: a change with a time of its own, a change in
    the 30 s next to a flight, or a thing that vanishes."""
    w = fresh(102)
    between, ids = gaps(w, M, 102), set(w.carried)
    before = w.ledger.finish()
    between(None, 0.0, frozenset({"R4"}))
    assert (w.ledger.finish(), set(w.carried)) == (before, ids)
    moved, edges = 0, [(100.0, 700.0), (700.0 + 3600.0, 4300.0 + 7200.0), (20000.0, 30800.0), (40000.0, 40060.0)]
    for lo, hi in edges:
        old = {(p.entity, p.t0) for p in w.ledger.finish()[0]}
        between(lo, hi, frozenset({"R4"}))
        new = [p for p in w.ledger.finish()[0] if (p.entity, p.t0) not in old]
        times = {p.t0 for p in new}
        assert len(times) <= 1, (lo, hi, sorted(times))
        for t in times:
            assert t == int(t) and lo + 30 <= t <= hi - 30
        moved += len(new)
    assert moved >= 6 and set(w.carried) == ids
    placed = Counter(p.entity for p in w.ledger.finish()[0] if p.t1 is None)
    assert all(placed[e] == 1 for e in ids)  # each thing is somewhere at the end
    with pytest.raises(ValueError, match="shorter than 60"):
        between(50000.0, 50059.0, frozenset({"R4"}))


def test_the_world_stays_valid_through_many_gaps():
    """Contract: after five gaps of 1 to 3 hours on each of worlds 101-103 the world `validate` reads (no flight in it)
    has no problem: every thing is on a surface of the map, wholly on it, and no two things overlap at any time.
    Regression caught: a process that lands a vehicle on another, off its surface, or on one it does not fit."""
    for seed in (101, 102, 103):
        w = fresh(seed)
        between, lo = gaps(w, M, seed), 0.0
        between(None, 0.0, frozenset({"R4"}))
        for k in range(5):
            hi = lo + (1 + k % 3) * HOUR
            between(lo + 1.0, hi, frozenset({"R4"}))
            lo = hi
        info = EpisodeInfo(M.name, seed, "drone", "drone/1", "2026-01-01T08:00:00")
        assert validate(to_timeline(M, w, Ego([], [], []), info), M) == [], seed
