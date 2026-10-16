"""`World.stale` and `World.steady` (questions_world): whether a place a thing had is out of date, and whether a stretch
of sightings keeps one place, by hand on `space_kit`'s run (one task of 28 keyframes, 1.5 s each: a thing that moves at
t 7.5 s moves at keyframe 5; the end look window is keyframes 22 to 27)."""

from __future__ import annotations

import pytest

from space_kit import inputs, record, vacuum_run
from stbench.questions_world import World
from stbench.rules import MOVED_NO_M, STALE_M

ROWS = {  # id -> (kind, where the placements put it before and after t 7.5 s, the keyframes it is recognisable at)
    "E0001": ("bag", (1.0, 1.0), (2.0, 1.0)),   # moves exactly STALE_M
    "E0002": ("bag", (1.0, 2.0), (1.9, 2.0)),   # moves 0.9 m
    "E0003": ("bag", (3.0, 1.0), (3.0, 2.0)),   # moves exactly STALE_M, along y
    "E0004": ("bag", (5.0, 1.0), (6.25, 1.0)),  # moves 1.25 m
    "E0006": ("toy", (2.0, 4.0), (2.1, 4.0)),   # moves 0.1 m: the same place
    "E0007": ("toy", (2.0, 5.0), (2.2, 5.0)),   # moves 0.2 m: another place
}


def moving_world() -> World:
    """The bags and toys of `ROWS`, each recognisable (1,500 px) at keyframe 1, before it moves, and at 10, after; and
    E0005, shoes at (1, 3) recognisable at 3 whose later sightings are not recognisable: a 500 px glimpse at keyframe
    8, where it has moved 2 m, and E0008, a bag at (4, 3) seen at 1 that is dark (lum 0.005, under the floor) at 10
    where it has moved 3 m."""
    s = vacuum_run()
    for eid, (kind, before, after) in ROWS.items():
        s.item(eid, kind, {"color": "red"}, [(0, 7.5, (*before, 0.0)), (7.5, None, (*after, 0.0))])
        s.see([1, 10], eid, 1500)
    s.item("E0005", "shoes", {"color": "red"}, [(0, 7.5, (1.0, 3.0, 0.0)), (7.5, None, (3.0, 3.0, 0.0))])
    s.see(3, "E0005", 1500)
    s.see(8, "E0005", 500)
    s.item("E0008", "bag", {"color": "red"}, [(0, 7.5, (4.0, 3.0, 0.0)), (7.5, None, (7.0, 3.0, 0.0))])
    s.see(1, "E0008", 1500)
    s.see(10, "E0008", 1500, lum=0.005)
    return World(inputs(s), record(s))


@pytest.fixture(scope="module")
def world() -> World:
    return moving_world()


def thing(w: World, eid: str):
    return next(e for e in w.tl.entities if e.id == eid)


def test_a_place_is_stale_when_a_later_recognisable_sighting_is_far_enough_away(world):
    """Contract: `stale(e, k, xy)` is true when the thing has a recognisable sighting after keyframe `k` (strictly) at
    least `STALE_M` plus the kind's widening from `xy`: `xy` is the place being judged, not the thing's placement at
    `k`. Hand values (the bag widens by 0: 0.374 / 2 - 0.25 < 0): E0001 sighted at (1, 1) at 1 and (2, 1) at 10 is
    stale at (1, 1) from k 1 and from k 9, not from k 10 (the sighting at 10 is not after 10) and not at (1.5, 1) or
    (2, 1), which are 0.5 and 0 m from where it was last seen; E0002, which moved 0.9 m, is not stale, E0003 and E0004,
    which moved 1 and 1.25 m, are. Regression caught: a stale place read as current (a moved thing answered at its old
    place, VISION: change over time), a thing stale from its own sighting, or the threshold off by a rung."""
    assert (STALE_M, MOVED_NO_M) == (1.0, 0.15)
    e = thing(world, "E0001")
    assert world.stale(e, 1, (1.0, 1.0)) and world.stale(e, 9, (1.0, 1.0))
    assert not world.stale(e, 10, (1.0, 1.0)) and not world.stale(e, 27, (1.0, 1.0))
    assert not world.stale(e, 1, (1.5, 1.0)) and not world.stale(e, 1, (2.0, 1.0))
    assert not world.stale(thing(world, "E0002"), 1, (1.0, 2.0))
    assert world.stale(thing(world, "E0003"), 1, (3.0, 1.0)) and world.stale(thing(world, "E0004"), 1, (5.0, 1.0))


def test_a_large_thing_needs_a_longer_move_to_be_stale(world, monkeypatch):
    """Contract: the distance is `STALE_M` plus `rules.widening` of the kind's catalog size, the widening every position
    rung of the kind gets. Hand-worked with a kind 1.0 x 0.4 x 0.5 m (widening max(0, 1.0 / 2 - 0.25) = 0.25, so 1.25
    m): E0001 and E0003, which moved 1 m, are not stale, E0004, which moved 1.25 m, is. Regression caught: a large
    thing called moved by the same distance as a small one, which its own extent explains."""
    monkeypatch.setattr("stbench.questions_world.item_size", lambda kind: (1.0, 0.4, 0.5))
    assert not world.stale(thing(world, "E0001"), 1, (1.0, 1.0))
    assert not world.stale(thing(world, "E0003"), 1, (3.0, 1.0))
    assert world.stale(thing(world, "E0004"), 1, (5.0, 1.0))


def test_only_a_recognisable_later_sighting_makes_a_place_stale(world):
    """Contract: a later sighting counts for `stale` only when it is recognisable (enough pixels and light): a 500 px
    glimpse and a dark sighting elsewhere do not. Hand-worked: E0005 (shoes) is recognisable at (1, 3) at keyframe 3
    and glimpsed at (3, 3), 2 m away, at 8; E0008 is recognisable at (4, 3) at 1 and dark at (7, 3), 3 m away, at 10:
    neither place is stale. Regression caught: a glimpse a person could not name moving a gold."""
    assert not world.stale(thing(world, "E0005"), 3, (1.0, 3.0))
    assert not world.stale(thing(world, "E0008"), 1, (4.0, 3.0))


def test_steady_is_every_sighting_in_the_stretch_near_the_place(world):
    """Contract: `steady(e, k, xy, until)` is true when every sighting of the thing from keyframe `k` to keyframe
    `until`, both included, lies closer than `MOVED_NO_M` to `xy`; a sighting after `until` is not looked at, and a
    stretch with none is steady. Hand values: E0001 (1, 1) at 1, (2, 1) at 10: steady at (1, 1) over 1 to 9, not to 10;
    steady at (2, 1) over 10 to 27, not at (1, 1) (the sighting at 10, the stretch's first keyframe, counts); E0006
    sighted 0.1 m apart is steady at its first place over 1 to 27, E0007 sighted 0.2 m apart is not. Regression caught:
    a thing that moved read as still, an off-by-one at either end of the stretch, or a sighting after the moment
    counted."""
    e = thing(world, "E0001")
    assert world.steady(e, 1, (1.0, 1.0), 9) and not world.steady(e, 1, (1.0, 1.0), 10)
    assert world.steady(e, 10, (2.0, 1.0), 27) and not world.steady(e, 10, (1.0, 1.0), 27)
    assert world.steady(e, 2, (1.0, 1.0), 9) and world.steady(e, 11, (1.0, 1.0), 27)  # no sighting in the stretch
    assert world.steady(thing(world, "E0006"), 1, (2.0, 4.0), 27)
    assert not world.steady(thing(world, "E0007"), 1, (2.0, 5.0), 27)


def test_any_sighting_breaks_steadiness_recognisable_or_not(world):
    """Contract: `steady` looks at every sighting, not only the recognisable ones (a thing glimpsed elsewhere is not
    known to have stayed), unlike `stale`. Hand-worked: E0005, recognisable at (1, 3) at keyframe 3 and glimpsed
    (500 px) at (3, 3) at 8, is steady at (1, 3) over 3 to 7 and not over 3 to 8; E0008, dark at (7, 3) at 10, is not
    steady at (4, 3) over 1 to 10. Regression caught: a person who walked off and was glimpsed read as still."""
    e = thing(world, "E0005")
    assert world.steady(e, 3, (1.0, 3.0), 7) and not world.steady(e, 3, (1.0, 3.0), 8)
    assert world.steady(thing(world, "E0008"), 1, (4.0, 3.0), 9)
    assert not world.steady(thing(world, "E0008"), 1, (4.0, 3.0), 10)


def test_a_thing_arrived_at_a_place_when_an_earlier_recognisable_sighting_was_far_from_it(world, monkeypatch):
    """Contract: `arrived(e, k, xy)` is `stale`'s mirror: true when the thing has a recognisable sighting before
    keyframe `k` (strictly) at least `STALE_M` plus the kind's widening from `xy`, the place the thing had at `k`; a
    glimpse and a dark sighting before `k` do not count. Hand values (the bag widens by 0): E0001, sighted at (1, 1) at
    1 and (2, 1) at 10, has arrived at (2, 1) at 10 and at 27, not at 1 (nothing before it), not at (1, 1) or (1.5, 1)
    (0.5 m from the sighting), E0002 (0.9 m) never, E0003 and E0004 (1 and 1.25 m) at their new places; E0005
    (recognisable at (1, 3) at 3, glimpsed 2 m away at 8) and E0008 (recognisable at (4, 3) at 1, dark 3 m away at 10)
    have not arrived at their first places by 12. With a kind 1.0 x 0.4 x 0.5 m (widening 0.25) a move of 1 m is not
    enough and 1.25 m is. Regression caught: a thing at the end of a move called steady (the notebook's latest answer
    asked of a thing that moved), or a glimpse counted as proof of a move."""
    e = thing(world, "E0001")
    assert world.arrived(e, 10, (2.0, 1.0)) and world.arrived(e, 27, (2.0, 1.0))
    assert not world.arrived(e, 1, (2.0, 1.0)) and not world.arrived(e, 10, (1.0, 1.0))
    assert not world.arrived(e, 10, (1.5, 1.0))
    assert not world.arrived(thing(world, "E0002"), 10, (1.9, 2.0))
    e3, e4 = thing(world, "E0003"), thing(world, "E0004")
    assert world.arrived(e3, 10, (3.0, 2.0)) and world.arrived(e4, 10, (6.25, 1.0))
    assert not world.arrived(thing(world, "E0005"), 12, (1.0, 3.0))
    assert not world.arrived(thing(world, "E0008"), 12, (4.0, 3.0))
    monkeypatch.setattr("stbench.questions_world.item_size", lambda kind: (1.0, 0.4, 0.5))
    assert not world.arrived(e, 10, (2.0, 1.0)) and not world.arrived(e3, 10, (3.0, 2.0))
    assert world.arrived(e4, 10, (6.25, 1.0))
