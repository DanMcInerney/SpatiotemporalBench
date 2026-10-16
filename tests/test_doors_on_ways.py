"""`world_processes.doors_on_ways`: the one rule of which doors a task needs open, read off the map's room graph. The
expected sets are worked by hand from `maps/apartment_a.json`: the hall R1 joins the kitchen R3 by D1, the living room
R2 by D2, the bathroom R5 by D3 and the bedroom R4 by D4; an arch O1 joins the living room and the kitchen; D0 is the
entrance."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench.mapspec import MapSpec, Opening, Region, load_map
from stbench.world_processes import doors_on_ways

SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def door(door_id: str, a: str, b: str, *, interior: bool = True) -> Opening:
    return Opening(door_id, "door", "W", 1.0, 0.9, 2.0, "a", "left", interior, False, (a, b), (0.0, 0.0), 0.0, b)


def three_rooms() -> MapSpec:
    """A, B and C in a row: A and B joined by D9, B and C by D8, and the entrance D0 from outside into A."""
    rooms = tuple(Region(r, r, SQUARE, (0.5, 0.5), 0, 0.0, 2.5) for r in "ABC")
    openings = (door("D0", "outside", "A", interior=False), door("D9", "A", "B"), door("D8", "B", "C"))
    return replace(load_map("apartment_a"), name="three_rooms", regions=rooms, openings=openings)


def test_doors_on_ways_by_hand():
    """Contract (design §8.1, "paths leak"): the doors of `doors_on_ways(m, regions)` are the interior doors on some
    simple path between two of the regions, so a door the robot's path could need is never left free to toggle. By hand,
    on apartment_a: {hall, kitchen} and {hall, living room} each give D1 and D2 (the kitchen is reached by D1, or by
    D2, the living room and the arch O1, and the living room by D2 or by D1, the kitchen and the arch), {hall, bedroom}
    D4, {hall, bathroom} D3, {hall} none, {hall, kitchen, bedroom} D1, D2 and D4; the arch is never a door of the set,
    nor is the entrance D0 (even with every room asked); on a row of three rooms a door past the last room asked is
    left out. Regression caught: a door on a way left out (a path that detours round it leaks it), a door off every way
    kept (a door that could never toggle), the entrance or an arch in the set, a set that depends on the order of the
    regions."""
    m = load_map("apartment_a")
    assert doors_on_ways(m, {"R1", "R3"}) == {"D1", "D2"}
    assert doors_on_ways(m, ["R1", "R2"]) == {"D1", "D2"}
    assert doors_on_ways(m, ["R3", "R2"]) == {"D1", "D2"}  # the arch O1 is a way, but not a door
    assert doors_on_ways(m, ["R1", "R4"]) == {"D4"}
    assert doors_on_ways(m, ["R1", "R5"]) == {"D3"}
    assert doors_on_ways(m, ["R1"]) == frozenset()
    assert doors_on_ways(m, []) == frozenset()
    assert doors_on_ways(m, ["R1", "R3", "R4"]) == {"D1", "D2", "D4"}
    assert doors_on_ways(m, ["R4", "R3", "R1"]) == {"D1", "D2", "D4"}
    assert doors_on_ways(m, ["R1", "R2", "R3", "R4", "R5"]) == {"D1", "D2", "D3", "D4"}
    three = three_rooms()
    assert doors_on_ways(three, ["A", "B"]) == {"D9"}  # B's other door, to C, is off the way
    assert doors_on_ways(three, ["A", "C"]) == {"D9", "D8"}


def test_doors_on_ways_refuses_an_unknown_region():
    """Contract: a region the map does not have is a KeyError naming it, not an empty set. Regression caught: a
    mistyped region id answered 'no door is on a way', which leaves every door free to toggle on the robot's path."""
    with pytest.raises(KeyError, match="R9"):
        doors_on_ways(load_map("apartment_a"), ["R1", "R9"])
