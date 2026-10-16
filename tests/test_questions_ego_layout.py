"""VISION rule 5 for the ego families: a thing whose place is a fact of the layout is not asked. Light furniture starts
within a metre of the place the public map gives it (`generate.light_start`), so a no-picture agent that learned the
layout answers it (`path_log` scored +0.20 on the AMR's `ego_now` over release-c's roll cages); a thing placed by a
random draw is asked as before."""

from __future__ import annotations

import pytest

from ego_kit import K, named, put, scene, world
from stbench.questions_ego import ego_mem, ego_now


def both(s):
    """The entities of the `ego_now` candidates, of the `ego_mem` answerable ones and of its unknown real things."""
    w = world(s)
    mem = ego_mem(w, K)
    unknown = {c.subjects[0][2:] for c in mem.unknown if c.subjects[0].startswith("e:")} if mem else set()
    return set(named(ego_now(w, K))), set(named(mem)), unknown


@pytest.mark.parametrize("mobility, asked", [("light", False), ("clutter", True)])
def test_a_layout_fixed_thing_is_not_asked_and_a_randomly_placed_one_of_its_kind_is(mobility, asked):
    """Contract (VISION rule 5, A3b): light furniture (a roll cage, a pallet jack, a fan on the floor) has a place that
    is a bounded draw around a public map place, so neither family asks about it, answerable or unknown; the same
    things placed by a random draw (here clutter) are asked in all three pools. By hand: a roll cage in view at the
    look point 13 (`ego_now`), a pallet jack seen at keyframe 2 and out of view in the look window (`ego_mem`), a floor
    fan no frame shows (the unknown pool), each of the mobility given. Regression caught: the AMR's `path_log` answering
    a roll cage from the map (+0.20 over `all_unknown`, over the 0.10 margin), or the screen dropping a kind that is
    placed at random."""
    s = scene()
    put(s, "N", "roll_cage", "grey", (3.0, 4.13), [K], mobility=mobility)
    put(s, "M", "pallet_jack", "oak", (5.0, 3.13), [2], mobility=mobility)
    put(s, "U", "floor_fan", "oak", (6.0, 2.0), mobility=mobility)
    put(s, "A", "bag", "red", (1.0, 4.13), [2])   # a carried bag keeps the point alive in the light case
    now, mem, unknown = both(s)
    assert ("N" in now) is asked and ("M" in mem) is asked and ("U" in unknown) is asked


def test_a_light_thing_still_hides_the_thing_it_could_be_taken_for():
    """Contract (A3b): light furniture is not asked but it is still in the picture, so it still makes a look-alike
    description ambiguous. By hand: a clutter stool in view is asked alone, and not beside a light stool. Regression
    caught: the screen removing light things from the confusers, which would ask 'the stool' with two in view."""
    s = scene()
    put(s, "S", "stool", "oak", (3.0, 4.13), [K], mobility="clutter")
    assert set(named(ego_now(world(s), K))) == {"S"}
    put(s, "L", "stool", "oak", (1.0, 1.13), [K], mobility="light")
    assert ego_now(world(s), K) is None
