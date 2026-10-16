"""The 0.6.0 look-dev review's fixes: two home clutter pairs that render alike hide each other in
`ego_now`, and a kind named by a plural noun is worded as a unit. The scene is `ego_kit`'s: the camera at (3.0, 1.13)
heading north at the look point 13."""

from __future__ import annotations

import pytest

from ego_kit import K, named, put, scene, world
from stbench.catalog import CATALOGS
from stbench.descriptions import lookalikes, phrase
from stbench.questions_ego import ego_now


def test_the_home_clutter_pairs_that_render_alike_are_each_others_lookalikes():
    """Contract: an air purifier and a space heater, a pedal bin and a bath bin are look-alike pairs in
    `descriptions.CONFUSABLE`, symmetric, and only pairs: the pedal bin takes no bin or drum, even for a thing never
    seen, and the bin group takes neither pair. Hand values: the purifier's are itself and the heater, the pedal bin's
    itself and the bath bin, and the reverses. Regression caught: a pair left out (the heater asked where the purifier
    stands), a pair made one-way, or a pair merged into the bin group (a pedal bin hiding a bin)."""
    assert lookalikes("air_purifier") == {"air_purifier", "space_heater"} == lookalikes("space_heater")
    assert lookalikes("pedal_bin") == {"pedal_bin", "bath_bin"} == lookalikes("bath_bin")
    assert lookalikes("pedal_bin", unseen=True) == {"pedal_bin", "bath_bin"}
    assert "pedal_bin" not in lookalikes("bin", unseen=True) and "air_purifier" not in lookalikes("box")


@pytest.mark.parametrize("kind, other, asked", [
    ("air_purifier", "space_heater", set()),     # a heater stands for the purifier, and the reverse
    ("space_heater", "air_purifier", set()),
    ("pedal_bin", "bath_bin", set()),            # a bath bin renders as a pedal bin, and the reverse
    ("bath_bin", "pedal_bin", set()),
    ("air_purifier", "floor_fan", {"C", "O"}),   # a fan is no purifier: both are asked
    ("pedal_bin", "bin", {"C", "O"}),            # a plain bin is no pedal bin (the groups are not merged): both asked
])
def test_a_clutter_pair_that_renders_alike_hides_each_other(kind, other, asked):
    """Contract: clutter is named by kind alone, and a purifier and a heater (two grey or black floor
    towers), a pedal bin and a bath bin (two pedal-bin cylinders of one colour range) cannot be told apart by the
    words, so neither is asked while the other is in frame, whatever its colour. By hand: one 1,500 px clutter thing "C"
    of the kind beside one "O" of the other kind; the ids asked are none for the four look-alike pairs and both for a
    fan beside a purifier and a bin beside a pedal bin, which `CONFUSABLE` does not pair. Regression caught: 'the air
    purifier' asked with a heater in plain view (helper case f9db83edda189b7a: the purifier mostly hidden, the grey
    heater in the open, an answer 2.4 m off), the same for a bath bin beside a pedal bin, or a new group merged into
    another."""
    s = scene()
    put(s, "C", kind, "grey", (4.0, 2.13), [K], size=(0.25, 0.25, 0.55), mobility="clutter")
    put(s, "O", other, "black", (2.0, 2.13), [K], size=(0.25, 0.25, 0.55),
        mobility="carried" if other == "bin" else "clutter")
    assert set(named(ego_now(world(s), K))) == asked


@pytest.mark.parametrize("kind, attrs, one, many", [
    ("dumbbells", {"color": "grey"}, "pair of grey dumbbells", "pairs of grey dumbbells"),
    ("folded_clothes", {}, "stack of folded clothes", "stacks of folded clothes"),
    ("shoes", {"color": "red"}, "pair of red shoes", "pairs of red shoes"),
])
def test_a_kind_named_by_a_plural_noun_is_worded_as_a_unit(kind, attrs, one, many):
    """Contract: a kind whose name is a plural noun is worded as a unit, so "Where is the ..." takes its
    singular verb: "the pair of grey dumbbells", "the stack of folded clothes", never "the dumbbells". Regression
    caught: 'Where is the dumbbells you can see now' (vacuum 613578, helper 0064f1 and 7e8771)."""
    assert phrase(kind, attrs) == one and phrase(kind, attrs, plural=True) == many


def test_no_catalog_kind_is_a_bare_plural_noun():
    """Contract: every kind of every catalog (tracked or clutter) whose last word ends in 's' (not 'ss', as
    a glass) is a unit in `phrase`. By hand the kinds are the dumbbells, the folded clothes and the shoes. Regression
    caught: a plural-named kind added to a catalog and asked as 'Where is the <plural>'."""
    names = {k for c in CATALOGS.values() for k in (*c.kinds, *(x.name for x in c.clutter))}
    plural = {k for k in names if k.split("_")[-1].endswith("s") and not k.endswith("ss")}
    assert plural == {"dumbbells", "folded_clothes", "shoes"}
    assert all(phrase(k, {}).split()[0] in {"pair", "stack"} for k in plural)
