"""The 0.7.0 given-name check's soft goods (`descriptions.CONFUSABLE`): folded clothes, a towel, a book and a jacket
rendered alike side by side and sat in no group. `ego_now` names the home clutter by kind alone and a towel or a book by
kind and colour; `object_where` names the towel and the book by colour and screens them with the clutter sighted. Hand-
built worlds of the kits, the sets of asked things written by hand. `ego_now`'s scene is `ego_kit`'s: the camera at
(3.0, 1.13) heading north at the look point 13."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import pytest

import family_kit as fk
from ego_kit import K, named, put, scene, world
from space_kit import JOB, RELEVANT_KINDS, vacuum_run
from stbench.job_registry import Relevance
from stbench.questions_ego import ego_now
from stbench.questions_where import pools

# the three groups by hand, in the catalog's kind names
SOFT_GOODS = (("folded_clothes", "towel", "book", "book_stack"),
              ("jacket", "hanging_coat", "clothes_pile", "folded_clothes"),
              ("cable_bundle", "extension_cord"))
SOFT_JOB = replace(JOB, relevance=lambda plan: Relevance(   # a job that asks a towel and a book by colour
    RELEVANT_KINDS | {"towel", "book"}, frozenset({"kitchen", "living room"}),
    frozenset({"cleaning", "transit", "docking"})))
CLUTTER = {"folded_clothes", "book_stack", "jacket", "hanging_coat", "clothes_pile", "cable_bundle",
           "extension_cord"}  # the others are tracked kinds


def beside(kind: str, other: str, colour: str, other_colour: str) -> set[str]:
    """The ids `ego_now` asks with one `C` of `kind` and one `O` of `other` side by side in plain view."""
    s = scene()
    for eid, k, c, x in (("C", kind, colour, 4.0), ("O", other, other_colour, 2.0)):
        put(s, eid, k, c, (x, 2.13), [K], size=(0.25, 0.25, 0.55), mobility="clutter" if k in CLUTTER else "carried")
    return set(named(ego_now(world(s), K)))


@pytest.mark.parametrize("kind, other", [pair for group in SOFT_GOODS for pair in permutations(group, 2)])
def test_each_soft_goods_pair_hides_each_other_in_frame_both_ways(kind, other):
    """Contract (`descriptions.CONFUSABLE`, 0.7.0): any two kinds of a soft-goods group are look-alikes, both ways, so
    neither is asked by `ego_now` while the other, in its colour, stands in the same frame. By hand: one `C` of `kind`
    beside one `O` of `other`, both purple and in plain view; for every ordered pair of the three groups (folded
    clothes, a towel, a book and a book stack, all four with each other; a jacket, a hanging coat, a clothes pile and
    folded clothes; a cable bundle and an extension cord) the ids asked are none. Regression caught: a pair left out
    of its group or made one-way (the given-name check's A12, 'the stack of folded clothes' asked beside a yellow book
    and a grey towel; the dev pilot's log review, a towel and a book that shared a group with the stack but not with
    each other, and a cable bundle answered at an extension cord), or a group dropped."""
    assert beside(kind, other, "purple", "purple") == set()


@pytest.mark.parametrize("kind, other", [
    ("jacket", "towel"),              # a jacket is a cloth heap, no towel
    ("hanging_coat", "book"),
    ("clothes_pile", "book_stack"),
    ("folded_clothes", "mug"),        # a kind of no group
    ("cable_bundle", "folded_clothes"),
    ("extension_cord", "book"),
])
def test_the_soft_goods_groups_are_not_one_merged_class(kind, other):
    """Contract (`lookalikes` unions only the groups that hold a kind, so the groups stay apart): two kinds that share
    no group do not hide each other, though both pair with folded clothes. By hand: each pair side by side in one
    colour, a jacket with a towel, a coat with a book, a clothes pile with a book stack, folded clothes with a mug, a
    cable bundle with folded clothes and an extension cord with a book: both ids are asked. Regression caught: the
    groups merged into one soft-goods class (a book hiding a jacket), or the screen widened to every cloth or every
    cable."""
    assert beside(kind, other, "purple", "purple") == {"C", "O"}


def test_a_towel_or_book_is_hidden_only_by_a_stack_of_its_colour_and_the_stack_by_any():
    """Contract (the colour half of the screen, `questions_ego._hidden`): a thing named by kind and colour (a towel) is
    hidden by a look-alike only when that one renders in its colour, and a thing named by kind alone (the stack) is
    hidden by a look-alike of any colour, since no word tells them apart. By hand, folded clothes `C` beside a towel
    `O`: both purple hide each other (the first test); a purple stack beside a blue towel leaves the towel asked and the
    stack not, for the stack is named by kind alone. Regression caught: the towel hidden by a stack of another colour
    (a screen read by kind only, which would cut every towel in a bedroom), or the stack asked beside a towel of
    another colour (what the check found)."""
    assert beside("folded_clothes", "towel", "purple", "blue") == {"O"}
    assert beside("folded_clothes", "book", "purple", "blue") == {"O"}


def sighted_pool(kind: str, other: str, other_colour: str, other_seen: bool = True) -> set[str]:
    """The ids `object_where` asks of a blue `kind` thing E0001 beside a clutter `other` E0002 of `other_colour`."""
    s = vacuum_run()
    s.item("E0001", kind, {"color": "blue"}, [(0, None, (1.0, 1.0, 0.0))], size=(0.1, 0.1, 0.1))
    s.see([1], "E0001", 1500)
    s.item("E0002", other, {"color": other_colour}, [(0, None, (3.0, 1.0, 0.0))], size=(0.1, 0.1, 0.1),
           mobility="clutter")
    if other_seen:
        s.see([1], "E0002", 1500)
    answerable, _ = pools(fk.world(s, job=SOFT_JOB))
    return {x[2:] for c in answerable for x in c.subjects if x.startswith("e:")}


def test_object_where_asks_no_towel_or_book_beside_a_sighted_stack_of_its_colour():
    """Contract (`World.sighted_looks`, `questions_where.lookalike_sighted`): a stack of folded clothes counts, once a
    `CONFUSABLE` group holds it, among the things a sighted look-alike hides: a blue towel (or book) is not asked in
    `object_where` while a blue stack of folded clothes is sighted anywhere in the window, and is when the stack has
    another colour or no frame shows it. By hand, one task: a towel (E0001) at (1.0, 1.0) and a stack (E0002) at (3.0,
    1.0), each seen recognisably at keyframe 1; the towel is asked when the stack is purple, and not when it is blue;
    nor is a blue book beside it; and it is when the blue stack is never sighted. Regression caught: the clutter kind
    left out of `World.sighted_looks` (it was before 0.7.0: a stack in no group), so the screen reads nothing of it."""
    assert sighted_pool("towel", "folded_clothes", "purple") == {"E0001"}
    assert sighted_pool("towel", "folded_clothes", "blue") == set()
    assert sighted_pool("book", "folded_clothes", "blue") == set()
    assert sighted_pool("towel", "folded_clothes", "blue", other_seen=False) == {"E0001"}
