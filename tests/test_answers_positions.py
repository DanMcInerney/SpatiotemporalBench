"""The `positions` answer type: a list of places, no change types, don't-care on a gold."""

from __future__ import annotations

import pytest

from stbench.answers import (
    DontCare,
    Malformed,
    Position,
    Positions,
    Unknown,
    answer_type_of,
    parse_answer,
    parse_gold,
    to_json,
)

PLACES = [
    ({"type": "positions", "items": []}, Positions(())),
    ({"type": "positions", "items": [{"xy": [1, 2]}]}, Positions((Position((1.0, 2.0)),))),
    ({"type": "positions", "items": [{"xyz": [1, 2, 0.5]}, {"xy": [-3.5, 4]}]},
     Positions((Position((1.0, 2.0, 0.5)), Position((-3.5, 4.0))))),
]


@pytest.mark.parametrize(("obj", "expected"), PLACES)
def test_a_positions_answer_parses_to_places_with_or_without_height_and_round_trips(obj, expected):
    """Contract: each item of a `positions` answer is `{xy}` or `{xyz}` and parses to a typed point (ints widen to
    floats), the whole to `Positions`; `to_json` writes the same object back. Regression caught: items kept as raw
    dicts (equal answers comparing unequal), an `xyz` item losing its height or becoming `xy` on the way back out."""
    parsed = parse_answer(obj)
    assert parsed == expected and answer_type_of(parsed) == "positions"
    assert to_json(parsed) == obj and parse_answer(to_json(parsed)) == parsed


def test_a_hundred_places_are_the_most():
    """Contract: a positions answer holds at most 100 items, as a set does (MAX_SET_ITEMS). Regression caught: no
    bound, so one answer could carry a grid of guesses that matches any gold."""
    item = {"xy": [0, 0]}
    assert isinstance(parse_answer({"type": "positions", "items": [item] * 100}), Positions)
    assert isinstance(parse_answer({"type": "positions", "items": [item] * 101}), Malformed)


BAD_PLACES = [
    {"type": "positions"},
    {"type": "positions", "items": "none"},
    {"type": "positions", "items": [[1, 2]]},
    {"type": "positions", "items": [{"xy": [0, 0], "change": "added"}]},   # a set's item
    {"type": "positions", "items": [{"xy": [0, 0], "xyz": [0, 0, 0]}]},
    {"type": "positions", "items": [{"type": "position", "xy": [0, 0]}]},
    {"type": "positions", "items": [{"xy": [0, 0], "note": "x"}]},
    {"type": "positions", "items": [{}]},
    {"type": "positions", "items": [{"xy": [0]}]},
    {"type": "positions", "items": [{"xy": [0, 0, 0]}]},
    {"type": "positions", "items": [{"xyz": [0, 0]}]},
    {"type": "positions", "items": [{"xy": [True, 0]}]},
    {"type": "positions", "items": [{"xy": [float("nan"), 0]}]},
    {"type": "positions", "items": [], "extra": 1},
    {"type": "positions", "items": [], "dont_care": []},
]


@pytest.mark.parametrize("obj", BAD_PLACES, ids=range(len(BAD_PLACES)))
def test_positions_schema_violations_are_malformed(obj):
    """Contract: an item with a `change` (a set's), a `type`, both `xy` and `xyz`, a wrong length, a bool or non-finite
    coordinate or no coordinates, a non-list `items`, an extra key and a `dont_care` on an answer are all Malformed,
    with a reason. Regression caught: a set's item accepted on an `inventory` question (a change type nobody scores), a
    lenient parser crediting `[True, 0]`, or an agent supplying its own don't-care regions."""
    parsed = parse_answer(obj)
    assert isinstance(parsed, Malformed) and parsed.reason


def test_a_positions_answer_and_a_set_answer_are_not_interchangeable():
    """Contract: a `positions` answer to a `set` question is Malformed, and a `set` answer to a `positions` question
    too (`unknown` is always allowed); each type is also refused the other's items. Regression caught: the two list
    shapes merged, so a question that wants events is answered with places, or the reverse."""
    places, events = {"type": "positions", "items": [{"xy": [0, 0]}]}, {"type": "set", "items": []}
    assert isinstance(parse_answer(places, "set"), Malformed)
    assert isinstance(parse_answer(events, "positions"), Malformed)
    assert parse_answer(places, "positions") == Positions((Position((0.0, 0.0)),))
    assert parse_answer({"type": "unknown"}, "positions") == Unknown()
    assert isinstance(parse_answer({"type": "set", "items": [{"xy": [0, 0]}]}), Malformed)  # no change type


def test_a_gold_may_carry_dont_care_regions_and_an_answer_may_not():
    """Contract: a gold `positions` parses `dont_care` regions (and `to_json` keeps them), an answer with them is
    Malformed, and a bad region raises. Regression caught: a gold's regions dropped when it is re-written (the scorer
    would then count the ambiguous places as misses), or an agent excusing its own false positives."""
    gold = {"type": "positions", "items": [{"xy": [0, 0]}], "dont_care": [{"xy": [5, 5], "radius": 0.5}]}
    parsed = parse_gold(gold)
    assert parsed == Positions((Position((0.0, 0.0)),), (DontCare((5.0, 5.0), 0.5),))
    assert to_json(parsed) == gold
    assert isinstance(parse_answer(gold), Malformed)
    for bad in ({"xy": [5, 5], "radius": 0}, {"xy": [5, 5]}, "near"):
        with pytest.raises(ValueError, match="invalid gold"):
            parse_gold({"type": "positions", "items": [], "dont_care": [bad]})


ALL_TYPES = ("['position', 'offset', 'time', 'yesno', 'count', 'number', 'set', 'positions', 'cells', 'route', "
             "'heading', 'unknown']")  # by hand


def test_with_no_question_named_positions_is_a_type_and_a_refusal_lists_all_twelve():
    """Contract (plan §2.6, one list): where no question is named (a gold, a scorer's raw answer) `positions` is a
    type, and the refusal of an unknown type lists every type of `answers.TYPES`, `positions` among them and `unknown`
    last, as it does for a question of any type (`test_every_type_parses_against_one_list` holds that matrix, a
    `positions` answer to a Recall question included, where Release 0.2 said no such type). Regression caught: the
    Recall/Perceive split returning on the paths that name no question, or a list that hides a type from the agent whose
    question asks for it."""
    assert parse_answer({"type": "blob"}) == Malformed(f"type must be one of {ALL_TYPES}, got 'blob'")
    assert isinstance(parse_answer({"type": "positions", "items": []}), Positions)
    with pytest.raises(ValueError) as refused:
        parse_gold({"type": "blob"})
    assert str(refused.value) == f"invalid gold value: type must be one of {ALL_TYPES}, got 'blob'"


@pytest.mark.parametrize(("answer", "reason"), [
    ("seven", "an answer must be a JSON object, got str"),
    ({"type": "set"}, "keys must be ['items', 'type']; missing ['items'], unexpected []"),
    ({"type": "set", "items": "x"}, "items must be a list of at most 100 entries"),
    ({"type": "set", "items": ["x", {"xy": [0, 0]}]}, "each item must be an object, got 'x'"),
    ({"type": "set", "items": [{"xy": [0, 0]}, "x"]},
     "keys must be ['change', 'xy']; missing ['change'], unexpected []"),
    ({"type": "set", "items": [{"xy": [0, 0], "change": "gone"}]},
     "change must be one of ['added', 'removed', 'moved'], got 'gone'"),
    ({"type": "set", "items": [{"xy": [0], "change": "added"}]}, "xy must be a list of 2 numbers, got [0]"),
    ({"type": "set", "items": [], "dont_care": []},
     "keys must be ['items', 'type']; missing [], unexpected ['dont_care']"),
])
def test_a_set_answer_is_refused_with_release_0_2s_reasons_in_release_0_2s_order(answer, reason):
    """Contract (review of the Perceive branch, finding 4): the reasons for a `set` answer that breaks the schema are
    Release 0.2's, copied from that release's `answers.py`, including which of two defects is reported (the items are
    checked one at a time, so a missing `change` in the first item is named before a non-object second). Regression
    caught: the shared list reader checking every item's type before any item's keys, which changed the message the
    reference agent sends back for a `set` answer with two defects."""
    assert parse_answer(answer, "set") == Malformed(reason)
