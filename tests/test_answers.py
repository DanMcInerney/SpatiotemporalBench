"""The answer schema (V0 §8): strict parsing, typed values, the question's answer_type, and gold shapes."""

from __future__ import annotations

from datetime import datetime

import pytest

from stbench import grid
from stbench.answers import (
    PROMPT_SHAPES,
    TYPES,
    Cells,
    Count,
    DontCare,
    Malformed,
    Number,
    Position,
    Route,
    RouteGold,
    SetAnswer,
    SetItem,
    Time,
    Unknown,
    YesNo,
    answer_type_of,
    parse_answer,
    parse_gold,
    to_json,
)

VALID = [
    ({"type": "position", "xy": [1.5, -2]}, Position((1.5, -2.0))),
    ({"type": "position", "xyz": [1, 2, 0.5]}, Position((1.0, 2.0, 0.5))),
    ({"type": "time", "clock": "2026-05-14T08:37:00"}, Time(datetime(2026, 5, 14, 8, 37, 0))),
    ({"type": "yesno", "value": False}, YesNo(False)),
    ({"type": "count", "value": 3}, Count(3)),
    ({"type": "number", "value": 2.5}, Number(2.5)),
    ({"type": "number", "value": 4}, Number(4.0)),
    ({"type": "set", "items": []}, SetAnswer(())),
    ({"type": "set", "items": [{"xy": [1, 2], "change": "moved"}]}, SetAnswer((SetItem((1.0, 2.0), "moved"),))),
    ({"type": "unknown"}, Unknown()),
]


@pytest.mark.parametrize(("obj", "expected"), VALID)
def test_every_answer_type_parses_to_its_typed_value(obj, expected):
    """Contract: each schema type of V0 §8 parses to exactly its typed value (ints widen to floats for metres).

    Regression caught: a type silently dropped from the parser, or coordinates kept as raw lists/ints so that equal
    answers compare unequal."""
    assert parse_answer(obj) == expected


@pytest.mark.parametrize(("obj", "_expected"), VALID)
def test_to_json_round_trips_every_answer_type(obj, _expected):
    """Contract: `parse_answer(to_json(a)) == a` and the JSON form is the canonical schema object.

    Regression caught: serialisation drifting from the parser (for example emitting `xy` for a 3-vector, or a clock the
    parser rejects), which would corrupt logged answers."""
    parsed = parse_answer(obj)
    assert parse_answer(to_json(parsed)) == parsed
    assert to_json(parsed)["type"] == obj["type"]


def test_to_json_of_position_keeps_the_key_that_matches_its_length():
    """Contract: a 3-vector serialises under `xyz`, a 2-vector under `xy`.

    Regression caught: always writing `xy`, which would lose z for metric-`xyz` questions."""
    assert to_json(Position((1.0, 2.0, 3.0))) == {"type": "position", "xyz": [1.0, 2.0, 3.0]}
    assert to_json(Position((1.0, 2.0))) == {"type": "position", "xy": [1.0, 2.0]}


BAD_SHAPES = [
    None, [], "unknown", 3, {}, {"type": "location", "xy": [0, 0]}, {"type": None},
    {"type": "unknown", "value": 1},
    {"type": "position", "xy": [0, 0], "extra": 1},
    {"type": "position", "xy": [0, 0], "xyz": [0, 0, 0]},
    {"type": "position"},
    {"type": "position", "xy": [0]},
    {"type": "position", "xy": [0, 0, 0]},
    {"type": "position", "xyz": [0, 0]},
    {"type": "position", "xy": ["1", 2]},
    {"type": "position", "xy": [True, 2]},
    {"type": "position", "xy": [float("nan"), 0]},
    {"type": "position", "xy": [float("inf"), 0]},
    {"type": "yesno", "value": "yes"},
    {"type": "yesno", "value": 1},
    {"type": "yesno"},
    {"type": "count", "value": 3.0},
    {"type": "count", "value": -1},
    {"type": "count", "value": True},
    {"type": "count", "value": "3"},
    {"type": "number", "value": "2.5"},
    {"type": "number", "value": True},
    {"type": "number", "value": float("nan")},
    {"type": "set", "items": "none"},
    {"type": "set", "items": [{"xy": [0, 0]}]},
    {"type": "set", "items": [{"xy": [0, 0], "change": "appeared"}]},
    {"type": "set", "items": [{"xy": [0, 0, 0], "change": "added"}]},
    {"type": "set", "items": [{"xy": [0, 0], "change": "added", "note": "x"}]},
    {"type": "set", "items": ["added"]},
    {"type": "set", "items": [{"xy": [0, 0], "change": "added"}] * 101},
    {"type": "set", "items": [], "dont_care": []},
]


@pytest.mark.parametrize("obj", BAD_SHAPES, ids=range(len(BAD_SHAPES)))
def test_schema_violations_return_malformed_and_never_raise(obj):
    """Contract: any object violating the schema (wrong JSON type, missing or unexpected key, bad length, non-finite
    number, bool posing as a number, oversize set, gold-only field) parses to Malformed with a reason.

    Regression caught: a lenient parser crediting `"yes"`, `3.0` or `[True, 2]`, or raising on garbage and crashing a
    scoring run instead of scoring 0."""
    parsed = parse_answer(obj)
    assert isinstance(parsed, Malformed)
    assert parsed.reason


@pytest.mark.parametrize(
    "clock",
    ["08:37", "2026-05-14", "2026-05-14T08:37:00Z", "2026-05-14T08:37:00+02:00", "2026-05-14T24:00:00",
     "2026-13-14T08:37:00", "2026-05-14T08:61:00", "20260514T083700", "2026-05-14T08:37:00.1234567", 837, None],
)
def test_clock_must_be_a_local_iso_datetime_without_a_zone(clock):
    """Contract: a time answer is a full local ISO date and time; a bare time of day, a zone, an impossible field or
    a non-string is Malformed.

    Regression caught: accepting `08:37` (ambiguous across midnight or days) or a zoned time, so the error in seconds
    against the gold clock would be computed on mismatched bases."""
    assert isinstance(parse_answer({"type": "time", "clock": clock}), Malformed)


def test_clock_accepts_minutes_only_fractions_and_a_space_separator():
    """Contract: `HH:MM`, fractional seconds and a space separator are valid local ISO and normalise to one datetime.

    Regression caught: over-strict rejection of valid ISO forms, which would score a correct time as 0."""
    assert parse_answer({"type": "time", "clock": "2026-05-14 08:37"}) == Time(datetime(2026, 5, 14, 8, 37))
    got = parse_answer({"type": "time", "clock": "2026-05-14T08:37:00.5"})
    assert got == Time(datetime(2026, 5, 14, 8, 37, 0, 500000))
    assert to_json(parse_answer({"type": "time", "clock": "2026-05-14 08:37"}))["clock"] == "2026-05-14T08:37:00"


def test_answer_must_match_the_questions_answer_type_unless_unknown():
    """Contract: with the question's `answer_type`, another type is Malformed, the same type and `unknown` pass.

    Regression caught: letting a count answer a `number` question, or an agent dodge a yes/no with a position, and
    `unknown` being rejected where abstaining is always allowed."""
    position = {"type": "position", "xy": [0, 0]}
    assert parse_answer(position, "position") == Position((0.0, 0.0))
    assert parse_answer({"type": "unknown"}, "position") == Unknown()
    assert isinstance(parse_answer(position, "yesno"), Malformed)
    assert isinstance(parse_answer({"type": "count", "value": 2}, "number"), Malformed)


def test_unrecognised_question_answer_type_is_a_caller_error():
    """Contract: an `answer_type` outside the schema vocabulary raises ValueError (a question-file bug, not a bad
    answer).

    Regression caught: a typo'd answer_type in a question silently marking every answer Malformed."""
    with pytest.raises(ValueError, match="answer_type"):
        parse_answer({"type": "unknown"}, "clock")


def test_dont_care_regions_exist_only_in_gold():
    """Contract: `dont_care` regions parse in gold and are Malformed in an answer; bad regions or gold raise.

    Regression caught: an agent supplying its own don't-care region to dodge false positives, or a gold with a zero
    radius or a non-object region being scored silently."""
    gold = {"type": "set", "items": [], "dont_care": [{"xy": [1, 2], "radius": 0.5}]}
    assert parse_gold(gold) == SetAnswer((), (DontCare((1.0, 2.0), 0.5),))
    assert isinstance(parse_answer(gold), Malformed)
    for bad in ({"xy": [1, 2], "radius": 0}, {"xy": [1, 2]}, "near"):
        with pytest.raises(ValueError, match="invalid gold"):
            parse_gold({"type": "set", "items": [], "dont_care": [bad]})
    with pytest.raises(ValueError, match="invalid gold"):
        parse_gold({"type": "yesno", "value": "yes"})


def test_to_json_carries_dont_care_and_refuses_malformed():
    """Contract: gold sets serialise their regions; a Malformed marker has no JSON form.

    Regression caught: dropping don't-care regions when gold is re-written, or serialising garbage as if valid."""
    gold = SetAnswer((SetItem((0.0, 1.0), "added"),), (DontCare((2.0, 3.0), 0.5),))
    assert to_json(gold) == {"type": "set", "items": [{"xy": [0.0, 1.0], "change": "added"}],
                             "dont_care": [{"xy": [2.0, 3.0], "radius": 0.5}]}
    with pytest.raises(ValueError, match="serialise"):
        to_json(Malformed("bad"))


def test_answer_type_of_names_every_variant():
    """Contract: each parsed variant, and Malformed, maps to its schema type name.

    Regression caught: a new variant added without a name, breaking the scorer's type-match check."""
    names = [answer_type_of(parse_answer(obj)) for obj, _ in VALID]
    assert names == ["position", "position", "time", "yesno", "count", "number", "number", "set", "set", "unknown"]
    assert answer_type_of(Malformed("x")) == "malformed"


def test_cells_parse_strictly():
    """Contract (plan §2.6): a `cells` answer is integer pairs only, at most 1,000 as given, with duplicates collapsing
    to a set; a float, a bool, a triple, a flat list, a gold-only `dont_care` or 1,001 items (duplicates included) is
    Malformed. Regression caught: a cells answer read two ways (`[1.0, 2]` kept as a float cell that matches no gold,
    `[True, 2]` as cell (1, 2)), or a cap applied after duplicates collapse, which a thousand copies would dodge."""
    assert parse_answer({"type": "cells", "items": [[1, 2], [1, 2], [-3, 0]]}) == Cells(frozenset({(1, 2), (-3, 0)}))
    assert parse_answer({"type": "cells", "items": []}) == Cells(frozenset())
    assert not isinstance(parse_answer({"type": "cells", "items": [[i, 0] for i in range(1000)]}, "cells"), Malformed)
    assert not isinstance(parse_answer({"type": "cells", "items": [[0, 0]] * 1000}), Malformed)
    for bad in ([[1.0, 2]], [[1, 2, 3]], [[True, 2]], [[1]], [1, 2], [[1, "2"]], "none",
                [[i, 0] for i in range(1001)], [[0, 0]] * 1001):
        assert isinstance(parse_answer({"type": "cells", "items": bad}), Malformed), bad
    assert isinstance(parse_answer({"type": "cells"}), Malformed)
    assert isinstance(parse_answer({"type": "cells", "items": [], "dont_care": []}), Malformed)


def test_cells_gold_rejects_overlap():
    """Contract (plan §2.6): a gold `cells` may carry `dont_care` cells, read the same way as `items`, and a cell in
    both lists is an invalid gold. Regression caught: a gold that both counts and ignores a cell, which the scorer's
    set difference would silently resolve in one direction."""
    ok = parse_gold({"type": "cells", "items": [[2, 2], [3, 2]], "dont_care": [[4, 2]]})
    assert ok == Cells(frozenset({(2, 2), (3, 2)}), frozenset({(4, 2)}))
    with pytest.raises(ValueError, match="invalid gold"):
        parse_gold({"type": "cells", "items": [[2, 2]], "dont_care": [[2, 2], [5, 5]]})
    with pytest.raises(ValueError, match="invalid gold"):
        parse_gold({"type": "cells", "items": [[2, 2]], "dont_care": [[1.5, 2]]})


def test_route_parse_strictly():
    """Contract (plan §2.6): a `route` answer is 1 to 50 points of two finite numbers; a gold is `regions` (a non-empty
    object of polygons of three or more points) and `valid` (non-empty lists of region names), every name a region, and
    neither shape may carry the other's key. Regression caught: a route gold no answer can match (a valid sequence
    naming a region that is not there), an empty route scored as an answer, or `points` accepted on a gold."""
    one, fifty = [[1, 2]], [[i, 0.5] for i in range(50)]
    assert parse_answer({"type": "route", "points": one}) == Route(((1.0, 2.0),))
    assert len(parse_answer({"type": "route", "points": fifty}, "route").points) == 50
    for bad in ([], [[i, 0] for i in range(51)], [[1, "a"]], [[1, 2, 3]], [[1]], [[float("inf"), 0]], [[True, 0]], 5):
        assert isinstance(parse_answer({"type": "route", "points": bad}), Malformed), bad
    assert isinstance(parse_answer({"type": "route", "points": one, "regions": {}}), Malformed)
    hall, attic = [[0, 0], [4, 0], [4, 3]], [[5, 0], [9, 0], [9, 3]]
    gold = {"type": "route", "regions": {"hall": hall, "attic": attic}, "valid": [["hall", "attic"], ["hall"]]}
    parsed = parse_gold(gold)
    assert isinstance(parsed, RouteGold) and [name for name, _ in parsed.regions] == ["hall", "attic"]
    assert parsed.valid == (("hall", "attic"), ("hall",))
    assert isinstance(parse_answer(gold), Malformed)  # `regions` and `valid` on an answer
    bad_golds = [
        {"type": "route", "regions": {"hall": hall}, "valid": [["hall", "attic"]]},   # no `attic` region
        {"type": "route", "regions": {"hall": hall}},
        {"type": "route", "valid": [["hall"]]},
        {"type": "route", "points": one, "regions": {"hall": hall}, "valid": [["hall"]]},
        {"type": "route", "regions": {}, "valid": [["hall"]]},
        {"type": "route", "regions": {"hall": hall[:2]}, "valid": [["hall"]]},
        {"type": "route", "regions": {"hall": hall}, "valid": []},
        {"type": "route", "regions": {"hall": hall}, "valid": [[]]},
        {"type": "route", "regions": {"hall": hall}, "valid": [[["hall"]]]},
    ]
    for bad in bad_golds:
        with pytest.raises(ValueError, match="invalid gold"):
            parse_gold(bad)


def test_new_shapes_round_trip():
    """Contract: `parse_answer(to_json(a)) == a` for a cells and a route answer and `parse_gold(to_json(g)) == g` for a
    cells gold with `dont_care` and a route gold; `to_json` writes cells sorted, `dont_care` only when there is one, and
    a route gold's regions in their order. Regression caught: a shape that changes when written and read back, or a
    cells list in set-iteration order, which would make a gold file differ run to run."""
    cells = Cells(frozenset({(2, 1), (-1, 3), (2, 0)}))
    route = Route(((1.0, 2.0), (0.5, -1.5)))
    gold = Cells(frozenset({(0, 0)}), frozenset({(1, 0), (-2, 5)}))
    hall, attic = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0)), ((5.0, 0.0), (9.0, 0.0), (9.0, 3.0))
    routes = RouteGold((("hall", hall), ("attic", attic)), (("hall", "attic"), ("hall",)))
    assert to_json(cells) == {"type": "cells", "items": [[-1, 3], [2, 0], [2, 1]]}
    assert parse_answer(to_json(cells)) == cells and parse_answer(to_json(route)) == route
    assert to_json(gold) == {"type": "cells", "items": [[0, 0]], "dont_care": [[-2, 5], [1, 0]]}
    assert parse_gold(to_json(gold)) == gold and parse_gold(to_json(routes)) == routes
    assert list(to_json(routes)["regions"]) == ["hall", "attic"]


def test_a_route_gold_and_a_route_answer_share_a_type_name():
    """Contract: a route answer, a route gold and a cells answer are named `route`, `route` and `cells`, so the scorer's
    check that an answer is the gold's type holds for the one type with two shapes. Regression caught: the gold named
    differently, which would score every correct route 0 as a type mismatch."""
    assert answer_type_of(Route(((0.0, 0.0),))) == answer_type_of(RouteGold((), ())) == "route"
    assert answer_type_of(Cells(frozenset())) == "cells"


ONE_OF_EACH = {
    "position": {"type": "position", "xy": [1.5, 2.0]},
    "offset": {"type": "offset", "ahead_m": 3.5, "left_m": -1.0},
    "time": {"type": "time", "clock": "2026-05-14T08:37:00"},
    "yesno": {"type": "yesno", "value": True},
    "count": {"type": "count", "value": 3},
    "number": {"type": "number", "value": 2.5},
    "set": {"type": "set", "items": [{"xy": [1.0, 2.0], "change": "moved"}]},
    "positions": {"type": "positions", "items": [{"xy": [1.0, 2.0]}]},
    "cells": {"type": "cells", "items": [[1, 2]]},
    "route": {"type": "route", "points": [[1.0, 2.0]]},
    "heading": {"type": "heading", "deg_left": -35.0},
}
ALL_TYPES = ("['position', 'offset', 'time', 'yesno', 'count', 'number', 'set', 'positions', 'cells', 'route', "
             "'heading', 'unknown']")  # by hand


@pytest.mark.parametrize("question_type", TYPES)
def test_every_type_parses_against_one_list(question_type):
    """Contract (plan §2.6): a question of any type in `TYPES` accepts an answer of its own type and `unknown`, refuses
    each of the other ten as a type mismatch (not as a type that does not exist), and refuses a type outside the list
    with all twelve named. Regression caught: the retired Recall/Perceive split returning, so that a question of one set
    refuses another set's type as nonexistent."""
    assert tuple(ONE_OF_EACH) == TYPES  # one hand-written answer per type, so the loop below covers them all
    assert not isinstance(parse_answer(ONE_OF_EACH[question_type], question_type), Malformed)
    assert parse_answer({"type": "unknown"}, question_type) == Unknown()
    for kind, answer in ONE_OF_EACH.items():
        if kind != question_type:
            assert parse_answer(answer, question_type) == Malformed(
                f"answer type {kind!r} does not match the question's answer_type {question_type!r}")
    assert parse_answer({"type": "blob"}, question_type) == Malformed(f"type must be one of {ALL_TYPES}, got 'blob'")


def test_every_type_has_a_prompt_shape():
    """Contract: `PROMPT_SHAPES` holds a shape for every type in `TYPES` and for `unknown`, and each shape shows its own
    `"type"`. Regression caught: a type the reference agent's prompt cannot show, so it could be asked a question it has
    never been told how to answer."""
    assert set(PROMPT_SHAPES) == set(TYPES) | {"unknown"}
    for kind, shape in PROMPT_SHAPES.items():
        assert f'"type": "{kind}"' in shape


def test_the_prompt_tells_the_agent_the_grids_cell_size_and_the_route_cap():
    """Contract: the `cells` shape names the cell size `grid.CELL_M` (floor(x / 0.5), floor(y / 0.5)) and the `route`
    shape the 50-point cap that `parse_answer` enforces. Regression caught: a prompt that teaches one cell size while
    the gold's grid uses another, so every cells answer lands one cell off."""
    assert f"i = floor(x / {grid.CELL_M})" in PROMPT_SHAPES["cells"]
    assert f"j = floor(y / {grid.CELL_M})" in PROMPT_SHAPES["cells"]
    assert "1 to 50 map positions" in PROMPT_SHAPES["route"]
