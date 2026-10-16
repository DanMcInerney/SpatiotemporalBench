"""The credits `cells`, `route`, `region` and `relative` (plan §2.6), the dispatch on a gold row's `credit`, and the fit
of a credit to its gold's type. Expected values are worked out by hand in each test."""

from __future__ import annotations

from datetime import datetime

import pytest

from score_kit import SQUARE, UNKNOWN, cells, clock, count, items, num, off, places, pos, rec
from stbench import records
from stbench.answers import Cells, Route, parse_gold
from stbench.score import credit, outcome, penalised
from stbench.score_set import cells_f1

# a hall (x 0-2), a kitchen overlapping it (x 1-4), a doorway gap (x 4-5) and a bedroom (x 5-7), each 2 m deep
ROUTE_GOLD = {"type": "route",
              "regions": {"hall": [[0, 0], [2, 0], [2, 2], [0, 2]], "kitchen": [[1, 0], [4, 0], [4, 2], [1, 2]],
                          "bedroom": [[5, 0], [7, 0], [7, 2], [5, 2]]},
              "valid": [["hall", "kitchen", "bedroom"], ["hall", "bedroom"]]}


def route(*points):
    return {"type": "route", "points": [list(p) for p in points]}


def test_cells_credit_by_hand():
    """Contract (plan §2.6): `cells` credit is F1 of the two sets, 1 when both are empty and 0 when exactly one is, the
    gold's don't-care cells dropped from both sides, duplicates collapsed, negative cells compared as they are, and an
    answer of 1,001 items malformed. Worked: gold {(0,0), (1,0), (0,1)}, answer {(0,0), (1,0), (5,5)} meets two of
    three on each side: 2*2/(3+3) = 2/3. Answer {(0,0), (1,0), (2,0)} against gold {(0,0), (1,0)} with (2,0) don't-care
    drops (2,0): 1. A duplicate cell in the answer is one cell: {(0,0), (0,0)} against {(0,0)} is 1, not 2/3.

    Regression caught: partial room cells counted as hits or misses, the downtown's negative cells mirrored or
    clipped, or a repeated cell inflating precision."""
    gold = rec(cells((0, 0), (1, 0), (0, 1)), credit="f1")
    assert credit(cells((0, 0), (1, 0), (0, 1)), gold) == 1.0
    assert credit(cells((0, 0), (1, 0), (5, 5)), gold) == pytest.approx(2 / 3)
    assert credit(cells(), rec(cells(), credit="f1")) == 1.0
    assert credit(cells((0, 0)), rec(cells(), credit="f1")) == 0.0
    assert credit(cells(), gold) == 0.0
    dc = rec(cells((0, 0), (1, 0), dont_care=[(2, 0)]), credit="f1")
    assert credit(cells((0, 0), (1, 0), (2, 0)), dc) == 1.0
    assert credit(cells((2, 0)), rec(cells((0, 0), dont_care=[(2, 0)]), credit="f1")) == 0.0  # A empty, G is not
    assert credit(cells((2, 0)), rec(cells(dont_care=[(2, 0)]), credit="f1")) == 1.0  # both empty once (2,0) is dropped
    assert credit({"type": "cells", "items": [[0, 0], [0, 0]]}, rec(cells((0, 0)), credit="f1")) == 1.0
    negative = rec(cells((-1, -1), (-2, 3)), credit="f1")
    assert credit(cells((-1, -1), (-2, 3)), negative) == 1.0
    assert credit(cells((1, 1), (2, 3)), negative) == 0.0
    thousand = {"type": "cells", "items": [[i, 0] for i in range(1000)]}
    over = {"type": "cells", "items": [[i, 0] for i in range(1001)]}
    assert credit(thousand, rec(thousand, credit="f1")) == 1.0
    assert credit(over, rec(thousand, credit="f1")) == 0.0 and outcome(over, rec(thousand, credit="f1")) == "wrong"


def test_cells_f1_drops_dont_care_from_the_gold_side_too():
    """Contract (map D-10): the don't-care cells are dropped from the gold as well as from the answer, a new rule beside
    the list credits', which drop answer items only. Worked on a gold object that lists a cell in both (which
    `parse_gold` refuses, so only the function sees one): gold {(0,0), (1,0)} with (1,0) don't-care leaves {(0,0)}, and
    the answer {(0,0)} scores 1, where dropping the answer side alone would score 2/3.

    Regression caught: a don't-care cell still counted as a gold cell the answer missed."""
    gold = Cells(frozenset({(0, 0), (1, 0)}), frozenset({(1, 0)}))
    assert cells_f1(Cells(frozenset({(0, 0)})), gold) == 1.0
    assert cells_f1(Cells(frozenset({(0, 0), (1, 0)})), gold) == 1.0


def test_a_partial_cells_answer_is_a_partial_outcome_scored_like_any_credit():
    """Contract: credit 2/3 is `partial`, its penalised score at t = 0.5 is 2/3 - 1/3 = 1/3, and `unknown` on a cells
    gold is an abstention (0). Regression caught: the new type bypassing the outcome classes or the penalty."""
    gold, answer = rec(cells((0, 0), (1, 0), (0, 1)), credit="f1"), cells((0, 0), (1, 0), (5, 5))
    assert outcome(answer, gold) == "partial" and penalised(answer, gold) == pytest.approx(1 / 3)
    assert outcome(UNKNOWN, gold) == "abstained" and penalised(UNKNOWN, gold) == 0.0


def test_route_credit_by_hand():
    """Contract (plan §2.6): a route is judged by the rooms it passes, not by its points. Each point goes to the first
    region (in the gold's order) that holds it, boundary included; a point in no region (the doorway gap) is dropped;
    consecutive repeats collapse; credit is 1 when the sequence is one of `valid`, else 0, and an empty sequence is 0.
    Worked on ROUTE_GOLD (the kitchen overlaps the hall for x 1-2): [(0.5,1), (3,1), (6,1)] is hall, kitchen, bedroom:
    1. [(1.5,1), (3,1), (6,1)] starts in the overlap, which is the hall (the first listed): the same sequence, 1; had
    the overlap gone to the kitchen it would read kitchen, kitchen, bedroom and score 0. (2,1) is on the hall's edge and
    in the kitchen: the hall. (4.5,1) is in the gap and dropped: [(0.5,1), (4.5,1), (6,1)] is hall, bedroom, the
    second valid route.

    Regression caught: a route judged by how close its points run to a path instead of by its rooms, a doorway point
    breaking a route, overlapping rooms resolved to the wrong one, or an unordered visit accepted."""
    gold = rec(ROUTE_GOLD, credit="route")
    assert credit(route((0.5, 1), (3, 1), (6, 1)), gold) == 1.0
    assert credit(route((0.5, 1), (0.6, 1), (3, 1), (3.5, 1), (6, 1)), gold) == 1.0  # repeats collapse
    assert credit(route((0.1, 0.1), (3.9, 1.9), (5.1, 0.1)), gold) == 1.0  # corners of the rooms, far from any path
    assert credit(route((1.5, 1), (3, 1), (6, 1)), gold) == 1.0
    assert credit(route((2, 1), (6, 1)), gold) == 1.0
    assert credit(route((0.5, 1), (4.5, 1), (6, 1)), gold) == 1.0
    assert credit(route((5, 1)), gold) == 0.0  # the bedroom alone is no valid route
    assert credit(route((6, 1), (0.5, 1)), gold) == 0.0  # the right rooms in the wrong order
    assert credit(route((4.5, 1)), gold) == 0.0  # no room at all: empty
    assert credit(route((0.5, 1), (3, 1), (0.5, 1), (6, 1)), gold) == 0.0  # hall, kitchen, hall, bedroom
    assert credit(parse_gold(ROUTE_GOLD), gold) == 0.0  # a gold handed in as an answer has no points


def test_region_credit_by_hand():
    """Contract (plan §2.6): a region position scores 1 when the answer's xy is inside the polygon, boundary included,
    and at least `erode_m` from its boundary (within 1e-9), else 0; an xyz answer is judged by its xy. Worked on a 4 m
    square with erosion 0.2: the centre 1; 0.1 m inside 0; exactly 0.2 m inside 1; 0.1 m outside 0. With erosion 0 a
    point on an edge or a corner is 1. On a triangle with a slanted edge, (2, 2) is on the edge and so is
    (2.0000000000000004, 2), which float noise puts a hair outside: 1. In an L-shaped room, (1, 3) is inside and (3, 3),
    in the notch, is outside.

    Regression caught: a boundary rule that flips with float noise, a polygon test wrong for a concave room, or an
    eroded polygon whose rounding moves a point across it."""
    room = rec(pos(2, 2), credit="region", region=SQUARE, tolerances={"erode_m": 0.2})
    assert credit(pos(2, 2), room) == 1.0
    assert credit(pos(0.1, 2), room) == 0.0
    assert credit(pos(0.2, 2), room) == 1.0
    assert credit(pos(-0.1, 2), room) == 0.0
    assert credit(pos(2, 2, 9.0), room) == 1.0
    zone = rec(pos(2, 2), credit="region", region=SQUARE, tolerances={"erode_m": 0})
    assert credit(pos(0, 2), zone) == 1.0 and credit(pos(4, 4), zone) == 1.0 and credit(pos(4.0000001, 2), zone) == 0.0
    triangle = rec(pos(1, 1), credit="region", region=[[0, 0], [4, 0], [0, 4]], tolerances={"erode_m": 0})
    assert credit(pos(2, 2), triangle) == 1.0 and credit(pos(2.0000000000000004, 2), triangle) == 1.0
    assert credit(pos(2.1, 2), triangle) == 0.0
    ell = rec(pos(1, 1), credit="region", region=[[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]],
              tolerances={"erode_m": 0})
    assert credit(pos(1, 3), ell) == 1.0 and credit(pos(3, 3), ell) == 0.0


def test_a_region_credit_needs_its_polygon_and_its_erosion():
    """Contract: a `region` gold with no polygon, or with no `tolerances.erode_m`, is a loud error, not a default.
    Regression caught: a region question silently scored with no margin (or as a plain position) when its row was
    written without them."""
    for bad in (rec(pos(2, 2), credit="region", tolerances={"erode_m": 0.2}),
                rec(pos(2, 2), credit="region", region=SQUARE),
                rec(pos(2, 2), credit="region", region=[[0, 0], [1, 1]], tolerances={"erode_m": 0.2})):
        with pytest.raises(ValueError, match="region"):
            credit(pos(2, 2), bad)


def test_relative_credit_by_hand():
    """Contract (map D-9): a number scored `relative` earns the mean over p = 50, 55, ..., 95 of [|e - y| * 100 <
    (100 - p) * y], strict. Worked: y = 2, e = 2.5 is 25% off, which meets p = 50..70 (50 < 100, 90, 80, 70, 60) but
    not 75 (50 < 50): 5 of 10, 0.5. An estimate exactly 50% off (3 or 1 for y = 2) meets nothing (100 < 100 fails at
    p = 50): 0. y = 10, e = 10.5 is 5% off, meets p up to 90 (50 < (100 - p) * 10 fails at 95): 0.9. An exact
    estimate is 1. A negative estimate is more than y away, so 0. A gold y of 0 or less is refused.

    Regression caught: MRA's strictness lost in floats (an error exactly on a bound crediting), or a negative estimate
    crediting, or a relative error against a zero gold dividing by zero."""
    gold = rec(num(2.0), credit="relative")
    assert credit(num(2.0), gold) == 1.0
    assert credit(num(2.5), gold) == pytest.approx(0.5)
    assert credit(num(1.5), gold) == pytest.approx(0.5)
    assert credit(num(3.0), gold) == 0.0 and credit(num(1.0), gold) == 0.0
    assert credit(num(-1.0), gold) == 0.0 and credit(num(0.0), gold) == 0.0
    assert credit(num(10.5), rec(num(10.0), credit="relative")) == pytest.approx(0.9)
    for y in (0.0, -2.0):
        with pytest.raises(ValueError, match="positive"):
            credit(num(1.0), rec(num(y), credit="relative"))


@pytest.mark.parametrize(
    ("record", "reason"),
    [(rec(num(1.0), credit="region", region=SQUARE, tolerances={"erode_m": 0.2}), "scores"),
     (rec(count(3), credit="ladder", tolerances={"ladder": [1.0]}), "scores"),
     (rec(pos(0, 0), credit="relative"), "scores"), (rec(pos(0, 0), credit="route"), "scores"),
     (rec(num(1.0), credit="mra"), "scores"), (rec(cells((0, 0)), credit="exact"), "scores"),
     (rec(num(1.0), credit="bogus"), "credit must be one of")],
)
def test_a_credit_must_fit_its_gold(record, reason):
    """Contract (`records.CREDIT_FITS`): a gold row whose credit does not score its gold's type raises, whatever the
    answer. Regression caught: a gold scored by the wrong rule (a region credit on a number, a relative credit on a
    position) and so worth something else than its row says."""
    with pytest.raises(ValueError, match=reason):
        credit(num(1.0), record)


HEADING = {"type": "heading", "deg_left": 0}   # the best of the profile below (entry 0, 2.0 m)

# each (credit, gold type) of `records.CREDIT_FITS`: a gold, its perfect answer and the tolerance the row holds
PERFECT = {
    ("ladder", "position"): (pos(1, 1), pos(1, 1), {"ladder": [0.25, 0.5, 1.0, 2.0]}),
    ("ladder", "offset"): (off(2.0, -1.0), off(2.0, -1.0), {"ladder": [0.1, 0.25, 0.5]}),
    ("ladder", "number"): (num(3.0), num(3.0), {"ladder": [0.25, 0.5, 1.0, 2.0]}),
    ("ladder", "time"): (clock("2026-05-14T08:37:00"), clock("2026-05-14T08:37:00"), {"ladder": [3.0, 10.0, 30.0]}),
    ("region", "position"): (pos(2, 2), pos(2, 2), {"erode_m": 0.2}),
    ("relative", "number"): (num(4.0), num(4.0), {}),
    ("mra", "count"): (count(5), count(5), {}),
    ("exact", "yesno"): ({"type": "yesno", "value": False}, {"type": "yesno", "value": False}, {}),
    ("f1", "set"): (items((0, 0, "added")), items((0, 0, "added")), {"match_m": 0.5}),
    ("f1", "positions"): (places((0, 0)), places((0, 0)), {"match_m": 0.5}),
    ("f1", "cells"): (cells((0, 0)), cells((0, 0)), {}),
    ("route", "route"): (ROUTE_GOLD, route((0.5, 1), (6, 1)), {}),
    ("graded", "offset"): (off(2.0, -1.0), off(2.0, -1.0), {"resolution_m": 0.02, "need_m": 0.1, "harm": "radial"}),
    ("graded", "number"): (num(3.0), num(3.0), {"resolution_m": 0.02, "need_m": 0.15, "harm": "over",
                                                "threshold_m": 0.49}),
    ("graded", "position"): (pos(1, 1), pos(1, 1), {"resolution_m": 0.02, "need_m": 1.0, "harm": None}),
    ("graded_set", "positions"): (places((0, 0), (3, 0)), places((0, 0), (3, 0)),
                                  {"match_m": 0.5, "resolutions_m": [0.02, 0.02], "fp_weight": 0.5}),
    ("regret", "heading"): (HEADING, HEADING, {"profile_m": [2.0, 1.0, 0.5, 1.0], "step_deg": 90,
                                               "resolution_m": 0.02, "need_m": 0.1, "threshold_m": 0.44}),
    ("route", "heading"): (HEADING, HEADING, {"profile_m": [0.5, -0.2, None, 0.3], "step_deg": 90, "step_m": 1.0,
                                              "resolution_m": 0.1, "need_m": 0.15}),
}


def test_every_credit_a_gold_row_may_name_scores_its_perfect_answer_as_one():
    """Contract: the scorer implements every (credit, gold type) pair that `records.CREDIT_FITS` allows a gold row to
    hold, with the tolerance key the row holds for it: the perfect answer to each earns 1, and the table below names
    exactly the pairs the records allow. Regression caught: a credit or a fit added to the records that the scorer does
    not know, which would raise on the first gold row that uses it."""
    allowed = {(name, kind) for name, fits in records.CREDIT_FITS.items() for kind in fits}
    assert set(PERFECT) == allowed
    for (name, kind), (gold, answer, tolerances) in PERFECT.items():
        extra = {"region": SQUARE} if name == "region" else {}
        assert credit(answer, rec(gold, credit=name, tolerances=tolerances, **extra)) == 1.0, (name, kind)
        assert set(tolerances) == set(records.CREDIT_FITS[name][kind]), (name, kind)  # what the records say


def test_a_record_with_no_credit_key_is_scored_by_its_golds_type():
    """Contract: a gold record without `credit` is scored by its gold's type: position, number and
    time on the ladder, count by MRA, yes/no exactly, set, positions and cells by F1, a route by its rooms. Worked: a
    position 0.3 m off meets 0.5, 1 and 2 m (0.75); a number 0.6 off meets 1 and 2 (0.5); a time 5 s off meets 10, 30
    and 90 s (0.75); 9 for a count of 10 meets theta up to 0.85 (0.8); true for true is 1; a set item right and one
    extra is 2*1/(2+1) = 2/3; cells {(0,0), (1,0)} against {(0,0)} is 2/3; a route of the right rooms is 1.

    Regression caught: the dispatch on `credit` changing what a type is worth when its record names no credit, which
    would make a stored record score differently from how its type always scored."""
    start = datetime(2026, 5, 14, 8, 37, 0)
    cases = [
        (pos(0.3, 0), pos(0, 0), 0.75), (num(3.6), num(3.0), 0.5),
        (clock((start.replace(second=5)).isoformat()), clock(start.isoformat()), 0.75),
        (count(9), count(10), 0.8), ({"type": "yesno", "value": True}, {"type": "yesno", "value": True}, 1.0),
        (items((0, 0, "added"), (9, 9, "added")), items((0, 0, "added")), 2 / 3),
        (places((0, 0), (9, 9)), places((0, 0)), 2 / 3), (cells((0, 0), (1, 0)), cells((0, 0)), 2 / 3),
        (route((0.5, 1), (3, 1), (6, 1)), ROUTE_GOLD, 1.0),
    ]
    for answer, gold, expected in cases:
        assert credit(answer, rec(gold)) == pytest.approx(expected), gold["type"]


@pytest.mark.parametrize(
    "gold",
    [rec(pos(0, 0)), rec(cells((0, 0)), credit="f1"), rec(ROUTE_GOLD, credit="route"), rec(count(3)),
     rec(UNKNOWN, slice="unknown")],
    ids=["position", "cells", "route", "count", "unknown"],
)
def test_a_failed_ask_scores_as_malformed_under_every_type(gold):
    """Contract (recommendations §2.4 item 1): a question the harness marks `{"type": "failed", ...}` (an ask that timed
    out) parses as malformed under every answer type, so it earns credit 0 and is a confident error: outcome `wrong`,
    penalised -1 at t = 0.5, on an answerable gold and on an unknown one alike (where only `unknown` is right).
    Regression caught: a failed question treated as `unknown` (an abstention, free) or crashing the scorer."""
    failed = {"type": "failed", "reason": "timeout"}
    assert credit(failed, gold) == 0.0 and outcome(failed, gold) == "wrong" and penalised(failed, gold) == -1.0


def test_parsed_cells_and_route_answers_are_scored_without_reparsing():
    """Contract: the scorer takes an already-parsed `Cells` or `Route` as well as a raw object. Regression caught:
    `_coerce` listing the old classes by hand, so a new class is re-parsed as a raw object and flagged malformed."""
    assert credit(Cells(frozenset({(0, 0)})), rec(cells((0, 0)), credit="f1")) == 1.0
    assert credit(Route(((0.5, 1.0), (6.0, 1.0))), rec(ROUTE_GOLD, credit="route")) == 1.0
