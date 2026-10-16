"""The `graded_set` credit (brief §3.4): a list scored per item. The answer's items are matched one to
one to the gold's by maximum summed credit, each pair within four match radii earning its own four quarters at its own
resolution, and its direction quarter only within the match radius M (a list item is a place); a missed item costs a
whole item and a false one `fp_weight` of one. Expected values are worked by hand."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, places, rec
from stbench.score import credit
from stbench.score_set import assignment

M = 0.5   # the match radius of the rows below; the reach is 4 M = 2 m


def set_row(gold_points, resolutions=None, *, dont_care=(), weight=0.5):
    """A `task_things` gold row: the places, a resolution of 0.02 m each unless given, a match radius of 0.5 m."""
    gold = places(*gold_points)
    if dont_care:
        gold["dont_care"] = [{"xy": [x, y], "radius": r} for x, y, r in dont_care]
    return rec(gold, credit="graded_set", family="task_things",
               tolerances={"match_m": M, "resolutions_m": resolutions or [0.02] * len(gold_points),
                           "fp_weight": weight})


def test_a_perfect_list_scores_one_a_missed_item_costs_one_and_a_false_item_half():
    """Contract (§3.4): s = sum of the matched items' credit / (|gold| + fp_weight * |false|). By hand, gold (0, 0) and
    (3, 0), rho 0.02:
    - both answered exactly: (1 + 1) / 2 = 1;
    - only (0, 0): (1 + 0) / 2 = 0.5, the missed item costing a whole one;
    - both and (10, 10): 2 / (2 + 0.5) = 0.8, the false item costing half;
    - the same item twice: the repeat is false, 1 / (1 + 0.5) for a gold of one.
    (An empty gold has its own test below.) Regression caught: matching by count (an F1 that scores a near miss as a
    hit), or a false item that costs nothing."""
    gold = set_row([(0, 0), (3, 0)])
    assert credit(places((0, 0), (3, 0)), gold) == pytest.approx(1.0)
    assert credit(places((0, 0)), gold) == pytest.approx(0.5)
    assert credit(places((0, 0), (3, 0), (10, 10)), gold) == pytest.approx(0.8)
    assert credit(places((0, 0), (0, 0)), set_row([(0, 0)])) == pytest.approx(1 / 1.5)


def test_a_placed_item_earns_its_four_quarters_at_its_own_golds_resolution():
    """Contract (§3.4, §3.7): a matched item earns the four quarters of its error against the match radius M and that
    gold item's own resolution (the row lists one a gold item, in the gold's order). By hand, M = 0.5, gold (0, 0)
    with rho 0.30 and (3, 0) with rho 0.02, answers 0.1 m and 0.4 m off:
    - the first is inside rho (N' = max(0.5, 0.6) = 0.6): 1;
    - the second: t = a = 1, f = 1 - ln 20 / ln 25 = 0.0693: (3.0693) / 4 = 0.7673;
    - the list: (1 + 0.7673) / 2 = 0.8837. With the resolutions swapped the first answer earns (3 + 1 - ln 5 / ln 25)
      / 4 = 0.875 (f = 0.5) and the second, 0.4 m off a resolution of 0.30 (N' = 0.60), (3 + 1 - ln(4/3) / ln 2) / 4
      = 0.8962: the list is 0.8856.
    (The direction quarter beyond M is the next tests'.) Regression caught: one resolution for the whole list, the
    resolutions read in another order, or the direction quarter withheld from a match within M."""
    answer = places((0.1, 0), (3.4, 0))
    assert credit(answer, set_row([(0, 0), (3, 0)], [0.30, 0.02])) == pytest.approx(0.8837, abs=1e-4)
    assert credit(answer, set_row([(0, 0), (3, 0)], [0.02, 0.30])) == pytest.approx(0.8856, abs=1e-4)


def test_the_assignment_is_the_one_of_largest_summed_credit_not_the_greedy_nearest_first():
    """Contract (§3.4): the pairs maximise the summed credit (the Hungarian method). By hand, M = 0.5, rho 0.02, gold
    at 0 and 1.5 on a line, answers at 0.4 and -0.5. The pair credits: 0.4 to gold 0 (0.4 m off, direction): t = a = 1,
    f = 1 - ln 20 / ln 25 = 0.0693, so 0.7673; 0.4 to gold 1.5 (1.1 m off, past M): no direction, t = (2 - 1.1) / 1
    = 0.9, a = 0, so 0.225; -0.5 to gold 0 (0.5 m off, exactly M): (1 + 1 + 1 + 0) / 4 = 0.75; -0.5 to gold 1.5 (2.0 m
    off, the reach): t = 0, so 0. The nearest pair is (0.4, 0), which strands the answer at -0.5 (false) and leaves
    the gold at 1.5 missed, so s = 0.7673 / (2 + 0.5) = 0.3069; the optimum is (0.4, 1.5) and (-0.5, 0), worth
    0.225 + 0.75 = 0.975, so s = 0.975 / 2 = 0.4875. `assignment` itself pairs the larger side's best rows: three
    answers for two gold items, credits [[.2, .9], [.8, .1], [.5, .5]], pair (0, 1) and (1, 0) (1.7), and the
    transposed matrix the same pairs.
    Regression caught: a greedy or count-based matching, which scores this list 0.3069."""
    assert credit(places((0.4, 0), (-0.5, 0)), set_row([(0, 0), (1.5, 0)])) == pytest.approx(0.4875, abs=1e-4)
    wide = [[0.2, 0.9], [0.8, 0.1], [0.5, 0.5]]
    assert assignment(wide) == [(0, 1), (1, 0)]
    assert assignment([list(c) for c in zip(*wide, strict=True)]) == [(0, 1), (1, 0)]
    assert assignment([]) == [] and assignment([[]]) == []


def test_a_list_item_is_a_place_its_direction_quarter_is_earned_within_the_match_radius_alone():
    """Contract (0.11.0, methodology rule 7): a list item is a place, so a pair earns its direction quarter within the
    match radius M, as a single place does within R, and not at any distance up to the reach 4 M. By hand, gold (0, 0),
    rho 0.02, M = 0.5 (t is 1 to 2 M = 1 and 0 at 4 M = 2, a is 1 to M and 0 at 2 M, f is 1 - ln(e / rho) / ln(M / rho)
    to 0 at M), one answer e metres off, so s is the pair's credit over 1 (no false place):
    - 0.4 m: d = t = a = 1, f = 1 - ln 20 / ln 25 = 0.0693, s = 3.0693 / 4 = 0.7673;
    - 0.5 m, exactly M: d = t = a = 1, f = 0, s = 0.75;
    - 0.6 m, just past M: d = 0, t = 1, a = (1 - 0.6) / 0.5 = 0.8, f = 0, s = 1.8 / 4 = 0.45;
    - 1.0 m: d = 0, t = 1, a = 0, f = 0, s = 0.25;
    - 1.5 m: d = 0, t = (2 - 1.5) / 1 = 0.5, s = 0.125;
    - 2.5 m, past the reach: no pair, the answer false and the gold missed, s = 0 / (1 + 0.5) = 0.
    Regression caught: the direction quarter given to every pair within the reach (a constant True before 0.11.0),
    which scores 1.0 m as 0.5 and 1.5 m as 0.375, so a sensorless prior is paid for the layout."""
    row = set_row([(0, 0)])
    for e, s in ((0.4, 0.7673), (0.5, 0.75), (0.6, 0.45), (1.0, 0.25), (1.5, 0.125), (2.5, 0.0)):
        assert credit(places((e, 0)), row) == pytest.approx(s, abs=1e-4), f"{e} m off"


def test_a_one_item_list_scores_as_a_single_place_with_the_same_radius_and_resolution():
    """Contract (0.11.0): one rule for every place answer. A list of one item against a gold of one, read through
    `score.credit`, equals a single `position` on a row whose `need_m` is the list's M and whose resolution is the
    same. By hand, M = 0.5, rho 0.02, e metres off:
    - 0.3 m: d = t = a = 1, f = 1 - ln 15 / ln 25 = 0.1587, 3.1587 / 4 = 0.7897;
    - 0.5 m: 0.75;
    - 0.8 m: d = 0, t = 1, a = (1 - 0.8) / 0.5 = 0.4, 1.4 / 4 = 0.35;
    - 1.2 m: d = 0, t = (2 - 1.2) / 1 = 0.8, a = 0, 0.2;
    - 1.9 m: t = (2 - 1.9) / 1 = 0.1, 0.025.
    Regression caught: the two place rules drifting apart (the list's direction by the reach, the place's by R)."""
    single = rec({"type": "position", "xy": [0, 0]}, credit="graded", family="object_where",
                 tolerances={"resolution_m": 0.02, "need_m": M, "harm": None})
    row = set_row([(0, 0)])
    for e, s in ((0.3, 0.7897), (0.5, 0.75), (0.8, 0.35), (1.2, 0.2), (1.9, 0.025)):
        listed, placed = credit(places((e, 0)), row), credit({"type": "position", "xy": [e, 0]}, single)
        assert listed == pytest.approx(placed) and listed == pytest.approx(s, abs=1e-4), f"{e} m off"


def test_an_empty_gold_is_one_thing_that_false_places_dilute():
    """Contract (0.11.0): where there was nothing to list, the empty list is the one thing to find, and a false place
    dilutes it as it dilutes a list of one thing: s = 1 / (1 + fp_weight |false|). By hand, fp_weight 0.5:
    - the empty list: 1; `unknown`: 0;
    - one false place: 1 / 1.5 = 2/3, three: 1 / (1 + 1.5) = 0.4;
    - fp_weight 1 and one false place: 1 / 2 = 0.5, so the row's weight is read and not a constant half;
    - a place inside a don't-care circle (radius 0.5 at (10, 10)) is excused: 1; with a second, outside it: 2/3.
    Regression caught: the cliff before 0.11.0, where one false place scored 0 / (0 + 0.5) = 0 as ten did."""
    empty = set_row([])
    assert credit(places(), empty) == 1.0 and credit(UNKNOWN, empty) == 0.0
    assert credit(places((5, 5)), empty) == pytest.approx(2 / 3)
    assert credit(places((5, 5), (7, 5), (9, 5)), empty) == pytest.approx(0.4)
    assert credit(places((5, 5)), set_row([], weight=1.0)) == pytest.approx(0.5)
    circled = set_row([], dont_care=[(10, 10, 0.5)])
    assert credit(places((10.1, 10)), circled) == 1.0
    assert credit(places((10.1, 10), (20, 20)), circled) == pytest.approx(2 / 3)


def test_a_dont_care_circle_drops_an_unmatched_answer_but_never_a_matched_one():
    """Contract (§3.4, 0.8.0's don't-care rule): an answered item that matches nothing and lies in a don't-care
    circle of the gold is neither true nor false; one that matches a gold item keeps its credit wherever it lies. By
    hand, gold (0, 0) with a circle of radius 0.5 at (10, 10) and one of radius 0.5 at (0.2, 0):
    - (0, 0) and (10.1, 10): the second is in the circle and dropped, s = 1;
    - (0, 0) and (10, 11): outside every circle, false, s = 1 / 1.5;
    - (0.2, 0) alone, inside a circle yet 0.2 m from the gold: matched, t = a = 1, f = 1 - ln 10 / ln 25 = 0.2846,
      (3.2846) / 4 = 0.8212, with no false item;
    - an empty gold and an answer inside a circle: 1.
    Regression caught: a dropped true positive, or a false item that a circle fails to excuse."""
    row = set_row([(0, 0)], dont_care=[(10, 10, 0.5), (0.2, 0, 0.5)])
    assert credit(places((0, 0), (10.1, 10)), row) == pytest.approx(1.0)
    assert credit(places((0, 0), (10, 11)), row) == pytest.approx(1 / 1.5)
    assert credit(places((0.2, 0)), row) == pytest.approx(0.8212, abs=1e-4)
    assert credit(places((10.1, 10)), set_row([], dont_care=[(10, 10, 0.5)])) == 1.0


def test_a_list_row_must_hold_a_resolution_for_each_gold_item():
    """Contract (records): a `graded_set` row with a resolution count other than its gold's, or no `match_m` or
    `fp_weight`, raises ValueError naming the key. Regression caught: a short list of resolutions read with a default,
    which would grade an item at a resolution the compiler never wrote."""
    for tolerances, key in (({"match_m": M, "resolutions_m": [0.02], "fp_weight": 0.5}, "resolutions_m"),
                            ({"match_m": M, "resolutions_m": [0.02, 0.02], "fp_weight": -1}, "fp_weight"),
                            ({"resolutions_m": [0.02, 0.02], "fp_weight": 0.5}, "match_m")):
        with pytest.raises(ValueError, match=key):
            credit(places((0, 0)), rec(places((0, 0), (1, 1)), credit="graded_set", tolerances=tolerances))
