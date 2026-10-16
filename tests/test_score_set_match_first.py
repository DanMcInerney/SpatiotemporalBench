"""The set scorer matches first (0.5.0 log review, defect 5): answers are matched to gold items, and only the answers
left unmatched are dropped when they lie inside a gold don't-care place. Expected values are worked out by hand in each
test, with made-up coordinates; the radius is 0.5 m (the default `match_m`) unless a test says otherwise."""

from __future__ import annotations

import pytest

from score_kit import items, places, rec
from stbench.score import credit


def gold(points, dont_care=()):
    """A `positions` gold row from (x, y) items and (x, y, radius) don't-care places."""
    value = {"type": "positions", "items": [{"xy": [x, y]} for x, y in points],
             "dont_care": [{"xy": [x, y], "radius": r} for x, y, r in dont_care]}
    return rec(value, family="out_of_view")


def test_a_correct_answer_inside_anothers_dont_care_place_is_matched():
    """Contract: an answer that matches a gold item counts as a true positive even when it also lies inside a
    don't-care place. The d107 shape: gold items G1 (10, 0) and G2 (20, 0), and a twin's don't-care place at (10.66, 0)
    r 0.5, which does not touch G1 (0.66 m) but holds the answer (10.17, 0) (0.49 m), 0.17 m from G1. The answers
    (10.17, 0) and (20, 0) match both: tp 2, kept 2, F1 = 2*2/(2+2) = 1. Dropping before matching leaves one answer
    against two items: 2*1/(1+2) = 2/3, the 0.33 a perfect answer lost. The answer (10.17, 0) alone finds one of the
    two: tp 1, kept 1, F1 = 2*1/(1+2) = 2/3 (dropped first, it scored 0). Regression caught: the drop-before-match
    order, which scored a right answer as if it had not been given."""
    g = gold([(10, 0), (20, 0)], dont_care=[(10.66, 0, 0.5)])
    assert credit(places((10.17, 0), (20, 0)), g) == 1.0
    assert credit(places((10.17, 0)), g) == pytest.approx(2 / 3)


def test_a_match_inside_a_dont_care_place_still_counts_for_a_set_with_change_types():
    """Contract: the same for a `set` answer, where the change type still has to agree to match. Gold: `moved` at
    (0, 0), and a don't-care place at (0.55, 0) r 0.5 (0.55 m from the gold item, 0.45 m from the answers below).
    `moved` at (0.1, 0) matches: F1 = 1. `added` at (0.1, 0) matches nothing, is unmatched inside the place and is
    dropped: nothing kept against one gold item, F1 = 0. Both together: tp 1, the `added` one dropped, kept 1:
    2*1/(1+1) = 1. Regression caught: a type-blind match inside a place, or the drop-before-match order scoring the
    first answer 0."""
    g = rec({"type": "set", "items": [{"xy": [0, 0], "change": "moved"}],
             "dont_care": [{"xy": [0.55, 0], "radius": 0.5}]}, family="changed")
    assert credit(items((0.1, 0, "moved")), g) == 1.0
    assert credit(items((0.1, 0, "added")), g) == 0.0
    assert credit(items((0.1, 0, "moved"), (0.1, 0, "added")), g) == 1.0


def test_an_unmatched_answer_inside_a_dont_care_place_is_ignored():
    """Contract: an answer that matches no gold item and lies within a don't-care radius (inclusive) is neither a true
    nor a false positive. Gold: G (0, 0) and a place at (5, 5) r 0.5. The answers (0, 0) and (5.3, 5): tp 1, the second
    dropped, kept 1: F1 = 2*1/(1+1) = 1; (5.5, 5), exactly on the radius, is dropped too. The same two answers against a
    gold without the place: kept 2, F1 = 2*1/(2+1) = 2/3, so the place is what changes the score. Regression caught: an
    unmatched answer inside a place counted as a false positive, or the radius made exclusive."""
    g = gold([(0, 0)], dont_care=[(5, 5, 0.5)])
    assert credit(places((0, 0), (5.3, 5)), g) == 1.0
    assert credit(places((0, 0), (5.5, 5)), g) == 1.0
    assert credit(places((0, 0), (5.3, 5)), gold([(0, 0)])) == pytest.approx(2 / 3)


def test_an_unmatched_answer_outside_every_dont_care_place_is_a_false_positive():
    """Contract: an unmatched answer outside every don't-care radius counts against precision. Gold: G (0, 0) and places
    at (5, 5) r 0.5 and (8, 8) r 0.5. The answers (0, 0) and (6, 5) (1 m from the first place): tp 1, kept 2, F1 =
    2*1/(2+1) = 2/3; the answers (0, 0), (5.3, 5) and (6, 5): tp 1, the middle one dropped, kept 2: 2/3 again. An answer
    0.6 m from G is unmatched with no place near it: (0.6, 0) alone is tp 0, F1 = 0. Regression caught: dropping
    every unmatched answer once any place exists, or a dropped answer making the next one free."""
    g = gold([(0, 0)], dont_care=[(5, 5, 0.5), (8, 8, 0.5)])
    assert credit(places((0, 0), (6, 5)), g) == pytest.approx(2 / 3)
    assert credit(places((0, 0), (5.3, 5), (6, 5)), g) == pytest.approx(2 / 3)
    assert credit(places((0.6, 0)), g) == 0.0


def test_empty_gold_is_one_only_when_every_answer_lies_in_a_dont_care_place():
    """Contract: with no gold items, every answer is unmatched, so the answers outside the places decide. Gold: none, a
    place at (5, 5) r 0.5. The answers (5.1, 5) and (4.8, 5), both inside: both dropped, nothing kept and nothing to
    find: F1 = 1. One answer (7, 7) outside: kept 1 against 0 gold items, F1 = 2*0/(1+0) = 0; that one beside an
    answer inside: 0 again. No answers at all: 1. Regression caught: the both-empty rule read before the drop (a lone
    answer in a place scoring 0), or an answer outside the places forgiven because the gold is empty."""
    g = gold([], dont_care=[(5, 5, 0.5)])
    assert credit(places((5.1, 5), (4.8, 5)), g) == 1.0
    assert credit(places((7, 7)), g) == 0.0
    assert credit(places((5.1, 5), (7, 7)), g) == 0.0
    assert credit(places(), g) == 1.0


def test_of_two_answers_for_one_gold_item_the_one_in_a_place_is_the_one_left_over():
    """Contract: two answers near one gold item match one gold item once, and the other is a false positive unless it
    lies in a don't-care place, in which case it is dropped; which of the two matches is the one outside the place,
    whatever the order they were given in. Gold: G (0, 0). A (0.1, 0) and B (0.2, 0) both reach G.
    - no place: tp 1, kept 2: F1 = 2*1/(2+1) = 2/3;
    - a place at (0.65, 0) r 0.5 holds B (0.45 m) but not A (0.55 m) nor G (0.65 m): A matches, B is dropped, kept 1:
      F1 = 2*1/(1+1) = 1, in either order (matching B first would leave A a false positive: 2/3);
    - a place at (0.55, 0) r 0.5 holds both (0.45 m and 0.35 m) but not G (0.55 m): one matches, the other is
      dropped, kept 1: F1 = 1.
    Regression caught: the score depending on the order of the answers, or the leftover answer in a place counted."""
    plain = gold([(0, 0)])
    assert credit(places((0.1, 0), (0.2, 0)), plain) == pytest.approx(2 / 3)
    holds_b = gold([(0, 0)], dont_care=[(0.65, 0, 0.5)])
    assert credit(places((0.1, 0), (0.2, 0)), holds_b) == 1.0
    assert credit(places((0.2, 0), (0.1, 0)), holds_b) == 1.0
    holds_both = gold([(0, 0)], dont_care=[(0.55, 0, 0.5)])
    assert credit(places((0.1, 0), (0.2, 0)), holds_both) == 1.0
    assert credit(places((0.2, 0), (0.1, 0)), holds_both) == 1.0


def test_an_answer_in_a_dont_care_place_still_takes_the_match_the_maximum_needs():
    """Contract: putting the answers outside the places first does not give up a maximum matching. Gold: G1 (0, 0), G2
    (0.6, 0); a place at (-0.9, 0) r 0.5. A (0.2, 0) is outside and reaches G1 (0.2 m) and G2 (0.4 m); B (-0.45, 0) is
    inside the place (0.45 m) and reaches G1 only (0.45 m; G2 is 1.05 m). The maximum is A to G2 and B to G1: tp 2,
    kept 2, F1 = 2*2/(2+2) = 1; a matching that gave A to G1 would strand B, drop it and score 2*1/(1+2) = 2/3.
    Regression caught: the outside-first order replacing the augmenting search with a greedy pass."""
    g = gold([(0, 0), (0.6, 0)], dont_care=[(-0.9, 0, 0.5)])
    assert credit(places((0.2, 0), (-0.45, 0)), g) == 1.0
    assert credit(places((-0.45, 0), (0.2, 0)), g) == 1.0
