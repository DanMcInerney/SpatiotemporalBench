"""Scoring a `positions` answer: F1 over places under one-to-one matching, don't-care discs
neither hit nor miss, `tolerances.match_m` the radius. Expected values are worked out by hand in each test."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, items, places, rec
from stbench.answers import Malformed, Position, Positions
from stbench.score import credit, outcome, penalised


def gold(*points, **kwargs):
    return rec(places(*points), family="inventory", **kwargs)


def test_credit_is_f1_over_the_places_matched_within_half_a_metre():
    """Contract: positions credit = F1 of one-to-one matches within 0.5 m (inclusive). Worked: gold (0,0) and (3,0);
    an answer with one place near the first and one far from both matches 1 of 2 and answers 2: F1 = 2*1/(2+2) = 0.5;
    both near: 1; an extra far place: 2*2/(3+2) = 0.8; 0.5 m apart matches, 0.51 m does not. Regression caught:
    precision and recall mixed up, an exclusive radius, a gold place counted without a match."""
    g = gold((0, 0), (3, 0))
    assert credit(places((0.2, 0), (9, 9)), g) == 0.5
    assert credit(places((0.2, 0), (3, 0.3)), g) == 1.0
    assert credit(places((0.2, 0), (3, 0.3), (9, 9)), g) == pytest.approx(0.8)
    assert credit(places((0.5, 0), (3.5, 0)), g) == 1.0
    assert credit(places((0.51, 0), (3, 0.51)), g) == 0.0


def test_a_place_answered_twice_matches_once():
    """Contract: matching is one-to-one, so a repeated answer place is a false positive. Worked: one gold place, the
    answer [(0,0), (0,0)] matches one: F1 = 2*1/(2+1) = 2/3; against two gold places 0.4 m apart the same two answers
    match both: 1. Regression caught: counting every answer within range as a hit (duplicates inflating precision, or
    one answer covering two gold places)."""
    assert credit(places((0, 0), (0, 0)), gold((0, 0))) == pytest.approx(2 / 3)
    assert credit(places((0.1, 0), (0.1, 0)), gold((0, 0), (0.4, 0))) == 1.0
    assert credit(places((0.2, 0)), gold((0, 0), (0.4, 0))) == pytest.approx(2 / 3)


def test_the_assignment_is_optimal_not_greedy():
    """Contract: places are assigned to maximise matches. Gold g1=(0,0), g2=(0.6,0); answers a1=(0.2,0), a2=(-0.45,0).
    a1 is nearest to g1, but a2 can only reach g1, so the optimum is a1->g2 and a2->g1: F1 = 1. A nearest-first greedy
    takes a1->g1, strands a2 and gets 2*1/(2+2) = 0.5. Regression caught: greedy matching under-scoring a correct
    answer."""
    g = gold((0, 0), (0.6, 0))
    assert credit(places((0.2, 0), (-0.45, 0)), g) == 1.0
    assert credit(places((-0.45, 0), (0.2, 0)), g) == 1.0


def test_match_m_in_the_gold_sets_the_radius():
    """Contract: `tolerances.match_m` replaces the 0.5 m radius (it carries the widening of a large thing). Gold (0,0),
    answer (0.9,0): no match by default (0), a match with match_m 1.0 (1.0); exactly on the radius matches. Regression
    caught: the radius ignored for this type, so a perfect perceiver's error on a pallet is a miss."""
    assert credit(places((0.9, 0)), gold((0, 0))) == 0.0
    assert credit(places((0.9, 0)), gold((0, 0), tolerances={"match_m": 1.0})) == 1.0
    assert credit(places((1.0, 0)), gold((0, 0), tolerances={"match_m": 1.0})) == 1.0
    assert credit(places((1.1, 0)), gold((0, 0), tolerances={"match_m": 1.0})) == 0.0


def test_dont_care_discs_remove_answer_places_from_hits_and_misses():
    """Contract: an unmatched answer place within a gold don't-care radius is neither a hit nor a false positive (it is
    matched first: `test_score_set_match_first.py`). Gold: one place
    (0,0) and a disc at (5,5) r=0.5. Answer (0,0) and (5.3,5): F1 = 1 (not 2*1/(2+1)); (0,0) and (6,5), outside the
    disc: 2/3; exactly on the radius counts as inside. A gold with no places and an answer only inside its disc is
    both-empty: 1. Regression caught: a glimpsed thing's place scored as a false positive, or the radius ignored."""
    disc = {"type": "positions", "items": [{"xy": [0, 0]}], "dont_care": [{"xy": [5, 5], "radius": 0.5}]}
    g = rec(disc, family="inventory")
    assert credit(places((0, 0), (5.3, 5)), g) == 1.0
    assert credit(places((0, 0), (6, 5)), g) == pytest.approx(2 / 3)
    assert credit(places((0, 0), (5.5, 5)), g) == 1.0
    only = rec({"type": "positions", "items": [], "dont_care": [{"xy": [5, 5], "radius": 0.5}]}, family="inventory")
    assert credit(places((5.1, 5)), only) == 1.0


def test_empty_lists_score_one_against_empty_and_zero_against_places():
    """Contract: an empty answer to an empty gold is 1, and exactly one empty side is 0 (and a wrong, not an abstained,
    answer: an empty list is an answer). Regression caught: 0/0 read as 0 (punishing a correct "none") or an empty
    answer to a non-empty gold passing as a vacuous 1, or as `unknown`."""
    assert credit(places(), gold()) == 1.0
    assert credit(places((1, 1)), gold()) == 0.0
    assert credit(places(), gold((1, 1))) == 0.0
    assert outcome(places(), gold((1, 1))) == "wrong" and penalised(places(), gold((1, 1))) == -1.0
    assert outcome(places(), gold()) == "correct"


def test_places_are_matched_in_the_floor_plane_whatever_the_height_or_the_metric():
    """Contract: heights are ignored (an `xyz` answer item matches an `xy` gold item), and a `metric` of `xyz` on the
    record does not make places 3D: positions are judged in the floor plane, as Recall's are. Regression caught: a 3D
    distance for a thing on a rack level (2.6 m of height added to every error), or an `xyz` answer scoring 0."""
    xyz = {"type": "positions", "items": [{"xyz": [0.1, 0, 2.6]}]}
    assert credit(xyz, gold((0, 0))) == 1.0
    assert credit(xyz, rec({"type": "positions", "items": [{"xyz": [0, 0, 0]}]}, family="inventory",
                           metric="xyz")) == 1.0


def test_positions_and_set_answers_never_earn_credit_for_each_other_and_unknown_abstains():
    """Contract: a `set` answer on a positions gold, a positions answer on a set gold, a malformed item (a `change`)
    and `unknown` earn credit 0; `unknown` is an abstention (penalised score 0, not -1). Regression caught: a set's
    events matched as places (or the reverse) through the shared core, or an abstention scored as a wrong answer."""
    g = gold((0, 0))
    changes = rec(items((0, 0, "added")), family="changed")
    assert credit(items((0, 0, "added")), g) == 0.0
    assert credit(places((0, 0)), changes) == 0.0
    assert credit({"type": "positions", "items": [{"xy": [0, 0], "change": "added"}]}, g) == 0.0
    assert credit(UNKNOWN, g) == 0.0 and outcome(UNKNOWN, g) == "abstained" and penalised(UNKNOWN, g) == 0.0


def test_a_parsed_positions_answer_is_scored_without_reparsing():
    """Contract: the scorer takes an already-parsed `Positions` as well as a raw object. Regression caught: `_coerce`
    not knowing the new type, re-parsing it as a raw object and flagging it Malformed (credit 0)."""
    g = gold((0, 0))
    assert credit(Positions((Position((0.0, 0.0)),)), g) == 1.0
    assert credit(Malformed("x"), g) == 0.0


def test_a_partial_list_is_a_partial_outcome_with_the_penalised_score_of_its_credit():
    """Contract: credit 0.5 is `partial` and its penalised score at t = 0.5 is s - (1 - s) = 0.0, as for any type.
    Regression caught: the new type bypassing the outcome classes or the penalty."""
    g = gold((0, 0), (3, 0))
    answer = places((0.2, 0), (9, 9))
    assert outcome(answer, g) == "partial" and penalised(answer, g) == pytest.approx(0.0)
