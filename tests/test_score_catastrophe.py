"""The reported score (`score.reported`) against the penalised one (`score.penalised`) on hand rows (0.10.0,
FD-questions-0.10 §4, review B4): a catastrophe scores minus its weight, anything else scores as it did. Every
expected number is worked by hand from `credit_kernel`'s quarters (the arithmetic is in each docstring). The number
golds lie at 1.0 m, within ten needs of C = 0.10 m, where the need is the job's own (a tenth of 1.0 m is C)."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, num, off, places, pos, rec
from stbench import score

INFO = {"case": "c1", "script": "s1", "layout": "apartment_a", "robot": "vacuum", "job": "vacuum/1", "length": "short"}


def number_row(rule: dict, *, gold: float = 1.0, threshold: float | None = 0.45, **extra) -> dict:
    """A `graded` number gold of a vacuum-like job (C = 10 cm, one pixel 2 cm, the body 35 cm long) with a catastrophe
    rule."""
    tolerances = {"resolution_m": 0.02, "need_m": 0.1, "harm": "over", "catastrophe": rule}
    if threshold is not None:
        tolerances["threshold_m"] = threshold
    return rec(num(gold), credit="graded", tolerances=tolerances, family="clear_path", **extra)


def offset_row(rule: dict, gold: dict, *, resolution: float = 0.05, need: float = 0.15) -> dict:
    return rec(gold, credit="graded", family="person_mem",
               tolerances={"resolution_m": resolution, "need_m": need, "harm": "radial", "catastrophe": rule})


def listed_row(rule: dict, gold: dict) -> dict:
    return rec(gold, credit="graded_set", family="tangle_ahead",
               tolerances={"match_m": 0.5, "resolutions_m": [0.1] * len(gold["items"]), "fp_weight": 0.5,
                           "catastrophe": rule})


OVER = {"weight": 2, "rule": "over", "margin_m": 0.1}


@pytest.mark.parametrize(("value", "reported", "penalised"), [
    (1.0, 1.0, 1.0),       # exact: s = 1
    (1.05, 0.5, 0.5),      # 5 cm long, inside the margin: harmful need 0.05, t = 1, a = 1, f = 0, d = 1: s = 3/4
    (1.10, 0.0, 0.0),      # exactly the margin long: not over it. t = 1, a = 0, f = 0, d = 1: s = 1/2
    (1.11, -2.0, -0.05),   # 1 cm past the margin: a catastrophe. t = 0.9, a = 0, f = 0, d = 1: s = 0.475
    (0.5, -0.5, -0.5),     # 50 cm short is harmless: need 0.1, t = a = f = 0, d = 1: s = 1/4
])
def test_an_over_rule_charges_its_weight_only_past_the_margin_and_only_on_the_long_side(value, reported, penalised):
    """Contract (FD-questions-0.10 §4.3, `over`): a number answered longer than the gold by MORE than `margin_m` is a
    catastrophe, scoring minus the rule's weight (2); at the margin, short of it, exact or shorter than the gold it
    scores as `penalised` does (worked by hand in the table: s = (d + t + a + f) / 4, score = 2s - 1). Regression
    caught: a margin read as inclusive, a catastrophe on the short side (the safe one), or a penalised score that
    moved."""
    row = number_row(OVER)
    assert score.reported(num(value), row) == pytest.approx(reported)
    assert score.penalised(num(value), row) == pytest.approx(penalised)
    assert score.catastrophe(num(value), row) == (2 if reported == -2.0 else 0)


@pytest.mark.parametrize(("value", "weight"), [
    (1.5, 3),    # gold 1.0 is under the threshold 1.45, the answer is over it and 0.5 > the margin 0.45
    (1.46, 3),   # 0.46 over the gold, 0.01 past the margin, and over the threshold
    (1.45, 0),   # 0.45 over the gold: not past the margin
    (1.4, 0),    # under the threshold: the answer still says the body does not fit
])
def test_an_over_rule_with_a_threshold_needs_the_gold_under_it_and_the_answer_at_or_over_it(value, weight):
    """Contract (FD-questions-0.10 §4.3, `closest_person`): where the rule has a `threshold_m`, a catastrophe is a
    person who came closer than the body has room for (the gold under it) said to be clear of it (the answer at or
    over it) by more than the margin max(C, 2 rho) = max(0.45, 0.1) = 0.45. Hand cases at gold 1.0, threshold 1.45.
    Regression caught: a threshold test dropped (every long answer a catastrophe) or a margin test dropped (every
    answer over the threshold)."""
    row = number_row({"weight": 3, "rule": "over", "margin_m": 0.45, "threshold_m": 1.45}, gold=1.0, threshold=1.45)
    assert score.catastrophe(num(value), row) == weight
    assert (score.reported(num(value), row) == -3.0) is (weight == 3)


def test_a_threshold_rule_spares_a_gold_that_is_already_over_the_threshold_or_far_under_it():
    """Contract (§4.3): a gold at or over the threshold (nobody is too close), and an answer that stays under it, are
    never catastrophes whatever the excess. Gold 2.0 answered 3.0 (excess 1.0 > 0.45, gold over 1.45) and gold 0.5
    answered 1.4 (excess 0.9 > 0.45, answer under 1.45). Regression caught: the threshold ignored when the excess is
    large."""
    rule = {"weight": 3, "rule": "over", "margin_m": 0.45, "threshold_m": 1.45}
    assert score.catastrophe(num(3.0), number_row(rule, gold=2.0, threshold=1.45)) == 0
    assert score.catastrophe(num(1.4), number_row(rule, gold=0.5, threshold=1.45)) == 0


@pytest.mark.parametrize(("answer", "weight"), [
    (off(2.31, 0.0), 3),     # 0.31 farther along the bearing, the margin is 0.3
    (off(2.3, 0.0), 0),      # exactly the margin: not over it
    (off(2.0, 0.5), 0),      # off to the side: no radial excess
    (off(1.0, 0.0), 0),      # nearer than it is: the safe side
])
def test_a_radial_rule_charges_a_point_placed_farther_along_its_bearing_than_the_margin(answer, weight):
    """Contract (§4.3, `radial`, `person_mem`): an offset that puts the person farther from the camera than they are,
    by more than the margin along the gold's bearing (`score_graded.radial_error`'s sign), is a catastrophe; a sideways
    error, a nearer answer and the margin itself are not. Gold (2.0 ahead, 0 left), margin 0.3. Regression caught: the
    offset's whole 2-D error read instead of its radial part, or the sign flipped."""
    row = offset_row({"weight": 3, "rule": "radial", "margin_m": 0.3}, off(2.0, 0.0))
    assert score.catastrophe(answer, row) == weight


def test_a_radial_rule_measures_along_a_diagonal_bearing_and_ignores_a_gold_on_the_camera():
    """Contract (§4.3): the excess is the component along the gold's bearing. Gold (3, 4), bearing (0.6, 0.8): the
    answer (3.6, 4.8) is (0.6, 0.8) away, 0.36 + 0.64 = 1.0 along it; (3.5, 4.6) is 0.30 + 0.48 = 0.78 along it. With
    a margin of 0.9 the first is a catastrophe and the second is not. A gold on the camera has no bearing: no answer is
    a catastrophe. Regression caught: a norm of the error used for the excess, or a crash on the zero bearing."""
    rule = {"weight": 3, "rule": "radial", "margin_m": 0.9}
    row = offset_row(rule, off(3.0, 4.0))
    assert score.catastrophe(off(3.6, 4.8), row) == 3 and score.catastrophe(off(3.5, 4.6), row) == 0
    assert score.catastrophe(off(5.0, 5.0), offset_row(rule, off(0.0, 0.0))) == 0


@pytest.mark.parametrize(("answer", "weight"), [
    (places((1.4, 1.0), (4, 4)), 0),    # the critical point has an item 0.4 m away; the rest is not critical
    (places((1.5, 1.0)), 0),            # exactly the matching radius from it: within
    (places((1.6, 1.0), (4, 4)), 3),    # 0.6 m away: the critical point is missed
    (places((4, 4)), 3),                # only the other thing named
    (places(), 3),                      # an empty list misses it
    (places((1.0, 1.0)), 0),            # the non-critical thing left out costs credit, not a catastrophe
])
def test_a_miss_rule_charges_a_critical_point_with_no_answered_item_within_the_matching_radius(answer, weight):
    """Contract (§4.3, `drop_zone_people`, `tangle_ahead`): the answer is a catastrophe when some point of `critical`
    has no answered item within the row's `match_m` (0.5 m); an item at the radius is within, a missed point that is
    not critical is only a loss of credit. Gold (1, 1) and (4, 4), (1, 1) critical. Regression caught: every miss a
    catastrophe, the radius read as the credit's four-radii reach, or an empty list let through."""
    row = listed_row({"weight": 3, "rule": "miss", "margin_m": 1.0, "critical": [[1.0, 1.0]]}, places((1, 1), (4, 4)))
    assert score.catastrophe(answer, row) == weight


def test_unknown_is_never_a_catastrophe_and_neither_is_an_answer_of_another_type_or_to_an_unknown_gold():
    """Contract (§4.3): `unknown` takes no action, so it scores 0 on an answerable catastrophe row; a malformed answer
    and one of another type than the gold's earn credit 0 (penalised -1) and are not catastrophes; a row whose gold is
    `unknown` has no catastrophe to commit (answering a number there is -1). Regression caught: a catastrophe charged
    to an answer that took no action, or to one that never parsed."""
    row = number_row(OVER)
    assert score.reported(UNKNOWN, row) == 0.0 and score.catastrophe(UNKNOWN, row) == 0
    for wrong in ({"type": "banana"}, off(9.0, 9.0)):
        assert score.reported(wrong, row) == -1.0 and score.catastrophe(wrong, row) == 0
    blank = rec(UNKNOWN, slice="unknown", credit="graded",
                tolerances={"resolution_m": 0.02, "need_m": 0.1, "harm": "over", "catastrophe": OVER})
    assert score.reported(num(99.0), blank) == -1.0 and score.reported(UNKNOWN, blank) == 1.0
    assert score.catastrophe(num(99.0), blank) == 0


def test_a_row_without_a_catastrophe_tolerance_scores_as_it_did():
    """Contract (§4.4): the key is optional, and a row without it (a family that has none, a cap gold the compile left
    bare, a gold of an earlier build) is scored by `penalised`. The same answer that is a catastrophe above (1.11 for
    gold 1.0) is -0.05 here. Regression caught: a default rule applied to a row that declares none."""
    row = rec(num(1.0), credit="graded", tolerances={"resolution_m": 0.02, "need_m": 0.1, "harm": "over",
                                                    "threshold_m": 0.45})
    assert score.reported(num(1.11), row) == pytest.approx(-0.05) == pytest.approx(score.penalised(num(1.11), row))
    assert score.catastrophe(num(1.11), row) == 0


def test_the_result_row_carries_both_scores_and_the_catastrophe_flag():
    """Contract (§4.2): a result row's `score` is the reported score, `score_gate` the penalised one and `catastrophe`
    says whether the weight was charged: for 1.11 against gold 1.0 (worked above) -2 and -0.05 and True; for 1.05 both
    are 0.5 and False; the gold row's `age_band` rides along (null where it has none). Regression caught: the gate
    score replaced by the reported one in the row (which would push every gate), or a flag that disagrees with the
    score."""
    gold = {**number_row(OVER, qid="q1"), "scope": "memory", "reference": None, "critical": True,
            "age_band": "task"}
    hit = score.row(num(1.11), gold, when="during", info=INFO)
    assert (hit["score"], hit["catastrophe"], hit["age_band"]) == (-2.0, True, "task")
    assert hit["score_gate"] == pytest.approx(-0.05)
    near = score.row(num(1.05), gold, when="during", info=INFO)
    assert (near["score"], near["score_gate"], near["catastrophe"]) == (pytest.approx(0.5), pytest.approx(0.5), False)
    bare = {**rec(pos(0, 0), qid="q2"), "scope": "now", "reference": None, "critical": False}   # no age_band key
    assert score.row(pos(0, 0), bare, when="end", info=INFO)["age_band"] is None
