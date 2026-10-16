"""The scorer (V0 §8): credit per answer type, the penalised-score table and outcome classes (aggregation is in
test_score_stats.py). Expected values are worked out by hand in each test."""

from __future__ import annotations

import pytest

from score_kit import ORIGIN, UNKNOWN, clock, count, items, num, pos, rec
from stbench.answers import Malformed, Position, SetAnswer, SetItem, Unknown, parse_answer
from stbench.score import credit, outcome, penalised, row
from stbench.score_stats import ROW_KEYS


@pytest.mark.parametrize(
    ("error_m", "expected"),
    [(0.0, 1.0), (0.25, 1.0), (0.2500001, 0.75), (0.5, 0.75), (0.5000001, 0.5), (1.0, 0.5), (1.0000001, 0.25),
     (2.0, 0.25), (2.0000001, 0.0), (50.0, 0.0)],
)
def test_position_credit_is_the_share_of_the_four_default_thresholds_met(error_m, expected):
    """Contract: position credit = share of {0.25, 0.5, 1, 2} m with error <= threshold (inclusive at the edge).

    Regression caught: a strict `<` dropping an answer exactly on a threshold, or a ladder step shifted."""
    assert credit(pos(error_m, 0), ORIGIN) == expected


def test_threshold_comparison_survives_float_noise():
    """Contract: an error that is exactly a threshold in real arithmetic counts as met even if the float result is a
    hair over. A 0.3 / 0.4 offset is 0.5 m but squares to 0.25000000000000006.

    Regression caught: a bare `<=` on floats marking a correct answer short of a rung."""
    assert credit(pos(0.3, 0.4), ORIGIN) == 0.75


def test_ladders_are_read_from_the_gold_record():
    """Contract: `tolerances.ladder` replaces the default thresholds (maps with other scales), and each rung counts.

    Regression caught: hard-coded defaults ignoring the per-map scaling stored with the gold."""
    scaled = rec(pos(0, 0), tolerances={"ladder": [0.5, 1.0, 2.0, 4.0]})
    assert credit(pos(3.0, 0), scaled) == 0.25
    assert credit(pos(1.0, 0), scaled) == 0.75
    for bad in ([], [0.5, -1], [True]):
        with pytest.raises(ValueError, match="ladder"):
            credit(pos(0, 0), rec(pos(0, 0), tolerances={"ladder": bad}))


def test_xy_metric_ignores_height_and_xyz_metric_uses_it():
    """Contract: metric `xy` judges horizontal error only (an xyz answer is fine); metric `xyz` uses the 3D distance
    and gives 0 to an answer without z.

    Regression caught: a floor-plan question punishing a wrong z, or a 3D question accepting a flat answer."""
    flat = rec(pos(0, 0, 0), metric="xy")
    full = rec(pos(0, 0, 0), metric="xyz")
    assert credit(pos(0, 0, 0.6), flat) == 1.0
    assert credit(pos(0, 0), flat) == 1.0
    assert credit(pos(0, 0, 0.6), full) == 0.5  # 3D error 0.6: meets 1 and 2 only
    assert credit(pos(0, 0), full) == 0.0
    assert credit(pos(0.3, 0, 0.4), full) == 0.75  # 3-4-5: error 0.5


def test_number_credit_uses_the_distance_ladder_on_the_absolute_difference():
    """Contract: number credit = share of the metre ladder met by |answer - gold|, whichever side the answer is on.

    Regression caught: a signed difference crediting every under-estimate."""
    g = rec(num(3.0), family="ego_distance")
    assert credit(num(3.9), g) == 0.5
    assert credit(num(2.1), g) == 0.5
    assert credit(num(1.5), g) == 0.25
    assert credit(num(2.0), g) == 0.5
    assert credit(num(3.25), g) == 1.0


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("2026-05-14T08:37:03", 1.0), ("2026-05-14T08:36:57", 1.0), ("2026-05-14T08:37:04", 0.75),
     ("2026-05-14T08:37:10", 0.75), ("2026-05-14T08:37:11", 0.5), ("2026-05-14T08:37:30", 0.5),
     ("2026-05-14T08:37:31", 0.25), ("2026-05-14T08:38:30", 0.25), ("2026-05-14T08:38:31", 0.0)],
)
def test_time_credit_is_the_share_of_the_second_ladder_met(answer, expected):
    """Contract: time credit = share of {3 s, 10 s, 30 s, 90 s} met, inclusive, on |answer - gold| (gold 08:37:00):
    the ladder is on the timescale of a visit (a keyframe is 1.5 s), not of a day.

    Regression caught: minutes mistaken for seconds, or a rung shifted by one."""
    assert credit(clock(answer), rec(clock("2026-05-14T08:37:00"), family="when_last")) == expected


def test_time_error_is_the_absolute_difference_across_midnight_and_dates():
    """Contract: time error is the real elapsed time between two datetimes: 40 s across midnight, a full day apart
    for the same clock on adjacent dates.

    Regression caught: comparing time of day only (same clock on another day scoring 1) or wrapping at midnight."""
    g = rec(clock("2026-05-14T23:59:30"), family="when_last")
    assert credit(clock("2026-05-15T00:00:10"), g) == 0.25  # 40 s: meets 90 only
    assert credit(clock("2026-05-14T23:59:30"), g) == 1.0
    assert credit(clock("2026-05-15T23:59:30"), g) == 0.0  # 86400 s


@pytest.mark.parametrize(
    ("truth", "estimate", "expected"),
    [(10, 10, 1.0), (10, 9, 0.8), (10, 8, 0.6), (10, 12, 0.6), (10, 7, 0.4), (10, 6, 0.2), (10, 5, 0.0),
     (10, 15, 0.0), (10, 0, 0.0), (20, 19, 0.9), (4, 3, 0.5), (1, 2, 0.0), (0, 0, 1.0), (0, 1, 0.0)],
)
def test_count_credit_is_mean_relative_accuracy_over_ten_thresholds(truth, estimate, expected):
    """Contract: count credit = VSI-Bench MRA, mean over theta = 0.50, 0.55, ..., 0.95 of [|est - y| / y < 1 - theta]
    (strict). Worked: y=10, est=9 has relative error 0.1, below 1-theta for theta <= 0.85, which is 8 of 10 steps;
    y=20, est=19 has error 0.05, met for theta <= 0.90 only (9 of 10); y=4, est=3 has error 0.25, met for theta <= 0.70
    (5 of 10).

    Regression caught: `<=` instead of `<` (an error equal to 1-theta crediting), or theta steps off by one."""
    assert credit(count(estimate), rec(count(truth), family="count")) == pytest.approx(expected)


def test_yesno_credit_is_exact_match():
    """Contract: yes/no is 1 for the matching boolean, else 0.

    Regression caught: truthiness comparison letting a wrong answer through."""
    g = rec({"type": "yesno", "value": True}, family="moved_between")
    assert credit({"type": "yesno", "value": True}, g) == 1.0
    assert credit({"type": "yesno", "value": False}, g) == 0.0


def test_set_credit_f1_perfect_partial_and_change_type():
    """Contract: set credit = F1 of matches within 0.5 m (inclusive) with the same change type. Worked: gold 2 items,
    answer 3 items of which 2 match: F1 = 2*2/(3+2) = 0.8; a right place with the wrong change type matches nothing;
    0.5 m apart matches, 0.51 m does not.

    Regression caught: ignoring the change type, an exclusive radius, or precision/recall mixed up."""
    g = rec(items((0, 0, "added"), (3, 0, "removed")), family="changed")
    assert credit(items((0, 0, "added"), (3, 0, "removed")), g) == 1.0
    assert credit(items((0.1, 0, "added"), (3, 0.1, "removed"), (8, 8, "moved")), g) == pytest.approx(0.8)
    assert credit(items((0, 0, "removed"), (3, 0, "added")), g) == 0.0
    assert credit(items((0.5, 0, "added"), (3, 0, "removed")), g) == 1.0
    assert credit(items((0.51, 0, "added"), (3, 0, "removed")), g) == pytest.approx(0.5)
    wide = rec(items((0, 0, "added")), family="changed", tolerances={"match_m": 1.0})
    assert credit(items((0.9, 0, "added")), wide) == 1.0


def test_set_credit_uses_an_optimal_assignment_not_a_greedy_one():
    """Contract: items are assigned one-to-one to maximise matches. Gold g1=(0,0), g2=(0.6,0); answers a1=(0.2,0),
    a2=(-0.45,0), all `added`. a1 is nearest to g1, but a2 can only reach g1, so the optimum is a1->g2 and a2->g1: two
    matches, F1 = 1. A nearest-first greedy takes a1->g1, strands a2 and gets F1 = 2*1/(2+2) = 0.5.

    Regression caught: replacing the assignment with greedy matching, which under-scores a correct answer."""
    g = rec(items((0, 0, "added"), (0.6, 0, "added")), family="changed")
    assert credit(items((0.2, 0, "added"), (-0.45, 0, "added")), g) == 1.0
    assert credit(items((-0.45, 0, "added"), (0.2, 0, "added")), g) == 1.0


def test_one_answer_item_cannot_match_two_gold_items():
    """Contract: matching is one-to-one. One answer between two gold items of one type matches one: F1 = 2*1/(1+2).

    Regression caught: counting every gold item within range as hit, inflating recall."""
    g = rec(items((0, 0, "moved"), (0.4, 0, "moved")), family="changed")
    assert credit(items((0.2, 0, "moved")), g) == pytest.approx(2 / 3)


def test_dont_care_regions_remove_nearby_answers_from_true_and_false_positives():
    """Contract: an answer item that matches no gold item and lies within a gold don't-care radius counts neither as a
    hit nor as a false positive (matching comes first: `test_score_set_match_first.py`).
    Gold: one item plus a region at (5,5) r=0.5. Answer: the item and a second at (5.3,5): F1 = 1 (not 2/3). An answer
    at (6,5), outside the region, is a false positive: F1 = 2*1/(2+1). A gold with no real items and an answer only
    inside the region is both-empty: F1 = 1.

    Regression caught: scoring identical-item ambiguity as a false positive, or exclusion ignoring the radius."""
    dc = {"type": "set", "items": [{"xy": [0, 0], "change": "moved"}], "dont_care": [{"xy": [5, 5], "radius": 0.5}]}
    g = rec(dc, family="changed")
    assert credit(items((0, 0, "moved"), (5.3, 5, "added")), g) == 1.0
    assert credit(items((0, 0, "moved"), (6, 5, "added")), g) == pytest.approx(2 / 3)
    assert credit(items((0, 0, "moved"), (5.5, 5, "added")), g) == 1.0  # exactly on the radius
    only_region = rec({"type": "set", "items": [], "dont_care": [{"xy": [5, 5], "radius": 0.5}]}, family="changed")
    assert credit(items((5.1, 5, "removed")), only_region) == 1.0


def test_empty_sets_score_one_when_both_empty_and_zero_when_only_one_is():
    """Contract: F1 = 1 for empty gold and empty answer; 0 when exactly one of them is empty.

    Regression caught: 0/0 read as 0 (punishing a correct 'nothing changed') or an empty answer to a non-empty gold
    scoring as a vacuous 1."""
    assert credit(items(), rec(items(), family="changed")) == 1.0
    assert credit(items((1, 1, "added")), rec(items(), family="changed")) == 0.0
    assert credit(items(), rec(items((1, 1, "added")), family="changed")) == 0.0


@pytest.mark.parametrize(
    ("answer", "gold"),
    [(pos(0, 0), num(0)), (num(1), count(1)), (count(1), num(1)), ({"type": "yesno", "value": True}, count(1)),
     (items(), pos(0, 0)), (clock("2026-05-14T08:37:00"), pos(0, 0)), (UNKNOWN, pos(0, 0)), (UNKNOWN, count(2)),
     ({"type": "position"}, pos(0, 0)), ({"type": "count", "value": 3.0}, count(3)), (None, pos(0, 0)),
     ("0,0", pos(0, 0)), ({"type": "yesno", "value": "yes"}, {"type": "yesno", "value": True})],
)
def test_wrong_type_malformed_and_unknown_on_an_answerable_gold_all_earn_zero(answer, gold):
    """Contract: an answer of another type than the gold's, a malformed one (as a raw object or a Malformed marker),
    and `unknown` earn credit 0 on an answerable gold.

    Regression caught: a number answer to a count gold (or vice versa) earning partial credit, or a malformed answer
    raising instead of scoring 0."""
    assert credit(answer, rec(gold)) == 0.0
    assert credit(parse_answer(answer), rec(gold)) == 0.0


def test_credit_accepts_raw_objects_and_parsed_answers_alike():
    """Contract: `credit` takes either a raw JSON answer or its parsed form, with the same result.

    Regression caught: the runner and the tests disagreeing because only one input form is scored."""
    g = rec(pos(0, 0))
    assert credit(pos(0.25, 0), g) == credit(Position((0.25, 0.0)), g) == 1.0


def test_credit_on_unknown_gold_is_one_only_for_unknown():
    """Contract: on an unknown gold, `unknown` has credit 1 and any answer (malformed included) has 0.

    Regression caught: crediting a confident answer about something never seen."""
    g = rec(UNKNOWN, slice="unknown")
    assert credit(UNKNOWN, g) == 1.0
    assert credit(pos(0, 0), g) == 0.0
    assert credit(Malformed("x"), g) == 0.0


def test_inconsistent_gold_record_is_a_loud_error():
    """Contract: a gold whose slice disagrees with its value, or whose value is invalid, raises ValueError.

    Regression caught: a compiler bug silently producing wrong penalised scores."""
    for bad in (rec(pos(0, 0), slice="unknown"), rec(UNKNOWN, slice="answerable"), rec({"type": "count", "value": -1})):
        with pytest.raises(ValueError, match="gold"):
            credit(UNKNOWN, bad)
        with pytest.raises(ValueError, match="gold"):
            penalised(pos(0, 0), bad)


@pytest.mark.parametrize(
    ("answer", "gold_rec", "t", "expected"),
    [
        (pos(0, 0), ORIGIN, 0.5, 1.0),                                       # answerable, s=1: 1 - 0
        (pos(0.6, 0), ORIGIN, 0.5, 0.0),                                     # s=0.5: 0.5 - 0.5*1
        (pos(0.3, 0), ORIGIN, 0.5, 0.5),                                     # s=0.75: 0.75 - 0.25
        (pos(9, 0), ORIGIN, 0.5, -1.0),                                      # s=0: 0 - 1*1
        (pos(0.6, 0), ORIGIN, 0.75, -1.0),                                   # k=3: 0.5 - 0.5*3
        (pos(9, 0), ORIGIN, 0.75, -3.0),                                     # k=3: 0 - 3
        (pos(0.6, 0), ORIGIN, 0.0, 0.5),                                     # k=0: no penalty, s
        (UNKNOWN, ORIGIN, 0.5, 0.0),                                         # answerable, unknown
        (UNKNOWN, ORIGIN, 0.75, 0.0),
        (UNKNOWN, rec(UNKNOWN, slice="unknown"), 0.5, 1.0),                  # unknown gold, unknown
        (UNKNOWN, rec(UNKNOWN, slice="unknown"), 0.75, 1.0),
        (pos(0, 0), rec(UNKNOWN, slice="unknown"), 0.5, -1.0),               # unknown gold, an answer: -k
        (pos(0, 0), rec(UNKNOWN, slice="unknown"), 0.75, -3.0),
        ({"type": "bogus"}, ORIGIN, 0.5, -1.0),                              # malformed: an answer with s=0
        ({"type": "bogus"}, rec(UNKNOWN, slice="unknown"), 0.5, -1.0),
    ],
)
def test_penalised_score_table(answer, gold_rec, t, expected):
    """Contract: V0 §8's table with k = t/(1-t): answerable+answer s-(1-s)k; answerable+unknown 0; unknown gold+unknown
    1; unknown gold+answer -k. Malformed counts as an answer.

    Regression caught: any row of the table mis-signed, `unknown` costing something, or k fixed at 1."""
    assert penalised(answer, gold_rec, t) == pytest.approx(expected)


def test_penalised_rejects_t_outside_zero_to_one():
    """Contract: t must be in [0, 1) (k is infinite at 1, negative below 0).

    Regression caught: a division by zero or a negative penalty accepted silently."""
    for t in (1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="t must be"):
            penalised(UNKNOWN, ORIGIN, t)


def test_outcome_classes():
    """Contract: correct = s of 1 or `unknown` on an unknown gold; partial = 0 < s < 1; wrong = answered with s = 0
    (malformed and an answer on an unknown gold included); abstained = `unknown` on an answerable gold.

    Regression caught: counting an abstention as an error, or an answer on an unknown gold as merely partial."""
    unk = rec(UNKNOWN, slice="unknown")
    assert outcome(pos(0, 0), ORIGIN) == "correct"
    assert outcome(pos(0.6, 0), ORIGIN) == "partial"
    assert outcome(pos(9, 0), ORIGIN) == "wrong"
    assert outcome({"type": "bogus"}, ORIGIN) == "wrong"
    assert outcome(UNKNOWN, ORIGIN) == "abstained"
    assert outcome(UNKNOWN, unk) == "correct"
    assert outcome(pos(0, 0), unk) == "wrong"


def test_row_builds_the_result_row_from_the_gold_the_question_and_the_case():
    """Contract: a result row holds exactly `ROW_KEYS`: the case's `info` (case, script, layout, robot, job, length),
    the question's `when`, the gold row's qid, family, scope, reference, critical and slice, and the credit, the
    reported score (`score`), the penalised one (`score_gate`), whether the answer is a catastrophe, the gold row's age
    band and the outcome. Worked: an answer 0.6 m off meets the 1 m and 2 m rungs, so s = 0.5, score = score_gate = 0.5
    - 0.5 = 0, no catastrophe (the gold row declares none), and the outcome is partial.

    Regression caught: a renamed or missing key that `aggregate` and the gates then fail on, or a tag (scope, own path,
    critical) read from the wrong place."""
    gold = {**rec(pos(0, 0), qid="qz", family="path_ahead"), "scope": "now", "reference": "path_log",
            "critical": True, "answer_class": "answerable"}
    info = {"case": "c9", "script": "s4", "layout": "apartment_a", "robot": "vacuum", "job": "vacuum/1",
            "length": "short"}
    made = row(pos(0.6, 0), gold, when="during", info=info)
    assert made == {"case": "c9", "script": "s4", "qid": "qz", "family": "path_ahead", "when": "during",
                    "scope": "now", "reference": "path_log", "critical": True, "slice": "answerable",
                    "answer_class": "answerable", "range_m": None, "age_band": None, "s": 0.5, "score": 0.0,
                    "score_gate": 0.0, "catastrophe": False, "outcome": "partial",
                    "layout": "apartment_a", "robot": "vacuum", "job": "vacuum/1", "length": "short"}
    assert tuple(made) == ROW_KEYS
    # a gold of a compile before 0.6.0 has no class: its row reads null, so an older run stays scorable
    older = {k: v for k, v in gold.items() if k != "answer_class"}
    assert row(pos(0.6, 0), older, when="during", info=info)["answer_class"] is None


def test_row_refuses_an_unknown_when_and_an_info_that_lacks_a_key():
    """Contract: `when` is during or end, and `info` names the case, script, layout, robot, job and length.
    Regression caught: a row filed under a `when` no split knows, or a case whose script is missing, which would put
    its questions in no cluster."""
    gold = {**ORIGIN, "scope": "memory", "reference": None, "critical": False}
    info = {"case": "c", "script": "s", "layout": "l", "robot": "r", "job": "j", "length": "short"}
    with pytest.raises(ValueError, match="when must be"):
        row(UNKNOWN, gold, when="after", info=info)
    with pytest.raises(ValueError, match="script"):
        row(UNKNOWN, gold, when="end", info={k: v for k, v in info.items() if k != "script"})


def test_parsed_answers_are_scored_without_reparsing():
    """Contract: the scorer accepts already-parsed answers (a SetAnswer, an Unknown) as well as raw objects.

    Regression caught: `_coerce` re-parsing a parsed answer as a raw object and flagging it Malformed."""
    g = rec(items((0, 0, "added")), family="changed")
    assert credit(SetAnswer((SetItem((0.0, 0.0), "added"),)), g) == 1.0
    assert credit(Unknown(), ORIGIN) == 0.0
