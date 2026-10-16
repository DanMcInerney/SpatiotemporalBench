"""The 0.10.0 gold-row contracts (FD-questions-0.10 §4.4-4.5, step 0): a harm row may carry a `catastrophe` rule beside
its credit's keys, and every row may carry its evidence age (`age_s`, `age_band`), which rows of earlier builds lack."""

from __future__ import annotations

import pytest

from stbench.look_windows import LOOK_WINDOW, recent
from stbench.records import AGE_BANDS, GOLD_OPTIONAL, check_gold


def row(**over) -> dict:
    """A valid `clear_path`-like gold row: a number with the graded credit's keys."""
    base = {"qid": "fedcba9876543210", "family": "clear_path", "gold": {"type": "number", "value": 1.4},
            "slice": "answerable", "answer_class": "answerable", "range_m": None, "credit": "graded", "region": None,
            "tolerances": {"resolution_m": 0.02, "need_m": 0.15, "harm": "over", "threshold_m": 0.49},
            "metric": None, "scope": "now", "reference": None, "critical": False, "subjects": ["w:12-12"],
            "evidence_k": [12], "entities": []}
    return {**base, **over}


def with_catastrophe(rule: dict) -> dict:
    r = row()
    r["tolerances"] = {**r["tolerances"], "catastrophe": rule}
    return r


def test_a_row_of_an_earlier_build_without_the_age_keys_stays_valid():
    """Contract: the age keys are optional, so the gold of 0.9.0's data still validates. Regression caught: adding the
    keys to the required set, which would make every earlier dataset unreadable."""
    assert GOLD_OPTIONAL == ("age_s", "age_band") and check_gold(row()) == []


@pytest.mark.parametrize("age_s, band", [(0.0, "now"), (6.5, "recent"), (310.0, "task"), (5400.0, "earlier"),
                                         (None, None)])
def test_the_age_keys_accept_a_non_negative_age_and_a_known_band(age_s, band):
    """Contract: `age_s` is null or at least 0 and `age_band` null or one of the four bands. Regression caught: a
    band name the reports do not split by."""
    assert AGE_BANDS == ("now", "recent", "task", "earlier")
    assert check_gold(row(age_s=age_s, age_band=band)) == []


@pytest.mark.parametrize("bad", [{"age_s": -1.0}, {"age_band": "soon"}, {"age_s": "3"}])
def test_a_bad_age_is_a_problem(bad):
    """Contract: a negative or non-numeric age, or an unknown band, is refused. Regression caught: a compile that
    writes ages the reports cannot band."""
    assert check_gold(row(**bad))


@pytest.mark.parametrize("rule", [
    {"weight": 2, "rule": "over", "margin_m": 0.15},
    {"weight": 3, "rule": "over", "margin_m": 0.15, "threshold_m": 0.49},
    {"weight": 3, "rule": "radial", "margin_m": 0.45},
    {"weight": 3, "rule": "miss", "margin_m": 0.3, "critical": [[1.0, 2.0], [3.5, 0.25]]},
    {"weight": 3, "rule": "miss", "margin_m": 0.3, "critical": []},
])
def test_a_catastrophe_rule_beside_the_credit_keys_is_valid(rule):
    """Contract: `catastrophe` is optional beside any credit's exact keys, in the step-0 shape. Regression caught: the
    credit's exact-keys check refusing the extra key, which would make every harm row invalid."""
    assert check_gold(with_catastrophe(rule)) == []


@pytest.mark.parametrize("rule", [
    {"weight": 1, "rule": "over", "margin_m": 0.15},                       # a weight that is neither 2 nor 3
    {"weight": 2, "rule": "under", "margin_m": 0.15},                      # an unknown rule
    {"weight": 2, "rule": "over", "margin_m": 0},                          # no margin
    {"weight": 3, "rule": "radial", "margin_m": 0.4, "threshold_m": 1.0},  # a threshold on a rule that reads none
    {"weight": 3, "rule": "miss", "margin_m": 0.3},                        # a miss rule with no critical points
    {"weight": 3, "rule": "over", "margin_m": 0.3, "critical": [[1, 2]]},  # critical points on an over rule
    {"weight": 3, "rule": "over", "margin_m": 0.3, "why": "x"},            # a key outside the shape
])
def test_a_malformed_catastrophe_rule_is_a_problem(rule):
    """Contract: the rule's weight, name, margin and own key are checked. Regression caught: a row whose catastrophe
    the reported score would read wrongly or not at all."""
    assert check_gold(with_catastrophe(rule))


@pytest.mark.parametrize("task_k0, k, want", [(0, 0, (0, 1)), (0, 3, (0, 4)), (0, 7, (0, 8)), (0, 20, (13, 21)),
                                              (100, 104, (100, 105)), (100, 130, (123, 131))])
def test_the_recent_view_is_the_last_window_of_frames_of_the_task(task_k0, k, want):
    """Contract: `recent(task_k0, k)` is the last `LOOK_WINDOW` (8) frames up to and including k, never before the
    task's first frame. Hand table. Regression caught: a recent view that reaches into an earlier task across its gap,
    or one that leaves out frame k."""
    assert LOOK_WINDOW == 8 and recent(task_k0, k) == range(*want)


def test_a_frame_before_its_task_is_refused():
    """Contract: k before the task's first frame is a caller's error. Regression caught: a silent empty range."""
    with pytest.raises(ValueError, match="before its task's first frame"):
        recent(10, 9)
