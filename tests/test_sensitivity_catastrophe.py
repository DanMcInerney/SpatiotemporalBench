"""The credit sensitivity scales a harm row's catastrophe rule with its clearance (`eval_sensitivity.scaled_need`,
0.10.0): the rule's margin is max(C, 2 rho) at compile time, so scaling C re-makes it, and its threshold moves with the
row's. Hand rows."""

from __future__ import annotations

import pytest

from stbench import eval_sensitivity, rules


def row(rho: float, need: float, rule: dict, *, threshold: float | None = 1.45) -> dict:
    tolerances = rules.graded_tolerances(rho, need, "over", threshold=threshold)
    return {"qid": "q", "family": "closest_person", "credit": "graded", "gold": {"type": "number", "value": 1.0},
            "tolerances": {**tolerances, "catastrophe": rule}}


@pytest.mark.parametrize(("factor", "need", "threshold", "margin"), [
    (2.0, 0.9, 1.9, 0.9),        # C 0.45 doubled: the threshold (body 1.0 + C) 1.45 -> 1.9, margin max(0.9, 0.1)
    (0.5, 0.225, 1.225, 0.225),  # halved: 1.45 - 0.225, margin max(0.225, 0.1)
    (0.1, 0.045, 1.045, 0.1),    # a tenth: 2 rho = 0.1 is now the larger, so the margin stops at the pixel
])
def test_scaling_the_clearance_remakes_the_margin_and_moves_the_threshold(factor, need, threshold, margin):
    """Contract (FD-0.7.0 §2.3, the C sensitivity, with the 0.10.0 rule): the C sensitivity scales a graded row's
    `need_m` and moves its `threshold_m` by the change in C; the row's catastrophe margin is `max(C, 2 rho)`, so it is
    made so again from the scaled C (the resolution is kept), and the rule's own `threshold_m` moves with the row's.
    Hand rows: C = 0.45 m, one pixel 0.05 m, threshold 1.45 m. Regression caught: a sensitivity that scales the credit
    but leaves a catastrophe at its old margin, so that the headline's C-sensitivity moves less than the scorer's."""
    gold = row(0.05, 0.45, {"weight": 3, "rule": "over", "margin_m": 0.45, "threshold_m": 1.45})
    scaled = eval_sensitivity.scaled_need(gold, factor)["tolerances"]
    assert scaled["need_m"] == pytest.approx(need) and scaled["threshold_m"] == pytest.approx(threshold)
    assert scaled["catastrophe"]["margin_m"] == pytest.approx(margin)
    assert scaled["catastrophe"]["threshold_m"] == pytest.approx(threshold)
    assert scaled["catastrophe"]["weight"] == 3 and scaled["resolution_m"] == 0.05


def test_a_miss_rule_and_a_row_without_one_are_scaled_as_they_were():
    """Contract: a `miss` rule (a list's critical points and the row's matching radius) has no margin of C, so scaling C
    leaves it whole; a row with no rule gains none. Regression caught: a list family's catastrophe rewritten by the C
    sensitivity, or a rule invented for a row that had none."""
    rule = {"weight": 3, "rule": "miss", "margin_m": 1.0, "critical": [[1.0, 1.0]]}
    scaled = eval_sensitivity.scaled_need(row(0.05, 0.45, rule), 2.0)["tolerances"]
    assert scaled["catastrophe"] == rule and scaled["need_m"] == pytest.approx(0.9)
    bare = {**row(0.05, 0.45, rule), "tolerances": rules.graded_tolerances(0.05, 0.45, "over", threshold=1.45)}
    assert "catastrophe" not in eval_sensitivity.scaled_need(bare, 2.0)["tolerances"]
