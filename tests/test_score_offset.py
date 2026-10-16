"""The credit of an `offset` answer (`score.py`): the ladder on the 2-D error in the camera's frame."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, off, pos, rec
from stbench.rules import relative_ladder
from stbench.score import credit, outcome, penalised

LADDER = {"ladder": [0.1, 0.25, 0.5]}
GOLD = rec(off(4.0, 1.0), credit="ladder", tolerances=LADDER, metric=None)


@pytest.mark.parametrize("answer, expected", [
    (off(4.0, 1.0), 1.0),            # exact
    (off(4.06, 1.08), 1.0),          # error hypot(0.06, 0.08) = 0.10, on the first rung (met up to 1e-9)
    (off(4.2, 1.15), 2 / 3),         # hypot(0.20, 0.15) = 0.25: the second and third rungs
    (off(4.2, 1.2), 1 / 3),          # hypot(0.2, 0.2) = 0.283: the third rung alone
    (off(4.0, 1.6), 0.0),            # 0.6 m: none
    (off(1.0, 4.0), 0.0),            # the axes swapped: hypot(3, 3)
    (off(4.0, -1.0), 0.0),           # the sign of left flipped: 2 m
])
def test_an_offset_is_credited_by_the_two_dimensional_error(answer, expected):
    """Contract: the credit of an `offset` against an `offset` gold is the share of the ladder's rungs met by
    `hypot(d_ahead, d_left)`, a rung met at its value up to 1e-9. By hand at gold (4, 1) and rungs 0.10, 0.25, 0.50 m.
    The third row is the discriminating one: 0.2 m off on each axis is 0.283 m off, which meets only the 0.5 rung (1/3),
    where an error read on one axis would meet 0.25 and 0.5 (2/3). Regression caught: one axis scored (the other free),
    an error summed instead of combined, the axes swapped, or a sign flipped."""
    assert credit(answer, GOLD) == pytest.approx(expected)


def test_a_position_for_an_offset_gold_and_an_offset_for_a_position_gold_score_nothing():
    """Contract: an answer of another type than the gold's has credit 0, whatever its numbers (the parse of the question
    already refuses it). Regression caught: a map position read as a camera-frame offset (or the reverse) by the same
    two numbers."""
    assert credit(pos(4.0, 1.0), GOLD) == 0.0
    assert credit(off(4.0, 1.0), rec(pos(4.0, 1.0), credit="ladder", tolerances=LADDER, metric="xy")) == 0.0


def test_an_unknown_offset_gold_and_the_penalised_score_follow_the_other_golds():
    """Contract: `unknown` on an offset gold of slice `unknown` is correct and an answer there costs 1; on an
    answerable offset gold `unknown` is 0 and a wrong answer costs 1 (t = 0.5), a half-credit one 0.5 - 0.5 = 0 by
    `s - (1 - s)k`. Regression caught: the new type leaving the penalty rules (the coin that decides `unknown`)."""
    unknown_gold = rec(UNKNOWN, slice="unknown", credit="ladder", tolerances=LADDER, metric=None)
    assert penalised(UNKNOWN, unknown_gold) == 1.0 and penalised(off(1.0, 1.0), unknown_gold) == -1.0
    assert penalised(UNKNOWN, GOLD) == 0.0 and penalised(off(9.0, 9.0), GOLD) == -1.0
    assert penalised(off(4.2, 1.15), GOLD) == pytest.approx(2 / 3 - 1 / 3)
    assert outcome(off(4.2, 1.15), GOLD) == "partial" and outcome(off(9.0, 9.0), GOLD) == "wrong"


def test_the_rungs_scale_with_the_slant_range_of_the_row():
    """Contract: a row built from `relative_ladder(r)` is judged against rungs that grow with the range. By hand: a
    0.3 m error at 1 m (rungs 0.10, 0.25, 0.50) and at 5 m (rungs 0.5, 1.25, 2.5) scores 1/3 and 1; at 20 m (rungs 2, 5,
    10) a 1.5 m error is full credit while at 0.3 m (rungs 0.05, 0.10, 0.20) it is nothing. Regression caught: absolute
    rungs for a far thing (every drone answer wrong) or relative rungs ignoring the floor for a near one."""
    for r, error, expected in ((1.0, 0.3, 1 / 3), (5.0, 0.3, 1.0), (20.0, 1.5, 1.0), (0.3, 1.5, 0.0)):
        row = rec(off(2.0, 0.0), credit="ladder", tolerances={"ladder": relative_ladder(r)}, metric=None)
        assert credit(off(2.0 + error, 0.0), row) == pytest.approx(expected), (r, error)
