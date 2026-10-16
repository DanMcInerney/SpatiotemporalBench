"""The credit kernel (`credit_kernel`, 0.9.0, docs/methodology.md rule 7): four quarters, every number worked by hand.

Throughout: the helper's clearance C = 0.15 m with a one-pixel resolution rho = 0.02 m (so N' = C, since C >= 2 rho)."""

from __future__ import annotations

import math

import pytest

from stbench.credit_kernel import ANGLE_DEG, FP_WEIGHT, HARM_FACTOR, KAPPA, MATCH_REACH, MIN_RHO_M, graded, tiers

C, RHO = 0.15, 0.02


def test_the_constants_are_the_designs():
    """Contract (brief §3, §9; FD-0.11 §3.1): the least resolution is 1 cm, the harmful side keeps half the need, a
    false list item costs half a missed one, an offset's direction is 45 degrees, a list item matches within four match
    radii and the far field's slack is a tenth of the distance. Regression caught: a kernel constant changed without the
    worked values of the brief changing with it."""
    assert (MIN_RHO_M, HARM_FACTOR, FP_WEIGHT, ANGLE_DEG, MATCH_REACH, KAPPA) == (0.01, 0.5, 0.5, 45, 4, 0.1)


@pytest.mark.parametrize("e", [0.0, 0.01, RHO])
def test_an_answer_within_the_resolution_scores_one(e):
    """Contract: within one pixel (rho) the three graded quarters are 1, and with the direction right the answer scores
    1. Regression caught: a full credit that takes less than the camera's own precision, so no reader could reach it."""
    assert tiers(e, RHO, C) == (1.0, 1.0, 1.0)
    assert graded(e, RHO, C, True, False) == 1.0


@pytest.mark.parametrize(("multiple", "with_direction", "without"), [(1, 3 / 4, 2 / 4), (2, 2 / 4, 1 / 4),
                                                                      (4, 1 / 4, 0.0), (6, 1 / 4, 0.0)])
def test_the_anchors_at_the_need_twice_and_four_times_it(multiple, with_direction, without):
    """Contract (brief §3.1): at the need N the answer scores 3/4 (the thing and the amount held, the full quarter
    spent), at 2N 1/2, at 4N 1/4 with the direction right, and beyond 4N the direction alone is worth that quarter.
    Hand values for C = 0.15, rho = 0.02: e = 0.15 gives t = 1, a = 1, f = 0; e = 0.30 gives t = 1, a = 0; e = 0.60
    gives t = 0. Regression caught: an anchor moved, or the direction quarter made part of the curve."""
    e = multiple * C
    assert graded(e, RHO, C, True, False) == pytest.approx(with_direction)
    assert graded(e, RHO, C, False, False) == pytest.approx(without)


def test_the_credit_falls_strictly_between_the_resolution_and_four_needs_and_has_no_step():
    """Contract (brief §3.1): between rho and 4N' the credit falls strictly with the error, and it is continuous at
    every anchor (rho, N', 2N', 4N'): its one step is the direction quarter. Regression caught: a flat top wider than
    one pixel (a plateau, an error of 2 cm scoring like 1 cm), or a step inside the curve (a rung)."""
    errors = [RHO + (4 * C - RHO) * i / 400 for i in range(401)]
    scores = [graded(e, RHO, C, True, False) for e in errors]
    assert all(a > b for a, b in zip(scores, scores[1:], strict=False))
    for anchor in (RHO, C, 2 * C, 4 * C):
        below, above = graded(anchor - 1e-9, RHO, C, True, False), graded(anchor + 1e-9, RHO, C, True, False)
        assert below == pytest.approx(above, abs=1e-6), anchor


def test_the_appendix_b_worked_examples():
    """Contract (brief Appendix B): the helper's reverse corridor, gold 0.80 m, rho 0.02, C 0.15. By hand, with
    f = 1 - ln(e / rho) / ln(N' / rho):
    - an answer 0.85 (harmful, N' = 0.075, e = 0.05): t = a = 1, f = 1 - ln 2.5 / ln 3.75 = 0.3068, s = 0.8267;
    - an answer 0.75 (cautious, N' = 0.15): f = 1 - ln 2.5 / ln 7.5 = 0.5453, s = 0.8863;
    - an answer 1.00 (harmful, e = 0.20): t = (0.30 - 0.20) / 0.15 = 2/3, a = f = 0, s = (1 + 2/3) / 4 = 5/12;
    - an answer 0.60 (cautious, e = 0.20): t = 1, a = (0.30 - 0.20) / 0.15 = 2/3, f = 0, s = (1 + 1 + 2/3) / 4 = 2/3;
    - an answer 3.00 (harmful, e = 2.2), direction wrong: 0.
    Regression caught: the kernel disagreeing with the design's own table."""
    assert graded(0.05, RHO, C, True, True) == pytest.approx(0.8267, abs=1e-4)
    assert graded(0.05, RHO, C, True, False) == pytest.approx(0.8863, abs=1e-4)
    assert graded(0.20, RHO, C, True, True) == pytest.approx(5 / 12)
    assert graded(0.20, RHO, C, True, False) == pytest.approx(2 / 3)
    assert graded(2.2, RHO, C, False, True) == 0.0


def test_the_harmful_side_halves_the_need_and_not_the_resolution():
    """Contract (brief §3.3): on the harmful side N becomes N / 2 in every anchor and rho stays. Hand case, e = 0.10:
    cautious (N' = 0.15) gives t = a = 1 and f = 1 - ln 5 / ln 7.5 = 0.2012, s = 0.8003; harmful (N' = 0.075) gives
    t = 1, a = (0.15 - 0.10) / 0.075 = 2/3 and f = 0, s = 2/3; and an answer within rho scores 1 on either side.
    Regression caught: a harm rule that is a no-op, one that halves rho too, or one applied to the wrong side."""
    assert graded(0.10, RHO, C, True, False) == pytest.approx(0.8003, abs=1e-4)
    assert graded(0.10, RHO, C, True, True) == pytest.approx(2 / 3)
    assert graded(RHO, RHO, C, True, True) == 1.0


def test_a_resolution_of_half_the_need_widens_the_need_to_twice_the_resolution():
    """Contract (brief §3.1: N' = max(N, 2 rho)): when one pixel is half the need or more, the need widens to 2 rho, so
    the full quarter is still a curve and not a step. Hand case: N = 0.04, rho = 0.03, N' = 0.06; e = 0.05 is inside N'
    (t = a = 1) with f = 1 - ln(0.05 / 0.03) / ln(2) = 0.2630, where N' = N = 0.04 would give a = 0.75. Regression
    caught: a need narrower than two pixels, which no reader of the frame could meet."""
    t, a, f = tiers(0.05, 0.03, 0.04)
    assert (t, a) == (1.0, 1.0) and f == pytest.approx(0.2630, abs=1e-4)
    assert tiers(0.06, 0.03, 0.04)[1] == pytest.approx(1.0) and tiers(0.06, 0.03, 0.04)[2] == pytest.approx(0.0)


def test_a_resolution_under_a_centimetre_is_read_as_a_centimetre():
    """Contract: a gold row's rho is at least `MIN_RHO_M` (an answer's natural precision and the pose's rounding), so an
    error of 1 cm against a stated rho of 0 or 1 mm still scores 1, and an error of 2 cm does not. Regression caught: a
    resolution below the pose's own rounding making the full credit unreachable."""
    assert tiers(0.01, 0.0, C) == (1.0, 1.0, 1.0) and tiers(0.01, 0.001, C) == (1.0, 1.0, 1.0)
    assert tiers(0.02, 0.0, C)[2] < 1.0
    assert tiers(0.02, 0.0, C)[2] == pytest.approx(1 - math.log(2) / math.log(15), abs=1e-9)
