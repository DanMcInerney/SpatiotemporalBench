"""An answer of any finite size is scored by its geometry, never by a floating-point overflow: a bearing, a harmful side
and a distance stay right where the products they are made of leave the range of a float (about 1.8e308)."""

from __future__ import annotations

import pytest

from score_kit import off, pos, rec
from stbench.answers import parse_answer
from stbench.eval_distance import measure
from stbench.score import credit, reported
from stbench.score_graded import bearing_deg, radial_error

HUGE = 1.7e308   # finite, and its square and its product with 5 are not


def offset_row(ahead, left):
    return rec(off(ahead, left), credit="graded", family="ego_now",
               tolerances={"resolution_m": 0.02, "need_m": 0.10, "harm": "radial"})


def test_a_huge_offset_has_the_bearing_it_points_along():
    """Contract: the angle between two offsets is the same for any finite lengths. By hand, gold (5, 5), 45 degrees to
    the left of ahead: (HUGE, -HUGE) is 90 degrees from it, (HUGE, HUGE) 0 and (-HUGE, -HUGE) 180. Their dot product
    overflows to inf - inf = NaN, which `min(1.0, nan)` read as a cosine of 1, 0 degrees: the answer 90 degrees off
    scored -0.50 (the direction quarter) and not -1.00, as (1e6, -1e6) does. Regression caught: a bearing taken from
    products of unscaled lengths."""
    gold = parse_answer(off(5, 5))
    for ahead, left, degrees in ((HUGE, -HUGE, 90.0), (HUGE, HUGE, 0.0), (-HUGE, -HUGE, 180.0), (1e6, -1e6, 90.0)):
        assert bearing_deg(parse_answer(off(ahead, left)), gold) == pytest.approx(degrees)
    row = offset_row(5, 5)
    assert reported(off(HUGE, -HUGE), row) == -1.0
    assert reported(off(1e6, -1e6), row) == -1.0
    assert credit(off(HUGE, HUGE), row) == pytest.approx(0.25)   # the right bearing alone, at any error


def test_a_huge_offset_is_harmful_when_it_points_farther_along_the_golds_bearing():
    """Contract: the error's component along the gold's bearing keeps its sign for any finite answer. By hand, gold
    (5, 5), answer (HUGE, -0.9 HUGE): the error is 0.1 HUGE ahead of the gold's bearing, so positive; its two products
    with the gold's coordinates overflow with opposite signs and summed to NaN, which is not above 0. Regression caught:
    a huge placement farther away not read as harmful."""
    gold = parse_answer(off(5, 5))
    assert radial_error(parse_answer(off(HUGE, -0.9 * HUGE)), gold) > 0
    assert radial_error(parse_answer(off(-HUGE, 0.9 * HUGE)), gold) < 0
    assert radial_error(parse_answer(off(7, 5)), gold) == pytest.approx(2 / 2 ** 0.5)


def test_a_huge_place_scores_zero_where_it_used_to_raise():
    """Contract: a `ladder` place answer of any finite size is far from the gold and earns nothing; the distance is
    `math.dist` and not a sum of squares, whose square of 1.7e308 raises OverflowError. Regression caught: a crash of
    the scorer on one extreme answer."""
    row = rec(pos(1, 1), tolerances={"ladder": [0.5]})
    assert credit(pos(HUGE, -HUGE), row) == 0.0
    row_z = rec(pos(1, 1, 1), metric="xyz", tolerances={"ladder": [0.5]})
    assert credit(pos(HUGE, -HUGE, HUGE), row_z) == 0.0


def test_the_distance_report_reads_the_same_bearing_as_the_scorer():
    """Contract: `eval_distance.measure` owns no bearing of its own; it reports `score_graded.bearing_deg`, so the
    report of a huge answer agrees with its score. Regression caught: a second copy of the cosine formula, with the
    same overflow."""
    got = measure(parse_answer(off(HUGE, -HUGE)), parse_answer(off(5, 5)), 7.0)
    assert got["bearing_deg"] == pytest.approx(90.0)
    assert measure(parse_answer(off(0, 0)), parse_answer(off(5, 5)), 7.0)["bearing_deg"] is None
