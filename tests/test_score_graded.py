"""The `graded` credit (FD-decision-questions §3) as the scorer applies it: what the scorer adds to the
kernel's four quarters (`test_credit_kernel.py` owns the quarters: the anchors at N, 2N and 4N, the harm halving, the
2-rho widening, the brief's worked corridor). Here: the direction quarter and the harmful side, decided by the gold's
type, an offset's bearing, a number's side of the threshold and a place's `e <= R`. Every expected value is worked by
hand from the kernel's formulas, never by the code under test: with a resolution rho, a need N, N' = max(N, 2 rho) and
an error e, s = (direction + t + a + f) / 4, t = 1 to 2N' then 0 at 4N', a = 1 to N' then 0 at 2N', f = 1 to rho then
1 - ln(e / rho) / ln(N' / rho) to 0 at N'; on the harmful side N is halved. The golds here lie within ten needs of the
camera, where N' is the job's own need (a tenth of the distance is no more than C: `test_credit_far_field.py` owns
what changes beyond)."""

from __future__ import annotations

import math

import pytest

from score_kit import UNKNOWN, num, off, pos, rec
from stbench.score import credit, outcome, penalised

ROOM = [[0, 0], [12, 0], [12, 12], [0, 12]]


def number_row(value, *, rho=0.02, need=0.15, harm="over", threshold=0.49):
    """A number gold as the compiler writes a corridor: the helper's C = 0.15 m, the body 0.34 m long so T = 0.49 m."""
    return rec(num(value), credit="graded", family="clear_path",
               tolerances={"resolution_m": rho, "need_m": need, "harm": harm, "threshold_m": threshold})


def offset_row(ahead, left, *, rho=0.02, need=0.10, harm="radial"):
    return rec(off(ahead, left), credit="graded", family="ego_now",
               tolerances={"resolution_m": rho, "need_m": need, "harm": harm})


def place_row(x, y, *, rho=0.02, need=1.0, **extra):
    return rec(pos(x, y), credit="graded", family="object_where",
               tolerances={"resolution_m": rho, "need_m": need, "harm": None, **extra})


def test_a_number_is_harmful_only_when_answered_longer_than_the_gold():
    """Contract (§3.3): the scorer decides the harmful side of a number: the answer larger than the gold's, when the row
    says `over`; the kernel then halves the need (its own test). By hand, gold 1.0, rho 0.02, N = 0.15, error 0.15:
    - answered 1.15 (longer, N' = 0.075): t = (0.30 - 0.15) / 0.15 = 1, a = (0.15 - 0.15) / 0.075 = 0, f = 0, so
      (1 + 1) / 4 = 0.5;
    - answered 0.85 (shorter, N' = 0.15): t = 1, a = 1, f = 0, so (1 + 1 + 1) / 4 = 0.75;
    - answered 1.15 on a row with no harmful side: 0.75.
    Regression caught: the sign of the harm rule reversed (the shorter answer punished), or a row's `harm` ignored."""
    assert credit(num(1.15), number_row(1.0)) == pytest.approx(0.5)
    assert credit(num(0.85), number_row(1.0)) == pytest.approx(0.75)
    assert credit(num(1.15), number_row(1.0, harm=None)) == pytest.approx(0.75)


def test_a_number_s_direction_is_the_side_of_the_threshold_and_a_factor_of_two():
    """Contract (§3.2): a number's direction quarter is earned when the answer and the gold lie on the same side of
    T = L + N and, where the gold is at least T, the answer is within a factor of two of it. By hand, need 0.05 and
    T = 0.39, gold 0.50 (rho 0.02), so every tier is zero at an error of 0.30 and the score is the direction alone:
    - 0.80 (error 0.30, harmful, in [0.25, 1.0]): 1/4;
    - 0.20 (error 0.30, under T and under g/2): 0;
    - gold 0.45, answer 0.35 (error 0.10, across T): t = 1, a = 0 and no direction, (1) / 4 = 0.25, where
      the answer 0.50 (same side, error 0.05, harmful so N = 0.025) is (1 + 1 + 0 + 0) / 4 = 0.5;
    - a gold under T (0.30, rho 0.01), answered 0.34: the same side is right, error 0.04 harmful so N = 0.025 and
      t = 1, a = (0.05 - 0.04) / 0.025 = 0.4: (1 + 1 + 0.4) / 4 = 0.6; answered 0.45: across T, error 0.15 past 4N: 0;
    - the corridor of gold 0.80 (T = 0.49, N = 0.15), answered 1.60, exactly twice: the error 0.80 is past 4N' so the
      score is the direction alone, 1/4; answered 3.00, past a factor of two: 0.
    Regression caught: a direction that ignores T, one that demands a factor of two below T, a factor of two that
    stops short of twice the gold, or one decided by the error alone."""
    row = number_row(0.50, need=0.05, threshold=0.39)
    assert credit(num(0.80), row) == pytest.approx(0.25)
    assert credit(num(0.20), row) == pytest.approx(0.0)
    across = number_row(0.45, need=0.05, threshold=0.39, rho=0.01)
    assert credit(num(0.35), across) == pytest.approx(0.25)
    assert credit(num(0.50), across) == pytest.approx(0.5)
    below = number_row(0.30, need=0.05, threshold=0.39, rho=0.01)
    assert credit(num(0.34), below) == pytest.approx(0.6)
    assert credit(num(0.45), below) == pytest.approx(0.0)
    corridor = number_row(0.80)   # T = 0.49: answered exactly twice (1.60) the direction stands, past it (3.00) it goes
    assert credit(num(1.60), corridor) == pytest.approx(0.25)   # the error is past 4N' (0.30), so only the direction
    assert credit(num(3.00), corridor) == pytest.approx(0.0)


def test_an_offset_s_direction_is_a_bearing_within_45_degrees_or_the_need():
    """Contract (§3.2): an offset's direction quarter is earned when the answered bearing is within 45 degrees of the
    gold's or the 2-D error is within N. By hand, gold 2.0 m dead ahead, N = 0.10: an answer 2.0 m out at 40 degrees
    (2 cos 40, 2 sin 40), error 2 * 2 * sin 20 = 1.368 m, beyond 4N so only the direction is left, scores 1/4; the
    same at 50 degrees scores 0. Regression caught: the bearing read in radians, or a direction that needs the error
    small."""
    row = offset_row(2.0, 0.0)
    at = lambda deg: off(2 * math.cos(math.radians(deg)), 2 * math.sin(math.radians(deg)))   # noqa: E731
    assert credit(at(40), row) == pytest.approx(0.25)
    assert credit(at(50), row) == pytest.approx(0.0)
    assert credit(at(-40), row) == pytest.approx(0.25)


def test_a_zero_offset_has_no_bearing_so_it_earns_the_direction_only_within_the_need():
    """Contract (0.9.0 review N3): an answer of (0, 0), or a gold on the camera, has no bearing, so its direction
    quarter is earned only when the 2-D error is within N, never for free. By hand, N = 0.10, rho 0.02:
    - gold (2, 0) answered (0, 0): error 2.0, beyond 4N, and no bearing, so 0 (not 1/4);
    - gold (0, 0) answered (2, 0): the same, 0;
    - gold (0.05, 0) answered (0, 0): error 0.05 is within N, so the direction is right; the answer is nearer than the
      gold, not harmful: t = 1, a = 1, f = 1 - ln 2.5 / ln 5 = 0.4307, so (3 + 0.4307) / 4 = 0.8577;
    - gold (0, 0) answered (0.08, 0): error 0.08 within N, the gold has no bearing to be harmful along: t = 1, a = 1,
      f = 1 - ln 4 / ln 5 = 0.1386, so 0.7847.
    Regression caught: a zero vector that is taken to point the right way, which would pay an answer of nothing the
    direction quarter against any gold."""
    assert credit(off(0.0, 0.0), offset_row(2.0, 0.0)) == pytest.approx(0.0)
    assert credit(off(2.0, 0.0), offset_row(0.0, 0.0)) == pytest.approx(0.0)
    assert credit(off(0.0, 0.0), offset_row(0.05, 0.0)) == pytest.approx(0.8577, abs=1e-4)
    assert credit(off(0.08, 0.0), offset_row(0.0, 0.0)) == pytest.approx(0.7847, abs=1e-4)


def test_an_offset_is_harmful_only_when_it_places_the_point_farther_along_the_golds_bearing():
    """Contract (§3.3): an offset's harmful side is a positive error along the gold's bearing (a thing placed farther
    than it is); the need is halved there. By hand, gold (1, 0), N = 0.10, rho 0.02, error 0.12:
    - (1.12, 0): farther, harmful, N = 0.05: t = (0.2 - 0.12) / 0.1 = 0.8 (2N is 0.10), a = 0: (1 + 0.8) / 4 = 0.45;
    - (0.88, 0): nearer, N = 0.10: t = 1, a = (0.2 - 0.12) / 0.1 = 0.8: (1 + 1 + 0.8) / 4 = 0.70;
    - (1.0, 0.12): sideways, no radial error, cautious: 0.70.
    Regression caught: the sign of the harm rule reversed (the thing placed nearer punished), or a sideways error
    counted as harmful."""
    row = offset_row(1.0, 0.0)
    assert credit(off(1.12, 0.0), row) == pytest.approx(0.45)
    assert credit(off(0.88, 0.0), row) == pytest.approx(0.70)
    assert credit(off(1.0, 0.12), row) == pytest.approx(0.70, abs=1e-3)
    assert credit(off(1.12, 0.0), offset_row(1.0, 0.0, harm=None)) == pytest.approx(0.70)   # a row with no harm side


def test_a_place_s_direction_is_within_the_need_of_the_gold_and_it_has_no_harmful_side():
    """Contract (§3.2, §3.3; 0.9.0 review B2): a place is judged by the floor distance, its direction quarter is earned
    within N = R of the gold and nowhere else (not for being in the gold's room), and it has no harmful side. By hand,
    gold (6, 6) in a 12 m room, R = 1, rho 0.02, N' = 1:
    - 0.9 m off: direction, t = 1, a = 1, f = 1 - ln 45 / ln 50 = 0.0269, (1 + 1 + 1 + 0.0269) / 4 = 0.7567;
    - exactly R off: direction, t = a = 1, f = 0, 0.75; 1.01 m off: no direction, t = 1, a = 0.99, f = 0, (1 + 0.99) /
      4 = 0.4975, the step of the direction quarter alone;
    - 1.5 m off, either way (never harmful): no direction, a = (2 - 1.5) / 1 = 0.5, f = 0: (1 + 0.5) / 4 = 0.375;
    - 5.9 m off, inside the room: past 4N', 0; the height of the answer is not in it.
    Regression caught: a place scored by the room, the harm rule applied to places, or the height read as part of it."""
    assert credit(pos(6.9, 6.0), place_row(6, 6)) == pytest.approx(0.7567, abs=1e-4)
    assert credit(pos(7.0, 6.0), place_row(6, 6)) == pytest.approx(0.75)
    assert credit(pos(7.01, 6.0), place_row(6, 6)) == pytest.approx(0.4975)
    for answer in (pos(7.5, 6.0), pos(4.5, 6.0)):
        assert credit(answer, place_row(6, 6)) == pytest.approx(0.375)
    assert credit(pos(11.9, 6.0), place_row(6, 6)) == pytest.approx(0.0)
    assert credit({"type": "position", "xyz": [6.0, 6.0, 1.7]}, place_row(6, 6)) == pytest.approx(1.0)   # the height
    # is not in it


def test_a_place_in_the_golds_room_beyond_r_earns_no_direction_quarter():
    """Contract (0.9.0 review B2): the room a thing is in is a fact of the map and the kind, which a prior without the
    camera knows, so a place beyond R of the gold earns no direction quarter even in the gold's room. By hand, gold
    (6, 6), R = 1, rho 0.02, N' = 1, the answer 1.33 m away, in the same 12 m room: t = 1 (inside 2N'), a = (2 - 1.33) /
    1 = 0.67, f = 0 (past N'), so (0 + 1 + 0.67 + 0) / 4 = 0.4175 and penalised 2 * 0.4175 - 1 = -0.165 (with the
    room's quarter it was 0.6675, +0.335: a prior's lead). The same answer 0.9 m away (within R) has the quarter. A
    row that still carried a room polygon around the answer (an older compile) is scored the same, the polygon unread.
    Regression caught: the room quarter returning, which would pay a sensorless layout prior for naming the room."""
    gold = place_row(6, 6)
    answer = pos(7.33, 6.0)
    assert credit(answer, gold) == pytest.approx(0.4175)
    assert penalised(answer, gold) == pytest.approx(-0.165)
    assert credit(answer, place_row(6, 6, room=ROOM)) == pytest.approx(0.4175)
    assert credit(pos(6.9, 6.0), gold) == pytest.approx(0.7567, abs=1e-4)


def test_a_graded_row_must_hold_every_key_it_reads_and_an_answer_of_another_type_scores_nothing():
    """Contract (§3, records): a `graded` row without `resolution_m`, `need_m`, `harm` (or `threshold_m` for a number)
    raises ValueError naming the key, so no default decides a credit unseen; an answer of another
    type than the gold's, a malformed one and `unknown` against an answerable gold score 0, and `unknown` against an
    unknown gold is 1. Regression caught: a missing tolerance read as 0 or a default, or an `unknown` that earns the
    direction quarter."""
    base = {"resolution_m": 0.02, "need_m": 0.15, "harm": "over", "threshold_m": 0.49}
    for key in base:
        with pytest.raises(ValueError, match=key):
            credit(num(1.0), rec(num(1.0), credit="graded", tolerances={k: v for k, v in base.items() if k != key}))
    with pytest.raises(ValueError, match="need_m"):
        credit(num(1.0), rec(num(1.0), credit="graded", tolerances={**base, "need_m": 0}))
    with pytest.raises(ValueError, match="harm"):
        credit(num(1.0), rec(num(1.0), credit="graded", tolerances={**base, "harm": "radial?"}))
    with pytest.raises(ValueError, match="need_m"):
        credit(pos(1, 1), rec(pos(1, 1), credit="graded", tolerances={"resolution_m": 0.02, "harm": None}))
    row = number_row(1.0)
    assert credit(off(1.0, 0.0), row) == 0.0 and credit({"type": "number", "value": "x"}, row) == 0.0
    assert credit(UNKNOWN, row) == 0.0
    unknown_gold = rec({"type": "unknown"}, slice="unknown", credit="graded", tolerances={})
    assert credit(UNKNOWN, unknown_gold) == 1.0 and credit(num(1.0), unknown_gold) == 0.0


def test_a_graded_answer_s_outcome_and_penalised_score_follow_its_credit():
    """Contract (§3.1, `penalised`, `outcome`): the penalised score stays 2s - 1 and an outcome stays correct only at
    s = 1, wrong only at s = 0. By hand, the corridor of gold 0.80: the answer 0.81 (s = 1) is correct and +1; 0.85
    (s = 0.8267) is partial and +0.6534; 3.00 (s = 0) is wrong and -1; `unknown` is abstained and 0. Regression caught:
    a partial score treated as correct, or a changed penalty."""
    row = number_row(0.80)
    assert (outcome(num(0.81), row), penalised(num(0.81), row)) == ("correct", pytest.approx(1.0))
    assert (outcome(num(0.85), row), penalised(num(0.85), row)) == ("partial", pytest.approx(0.6534, abs=1e-4))
    assert (outcome(num(3.0), row), penalised(num(3.0), row)) == ("wrong", pytest.approx(-1.0))
    assert (outcome(UNKNOWN, row), penalised(UNKNOWN, row)) == ("abstained", 0.0)


def test_a_ladder_row_is_scored_by_its_ladder_and_not_graded():
    """Contract (§0: the ladder credits score as they always did): a gold row of the ladder credit is not graded. By
    hand, an offset on the ladder [0.1, 0.25, 0.5] answered 0.2 off meets two rungs: 2/3; a number on [0.25, 0.5, 1, 2]
    0.6 off meets the last two of four: 1/2. Regression caught: the graded credit reaching a ladder row, so that its
    score moves."""
    assert credit(off(1.2, 0), rec(off(1.0, 0), credit="ladder", tolerances={"ladder": [0.1, 0.25, 0.5]})) == (
        pytest.approx(2 / 3))
    assert credit(num(2.4), rec(num(3.0), credit="ladder", tolerances={"ladder": [0.25, 0.5, 1.0, 2.0]})) == (
        pytest.approx(1 / 2))
