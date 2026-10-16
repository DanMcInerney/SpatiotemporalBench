"""The one need rule for every distance (FD-0.11 §3.1): N' = max(N, KAPPA g, 2 rho), the first two halved on the harmful
side, `g` the distance the answer is judged over. Hand values throughout (s = (d + t + a + f) / 4, the score 2s - 1)."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, num, off, pos, rec
from stbench.credit_kernel import graded
from stbench.score import credit, penalised

HELPER_RHO = 0.1365   # one pixel of the helper's camera at 5 m: 0.0273 of the range (FD-0.11 evidence §1.1)


def number_row(value, *, rho=0.02, need=0.15, harm="over", threshold=0.49):
    """A corridor gold as the compiler writes it: the helper's C = 0.15 m, the body 0.34 m long so T = 0.49 m."""
    return rec(num(value), credit="graded", family="clear_path",
               tolerances={"resolution_m": rho, "need_m": need, "harm": harm, "threshold_m": threshold})


def offset_row(ahead, left, *, rho, need, harm="radial"):
    return rec(off(ahead, left), credit="graded", family="ego_now",
               tolerances={"resolution_m": rho, "need_m": need, "harm": harm})


def test_the_kernel_widens_the_need_to_a_tenth_of_g_before_it_halves_it():
    """Contract (FD-0.11 §3.1): `graded(..., g)` takes the need max(N, a tenth of g) and halves that on the harmful
    side, and `g` of 0 (a place, a list item) leaves N alone. By hand, N = 0.15, rho 0.02, error 0.5, direction right:
    g = 5 gives N' = 0.5 (t = 1, a = (1.0 - 0.5) / 0.5 = 1, f = 0): 3/4; the same harmful gives N' = 0.25 (t = 1,
    a = (0.5 - 0.5) / 0.25 = 0, f = 0): 1/2; g = 1.5 is a tenth of exactly N, so it changes nothing, and neither does
    g = 0: at error 0.15 both give t = a = 1, f = 0, 3/4. Regression caught: a far error scored on N alone, the
    distance term left out of the halving, or a term that bites inside ten needs."""
    assert graded(0.5, 0.02, 0.15, True, False, g=5.0) == pytest.approx(0.75)
    assert graded(0.5, 0.02, 0.15, True, True, g=5.0) == pytest.approx(0.5)
    assert graded(0.15, 0.02, 0.15, True, False, g=1.5) == pytest.approx(0.75)
    assert graded(0.15, 0.02, 0.15, True, False) == pytest.approx(0.75)
    assert graded(0.5, 0.02, 0.15, True, False) < graded(0.5, 0.02, 0.15, True, False, g=5.0)


def test_the_need_grows_to_a_tenth_of_the_gold_distance_beyond_ten_clearances():
    """Contract (FD-0.11 §3.1, §4.5): the helper's door 5 m away (C = 0.15, one pixel 0.1365 m there, so the tenth, 0.5,
    is the need). By hand: answered 4.0 (error 1.0, N' = 0.5): t = (2.0 - 1.0) / 1.0 = 1, a = (1.0 - 1.0) / 0.5 = 0, f
    = 0 (the error is past N'), s = 1/2 and the score 0.00, as good as `unknown` (before: -0.41); answered 4.5 (error
    0.5): t = a = 1, f = 0, s = 3/4, score +0.50; answered 6.0 (harmful: N' = 0.25 is under two pixels, 0.273, so
    N' = 0.273): t = (1.092 - 1.0) / 0.546 = 0.1685, a = f = 0, s = 0.2921, score -0.416. Within ten clearances the
    need is C: gold 1.5 (a tenth is C) answered 1.2 is error 0.3 = 2C, s = 1/2, score 0; gold 1.6 (a tenth 0.16) the
    same error has a = (0.32 - 0.3) / 0.16 = 0.125, s = 0.53125, score +0.0625. Regression caught: a far error scored on
    C alone (the three ground robots then disagree with each other by their cameras), or a rule that bites near."""
    door = number_row(5.0, rho=HELPER_RHO)
    assert penalised(num(4.0), door) == pytest.approx(0.0, abs=1e-9)
    assert penalised(num(4.5), door) == pytest.approx(0.5)
    assert penalised(num(6.0), door) == pytest.approx(-0.4158, abs=1e-4)
    assert penalised(num(1.2), number_row(1.5)) == pytest.approx(0.0, abs=1e-9)
    assert penalised(num(1.3), number_row(1.6)) == pytest.approx(0.0625)


def test_the_harmful_side_halves_the_distance_term_too():
    """Contract (FD-0.11 §3.1, §4.5): an answer too long by a given error scores below one too short by it, at every
    distance, because the halving applies to the tenth as to C. By hand, gold 5.0, rho 0.02, C = 0.15, error 0.5: 5.5
    (harmful, N' = 0.25): t = 1, a = (0.5 - 0.5) / 0.25 = 0, f = 0, s = 1/2, score 0; 4.5 (cautious, N' = 0.5): t = a =
    1, f = 0, s = 3/4, score +0.5; error 1.0: 6.0 has t = 0 (4N' = 1.0), s = 1/4, score -0.5, and 4.0 has t = 1, a = 0,
    s = 1/2, score 0. Regression caught: an over-estimate scored like an under-estimate (6.0 no worse than 4.0), the
    case where the need is the camera's pixel having hidden it."""
    row = number_row(5.0)
    assert (penalised(num(5.5), row), penalised(num(4.5), row)) == pytest.approx((0.0, 0.5))
    assert (penalised(num(6.0), row), penalised(num(4.0), row)) == pytest.approx((-0.5, 0.0))
    assert penalised(num(6.0), row) < penalised(num(4.0), row)


def test_an_offsets_distance_is_its_range_from_the_camera_and_not_its_ahead_component():
    """Contract (FD-0.11 §3.1): an offset is judged over its ground range, hypot(ahead, left). By hand, the AMR (C =
    0.45, one pixel 0.108 m at 5 m): the thing at (3, 4), 5 m away, answered (2.4, 3.2), 1.0 m nearer along the same
    bearing: N' = max(0.45, 0.5) = 0.5, t = (2.0 - 1.0) / 1.0 = 1, a = (1.0 - 1.0) / 0.5 = 0, f = 0, direction right,
    s = 1/2, score 0.00; answered (6.0, 4.8), 1.0 m farther (harmful, N' = 0.25 over two pixels, 0.216): t = (1.0 - 1.0)
    / 0.5 = 0, s = 1/4, score -0.5. Judged over the 3 m ahead alone the tenth would be 0.3, under C, and the first would
    score -0.056. Regression caught: the range read as the ahead component, or the tenth taken of the error."""
    row = offset_row(3.0, 4.0, rho=0.108, need=0.45)
    assert penalised(off(2.4, 3.2), row) == pytest.approx(0.0, abs=1e-9)
    assert penalised(off(6.0, 4.8), row) == pytest.approx(-0.5)
    straight = offset_row(5.0, 0.0, rho=0.108, need=0.45)
    assert penalised(off(4.0, 0.0), straight) == pytest.approx(0.0, abs=1e-9)


def test_a_place_and_a_list_item_never_get_the_distance_term():
    """Contract (FD-0.11 §3.1): a place keeps its re-find radius R and a list item its match radius M however far away
    the thing is: the gold row's `range_m` (the slant range a ladder was built from) is not a distance to widen the
    need by. By hand, a place R = 1, rho 0.02, with `range_m` 50 on the row (a tenth would be 5 m), answered 1.5 m off:
    no direction (past R), a = (2 - 1.5) / 1 = 0.5, t = 1, f = 0: 3/8 as for any place; a list of one thing at (40, 0),
    match radius 0.5, answered 0.2 m off: paired, t = a = 1, f = 1 - ln 10 / ln 25 = 0.2847, s = 0.8212. Regression
    caught: R or M widened by range, which would score a far place nearly whole."""
    far = rec(pos(6, 6), credit="graded", range_m=50.0, tolerances={"resolution_m": 0.02, "need_m": 1.0, "harm": None})
    assert credit(pos(7.5, 6.0), far) == pytest.approx(0.375)
    listed = rec({"type": "positions", "items": [{"xy": [40.0, 0.0]}]}, credit="graded_set", family="task_things",
                 tolerances={"match_m": 0.5, "resolutions_m": [0.02], "fp_weight": 0.5})
    assert credit({"type": "positions", "items": [{"xy": [40.2, 0.0]}]}, listed) == pytest.approx(0.8212, abs=1e-4)
    assert credit(UNKNOWN, far) == 0.0
