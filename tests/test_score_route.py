"""The `route` credit for a heading (`score_graded.route_credit`, FD-0.11 §3.4.2): the way to a remembered thing is
scored by how much nearer one `step_m` of driving along the answered heading brings the robot, against the best
heading's, on the four quarters. The profile on the gold row lists the outcome at each heading, entry i the heading
i * step_deg degrees to the left of the way the robot faces (null: no room to move or space never seen). Every
expected value is worked by hand."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, rec
from stbench.score import credit, outcome, penalised

# a step of 90 degrees: ahead 0.8 m nearer, left 0.3, behind none (no room), right 0.2 m farther away
PROFILE = [0.8, 0.3, None, -0.2]
TOLERANCES = {"profile_m": PROFILE, "step_deg": 90, "step_m": 1.0, "resolution_m": 0.1, "need_m": 0.15}


def heading(deg):
    return {"type": "heading", "deg_left": deg}


def route_row(**over):
    return rec(heading(0), credit="route", family="way_to", tolerances={**TOLERANCES, **over})


def test_the_best_heading_scores_one_and_a_heading_with_no_room_scores_minus_one():
    """Contract (FD-0.11 §3.4.2, §4.5): the best outcome earns 1 (+1 penalised, `correct`); a null entry (no room to
    move, or space never seen) has the error best + `step_m` = 1.8, no direction and the harmful side, all four
    quarters 0, so it scores -1 (`wrong`). By hand, rho 0.1, C = 0.15 (two pixels, 0.2, are the need). Regression
    caught: a heading with no room read as an outcome of 0, which would pay it for being no worse than an average
    heading, or a best heading short of full credit."""
    row = route_row()
    assert credit(heading(0), row) == pytest.approx(1.0) and outcome(heading(0), row) == "correct"
    assert penalised(heading(0), row) == pytest.approx(1.0)
    assert credit(heading(180), row) == pytest.approx(0.0) and outcome(heading(180), row) == "wrong"
    assert penalised(heading(180), row) == pytest.approx(-1.0)


def test_a_heading_that_makes_less_progress_loses_by_its_shortfall_and_keeps_the_direction():
    """Contract (FD-0.11 §3.4.2): any heading with room to move has the direction quarter, even one that leads away
    from the thing, and loses by the shortfall `best - outcome` on N' = max(C, a tenth of `step_m`, 2 rho) = 0.2. By
    hand: the left heading (0.3, shortfall 0.5): t = (0.8 - 0.5) / 0.4 = 0.75, a = f = 0, s = (1 + 0.75) / 4 = 0.4375,
    score -0.125; the right heading (-0.2, shortfall 1.0, past 4N' = 0.8): the direction alone, s = 1/4, score -0.5.
    Regression caught: a heading leading away given no direction (it has room to move), or a shortfall measured from
    the straight bearing and not from the best way."""
    row = route_row()
    assert penalised(heading(90), row) == pytest.approx(-0.125)
    assert penalised(heading(-90), row) == pytest.approx(-0.5)


def test_the_answer_is_rounded_to_the_profiles_step_and_180_wraps():
    """Contract (as `regret`): the answered heading rounds to the nearest entry (45 degrees is half a step of 90 and
    rounds up to entry 1, 44.9 stays at entry 0), and -180 and 180 are the same entry. By hand: 44.9 scores as the best
    (+1), 45 as the left heading (-0.125), and both -180 and 180 as the null (-1). Regression caught: a floor that reads
    a neighbouring entry, or -180 treated as a heading of its own."""
    row = route_row()
    assert penalised(heading(44.9), row) == pytest.approx(1.0) and penalised(heading(45), row) == pytest.approx(-0.125)
    assert penalised(heading(180), row) == pytest.approx(-1.0) == penalised(heading(-180), row)


def test_the_distance_the_outcome_is_judged_over_widens_the_need_as_every_distance_does():
    """Contract (FD-0.11 §3.1, §3.4.2): the outcome is measured over `step_m` of driving, so the need is max(C, a tenth
    of `step_m`, 2 rho). By hand, C = 0.05, rho 0.01, `step_m` 1.0 (a tenth 0.10) and the left heading 0.1 short of the
    best (0.7 against 0.8): N' = 0.10, t = 1, a = (0.2 - 0.1) / 0.1 = 1, f = 1 - ln 10 / ln 10 = 0, s = 3/4, score +0.5;
    on C alone (N' = 0.05) it would be t = 1, a = 0, s = 1/2, score 0. Regression caught: the route's need left at C
    when the step is long."""
    row = route_row(profile_m=[0.8, 0.7, None, -0.2], need_m=0.05, resolution_m=0.01)
    assert penalised(heading(90), row) == pytest.approx(0.5)


def test_a_route_row_must_hold_every_key_it_reads_and_an_unknown_gold_is_answered_by_unknown():
    """Contract (records): a `route` row of a heading without `step_m`, `resolution_m`, `need_m` or `step_deg`, or with
    a profile of the wrong length or with no heading that has an outcome, raises ValueError naming the key (so no
    default decides a credit unseen); an outcome may be negative; `unknown` against an answerable route gold scores 0
    and against an `unknown` gold 1. Regression caught: a missing step read as 0, a short profile read modulo its
    length, or a negative outcome refused as if it were a run."""
    row = route_row()
    for key in ("step_m", "resolution_m", "need_m", "step_deg"):
        with pytest.raises(ValueError, match=key):
            credit(heading(0), {**row, "tolerances": {k: v for k, v in TOLERANCES.items() if k != key}})
    for bad in ({"profile_m": [0.8, 0.3, None]}, {"profile_m": [None] * 4}):
        with pytest.raises(ValueError, match="profile_m"):
            credit(heading(0), route_row(**bad))
    assert credit(UNKNOWN, row) == 0.0
    blank = rec(UNKNOWN, slice="unknown", credit="route", family="way_to", tolerances={})
    assert credit(UNKNOWN, blank) == 1.0 and credit(heading(0), blank) == 0.0
