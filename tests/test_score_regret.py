"""The `regret` credit (brief §3.5; room to move and the one need rule, FD-0.11 §3.3): a heading is scored by the room
it gives up against the best heading, the four quarters on that regret, the need the larger of the clearance C and a
tenth of the best run. The profile on the gold row lists the run at each heading, entry i the heading i * step_deg
degrees to the left of the way the robot faces. Expected values are worked by hand from the rule."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, rec
from stbench.score import credit, outcome, penalised

VACUUM = {"resolution_m": 0.02, "need_m": 0.10, "threshold_m": 0.44}   # C = 0.10, T = 0.34 + 0.10


def heading(deg):
    return {"type": "heading", "deg_left": deg}


def heading_row(runs: dict[int, float | None], *, step=1, fill=None, **over):
    """A heading gold row with a 360 / step profile of `fill` (a run never seen) but for `runs` (index -> run)."""
    profile = [fill] * round(360 / step)
    for index, run in runs.items():
        profile[index] = run
    return rec(heading(0), credit="regret", family="open_heading",
               tolerances={**VACUUM, "step_deg": step, "profile_m": profile, **over})


# the best run 2.60 m dead ahead (heading 0); 2.55 m at 90 (left); 2.30 at 180 (back); 0.30 at 270 (right); the
# rest unseen
RUNS = {0: 2.60, 90: 2.55, 180: 2.30, 270: 0.30}


def test_the_brief_s_worked_headings_score_by_their_regret():
    """Contract (brief Appendix B, §3.5; FD-0.11 §3.3): the vacuum (C = 0.10, T = 0.44, rho 0.02) with a best run of
    2.60 m scores a heading by its regret r = best - run on the need N' = max(C, a tenth of 2.60) = 0.26. By hand:
    - the best heading (r = 0): 1;
    - a heading running 2.55 m (r = 0.05, within N'): (1 + 1 + 1 + (1 - ln 2.5 / ln 13 = 0.6428)) / 4 = 0.9107;
    - a heading running 2.30 m (r = 0.30, past N' = 0.26, within 2N' = 0.52): room to move, so the direction; t = 1,
      a = (0.52 - 0.30) / 0.26 = 0.8462, f = 0: (1 + 1 + 0.8462) / 4 = 0.7115;
    - a heading running 0.30 m (under T, so not room to move: no direction and harmful, N' = 0.13; r = 2.30 past
      4N' = 0.52): 0.
    Regression caught: a regret read from the wrong bin, a harmful side that is not the run under T, a need that stays
    C when the best run is long, or full credit for a run that is merely within the need."""
    row = heading_row(RUNS)
    for deg, want in ((0, 1.0), (90, 0.9107), (180, 0.7115), (-90, 0.0)):
        assert credit(heading(deg), row) == pytest.approx(want, abs=1e-4), deg
    assert outcome(heading(0), row) == "correct" and outcome(heading(90), row) == "partial"
    assert outcome(heading(-90), row) == "wrong"


def test_a_heading_into_space_never_seen_runs_nothing_and_is_harmful_with_the_full_regret():
    """Contract (§3.5): a null run counts 0, so a heading into an unseen bin has the full regret of the best run, is
    harmful (0 is under T) and has no direction (0 has no room to move). By hand: heading 45 is unseen: r = 2.60,
    N' = 0.13 (the harmful side's half of 0.26), every tier 0 and no direction: 0. Regression caught: a null read as
    'the best' or skipped, which would score an answer nobody could ground."""
    row = heading_row(RUNS)
    assert credit(heading(45), row) == pytest.approx(0.0)
    assert credit(UNKNOWN, row) == 0.0   # `unknown` is an abstention on an answerable gold


def test_minus_180_and_180_are_one_heading_and_left_is_positive():
    """Contract (§3.5, `answers.Heading`): headings wrap at +-180, so -180 and 180 are the same entry (180); 90 is to
    the robot's left and -90 (entry 270) to its right. By hand: with the best run (2.60) only at the left (90) and 0.30
    at entry 270, the answer 90 scores 1 and -90 scores 0; with 2.30 at entry 180, both 180 and -180 score 0.7115 (a
    regret of 0.30 on N' = 0.26: (1 + 1 + (0.52 - 0.30) / 0.26) / 4). Regression caught: -180 treated as a heading of
    its own, or left and right swapped."""
    row = heading_row({90: 2.60, 270: 0.30, 180: 2.30})
    assert credit(heading(90), row) == pytest.approx(1.0) and credit(heading(-90), row) == pytest.approx(0.0)
    assert credit(heading(180), row) == pytest.approx(0.7115, abs=1e-4) == credit(heading(-180), row)


def test_the_answer_is_rounded_to_the_profiles_step():
    """Contract (§3.5): the answered heading is rounded to the nearest entry of the profile before its run is read. By
    hand, a step of 5 degrees (72 entries) with 2.60 m at entry 1 (5 degrees) and 1.00 m at entry 2 (10 degrees): 7.4
    rounds to 5 and scores 1; 7.5 rounds up to 10, the run 1.00 (room to move, so the direction; r = 1.6 past 4N' =
    1.04): 1/4. At a step of 1 degree 0.4 is entry 0 and 0.5 entry 1. Regression caught: a floor or a truncation that
    reads a neighbouring bin, or a step the scorer ignores."""
    row = heading_row({1: 2.60, 2: 1.00}, step=5)
    assert credit(heading(7.4), row) == pytest.approx(1.0)
    assert credit(heading(7.5), row) == pytest.approx(0.25)
    fine = heading_row({0: 2.60, 1: 1.00})
    assert credit(heading(0.4), fine) == pytest.approx(1.0) and credit(heading(0.5), fine) == pytest.approx(0.25)


def test_a_heading_with_room_to_move_earns_the_direction_and_one_without_is_harmful():
    """Contract (FD-0.11 §3.3, FD-heading-credit §7): the right direction is room to move, a run of T or the best run
    less a pixel, whichever is shorter, and the harmful side is the rest. By hand, the vacuum (C = 0.10, T = 0.44,
    rho 0.02), best run 4.0 m (N' = 0.40; harmful 0.20):
    - a run of 1.0 m (under half the best, but over T): the direction alone, r = 3.0 past 4N': 1/4, score -0.5 (where a
      direction of half the best made it 0, score -1);
    - a run of 0.45 m (just over T): the direction alone, 1/4;
    - a run of 0.30 m (under T): no direction, harmful, r = 3.7: 0.
    With no heading having T of room, best 0.30: the best itself has room to move (1), a run of 0.20 does not (r = 0.10
    on the harmful need 0.05: t = (0.2 - 0.1) / 0.1 = 1, a = 0: no direction, 1/4 of the credit, 0.25).
    Regression caught: a clear heading under half the best scored as a miss, a heading under T given the direction, or a
    best run under T that cannot earn full credit."""
    row = heading_row({0: 4.0, 90: 1.0, 180: 0.45, 270: 0.30})
    assert credit(heading(0), row) == pytest.approx(1.0)
    assert credit(heading(90), row) == pytest.approx(0.25) and credit(heading(180), row) == pytest.approx(0.25)
    assert credit(heading(-90), row) == pytest.approx(0.0)
    short = heading_row({0: 0.30, 90: 0.20})
    assert credit(heading(0), short) == pytest.approx(1.0) and credit(heading(90), short) == pytest.approx(0.25)


def test_a_headings_need_is_the_larger_of_the_clearance_and_a_tenth_of_the_best_run():
    """Contract (FD-0.11 §3.1, §3.3): the heading's need N' is max(C, a tenth of the best run, 2 rho), so the regret of
    a far heading is judged on the same rule as a distance. By hand, the vacuum (C = 0.10, rho 0.02), best 5.0 m, a
    heading running 3.5 m (r = 1.5, room to move): N' = 0.50, t = (2.0 - 1.5) / 1.0 = 0.5, a = f = 0, s = (1 + 0.5) / 4
    = 0.375, score -0.25; with best 0.8 m the tenth is 0.08 under C, so N' = 0.10: a run of 0.6 m (r = 0.2) is
    t = 1, a = 0, s = 0.5. Regression caught: a long run's regret scored on C alone (a regret of 1.5 m then is -0.5,
    the direction alone)."""
    far = heading_row({0: 5.0, 90: 3.5})
    assert credit(heading(90), far) == pytest.approx(0.375) and penalised(heading(90), far) == pytest.approx(-0.25)
    near = heading_row({0: 0.8, 90: 0.6})
    assert credit(heading(90), near) == pytest.approx(0.5)


def test_a_heading_row_must_hold_a_profile_of_its_step_and_every_key():
    """Contract (records): a `regret` row whose profile length is not 360 / step_deg, with a negative run, with no
    heading seen, or without `resolution_m`, `need_m`, `threshold_m` or `step_deg` raises ValueError naming the key.
    Regression caught: a profile shorter than its step read modulo its length, which would score the wrong bearing
    silently."""
    good = heading_row(RUNS)
    bad = [{"profile_m": [1.0] * 359}, {"profile_m": [-1.0] + [1.0] * 359}, {"profile_m": [None] * 360},
           {"step_deg": 5}]
    for change in bad:
        with pytest.raises(ValueError, match="profile_m"):
            credit(heading(0), {**good, "tolerances": {**good["tolerances"], **change}})
    for key in ("resolution_m", "need_m", "threshold_m", "step_deg"):
        with pytest.raises(ValueError, match=key):
            credit(heading(0), {**good, "tolerances": {k: v for k, v in good["tolerances"].items() if k != key}})
