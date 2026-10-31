"""The worked scoring examples of the README and `docs/question-design.md`, scored by the real scorer: a number in a
document is a claim about the code, and these tests are where it is checked. Each expected value is the document's own,
to its printed decimals (so within half of the last one)."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, num, off, pos, rec
from stbench import credit_kernel
from stbench.score import credit, reported

PRINTED = 0.005 + 1e-9   # half of the last printed decimal


def corridor(gold, rho, need, threshold):
    """A `clear_path` gold as the compiler writes it, with the catastrophe rule of an `over` row."""
    return rec(num(gold), credit="graded", family="clear_path",
               tolerances={"resolution_m": rho, "need_m": need, "harm": "over", "threshold_m": threshold,
                           "catastrophe": {"weight": 2, "rule": "over", "margin_m": need, "threshold_m": threshold}})


def offset(ahead, left, rho, need=0.15):
    return rec(off(ahead, left), credit="graded", family="ego_mem",
               tolerances={"resolution_m": rho, "need_m": need, "harm": "radial"})


def place(x, y, rho):
    return rec(pos(x, y), credit="graded", family="object_where",
               tolerances={"resolution_m": rho, "need_m": 1.0, "harm": None})


def test_the_readmes_corridor_example():
    """Contract (README, Scoring, Worked example): the helper, C = 0.15 m, body 0.34 m so T = 0.49 m, a corridor of
    0.307 m with one pixel 4.7 cm. By hand: 0.45 m is 14 cm too long, harmful, N = 7.5 cm widened to 2 rho = 9.4 cm, so
    t = 1, a = 0.48, f = 0: credit 0.62, score +0.24; 0.164 m is 14 cm short, N = 15 cm: a = 1, f = 0.04, credit 0.76,
    score +0.52; 1.0 m is across T, credit 0, and a catastrophe: -2.00. Regression caught: a document that says what the
    scorer stopped doing."""
    row = corridor(0.307, 0.047167, 0.15, 0.49)
    assert credit(num(0.45), row) == pytest.approx(0.62, abs=PRINTED)
    assert credit(num(0.164), row) == pytest.approx(0.76, abs=PRINTED)
    assert credit(num(1.0), row) == 0.0
    assert reported(num(0.35), row) == pytest.approx(1.0)
    assert reported(num(0.45), row) == pytest.approx(0.24, abs=PRINTED)
    assert reported(num(0.164), row) == pytest.approx(0.52, abs=PRINTED)
    assert reported(num(1.0), row) == -2.0


def test_the_readmes_far_field_example():
    """Contract (README, Scoring, The far field): the helper (C = 0.15 m) and a door 5 m away, one pixel there 13.65 cm,
    so the margin is a tenth of the distance, 0.50 m. By hand: 4.5 m is 0.5 m short, credit 3/4, +0.50; 4.0 m is 1.0 m
    short, credit 1/2, 0.00; 6.0 m is 1.0 m long, harmful, N' = 0.25 widened to 2 rho = 0.273, t = 0.17, credit 0.29,
    score -0.42. (13.7 cm, the rounded pixel, would give -0.41.) Regression caught: the far field's margin falling back
    to the camera's two pixels or to C."""
    door = corridor(5.0, 0.1365, 0.15, 0.49)
    assert credit(num(6.0), door) == pytest.approx(0.29, abs=PRINTED)
    assert [reported(num(a), door) for a in (4.5, 4.0)] == pytest.approx([0.5, 0.0], abs=PRINTED)
    assert reported(num(6.0), door) == pytest.approx(-0.42, abs=PRINTED)


# the table of `question-design.md`, section 6: a door 5 m away, the score of each answer on each job as
# `(0.10.0, 0.11)`; the key is the pixel at 5 m, unrounded
DOOR = {
    "vacuum": {6.0: (-0.03, -0.03), 5.5: (0.47, 0.47), 5.0: (1.0, 1.0), 4.75: (0.96, 0.96), 4.5: (0.47, 0.50),
               4.0: (-0.03, 0.0), 3.0: (-0.5, -0.5), 2.5: (-0.5, -0.5)},
    "helper": {6.0: (-0.42, -0.42), 5.5: (0.08, 0.08), 5.0: (1.0, 1.0), 4.75: (0.56, 0.77), 4.5: (0.08, 0.50),
               4.0: (-0.42, 0.0), 3.0: (-0.5, -0.5), 2.5: (-0.5, -0.5)},
    "amr": {6.0: (-0.5, -0.5), 5.5: (-0.06, 0.0), 5.0: (1.0, 1.0), 4.75: (0.71, 0.73), 4.5: (0.44, 0.50),
            4.0: (-0.06, 0.0), 3.0: (-0.5, -0.5), 2.5: (-0.5, -0.5)},
    "drone": {6.0: (0.53, 0.53), 5.5: (0.62, 0.62), 5.0: (1.0, 1.0), 4.75: (0.77, 0.77), 4.5: (0.68, 0.68),
              4.0: (0.60, 0.60), 3.0: (0.02, 0.02), 2.5: (-0.02, -0.02)},
}
DOOR_ROWS = {   # the AMR asks no corridor, so its door is a thing straight ahead
    "vacuum": corridor(5.0, 0.2365, 0.10, 0.44),
    "helper": corridor(5.0, 0.1365, 0.15, 0.49),
    "amr": offset(5.0, 0.0, 0.108, 0.45),
    "drone": corridor(5.0, 0.034, 2.4, 3.7),
}


@pytest.mark.parametrize("version, kappa", [(0, 0.0), (1, credit_kernel.KAPPA)])
@pytest.mark.parametrize("job", sorted(DOOR))
def test_the_door_table_of_question_design(job, version, kappa, monkeypatch):
    """Contract (`question-design.md`, section 6, The door, scored): each printed score is what the scorer gives that
    job's door 5 m away for the answer, at 0.11.0's tenth of the distance and, with the tenth switched off, at 0.10.0's
    two pixels. Regression caught: the table and the kernel drifting apart, as the helper's 6.0 m did when the example
    quoted a rounded pixel."""
    monkeypatch.setattr(credit_kernel, "KAPPA", kappa)
    for answer, printed in DOOR[job].items():
        given = off(answer, 0.0) if job == "amr" else num(answer)
        assert reported(given, DOOR_ROWS[job]) == pytest.approx(printed[version], abs=PRINTED), (job, answer)


def test_the_readmes_case_table():
    """Contract (README, A case, the whole case): the example answers of the case's golds score as printed. The golds
    are the case's own, with their resolutions: an offset 0.76 m ahead (rho 0.01) and three remembered ones 2.8 m, 4.8 m
    and 0.4 m away (the margin is a tenth of the first two), two places (R = 1 m) and a list of one thing (match radius
    0.5 m). Regression caught: a printed score the scorer no longer gives."""
    for row, answers, printed in (
            (offset(0.7595, 0.0825, 0.01), [(0.77, 0.10), (0.85, 0.10), (1.00, 0.10)], [0.82, 0.39, -0.30]),
            (offset(-2.324, 1.5917, 0.01), [(-2.35, 1.60), (-2.20, 1.70), (-2.90, 1.60)], [0.81, 0.58, -0.50]),
            (offset(0.2454, -4.8171, 0.013801), [(0.25, -4.80), (0.50, -4.80), (0.25, -4.00)], [0.96, 0.59, 0.15]),
            (offset(-0.3157, 0.2568, 0.01), [(-0.32, 0.26), (-0.45, 0.26), (-0.60, 0.26), (0.30, 0.26)],
             [1.00, 0.10, -0.45, -1.00])):
        assert [reported(off(a, b), row) for a, b in answers] == pytest.approx(printed, abs=PRINTED)
    for (x, y), rho, printed in (((0.8824, 4.4203), 0.011163, [0.78, 0.51, -0.10]),
                                 ((3.1765, 4.1621), 0.010088, [0.77, 0.51, -0.10])):
        assert [reported(pos(x + d, y), place(x, y, rho)) for d in (0.08, 0.9, 1.2)] == pytest.approx(printed,
                                                                                                    abs=PRINTED)
    things = rec({"type": "positions", "items": [{"xy": [2.7029, 3.7943]}]}, credit="graded_set", family="task_things",
                 tolerances={"match_m": 0.5, "resolutions_m": [0.0113], "fp_weight": 0.5})
    listed = [reported({"type": "positions", "items": [{"xy": list(p)} for p in ps]}, things)
              for ps in ([(2.8, 3.8)], [(3.1, 3.8)], [])]
    assert listed == pytest.approx([0.72, 0.53, -1.0], abs=PRINTED)
    unknown = rec(UNKNOWN, slice="unknown", credit="graded", tolerances={})
    assert reported(UNKNOWN, unknown) == 1.0 and reported(off(-90.0, 0.0), unknown) == -1.0


def test_the_prompts_anchors_of_the_score():
    """Contract (`llm_credit._penalised`, told to the agent): with the margin C, an answer off by C scores 0.5 and one
    off by 2C scores 0; on the harmful side an answer off by C (twice the halved margin) scores 0; the direction alone
    scores -0.5 and a wrong direction -1. By hand, C = 0.15, one pixel 0.02, a corridor of 1.0 m (within ten margins, so
    the need is C): 0.85 (short by C): credit 3/4; 0.7 (by 2C): credit 1/2; 1.15 (long by C, N' = 0.075): credit 1/2;
    1.6 (long, direction right, past 4N'): credit 1/4; 0.2 (across T): 0. Regression caught: the prompt promising a
    score the scorer does not give."""
    row = corridor(1.0, 0.02, 0.15, 0.49)
    assert [reported(num(a), row) for a in (0.85, 0.7, 1.15, 1.6, 0.2)] == pytest.approx([0.5, 0.0, 0.0, -0.5, -1.0])
