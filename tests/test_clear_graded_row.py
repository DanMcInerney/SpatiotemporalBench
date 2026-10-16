"""A `clear_path` row is scored by the graded credit (0.9.0, `clear_ground.moment`, `score.credit`): its tolerances hold
exactly the keys `records.CREDIT_KEYS` lists for a graded number, with the harmful side the longer answer. `clear_kit`'s
tote AMR, a wall 2.01 m ahead of the front edge (gold 2.0, resolution 0.0229 m), clearance C 0.45 m, body 0.8 m long."""

from __future__ import annotations

import pytest

from clear_kit import LAST, drive_run, wall, world
from stbench import clear_ground, score
from stbench.records import CREDIT_KEYS

FACE = wall(3.01)


def gold_row():
    m = clear_ground.moment(world(drive_run(boxes=[FACE.at]), [FACE]), LAST)
    (cand,) = m.answerable
    return {"qid": "q", "family": "clear_path", "gold": cand.gold, "slice": "answerable", "credit": "graded",
            "tolerances": cand.tolerances}


def test_the_row_holds_the_graded_number_keys_and_scores_by_them():
    """Contract (plan G1): the tolerances are exactly `CREDIT_KEYS["graded"]["number"]` (resolution, need, harm,
    threshold), and the scorer reads them: the gold scores 1; an answer short by the need (0.45) is within the need but
    not a pixel: (1 + 1 + 1 + 0) / 4 = 0.75; an answer long by the need is on the harmful side, whose need is halved
    (0.225): the direction and twice-the-need quarters hold and the others are gone: (1 + 1 + 0 + 0) / 4 = 0.5.
    Hand values from the kernel with N 0.45, e 0.45. Regression caught: a row the scorer reads another credit's keys
    from, a harm rule on the wrong side, or a resolution below the answer's reach."""
    row = gold_row()
    assert set(row["tolerances"]) == set(CREDIT_KEYS["graded"]["number"])
    assert row["tolerances"]["harm"] == "over" and row["tolerances"]["threshold_m"] == pytest.approx(1.25)
    assert score.credit({"type": "number", "value": 2.0}, row) == 1.0
    assert score.credit({"type": "number", "value": 1.55}, row) == pytest.approx(0.75)
    assert score.credit({"type": "number", "value": 2.45}, row) == pytest.approx(0.5)
    assert score.credit({"type": "unknown"}, row) == 0.0
