"""`path_log`'s ego probe over golds that carry the graded credit: the prior's offset golds are put on the
map as position golds and scored with the rows' resolution and need alone (a place holds no room, 0.9.0 review B2)."""

from __future__ import annotations

import pytest

from baseline_kit import step
from test_path_log_ego_places import asked, two_spots


def graded(records: list[dict]) -> list[dict]:
    """The same records with the credit the 0.9.0 compile writes on an ego gold: `graded`, its resolution, its need
    (the vacuum's C, 0.10 m) and its harmful side, with no `ladder`."""
    for r in records:
        r["gold"]["credit"] = "graded"
        r["gold"]["tolerances"] = {"resolution_m": 0.02, "need_m": 0.1, "harm": "radial"}
    return records


def test_a_place_is_learned_from_graded_ego_golds(tmp_path):
    """Contract (`baselines_path_ego.on_the_map`, `score_graded._position`): the probe judges a place by scoring the
    prior's golds as positions, so a prior whose golds are `graded` (an offset's `radial` harm among them, which a place
    ignores) is scored with the rows' resolution and need. By hand: the two-spot case of `test_path_log_ego_places`
    (eight scripts, a chair 2.87 m ahead of each base, four at x = 1.0 and four at x = 9.0) with graded golds. Asked
    from the base (1.5, 1.0) heading north the answer is place A's point from the camera (1.5, 1.13): 2.87 ahead and
    0.5 left, as with the ladder. Regression caught: a graded position the scorer refuses, so every `ego_mem` or
    `ego_now` question the learned place was judged on ended in an ask error (an unknown, in the check, and a failed
    instrument)."""
    records = graded(two_spots("ego_now", lambda x: (step(0, x, 1.0, 90.0)["camera"],)))
    got = asked(tmp_path, "ego_now", records, [step(0, 1.5, 1.0, 90.0)])
    assert (got["ahead_m"], got["left_m"]) == pytest.approx((2.87, 0.5), abs=1e-3)
