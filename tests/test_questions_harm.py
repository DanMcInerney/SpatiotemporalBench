"""The pieces the harm families share (`questions_harm`): the catastrophe margin, the `over` and `miss` tolerances, the
list row and the empty-gold draw. Hand-written values; the shapes are the ones `records.check_gold` validates."""

from __future__ import annotations

import pytest

import approach_kit as ak
import person_kit as pk
from stbench import questions_harm as qh
from stbench.records import _catastrophe_ok
from stbench.seeding import stable_hash


def test_the_margin_is_the_clearance_or_twice_the_resolution_whichever_is_larger():
    """Contract (FD-questions-0.10 section 4.3): a catastrophe needs a harmful error larger than max(C, 2 rho), so an
    error inside the job's clearance or inside twice the pixel of the point that sets it is none. By hand: C = 0.45 and
    rho = 0.1 give 0.45; C = 0.1 and rho = 0.3 give 0.6; the margin is to the micrometre. Regression caught: a margin of
    C alone (a far point's pixel error, 0.5 m at 6 m, then makes a catastrophe of a good answer) or of 2 rho alone."""
    assert qh.margin_m(0.45, 0.1) == 0.45 and qh.margin_m(0.1, 0.3) == 0.6
    assert qh.margin_m(0.15, 0.0333333333) == 0.15 and qh.margin_m(0.1, 0.0123456789) == 0.1


def test_over_and_miss_are_the_shapes_the_validator_takes():
    """Contract (records, section 4.4): `over` holds the weight 3, the rule, the margin and the threshold; `miss` the
    weight 3, the margin and the critical places sorted and rounded to 4 decimals, whose margin takes the largest
    resolution of the items (an empty list takes C). Both pass the validator `records._catastrophe_ok`. Regression
    caught: a tolerance the compile would refuse, or critical places in an order the row does not fix."""
    over = qh.over(0.45, 0.1, 1.25)
    assert over == {"weight": 3, "rule": "over", "margin_m": 0.45, "threshold_m": 1.25} and _catastrophe_ok(over)
    gold, tol = qh.list_row([((3.0, 1.8), 0.3), ((2.2, 1.13), 0.2)], [((5.0, 5.0), 0.55), ((1.0, 1.0), 0.5)], 0.5, 0.1,
                            [(3.0, 1.8), (2.2, 1.13)])
    assert gold == {"type": "positions", "items": [{"xy": [2.2, 1.13]}, {"xy": [3.0, 1.8]}],
                    "dont_care": [{"xy": [1.0, 1.0], "radius": 0.5}, {"xy": [5.0, 5.0], "radius": 0.55}]}
    assert tol == {"match_m": 0.5, "resolutions_m": [0.2, 0.3], "fp_weight": 0.5,
                   "catastrophe": {"weight": 3, "rule": "miss", "margin_m": 0.6, "critical": [[2.2, 1.13], [3.0, 1.8]]}}
    assert _catastrophe_ok(tol["catastrophe"])
    empty_gold, empty_tol = qh.list_row([], [], 2.0, 2.4, [])
    assert empty_gold == {"type": "positions", "items": []} and empty_tol["resolutions_m"] == []
    assert empty_tol["catastrophe"] == {"weight": 3, "rule": "miss", "margin_m": 2.4, "critical": []}


def test_an_empty_gold_is_kept_when_its_hash_is_under_the_percent_and_a_percent_of_100_keeps_all():
    """Contract (the balance rule): `kept` is `stable_hash(case id, family, point, "empty") % 100 < percent`: percent 0
    keeps none, 100 keeps all, the hash itself is the boundary. By hand from the hash of case `c0`. Regression caught:
    the draw keyed on the wrong strings (two families at one point would then keep or drop together) or read the other
    way."""
    w = pk.world(ak.scene())
    h = stable_hash("c0", "closest_approach", "13", "empty") % 100
    assert not qh.kept(w, "closest_approach", 13, 0) and qh.kept(w, "closest_approach", 13, 100)
    assert not qh.kept(w, "closest_approach", 13, h) and qh.kept(w, "closest_approach", 13, h + 1)
    assert stable_hash("c0", "tangle_ahead", "13", "empty") != stable_hash("c0", "closest_approach", "13", "empty")


@pytest.mark.parametrize("k, name", [(0, "w:0-0"), (4, "w:0-4"), (7, "w:0-7"), (8, "w:1-8"), (13, "w:6-13"),
                                    (27, "w:20-27")])
def test_the_subject_is_the_last_eight_frames_of_the_task_up_to_the_point(k, name):
    """Contract (FD-questions-0.10 S7): the subject of a point is `w:<first>-<k>`, the first being the task's first
    frame or 7 before `k`. Regression caught: a stretch that starts before the task (it would name the previous task's
    frames) or one of another length than the look window."""
    assert qh.stretch(pk.world(ak.scene()), k) == (name,)
