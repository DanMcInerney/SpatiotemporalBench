"""The interfaces 0.7.0's credit by clearance builds on (lane D0): `rules.clearance_ladder`, `rules.hidden_widening`
and `JobSpec.clearance` / `reacquire`. No family uses them yet; each lane after reads them as they are here. Expected
values are worked by hand."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench import job_registry, score
from stbench.job_registry import JobSpec
from stbench.rules import clearance_ladder, hidden_widening, relative_ladder, widening

VACUUM = job_registry.get("vacuum/1")


def bare(**more) -> JobSpec:
    """A job built with the required fields only, as every job file but the ones D1 edits is."""
    return JobSpec(id="bare/1", robot="vacuum", domain="home", task_kinds=("clean_room",), decisions={}, events={},
                   plan=VACUUM.plan, script=VACUUM.script, relevance=VACUUM.relevance, checks=VACUUM.checks,
                   anchors=frozenset(), assignments={}, **more)


def offset(ahead: float) -> dict:
    return {"type": "offset", "ahead_m": ahead, "left_m": 0.0}


@pytest.mark.parametrize("c, rungs", [(0.10, [0.05, 0.10, 0.20]), (0.15, [0.075, 0.15, 0.30]),
                                      (0.5, [0.25, 0.5, 1.0]), (0.6, [0.3, 0.6, 1.2])])
def test_the_clearance_rungs_are_half_the_clearance_the_clearance_and_twice_it(c, rungs):
    """Contract: `clearance_ladder(c)` is `[c/2, c, 2c]`, absolute (they do not depend on how far the thing is), by hand
    for the four clearances the brief lists (0.10, 0.15, 0.50, 0.60 m). Regression caught: the rungs taken as shares of
    a range (as `relative_ladder` does), a rung dropped, or the middle rung moved off the clearance itself."""
    assert clearance_ladder(c) == pytest.approx(rungs)


def test_the_clearance_rungs_have_the_shape_of_a_row_s_ladder():
    """Contract: the rungs are a list of three strictly increasing positive floats rounded to the micrometre, the shape
    `relative_ladder` returns, so a gold row carries them as `tolerances.ladder` and the unchanged scorer
    credits an offset error of 0.2 m (under c = 0.5 m's first rung 0.25) in full, 0.5 m (the second) two thirds, 1.0 m
    (the third) one third and 1.1 m nothing, by hand. Regression caught: a tuple or a four-rung list (a different
    share per rung), and a rounding that moves a rung off its error so the credit steps the wrong way."""
    rungs = clearance_ladder(0.123456789)
    assert isinstance(rungs, list) and len(rungs) == len(relative_ladder(1.0)) == 3
    assert rungs == [0.061728, 0.123457, 0.246914]
    assert all(a < b for a, b in zip(rungs, rungs[1:], strict=False))
    row = {"qid": "q", "gold": offset(2.0), "slice": "answerable", "credit": "ladder",
           "tolerances": {"ladder": clearance_ladder(0.5)}}
    assert [score.credit(offset(2.0 + e), row) for e in (0.2, 0.5, 1.0, 1.1)] == [1.0, 2 / 3, 1 / 3, 0.0]


def test_the_clearance_refuses_what_is_not_a_clearance():
    """Contract: a clearance that is zero, negative, not finite or not a number is refused (a zero clearance makes every
    rung meet at 0, a NaN one makes every error meet them). Regression caught: a job whose unset or mis-typed
    clearance silently yields rungs that credit every answer."""
    for bad in (0.0, -0.1, float("nan"), float("inf"), "0.5", None, True):
        with pytest.raises(ValueError, match="c_m"):
            clearance_ladder(bad)


def test_the_hidden_top_widening_is_zero_when_the_top_is_in_view_and_positive_when_it_is_not():
    """Contract: `hidden_widening(size, cam_z, top_z)` is `widening(size)` for a thing whose top is above the camera and
    0 for one whose top is below it. By hand: a 2.0 x 1.0 x 1.5 m thing widens by max(2.0, 1.0) / 2 - 0.25 = 0.75 m
    (a camera at 0.4 m under its 1.5 m top: 0.75; at 2.0 m, above it: 0); a 0.4 m tote widens by 0.2 - 0.25 < 0, so 0 in
    either case. Regression caught: the widening applied whatever the camera saw (the drone's, whose camera is above
    every top), or never applied (a van seen from a vacuum's low camera answered at its middle)."""
    van = (2.0, 1.0, 1.5)
    assert hidden_widening(van, cam_z=0.4, top_z=1.5) == pytest.approx(0.75)
    assert hidden_widening(van, cam_z=2.0, top_z=1.5) == 0.0
    assert hidden_widening((0.4, 0.3, 0.2), cam_z=0.1, top_z=0.2) == 0.0
    assert hidden_widening(van, cam_z=0.4, top_z=1.5) == widening(van)


def test_a_top_level_with_or_below_the_camera_is_not_widened():
    """Contract: only a top strictly above the camera is hidden; a top at the camera's height is edge-on and a camera at
    or above it is not widened for. Regression caught: the comparison flipped (a drone's 15 m camera widened for a car,
    an AMR's 0.3 m camera not widened for a 2 m pallet stack)."""
    pallets = (1.2, 1.0, 2.0)   # 1.2 / 2 - 0.25 = 0.35 m
    assert hidden_widening(pallets, cam_z=0.3, top_z=2.0) == pytest.approx(0.35)
    assert hidden_widening(pallets, cam_z=2.0, top_z=2.0) == 0.0
    assert hidden_widening(pallets, cam_z=15.0, top_z=2.0) == 0.0


def test_the_re_find_radius_is_the_matching_radius_unless_the_job_declares_its_own():
    """Contract: `JobSpec.reacquire_m` is the job's `match_m` (twice its unit: 0.5 m for the vacuum's 0.25 m ladder, 2.0
    m for a 1.0 m one) when `reacquire` is not set, and the declared radius when it is. Regression caught: a default
    that is not the matching radius (the `task_things` radius and the re-find radius drifting apart), a declared radius
    that is ignored, and a stale default after `dataclasses.replace` changes the ladder."""
    assert bare().reacquire_m == bare().match_m == 0.5
    assert bare(reacquire=0.8).reacquire_m == 0.8
    assert replace(bare(), ladder=(1.0, 2.0, 4.0)).reacquire_m == 2.0


def test_an_unset_clearance_refuses_and_a_declared_one_is_read_back():
    """Contract: until a job file declares its clearance, `JobSpec.clearance_m` raises a ValueError that names the job
    (no job states one in D0 and none may silently get a default); once declared it is the declared number. Regression
    caught: a default clearance (0.0 or a guessed one) that lets a family compile credit nobody chose, and a message
    that does not say which job lacks it."""
    job = bare()
    with pytest.raises(ValueError, match=r"'bare/1' declares no clearance"):
        _ = job.clearance_m
    assert bare(clearance=0.25).clearance_m == 0.25
    assert clearance_ladder(bare(clearance=0.25).clearance_m) == [0.125, 0.25, 0.5]
