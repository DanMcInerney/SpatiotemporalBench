"""The own-log screen (`heading_screen.told_by_log`) and its switch (`FamilySpec.own_log_screen`). The robot (the home
helper) drives east half a metre a keyframe for twenty keyframes, so at the ask (keyframe 19, the base at (9.5, 0.5)
facing +x) the pose eight steps earlier is keyframe 10's (5.0, 0.5): `path_log`'s `way_to` rule answers 180, straight
behind. The candidates' profiles are set by hand: a null everywhere but the two headings named."""

from __future__ import annotations

from dataclasses import replace

from stbench import heading_screen, questions_heading
from stbench.questions_select import Cand
from stbench.records import check_family
from way_kit import FRAMES, LAST, way_world

NAME = {"kind": "pallet", "attributes": {"color": "grey"}, "reach_m": 1.0, "room_m": 0.49}


def drive():
    return way_world({}, {k: (0.5 * k, 0.5, 0.0) for k in range(FRAMES)})


def cand(best: int, other: int) -> Cand:
    """A `way_to` candidate whose outcomes are 1.0 at heading `best` and 0.0 at `other`, null elsewhere."""
    profile = [None] * 360
    profile[best], profile[other] = 1.0, 0.0
    tol = {"profile_m": profile, "step_deg": 1, "step_m": 1.0, "resolution_m": 0.1, "need_m": 0.15}
    return Cand("way_to", dict(NAME), {"type": "heading", "deg_left": float(best)}, "answerable", (3,), (), tol,
                decision_k=LAST, subjects=("e:T1",), answer_class="answerable")


def test_the_screen_drops_the_point_the_own_log_answers_and_keeps_the_one_it_gets_wrong():
    """Contract (FD-0.11 §3.3): a point is told by the log when `path_log`'s unbarred own-log heading would score 0 or
    more on the point's own row. Hand cases (`way_to`, route credit): the best way is behind (180, outcome 1.0) and the
    log says 180: full credit, screened. The best way is ahead (0, outcome 1.0), the log says 180 with outcome 0.0, a
    regret of 1.0 m against a need of 0.15: the direction quarter alone, 0.25, penalised -0.5: kept. Early in the
    window (keyframe 5: only six poses, none eight steps back) the log gives no heading, and nothing is told. Regression
    caught: a screen that drops every point, none, or one that reads the gold's heading instead of the log's."""
    w = drive()
    assert heading_screen.told_by_log(w, cand(180, 0), LAST, "route")
    assert not heading_screen.told_by_log(w, cand(0, 180), LAST, "route")
    assert not heading_screen.told_by_log(w, cand(180, 0), 5, "route")


def test_a_family_turns_the_screen_on_with_its_spec_and_only_a_heading_family_may(monkeypatch):
    """Contract: `FamilySpec.own_log_screen` is a boolean that is False unless a family sets it, and only a heading
    family may. `open_heading` honours it: with the switch on and the screen saying the log answers, the point is a dead
    band; with it off the screen is not asked. Regression caught: a switch on a family that has no heading, or a family
    that ignores its own switch."""
    spec = questions_heading.SPECS[0]
    assert spec.own_log_screen is False and check_family(spec) == []
    assert check_family(replace(spec, own_log_screen=1))
    assert check_family(replace(spec, answer_type="offset", credit="graded", own_log_screen=True))
    calls = []
    monkeypatch.setattr(heading_screen, "told_by_log", lambda *a: calls.append(a) or True)
    from heading_kit import heading_world, wall_cells
    w = heading_world({LAST: wall_cells()})
    assert questions_heading.open_heading(w, LAST) is not None and calls == []
    monkeypatch.setattr(questions_heading, "SPECS", (replace(spec, own_log_screen=True),))
    assert questions_heading.open_heading(w, LAST) is None and len(calls) == 1
