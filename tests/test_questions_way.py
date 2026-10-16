"""The `way_to` gold (`questions_way.way_to`) on `way_kit`'s window: the helper's body at (2.0, 0.5) facing +x for
twenty keyframes in `transit`, the thing `T1` at (5.0, 0.5) behind the wall of `door_scene`. The subjects are
`ego_mem`'s and are the tests of `test_questions_ego_mem.py`; here `ego_mem` is replaced by a fixed moment, so what is
under test is the route: the row, the dead bands and what the gold reads. The look point is keyframe 19 (`LAST`)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench import heading_screen, questions, questions_way
from stbench.questions_select import Cand, Moment
from stbench.robots import ROBOTS
from way_kit import JOB, LAST, door_scene, way_world

NAME = {"kind": "pallet", "attributes": {"color": "grey"}}
SEEN_AT = 5   # the keyframe whose depth saw the rooms and the thing


def fixed_ego(monkeypatch, k=LAST):
    """`ego_mem` replaced by one answerable subject, `T1`, last seen at keyframe 5, and one unknown real thing."""
    thing = Cand("ego_mem", dict(NAME), {"type": "offset", "ahead_m": 3.0, "left_m": 0.0}, "answerable", (SEEN_AT,),
                 ("T1",), {}, decision_k=k, subjects=("e:T1",), answer_class="answerable")
    nobody = Cand("ego_mem", dict(NAME), {"type": "unknown"}, "unknown", (), (), {}, decision_k=k, subjects=("e:U1",),
                  answer_class="unknown")
    monkeypatch.setattr(questions_way, "ego_mem", lambda w, point: Moment(str(point), [thing], [nobody]))


def asked(w, k=LAST):
    return questions_way.way_to(w, k)


def test_the_gold_is_the_best_heading_and_the_row_carries_the_outcome_of_every_heading(monkeypatch):
    """Contract: the answerable candidate is the heading of the greatest outcome on its own row, the row holds the keys
    of credit `route` (the 360 outcomes, the 1 degree step, the reach R it is told over, the field's fine cell as the
    resolution and the job's C as the need) and the slots the table needs (R and the room T = 0.34 + 0.15). Hand values:
    the door at bearing 60 degrees from (2.0, 0.5) (`test_route_field.py`), R 1.0, T 0.49, C 0.15, the fine cell 0.1. It
    is `memory` evidence: the thing's sighting at keyframe 5 and the cells that certified the route (also keyframe 5);
    it names the thing as a subject and no entity. The compile writes it as a valid `route` row (`check_gold`) whose
    text names the thing. Regression caught: a gold off its own profile, a row missing a key the scorer reads, a table
    left without R and T, or a row the compile refuses."""
    fixed_ego(monkeypatch)
    w = way_world({SEEN_AT: door_scene()})
    m = asked(w)
    assert m is not None and m.key == str(LAST) and len(m.answerable) == 1
    c = m.answerable[0]
    t = c.tolerances
    assert set(t) == {"profile_m", "step_deg", "step_m", "resolution_m", "need_m"}
    assert (len(t["profile_m"]), t["step_deg"], t["step_m"], t["resolution_m"], t["need_m"]) == (360, 1, 1.0, 0.1, 0.15)
    best = max(v for v in t["profile_m"] if v is not None)
    assert c.gold["type"] == "heading" and t["profile_m"][round(c.gold["deg_left"]) % 360] == best
    assert 45 <= c.gold["deg_left"] <= 75
    assert c.slots == {**NAME, "reach_m": 1.0, "room_m": 0.49} and c.subjects == ("e:T1",) and c.entities == ()
    assert (c.slice, c.answer_class, c.decision_k, c.critical, c.evidence_k) == ("answerable", "answerable", LAST, True,
                                                                               (SEEN_AT,))
    qs, gs = questions.records([c], w)
    assert qs[0]["template"] == "way_to" and qs[0]["text"].startswith("If you had to go now to the grey pallet you saw")
    assert gs[0]["credit"] == "route" and gs[0]["scope"] == "memory" and gs[0]["age_band"] == "task"


def test_the_unknown_pool_is_egos_under_this_family_with_the_same_slots(monkeypatch):
    """Contract: the unknown candidates are `ego_mem`'s pool, re-filed under `way_to` with the slots of the family and
    the tolerances an unknown row holds (no profile), still an `unknown` gold, so the question reads the same whichever
    pool it came from. Regression caught: an unknown candidate left in family `ego_mem` (an `ego_mem` row asked at a
    `way_to` point), or one whose slots differ from an answerable one's (the kind would predict the slice)."""
    fixed_ego(monkeypatch)
    m = asked(way_world({SEEN_AT: door_scene()}))
    u = m.unknown[0]
    assert (u.family, u.gold, u.slice, u.answer_class, u.subjects) == ("way_to", {"type": "unknown"}, "unknown",
                                                                        "unknown", ("e:U1",))
    assert u.slots == m.answerable[0].slots
    assert u.tolerances == {"resolution_m": 0.1, "need_m": 0.15, "step_deg": 1, "step_m": 1.0}


def test_no_route_through_floor_seen_is_a_dead_band(monkeypatch):
    """Contract: where no route exists through floor seen or driven, nothing is asked (never an `unknown` gold, so the
    slice stays unpredictable from the question). Hand case: the door cells unseen, the robot never having driven there
    (`door_scene(door=False)`): None. Regression caught: a seen thing with an unreachable route asked as `unknown`
    (the form would tell the slice), or a route through the unseen door."""
    fixed_ego(monkeypatch)
    assert asked(way_world({SEEN_AT: door_scene(door=False)})) is None


def test_the_gold_reads_the_cells_up_to_the_point_and_none_after_it(monkeypatch):
    """Contract (truth rule 2): the walkable floor is what the window's frames up to the point had seen. Hand case: the
    rooms seen only at keyframe 19: asked at 18 there is no floor at all (a dead band), at 19 the row of the first
    test. Regression caught: a gold read from the frames after the ask, which no system could have been sent."""
    fixed_ego(monkeypatch, 18)
    w = way_world({LAST: door_scene()})
    assert asked(w, 18) is None
    fixed_ego(monkeypatch, LAST)
    assert asked(w, LAST) is not None


def test_the_route_goes_to_where_the_thing_was_last_seen_not_where_it_is_now(monkeypatch):
    """Contract (truth rule 2, `ego_mem`'s own: as you last saw it): the goal is the thing's place at its last
    recognisable sighting (keyframe 5), whatever has happened since. Hand case: `T1` at (5.0, 0.5) until keyframe 6,
    when it stands at (0.6, 2.0) in the west room, 1.4 m from the robot; the route is still the door's (best heading
    within 15 degrees of 60), not the short way to where it stands now. Regression caught: a gold taken from the
    thing's place at the ask, which a system that has not seen it move cannot know."""
    fixed_ego(monkeypatch)
    m = asked(way_world({SEEN_AT: door_scene()}, moved=(6, (0.6, 2.0))))
    assert 45 <= m.answerable[0].gold["deg_left"] <= 75


def test_it_is_asked_only_where_the_robot_drives(monkeypatch):
    """Contract: as `clear_path`, the family is asked in the status phases the job declares in `look_phases` (the robot
    drives; elsewhere its status answers), never by a flier, and a job that asks it without declaring them raises
    (which phases the robot drives in is the job's to say). The same scene is asked in `transit` (keyframe 19's phase),
    not when the job names only `waiting`. Regression caught: a route asked at a dock, or a phase chosen by default."""
    fixed_ego(monkeypatch)
    cells = {SEEN_AT: door_scene()}
    assert asked(way_world(cells, job=replace(JOB, look_phases={"way_to": frozenset({"waiting"})}))) is None
    with pytest.raises(ValueError, match="look_phases"):
        asked(way_world(cells, job=replace(JOB, look_phases={})))
    w = way_world(cells)
    flier = type(w)(replace(w.inp, robot=ROBOTS["drone"]), w.rec)
    assert asked(flier) is None and asked(w) is not None


def test_the_own_log_screen_is_off_by_default_and_drops_the_point_the_log_answers(monkeypatch):
    """Contract (FD-0.11 §3.3, the declared fallback): with `FamilySpec.own_log_screen` the point where
    `heading_screen.told_by_log` says the robot's own log answers is a dead band; by default the screen is not even
    asked. The screen itself is tested in `test_heading_screen.py`; here it is stubbed. Regression caught: a screen on
    by default (it would remove points no gate asked it to), or a family that ignores its own switch."""
    fixed_ego(monkeypatch)
    w = way_world({SEEN_AT: door_scene()})
    calls = []
    monkeypatch.setattr(heading_screen, "told_by_log", lambda *a: calls.append(a) or True)
    assert asked(w) is not None and calls == []
    monkeypatch.setattr(questions_way, "SPECS", (replace(questions_way.SPECS[0], own_log_screen=True),))
    assert asked(w) is None and len(calls) == 1
