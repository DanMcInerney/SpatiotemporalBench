"""The `open_heading` gold (`questions_heading.open_heading`), on `heading_kit`'s tote AMR (a box 0.80 x 0.58 m, front
edge 0.4 m ahead of its base, half width 0.29, cap 5 m, the job's C 0.45 m) standing at the origin heading +x for twenty
keyframes in `transit`; the look point is keyframe 19, which ends the look window of keyframes 16 to 19. The free-space
cells are set by hand per keyframe. `wall_cells()` is free floor west of a wall with its centre line at x 1.01, so the
body turned theta from +x touches it after `1.01 / cos(theta) - 0.4 - 0.29 tan(theta)` metres (the corner of its leading
side), less half a sub-cell: 0.6 ahead; at 82 degrees about 4.79; from 83 degrees on past the 5 m cap."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from heading_kit import HEADING_JOB, LAST, heading_world, occupied, room, sub, wall_cells
from stbench import corridor_screens, questions_heading
from stbench.questions_select import Cand


def asked(w, k=LAST) -> Cand | None:
    m = questions_heading.open_heading(w, k)
    cands = [] if m is None else [*m.answerable, *m.unknown]
    assert m is None or (m.key == str(k) and m.unknown == [] and len(cands) == 1)
    return cands[0] if cands else None


def test_the_gold_is_the_best_heading_nearest_ahead_with_the_profile_on_its_row():
    """Contract: the gold heading is the one nearest the robot's own among those within a sub-cell of the longest run of
    the window's profile, in degrees to the left; the row carries the whole profile (360 runs at 1 degree), the best
    run's resolution (the 2 cm sub-cell: the run is the cap, no stopper), the job's C and the threshold T = L + C. Hand
    values on `wall_cells()` (the module docstring): ahead 0.6 (the wall's centre line 1.01 less the 0.4 front edge and
    half a sub-cell); 83 degrees runs to the cap and 82 stops short of it, 4.79; left and right are as near, so
    the left one: 83.0. T = 0.8 + 0.45 = 1.25. The candidate is answerable, critical, with no subject (it names
    nothing), the cap as its slot and keyframe 19 as its evidence (`now`). Regression caught: a gold from the wrong
    end of the profile (the farthest ahead, the first by index), one that rests on a heading short of the best, or a
    profile that is not on the row."""
    c = asked(heading_world({LAST: wall_cells()}))
    assert c is not None and c.slice == "answerable" and c.answer_class == "answerable"
    assert c.gold == {"type": "heading", "deg_left": 83.0}
    assert c.slots == {"cap": 5.0} and (c.scope, c.critical, c.decision_k) == ("now", True, LAST)
    assert c.subjects == () and c.range_m is None and c.evidence_k == (LAST,)
    t = c.tolerances
    assert set(t) == {"profile_m", "step_deg", "resolution_m", "need_m", "threshold_m"}
    assert (t["step_deg"], t["resolution_m"], t["need_m"], t["threshold_m"]) == (1, 0.02, 0.45, 1.25)
    assert len(t["profile_m"]) == 360 and t["profile_m"][0] == 0.6 and t["profile_m"][83] == 5.0
    assert 4.7 < t["profile_m"][82] < 4.98 and t["profile_m"][277] == 5.0 and t["profile_m"][180] == 5.0


def test_the_nearest_ahead_is_on_the_side_that_is_nearer():
    """Contract: of two openings the nearer to the robot's own heading wins, left or right. Hand case: `wall_cells()`
    and a side wall on the left (sub-cells at y 1.01 over x 0 to 1): the left headings are blocked by it up to well
    past 90 degrees (the body's leading side meets it), so the nearest heading that runs to the cap is the right one
    at 83 degrees right, -83.0, and not the left opening at more than 100. Regression caught: a left bias in the
    choice."""
    side = occupied((a, sub(1.01)) for a in range(0, 50))
    c = asked(heading_world({LAST: {**wall_cells(), **side}}))
    assert c is not None and c.gold == {"type": "heading", "deg_left": -83.0}
    assert c.tolerances["profile_m"][83] < 1.0   # the left heading that was the answer is blocked


def test_the_gold_reads_the_window_and_the_scope_says_memory_when_the_frame_cannot_answer():
    """Contract: the gold is over the latest cells of every keyframe of the window up to the point, not the point's
    frame alone (truth rule 2: what the robot last saw of each cell). Hand case: the wall seen at keyframe 5 and
    nothing in keyframes 16 to 19: the gold is still 83.0 on the same profile, and it is `memory` (the point's frame
    has no heading at all). The evidence is keyframe 5. Regression caught: a last-frame-only gold (it would be a dead
    band here), or evidence that names a frame that saw nothing."""
    c = asked(heading_world({5: wall_cells()}))
    assert c is not None and c.gold == {"type": "heading", "deg_left": 83.0}
    assert (c.scope, c.evidence_k) == ("memory", (5,))


def test_nothing_seen_asks_nothing():
    """Contract: a point where no heading's corridor was seen empty asks nothing (there is no `unknown` gold: the
    question names nothing for the robot to have never looked at). Regression caught: a default heading, or an
    `unknown` candidate, from an empty window."""
    assert questions_heading.open_heading(heading_world({}), LAST) is None


def test_it_is_asked_only_where_the_robot_drives():
    """Contract: as `clear_path`, the family is asked in the phases of `JobSpec.clear_phases` (the robot drives;
    where it sits the status answers). The same scene is asked in `transit` (keyframe 19's phase) and not when the
    job names only `docking`, or none. Regression caught: a heading asked at a station or a dock."""
    for phases in (frozenset({"docking"}), frozenset()):
        assert asked(heading_world({LAST: wall_cells()}, replace(HEADING_JOB, clear_phases=phases))) is None
    assert asked(heading_world({LAST: wall_cells()})) is not None


def test_a_point_may_be_any_frame_and_reads_no_frame_after_it():
    """Contract (0.10.0, one frame per `observe`): `open_heading` is asked at any frame, reading the cells up to it and
    none after. Hand case: `wall_cells()` seen only at keyframe 19; asked at 16, which ends no decision window, nothing
    has been seen (a dead band, None), and at 19 the heading 83.0 of the first test. Regression caught: a heading
    refused at a mid-window frame, or one read from the frames after it."""
    w = heading_world({LAST: wall_cells()})
    assert questions_heading.open_heading(w, 16) is None
    assert questions_heading.open_heading(w, LAST).answerable[0].gold == {"type": "heading", "deg_left": 83.0}


def test_a_box_whose_turn_would_sweep_something_it_saw_asks_nothing():
    """Contract (the turn sweep): a box body turning in place sweeps the disc of its half-diagonal (0.494 m for the
    tote AMR); an obstacle sub-cell seen inside it and outside the footprint makes the point ask nothing, because the
    heading cannot be reached by turning in place. Hand case: `wall_cells()` with a sub-cell at (0.31, 0.35), 0.468 m
    out and beside the footprint (|y| > 0.29): not asked. The same scene with the sub-cell inside the footprint,
    (0.21, 0.21), is asked. Regression caught: a heading through a wall beside the robot, or a turn refused for what
    the body holds."""
    beside = occupied([(sub(0.31), sub(0.35))])
    inside = occupied([(sub(0.21), sub(0.21))])
    assert asked(heading_world({LAST: {**wall_cells(), **beside}})) is None
    assert asked(heading_world({LAST: {**wall_cells(), **inside}})) is not None


@pytest.fixture
def screens(monkeypatch):
    """The three screens of `corridor_screens` as values the test sets: each passes until a test says otherwise, and the
    stopper has the given one-pixel error and tie gap (`stopper(...)` reads frames the kit does not render)."""
    state = SimpleNamespace(height=True, side=True, tied=True, visible=True, error=0.0, gap=0.0)
    monkeypatch.setattr(corridor_screens, "height_clear", lambda *a: state.height)
    monkeypatch.setattr(corridor_screens, "side_stable", lambda *a: state.side)
    monkeypatch.setattr(corridor_screens, "stopper", lambda *a: SimpleNamespace(
        tied=state.tied, visible=state.visible, error_m=state.error, gap_m=state.gap), raising=False)
    return state


def row() -> Cand | None:
    """The candidate asked in the 4 m room, or None."""
    return asked(heading_world({LAST: room(2.0)}))


def test_a_best_run_short_of_the_cap_must_pass_the_screens(screens):
    """Contract (the screens): a best run that does not reach the cap is asked only when its corridor passes
    `clear_path`'s screens: the height and the side, and a stopper that is tied, visible and read to at most C/2
    (0.225 m for this job) at one pixel and in the tie gap. Hand case: the 4 m room (the walls' centres at x, y =
    +-2.01), whose best run is short of the 5 m cap: asked with every screen passing; not asked with the height
    screen failing, the side screen failing, the stopper untied, the stopper not visible, its error 0.3 or its gap
    0.3. Regression caught: a gold that rests on a stop a camera cannot place (an untied stopper), or a screen
    computed and not consulted."""
    assert row() is not None
    for field, bad in (("height", False), ("side", False), ("tied", False), ("visible", False), ("error", 0.3),
                       ("gap", 0.3)):
        good = getattr(screens, field)
        setattr(screens, field, bad)
        assert row() is None, field
        setattr(screens, field, good)


def test_the_resolution_is_the_stoppers_error_or_its_gap_or_two_centimetres(screens):
    """Contract: the row's `resolution_m` is the larger of the 2 cm sub-cell, the stopper's one-pixel error and the tie
    gap, as `clear_path`'s. Hand cases on the room: error 0.05 and gap 0.03 give 0.05; error 0.01 and gap 0.04 give
    0.04; error 0.01 and gap 0 give 0.02. A run to the cap needs no stopper: 0.02 (the wall scene, where the stopper is
    never called). Regression caught: a resolution under the sub-cell, or one read from the wrong field."""
    for error, gap, want in ((0.05, 0.03, 0.05), (0.01, 0.04, 0.04), (0.01, 0.0, 0.02)):
        screens.error, screens.gap = error, gap
        assert row().tolerances["resolution_m"] == want
    screens.tied = False   # a refusing stopper does not matter at the cap
    assert asked(heading_world({LAST: wall_cells()})).tolerances["resolution_m"] == 0.02
