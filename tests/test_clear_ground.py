"""The ground robot's `clear_path` gold (`clear_ground.moment`), on `clear_kit`'s tote AMR: base (0.6, 0) heading +x, so
its front edge is at x 1.0, 0.4 m ahead of the base and 0.02 m ahead of the camera (mount 0.38 m); the corridor is 0.58
m wide (half 0.29), cap 5 m; the look point is keyframe 19, which ends the look window of keyframes 16 to 19."""

from __future__ import annotations

from dataclasses import replace

import pytest

from clear_kit import JOB, LAST, Owned, drive_run, wall, world
from space_kit import box
from stbench import clear_ground, corridor_screens
from stbench.questions_select import Cand

FACE = wall(3.01, "")   # structure: its near face is 2.01 m ahead of the front edge


def asked(w, k=LAST) -> Cand | None:
    m = clear_ground.moment(w, k)
    cands = [] if m is None else [*m.answerable, *m.unknown]
    assert len(cands) <= 1   # a point offers one corridor, answerable or not
    return cands[0] if cands else None


def test_the_gold_is_the_corridor_from_the_front_edge():
    """Contract: the gold is `grid.corridor_clear` over the cells the case holds up to the look point, from the body's
    front edge (the base plus half the 0.8 m length), never the base or the camera. Hand values: a wall whose face is at
    x 3.01 lies in the sub-cell [3.00, 3.02), centre 3.01, which is 3.01 - 1.0 = 2.01 from the front edge: 2.01 - 0.01 =
    2.0 (from the base 2.4, from the camera, 0.02 behind the edge, 2.02); `range_m` is the gold plus the 0.02 m from the
    camera to the edge, 2.02. The candidate is answerable, critical, scope `now` (the look window alone gives 2.0),
    graded on the job's clearance 0.45 (the harmful side the longer answer, the threshold the body's 0.8 m plus 0.45),
    its resolution the stopper's one pixel from the camera 0.5 m up and 2.03 m from the wall's foot: (2.03^2 + 0.5^2) /
    (0.5 x 381.36) = 0.0229 m. The slots are those of a driven corridor ahead. Regression caught: a gold from the base
    or the camera, from the 0.1 m cell's near edge (2.0 either way here: the sub-cell is checked by
    `test_corridor_subcell.py`), a range that leaves out the camera's setback, or a resolution that is not the
    camera's."""
    w = world(drive_run(boxes=[FACE.at]), [FACE])
    c = asked(w)
    assert c is not None and c.slice == "answerable" and c.answer_class == "answerable"
    assert c.gold == {"type": "number", "value": 2.0}
    assert c.slots == {"mode": "drive", "direction": "ahead", "cap": 5.0}
    assert (c.range_m, c.scope, c.critical, c.decision_k, c.subjects) == (2.02, "now", True, LAST, (f"w:{LAST}",))
    assert c.tolerances == {"resolution_m": pytest.approx(0.0229, abs=1e-4), "need_m": 0.45, "harm": "over",
                            "threshold_m": 1.25}
    assert set(c.evidence_k) <= set(range(16, 20)) and c.evidence_k


def test_a_corridor_that_was_not_seen_empty_is_an_unknown_candidate():
    """Contract: where the look window has no depth, the corridor was not seen empty: the gold is `unknown`, the
    `unknown` pool's only candidate (a gold, never a dead band), with no evidence and no range, and no screen applies to
    it (there is no stop to screen: the inputs carry no frame here, and nothing raises). Regression caught: an `unknown`
    that is silently dropped as a dead band, or screened against a stopper it has not got."""
    w = world(drive_run(depth_ks=()), None)
    m = clear_ground.moment(w, LAST)
    assert m is not None and m.answerable == [] and len(m.unknown) == 1
    (c,) = m.unknown
    assert (c.gold, c.slice, c.answer_class, c.evidence_k, c.range_m) == ({"type": "unknown"}, "unknown", "unknown", (),
                                                                        None)


def test_nothing_is_asked_outside_the_clear_phases():
    """Contract (review 2.5): the corridor is asked only in the phases of `JobSpec.clear_phases`, the ones in which the
    robot drives; where it sits the status answers and a held-out constant scores. The same scene is asked in `transit`
    (keyframe 19's phase) and not when the job names only `docking`, or none at all (the AMR's table). Regression
    caught: a corridor asked at a station or a dock."""
    script = drive_run(boxes=[FACE.at])
    assert clear_ground.moment(world(script, [FACE]), LAST) is not None
    for phases in (frozenset({"docking"}), frozenset()):
        assert clear_ground.moment(world(script, [FACE], replace(JOB, clear_phases=phases)), LAST) is None


def test_a_gold_the_screens_refuse_is_not_asked(monkeypatch):
    """Contract: an answerable gold is asked only if the screens pass (`corridor_screens.screen`: the stopper, side and
    height; it owns what each means and gives the resolution); a refusal makes the moment None, not an `unknown` and
    not another family's candidate. Regression caught: a screen computed and not consulted."""
    w = world(drive_run(boxes=[FACE.at]), [FACE])
    monkeypatch.setattr(corridor_screens, "screen", lambda *a: None)
    assert clear_ground.moment(w, LAST) is None


def test_a_wall_beyond_the_cap_is_the_cap():
    """Contract: with nothing within the cap the gold is the cap, answerable (a corridor seen empty to its end), and
    needs no stopper (`corridor_screens`: a cap gold needs neither a tie nor a visible stopper), so the point is asked
    though the inputs hold no frame. Hand value: a wall at x 6.5 is 5.5 m from the front edge, past the 5 m cap."""
    far = Owned(box(6.5, 6.7, -3.0, 3.0, 0.0, 2.0))
    c = asked(world(drive_run(boxes=[far.at]), None))
    assert c is not None and c.gold == {"type": "number", "value": 5.0}


@pytest.mark.parametrize("k", [16, 18])
def test_a_point_may_be_any_frame_and_reads_no_frame_after_it(k):
    """Contract (0.10.0, one frame per `observe`): `moment` is asked at any frame, reading the cells up to it and none
    after. Hand case: the wall of `FACE` is shown by every frame from 16 (the look window 16-19), so asked at 16 or at
    18, which end no decision window, the gold is 2.0 and the evidence lies in 16 to k (never 19), the point is k itself
    and the scope `now`. Regression caught: a corridor refused at a mid-window frame, or read from the frames after
    it."""
    c = asked(world(drive_run(boxes=[FACE.at]), [FACE]), k)
    assert c is not None and c.gold == {"type": "number", "value": 2.0}
    assert (c.decision_k, c.scope) == (k, "now") and c.evidence_k and set(c.evidence_k) <= set(range(16, k + 1))
