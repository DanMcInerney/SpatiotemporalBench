"""The table systems answer `open_heading` with the gold, through the real compile and the stream (the ceiling): the
oracle and `memory_table` give every asked gold's heading, `current_view` gives it wherever the latest frame alone is
enough. `clear_kit`'s tote AMR stands at (0.6, 0) heading +x for twenty keyframes in `transit`, looking at a wall
whose face is at x 4.51 across 6 m (a sub-cell centre at 4.51, 3.51 m from the front edge: 3.5 ahead), with depth in
every keyframe. The wall is far enough that the 0.53 m the camera cannot see in front of the body is under a fifth of
the corridor to it, so the corridor is seen empty (a nearer wall would leave it `unknown`: the profile counts every
cell)."""

from __future__ import annotations

from dataclasses import replace

from answers_kit import compiled_case
from baseline_kit import rows
from clear_kit import FRAMES, JOB, drive_run, wall
from space_kit import use_job
from stbench import corridor_screens
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.stream import run_case

HEADING_JOB = replace(JOB, looks=("open_heading",))
EVENTS = dict(JOB.events)


def test_the_tables_answer_the_gold_heading_on_the_depth_they_were_sent(tmp_path, monkeypatch):
    """Contract (the ceiling): through `run_case` on `detections` the oracle and `memory_table` answer every asked
    `open_heading` with the gold row's heading: the profile over the cells of everything sent up to the ask, its best
    heading nearest ahead, at the ask's delivered pose; `current_view`, whose store is the latest frame, answers the
    same wherever the row is `now` (the frame alone gives the gold). Hand values: the base never moves and every
    keyframe has depth, so the row's profile runs 3.5 ahead (the wall's 4.51 less the front edge at 1.0 and half a
    sub-cell) at every look point. The stopper screen needs the instance images no kit writes: it is stubbed
    tied and visible, and `test_corridor_screens.py` holds it. Regression caught: a table reading other cells than the
    gold (the frame for a memory table, the camera for the base), another best heading, or a heading in another frame,
    which would fail the ceiling gate for the wrong reason."""
    monkeypatch.setattr(corridor_screens, "stopper", lambda *a: corridor_screens.NO_STOP)
    use_job(monkeypatch, HEADING_JOB)
    case = compiled_case(tmp_path / "ds", "c0", drive_run(boxes=[wall(4.51).at], depth_ks=range(FRAMES), seed=1),
                         HEADING_JOB, 0, 0)
    qs, golds = rows(case)
    assert qs and {q["template"] for q in qs} == {"open_heading"}
    assert all(g["gold"]["type"] == "heading" and g["tolerances"]["profile_m"][0] == 3.5 for g in golds)
    assert {g["scope"] for g in golds} >= {"now"}   # the current view is asked something it can answer
    for cls in (Oracle, MemoryTable, CurrentView):
        run = run_case(lambda state, work, cls=cls: cls(state, work, events=EVENTS), case, channel="detections")
        assert run.status == "complete" and run.failures == {}, cls.__name__
        for g in golds:
            if cls is CurrentView and g["scope"] != "now":
                continue
            assert run.answers[g["qid"]] == g["gold"], (cls.__name__, g["scope"])
