"""The table systems' answer to a ground `clear_path` (`table_clear`): the gold's own rule over the cells the system's
mapper made of the frame it was sent. Cases are `clear_kit`'s tote AMR standing at (0.6, 0) heading +x for twenty
keyframes in `transit`, its front edge at x 1.0, looking at a wall whose face is at x 3.01 (the sub-cell [3.00, 3.02),
2.01 from the
edge: the corridor stops at 2.0)."""

from __future__ import annotations

from answers_kit import compiled_case
from baseline_kit import UNKNOWN, number, rows
from clear_kit import FRAMES, JOB, drive_run, wall
from space_kit import use_job
from stbench import corridor_screens
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.stream import run_case

EVENTS = dict(JOB.events)


def test_the_tables_answer_the_gold_on_the_depth_they_were_sent(tmp_path, monkeypatch):
    """Contract (the ceiling): through `run_case` on `detections` the oracle, `memory_table` and `current_view` answer
    every asked `clear_path` with the gold: `grid.corridor_clear` over the ask's frame from the front edge at the ask's
    delivered pose. Hand values: the wall's sub-cell centre 3.01 is 2.01 from the front edge, so 2.0 at every look point
    (the base never moves and every keyframe has depth); a case with no depth at all has no corridor seen empty, so
    `unknown`, which the tables answer too. The screens read the instance images no kit writes: they are stubbed to
    pass, and `test_corridor_screens.py` holds it. Regression caught: the table reading other cells than the gold (a
    blind band where the gold has one, the camera or the base for the start), which would fail the ceiling gate for the
    wrong reason."""
    monkeypatch.setattr(corridor_screens, "screen", lambda *a: 0.05)
    use_job(monkeypatch, JOB)
    dataset = tmp_path / "ds"
    cases = {"wall": (compiled_case(dataset, "c0", drive_run(boxes=[wall(3.01).at], depth_ks=range(FRAMES), seed=1),
                                    JOB, 0, 0), number(2.0), "now"),
             "blind": (compiled_case(dataset, "c1", drive_run(depth_ks=(), seed=2), JOB, 0, 0), UNKNOWN, "memory")}
    for name, (case, want, scope) in cases.items():
        qs, golds = rows(case)
        assert qs and {q["template"] for q in qs} == {"clear_path"}, name
        assert all(g["gold"] == want and g["scope"] == scope for g in golds), name
        for cls in (Oracle, MemoryTable, CurrentView):
            run = run_case(lambda state, work, cls=cls: cls(state, work, events=EVENTS), case, channel="detections")
            assert run.status == "complete" and run.failures == {}, (name, cls.__name__)
            assert run.answers == {q["qid"]: want for q in qs}, (name, cls.__name__)
