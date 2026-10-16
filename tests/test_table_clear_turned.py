"""The table systems' answer to a turned `clear_path` (`table_clear`): the gold's own rule over the cells the system's
mapper made of every step it was sent up to the ask, from the edge of the asked move. Cases are `clear_kit`'s closed
room (`room_run`): the tote AMR's camera looked back, left and right before the point, and the look window (keyframes 16
to 19) looks ahead, so the turned corridors are `memory` rows only a system that kept the older frames answers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from answers_kit import compiled_case
from baseline_kit import rows
from clear_kit import JOB, room_run
from space_kit import use_job
from stbench import corridor_screens
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.stream import run_case

EVENTS = dict(JOB.events)
CASE = "c2"   # a case id whose draw asks the point 19 (and 7), found from the looks hash


@pytest.mark.parametrize("direction", ["back", "left", "right"])
def test_the_tables_answer_a_turned_corridor_from_the_cells_they_kept(tmp_path, monkeypatch, direction):
    """Contract (the ceiling): through `run_case` on `detections` the oracle and `memory_table` answer every asked
    turned `clear_path` with the gold, `grid.corridor_clear` over the latest cells of every step up to the ask from the
    edge of the move (back: the rear edge; left and right: the front edge after the turn), where the older frames the
    latest frame does not hold are the evidence. `current_view`, which holds the latest frame alone, gives the gold on a
    `now` row and cannot give a number on a `memory` row. The screens read instance images no kit writes: they are
    stubbed to pass, and `test_corridor_screens.py` holds them. Regression caught: a table that reads the latest frame
    only (the ceiling gate false on every `memory` row), the cells of the wrong edge, or a `memory` row the current view
    answers."""
    monkeypatch.setattr(corridor_screens, "screen", lambda *a: 0.05)
    job = replace(JOB, clear_directions=(direction,))
    use_job(monkeypatch, job)
    case = compiled_case(tmp_path / "ds", CASE, room_run(), job, 0, 0)
    qs, golds = rows(case)
    assert qs and {q["template"] for q in qs} == {"clear_path"} and {q["slots"]["direction"] for q in qs} == {direction}
    assert any(g["scope"] == "memory" and g["gold"]["type"] == "number" for g in golds)
    for cls in (Oracle, MemoryTable):
        run = run_case(lambda state, work, cls=cls: cls(state, work, events=EVENTS), case, channel="detections")
        assert run.status == "complete" and run.failures == {}, cls.__name__
        assert run.answers == {g["qid"]: g["gold"] for g in golds}, cls.__name__
    run = run_case(lambda state, work: CurrentView(state, work, events=EVENTS), case, channel="detections")
    for g in golds:
        if g["scope"] == "now":
            assert run.answers[g["qid"]] == g["gold"]
        elif g["gold"]["type"] == "number":
            assert run.answers[g["qid"]] != g["gold"]
