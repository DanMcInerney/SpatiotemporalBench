"""Helpers for the tests of what `stbench run` scores: `stream_kit`'s synthetic cases, with the gold file a run is
scored against. `golds` writes one answerable gold row a question (`stream_kit.gold_row`: a number 1.4 for a during
question, a position (3.2, 1.5) for an end one) unless the test gives its own."""

from __future__ import annotations

from pathlib import Path

from stbench import case as case_mod
from stbench import episode as ep
from stream_kit import QUESTIONS, case_on, gold_row, scripted_case
from synth import Synth


def golds(case: case_mod.CasePaths, questions: list[dict] = QUESTIONS, rows: list[dict] | None = None) -> None:
    """`case`'s `gold.jsonl`: the rows given, else one `gold_row` a question."""
    ep.write_jsonl(case.gold, rows if rows is not None else [gold_row(q) for q in questions])


def scored_case(dataset: Path, synth: Synth, questions: list[dict], **kw) -> case_mod.CasePaths:
    """`stream_kit.case_on`, then a gold row for each question."""
    case = case_on(dataset, synth, questions, **kw)
    golds(case, questions)
    return case


def scored_scripted(tmp_path: Path) -> tuple[case_mod.CasePaths, Path]:
    """`stream_kit.scripted_case`, with a gold row for each of its questions."""
    case, file = scripted_case(tmp_path)
    golds(case)
    return case, file
