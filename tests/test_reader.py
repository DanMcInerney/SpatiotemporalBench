"""A reader that keeps nothing between observes scores at the floor on memory questions: the stream does not let
the past reach the present."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, IDS, compiled_case, play, rows, scores
from stbench import descriptions
from stbench.baselines import MemoryTable
from stbench.stream_protocol import System


class StatelessReader(System):
    """Replaces what it knows with each frame's detections, so it answers only about what the last frame showed."""

    structured = True
    channels = ("detections",)

    def observe(self, steps: list[dict]) -> None:
        self.seen = [d for s in steps for d in s["detections"]]

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        out = {}
        for q in questions:
            match = [d for d in self.seen if q["template"] == "object_where"
                     and descriptions.matches(d["kind"], d["attributes"], q["slots"]["kind"], q["slots"]["attributes"])]
            out[q["qid"]] = ({"type": "position", "xy": match[0]["center_map"][:2]} if match else {"type": "unknown"})
        return out


def test_a_stateless_reader_scores_at_the_floor(tmp_path):
    """Contract (plan §4.7): a reader that keeps nothing between observes scores, on `memory` questions, the share
    of them whose gold is `unknown` (it answers `unknown` to the rest, for 0), because everything an end question
    asks about was last seen before the end look window. Hand case (the second of `baseline_kit.IDS`, whose draw asks
    an unknown gold): the compiled case asks two `object_where` questions, one of them an `unknown`
    gold, and the end look window (keyframes 22-27) shows only green shoes that the compiler never asks about, so the
    reader scores (1 + 0) / 2 = 1/2, where `memory_table` scores 1. Regression caught: the stream
    letting the past reach the present (an earlier frame's detections or files left in the inbox), which would lift
    this reader above the floor and every memoryless gate with it."""
    case = compiled_case(tmp_path, IDS[1])
    _, gold = rows(case)
    memory = [g["qid"] for g in gold if g["scope"] == "memory"]
    reader, table = scores(case, play(case, StatelessReader)[0]), scores(case, play(case, MemoryTable,
                                                                                     events=EVENTS)[0])
    assert len(memory) == 2 and sum(g["slice"] == "unknown" for g in gold if g["scope"] == "memory") == 1
    assert sum(reader[q] for q in memory) / 2 == pytest.approx(1 / 2)
    assert sum(table[q] for q in memory) / 2 == 1.0
