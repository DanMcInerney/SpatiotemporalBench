"""Helpers for the tests of the priors, `stbench check` and the form adversary.

`synthetic_dataset` is `space_kit.things_run`'s script a few times over, each with a case compiled by the registered
`vacuum/1` under a fixed id (so the questions it draws repeat): the worlds are alike, so every prior's key finds its
records and the oracle is exact on all of them. `empty_case` is a compiled case that asks nothing (the compiler's
output for a window in which no question could be drawn). `hand_synth` and `hand_case` make scripts of 20-step kitchen
tasks with hand-written questions and gold (no pictures are needed: nothing runs over them), for the numbers a record,
an option or a report must give by hand. `asked_records` plays a prior system over a case and keeps the record it
builds at each ask.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from baseline_kit import compiled_case, play
from run_kit import golds
from space_kit import things_run
from stbench import case as case_mod
from stbench import episode as ep
from stbench.case import CasePaths
from stbench.stream_protocol import System
from stream_kit import UNKNOWN, case_on, during_q, end_q, gold_row
from synth import HOME, Stage, Synth

EVENT = "when you left the dock"   # one of `vacuum/1`'s named moments
TASK_STEPS = 20


def synthetic_dataset(root: Path, n: int) -> list[CasePaths]:
    """`n` cases of `things_run` scripts (seeds 1 to `n`), compiled, the case ids 1 to `n` as 16 hex digits."""
    return [compiled_case(root, f"{i:016x}", things_run(seed=i)) for i in range(1, n + 1)]


def empty_case(root: Path, script=None, cid: str = "00000000000000ee") -> CasePaths:
    """A case of `script` (default: a `things_run` one, of the layout and job of `synthetic_dataset`'s) under the id
    `cid`, whose `questions.jsonl` and `gold.jsonl` exist and are empty, as `questions.write` leaves a window that drew
    no question."""
    with mock.patch.object(case_mod, "claim_id", lambda dataset, entry: cid):
        case = (script or things_run(seed=99)).write_case(root)
    golds(case, rows=[])
    ep.write_jsonl(case.questions, [])
    return case


def asked_records(case: CasePaths, cls: type[System], **opts) -> dict[str, dict]:
    """`cls(**opts)` over the case on the gate channel: `{qid: the record its `ask_record` built when asked}`."""
    seen: dict[str, dict] = {}

    class Spy(cls):
        def ask(self, when, now, questions):
            for q in questions:
                seen[q["qid"]] = self.ask_record(q, now, when=when)
            return super().ask(when, now, questions)

    play(case, Spy, **opts)
    return seen


def hand_synth(seed: int, *, tasks: int = 1, room: str = HOME.rooms[0], stage: Stage = HOME) -> Synth:
    """A script of `tasks` tasks of `TASK_STEPS` steps in `room`, `before_doorway` decision points at steps 3 and 12 of
    each task."""
    return Synth(tasks=[[(room, TASK_STEPS)] for _ in range(tasks)], seed=seed, stage=stage,
                 decisions=[(TASK_STEPS * t + k, "before_doorway") for t in range(tasks) for k in (3, 12)])


def hand_case(root: Path, synth: Synth, a: int = 0, *, cap: float = 2, color: str = "red", event: str = EVENT,
              unknown: bool = False) -> CasePaths:
    """A case over task `a` of the script, asking `path_ahead` after its step 12 (slot `cap`) and, at the end, where a
    mug of `color` was `event`. The golds are hand-written: 1.4 m, and the position (3.2, 1.5) or, with `unknown`,
    `unknown`."""
    qs = [during_q(1, TASK_STEPS * a + 12, slots={"cap": cap}),
          end_q(2, slots={"kind": "mug", "attributes": {"color": color}, "event": event})]
    case = case_on(root, synth, qs, a=a)
    end = {"gold": UNKNOWN, "slice": "unknown"} if unknown else {}
    golds(case, rows=[gold_row(qs[0]), gold_row(qs[1], **end)])
    return case
