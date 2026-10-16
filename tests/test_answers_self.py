"""The answerers of `where_was_i` (`table_self` for the table systems, `baselines_path_self` for `path_log`) on
`self_kit`'s tasks: 96 steps 1.5 s apart, the base of step `o` of task `t` at (0.5 + 0.05 o, 0.5 + 0.1 t), decision
points at offsets 40 and 70 of each task. The compiled window is tasks 0 and 1 as the case "c0": four during questions
asked at the script's keyframes 40, 70, 136 and 166, each naming a step by the gold's own draw. Expected positions are
worked out from the script's formula; the question's clock is the case clock the stream delivered."""

from __future__ import annotations

from pathlib import Path

import pytest

from answers_kit import compiled_case, scored
from baseline_kit import UNKNOWN, at, question, rows, started, step
from self_kit import JOB, TASK_FRAMES, self_script
from space_kit import EVENTS
from stbench import mapspec, table_self
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.baselines_path_self import where_was_i as path_where_was_i
from stbench.stream import run_case
from stbench.table import Store
from stream_kit import clock


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    """Tasks 0 and 1 of `self_script(2, decisions=(40, 70))` as the case "c0" with its questions and gold compiled:
    `(dataset, case, questions, golds)`."""
    dataset = tmp_path_factory.mktemp("self")
    case = compiled_case(dataset, "c0", self_script(2, decisions=(40, 70)), JOB, 0, 1)
    return (dataset, case, *rows(case))


def base_at(k: int) -> dict:
    """The answer by hand for script keyframe `k` (also the case's step `k`: the window starts at 0)."""
    task, offset = divmod(k, TASK_FRAMES)
    return at(round(0.5 + 0.05 * offset, 4), round(0.5 + 0.1 * task, 4))


def named(golds: list[dict]) -> list[int]:
    """The step each gold row names."""
    return [g["evidence_k"][0] for g in golds]


def test_the_oracle_answers_where_was_i(window):
    """Contract (E3b §1, §3): through `run_case` on `detections`, the oracle and `memory_table` give the gold on every
    `where_was_i` question of the window and the answer is the named step's base worked out from the script. By hand:
    decision 40 (60 s in) can only name step 0, (0.5, 0.5); the questions asked in the second task name steps spread
    over both tasks, so one of them reaches back into the first task's log (a step below 96, asked at keyframe 136 or
    166), which a reader of the last task's steps alone would miss. Regression caught: an answerer missing or wrong,
    seen only as a ceiling failure of the memory-needed gate, or one that reads the current task and not the log."""
    _, case, qs, golds = window
    assert [q["decision_k"] for q in qs] == [40, 70, 136, 166] and named(golds)[0] == 0
    assert any(k < TASK_FRAMES <= q["decision_k"] for q, k in zip(qs, named(golds), strict=True))
    for cls in (Oracle, MemoryTable):
        run = run_case(lambda state, work, cls=cls: cls(state, work, events=EVENTS), case, channel="detections")
        assert run.status == "complete" and run.failures == {}, cls.__name__
        assert scored(golds, run.answers) == dict.fromkeys((q["qid"] for q in qs), 1.0), cls.__name__
        assert [run.answers[q["qid"]] for q in qs] == [base_at(k) for k in named(golds)], cls.__name__


def test_current_view_answers_where_the_look_window_began(tmp_path):
    """Contract (E3b §1): `current_view`'s store holds the latest look window alone, and the step a clock names is a
    minute or more before it, so its `where_was_i` is the look window's first base: the memoryless guess the
    memory-needed gate must hold under the floor, where `memory_table` fed the same steps answers the named step. Hand
    case: steps 0-3 at (1.0 + 0.5 k, 1.0) and then steps 4-7 at (5.0 + 0.5 (k - 4), 2.0), one second apart from
    08:00:00; asked at step 7 for the clock 08:00:01 (step 1), `memory_table` says (1.5, 1.0) and `current_view` (5.0,
    2.0), the first base of the look window it holds. Regression caught: a memoryless baseline weaker than a memoryless
    agent (`unknown`, or a base that is not where the look window began) or a memory system that forgets."""
    earlier = [step(k, 1.0 + 0.5 * k, 1.0) for k in range(4)]
    window = [step(k, 5.0 + 0.5 * (k - 4), 2.0) for k in range(4, 8)]
    asked = question("where_was_i", {"clock": "2026-03-02T08:00:01"})
    now = {"k": 7, "clock": "2026-03-02T08:00:07"}
    answers = {}
    for cls in (CurrentView, MemoryTable):
        system = started(cls, tmp_path / cls.__name__, events=EVENTS)
        system.observe(earlier)
        system.observe(window)
        answers[cls.__name__] = system.ask("during", now, [asked])["q1"]
    assert answers == {"CurrentView": at(5.0, 2.0), "MemoryTable": at(1.5, 1.0)}


def test_other_case_asks_where_it_was_then(window):
    """Contract (E3b §1 with E3a's `clock_map`): fed another case's stream, `other_case` (`memory_table` with the maps)
    answers the other's base at the translated clock, the same moment of the stream it was fed. Hand case: the other
    script is `self_script` with every base moved (x = 3.0 + 0.02 o, y = 2.0 + 0.1 t) and its case "c1" starts at
    another clock; each own clock goes to the other's clock for the same case time (1.5 s a step, 7200 s a task,
    rounded half up) with the tasks mapped to themselves, so the answer to each question is the other's base at the step
    the own gold names, (3.0 + 0.02 o, 2.0 + 0.1 t) worked out here. Regression caught: an `other_case` that never
    finds the clock (its own clock, which the other stream never delivered, answers an end of the stream)."""
    dataset, case, qs, golds = window
    bases = {k: (round(3.0 + 0.02 * (k % TASK_FRAMES), 3), round(2.0 + 0.1 * (k // TASK_FRAMES), 3))
             for k in range(2 * TASK_FRAMES)}
    # another seed: the index gives equal script entries (same seed and task count) one id, and the second write would
    # overwrite the first script's files
    other = compiled_case(dataset, "c1", self_script(2, decisions=(40, 70), seed=2, poses=bases), JOB, 0, 1)

    def other_clock(k: int) -> str:
        task, offset = divmod(k, TASK_FRAMES)
        return clock(other, int(1.5 * offset + 7200 * task + 0.5))

    clock_map = {q["slots"]["clock"]: other_clock(k) for q, k in zip(qs, named(golds), strict=True)}
    assert all(own != theirs for own, theirs in clock_map.items())   # the two streams' clocks are not the same clocks
    ask_map = {q["decision_k"]: q["decision_k"] for q in qs}   # the other has its decision points at the same keyframes
    run = run_case(lambda state, work: MemoryTable(state, work, events=EVENTS, clock_map=clock_map,
                                                   task_map={0: 0, 1: 1}),
                   case, channel="detections", stream_from=other, ask_map=ask_map)
    assert run.status == "complete" and run.failures == {}
    assert [run.answers[q["qid"]] for q in qs] == [at(*bases[k]) for k in named(golds)]


def test_path_log_is_the_reference_on_where_was_i(window):
    """Contract (E3b §2, the family's `reference` is `path_log`): `path_log` gives the gold on every question of the
    window from the pose log alone, with the pictures, depth and detections taken from every step it is sent (the
    steps the `detections` channel delivers carry the last two; both are dropped and counted here, so the strip is not
    vacuous), and the answer is the named step's base worked out from the script. Regression caught: a reference that
    sees, answering from a picture, or one that cannot take the log of two tasks."""
    _, case, qs, golds = window
    seen = set()

    class Blindfolded(PathLog):
        def observe(self, steps):
            seen.update(key for s in steps for key in ("rgb", "depth", "detections") if key in s)
            super().observe([{k: v for k, v in s.items() if k not in ("rgb", "depth", "detections")} for s in steps])

    run = run_case(lambda state, work: Blindfolded(state, work, map=mapspec.load_map("apartment_a"), events=EVENTS,
                                                   prior=[]), case, channel="detections")
    assert seen == {"depth", "detections"}
    assert run.status == "complete" and run.failures == {}
    assert scored(golds, run.answers) == dict.fromkeys((q["qid"] for q in qs), 1.0)
    assert [run.answers[q["qid"]] for q in qs] == [base_at(k) for k in named(golds)]


def test_a_log_with_no_step_answers_unknown(tmp_path):
    """Contract (E3b §1): a system that has been sent no step has no base to name and answers `unknown`, the table's
    store and `path_log`'s log alike, and neither raises. Hand case: the table's empty `Store` and a fresh `path_log`
    asked for 08:00:01 (a question the harness never puts to an empty log, so the answerers are called directly).
    Regression caught: an answerer that raises on an empty log, or a position at the origin for a step never seen."""
    clock_slot = {"clock": "2026-03-02T08:00:01"}
    assert table_self.where_was_i(Store(), clock_slot, {}) == UNKNOWN
    system = started(PathLog, Path(tmp_path), map=mapspec.load_map("apartment_a"), events=EVENTS)
    assert path_where_was_i(system, question("where_was_i", clock_slot), {}) == UNKNOWN
