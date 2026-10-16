"""The answer registries and what an answer reads across tasks (plan §5.1, E3a): the family modules found through the
pre-seeded lists, the hand 3-task case of `answers_kit` answered by every system, and the task a question names.
Expected values are worked out by hand in `answers_kit` and below."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

import stbench
from answers_kit import BAG, BOX, compiled_case, scored, three_task_script, three_tasks, where
from baseline_kit import EVENTS as VACUUM_EVENTS
from baseline_kit import UNKNOWN, at, record, rows, started, step
from baseline_kit import where as hand_where
from space_kit import EVENTS, JOB
from stbench import answer_registry, baselines_path, mapspec, table_answers
from stbench.baselines import BASELINES, Oracle
from stbench.baselines_path import PathLog
from stbench.case import read_case
from stbench.stream import run_case


def fake_module(name: str, **attrs) -> types.ModuleType:
    """A module `stbench.<name>` holding `attrs`, to put in `sys.modules`."""
    module = types.ModuleType(f"stbench.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def table_spy(store, slots, ctx):
    return {"type": "count", "value": 7}


def path_spy(pl, question, rec):
    return {"type": "count", "value": len(pl.log.steps)}


def test_a_missing_answer_module_is_skipped(monkeypatch, tmp_path):
    """Contract (E3a §1): a name of `ANSWER_MODULES` or `PATH_MODULES` whose module does not exist adds nothing (the
    later milestones' names cost nothing until written); a module that exists adds its families, to the table systems
    and bound to `path_log`'s instance; a family answered twice raises, naming it and both claimants, `object_where`
    and `path_ahead` (the built-ins) included; a module that fails to import for another reason is raised, not
    skipped. Hand case: `table_fake` and `baselines_path_fake` answer `ghost`; the missing `table_gone` beside them
    changes nothing; `table_other` answering `ghost` too raises 'ghost' naming `table_fake` and `table_other`;
    `table_claims` answering `object_where` and a path module answering it name the built-in; a module file that
    imports a package that does not exist raises that error. Regression caught: a family lane's answerer unregistered,
    two lanes claiming one family, or a missing dependency hidden as a missing module."""
    plain = table_answers.BUILT_IN
    assert set(plain) == {"path_ahead", "object_where"} <= set(table_answers.ANSWERERS)
    ghost, claims = {"ghost": table_spy}, {"object_where": table_spy}
    for name, attrs in (("table_fake", {"ANSWERERS": ghost}), ("table_other", {"ANSWERERS": ghost}),
                        ("table_claims", {"ANSWERERS": claims}), ("table_bare", {}),
                        ("baselines_path_fake", {"PATH_ANSWERERS": {"ghost": path_spy}}),
                        ("baselines_path_claims", {"PATH_ANSWERERS": {"object_where": path_spy}})):
        monkeypatch.setitem(sys.modules, f"stbench.{name}", fake_module(name, **attrs))
    assert table_answers.registry(("table_gone",)) == plain
    assert table_answers.registry(("table_gone", "table_fake")) == {**plain, "ghost": table_spy}
    assert baselines_path.path_registry(("baselines_path_gone",)) == {}
    assert baselines_path.path_registry(("baselines_path_gone", "baselines_path_fake")) == {"ghost": path_spy}
    with pytest.raises(ValueError, match="'ghost' is answered twice: by table_fake and by table_other"):
        table_answers.registry(("table_fake", "table_other"))
    twice = "'object_where' is answered twice: by a built-in answerer and by "
    with pytest.raises(ValueError, match=twice + "table_claims"):
        table_answers.registry(("table_claims",))
    with pytest.raises(ValueError, match=twice + "baselines_path_claims"):
        baselines_path.path_registry(("baselines_path_claims",))
    with pytest.raises(ValueError, match="table_bare defines no ANSWERERS"):
        table_answers.registry(("table_bare",))
    (tmp_path / "table_broken.py").write_text("import a_package_that_does_not_exist\n", encoding="utf-8")
    monkeypatch.setattr(stbench, "__path__", [*stbench.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    with pytest.raises(ModuleNotFoundError, match="a_package_that_does_not_exist"):
        answer_registry.collect(("table_broken",), "ANSWERERS")
    monkeypatch.setattr(baselines_path, "PATH_ANSWERERS", baselines_path.path_registry(("baselines_path_fake",)))
    system = PathLog(Path("."), Path("."), map=None, events={})
    system.log.add(step(0, 0.0, 0.0))
    assert system.answerers["ghost"]({}, {}) == {"type": "count", "value": 1}
    assert set(system.answerers) == {"object_where", "ghost"}


def test_the_two_registries_pair_module_by_module():
    """Contract (E3a §1): `ANSWER_MODULES` and `PATH_MODULES` hold the same eighteen group names (0.10.0 adds
    `approach` and `hazard`, FD-questions-0.10 step 0; 0.11 adds `way`), in the same order,
    `table_` and `baselines_path_` before each, which pair with C2a's family modules (`questions_<group>`;
    `object_where`'s `questions_where` is the built-in's), so a family lane adds one file to each side and edits no
    list. Regression caught: a name on one side only, so a family is answered by the table systems and not `path_log`
    (or the reverse), which `test_families_complete` finds only once the family exists."""
    groups = ["self", "space", "cover", "view", "changed", "things", "route", "zone", "adjacent", "change", "ego",
              "task", "clear", "people", "heading", "approach", "hazard", "way"]
    assert [m.removeprefix("table_") for m in table_answers.ANSWER_MODULES] == groups
    assert [m.removeprefix("baselines_path_") for m in baselines_path.PATH_MODULES] == groups


# -- a window of several tasks --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    """The hand 3-task case, its question rows and gold rows, and the oracle run over it: `(case, questions, golds, run,
    system)`."""
    case, qs, golds = three_tasks(tmp_path_factory.mktemp("hand"))
    built = []

    def make(state, work):
        built.append(Oracle(state, work, events=EVENTS))
        return built[0]

    return case, qs, golds, run_case(make, case, channel="detections"), built[0]


def test_every_system_runs_the_multi_task_case(hand):
    """Contract (E3a exit): through `run_case` on `detections`, the oracle scores 1 on every question of the hand 3-task
    case (the bag at (2, 1), (5, 1), (1, 2) at the end of tasks 0, 1 and 2, the box at (3.5, 2) in task 1), and every
    other system of `BASELINES`, given the options its flags say, completes it with an answer to each question.
    Regression caught: a system that cannot take a window of several tasks (a task message it drops, an event resolved
    over the last task alone), or an oracle that is not exact once the world changes between tasks."""
    case, qs, golds, run, _ = hand
    assert run.status == "complete" and run.failures == {}
    assert scored(golds, run.answers) == dict.fromkeys((q["qid"] for q in qs), 1.0)
    for name, b in BASELINES.items():
        opts = {key: value for key, value, wanted in (("map", mapspec.load_map("apartment_a"), b.needs_map),
                                                      ("events", EVENTS, b.needs_events),
                                                      ("prior", [], b.needs_prior)) if wanted}
        other = {"stream_from": case, "ask_map": {}} if b.other_stream else {}
        got = run_case(lambda state, work, cls=b.cls, opts=opts: cls(state, work, **opts), case, channel="detections",
                       **other)
        assert got.status == "complete" and got.failures == {}, name
        assert set(got.answers) == {q["qid"] for q in qs}, name


def with_task(question: dict, task: int | None) -> dict:
    """The question naming the case task `task` (None: the window's only task)."""
    return {**question, "slots": {**question["slots"], "task": task}}


def test_object_where_reads_the_named_task(hand, tmp_path):
    """Contract (E3a §2, the task slot): the table and `path_log` resolve an `object_where` event over the steps of the
    task `slots["task"]` names (the last when None; a task the stream never had resolves to nothing, so `unknown`, never
    a raise). The table, hand case of `answers_kit`: the bag is at (2, 1) after task 0, (5, 1) after task 1 and (1, 2)
    after task 2, which a reader of the last task alone would answer for every task; the box is unseen at the end of
    task 0 (its one sighting, keyframe 40, is later) and at (3.5, 2) from task 1; task 4, which the window lacks, is
    `unknown`. `path_log`, three tasks of six steps at (0.5, 1.9) with the camera facing the kitchen's dense place
    (2, 2) in task 2 only, its brief naming the kitchen and five other scripts' golds there (asked about tasks 0 to 4):
    it answers (2, 2) for task 2 and for None (the last task), and `unknown` for task 0, task 1 (its camera has not
    faced the place by the step the event resolves to there, keyframe 9) and task 4. Regression caught: the table and
    the gold reading different tasks (a ceiling failure), or `path_log` resolving every event over the last task."""
    _, _, _, _, oracle = hand
    now = {"k": 83, "clock": ""}

    def table(task, kind=BAG, color="red"):
        return oracle.answer_at(where(9, kind, color, task), now)

    assert [table(task) for task in (0, 1, 2, None)] == [at(2.0, 1.0), at(5.0, 1.0), at(1.0, 2.0), at(1.0, 2.0)]
    assert [table(task, BOX, "blue") for task in (0, 1, 4)] == [UNKNOWN, at(3.5, 2.0), UNKNOWN]
    assert table(4) == UNKNOWN

    brown = hand_where("box", "brown", "when you started cleaning")
    prior = [record(with_task(brown, i), at(2.0, 2.0), script=f"s{i}", rooms=("kitchen",)) for i in range(5)]
    system = started(PathLog, tmp_path, map=mapspec.load_map("apartment_a"), events=VACUUM_EVENTS, prior=prior)
    phases = ["docked", "undocking", "undocking", "cleaning", "cleaning", "docking"]
    for task in range(3):
        system.task(task, "2026-03-02T08:00:00", 0.0, "Its current assignment is to clean the kitchen.")
        system.observe([step(6 * task + i, 0.5, 1.9, 0.0 if task == 2 else 180.0, phase=phase)
                        for i, phase in enumerate(phases)])
    path = [system.ask("end", {"k": 17, "clock": ""}, [with_task(brown, task)])["q1"] for task in (0, 1, 2, None, 4)]
    assert path == [UNKNOWN, UNKNOWN, at(2.0, 2.0), at(2.0, 2.0), UNKNOWN]


def test_a_compiled_case_names_its_length_and_id(tmp_path):
    """Contract (E3a's `answers_kit.compiled_case`, the review's m7): tasks a..b of a script are written as a case of
    the fixed id given, of the length `case.LENGTHS` gives their count (1 short, 2-3 medium, 4-6 long), and compiled
    with the job given. Hand case: the hand 3-task script's tasks 1..2 are a medium window, and the case is under the
    id given with its questions and gold written (one gold row to each question). Regression caught: a family lane's
    oracle test over a window whose length, id, tasks or job is not the one it asked for."""
    case = compiled_case(tmp_path, "00000000000000a1", three_task_script(), JOB, 1, 2)
    record_ = read_case(case)
    assert (case.id, record_.length, record_.a, record_.b) == ("00000000000000a1", "medium", 1, 2)
    qs, golds = rows(case)
    assert qs and [g["qid"] for g in golds] == [q["qid"] for q in qs]
