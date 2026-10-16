"""The suite runner (`suite.py`, plan §4.4's B.2): `run_suite` over cases with injected options and waves, the spawn
pool, and the scoring of a run into rows. The systems are fakes registered in `baselines.BASELINES` (the parent resolves
a name, and the pool is sent the class, which pickles by reference); the cases are `stream_kit`'s synthetic ones."""

from __future__ import annotations

import multiprocessing
import pickle
from types import SimpleNamespace

import pytest

from run_kit import golds, scored_case
from stbench import baselines, job_registry
from stbench.mapspec import load_map
from stbench.stream import CaseRun
from stbench.stream_protocol import System
from stbench.suite import RunOptions, case_info, cases_like, run_suite, score_rows
from stream_kit import QUESTIONS, Recorder, case_on, during_q, end_q, gold_row, hand20, qid
from synth import WAREHOUSE, Synth

CHANNEL = "rgb_pose"
UNKNOWN = {"type": "unknown"}


class Lengths(System):
    """Answers each question with a number: the length of its text plus the run's `bias` option."""

    def observe(self, steps):
        pass

    def ask(self, when, now, questions):
        return {q["qid"]: {"type": "number", "value": float(len(q["text"]) + self.opts.get("bias", 0))}
                for q in questions}


class Steps(System):
    """Answers each question with a number: the steps it has been shown plus the run's `bias` option."""

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.seen = 0

    def observe(self, steps):
        self.seen += len(steps)

    def ask(self, when, now, questions):
        return {q["qid"]: {"type": "number", "value": float(self.seen + self.opts.get("bias", 0))} for q in questions}


class Notes(System):
    """Answers `unknown`, and notes in the run's `ran` list, when it is built and when it finishes, its `tag`."""

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        opts.get("built", []).append(opts.get("tag"))

    def observe(self, steps):
        pass

    def ask(self, when, now, questions):
        return {q["qid"]: dict(UNKNOWN) for q in questions}

    def finish(self):
        self.opts["ran"].append(self.opts["tag"])
        return {}


class Narrow(System):
    """A system that runs on the detections channel only: `run_case` refuses it on any other."""

    channels = ("detections",)


@pytest.fixture
def fakes(monkeypatch):
    for name, cls in (("lengths", Lengths), ("steps", Steps), ("notes", Notes), ("narrow", Narrow), ("rec", Recorder)):
        monkeypatch.setitem(baselines.BASELINES, name, SimpleNamespace(cls=cls))


def questions_of(n: int) -> list[dict]:
    """Case n's (1-based) three questions: texts 3n, 4n and 5n characters long, so a system that answers by length
    answers each case differently."""
    return [during_q(10 * n + 1, 12, text="d" * 3 * n), end_q(10 * n + 2, text="e" * 4 * n),
            end_q(10 * n + 3, text="f" * 5 * n)]


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """Three scripts of the 20-step hand case, a case each holding case n's questions: read-only, so the tests of the
    module share them."""
    out = []
    for n in (1, 2, 3):
        synth = hand20()
        synth.seed = n
        out.append(case_on(tmp_path_factory.mktemp("suite"), synth, questions_of(n)))
    return out


def values(runs: dict[str, list[CaseRun]]) -> dict:
    """What a run gives that does not depend on the clock or on a random session token: per system and case, the
    numbers it answered by question, its status and its failures."""
    return {name: [({q: a["value"] for q, a in run.answers.items()}, run.status, run.failures) for run in cases]
            for name, cases in runs.items()}


def feed(name, case, cases, done):
    """`lengths` is given its case's place as `bias`; `steps` (the second wave) the number `lengths` answered to the
    case's first end question, from `done`: an options function that needs the earlier wave's runs."""
    place = cases.index(case) + 1
    if name == "lengths":
        return RunOptions({"bias": place - 1})
    return RunOptions({"bias": int(done["lengths", case.id].answers[qid(10 * place + 2)]["value"])})


def test_the_pool_equals_serial(cases, fakes):
    """Contract (plan §4.4): `run_suite` over three cases and two systems in two waves, the second wave's options built
    from the first's answers, gives the same answers, statuses and failures, in the same order, with two worker
    processes as with one; the systems come in wave order, the logs give a line a run, and no worker is left. The
    serial result is checked against numbers worked by hand (a system's answer is the text length plus its bias; the
    other's the steps seen plus the first's end answer). Regression caught: a worker reordering rows (here every case's
    answers differ), options or class lost on the way to a spawned process, or a second wave built without the first's
    answers."""
    waves = [["lengths"], ["steps"]]
    logs, before = [], set(multiprocessing.active_children())
    serial = run_suite(cases, ["lengths", "steps"], CHANNEL, options=feed, waves=waves)
    pooled = run_suite(cases, ["lengths", "steps"], CHANNEL, options=feed, waves=waves, workers=2, log=logs.append)
    assert set(multiprocessing.active_children()) == before
    hand = {  # per case: the length plus the place less one (3, 4, 5 for case 1); the steps seen plus that end answer
        "lengths": [{qid(11): 3.0, qid(12): 4.0, qid(13): 5.0}, {qid(21): 7.0, qid(22): 9.0, qid(23): 11.0},
                    {qid(31): 11.0, qid(32): 14.0, qid(33): 17.0}],
        "steps": [{qid(11): 17.0, qid(12): 24.0, qid(13): 24.0}, {qid(21): 22.0, qid(22): 29.0, qid(23): 29.0},
                  {qid(31): 27.0, qid(32): 34.0, qid(33): 34.0}]}  # 13 steps seen at the during ask, 20 at the end
    assert values(serial) == {name: [(answers, "complete", {}) for answers in rows] for name, rows in hand.items()}
    assert values(pooled) == values(serial) and list(pooled) == ["lengths", "steps"]
    assert sorted(m.split(":")[0] for m in logs) == sorted(f"{name} {c.id}" for name in hand for c in cases)


def test_waves_feed_the_options(cases, monkeypatch):
    """Contract (plan §4.4): with two waves and an `OptionsFn` that records its calls, every call of the second wave
    sees every run of the first (by system and case id) in `done` and every case of the run in `cases`, no options are
    built for the second wave before the first wave has finished, and the first wave's calls (all of them, built before
    its first run starts) see an empty `done`. Regression caught: a prior keyed on answers that do not exist yet, which
    falls to its template level."""
    ran, calls = [], []

    def options(name, case, all_cases, done):
        calls.append({"name": name, "case": case.id, "cases": [c.id for c in all_cases], "ran": list(ran),
                      "done": {key: run.status for key, run in done.items()}})
        return RunOptions({"ran": ran, "tag": name})

    for name in ("first", "second"):
        monkeypatch.setitem(baselines.BASELINES, name, SimpleNamespace(cls=Notes))
    run_suite(cases, ["first", "second"], CHANNEL, options=options, waves=[["first"], ["second"]])
    ids = [c.id for c in cases]
    assert [(c["name"], c["case"]) for c in calls] == [(n, i) for n in ("first", "second") for i in ids]
    assert all(c["cases"] == ids for c in calls)
    first, second = calls[:3], calls[3:]
    assert all(c["done"] == {} and c["ran"] == [] for c in first)
    assert all(c["done"] == {("first", i): "complete" for i in ids} and c["ran"] == ["first"] * 3 for c in second)


def test_options_stream_from_and_ask_map_reach_the_run(tmp_path, fakes):
    """Contract (plan §4.4): the `stream_from` and `ask_map` of a `RunOptions` reach `run_case`: the system is fed the
    other case's stream (here a script whose first step stands elsewhere) and the case's during question is asked at
    the keyframe the map names; a case given neither is fed its own and asked at its own decision point. Regression
    caught: an `other_case` run silently fed its own stream, so a gate that depends on the stream proves nothing."""
    other = hand20()
    other.seed, other.poses = 2, {0: (3.0, 2.0)}  # a script whose first step stands elsewhere (and has its own id)
    fed, plain, stream = (case_on(tmp_path, synth, QUESTIONS) for synth in (_seeded(1), _seeded(3), other))
    logs = {fed.id: [], plain.id: []}

    def options(name, case, cases, done):
        given = {"stream_from": stream, "ask_map": {12: 3}} if case.id == fed.id else {}
        return RunOptions({"log": logs[case.id]}, **given)

    run_suite([fed, plain], ["rec"], CHANNEL, options=options)

    def first_step(log):
        return next(entry for entry in log if entry[0] == "observe")[1][0]["pose"]["pos"][:2]

    def asked_at(log):
        return [entry[2]["k"] for entry in log if entry[0] == "ask" and entry[1] == "during"]

    assert (first_step(logs[fed.id]), asked_at(logs[fed.id])) == ([3.0, 2.0], [3])  # the other stream, asked at its k 3
    assert (first_step(logs[plain.id]), asked_at(logs[plain.id])) == ([2.0, 1.5], [12])  # its own


def test_run_options_cross_into_a_spawned_worker(cases):
    """Contract (plan §4.4): a `RunOptions` pickles with what E2 puts in it (the public map, a job's event table,
    another case's paths and an ask map) and comes back equal in content. Regression caught: a field that cannot cross
    into a spawned worker (a lambda, an open file), found only when a check runs with workers, hours in."""
    ro = RunOptions({"map": load_map("apartment_a"), "events": job_registry.get("vacuum/1").events},
                    stream_from=cases[1], ask_map={12: 3})
    back = pickle.loads(pickle.dumps(ro))
    assert (back.opts, back.ask_map) == (ro.opts, {12: 3})
    assert (back.stream_from.id, back.stream_from.private, back.stream_from.script.root) == (
        cases[1].id, cases[1].private, cases[1].script.root)


def test_a_failed_run_in_a_worker_raises_its_error_and_leaves_no_process(cases, fakes):
    """Contract (plan §4.4): a run `run_case` refuses in a worker (a system that runs on another channel) reaches the
    caller as the error itself, the queued runs are cancelled and no worker outlives the call. Regression caught: a
    parallel run that hangs, swallows the failure into missing results, or leaves processes running after it raised."""
    before = set(multiprocessing.active_children())
    with pytest.raises(ValueError, match="runs on the detections channel only"):
        run_suite(cases[:2], ["narrow", "lengths"], CHANNEL, workers=2)
    assert set(multiprocessing.active_children()) == before


def test_a_bad_request_is_refused_before_any_run(cases, fakes):
    """Contract (plan §4.4): fewer than one worker, a system that is not a baseline, a system named twice among the
    waves and one of `names` in no wave are refused with ValueError before any system is built; a wave may name a helper
    that `names` does not, and its runs are in the result. Regression caught: a typo in a system's name found after the
    first wave's hours, a system silently dropped, or run twice."""
    built = []
    notes = RunOptions({"ran": [], "built": built, "tag": "notes"})
    for names, waves, workers, why in [(["notes"], None, 0, "workers must be at least 1"),
                                       (["notes", "ghost"], None, 1, "no baseline named 'ghost'"),
                                       (["notes", "steps"], [["notes"], ["notes"]], 1, "exactly once"),
                                       (["notes", "steps"], [["notes"]], 1, "exactly once")]:
        with pytest.raises(ValueError, match=why):
            run_suite(cases, names, CHANNEL, options=lambda *a: notes, waves=waves, workers=workers)
    assert built == []
    helped = run_suite(cases, ["steps"], CHANNEL, options=feed, waves=[["lengths"], ["steps"]])
    assert list(helped) == ["lengths", "steps"] and len(helped["lengths"]) == 3


def test_score_rows_by_hand(tmp_path):
    """Contract (plan §2.6, D's `score.row`): a case with three golds (a number on the ladder, a position and an
    `unknown`) answered 1.4 for 1.45, a place 0.6 m off the gold's, and `unknown`, gives its rows in gold order with the
    case's keys (case, script, layout, robot, job, length) and the question's `when`, the hand credits (1.0 on every
    rung of [0.25, 0.5, 1, 2]; 0.5 on [0.5, 1]; 1.0 for the right `unknown`) and the penalised scores at t = 0.5 (1.0;
    0.5 - 0.5 = 0.0; 1.0), which are both the reported `score` and the gate's `score_gate` since no gold row declares a
    catastrophe (0.10.0), with no age band (the hand golds carry none). Regression caught: rows that drop the case,
    script or layout the gates group by, or a `when` read from the wrong file."""
    questions = [during_q(1, 12), end_q(2), end_q(3)]
    case = case_on(tmp_path, hand20(), questions)
    golds(case, rows=[gold_row(questions[0], gold={"type": "number", "value": 1.45}), gold_row(questions[1]),
                      gold_row(questions[2], gold=UNKNOWN, slice="unknown")])
    answers = {qid(1): {"type": "number", "value": 1.4}, qid(2): {"type": "position", "xy": [3.2, 2.1]},
               qid(3): dict(UNKNOWN)}
    run = CaseRun(answers, "complete", {}, {}, {}, None, "0" * 16)
    where = {"case": case.id, "script": case.script.id, "layout": "apartment_a", "robot": "vacuum", "job": "vacuum/1",
             "length": "short"}
    assert case_info(case) == where
    assert score_rows(case, run) == [
        {**where, "qid": qid(1), "family": "path_ahead", "when": "during", "scope": "now", "reference": None,
         "critical": False, "slice": "answerable", "answer_class": None, "range_m": None, "age_band": None,
         "s": 1.0, "score": 1.0, "score_gate": 1.0, "catastrophe": False, "outcome": "correct"},
        {**where, "qid": qid(2), "family": "object_where", "when": "end", "scope": "memory", "reference": None,
         "critical": False, "slice": "answerable", "answer_class": None, "range_m": None, "age_band": None,
         "s": 0.5, "score": 0.0, "score_gate": 0.0, "catastrophe": False, "outcome": "partial"},
        {**where, "qid": qid(3), "family": "object_where", "when": "end", "scope": "memory", "reference": None,
         "critical": False, "slice": "unknown", "answer_class": None, "range_m": None, "age_band": None,
         "s": 1.0, "score": 1.0, "score_gate": 1.0, "catastrophe": False, "outcome": "correct"}]


def test_an_unanswered_question_scores_as_failed(tmp_path):
    """Contract (map X-H15; recommendations §2.4 item 1): a run with no answer for a gold gives that gold a row, scored
    as a failed (malformed) answer: credit 0, penalised -1 on an answerable gold and -1 on an unknown one (an answer
    where `unknown` was right); and with `during` false the during question's gold has no row, the run not having asked
    it. Regression caught: a gold with no row, or a missing answer raising so that no run of a dataset scores."""
    questions = [during_q(1, 12), end_q(2), end_q(3)]
    case = case_on(tmp_path, hand20(), questions)
    golds(case, rows=[gold_row(questions[0]), gold_row(questions[1]), gold_row(questions[2], gold=UNKNOWN,
                                                                                  slice="unknown")])
    run = CaseRun({qid(1): {"type": "number", "value": 1.4}}, "complete", {}, {}, {}, None, "0" * 16)
    rows = score_rows(case, run)
    assert [(r["qid"], r["s"], r["score"], r["outcome"]) for r in rows] == [
        (qid(1), 1.0, 1.0, "correct"), (qid(2), 0.0, -1.0, "wrong"), (qid(3), 0.0, -1.0, "wrong")]
    assert [r["qid"] for r in score_rows(case, run, during=False)] == [qid(2), qid(3)]


def test_cases_like_are_the_compiled_cases_of_the_same_layout_and_job(tmp_path):
    """Contract (plan §4.4, `stbench run`'s options): the cases `stbench run` hands the options function are the
    dataset's compiled cases (questions and gold on disk) on the case's layout and job, its own among them, in id
    order. Regression caught: a prior built from another layout's gold (a warehouse's questions in a flat's pool) or
    from a case that was never compiled."""
    flat, other_flat = scored_case(tmp_path, hand20(), QUESTIONS), scored_case(tmp_path, _seeded(2), QUESTIONS)
    uncompiled = case_on(tmp_path, _seeded(3), QUESTIONS)
    hall = scored_case(tmp_path, Synth(tasks=[[(WAREHOUSE.rooms[0], 8)]], stage=WAREHOUSE), QUESTIONS)
    assert [c.id for c in cases_like(flat)] == sorted([flat.id, other_flat.id])
    assert [c.id for c in cases_like(hall)] == [hall.id]
    assert uncompiled.id not in [c.id for c in cases_like(flat)]
    assert (case_info(hall)["layout"], case_info(hall)["robot"], case_info(hall)["job"]) == (
        "warehouse_a", "amr", "amr/1")


def _seeded(seed: int) -> Synth:
    synth = hand20()
    synth.seed = seed
    return synth
