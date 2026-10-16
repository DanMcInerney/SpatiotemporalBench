"""What `stbench run` scores (`cmd_run.py`, plan §4.4's B.2): the rows and the scores block beside the answers, the
options a baseline is built with, and the refusals that keep a run from being unscorable. B.1's run tests are in
test_cmd_run.py; the rows themselves, from hand, in test_suite_runner.py."""

from __future__ import annotations

import json

import pytest

from run_kit import golds, scored_case
from stbench import baselines, cli, cmd_run, suite_priors
from stbench.stream_protocol import System
from stbench.suite import RunOptions
from stream_kit import AGENT_CALLS, QUESTIONS, case_on, gold_row, hand20, qid
from synth import WAREHOUSE, Synth

UNKNOWN = {"type": "unknown"}


def record(cls, *, prior_nm: bool = False) -> baselines.Baseline:
    """A registry record for a test system: every flag stated, so none is read off a stand-in."""
    return baselines.Baseline(cls, needs_prior=False, prior_nm=prior_nm, needs_map=False, needs_events=False,
                              other_stream=False)


def run(*argv) -> int:
    return cli.main(["run", *map(str, argv)])


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Unknowing(System):
    """A system that answers every question `unknown`."""

    def observe(self, steps):
        pass

    def answer(self, question):
        return dict(UNKNOWN)


class Biased(System):
    """Answers every question with the number its `bias` option holds (-1 without one): what it was built with."""

    def observe(self, steps):
        pass

    def answer(self, question):
        return {"type": "number", "value": float(self.opts.get("bias", -1))}


def test_run_scores_a_case(tmp_path, capsys):
    """Contract (plan §4.4's exit): `stbench run all_unknown CASE_DIR` writes, beside `answers.jsonl`, `rows.jsonl` with
    a row per gold in gold order, and a `scores` block in `summary.json` (the headline, the splits and the outcome
    counts). On a case of two answerable golds and an unknown one, all-`unknown` abstains twice (0, 0) and is right
    once (1): the during question is the `now` column (0) and the two end ones the `memory` column ((0 + 1)/2 = 1/2), so
    the headline is (0 + 1/2)/2 = 1/4 (pooled it would be 1/3); the during rows (0) and the end ones' (0, 1) average 0
    and 1/2; the abstentions are 2 and the correct answers 1. Regression caught: a smoke run that cannot be scored, a
    summary without its scores, or a headline pooled over the columns."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    golds(case, rows=[gold_row(QUESTIONS[0]), gold_row(QUESTIONS[1]),
                      gold_row(QUESTIONS[2], gold=UNKNOWN, slice="unknown")])
    assert run("all_unknown", case.private) == 0
    out = tmp_path / "runs" / f"all_unknown-{case.id}"
    rows = read_rows(out / "rows.jsonl")
    assert [(r["qid"], r["when"], r["slice"], r["score"], r["outcome"]) for r in rows] == [
        (qid(1), "during", "answerable", 0.0, "abstained"), (qid(2), "end", "answerable", 0.0, "abstained"),
        (qid(3), "end", "unknown", 1.0, "correct")]
    scores = json.loads((out / "summary.json").read_text(encoding="utf-8"))["scores"]
    assert scores["headline"]["mean"] == pytest.approx(1 / 4) and scores["headline"]["ci"] is None  # one script
    assert scores["counts"] == {"correct": 1, "partial": 0, "wrong": 0, "abstained": 2}
    assert (scores["by"]["when"]["during"]["mean"], scores["by"]["when"]["end"]["mean"]) == (0.0, 0.5)
    assert "scores: headline 0.250 over 3 questions" in capsys.readouterr().out


def test_a_run_with_the_during_questions_off_scores_only_what_it_asked(tmp_path):
    """Contract (plan §4.4): with `--during off` the during question was not asked, so its gold has no row and the
    scores cover the end questions only; the summary says the during flag was off. Regression caught: a leak-check arm
    charged for questions it was never asked, which reads as a score it did not earn."""
    case = scored_case(tmp_path, hand20(), QUESTIONS)
    assert run("all_unknown", case.private, "--during", "off") == 0
    out = tmp_path / "runs" / f"all_unknown-{case.id}"
    assert [r["qid"] for r in read_rows(out / "rows.jsonl")] == [qid(2), qid(3)]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["during"] is False and sum(summary["scores"]["counts"].values()) == 2


def test_prior_nm_baselines_are_refused_by_run(tmp_path, monkeypatch, capsys):
    """Contract (the coordinator's call on Q9): a baseline whose record sets `prior_nm` (its priors carry another
    system's answers, which only `stbench check`'s first wave makes) is refused with exit 2, naming the flag and
    `stbench check`, before anything is written; the same record without the flag runs. Regression caught: a single-case
    run whose prior silently lacks its first-wave answers."""
    case = scored_case(tmp_path, hand20(), QUESTIONS)
    monkeypatch.setitem(baselines.BASELINES, "chained", record(Unknowing, prior_nm=True))
    assert run("chained", case.private, "--out", tmp_path / "no") == 2
    err = capsys.readouterr().err
    assert "prior_nm" in err and "stbench check" in err and not (tmp_path / "no").exists()
    monkeypatch.setitem(baselines.BASELINES, "chained", record(Unknowing))
    assert run("chained", case.private, "--out", tmp_path / "fine") == 0


def test_a_baseline_is_built_with_the_options_suite_priors_gives_it(tmp_path, monkeypatch):
    """Contract (plan §4.4, N45): `stbench run BASELINE` builds the baseline with the options of
    `suite_priors.options_for(name, case, cases, {})`, `cases` being the dataset's compiled cases of the case's layout
    and job (its own included; another layout's and an uncompiled case left out), with an empty `done`. Regression
    caught: baselines run without their priors or map, or priors built from the wrong cases."""
    flat = scored_case(tmp_path, hand20(), QUESTIONS)
    other = hand20()
    other.seed = 2
    twin = scored_case(tmp_path, other, QUESTIONS)
    third = hand20()
    third.seed = 3
    case_on(tmp_path, third, QUESTIONS)  # never compiled: no gold
    scored_case(tmp_path, Synth(tasks=[[(WAREHOUSE.rooms[0], 8)]], stage=WAREHOUSE), QUESTIONS)  # another layout
    monkeypatch.setitem(baselines.BASELINES, "biased", record(Biased))
    calls = []

    def options_for(name, case, cases, done):
        calls.append((name, case.id, [c.id for c in cases], dict(done)))
        return RunOptions({"bias": 7})

    monkeypatch.setattr(suite_priors, "options_for", options_for)
    assert run("biased", flat.private, "--out", tmp_path / "with") == 0
    assert calls == [("biased", flat.id, sorted([flat.id, twin.id]), {})]
    assert {r["answer"]["value"] for r in read_rows(tmp_path / "with" / "answers.jsonl")} == {7.0}


def test_a_case_without_gold_is_refused_before_anything_runs(tmp_path, monkeypatch, capsys):
    """Contract (plan §4.4): a run scores what it ran, so a case with no `gold.jsonl` is refused with exit 2 before a
    baseline is built or an agent is called, and nothing is written. Regression caught: an agent's run (hours of plan
    quota) that ends and then cannot be scored."""
    monkeypatch.setitem(cmd_run.AGENTS, "fake", "stream_kit.fake_agent")
    case = case_on(tmp_path, hand20(), QUESTIONS)
    AGENT_CALLS.clear()
    for system in ("all_unknown", "fake:model"):
        assert run(system, case.private, "--out", tmp_path / "no") == 2
        assert "has no gold.jsonl" in capsys.readouterr().err
    assert AGENT_CALLS == [] and not (tmp_path / "no").exists()


def test_a_gold_the_scorer_rejects_leaves_the_answers_and_summary(tmp_path, capsys):
    """Contract (plan §4.4): a run whose gold the scorer cannot read (a slice that disagrees with the gold value) exits
    1 with a message saying so, after writing `answers.jsonl` and `summary.json` (without scores) and no `rows.jsonl`.
    Regression caught: an agent's answers lost because the scoring raised."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    golds(case, rows=[gold_row(QUESTIONS[0]), gold_row(QUESTIONS[1], slice="unknown"), gold_row(QUESTIONS[2])])
    out = tmp_path / "out"
    assert run("all_unknown", case.private, "--out", out) == 1
    assert "could not be scored" in capsys.readouterr().err
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and "scores" not in summary and not (out / "rows.jsonl").exists()
    assert len(read_rows(out / "answers.jsonl")) == 3

