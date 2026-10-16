"""`stbench run codex:<model>` (`cmd_codex.py`), over the fake command (`codex_fake`) and a synthetic case: the run
driver of the Codex reference agent, from the command line through `stream.run_case` to the files it leaves. No model,
no `codex`, no network. Expected values are hand-written: the fake "model" notes the frames it is shown and answers
from how many it noted, and the exec it runs adds 100 tokens in (40 cached) and 10 out.

The case is six steps of one task with a decision point at step 3, so the stream is a task message, the frames 0 to 3
one at a time, the during ask, the frames 4 and 5 and the end ask: one exec, in which the fake "model" pulls it all.
The fake's answers are wrong by the gold (a number 4 for 1.4 is past the ladder's 2 m, a position (6, 0) for (3.2, 1.5)
is 3.18 m off, past its 1 m), so each scores 0 and, penalised at t = 0.5, -1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace

import pytest

from codex_kit import Fake, job_also_asks
from run_kit import scored_case
from stbench import cli, cmd_codex, job_registry
from stbench.cmd_run import Refused
from stream_kit import K, during_q, end_q, qid
from synth import Synth

KEYS = ("start", "task", "observe", "ask_during", "ask_end", "finish")
TOOLS = ["next_frame", "submit_answer"]  # the reference agent has no other tool, on either channel


@pytest.fixture(autouse=True)
def the_job_asks_what_the_case_asks(monkeypatch):
    """`small_case` holds a `number` during question and a `position` end one, and `vacuum/1` asks only `object_where`
    (a `position`): the job also asks `distance_between`, so the prompt shows the form of every type the case holds."""
    job_also_asks(monkeypatch, "distance_between")


def small_case(root):
    """Six steps in the kitchen, a decision point at step 3: a during question there and one end question, each with
    its gold (a number 1.4 and a position (3.2, 1.5), `stream_kit.gold_row`): `stbench run` scores what it runs."""
    synth = Synth(tasks=[[(K, 6)]], decisions=[(3, "before_doorway")])
    synth.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    synth.see(range(6), "E01")
    return scored_case(root, synth, [during_q(1, 3), end_q(2)])


def run(monkeypatch, fake: Fake, *argv) -> int:
    """`stbench run ARGV` with the Codex transport replaced by one over `fake`, built as the real one is (the model, the
    effort and the per-message timeouts come from the command line)."""
    monkeypatch.setattr(cmd_codex, "make_codex", lambda model, reasoning: fake.new(model=model, reasoning=reasoning))
    return cli.main(["run", *map(str, argv)])


def jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_fake_codex_case_runs_end_to_end(tmp_path, monkeypatch, capsys):
    """Contract (plan 4.9's exit): `stbench run codex:fake-model CASE_DIR` with the transport replaced by the fake
    completes a synthetic case on the default channel (`rgb_pose`, the camera image alone): the status is complete,
    every question is answered (the during one after 4 frames, the end one after 6: the fake counts what it was
    shown), the harness's `answers.jsonl` and `summary.json` are written
    and beside them `codex.summary.json` with the model, the effort given on the command line, the Codex version
    (no transport version, catalog digest or commit), the token totals of the one exec (100 in, 40 of them cached,
    10 out), the execs by kind, the six frames served and the seven pulls (one a frame, or a frame with its during or
    end questions, or the end: FD-questions-0.10 §1.5), no failures and a clean scan; the journal opens with the run
    event (the call bound and resume limit, the timeouts as given, the tools).
    Regression caught: the driver unwired (a run that does not reach the agent, or whose files cannot be read back), a
    summary that drops the model, the effort or its token totals, or a `--timeout` and `--reasoning` that never
    arrive. The harness also scores what ran: `rows.jsonl` holds a row a gold (both of the fake's answers are wrong,
    by hand: 4 against 1.4 is past the 2 m rung, 3.18 m against 1 m past the last) and `summary.json` the scores
    block."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--reasoning", "medium", "--timeout", "90") == 0
    out = tmp_path / "data" / "runs" / f"codex_fake-model-{case.id}"
    assert jsonl(out / "answers.jsonl") == [{"qid": qid(1), "answer": {"type": "number", "value": 4}},
                                            {"qid": qid(2), "answer": {"type": "position", "xy": [6.0, 0.0]}}]
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (harness["status"], harness["failures"], harness["sandbox"]) == ("complete", {}, "in-process")
    assert harness["usage"] == {"self_reported": True, "models": {"fake-model": {
        "tokens_in": 100, "tokens_cached_in": 40, "tokens_cache_write_in": 0, "tokens_out": 10}},
        "peak_context": None, "remote_store_bytes": None}
    summary = json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))
    assert {k: summary[k] for k in ("model", "reasoning", "codex", "case", "channel", "brief", "during", "status",
                                    "failures", "invalid", "scan")} == {
        "model": "fake-model", "reasoning": "medium", "codex": "codex-cli 9.9.9-fake", "case": case.id,
        "channel": "rgb_pose", "brief": "correct", "during": True, "status": "complete", "failures": {},
        "invalid": False, "scan": []}
    assert summary["execs"] == {"total": 1, "by_kind": {"start": 1}}
    assert (summary["frames"], summary["pulls"], summary["premature_pulls"], summary["resumes"]) == (6, 7, 0, 0)
    assert (summary["frames_asked"], summary["frame_latency"]["n"], summary["asked_latency"]["n"]) == (2, 4, 2)
    assert summary["tokens_lower_bound"] is False
    assert summary["tokens"] == {"in": 100, "cached_in": 40, "out": 10, "reasoning": 0}
    assert summary["answers"] == {"submitted": 2, "repaired": 0, "failed": {"malformed": 0, "no_submit": 0},
                                  "timed_out": 0} and summary["exec_failures"] == {}
    assert {key: len(times) for key, times in summary["wall_s"].items()} == dict(
        zip(KEYS, (1, 1, 6, 1, 1, 1), strict=True))
    assert not {"transport", "catalog_sha256", "git"} & set(summary)   # the facts of what ran, no stamp of ours
    first, *rest = jsonl(out / "codex.calls.jsonl")
    assert first["event"] == "run" and (first["model"], first["reasoning"]) == ("fake-model", "medium")
    assert not {"transport", "versions"} & set(first)
    assert first["settings"] == {"max_calls_item": 100, "max_resumes": 3, "finish_wait_s": 60.0}
    assert first["timeouts"] == dict.fromkeys(KEYS, 90.0) and first["tools"] == TOOLS
    assert first["case"] == case.id
    assert len([r for r in rest if "event" not in r]) == 1 and len(fake.execs()) == 1
    (only,) = fake.execs()  # one exec: six frames, a camera image each as an image block, no depth; frames 3 and 5
    assert [c["blocks"] for c in only["calls"] if c["text"].startswith("[k")] == [  # come with their questions
        ["text", "image"]] * 3 + [["text", "image", "text"], ["text", "image"], ["text", "image", "text"]]
    rows = jsonl(out / "rows.jsonl")
    assert [(r["qid"], r["when"], r["outcome"], r["s"], r["score"]) for r in rows] == [
        (qid(1), "during", "wrong", 0.0, -1.0), (qid(2), "end", "wrong", 0.0, -1.0)]
    assert harness["scores"]["headline"]["mean"] == -1.0 and harness["scores"]["counts"] == {
        "correct": 0, "partial": 0, "wrong": 2, "abstained": 0}
    printed = capsys.readouterr().out
    assert "complete: 2 answers" in printed and "scores: headline -1.000 over 2 questions" in printed


def test_a_fake_codex_case_runs_on_detections_with_text_for_images(tmp_path, monkeypatch):
    """Contract (the detections arm; plan 4.9): `stbench run codex:fake-model CASE --channel detections`
    is not refused and runs the same case to the end as on pictures: complete, the one exec, both questions
    answered (the fake counts the frames from their header lines, 4 and 6, as on pictures), the run event and the
    summary on the `detections` channel, with the tools of both channels;
    no image block was sent, the depth image the stream delivers included, each frame's reply holds the frame's
    detections as lines (the mug of the case, in the map frame), and neither the entity's id nor a path of the
    dataset is in anything the model was sent, the leak scan finding nothing. Regression caught: the gate channel
    still refused for the Codex agent, a detections run that sends images (the depth ones too), a run recorded as the
    picture agent, or an entity id reaching the model through a detection."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--channel", "detections") == 0
    out = tmp_path / "data" / "runs" / f"codex_fake-model-{case.id}"
    assert [a["answer"] for a in jsonl(out / "answers.jsonl")] == [{"type": "number", "value": 4},
                                                                     {"type": "position", "xy": [6.0, 0.0]}]
    summary = json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))
    assert {k: summary[k] for k in ("channel", "status", "failures", "invalid", "scan")} == {
        "channel": "detections", "status": "complete", "failures": {}, "invalid": False, "scan": []}
    first, *rest = jsonl(out / "codex.calls.jsonl")
    assert first["channel"] == "detections"
    assert first["tools"] == TOOLS
    (only,) = fake.execs()
    assert all(c["images"] == [] and set(c["blocks"]) == {"text"}   # a frame with questions: two text blocks
               for c in only["calls"] if c["tool"] == "next_frame")
    frames = [c["text"] for c in only["calls"] if c["text"].startswith("[k")]
    assert len(frames) == 6 and all("Detections in frame k" in f and "- mug (color red): map (2.00, 1.00, " in f
                                    for f in frames)
    sent = "\n".join([only["prompt"], *(c["text"] for c in only["calls"])])
    assert "E01" not in sent and str(tmp_path) not in sent and "private" not in sent


@pytest.mark.parametrize("asking", [
    {"decisions": {"before_doorway": ("out_of_view",)}},   # a family at a decision point (before 0.6.0's jobs)
    {"looks": ("ego_now", "ego_mem")},                     # families at look points (0.6.0's jobs; no decision asks)
])
def test_the_first_prompt_tells_of_questions_during_only_for_a_job_that_asks_during(tmp_path, monkeypatch, asking):
    """Contract (0.5.0's pilot fix round; 0.6.0: look points): `stbench run codex:` gives the agent the fact that its
    job asks during the job, read from the job's `decisions` and its `looks` (the two ways a job asks during): the
    start exec's prompt says questions come "after the job" for `vacuum/1` (it lists no family at its decision points
    and no look) and "at some points during the job and after it" for the same job made to ask `out_of_view` at a
    doorway, and for the same job made to look (`ego_now`, `ego_mem`) with no decision asking; the run's `--during`
    arm changes none of them (the arms of the leak check send the same prompt). Hand-written clauses. Regression
    caught: a prompt that promises questions during the job to a job that asks none (the vacuum, the helper, the
    drone, before their looks), one that reads the arm and not the job, or one read from `decisions` alone, which says
    "after the job" to a job whose every during question is a look and so tells the agent nothing of them."""
    case = small_case(tmp_path / "data")
    asking = replace(job_registry.get("vacuum/1"), **asking)
    real = job_registry.get

    def first_prompt(name: str, asks: bool, *argv: str) -> str:
        (tmp_path / name).mkdir()
        monkeypatch.setattr(job_registry, "get", lambda job_id: asking if asks else real(job_id))
        fake = Fake(tmp_path / name)
        assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", tmp_path / f"out-{name}", *argv) == 0
        return fake.execs()[0]["prompt"]

    for name, asks, argv in (("quiet", False, ()), ("quiet-off", False, ("--during", "off")),
                             ("asking", True, ()), ("asking-off", True, ("--during", "off"))):
        prompt = first_prompt(name, asks, *argv)
        clause = "at some points during the job and after it" if asks else "after the job"
        assert f"- Questions come {clause}, numbered Q1, Q2, ..." in prompt, name
        assert "decision point" not in prompt.split("\n\nTask 0")[0], name  # the system prompt; the brief has its own


def test_codex_failures_end_the_case(tmp_path, monkeypatch, capsys):
    """Contract (plan 4.9, 6.3): a usage limit that ends the exec while the during ask is answered (after the four
    frames before it) ends the case as a transport failure with that reason: the status says so, the questions not
    answered (the one that was asked and the end one) are failed and counted by reason, the run exits 1, no exec
    follows the failed one, and `codex.summary.json` is written with the failed exec counted by its reason, the four
    frames served and the token count marked a lower bound (a failed turn prints no usage); without `--reasoning` the
    run uses the effort the reference rows run at, and without `--timeout` each message its own timeout. The failed
    questions are scored as malformed answers (credit 0, penalised -1), not dropped. Regression caught: a stopped run
    scored as answered, a case that goes on after a usage limit (an ask's failure used to fail only its own
    questions), or a summary that is lost when the run fails."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"pulls": 4, "fail_late": "limit"}})  # F0-F2 and F3 with the ask are pulls 1-4; the limit
    assert run(monkeypatch, fake, "codex:fake-model", case.private) == 1
    out = tmp_path / "data" / "runs" / f"codex_fake-model-{case.id}"
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert harness["status"] == "ended: ask: TransportError: limit"
    assert harness["failures"] == {"case ended: ask": 2}
    assert [r["answer"] for r in jsonl(out / "answers.jsonl")] == [{"type": "failed", "reason": "case ended: ask"}] * 2
    assert [(r["outcome"], r["s"], r["score"]) for r in jsonl(out / "rows.jsonl")] == [("wrong", 0.0, -1.0)] * 2
    summary = json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == harness["status"] and summary["failures"] == {"case ended: ask": 2}
    assert summary["exec_failures"] == {"limit": 1} and summary["execs"] == {"total": 0, "by_kind": {}}
    assert summary["answers"]["timed_out"] == 0 and summary["invalid"] is False and summary["reasoning"] == "low"
    assert summary["frames"] == 4 and summary["tokens_lower_bound"] is True
    assert len(fake.execs()) == 1  # the failed exec was the only one: nothing was sent after it
    run_event = jsonl(out / "codex.calls.jsonl")[0]
    assert run_event["timeouts"] == {"start": 120.0, "task": 600.0, "observe": 600.0, "ask_during": 600.0,
                                     "ask_end": 3600.0, "finish": 600.0}
    assert "ended: ask: TransportError: limit" in capsys.readouterr().out


def test_a_pull_that_does_not_come_in_time_ends_the_case_and_fails_its_questions(tmp_path, monkeypatch, capsys):
    """Contract (FD-frame-pull §3.3): `--timeout 5` bounds each message, and a model that stops pulling after the second
    frame (it hangs, with the exec alive) is killed at the fifth second: the case ends as `ended: observe:
    TransportError: timeout`, both questions fail as `case ended: observe` and are scored as malformed answers, the
    run exits 1, and `codex.summary.json` counts the killed exec's failure as a `timeout` and the token count as a lower
    bound. Regression caught: a case that never ends because a model went quiet, or questions left unanswered rather
    than failed."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"hang": 2, "hang_s": 120}})
    began = time.monotonic()
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--timeout", "5") == 1
    assert 4.5 < time.monotonic() - began < 90
    out = tmp_path / "data" / "runs" / f"codex_fake-model-{case.id}"
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert harness["status"] == "ended: observe: TransportError: timeout"
    assert harness["failures"] == {"case ended: observe": 2}
    failed = {"type": "failed", "reason": "case ended: observe"}
    assert [r["answer"] for r in jsonl(out / "answers.jsonl")] == [failed] * 2
    summary = json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))
    assert summary["exec_failures"] == {"timeout": 1} and summary["frames"] == 2
    assert summary["tokens_lower_bound"] is True
    assert summary["execs"] == {"total": 1, "by_kind": {"start": 1}} and summary["invalid"] is False
    (only,) = fake.execs()
    assert not only["finished"] and "ended: observe: TransportError: timeout" in capsys.readouterr().out


def test_a_run_that_never_ran_leaves_no_journal_and_can_be_run_again(tmp_path, monkeypatch):
    """Contract (plan 4.9): when no exec finished (not logged in: the first exec fails), the run ends as a failed case,
    `codex.summary.json` says why (the failed exec counted as `auth`), and no journal is left, so the same `--out` is
    not refused as a run that cannot be repeated: the second run reaches Codex again. A run in which an exec did finish
    keeps its journal. Regression caught: a login failure that blocks every rerun in the folder with a journal that
    records no model call."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"fail": "auth"}, 1: {"fail": "auth"}})
    out = tmp_path / "out"
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", out) == 1
    assert not (out / "codex.calls.jsonl").exists()
    summary = json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))
    assert summary["exec_failures"] == {"auth": 1} and summary["status"] == "ended: task: TransportError: auth"
    assert summary["execs"]["total"] == 0 and summary["invalid"] is False
    # the failed exec's own words survive the removed journal (an outage must not read only "error", FINDINGS
    # 2026-10-14); regression caught: a summary that keeps the reason and loses the message
    assert summary["exec_failure_message"] and "log" in summary["exec_failure_message"].lower()
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", out) == 1  # not 2: not refused
    assert len(fake.execs()) == 2


@pytest.mark.parametrize("cause", ["an escaped tool", "a scan hit"])
def test_an_invalid_run_is_marked_and_scores_nothing(tmp_path, monkeypatch, capsys, cause):
    """Contract (plan 4.9: the closing scan): an exec that used something outside the tool server (`escaped`), or a
    leak-scan hit in what the harness sent, makes the run invalid: `codex.summary.json` says `"invalid": true` (with the
    hits, for a scan), the run raises so that `stbench run` exits 1 and says why, and nothing is scored (no answers, no
    rows, no `summary.json`). The journal stays even though no exec finished (the first one escaped or failed): its
    `exec_failed` record is the only evidence of why and names the tool or the limit. Regression caught: a run whose
    restriction failed, or that was shown the future, reported as a clean one; an escape on the first exec that
    deletes the journal naming the tool (the review's finding)."""
    case = small_case(tmp_path / "data")
    out = tmp_path / "out"
    if cause == "an escaped tool":
        fake = Fake(tmp_path, {0: {"escape": "command_execution"}})
    else:
        fake = Fake(tmp_path, {0: {"fail": "limit"}})
        monkeypatch.setattr(cmd_codex, "scan_log", lambda *a, **k: ["call 0, message 0 (user): truth 'truth'"])
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", out) == 1
    summary = json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))
    assert summary["invalid"] is True
    assert sorted(p.name for p in out.iterdir()) == ["codex.calls.jsonl", "codex.summary.json"]  # nothing scored
    events = jsonl(out / "codex.calls.jsonl")
    reason, named = ("escaped", "command_execution") if cause == "an escaped tool" else ("limit", "usage limit")
    names = [e["event"] for e in events]  # no exec record; the stream may send frames while the kill is on its way
    assert names[:3] == ["run", "served", "exec_started"] and names[-1] == "exec_failed"
    assert set(names[3:-1]) <= {"served", "call"} and "exec" not in names
    assert events[-1]["reason"] == reason and named in events[-1]["message"]
    assert summary["exec_failures"] == ({"escaped": 1} if cause == "an escaped tool" else {"limit": 1})
    assert summary["scan"] == ([] if cause == "an escaped tool" else ["call 0, message 0 (user): truth 'truth'"])
    err = capsys.readouterr().err
    why = "outside the server's" if cause == "an escaped tool" else "leak scan"
    assert "the run is invalid" in err and why in err


def test_a_run_that_cannot_start_is_refused_before_anything_is_written(tmp_path, monkeypatch, capsys):
    """Contract (plan 4.9: the preflight): a channel the agent cannot read (a name that is not one of the two it reads,
    `rgb_pose` and `detections`; `rgbd_pose` is retired), the text-only option on `detections`, a model the installed
    Codex does not list, a reasoning effort its catalog entry does not list or that is no effort at all, no `codex` on
    PATH and an `--out` that already holds a journal (a Codex run cannot be replayed) are each refused with exit 2 and
    the reason, and nothing is created: no output folder, no journal, and an existing journal is as it was. Regression
    caught: a refusal that leaves a half-made run folder, a run into a folder with the journal of another, a bad
    effort that reaches the first exec, or a channel the agent has no reading for reaching it once the detections arm
    opened the gate."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    out = tmp_path / "out"
    refusals = [(("--reasoning", "turbo"), "reasoning must be one of")]
    for options, why in refusals:
        assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", out, *options) == 2
        assert why in capsys.readouterr().err and not out.exists()
    for retired in ("rgb", "rgbd_pose"):  # retired names; the CLI's choices stop them sooner
        with pytest.raises(Refused, match=f"channels only, not '{retired}'"):
            cmd_codex._transport("fake-model",
                                 argparse.Namespace(channel=retired, out=out, reasoning=None, timeout=None))
    with pytest.raises(Refused, match="text_only withholds the camera images of rgb_pose, and 'detections' sends none"):
        cmd_codex._transport("fake-model", argparse.Namespace(channel="detections", out=out, reasoning=None,
                                                              timeout=None, text_only=True))
    assert run(monkeypatch, fake, "codex:gpt-nine", case.private, "--out", out) == 2
    assert "does not list the model 'gpt-nine'" in capsys.readouterr().err and not out.exists()
    assert run(monkeypatch, fake, "codex:limited-model", case.private, "--out", out, "--reasoning", "xhigh") == 2
    assert "at the reasoning efforts low, medium, high, not 'xhigh'" in capsys.readouterr().err and not out.exists()
    out.mkdir()
    (out / "codex.calls.jsonl").write_text('{"event": "run"}\n', encoding="utf-8")
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", out) == 2
    assert "a Codex run cannot be replayed" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["codex.calls.jsonl"]
    assert (out / "codex.calls.jsonl").read_text(encoding="utf-8") == '{"event": "run"}\n'
    assert fake.execs() == []


def test_the_real_transport_is_codex_on_path(tmp_path, monkeypatch, capsys):
    """Contract: unreplaced, `make_codex` is a `Codex` for the model and effort asked for (no timeout of its own: the
    exec as a whole is bounded only by the case), and a machine with no `codex` on PATH is refused with the install
    hint (exit 2), not a traceback. Regression caught: the seam that tests replace being the only thing that was ever
    run, so the real one is never built right."""
    codex = cmd_codex.make_codex("some-model", "medium")
    assert (codex.model, codex.reasoning) == ("some-model", "medium") and not hasattr(codex, "timeout")
    case = small_case(tmp_path / "data")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert cli.main(["run", "codex:some-model", str(case.private)]) == 2
    assert "not on PATH" in capsys.readouterr().err
