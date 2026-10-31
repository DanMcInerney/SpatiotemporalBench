"""`stbench run claude:<model>` and `stbench eval claude:<model>` (`cmd_claude.py`, `claude_system.py`,
`claude_live.py`), over the fake command (`claude_cli_fake`) and a synthetic case: the Claude Code reference agent,
from the command line through `stream.run_case` to the files it leaves. No model, no `claude`, no network. Expected
values are hand-written: the fake "model" notes the frames it is shown and answers from how many it noted, and each
turn it runs adds 10 input tokens, 60 cache reads, 30 cache writes and 10 output ones (4 of them thinking) to the
session's running totals, and 0.25 dollars.

The case is `test_cmd_codex.small_case`: a task message, the frames 0 to 3 one at a time, the during ask, the frames 4
and 5 and the end ask, all pulled by the one process the fake "model" is.
"""

from __future__ import annotations

import json
import shutil
import time

import pytest

from claude_kit import Fake
from codex_kit import job_also_asks
from eval_kit import cli as eval_cli
from eval_kit import compiled, results, short_script
from stbench import cli, cmd_claude, cmd_codex, cmd_run
from stbench.cmd_run import Refused
from stbench.llm_render import RESUME_MESSAGE
from stream_kit import end_q, qid
from test_cmd_codex import small_case

KEYS = ("start", "task", "observe", "ask_during", "ask_end", "finish")
USED = {"in": 100, "cached_in": 60, "out": 10, "reasoning": 4}   # one turn, by hand: 10 + 60 + 30, 60, 10, 4


@pytest.fixture(autouse=True)
def the_job_asks_what_the_case_asks(monkeypatch):
    """`small_case` holds a `number` during question and a `position` end one, and `vacuum/1` asks only
    `object_where` (a `position`): the job also asks `distance_between`, so the prompt shows the form of every type
    the case holds."""
    job_also_asks(monkeypatch, "distance_between")


def run(monkeypatch, fake: Fake, *argv) -> int:
    """`stbench run ARGV` with the Claude transport replaced by one over `fake`, built as the real one is (the model
    and the effort come from the command line)."""
    monkeypatch.setattr(cmd_claude, "make_claude", lambda model, reasoning: fake.new(model=model, reasoning=reasoning))
    return cli.main(["run", *map(str, argv)])


def jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def folder(tmp_path, case, name: str = "claude_sonnet"):
    return tmp_path / "runs" / f"{name}-{case.id}"


def test_a_fake_claude_case_runs_end_to_end(tmp_path, monkeypatch, capsys):
    """Contract (the Codex agent's, for Claude Code): `stbench run claude:sonnet CASE_DIR` completes a synthetic case
    on the default channel with ONE process: the status is complete, both questions are answered (the fake counts
    what it was shown: 4 frames at the during ask, 6 at the end), the harness's `answers.jsonl`, `rows.jsonl` and
    `summary.json` are the shape a Codex case writes, and beside them `claude.summary.json` has the model id
    (`sonnet` is `claude-sonnet-5-5`), the effort from the command line, `provider` `claude`, the CLI's version under
    `claude`, the tokens of the one turn (100 in, 60 of them cached, 10 out, 4 thinking), the CLI's own 0.25 dollars
    with a note that a subscription does not pay them, the six frames served and the seven pulls, no failures, no
    warnings and a clean scan. The harness's usage is keyed by the id, and its `cost_usd` is the price table's figure
    for those tokens, not the CLI's. The process got the isolation: the system prompt as `--system-prompt` and the
    journal's first prompt that prompt and the task, the task alone as the first user message, an empty working
    folder, and the environment that turns off CLAUDE.md and auto memory. Regression caught: the driver
    unwired, a summary that drops the provider, the id or the tokens, the CLI's dollars written over the harness's own
    figure, a system prompt sent as a user message (or twice), a model that runs in the repository or sees the user's
    CLAUDE.md, and an image dropped on the way (each frame reply holds its camera image block, a frame with questions
    the questions' text after it)."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--reasoning", "medium", "--timeout", "90") == 0
    out = folder(tmp_path, case)
    assert [r["answer"] for r in jsonl(out / "answers.jsonl")] == [{"type": "number", "value": 4},
                                                                     {"type": "position", "xy": [6.0, 0.0]}]
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (harness["status"], harness["failures"], harness["sandbox"]) == ("complete", {}, "in-process")
    assert harness["usage"] == {"self_reported": True, "models": {"claude-sonnet-5-5": {
        "tokens_in": 100, "tokens_cached_in": 60, "tokens_cache_write_in": 30, "tokens_out": 10}},
        "peak_context": None, "remote_store_bytes": None}
    assert harness["cost_note"] is None and 0 < harness["cost_usd"] < 0.01   # the table's, not the CLI's 0.25
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert {k: summary[k] for k in ("model", "reasoning", "provider", "claude", "case", "channel", "brief", "during",
                                    "status", "failures", "invalid", "scan")} == {
        "model": "claude-sonnet-5-5", "reasoning": "medium", "provider": "claude", "claude": "9.9.9 (Claude Code fake)",
        "case": case.id, "channel": "rgb_pose", "brief": "correct", "during": True, "status": "complete",
        "failures": {}, "invalid": False, "scan": []}
    assert summary["execs"] == {"total": 1, "by_kind": {"start": 1}} and summary["exec_failures"] == {}
    assert (summary["frames"], summary["pulls"], summary["premature_pulls"], summary["resumes"]) == (6, 7, 0, 0)
    assert (summary["frames_asked"], summary["frame_latency"]["n"], summary["asked_latency"]["n"]) == (2, 4, 2)
    assert summary["tokens"] == USED and summary["tokens_lower_bound"] is False and summary["warnings"] == 0
    assert summary["cost_usd"] == 0.25 and "subscription pays by quota" in summary["cost_note"]
    assert summary["answers"] == {"submitted": 2, "repaired": 0, "failed": {"malformed": 0, "no_submit": 0},
                                  "timed_out": 0}
    assert {key: len(times) for key, times in summary["wall_s"].items()} == dict(
        zip(KEYS, (1, 1, 6, 1, 1, 1), strict=True))
    journal = jsonl(out / "claude.calls.jsonl")
    first, record = journal[0], next(r for r in journal if "event" not in r)
    assert first["event"] == "run" and (first["model"], first["reasoning"]) == ("claude-sonnet-5-5", "medium")
    assert (first["provider"], first["claude"]) == ("claude", "9.9.9 (Claude Code fake)")
    assert first["timeouts"] == dict.fromkeys(KEYS, 90.0) and first["tools"] == ["next_frame", "submit_answer"]
    assert (record["provider"], record["kind"], record["cost"], record["killed"]) == ("claude", "start", 0.25, False)
    assert record["thread"] == "session-0000" and record["usage"]["prompt_tokens"] == 100
    assert "/9j/" not in (out / "claude.calls.jsonl").read_text(encoding="utf-8")   # no frame as base64 in the journal
    (only,) = fake.processes()
    argv = only["argv"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5" and argv[argv.index("--effort") + 1] == "medium"
    assert argv[argv.index("--tools") + 1] == "" and "--bare" not in argv
    assert argv[argv.index("--system-prompt") + 1] == only["system"]
    assert len(only["messages"]) == 1 and only["messages"][0].startswith("Task 0 begins at ")
    started = next(r for r in journal if r.get("event") == "exec_started")
    assert started["prompt"] == only["system"] + "\n\n" + only["messages"][0]   # what the model was shown, as for Codex
    assert only["system"].startswith("You are ") and "Task 0" not in only["system"]
    assert only["cwd_files"] == [] and only["cwd"].replace("\\", "/").endswith("/in")
    assert only["env"] == {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                           "MCP_TIMEOUT": "60000", "MCP_TOOL_TIMEOUT": "600000"}
    assert [c["blocks"] for c in only["calls"] if c["text"].startswith("[k")] == [
        ["text", "image"]] * 3 + [["text", "image", "text"], ["text", "image"], ["text", "image", "text"]]
    rows = jsonl(out / "rows.jsonl")
    assert [(r["qid"], r["when"], r["outcome"], r["s"], r["score"]) for r in rows] == [
        (qid(1), "during", "wrong", 0.0, -1.0), (qid(2), "end", "wrong", 0.0, -1.0)]
    printed = capsys.readouterr().out
    assert "complete: 2 answers" in printed and "cost: 0.25 USD, API-equivalent dollars" in printed


def test_a_turn_that_ends_early_is_resumed_on_the_same_stdin(tmp_path, monkeypatch):
    """Contract (the Codex agent's resume, over one process): when the model ends its turn after two frames without
    pulling again, the harness writes the fixed line (`RESUME_MESSAGE`) to the SAME process's stdin as the next user
    message, and the case goes on and completes: one process, two user messages (the task, then the line), the
    journal holding a `start` turn and a `resume` turn, one resume counted, and the tokens of the two turns summed
    once each (the result prints the session's running totals: 100 then 200, so 100 + 100 = 200, not 300) with the
    CLI's dollars the same way (0.25 then 0.50). Regression caught: a resume that starts another process (the
    conversation lost), tokens counted from each total in full, or a case that ends where a model merely paused."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"pulls": 2}})
    assert run(monkeypatch, fake, "claude:sonnet", case.private) == 0
    out = folder(tmp_path, case)
    (only,) = fake.processes()
    assert only["messages"][1:] == [RESUME_MESSAGE] and len(only["messages"]) == 2
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["execs"] == {"total": 2, "by_kind": {"start": 1, "resume": 1}}
    assert (summary["resumes"], summary["frames"], summary["exec_failures"]) == (1, 6, {})
    assert summary["tokens"] == {"in": 200, "cached_in": 120, "out": 20, "reasoning": 8}
    assert summary["cost_usd"] == 0.5 and summary["tokens_lower_bound"] is False
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert harness["usage"]["models"]["claude-sonnet-5-5"] == {
        "tokens_in": 200, "tokens_cached_in": 120, "tokens_cache_write_in": 60, "tokens_out": 20}
    assert [r["answer"] for r in jsonl(out / "answers.jsonl")] == [{"type": "number", "value": 4},
                                                                     {"type": "position", "xy": [6.0, 0.0]}]


def test_a_model_that_keeps_ending_its_turn_stalls_after_three_resumes(tmp_path, monkeypatch):
    """Contract (FD-frame-pull: at most `max_resumes` = 3 resumes in a row with no pull between): a model that pulls
    one frame and then ends every turn at once is sent the line three times, on the one stdin (four user messages in
    all), and the fourth early end is `stalled`: the status says so, both questions fail, the run exits 1, and the
    summary counts the failure by that reason and the four turns. Regression caught: a case that loops on a model
    that will not pull, or that gives up after one pause."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"pulls": 1}, 1: {"pulls": 0}, 2: {"pulls": 0}, 3: {"pulls": 0}})
    assert run(monkeypatch, fake, "claude:sonnet", case.private) == 1
    out = folder(tmp_path, case)
    (only,) = fake.processes()
    assert only["messages"][1:] == [RESUME_MESSAGE] * 3 and len(only["messages"]) == 4
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert harness["status"].endswith("TransportError: stalled") and sum(harness["failures"].values()) == 2
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert summary["exec_failures"] == {"stalled": 1} and summary["resumes"] == 3
    assert summary["execs"] == {"total": 4, "by_kind": {"start": 1, "resume": 3}} and summary["frames"] == 1


@pytest.mark.parametrize(("step", "reason", "named"), [
    ({"fail": "limit"}, "limit", "session limit"),
    ({"fail": "auth"}, "auth", "Invalid API key"),
    ({"fail": "context"}, "context", "Prompt is too long"),
    ({"fail": "overloaded"}, "limit", "529"),
    ({"fail": "boom"}, "error", "exploded"),
    ({"permission": True}, "error", "held for approval")])
def test_a_failed_turn_ends_the_case_as_its_reason(tmp_path, monkeypatch, step, reason, named):
    """Contract (error classification from the stream, plan 6.3): the error `result` events Claude Code prints (a
    usage limit, a login problem, a full context window, an overloaded API, anything else) and a permission request
    each end the case as the transport failure they are, with the word that counts them in `exec_failures` and their
    own words in the summary, the questions failed, the exit 1, and nothing sent after the failure (one process, one
    message). Regression caught: a failure read as a good turn (the answers taken as made), a limit named `error` (so
    a rerun is not told to wait), or a case that goes on after a failure."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: step})
    assert run(monkeypatch, fake, "claude:sonnet", case.private) == 1
    out = folder(tmp_path, case)
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert summary["exec_failures"] == {reason: 1} and summary["execs"]["total"] == 0 and summary["invalid"] is False
    assert named in summary["exec_failure_message"] and summary["status"].endswith(f"TransportError: {reason}")
    (only,) = fake.processes()
    assert len(only["messages"]) == 1
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [r["answer"]["type"] for r in jsonl(out / "answers.jsonl")] == ["failed"] * 2 and harness["failures"]


def test_a_limit_after_some_frames_keeps_the_count_a_lower_bound(tmp_path, monkeypatch, capsys):
    """Contract (the Codex test of the same name): a usage limit that ends the turn after four pulls (three frames
    and the fourth with the during ask) ends the case as `ended: ask: TransportError: limit` with both questions
    failed (the one asked and the end one), the summary counts the failed turn by its reason, the four frames served
    and the token count a lower bound (a failed turn prints no usage); without `--reasoning` the effort is `low`, and
    without `--timeout` each message has its own. Regression caught: a stopped run scored as answered, or a summary
    lost when the run fails."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"pulls": 4, "fail_late": "limit"}})
    assert run(monkeypatch, fake, "claude:sonnet", case.private) == 1
    out = folder(tmp_path, case)
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert harness["status"] == "ended: ask: TransportError: limit" and harness["failures"] == {"case ended: ask": 2}
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert summary["exec_failures"] == {"limit": 1} and summary["frames"] == 4 and summary["tokens_lower_bound"]
    assert summary["reasoning"] == "low" and summary["execs"] == {"total": 0, "by_kind": {}}
    run_event = jsonl(out / "claude.calls.jsonl")[0]
    assert run_event["timeouts"] == {"start": 120.0, "task": 600.0, "observe": 600.0, "ask_during": 600.0,
                                     "ask_end": 3600.0, "finish": 600.0}
    assert "ended: ask: TransportError: limit" in capsys.readouterr().out


def test_a_process_that_dies_without_a_result_is_named_by_its_stderr(tmp_path, monkeypatch):
    """Contract: a process that exits 1 with the login message on stderr and no `result` ends the case as `auth` (the
    stderr is the only evidence), nothing was spent so no journal is left and the same `--out` is not refused, and the
    summary keeps the message. Regression caught: a login failure that blocks every rerun in the folder with a
    journal that records no model call, or one that reads only `error`."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"crash": True}, 1: {"crash": True}})
    out = tmp_path / "out"
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--out", out) == 1
    assert not (out / "claude.calls.jsonl").exists()
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert summary["exec_failures"] == {"auth": 1} and summary["status"] == "ended: task: TransportError: auth"
    assert "Invalid API key" in summary["exec_failure_message"]
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--out", out) == 1   # not 2: not refused
    assert len(fake.processes()) == 2


def test_a_pull_that_does_not_come_in_time_kills_the_process(tmp_path, monkeypatch, capsys):
    """Contract (the Codex test of the same name): `--timeout 5` bounds each message, and a model that stops pulling
    after the second frame (it hangs, the process alive) is killed with its tool server at the fifth second: the case
    ends as `ended: observe: TransportError: timeout`, both questions fail, the run exits 1, and
    `claude.summary.json` counts a `timeout`, the killed turn as an exec and the tokens a lower bound. Regression
    caught: a case that never ends because a model went quiet, or a kill that leaves the process running after the
    harness has given up."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"hang": 2, "hang_s": 120}})
    began = time.monotonic()
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--timeout", "5") == 1
    assert 4.5 < time.monotonic() - began < 90
    out = folder(tmp_path, case)
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert harness["status"] == "ended: observe: TransportError: timeout"
    assert harness["failures"] == {"case ended: observe": 2}
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert summary["exec_failures"] == {"timeout": 1} and summary["frames"] == 2
    assert summary["tokens_lower_bound"] is True and summary["invalid"] is False
    assert summary["execs"] == {"total": 1, "by_kind": {"start": 1}}
    (record,) = [r for r in jsonl(out / "claude.calls.jsonl") if "event" not in r]
    assert record["killed"] is True and record["usage"]["prompt_tokens"] == 0
    assert "ended: observe: TransportError: timeout" in capsys.readouterr().out


def test_a_tool_outside_the_server_makes_the_run_invalid(tmp_path, monkeypatch, capsys):
    """Contract (the isolation holds or the run is not scored): a `tool_use` of a built-in tool (here `Bash`) kills
    the process at once and the run is invalid: `claude.summary.json` says `"invalid": true` and counts an `escaped`
    failure, `stbench run` exits 1 and says why, nothing is scored, and the journal stays with an `exec_failed`
    record that names the tool. Regression caught: a model that reached a shell reported as a clean run."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"escape": "Bash"}})
    out = tmp_path / "out"
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--out", out) == 1
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert summary["invalid"] is True and summary["exec_failures"] == {"escaped": 1}
    assert sorted(p.name for p in out.iterdir()) == ["claude.calls.jsonl", "claude.summary.json"]   # nothing scored
    last = jsonl(out / "claude.calls.jsonl")[-1]
    assert last["event"] == "exec_failed" and last["reason"] == "escaped" and "'Bash'" in last["message"]
    err = capsys.readouterr().err
    assert "the run is invalid" in err and "outside the server's" in err


def test_a_line_that_is_not_json_is_kept_as_a_warning_and_does_not_fail_the_turn(tmp_path, monkeypatch):
    """Contract (the unrecognised-model notice the CLI prints before the stream): a line of stdout that is not a JSON
    object is journaled as a warning of the turn and counted in the summary, and the case completes. A model id the
    price table lacks has no harness cost, with the reason. Regression caught: a notice that fails every run with a
    model id this CLI does not list yet, or one dropped without a trace."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"noise": True}})
    assert run(monkeypatch, fake, "claude:claude-unpriced-0", case.private) == 0
    out = folder(tmp_path, case, "claude_claude-unpriced-0")
    (record,) = [r for r in jsonl(out / "claude.calls.jsonl") if "event" not in r]
    assert record["warnings"] == ['[claude-code:unrecognized_model] {"model":"claude-fake"}']
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert (summary["status"], summary["warnings"], summary["model"]) == ("complete", 1, "claude-unpriced-0")
    harness = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert harness["cost_usd"] is None and harness["cost_note"].startswith("no price for claude-unpriced-0")


def test_a_run_that_cannot_start_is_refused_before_anything_is_written(tmp_path, monkeypatch, capsys):
    """Contract (the Codex test of the same name): a name that is no Claude model, an effort the CLI has no level for,
    the text-only option on `detections`, an `--out` that already holds a journal (a Claude Code run cannot be
    replayed) and no `claude` on PATH are each refused with exit 2 and the reason, and nothing is created: no output
    folder, no process, and an existing journal is as it was. Regression caught: a refusal that leaves a half-made
    run folder, a run into a folder with the journal of another, a Codex model id spent on `--model`, or an effort
    that reaches the first call."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    out = tmp_path / "out"
    assert run(monkeypatch, fake, "claude:gpt-nine", case.private, "--out", out) == 2
    assert "must be haiku, sonnet, opus" in capsys.readouterr().err and not out.exists()
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--out", out, "--reasoning", "none") == 2
    assert "reasoning must be one of" in capsys.readouterr().err and not out.exists()
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--out", out, "--channel", "detections",
               "--text-only") == 2
    assert "withholds the image" in capsys.readouterr().err and not out.exists()
    out.mkdir()
    (out / "claude.calls.jsonl").write_text('{"event": "run"}\n', encoding="utf-8")
    assert run(monkeypatch, fake, "claude:sonnet", case.private, "--out", out) == 2
    assert "a Claude Code run cannot be replayed" in capsys.readouterr().err
    assert (out / "claude.calls.jsonl").read_text(encoding="utf-8") == '{"event": "run"}\n'
    assert sorted(p.name for p in out.iterdir()) == ["claude.calls.jsonl"] and fake.processes() == []
    monkeypatch.undo()   # the real transport, on a machine with no claude
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert cli.main(["run", "claude:sonnet", str(case.private), "--out", str(tmp_path / "none")]) == 2
    assert "not on PATH" in capsys.readouterr().err and not (tmp_path / "none").exists()


def test_claude_and_codex_resolve_through_the_one_registry_and_take_the_same_options(tmp_path, monkeypatch):
    """Contract (switching between the two agents is only the prefix): `claude` and `codex` are both in `AGENTS`, both
    take `--text-only`, and the same command line with either prefix reaches that agent's runner with the same model
    string, effort, channel, timeout, brief, during and text-only options. Regression caught: an option that one
    agent takes and the other silently drops, or an agent missing from the registry."""
    assert cmd_run.AGENTS["claude"] == "stbench.cmd_claude.run" and cmd_run.AGENTS["codex"] == "stbench.cmd_codex.run"
    assert {"claude", "codex"} <= cmd_run.TEXT_ONLY_AGENTS
    case = small_case(tmp_path / "data")
    seen = []

    def recorder(agent: str):
        def runner(model, case_paths, args):
            seen.append((agent, model, args.reasoning, args.channel, args.timeout, args.brief, args.during,
                         args.text_only))
            raise Refused("stop here")
        return runner

    monkeypatch.setattr(cmd_claude, "run", recorder("claude"))
    monkeypatch.setattr(cmd_codex, "run", recorder("codex"))
    for agent in ("claude", "codex"):
        assert cli.main(["run", f"{agent}:the-model", str(case.private), "--reasoning", "high", "--timeout", "90",
                         "--brief", "neutral", "--during", "off", "--text-only"]) == 2
    assert seen == [(agent, "the-model", "high", "rgb_pose", 90.0, "neutral", "off", True)
                    for agent in ("claude", "codex")]


def test_eval_runs_the_claude_agent_and_reruns_a_stopped_case_from_scratch(tmp_path, monkeypatch):
    """Contract (docs/claude.md; the Codex test of the same name): `eval claude:sonnet --reasoning high` runs the
    agent through `cmd_run.AGENTS` with the options of the command; a case stopped part-way (its turn hit a usage
    limit after the first frame: `ended: observe: TransportError: limit`) leaves its journal, and the rerun removes
    the folder and runs the case from its start (a Claude Code run cannot be replayed: `refuse` would otherwise block
    it): the second fake is run once, the journal holds that turn and no stale one, and the result is complete,
    stamped with the system and the effort, with the tokens of the case. Regression caught: `eval` ignoring an agent
    that is not Codex, a refusal blocking a resume, a stale journal mixed into a rerun, or a result that does not say
    which agent ran."""
    data = tmp_path / "data"
    case = compiled(data, short_script(1), [end_q(1)])
    where = tmp_path / "runs" / "claude_sonnet-rgb_pose" / "cases" / case.id
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    first, second = Fake(tmp_path / "one", {0: {"pulls": 1, "fail_late": "limit"}}), Fake(tmp_path / "two")

    def evaluate(fake: Fake) -> int:
        monkeypatch.setattr(cmd_claude, "make_claude",
                            lambda model, reasoning: fake.new(model=model, reasoning=reasoning))
        return eval_cli("claude:sonnet", "--data", data, "--reasoning", "high")

    assert evaluate(first) == 1
    assert (where / "claude.calls.jsonl").exists() and len(first.processes()) == 1
    doc = results(where.parents[1])
    assert [c["status"] for c in doc["cases"]] == ["ended: observe: TransportError: limit"] and doc["partial"]
    assert evaluate(second) == 0
    (only,) = second.processes()
    assert only["argv"][only["argv"].index("--effort") + 1] == "high"
    journal = jsonl(where / "claude.calls.jsonl")
    assert len([r for r in journal if "event" not in r]) == 1 and journal[0]["event"] == "run"
    doc = results(where.parents[1])
    (done,) = doc["cases"]
    assert (done["status"], doc["partial"], doc["reasoning"]) == ("complete", False, "high")
    assert done["tokens"] == {"tokens_in": 100, "tokens_cached_in": 60, "tokens_cache_write_in": 30, "tokens_out": 10}
    assert doc["system"] == "claude:sonnet" and len(doc["rows"]) == 1
    assert done["cost_note"] is None and 0 < doc["cost_usd"] < 0.01
