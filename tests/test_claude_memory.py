"""A memory beside the Claude Code reference agent (`--memory SPEC` on `claude:<model>`: `claude_exec`, `claude_live`,
`claude_system`, `cmd_claude`, `codex_memory`, `codex_log`): a second MCP server in the one process, its guide in the
system prompt, its calls journaled, and the robot's sensors fed to its store by one long-lived process. Over the fake
command (`claude_cli_fake`), a tiny memory server and a tiny sensor feed (`memory_fake_server`, `memory_fake_sense`); no
model, no `claude`, no network, no real memory system. The invariant that matters most is that nothing the model is sent
changes with a memory (`test_a_memory_changes_no_message_the_model_is_sent`). Expected values are written by hand."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import pytest

from claude_kit import Fake, claude_run, pulled
from claude_memory_kit import CLAIM, PICTURE, PLAN, VIEW, made, play, sensed, sent
from codex_kit import GUIDE, job_also_asks, memory
from eval_kit import cli as eval_cli
from eval_kit import compiled, results, short_script
from stbench import cmd_claude
from stbench.claude_exec import ALLOWED, Claude, check_event, digest, memory_results, memory_uses
from stbench.codex_exec import CodexError, ToolEscaped
from stbench.codex_log import read_log
from stbench.llm_scan import scan_log
from stream_kit import end_q
from test_cmd_codex import jsonl, small_case


@pytest.fixture(autouse=True)
def the_job_asks_what_the_case_asks(monkeypatch):
    """`small_case` holds a `number` during question, which `vacuum/1` asks none of: see `test_cmd_claude.py`."""
    job_also_asks(monkeypatch, "distance_between")


# -- what the model is sent -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("early", [{}, {"pulls": 3}], ids=["one turn", "a resume"])
def test_a_memory_changes_no_message_the_model_is_sent(tmp_path, monkeypatch, early):
    """Contract: with a memory every message and item the model is sent (the first user message, each task, frame, ask,
    resume line and the end item) is byte for byte what the reference agent is sent: the same case run without and
    with a memory, the model calling the memory's tools after every frame and at every ask, gives the same user
    messages on stdin, the same replies to `next_frame` (text, blocks and images) and the same journal of what was
    sent and answered. What differs is the system prompt (the guide follows it after a blank line, and the journal's
    first prompt stays the reference agent's), the second MCP server in the one process, the permission for it, and the
    memory calls. Regression caught: a guide or a reply rule appended to a task or a user message (the memory would
    change what the model is asked, not only what it has), a memory server in the first process only, or a resume
    that forgets the memory."""
    case, held = small_case(tmp_path / "data"), memory(tmp_path / "spec")   # the one case, run twice
    plain = play(tmp_path / "without", monkeypatch, {0: early} if early else None, case=case)
    step = {**early, "memory": PLAN["memory"], "memory_ask": PLAN["memory_ask"]}
    with_it = play(tmp_path / "with", monkeypatch, {0: step, 1: {k: v for k, v in step.items() if k != "pulls"}}, held,
                   case=case)
    assert sent(plain) == sent(with_it)
    (alone,), (kept,) = plain.processes, with_it.processes
    assert len(alone["messages"]) == len(kept["messages"]) == (2 if early else 1)
    assert kept["system"] == f"{alone['system']}\n\n{GUIDE}"
    assert [p["servers"] for p in plain.processes] == [["stbench"]]
    assert [p["servers"] for p in with_it.processes] == [["mem", "stbench"]]   # one process: a resume has the same
    argv = kept["argv"]
    allowed = argv[argv.index("--allowedTools") + 1:argv.index("--permission-mode")]
    assert allowed == [*ALLOWED, "mcp__mem"] and kept["mcp"]["mem"]["type"] == "stdio"
    names = [(t, a) for t, a in made(with_it)]
    assert names[:3] == [("claim", {"text": "frame 0"}), ("view", {}), ("snapshot", {})] and len(names) > 6
    assert all(not c["error"] for c in kept["calls"])   # `--allowedTools` allowed every one of the memory's calls
    assert made(plain) == [] and not [e for e in plain.journal if e.get("event") in ("memory_call", "memory_failed")]
    assert [e["tool"] for e in with_it.journal if e.get("event") == "memory_call"][:3] == ["claim", "view", "snapshot"]
    assert with_it.summary["warnings"] == 0   # the memory's tools are not "tools offered beyond the server's"


def test_the_argv_and_the_mcp_config_add_the_memory_and_its_guide_and_nothing_else(tmp_path):
    """Contract (`claude_exec.Claude`): the argv of a process with a memory is the reference argv plus the memory's
    name in `--allowedTools` (`mcp__NAME`, the whole server) and the guide after the system prompt (a blank line
    between); every other flag and its place is the same; the `--mcp-config` file names the tool server and the memory's
    server (its command, its arguments and `stdio`) and no other. A transport with no guide leaves the prompt as it is.
    Regression caught: a memory whose tools are denied under `dontAsk` (the model could not use it), a guide that
    replaces the system prompt, a changed flag for the reference agent, or a second server with the tool server's
    name."""
    server = ("mem", ["C:/py/python.exe", "-m", "memory_server", "C:/state/mem"])
    plain, guided = Claude("sonnet", command=["claude"]), Claude("sonnet", command=["claude"], guide=GUIDE)
    without = plain.argv(Path("mcp.json"), "THE PROMPT")
    with_it = guided.argv(Path("mcp.json"), "THE PROMPT", memory=server)
    expected = list(without)
    expected.insert(expected.index("--permission-mode"), "mcp__mem")
    expected[-1] = f"THE PROMPT\n\n{GUIDE}"
    assert with_it == expected
    assert guided.argv(Path("mcp.json"), "THE PROMPT") == [*without[:-1], f"THE PROMPT\n\n{GUIDE}"]   # a guide alone
    assert plain.argv(Path("mcp.json"), "THE PROMPT", memory=server)[-1] == "THE PROMPT"
    config = guided.mcp_config(Path("server.json"), server)
    assert sorted(config["mcpServers"]) == ["mem", "stbench"]
    assert config["mcpServers"]["mem"] == {"type": "stdio", "command": "C:/py/python.exe",
                                          "args": ["-m", "memory_server", "C:/state/mem"]}
    assert plain.mcp_config(Path("server.json")) == {"mcpServers": {"stbench": config["mcpServers"]["stbench"]}}


# -- the event check --------------------------------------------------------------------------------------------------
def use(name: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": name, "input": {}}]}}


@pytest.mark.parametrize(("name", "memory_name", "outcome"), [
    ("mcp__mem__claim", "mem", None), ("mcp__stbench__next_frame", "mem", None),
    ("mcp__mem__claim", None, ToolEscaped), ("mcp__other__fetch", "mem", ToolEscaped),
    ("mcp__memory__claim", "mem", ToolEscaped), ("mcp__mem", "mem", ToolEscaped), ("Bash", "mem", ToolEscaped),
    ("ListMcpResourcesTool", "mem", None)])
def test_the_event_check_lets_the_memory_servers_calls_through_and_no_other_server_s(name, memory_name, outcome):
    """Contract (`claude_exec.check_event`): a call of the run's memory server (`mcp__NAME__TOOL`) passes, as the tool
    server's does; a call of any other server, of the memory's name when the run has no memory, of a server whose name
    only starts like it, of a built-in tool and Claude Code's resource helpers are as they were (the helpers pass).
    Regression caught: a memory that opens the door to every server, a prefix match that lets `memory` through for
    `mem`, or a reference run that lets a stray server name through."""
    if outcome is None:
        check_event(use(name), memory_name)
    else:
        with pytest.raises(outcome):
            check_event(use(name), memory_name)
    if memory_name is not None:   # a run with a memory is still held to the allow-list's own failures
        with pytest.raises(CodexError, match="held for approval"):
            check_event({"type": "control_request"}, memory_name)


def test_a_memory_calls_event_is_the_journals_one_shape():
    """Contract (`claude_exec.memory_uses`, `memory_results`): a `tool_use` of the memory's server and the `tool_result`
    that answers it become one `memory_call` event (server, tool without its `mcp__NAME__` prefix, arguments, the text
    blocks joined, each image block as the hash `scrubbed` made, `error` from `is_error`), in the order the answers
    come; a result as a plain string is its text; a call of the tool server, or a result that answers nothing the
    memory called, makes none. Regression caught: an error call journaled as a success, the prefix left on the tool,
    or an image written to the journal as base64."""
    started = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "mcp__mem__view", "input": {"q": 1}},
        {"type": "tool_use", "id": "b", "name": "mcp__mem__claim", "input": {"text": "t"}},
        {"type": "tool_use", "id": "c", "name": "mcp__stbench__next_frame", "input": {}}]}}
    pending = memory_uses(started, "mem")
    assert pending == {"a": ("view", {"q": 1}), "b": ("claim", {"text": "t"})} and memory_uses(started, None) == {}
    blocks = [{"type": "text", "text": "a"}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                                          "hash": "<jpeg 1 bytes sha256 x>"}},
              {"type": "text", "text": "b"}]
    answers = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "c", "content": "ignored"},
        {"type": "tool_result", "tool_use_id": "b", "is_error": True, "content": "no store"},
        {"type": "tool_result", "tool_use_id": "a", "content": blocks}]}}
    first, second = memory_results(answers, "mem", pending)
    keep = ("event", "server", "tool", "arguments", "error", "result", "images")
    assert {k: v for k, v in first.items() if k in keep} == {
        "event": "memory_call", "server": "mem", "tool": "claim", "arguments": {"text": "t"}, "error": True,
        "result": "no store", "images": []}
    assert {k: v for k, v in second.items() if k in keep} == {
        "event": "memory_call", "server": "mem", "tool": "view", "arguments": {"q": 1}, "error": False,
        "result": "a\nb", "images": ["<jpeg 1 bytes sha256 x>"]}
    assert pending == {} and memory_results(answers, "mem", pending) == [] and memory_results(answers, None, {}) == []


def test_the_tools_a_process_lists_are_not_a_warning_when_they_are_the_memory_s():
    """Contract (`claude_exec.digest`): the init event's tool list names tools beyond the tool server's as a warning,
    except the memory's own (`mcp__NAME__*`) and Claude Code's two resource helpers. Regression caught: a warning on
    every memory run (the summary would count a warning in each, and a real stray tool would drown in them)."""
    init = {"type": "system", "subtype": "init", "tools": [*ALLOWED, "mcp__mem__claim", "ListMcpResourcesTool",
                                                           "mcp__mem2__claim", "Bash"]}
    assert digest([init], [], "mem")["warnings"] == ["tools offered beyond the server's: mcp__mem2__claim, Bash"]
    assert digest([init], [])["warnings"] == [
        "tools offered beyond the server's: mcp__mem__claim, mcp__mem2__claim, Bash"]


# -- the journal and the leak scan ------------------------------------------------------------------------------------
def test_memory_calls_are_journaled_with_their_images_as_hashes_and_summarised_by_name(tmp_path, monkeypatch):
    """Contract: each completed call of the memory's tools is a `memory_call` event (server, tool, arguments, the
    reply's text, each image as `<jpeg N bytes sha256 HEX16>`, in the order of the other events); the turn record's
    events carry the images as hashes too, so no base64 of an image is anywhere in the journal; the summary counts the
    tool calls by name beside the tool server's (`claim` once for each frame) and the tool server's pulls are still its
    own. Regression caught: a memory image written to the journal as base64 (a journal of hundreds of megabytes), or
    memory calls missing from the summary (no way to count claims per frame)."""
    done = play(tmp_path, monkeypatch, {0: PLAN}, memory(tmp_path / "spec"))
    text = (done.out / "claude.calls.jsonl").read_text(encoding="utf-8")
    assert base64.b64encode(PICTURE).decode("ascii") not in text
    calls = [e for e in done.journal if e.get("event") == "memory_call"]
    assert [(c["server"], c["tool"]) for c in calls[:3]] == [("mem", "claim"), ("mem", "view"), ("mem", "snapshot")]
    assert calls[0]["arguments"] == {"text": "frame 0"} and calls[0]["result"] == "ok" and calls[0]["images"] == []
    assert calls[1]["result"].splitlines()[-1] == "frames 0" and not calls[1]["error"]
    assert calls[2]["result"] == "a picture"
    assert calls[2]["images"] == [f"<jpeg 20 bytes sha256 {hashlib.sha256(PICTURE).hexdigest()[:16]}>"]
    assert done.summary["tool_calls"] == {"next_frame": 7, "submit_answer": 2, "claim": 6, "view": 8, "snapshot": 6}
    assert done.summary["pulls"] == 7 and done.summary["memory"] == "mem"
    assert {c["result"] for c in calls if c["tool"] == "claim"} == {"ok"}


def test_the_leak_scan_reads_what_the_memory_showed_the_model(tmp_path, monkeypatch):
    """Contract (`codex_log.transcripts`, `llm_scan`): a memory's tool results are tool messages of the transcripts, in
    the order they were read, so a question's text in what the memory showed the model before that question's ask is a
    hit, as it is in an item the feed served; a clean run has none. Regression caught: memory results invisible to the
    scan (the store could carry a future question to the model unseen)."""
    done = play(tmp_path, monkeypatch, {0: PLAN}, memory(tmp_path / "spec"))
    questions = jsonl(done.case.questions)
    path = done.out / "claude.calls.jsonl"
    assert done.summary["scan"] == [] and scan_log(path, [], questions=questions) == []
    end = next(q for q in questions if q["when"] == "end")
    records = read_log(path)
    planted = next(r for r in records if r.get("event") == "memory_call" and r["tool"] == "view")
    planted["result"] += f"\n{end['text']}"
    bad = tmp_path / "planted.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    hits = scan_log(bad, [], questions=questions)
    assert hits and all("(tool)" in hit and f"question {end['qid']} before its ask" in hit for hit in hits), hits


def test_a_call_of_another_server_ends_a_memory_run_as_escaped_while_the_memory_s_own_calls_stand(tmp_path,
                                                                                                 monkeypatch):
    """Contract: in a run with a memory, the model calling the memory's tools is no escape (its calls are made and
    journaled), and a `tool_use` of a server that is neither the tool server nor the memory kills the process when the
    event arrives and makes the run invalid with an `escaped` failure. Regression caught: the memory's door left open to
    all servers, or the memory's own calls read as an escape."""
    plan = {0: {"server": "other", "tool": "x", "memory": [CLAIM]}}
    done = play(tmp_path, monkeypatch, plan, memory(tmp_path / "spec"), status=1)
    assert done.summary["invalid"] is True and done.summary["exec_failures"] == {"escaped": 1}
    last = done.journal[-1]
    assert last["event"] == "exec_failed" and last["reason"] == "escaped" and "mcp__other__x" in last["message"]
    assert (done.out / "claude.calls.jsonl").exists() and not (done.out / "rows.jsonl").exists()


# -- the sensor feed and the store ------------------------------------------------------------------------------------
def test_the_sensor_feed_is_one_process_sent_every_message_but_never_a_question(tmp_path, monkeypatch):
    """Contract (`codex_memory`, `codex_system`, unchanged for Claude): the feed is started once for the case, sent one
    JSON line for `start`, each `task` and each frame (`observe`, exactly one step), each answered by one line; it is
    never sent an ask, a question (no qid, no text, no label) or the end; each frame's image is an absolute path that
    exists while the message is handled and holds the frame; and it is closed when the case is. The frame is in the
    store as the model is served it, and not before: a `view` the model makes after being shown frame k lists exactly
    the frames 0 to k, never k + 1 (`test_codex_memory` pins it at the system). Regression caught: a feed run once for
    each message, one given the questions (the store would know what is asked), a frame fed after the model has it (a
    memory the model reads one frame behind), a frame fed when it is read ahead (a one-frame lookahead), or a process
    left running."""
    done = play(tmp_path, monkeypatch, {0: {"memory": [VIEW]}}, memory(tmp_path / "spec"))
    log = sensed(done)
    messages = [m for m in log if "message" in m]
    kinds = [next(iter(m["message"])) for m in messages]
    assert kinds == ["start", "task"] + ["observe"] * 6
    assert [m["n"] for m in messages] == list(range(1, 9)) and len({m["pid"] for m in log}) == 1
    assert log[-1] == {"pid": log[0]["pid"], "closed": True}
    frames = [m for m in messages if "observe" in m["message"]]
    assert [m["message"]["observe"]["steps"][0]["k"] for m in frames] == [0, 1, 2, 3, 4, 5]
    assert all(m["rgb"]["absolute"] and m["rgb"]["exists"] for m in frames)
    questions = jsonl(done.case.questions)
    for text in [q[k] for q in questions for k in ("qid", "text")] + ["Questions", "Q1"]:
        assert text not in json.dumps(log)
    views = [[int(n) for n in c["text"].splitlines()[-1].split()[1:]]
             for c in done.processes[0]["calls"] if c["server"] == "mem"]
    assert views == [list(range(shown + 1)) for shown in range(6)]   # exactly the frames served so far, frames 3 and 5
    # (served with their questions) included


def test_the_store_is_kept_beside_the_case_without_its_cache(tmp_path, monkeypatch):
    """Contract (`codex_system.close`, unchanged for Claude): when the case closes the memory's store is copied beside
    the case's files as `<name>-store` with what the memory and the feed wrote (claims, the server's log, the feed's
    log) and without its `cache/`; the memory's server started once, for the whole case (one process), and was
    called for every claim. Regression caught: a store lost with the case's temp folder (nothing to look at after the
    run), a cache copied (megabytes of rebuildable index), or a server started again for a resume."""
    done = play(tmp_path, monkeypatch, {0: {"pulls": 3, "memory": [CLAIM]}, 1: {"memory": [CLAIM]}},
                memory(tmp_path / "spec"))
    kept = done.out / "mem-store"
    assert sorted(p.name for p in kept.iterdir()) == ["claims.jsonl", "sense.jsonl", "server.jsonl"]
    claims = [json.loads(line) for line in (kept / "claims.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["text"] for c in claims] == [f"frame {k}" for k in range(6)]
    served = (kept / "server.jsonl").read_text(encoding="utf-8").splitlines()
    assert len({json.loads(line)["pid"] for line in served}) == 1
    assert done.summary["execs"] == {"total": 2, "by_kind": {"start": 1, "resume": 1}}


@pytest.mark.parametrize(("fault", "why"), [
    ("exit:5", r"ended \(exit 3\) on 'observe': the feed broke"), ("text:5", "printed no JSON object on 'observe'")])
def test_a_feed_that_fails_ends_the_case_as_memory(tmp_path, monkeypatch, fault, why):
    """Contract (`codex_system`, unchanged for Claude): a feed that exits or prints a line that is no JSON object ends
    the case as `TransportError("memory")` at the message it failed on (here the third frame, the fifth message),
    journals `memory_failed` with that message and the reason, the summary keeps the reason, and the model was not
    shown the frame it failed on (frame 2, fed as it was to be served) and has the two before it. Regression caught: a
    dead feed that lets the case go on with a store that stopped learning (the memory arm would measure nothing)."""
    done = play(tmp_path, monkeypatch, None, memory(tmp_path / "spec", fault=fault), status=1)
    (failed,) = [e for e in done.journal if e.get("event") == "memory_failed"]
    assert failed["message"] == "observe" and re.search(why, failed["error"]), failed
    assert done.summary["status"].endswith("TransportError: memory")
    assert done.summary["memory_failure"] == failed["error"]
    (only,) = done.processes
    assert [c["text"].split(" ")[0] for c in pulled(only) if c["text"].startswith("[k")] == ["[k0", "[k1"]


# -- the options a memory cannot be given -----------------------------------------------------------------------------
def test_a_memory_is_refused_where_it_cannot_be_fed_or_loaded_and_nothing_is_made(tmp_path, monkeypatch, capsys):
    """Contract (`cmd_run.classify`, `cmd_run.load_memory`, for `claude:`): `--memory` is refused with exit 2 on the
    detections channel and with `--text-only` (the memory is fed the camera images), for a SPEC that cannot be read and
    for a system that is no Codex or Claude agent, before anything is sent or made: no output folder, no process.
    Regression caught: a memory run on a channel that sends it no image (a store that sees nothing, measured as if it
    saw), or a SPEC mistake found after the first message."""
    case = small_case(tmp_path / "data")
    held, fake = memory(tmp_path / "spec"), Fake(tmp_path)
    images = "--memory feeds the memory the camera images"
    cases = [(["claude:sonnet", "--channel", "detections"], held.path, images),
             (["claude:sonnet", "--text-only"], held.path, images),
             (["all_unknown"], held.path, "--memory applies to claude or codex agents"),
             (["claude:sonnet"], tmp_path / "no-such.json", "cannot read the memory SPEC")]
    for options, spec, message in cases:
        out = tmp_path / "out"
        system, *rest = options
        assert claude_run(monkeypatch, fake, system, case.private, *rest, "--memory", spec, "--out", out) == 2, options
        assert message in capsys.readouterr().err and not out.exists(), options
    assert fake.processes() == []


def test_eval_runs_the_claude_agent_with_a_memory_and_names_it(tmp_path, monkeypatch):
    """Contract (`eval_cmd`, `eval_results`, `cmd_claude`): `eval claude:sonnet --memory SPEC` hands every case's runner
    the SPEC, so the process has the memory's server and guide; `results.json` holds `"memory": "<name>"` (the SPEC's
    name, not its path), `results.md` names the system `claude:sonnet with memory mem`, a default folder gains
    `-memory-<SPEC's stem>`, and the case keeps its store beside its files. Regression caught: `eval` dropping the
    memory for a Claude agent (a row labelled with a memory it did not have, or the reverse), or a memory row whose
    results read as the reference's."""
    data, held = tmp_path / "data", memory(tmp_path / "spec")
    case = compiled(data, short_script(1), [end_q(1)])
    fake = Fake(tmp_path)
    monkeypatch.setattr(cmd_claude, "make_claude", lambda model, reasoning, **given: fake.new(
        model=model, reasoning=reasoning, **given))
    assert eval_cli("claude:sonnet", "--data", data, "--memory", held.path) == 0
    out = data / "runs" / "claude_sonnet-rgb_pose-memory-memory"
    report = (out / "results.md").read_text(encoding="utf-8")
    assert results(out)["memory"] == "mem" and "claude:sonnet with memory mem" in report
    (only,) = fake.processes()
    assert only["servers"] == ["mem", "stbench"] and only["system"].endswith(GUIDE)
    where = out / "cases" / case.id
    assert (where / "mem-store" / "sense.jsonl").is_file()
    assert jsonl(where / "claude.calls.jsonl")[0]["memory"]["name"] == "mem"
