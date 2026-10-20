"""A memory beside the Codex reference agent (`codex_memory`, `codex_system`, `codex_exec`, `codex_live`, `codex_log`;
`--memory SPEC`): a second MCP server in the exec, its guide in the base instructions, and the robot's sensors fed to
its store by one long-lived process. Over the fake command (`codex_fake`), a tiny memory server and a tiny sensor feed
(`memory_fake_server`, `memory_fake_sense`); no model, no `codex`, no network, no real memory system. The invariant that
matters most is that nothing the model is sent changes with a memory (`test_a_memory_changes_no_message...`). Expected
values are written by hand."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

from codex_kit import CAMERA, CREDIT, GUIDE, ROBOT, Fake, Run, jpeg_hash, memory, pulled
from stbench import codex_memory
from stbench.codex_exec import BASE_INSTRUCTIONS, Codex, CodexError, ToolEscaped, check_event
from stbench.codex_log import memory_call, read_log
from stbench.codex_memory import MemoryFailure, feed_step, load
from stbench.codex_system import Settings
from stbench.llm_scan import scan_log
from stbench.llm_summary import summarize
from stbench.stream_protocol import TransportError

FIRST, BRIEF = "2026-03-02T08:00:00", "Your assignment is to clean up the café."
COUNT = {"qid": "a1b2c3d4e5f60718", "text": "How many frames have you seen?", "answer_type": "count"}
POSITION = {"qid": "0f1e2d3c4b5a6978", "text": "Where is the robot now?", "answer_type": "position"}
PICTURE = b"\xff\xd8\xff\xe0" + b"memory-picture" + b"\xff\xd9"  # what the fake memory server's `snapshot` shows back
CLAIM = {"server": "mem", "tool": "claim", "arguments": {"text": "frame {k}"}}
VIEW = {"server": "mem", "tool": "view", "arguments": {}}
SNAPSHOT = {"server": "mem", "tool": "snapshot", "arguments": {}}
PLAN = {"memory": [CLAIM, VIEW, SNAPSHOT], "memory_ask": [VIEW]}


def scenario(run: Run) -> None:
    """Six frames in two tasks, a during ask at frame 2 (it joins that frame) and the end ask at frame 5."""
    system = run.system
    system.task(0, FIRST, 0.0, BRIEF)
    run.stream([0, 1, 2])
    system.ask("during", {"k": 2, "clock": "2026-03-02T08:00:02"}, [COUNT])
    run.stream([3])
    system.task(1, "2026-03-02T08:10:00", 596.0, "Now sweep the hall.")
    run.stream([4, 5])
    system.ask("end", {"k": 5, "clock": "2026-03-02T08:00:05"}, [POSITION])
    system.finish()


def sensed(run: Run, name: str = "mem") -> list[dict]:
    """The lines the fake sensor feed wrote in the case's store, in order."""
    path = run.state / name / "sense.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def frames(record: dict) -> list[str]:
    """The `[kN` headers of the frames an exec has been shown, in order."""
    return [c["text"].split(" ")[0] for c in pulled(record) if c["text"].startswith("[k")]


def sent(execs: list[dict], log: list[dict]) -> dict:
    """Everything the model was sent, as the fake and the journal saw it, without the clocks."""
    return {"prompts": [e["prompt"] for e in execs],
            "pulls": [[(c["text"], c["blocks"], c["images"]) for c in pulled(e)] for e in execs],
            "journal": [{k: v for k, v in e.items() if k not in ("at", "time")} for e in log
                        if e.get("event") in ("exec_started", "served", "answer_outcome")]}


# -- the SPEC ---------------------------------------------------------------------------------------------------------
def spec_file(folder: Path, **changes) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "guide.md").write_text("How to use it.\n", encoding="utf-8")
    spec = {"name": "mem", "serve": ["python", "serve.py", "{store}"], "sense": ["python", "sense.py", "{store}"],
            "prompt": "guide.md", **changes}
    (folder / "memory.json").write_text(json.dumps({k: v for k, v in spec.items() if v is not None}), encoding="utf-8")
    return folder / "memory.json"


def test_a_spec_is_loaded_with_its_guide_and_the_store_and_model_filled_in(tmp_path):
    """Contract (`codex_memory`): a SPEC with `name`, `serve`, optional `sense` and a `prompt` file beside it loads to a
    memory whose guide is the file's stripped text, whose server argv has the case's store (`<state>/<name>`) and the
    model id filled in, and whose record (the journal's) holds the name, both argvs as loaded and the guide's text and
    nothing else: no digest of any kind. Regression caught: a store path that is not under the state folder (it would
    not be counted, nor carried away), a placeholder left in the argv, or a stamp that is a hash and not the text."""
    path = spec_file(tmp_path / "s", serve=["python", "-m", "serve", "--model", "{model}", "{store}"], sense=None)
    held = load(path)
    assert (held.name, held.guide, held.sense, held.path) == ("mem", "How to use it.", None, path.resolve())
    state = tmp_path / "state"
    assert held.server(state, "some-model") == ("mem", ["python", "-m", "serve", "--model", "some-model",
                                                         str(state / "mem")])
    assert held.open_sense(state, "some-model") is None
    assert held.record() == {"name": "mem", "serve": ["python", "-m", "serve", "--model", "{model}", "{store}"],
                             "sense": None, "guide": "How to use it."}
    assert load(spec_file(tmp_path / "t")).record()["sense"] == ["python", "sense.py", "{store}"]


@pytest.mark.parametrize(("changes", "message"), [
    ({"name": "MyMemory"}, "name must match"), ({"name": "x" * 40}, "name must match"),
    ({"name": "stbench"}, "none of \\['stbench', 'codex'\\]"), ({"name": "codex"}, "none of"),
    ({"serve": ["python", "serve.py"]}, "serve must name the case's store as \\{store\\}"),
    ({"serve": ["python", "it's", "{store}"]}, "quote or a line break"),
    ({"serve": "python serve.py {store}"}, "serve is a non-empty list of strings"),
    ({"serve": []}, "serve is a non-empty list of strings"),
    ({"sense": ["python", ""]}, "sense is a non-empty list of strings"),
    ({"prompt": "missing.md"}, "cannot read the prompt"),
    ({"extra": 1}, "unknown keys \\['extra'\\]")])
def test_a_spec_that_cannot_be_used_is_refused_saying_why(tmp_path, changes, message):
    """Contract (`codex_memory.load`): a SPEC with a bad name (the tool server's, Codex's own, or not a plain
    identifier), a `serve` that is no argv or does not name `{store}` or that Codex could not be passed, a bad `sense`,
    a missing prompt file or a key it does not know is refused with `MemoryFailure` before anything runs. Regression
    caught: a server that would shadow the tool server's name, a memory whose store is nowhere in its command (it would
    share one folder between cases), or a SPEC mistake that surfaces as a failed exec in the middle of a run."""
    with pytest.raises(MemoryFailure, match=message):
        load(spec_file(tmp_path / "s", **changes))
    empty = tmp_path / "e"
    empty.mkdir()
    (empty / "guide.md").write_text(" \n", encoding="utf-8")
    (empty / "memory.json").write_text(json.dumps({"name": "m", "serve": ["x", "{store}"], "prompt": "guide.md"}),
                                       encoding="utf-8")
    with pytest.raises(MemoryFailure, match="is empty"):
        load(empty / "memory.json")
    for text in ("{", "[]"):
        (empty / "memory.json").write_text(text, encoding="utf-8")
        with pytest.raises(MemoryFailure, match="cannot read the memory SPEC|is an object with name"):
            load(empty / "memory.json")


# -- what the model is sent -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("early", [{}, {"pulls": 3}], ids=["one exec", "a resume"])
def test_a_memory_changes_no_message_the_model_is_sent(tmp_path, early):
    """Contract: with a memory every message and item the model is sent (the first prompt, each task, frame, ask,
    resume line and the end item) is byte for byte what the reference agent is sent: the same case run without and with
    a memory, the model calling the memory's tools after every frame and at every ask, gives the same first prompt, the
    same replies to `next_frame` (text, blocks and images) and the same journal of what was sent and answered. What
    differs is the base instructions (the guide follows the one line), the second MCP server, and the memory calls, in
    the execs of the thread, the resume included. Regression caught: a guide or a reply rule appended to the system
    prompt or to a task (the memory would change what the model is asked, not only what it has), a memory server in the
    first exec only, or a resume that forgets the memory."""
    runs = []
    for name, held in (("without", None), ("with", "mem")):
        folder = tmp_path / name
        folder.mkdir()
        step = {**early, **({"memory": PLAN["memory"], "memory_ask": PLAN["memory_ask"]} if held else {})}
        extra = {"memory": memory(folder / "spec")} if held else {}
        with Run(folder, {0: step, 1: {k: v for k, v in step.items() if k != "pulls"}}, asks_during=True, **CREDIT,
                 **extra) as run:
            scenario(run)
            execs = run.fake.execs()
        runs.append((execs, read_log(run.log)))
    (plain, plain_log), (held, held_log) = runs
    assert sent(plain, plain_log) == sent(held, held_log)
    assert len(plain) == len(held) == (2 if early else 1)
    assert [e["instructions"] for e in plain] == [BASE_INSTRUCTIONS] * len(plain)
    assert [e["instructions"] for e in held] == [f"{BASE_INSTRUCTIONS}\n\n{GUIDE}"] * len(held)
    assert [e["servers"] for e in plain] == [["stbench"]] * len(plain)
    assert [e["servers"] for e in held] == [["mem", "stbench"]] * len(held)  # every exec, a resume included
    made = [(c["tool"], c["arguments"]) for c in held[0]["calls"] if c["server"] == "mem"]
    assert made[:3] == [("claim", {"text": "frame 0"}), ("view", {}), ("snapshot", {})] and len(made) > 6
    assert not [c for e in plain for c in e["calls"] if c["server"] != "stbench"]
    assert not [e for e in plain_log if e.get("event") in ("memory_call", "memory_failed")]
    assert [e["tool"] for e in held_log if e.get("event") == "memory_call"][:3] == ["claim", "view", "snapshot"]


def test_a_memory_adds_its_server_to_the_argv_and_its_guide_to_the_instructions_and_nothing_else(tmp_path):
    """Contract (`codex_exec`): the argv of an exec with a memory is the reference argv plus the memory's server,
    configured as the tool server is (required, its calls pre-approved, the startup timeout, the tool timeout of 600 s),
    in the same exec, and no other setting or flag differs; the instructions file holds the one line alone without a
    guide and the line, a blank line and the guide with one. Regression caught: a memory server that is not required
    (a store that did not start would go unseen), calls that wait for an approval nobody gives, a changed flag for the
    reference agent, or a guide that replaces the base instruction."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    plain, guided = Fake(tmp_path / "a"), Fake(tmp_path / "b", guide=GUIDE)
    plain.codex.prepare()
    guided.codex.prepare()
    assert Path(plain.codex.instructions).read_text(encoding="utf-8") == BASE_INSTRUCTIONS
    assert Path(guided.codex.instructions).read_text(encoding="utf-8") == f"{BASE_INSTRUCTIONS}\n\n{GUIDE}"
    store = tmp_path / "state" / "mem"
    server = ("mem", [sys.executable, "-m", "memory_server", str(store)])
    without = guided.codex.argv("start", None, tmp_path / "server.json", tmp_path / "empty")
    with_it = guided.codex.argv("start", None, tmp_path / "server.json", tmp_path / "empty", memory=server)
    resumed = guided.codex.argv("resume", "thread-0000", tmp_path / "server.json", tmp_path / "empty", memory=server)

    def settings(argv):
        return [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]

    def flags(argv):
        return [a for i, a in enumerate(argv) if a != "-c" and (i == 0 or argv[i - 1] != "-c")]

    ours = [s for s in settings(with_it) if s.startswith("mcp_servers.mem.")]
    assert ours == [f"mcp_servers.mem.command='{Path(sys.executable).as_posix()}'",
                    f"mcp_servers.mem.args=['-m','memory_server','{store.as_posix()}']",
                    "mcp_servers.mem.required=true", "mcp_servers.mem.default_tools_approval_mode=approve",
                    "mcp_servers.mem.startup_timeout_sec=60", "mcp_servers.mem.tool_timeout_sec=600"]
    assert [s for s in settings(with_it) if s not in ours] == settings(without)
    assert flags(with_it) == flags(without)  # every flag, `--disable` and the folder, the same
    assert {s.split(".")[1] for s in settings(resumed) if s.startswith("mcp_servers.")} == {"mem", "stbench"}
    assert flags(resumed)[-3:] == ["--", "thread-0000", "-"]
    assert {s.split(".")[1] for s in settings(without) if s.startswith("mcp_servers.")} == {"stbench"}


# -- the sensor feed --------------------------------------------------------------------------------------------------
def test_the_sensor_feed_is_one_process_sent_every_message_but_never_a_question(tmp_path):
    """Contract (`codex_memory`): the feed is started once for the case, before anything is sent to the model, and sent
    one JSON line for `start` (the public camera and the robot), each `task` (index, clock, gap, brief) and each frame
    (`observe`: the camera and exactly one step), each answered by one line; it is never sent an ask, a question (no
    qid, no question text, no label) or the end; non-ASCII text arrives whole; a step it is sent holds its whitelisted
    keys and nothing else, however many privileged keys the step carries (`identity`, `detections`, `depth`), with the
    image as an absolute path of a file that exists while the message is handled and holds the frame; and it is closed
    when the system is. Regression caught: a feed run once for each message (a start-up each frame), one given the
    questions (the store would know what is asked), a step copied whole (privileged keys reach the store), a relative
    image path, or a process left running."""
    held = memory(tmp_path / "spec")
    with Run(tmp_path, memory=held, asks_during=True, **CREDIT) as run:
        assert [m["message"] for m in sensed(run)] == [{"start": {"camera": CAMERA, "robot": ROBOT}}]  # before any exec
        assert not (run.fake.home / "execs.jsonl").exists()
        run.system.task(0, FIRST, 0.0, BRIEF)
        for k in range(3):
            (step,) = run.batch([k])
            step |= {"identity": {"d1": "E01"}, "detections": [{"kind": "mug"}], "depth": "inbox/depth/x.png",
                     "velocity": [0.0, -0.25], "subgoal": "cleaning the kitchen"}
            run.system.observe([step])
        run.system.ask("during", {"k": 2, "clock": "2026-03-02T08:00:02"}, [COUNT])
        run.system.task(1, "2026-03-02T08:10:00", 596.0, "Now sweep the hall.")
        run.stream([3])
        run.system.ask("end", {"k": 3, "clock": "2026-03-02T08:00:03"}, [POSITION])
        run.system.finish()
    log = sensed(run)  # after the system closed
    messages = [m for m in log if "message" in m]
    assert [next(iter(m["message"])) for m in messages] == ["start", "task", "observe", "observe", "observe", "task",
                                                             "observe"]
    assert messages[1]["message"] == {"task": {"index": 0, "clock": FIRST, "gap_s": 0.0, "brief": BRIEF}}
    assert messages[5]["message"]["task"]["gap_s"] == 596.0
    assert [m["n"] for m in messages] == list(range(1, 8)) and len({m["pid"] for m in log}) == 1
    assert log[0]["pid"] != os.getpid() and log[-1] == {"pid": log[0]["pid"], "closed": True}  # closed with the system
    for m, k in zip((messages[2], messages[3], messages[4], messages[6]), (0, 1, 2, 3), strict=True):
        body = m["message"]["observe"]
        assert body["camera"] == CAMERA and len(body["steps"]) == 1
        (step,) = body["steps"]
        plain = {"k", "clock", "pose", "camera", "status", "rgb"}
        assert set(step) == plain | ({"velocity", "subgoal"} if k < 3 else set())
        assert set(step["pose"]) == {"pos", "yaw_deg"} and set(step["camera"]) == {"pos", "rot", "pitch_deg"}
        assert Path(step["rgb"]).name == f"{k:06d}.jpg" and m["rgb"] == {"absolute": True, "exists": True,
                                                                        "sha": jpeg_hash(k)}
    for text in (COUNT["qid"], COUNT["text"], POSITION["qid"], POSITION["text"], "Questions", "Q1"):
        assert text not in json.dumps(log)


def test_the_memory_holds_exactly_the_frames_the_model_has_been_served(tmp_path):
    """Contract (`codex_system._feed_held`): a frame is written to the sense feed as it is served to the model, and not
    when the runner reads ahead to the next message to learn whether questions join it. So whenever `observe` of
    frame k has returned, the feed holds the frames the model has been shown (the ones before k: frame k is held), and
    whenever the model is served a frame (alone, or joined with its questions, a during ask or the end one) a view of
    the store it makes then lists exactly the frames served so far, frame k included and frame k + 1 not. Regression
    caught: a frame fed when it is read in (a memory that holds a frame the model has not seen yet: a one-frame
    lookahead in every memory arm), a frame fed after the model has it, or one fed that is never served."""
    plan = {0: {"memory": [VIEW], "memory_ask": [VIEW]}}
    with Run(tmp_path, plan, memory=memory(tmp_path / "spec"), asks_during=True, **CREDIT) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        for k in range(4):
            run.system.observe(run.batch([k]))
            fed = [m["message"]["observe"]["steps"][0]["k"] for m in sensed(run) if "observe" in m.get("message", {})]
            (live,) = run.fake.execs()
            assert fed == list(range(k)) and frames(live) == [f"[k{n}" for n in range(k)]
        run.system.ask("during", {"k": 3, "clock": "2026-03-02T08:00:03"}, [COUNT])   # joins frame 3, held
        run.stream([4, 5])
        run.system.ask("end", {"k": 5, "clock": "2026-03-02T08:00:05"}, [POSITION])   # joins frame 5, held
        run.system.finish()
        (live,) = run.fake.execs()
    views = [[int(n) for n in c["text"].splitlines()[-1].split()[1:]] for c in live["calls"] if c["server"] == "mem"]
    # a view after each frame, and one at each ask, which is served with the frame it is about (3 and 5)
    assert views == [list(range(k + 1)) for k in (0, 1, 2, 3, 3, 4, 5, 5)]


@pytest.mark.parametrize(("fault", "why"), [
    ("exit:3", r"ended \(exit 3\) on 'observe': the feed broke"), ("eof:3", r"ended \(exit 0\) on 'observe'"),
    ("text:3", "printed no JSON object on 'observe': 'not json'"),
    ("array:3", r"printed no JSON object on 'observe': '\[1, 2\]'"),
    ("hang:3", "did not answer 'observe' within 1 s")])
def test_a_feed_that_fails_ends_the_case_as_memory(tmp_path, monkeypatch, fault, why):
    """Contract: a feed that exits non-zero, closes its output, prints a line that is not a JSON object (text, or an
    array) or does not answer within `SENSE_TIMEOUT` ends the case as `TransportError("memory")` at the message it
    failed on, journals `memory_failed` with that message and the reason (the exit code and the end of its stderr, the
    line, the time), and the summary keeps the reason (a case whose journal is removed for having spent nothing is
    otherwise silent). Regression caught: a dead feed that lets the case go on with a store that stopped learning (the
    memory arm would measure nothing), or a hang that is waited for until the case's 12 hours."""
    with Run(tmp_path, memory=memory(tmp_path / "spec", fault=fault)) as run:  # 1 start, 2 task, 3 the first frame
        run.system.task(0, FIRST, 0.0, BRIEF)
        monkeypatch.setattr(codex_memory, "SENSE_TIMEOUT", 1.0)  # the feed is up: only the one that hangs waits it out
        began = time.monotonic()
        run.system.observe(run.batch([0]))   # held, not yet fed
        with pytest.raises(TransportError, match="^memory$") as ended:
            run.system.observe(run.batch([1]))   # frame 0 is fed as it is served, and the feed fails on it
    assert time.monotonic() - began < 30 and isinstance(ended.value.__cause__, MemoryFailure)
    (failed,) = [e for e in read_log(run.log) if e.get("event") == "memory_failed"]
    assert failed["message"] == "observe" and re.search(why, failed["error"]), failed
    assert re.search(why, summarize(run.log)["memory_failure"])
    assert frames(run.fake.execs()[0]) == []  # the model was not shown the frame


def test_a_feed_that_cannot_start_ends_the_case_before_anything_is_sent(tmp_path):
    """Contract: a `sense` command that cannot be started is a `memory` failure at `start`, journaled, and no exec has
    begun. Regression caught: a missing feed discovered at the first frame, after the model was already asked."""
    held = dataclasses.replace(memory(tmp_path / "spec"), sense=(str(tmp_path / "no-such-program"), "{store}"))
    with pytest.raises(TransportError, match="^memory$"):
        Run(tmp_path, memory=held)
    journal = read_log(tmp_path / "codex.calls.jsonl")
    assert [(e["event"], e["message"]) for e in journal] == [("memory_failed", "start")]
    assert not (tmp_path / "fake-codex" / "execs.jsonl").exists()


def test_feed_step_keeps_only_what_the_header_shows_and_the_camera_pose(tmp_path):
    """Contract (`codex_memory.feed_step`): a whitelist, by name and one level down: a step is reduced to its number,
    clock, base position and heading, velocity, camera position, rotation and pitch, status, subgoal and the absolute
    path of its image; any other key (privileged or new), and any other key of the pose or the camera, is dropped.
    Regression caught: `{**step}` (anything a later build adds to a step reaches the store)."""
    step = {"k": 4, "clock": "c", "pose": {"pos": [1, 2, 0], "rot": [1, 0, 0, 0], "yaw_deg": 90.0, "secret": 1},
            "camera": {"pos": [1, 2, 1], "rot": [1, 0, 0, 0], "pitch_deg": 5.0, "tilt": 1}, "status": {"phase": "x"},
            "rgb": "inbox/rgb/000004.jpg", "identity": {"d": "E1"}, "detections": [], "depth": "d.png", "extra": 1}
    assert feed_step(step, tmp_path) == {
        "k": 4, "clock": "c", "pose": {"pos": [1, 2, 0], "yaw_deg": 90.0}, "status": {"phase": "x"},
        "camera": {"pos": [1, 2, 1], "rot": [1, 0, 0, 0], "pitch_deg": 5.0},
        "rgb": str(tmp_path / "inbox/rgb/000004.jpg")}


# -- the tools --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("event", "memory_name", "outcome"), [
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mem", "tool": "claim"}}, "mem", None),
    ({"type": "item.started", "item": {"type": "mcp_tool_call", "server": "mem", "tool": "claim"}}, "mem", None),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "stbench", "tool": "next_frame"}}, "mem",
     None),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mem", "tool": "claim"}}, None,
     ToolEscaped),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "other", "tool": "fetch"}}, "mem",
     ToolEscaped),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "claim"}}, None, ToolEscaped),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "codex", "tool": "read_mcp_resource"}},
     "mem", ToolEscaped),
    ({"type": "item.started", "item": {"type": "command_execution", "command": "ls"}}, "mem", ToolEscaped),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mem", "tool": "claim",
                                         "error": {"message": "tool call requires approval"}}}, "mem", CodexError)])
def test_the_event_check_lets_the_memory_servers_calls_through_and_no_other_server_s(event, memory_name, outcome):
    """Contract (`codex_exec.check_event`): a call of the run's memory server passes, as the tool server's does; a call
    of any other server, of this memory's name when the run has no memory, one with no server, any other item and
    Codex's resource reader are `ToolEscaped`, and a memory call held for approval is an error. Regression caught: a
    memory that opens the door to every server, or a reference run that lets a stray server name through."""
    if outcome is None:
        check_event(event, memory_name)
    else:
        with pytest.raises(outcome):
            check_event(event, memory_name)


@pytest.mark.skipif(os.name != "nt", reason="a .cmd shim is run by cmd.exe on Windows only")
def test_a_cmd_shim_is_not_given_a_memory_server_that_cmd_exe_would_misread(tmp_path):
    """Contract (`Codex.popen`): the memory's server settings are arguments of the same command line as the tool
    server's, so the check for a `.cmd` shim (cmd.exe reads `&`, `|`, `%`, ... as syntax) covers them: an argv whose
    memory server holds such a character is refused with the reason, before anything is run. Regression caught: a
    memory path with `&` that cmd.exe splits into a second command."""
    fake = Fake(tmp_path)
    fake.codex.prepare()
    shim = tmp_path / "codex.cmd"
    server = ("mem", [sys.executable, "-m", "memory_server", str(tmp_path / "a&b")])
    argv = fake.codex.argv("start", None, tmp_path / "server.json", tmp_path / "empty", memory=server)
    with pytest.raises(CodexError, match="cmd.exe reads as syntax"):
        Codex("fake-model", command=[str(shim)]).popen([str(shim), *argv[len(fake.codex.command):]], tmp_path, None)


def test_a_call_of_another_server_ends_a_memory_run_as_escaped_while_the_memory_s_own_calls_stand(tmp_path):
    """Contract: in a run with a memory, the model calling the memory's tools is no escape (its first call is made and
    journaled), and a call of a server that is neither the tool server nor the memory kills the exec when the event
    arrives and ends the case as `TransportError("escaped")`. Regression caught: the memory's door left open to all
    servers, or the memory's own calls read as an escape."""
    plan = {0: {"server": "other", "tool": "x", "escape_at": 1, "hang": 1, "hang_s": 60, "memory": [CLAIM]}}
    with Run(tmp_path, plan, memory=memory(tmp_path / "spec")) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.system.observe(run.batch([0]))
        with pytest.raises(TransportError, match="^escaped$") as escaped:
            run.serve()
    assert isinstance(escaped.value.__cause__, ToolEscaped) and "'other'" in str(escaped.value.__cause__)
    assert [e["tool"] for e in read_log(run.log) if e.get("event") == "memory_call"] == ["claim"]


def test_memory_calls_are_journaled_with_their_images_as_hashes_and_summarised_by_name(tmp_path):
    """Contract: each completed call of the memory's tools is a `memory_call` event (server, tool, arguments, the
    reply's text, each image as `<jpeg N bytes sha256 HEX16>`, in the order of the other events), the exec record's
    events carry the images as hashes too, so no base64 of an image is anywhere in the journal; the summary counts the
    tool calls by name beside the tool server's (`claim` once for each frame) and the tool server's pulls are still
    its own; and the calls do not count toward `max_calls_item` (here 1, with three memory calls after every frame).
    Regression caught: a memory image written to the journal as base64 (a journal of hundreds of megabytes), memory
    calls missing from the summary (no way to count claims per frame), or memory calls capped as the pull's are."""
    with Run(tmp_path, {0: PLAN}, memory=memory(tmp_path / "spec"), settings=Settings(max_calls_item=1),
             asks_during=True, **CREDIT) as run:
        scenario(run)
    text = run.log.read_text(encoding="utf-8")
    assert base64.b64encode(PICTURE).decode("ascii") not in text and "base64" not in text
    calls = [e for e in read_log(run.log) if e.get("event") == "memory_call"]
    assert [(c["server"], c["tool"]) for c in calls[:3]] == [("mem", "claim"), ("mem", "view"), ("mem", "snapshot")]
    assert calls[0]["arguments"] == {"text": "frame 0"} and calls[0]["result"] == "ok" and calls[0]["images"] == []
    assert calls[1]["result"].splitlines()[-1] == "frames 0" and not calls[1]["error"]
    assert calls[2]["result"] == "a picture"
    assert calls[2]["images"] == [f"<jpeg 20 bytes sha256 {hashlib.sha256(PICTURE).hexdigest()[:16]}>"]
    summary = summarize(run.log)
    assert summary["tool_calls"] == {"next_frame": 8, "submit_answer": 4, "claim": 6, "view": 8, "snapshot": 6}
    assert summary["tool_errors"] == 2 and summary["pulls"] == 8  # the two repeated submissions the fake makes
    assert {c["result"] for c in calls if c["tool"] == "claim"} == {"ok"}


def test_the_leak_scan_reads_what_the_memory_showed_the_model(tmp_path):
    """Contract (`codex_log.transcripts`, `llm_scan`): a memory's tool results are tool messages of the transcripts, in
    the order they were read, so a question's text in what the memory showed the model before that question's ask is a
    hit, as it is in an item the feed served; a clean run has none. Regression caught: memory results invisible to the
    scan (the store could carry a future question to the model unseen)."""
    with Run(tmp_path, {0: PLAN}, memory=memory(tmp_path / "spec"), asks_during=True, **CREDIT) as run:
        scenario(run)
    questions = [COUNT, POSITION]
    assert scan_log(run.log, [], questions=questions) == []
    records = read_log(run.log)
    planted = next(r for r in records if r.get("event") == "memory_call" and r["tool"] == "view")
    planted["result"] += f"\n{POSITION['text']}"
    path = tmp_path / "planted.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    hits = scan_log(path, [], questions=questions)
    assert hits and all("(tool)" in hit and f"question {POSITION['qid']} before its ask" in hit for hit in hits), hits


def test_a_memory_calls_event_is_the_journals_one_shape(tmp_path):
    """Contract (`codex_log.memory_call`): a completed call becomes the event with its text blocks joined, its image
    blocks as the hashes `scrubbed` made, and `error` true for a call that failed (an error message, or status
    failed), whose `result` is then the error. Regression caught: an error call journaled as an empty success."""
    blocks = [{"type": "text", "text": "a"}, {"type": "image", "hash": "<jpeg 1 bytes sha256 x>"},
              {"type": "text", "text": "b"}]
    ok = {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mem", "tool": "view", "arguments": {},
                                             "result": {"content": blocks}}}
    bad = {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mem", "tool": "claim",
                                              "arguments": {"text": "t"}, "result": None, "status": "failed",
                                              "error": {"message": "no store"}}}
    first, second = memory_call(ok), memory_call(bad)
    assert {k: v for k, v in first.items() if k not in ("at", "time")} == {
        "event": "memory_call", "server": "mem", "tool": "view", "arguments": {}, "error": False, "result": "a\nb",
        "images": ["<jpeg 1 bytes sha256 x>"]}
    assert (second["error"], second["result"], second["arguments"]) == (True, '{"message": "no store"}', {"text": "t"})


# -- the store --------------------------------------------------------------------------------------------------------
def test_the_store_is_kept_beside_the_case_without_its_cache_replacing_an_earlier_one(tmp_path):
    """Contract (`codex_system.close`): when the case closes the memory's store is copied to `keep` with what the
    memory and the feed wrote (claims, the server's log, the feed's log) and without its `cache/`, replacing whatever
    an earlier run of the case left there, while the store in the state folder is the one the memory worked in.
    Regression caught: a store lost with the case's temp folder (nothing to look at after the run), a cache copied
    (megabytes of rebuildable index), or a rerun's files mixed with the last one's."""
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "stale.txt").write_text("from an earlier run", encoding="utf-8")
    with Run(tmp_path, {0: PLAN}, memory=memory(tmp_path / "spec"), keep=kept, asks_during=True, **CREDIT) as run:
        scenario(run)
        assert (run.state / "mem" / "cache" / "index.bin").is_file() and not (kept / "claims.jsonl").exists()
    assert sorted(p.name for p in kept.iterdir()) == ["claims.jsonl", "sense.jsonl", "server.jsonl"]
    claims = [json.loads(line) for line in (kept / "claims.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["text"] for c in claims] == [f"frame {k}" for k in range(6)]
    assert (kept / "sense.jsonl").read_text(encoding="utf-8") == (run.state / "mem" / "sense.jsonl").read_text(
        encoding="utf-8")
