"""`examples/claude_code_agent`: Claude Code as the robot's brain, run here on `claude_fake.py` (no model, no network).

Each test builds the agent in process with `CLAUDE_AGENT_COMMAND` the fake, `CLAUDE_CONFIG_DIR` a temp folder standing
in for the case's home and `FAKE_CLAUDE_LOG` a temp file, and reads what the fake was sent from that file. Every number
a test expects is written out by hand from the fake's documented usage (the n-th invocation of a session reads 600 n
cached tokens, writes 200, and so on).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from run_kit import scored_case
from stbench import cli
from stbench.answers import TYPES, Malformed, parse_answer
from stbench.stream_protocol import usage_problems
from stream_kit import QUESTIONS, hand20

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "claude_code_agent" / "agent.py"
FAKE = Path(__file__).with_name("claude_fake.py")
ROBOT = {"name": "robot vacuum"}
CAMERA = {"width": 64, "height": 48, "hfov_deg": 90.0}
STATUS = {"phase": "cleaning", "task": 0, "battery_pct": 97, "brush": True}
FRAME_3, FRAME_4 = b"\xff\xd8 the pixels of frame three", b"\xff\xd8 the pixels of frame four"
TOOLS = "Read,Write,Edit,Glob,Grep"


def load():
    spec = importlib.util.spec_from_file_location("claude_code_agent_under_test", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Rig:
    """One agent over a state folder under `tmp_path / name`, with the fake as its `claude`; its records read back."""

    def __init__(self, tmp_path: Path, monkeypatch, name: str = "run", plan: dict | None = None,
                 channel: str = "rgb_pose"):
        self.root = tmp_path / name
        self.state, self.work = self.root / "a" / "state", self.root / "work"
        for folder in (self.state, self.work / "public", self.work / "inbox" / "rgb"):
            folder.mkdir(parents=True)
        (self.work / "public" / "camera.json").write_text(json.dumps(CAMERA), encoding="utf-8")
        # the agent's log is in a folder that does not exist yet, as `stbench run --out DIR` leaves it until the end
        self.fake_log, self.agent_log = self.root / "fake.jsonl", self.root / "out" / "agent.jsonl"
        for var, value in (("CLAUDE_AGENT_COMMAND", FAKE), ("CLAUDE_CONFIG_DIR", self.root / "home" / ".claude"),
                           ("FAKE_CLAUDE_LOG", self.fake_log), ("CLAUDE_AGENT_LOG", self.agent_log)):
            monkeypatch.setenv(var, str(value))
        if plan is not None:
            (self.root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            monkeypatch.setenv("FAKE_CLAUDE_PLAN", str(self.root / "plan.json"))
        self.agent = load().ClaudeCodeAgent(self.state, self.work)
        self.agent.start("0123456789abcdef", ROBOT, channel)

    def step(self, k: int, jpeg: bytes | None = None, **extra) -> dict:
        step = {"k": k, "clock": f"2026-03-02T08:00:{k:02d}", "pose": {"pos": [5.6, 5.05, 0.0], "rot": [1, 0, 0, 0],
                                                                      "yaw_deg": -90.0},
                "camera": {"pos": [5.6, 5.0, 1.0], "rot": [1, 0, 0, 0], "pitch_deg": 5.0}, "status": STATUS, **extra}
        if jpeg is not None:
            (self.work / "inbox" / "rgb" / f"{k:06d}.jpg").write_bytes(jpeg)
            step["rgb"] = f"inbox/rgb/{k:06d}.jpg"
            step["depth"] = f"inbox/depth/{k:06d}.png"
        return step

    def four_messages(self) -> None:
        """A task, two frames and an end ask: four invocations."""
        self.agent.task(0, "2026-03-02T08:00:00", 0, "Brief: tidy the place.")
        self.agent.observe([self.step(3, FRAME_3)])
        self.agent.observe([self.step(4, FRAME_4)])
        self.agent.ask("end", {"k": 4, "clock": "2026-03-02T08:00:04"}, [question(1)])

    def fake_calls(self) -> list[dict]:
        return [json.loads(line) for line in self.fake_log.read_text(encoding="utf-8").splitlines()]


def question(n: int, answer_type: str = "position") -> dict:
    return {"qid": f"{n:016x}", "text": f"Where was thing {n} last seen?", "answer_type": answer_type}


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def test_one_session_a_case_resumed_by_every_message(tmp_path, monkeypatch):
    """Contract (docs/agent.md; the example's docstring): a task, two frames and an end ask are four invocations of
    `claude -p`; the first starts a session (no `--resume`) and the other three resume the one it gave, each with the
    same model, exactly the tools Read, Write, Edit, Glob and Grep, in the state folder and with the two quiet
    variables. Regression caught: a session a message (the agent would have no memory), or a shell, the web or a
    subagent within the model's reach."""
    rig = Rig(tmp_path, monkeypatch)
    rig.four_messages()
    calls = rig.fake_calls()
    sid = calls[0]["session"]
    base = ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--model",
            "claude-sonnet-5-5", "--tools", TOOLS, "--permission-mode", "acceptEdits", "--max-turns", "40"]
    assert [c["args"] for c in calls] == [base] + [[*base, "--resume", sid]] * 3
    assert [c["resumed"] for c in calls] == [False, True, True, True]
    assert [c["n"] for c in calls] == [1, 2, 3, 4]
    for c in calls:
        assert Path(c["cwd"]).resolve() == rig.state.resolve()
        assert c["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1" and c["env"]["DISABLE_TELEMETRY"] == "1"


def test_the_first_message_is_the_introduction_then_the_brief(tmp_path, monkeypatch):
    """Contract (the example's docstring): the first message holds the introduction (the robot's onboard assistant, the
    frame of reference, the camera from `public/camera.json`, the working folder as its memory, every answer shape and
    the unknown rule) and then the first task line and brief; a later task message is the task line, the gap and its
    brief alone; neither names the case's folders. Regression caught: a brief sent without the introduction, the
    camera or the answer shapes missing, a second introduction, or a dataset path leaking into the prompt."""
    rig = Rig(tmp_path, monkeypatch)
    rig.agent.task(0, "2026-03-02T08:00:00", 0, "Brief: you work on a floor.")
    rig.agent.task(1, "2026-03-02T09:00:00", 1800.0, "Brief: now the kitchen.")
    first, second = (c["text"] for c in rig.fake_calls())
    intro, _, rest = first.partition("Task 0 begins at 2026-03-02T08:00:00.\n")
    assert rest.startswith("Brief: you work on a floor.\n")
    assert intro.startswith("You are the onboard assistant of a robot.")
    assert "x points east, y north and z up" in intro
    assert "64 x 48 pixels with a horizontal field of view of 90 degrees" in intro
    assert "working folder" in intro and "Answer only if you are more likely right than wrong" in intro
    assert all(f'- {kind}: {{"type": "{kind}"' in intro for kind in (*TYPES, "unknown"))
    assert second.splitlines()[:3] == ["Task 1 begins at 2026-03-02T09:00:00.",
                                       "The robot went unobserved for 1800 s since the last task's final frame.",
                                       "Brief: now the kitchen."]
    assert "onboard assistant" not in second
    assert str(tmp_path) not in first + second


def test_frames_are_attached_in_order_with_their_headers(tmp_path, monkeypatch):
    """Contract (the example's docstring): an `observe` is one message with one header line a frame, in step order, with
    its k, clock, base position, heading, camera pitch and status, a line naming the image of each frame, and the
    frames' JPEG bytes attached in that order; the depth path is not sent. Regression caught: the model told the wrong
    frame for a picture, the camera's position shown as the base's, or a path of the inbox leaking into the message."""
    rig = Rig(tmp_path, monkeypatch)
    rig.agent.task(0, "2026-03-02T08:00:00", 0, "Brief.")
    rig.agent.observe([rig.step(3, FRAME_3), rig.step(4, FRAME_4)])
    call = rig.fake_calls()[1]
    assert call["images"] == [fingerprint(FRAME_3), fingerprint(FRAME_4)]
    assert call["text"].splitlines() == [
        "Frames k3 to k4 (2026-03-02T08:00:03 to 2026-03-02T08:00:04), with 2 camera images attached, in frame order:",
        "[k3 2026-03-02T08:00:03] base (5.60, 5.05, 0.00) heading -90 deg; camera pitch 5 deg; "
        "status: phase cleaning, task 0, battery_pct 97, brush true",
        "(image 1 is frame k3)",
        "[k4 2026-03-02T08:00:04] base (5.60, 5.05, 0.00) heading -90 deg; camera pitch 5 deg; "
        "status: phase cleaning, task 0, battery_pct 97, brush true",
        "(image 2 is frame k4)",
        "Make any notes you want in your working folder now, then reply with one short line."]
    assert "inbox" not in call["text"] and "depth" not in call["text"]


def test_a_header_shows_the_robots_own_velocity_when_the_step_has_one(tmp_path, monkeypatch):
    """Contract (final review S3, the example's docstring): a step that carries the robot's own `velocity` (the
    stream's, metres a second in the map frame) has it on its header, three decimals and never "-0.000", between the
    heading and the camera's pitch; a step without one is headed as before. Regression caught: a velocity the stream
    delivers and the example's model is never shown."""
    rig = Rig(tmp_path, monkeypatch)
    rig.agent.task(0, "2026-03-02T08:00:00", 0, "Brief.")
    rig.agent.observe([rig.step(3, FRAME_3, velocity=[-0.0, 0.25])])
    header = ("[k3 2026-03-02T08:00:03] base (5.60, 5.05, 0.00) heading -90 deg; velocity (0.000, 0.250) m/s; "
              "camera pitch 5 deg; status: phase cleaning, task 0, battery_pct 97, brush true")
    assert header in rig.fake_calls()[1]["text"].splitlines()


def test_detections_ride_as_one_compact_line_and_no_picture(tmp_path, monkeypatch):
    """Contract (the example's docstring): on the `detections` channel a step has no picture, so none is attached and no
    image line is written, and its detections follow its header as one compact JSON line. Regression caught: an
    attachment looked up for a step with no `rgb`, or detections dropped or pretty-printed over many lines."""
    rig = Rig(tmp_path, monkeypatch, channel="detections")
    detections = [{"d": "a1", "kind": "mug", "attributes": {"color": "red"}, "center_map": [2.0, 1.0, 0.9]}]
    rig.agent.task(0, "2026-03-02T08:00:00", 0, "Brief.")
    rig.agent.observe([rig.step(3, detections=detections)])
    call = rig.fake_calls()[1]
    assert call["images"] == []
    lines = call["text"].splitlines()
    assert lines[0] == "Frames k3 to k3 (2026-03-02T08:00:03 to 2026-03-02T08:00:03):"
    assert lines[2] == ('detections: [{"d":"a1","kind":"mug","attributes":{"color":"red"},'
                        '"center_map":[2.0,1.0,0.9]}]')
    assert "image" not in call["text"]


def test_asks_use_labels_and_map_back(tmp_path, monkeypatch):
    """Contract (the example's docstring): an ask shows its questions as Q1..Qn with text and answer type and never a
    qid; the last JSON object of the reply maps back to the qids, a label whose value is no object with a string `type`
    is missing, and one nudge message names exactly the labels still missing, after which what is missing is `unknown`.
    Regression caught: qids shown to the model, answers mapped to the wrong question, an endless retry, or an answer
    lost between the first reply and the nudge."""
    questions = [question(1, "count"), question(2, "count"), question(3, "yesno")]
    now = {"k": 19, "clock": "2026-03-02T08:00:19"}

    # one label unreadable in the reply (prose and a stray brace around the object): one nudge, for that label alone
    reply = ('I looked {as asked}. {"Q1": {"type": "count", "value": 2}, "Q2": {"value": 3}, '
             '"Q3": {"type": "yesno", "value": true}} Hope that helps.')
    nudged = '{"Q2": {"type": "count", "value": 3}}'
    rig = Rig(tmp_path, monkeypatch, "partial", plan={"0": {"reply": reply}, "1": {"reply": nudged}})
    answers = rig.agent.ask("end", now, questions)
    first, nudge = (c["text"] for c in rig.fake_calls())
    assert answers == {question(1)["qid"]: {"type": "count", "value": 2},
                       question(2)["qid"]: {"type": "count", "value": 3},
                       question(3)["qid"]: {"type": "yesno", "value": True}}
    assert "Q1: Where was thing 1 last seen?\nAnswer type: count." in first and "Answer type: yesno." in first
    assert "Questions after the job, at the robot's last frame (k19, 2026-03-02T08:00:19)" in first
    assert not re.search(r"[0-9a-f]{16}", first + nudge)
    assert set(re.findall(r"\bQ\d+\b", nudge)) == {"Q2"}

    # nothing readable twice: one nudge naming all three, then `unknown` for each, and no third message
    rig = Rig(tmp_path, monkeypatch, "prose", plan={"0": {"reply": "I cannot tell."}, "1": {"reply": "Still unsure."}})
    answers = rig.agent.ask("end", now, questions)
    calls = rig.fake_calls()
    assert answers == {q["qid"]: {"type": "unknown"} for q in questions}
    assert len(calls) == 2 and set(re.findall(r"\bQ\d+\b", calls[1]["text"])) == {"Q1", "Q2", "Q3"}
    assert not re.search(r"[0-9a-f]{16}", calls[1]["text"])


def test_finish_reports_the_sessions_tokens_per_model(tmp_path, monkeypatch):
    """Contract (docs/agent.md; the example's docstring): after the four invocations of a short case,
    `finish` reports each model's tokens from the last result's cumulative `modelUsage` (`tokens_in` counts the cache
    reads and writes, `tokens_cached_in` the reads, `tokens_cache_write_in` the writes), the largest context one
    request held, and no remote store; with no invocation it reports `models` null. By hand, sonnet: in 100 x 4 = 400,
    cached 600 (1 + 2 + 3 + 4) = 6,000, written 200 x 4 = 800, so `tokens_in` 7,200, out 50 x 4 = 200; haiku, used
    once: 10 in, 5 out; the peak is the fourth request's 100 + 2,400 + 200 = 2,700. Regression caught: running totals
    summed twice, cache writes dropped from the input, an unreported model, or Claude Code's own dollar figure
    reported."""
    rig = Rig(tmp_path, monkeypatch)
    assert rig.agent.finish() == {"models": None, "peak_context": None, "remote_store_bytes": None}
    rig.four_messages()
    assert rig.agent.finish() == {
        "models": {"claude-sonnet-5-5": {"tokens_in": 7200, "tokens_cached_in": 6000, "tokens_cache_write_in": 800,
                                         "tokens_out": 200},
                   "claude-haiku-4-5": {"tokens_in": 10, "tokens_cached_in": 0, "tokens_cache_write_in": 0,
                                        "tokens_out": 5}},
        "peak_context": 2700, "remote_store_bytes": None}

    # the context is the largest one seen, not the latest: a compaction shrinks it (900 tokens, then 100 + 100 + 200)
    rig = Rig(tmp_path, monkeypatch, "compacted", plan={"1": {"shrink": True}})
    rig.agent.task(0, "2026-03-02T08:00:00", 0, "Brief.")
    rig.agent.observe([rig.step(3, FRAME_3)])
    assert rig.agent.finish()["peak_context"] == 900


def test_a_failed_invocation_raises_with_its_reason(tmp_path, monkeypatch):
    """Contract (docs/agent.md, "When something goes wrong"): an invocation that fails (here the fake's invalid-key
    failure on the second one: a non-zero exit with an error result) raises `RuntimeError` holding the end of Claude
    Code's stderr and the result's subtype, so the harness ends the case with the reason. Regression caught: a failure
    answered silently, as an empty reply, so a case runs on with a model that never saw a frame."""
    rig = Rig(tmp_path, monkeypatch, plan={"1": {"fail": "auth"}})
    rig.agent.task(0, "2026-03-02T08:00:00", 0, "Brief.")
    with pytest.raises(RuntimeError) as failed:
        rig.agent.observe([rig.step(3, FRAME_3)])
    assert "Invalid API key · Please run /login" in str(failed.value)
    assert "error_during_execution" in str(failed.value) and "claude observe failed" in str(failed.value)


def test_it_refuses_a_claude_md_above_its_folder(tmp_path, monkeypatch):
    """Contract (the example's docstring; M5 calls): `start` raises, naming the file, when a `CLAUDE.md` (or
    `CLAUDE.local.md`) sits in a folder above the state folder, because Claude Code would add it to the context and the
    run would be another agent; before it nobody was invoked. Regression caught: a user's own instructions in the
    example's context."""
    rig = Rig(tmp_path, monkeypatch, "clean")  # a folder with none above it starts
    for name in ("CLAUDE.md", "CLAUDE.local.md"):
        marked = tmp_path / name.replace(".", "_") / "a" / "state"
        marked.mkdir(parents=True)
        (marked.parent.parent / name).write_text("Always say hello.", encoding="utf-8")
        with pytest.raises(RuntimeError, match=re.escape(str(marked.parent.parent.resolve() / name))):
            load().ClaudeCodeAgent(marked, rig.work).start("0123456789abcdef", ROBOT, "rgb_pose")
    assert not rig.fake_log.exists()


def test_it_refuses_to_start_without_claude(tmp_path, monkeypatch):
    """Contract (the example's docstring): `start` raises when the command it was told to run (a `.py` file) is not
    there, and when there is no `claude` on the path. Regression caught: a case that starts and then fails on its first
    message with an obscure OS error."""
    rig = Rig(tmp_path, monkeypatch)
    agent = load().ClaudeCodeAgent(rig.state, rig.work)
    monkeypatch.setenv("CLAUDE_AGENT_COMMAND", str(tmp_path / "nowhere" / "claude.py"))
    with pytest.raises(RuntimeError, match="not a file"):
        agent.start("0123456789abcdef", ROBOT, "rgb_pose")
    monkeypatch.setenv("CLAUDE_AGENT_COMMAND", "no-such-claude-command")
    with pytest.raises(RuntimeError, match="no `claude` command"):
        agent.start("0123456789abcdef", ROBOT, "rgb_pose")


def test_the_agent_log_has_one_line_an_invocation_and_no_prompt_or_secret(tmp_path, monkeypatch):
    """Contract (the example's docstring): `CLAUDE_AGENT_LOG` gets one JSON line an invocation with the op, seconds,
    whether it resumed, the session, the result's subtype, `is_error`, turns, the image count, Claude Code's cost,
    `usage` and `modelUsage`, and never a prompt's text or an environment value. Regression caught: a log that records
    the prompts (the questions, the brief) or the credential in the environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    rig = Rig(tmp_path, monkeypatch)
    rig.agent.task(0, "2026-03-02T08:00:00", 0, "Brief: a secret mission.")
    rig.agent.observe([rig.step(3, FRAME_3)])
    raw = rig.agent_log.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    sid = rig.fake_calls()[0]["session"]
    assert [(r["op"], r["resumed"], r["session"], r["images"], r["subtype"], r["is_error"], r["num_turns"],
             r["total_cost_usd"]) for r in rows] == [("task", False, sid, 0, "success", False, 1, 0.01),
                                                     ("observe", True, sid, 1, "success", False, 1, 0.02)]
    assert all(set(r) == {"op", "seconds", "resumed", "session", "subtype", "is_error", "num_turns", "images",
                          "total_cost_usd", "usage", "modelUsage"} and r["seconds"] >= 0 for r in rows)
    assert rows[1]["usage"]["cache_read_input_tokens"] == 1200
    assert rows[1]["modelUsage"]["claude-haiku-4-5"]["inputTokens"] == 10
    assert "secret mission" not in raw and "sk-ant" not in raw and "Frames k3" not in raw


def test_two_cases_through_stbench_run_and_the_second_starts_empty(tmp_path, monkeypatch):
    """Contract (docs/agent.md "Memory", M5's calls; the example's docstring): `stbench run FILE.py:ClaudeCodeAgent CASE
    --local` completes a case on the fake `claude` (exit 0, status complete) and the agent's `finish` is a valid
    stream-protocol report that the summary stamps self-reported; each case's `claude` runs with the case's own home, a
    `home/.claude` of its state folder (a different one a case, `HOME` and `USERPROFILE` beside it) and the case's
    session token as `STBENCH_CASE_NONCE`; and the second case's `claude` finds none of the auto memory the first one
    wrote. By hand, the case is 23 messages (a task, the 20 frames one an `observe`, the during ask after the thirteenth
    and the end ask), so the n-th reads 600 n cached tokens: sonnet in 23 x 100 + 600 x 276 + 23 x 200 = 172,500, cached
    165,600, written 4,600, out 1,150; haiku 10 in, 5 out; peak 100 + 13,800 + 200 = 14,100. The auto memory first
    exists for the third message (the first frame is the first to name frames). Regression caught: the example losing
    the case's environment (its `claude` reading the user's own `~/.claude`, so a case remembers the last), or failing
    the documented command."""
    monkeypatch.setenv("CLAUDE_AGENT_COMMAND", str(FAKE))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(tmp_path / "fake.jsonl"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)  # the shim sets the case's, or the fake has none
    summaries = []
    for seed in (1, 2):
        synth = hand20()
        synth.seed = seed  # another script, so another case
        case = scored_case(tmp_path / "data", synth, QUESTIONS)
        out = tmp_path / f"out{seed}"
        assert cli.main(["run", f"{EXAMPLE}:ClaudeCodeAgent", str(case.private), "--local", "--timeout", "60",
                         "--out", str(out)]) == 0
        summaries.append(json.loads((out / "summary.json").read_text(encoding="utf-8")))
    report = {"self_reported": True,
              "models": {"claude-sonnet-5-5": {"tokens_in": 172500, "tokens_cached_in": 165600,
                                               "tokens_cache_write_in": 4600, "tokens_out": 1150},
                         "claude-haiku-4-5": {"tokens_in": 10, "tokens_cached_in": 0, "tokens_cache_write_in": 0,
                                              "tokens_out": 5}},
              "peak_context": 14100, "remote_store_bytes": None}
    for summary in summaries:
        assert (summary["status"], summary["failures"], summary["sandbox"]) == ("complete", {}, "none")
        assert summary["usage"] == report and usage_problems(summary["usage"]) == []
    rows = [json.loads(line) for line in (tmp_path / "fake.jsonl").read_text(encoding="utf-8").splitlines()]
    first, second = rows[:23], rows[23:]
    assert len(second) == 23
    for case_rows, summary in ((first, summaries[0]), (second, summaries[1])):
        assert [r["resumed"] for r in case_rows] == [False] + [True] * 22
        assert len({r["session"] for r in case_rows}) == 1
        assert {r["env"]["STBENCH_CASE_NONCE"] for r in case_rows} == {summary["session"]}
        home = {Path(r["env"]["CLAUDE_CONFIG_DIR"]) for r in case_rows}
        assert len(home) == 1 and home.pop().parts[-3:] == ("state", "home", ".claude")
        assert {Path(r["env"]["HOME"]) for r in case_rows} == {Path(r["env"]["USERPROFILE"]) for r in case_rows}
        assert [r["memory_now"] for r in case_rows] == [False, False] + [True] * 21
    assert first[0]["env"]["CLAUDE_CONFIG_DIR"] != second[0]["env"]["CLAUDE_CONFIG_DIR"]
    assert second[0]["memory_at_start"] is False


@pytest.mark.parametrize("answer_type", TYPES)
def test_every_answer_shape_the_model_is_shown_is_accepted_by_the_strict_parser(answer_type):
    """Contract (the example's docstring: the template's shapes, copied): the example's `ANSWERS[answer_type]` exists
    for every type a question may name and the strict parser (`answers.parse_answer`) accepts it. Regression caught: a
    copy that fell behind the template and the parser, so the model is shown a shape that scores as malformed (-1)."""
    shape = load().ANSWERS[answer_type]
    assert shape["type"] == answer_type
    assert not isinstance(parse_answer(shape, answer_type), Malformed)


def test_the_example_imports_only_the_standard_library_and_the_shim():
    """Contract (docs/agent.md): an example runs unchanged in the Docker sandbox, which has the standard library and
    what the agent's folder vendors, so it imports nothing else but the shim's own `system_shim`. Regression caught: a
    third-party client library added to talk to Claude, which the sandbox lacks."""
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= set(sys.stdlib_module_names) | {"system_shim"}
