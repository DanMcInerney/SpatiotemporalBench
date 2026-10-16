"""The sandbox's end of the system protocol (`stbench/shim.py`), run as the container runs it: a fresh interpreter,
`python -I -B shim.py FILE CLASS`, JSON lines on stdin and stdout. No Docker: the file is plain stdlib Python."""

from __future__ import annotations

import inspect
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from stbench import shim, stream_protocol
from stbench.stream_protocol import start_message

SHIM = Path(__file__).resolve().parents[1] / "src" / "stbench" / "shim.py"

ECHO = """
import os
from system_shim import System

class Echo(System):
    fed = 0

    def observe(self, steps):
        self.fed += len(steps)
        print("noise")
        os.write(1, b"raw\\n")

    def answer(self, question):
        if question.get("boom"):
            raise ValueError("bad")
        return {"type": "count", "value": self.fed}
"""

LINGER = """
import threading, time
from system_shim import System

class Echo(System):
    def observe(self, steps):
        threading.Thread(target=time.sleep, args=(60,)).start()  # non-daemon: keeps an ordinary interpreter alive

    def close(self):
        (self.state_dir / "closed.txt").write_text("closed", encoding="utf-8")
"""

# A system that implements every op and writes down each call it gets, with its arguments, in its state directory.
RECORDER = """
import json
from pathlib import Path

class Echo:
    def __init__(self, state_dir, workdir, **opts):
        self.log, self.calls = Path(state_dir) / "calls.json", [["init", opts]]

    def note(self, *call):
        self.calls.append(list(call))
        self.log.write_text(json.dumps(self.calls))

    def start(self, session, robot, channel):
        self.note("start", session, robot, channel)

    def task(self, index, clock, gap_s, brief):
        self.note("task", index, clock, gap_s, brief)

    def observe(self, steps):
        self.note("observe", steps)

    def ask(self, when, now, questions):
        self.note("ask", when, now, questions)
        return {q["qid"]: {"type": "count", "value": len(self.calls)} for q in questions}

    def finish(self):
        self.note("finish")
        return {"tokens_in": 12}
"""

# A system with no `start`, `task`, `ask` or `finish`: the shim's defaults serve it.
PLAIN = """
class Echo:
    def __init__(self, state_dir, workdir, **opts):
        pass

    def observe(self, steps):
        pass

    def answer(self, question):
        return {"type": "text", "value": question["qid"]}
"""


def run_shim(folder: Path, source: str, requests: list[dict],
             timeout: float) -> tuple[subprocess.CompletedProcess, list]:
    """Run the shim on a system written from `source` (class `Echo`), feed it `requests` and close its stdin. Returns
    the finished process and its stdout parsed one JSON object per line; a shim that does not exit fails the test."""
    (folder / "state").mkdir()
    (folder / "work").mkdir()
    child = folder / "child.py"
    child.write_text(textwrap.dedent(source), encoding="utf-8")
    stdin = b"".join(json.dumps(r).encode() + b"\n" for r in requests)
    try:
        proc = subprocess.run([sys.executable, "-I", "-B", str(SHIM), str(child), "Echo"], input=stdin,
                              capture_output=True, timeout=timeout, cwd=folder)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the shim did not exit within {timeout} s of the end of its input")
    return proc, [json.loads(line) for line in proc.stdout.decode().splitlines()]


def start(folder: Path, **opts) -> dict:
    """The harness's `start` message for a system in `folder`'s `state` and `work`, on `rgb_pose`."""
    return start_message("0123456789abcdef", {"name": "robot vacuum"}, "rgb_pose", str(folder / "work"),
                         str(folder / "state"), opts)


def ask(*qids: str, when: str = "end") -> dict:
    return {"op": "ask", "when": when, "now": {"k": 1, "clock": "2026-03-02T08:00:01"},
            "questions": [{"qid": qid, "text": f"{qid}?", "answer_type": "count"} for qid in qids]}


def test_the_shim_answers_each_request_with_one_line_and_keeps_stdout_for_replies(tmp_path):
    """Contract (V0 §9, the worker protocol kept in the sandbox): exactly one JSON line per request, a failure becomes
    an `error` reply and the shim keeps serving, and stdout carries nothing but replies, because a stray `print` or
    a raw write to fd 1 is sent to stderr. Regression caught: a stray print corrupting a reply, and a failing system
    killing the shim instead of being reported. (A subprocess's inherited fd 1 takes the same fd-level redirect; it is
    not asserted here because handle inheritance on a Windows host differs from the container's Linux.)"""
    boom = {**ask("q"), "questions": [{"qid": "q", "text": "?", "answer_type": "count", "boom": True}]}
    requests = [{"op": "observe", "steps": [{"k": 1}]}, start(tmp_path), {"op": "observe", "steps": [{"k": 1}]},
                ask("q"), {"op": "bogus"}, boom]
    proc, replies = run_shim(tmp_path, ECHO, requests, timeout=30)
    assert replies == [{"error": "the system is not started"}, {"ok": True}, {"ok": True},
                       {"answers": [{"qid": "q", "answer": {"type": "count", "value": 1}}]},
                       {"error": "unknown op 'bogus'"}, {"error": "ValueError: bad"}]
    assert proc.stdout.count(b"\n") == len(requests)  # nothing else was written to stdout
    assert {"noise", "raw"} <= set(proc.stderr.decode().split())


def test_every_op_reaches_the_system_with_its_arguments(tmp_path):
    """Contract (plan §2.4): `start` builds the class with the message's folders and options and then calls
    its `start(session, robot, channel)`; `task`, `observe`, `ask` and `finish` call the system's methods with the
    message's fields; the replies are `ok`, `ok`, `ok`, an `answers` list in the order of the dict the system returns,
    and `ok` with the system's usage. Hand-written calls and replies. Regression caught: a shim that drops an op or an
    argument (the session never reaching an external system, an ask never calling `ask`)."""
    robot = {"name": "robot vacuum"}
    requests = [start(tmp_path, depth=2),
                {"op": "task", "index": 0, "clock": "2026-03-02T08:00:00", "gap_s": 0.0, "brief": "Clean it."},
                {"op": "observe", "steps": [{"k": 0}, {"k": 1}]}, ask("q1", "q2", when="during"), {"op": "finish"}]
    proc, replies = run_shim(tmp_path, RECORDER, requests, timeout=30)
    assert replies == [{"ok": True}, {"ok": True}, {"ok": True},
                       {"answers": [{"qid": "q1", "answer": {"type": "count", "value": 5}},
                                    {"qid": "q2", "answer": {"type": "count", "value": 5}}]},
                       {"ok": True, "usage": {"tokens_in": 12}}], proc.stderr.decode()
    questions = [{"qid": q, "text": f"{q}?", "answer_type": "count"} for q in ("q1", "q2")]
    assert json.loads((tmp_path / "state" / "calls.json").read_text(encoding="utf-8")) == [
        ["init", {"depth": 2}], ["start", "0123456789abcdef", robot, "rgb_pose"],
        ["task", 0, "2026-03-02T08:00:00", 0.0, "Clean it."], ["observe", [{"k": 0}, {"k": 1}]],
        ["ask", "during", {"k": 1, "clock": "2026-03-02T08:00:01"}, questions], ["finish"]]


def test_a_class_without_the_new_methods_is_served_by_the_defaults(tmp_path):
    """Contract (plan §2.4): a class with only `observe` and `answer` (no `start`, `task`, `ask` or `finish`, and no
    base class) is served: `task` is acknowledged, `ask` calls `answer` for each question, in order, and `finish`
    reports an empty usage. An old op, `answer`, is refused as unknown. Regression caught: a system forced to implement
    every op, or a shim that still speaks the old wire."""
    requests = [start(tmp_path), {"op": "task", "index": 0, "clock": "c", "gap_s": 0.0, "brief": "b"},
                {"op": "observe", "steps": []}, ask("b", "a"), {"op": "finish"},
                {"op": "answer", "question": {"qid": "q"}}]
    _, replies = run_shim(tmp_path, PLAIN, requests, timeout=30)
    assert replies == [{"ok": True}, {"ok": True}, {"ok": True},
                       {"answers": [{"qid": "b", "answer": {"type": "text", "value": "b"}},
                                    {"qid": "a", "answer": {"type": "text", "value": "a"}}]},
                       {"ok": True, "usage": {}}, {"error": "unknown op 'answer'"}]


def test_the_disks_are_synced_at_finish_and_at_the_end_only(tmp_path, monkeypatch):
    """Contract (plan §2.4, map HA-H29): the shim syncs the disks before it replies to `finish` and again when its input
    ends, and never after an `observe` (the harness no longer copies the state at a task's end, so a sync per observe
    would only slow the stream). The shim is served in this process on in-memory streams with `_sync` replaced by a
    recorder of how many replies had been written when it ran. Regression caught: the sync left on every observe, or
    none at finish (a state directory whose size the harness reads before the writes have landed)."""
    state, work = tmp_path / "state", tmp_path / "work"
    state.mkdir()
    work.mkdir()
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(PLAIN), encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))  # `load` puts the system's folder first
    monkeypatch.setitem(sys.modules, "system_under_test", None)
    requests = [start(tmp_path), {"op": "observe", "steps": [{"k": 0}]}, {"op": "observe", "steps": [{"k": 1}]},
                ask("q"), {"op": "finish"}]
    out, synced = io.BytesIO(), []
    monkeypatch.setattr(shim, "_sync", lambda: synced.append(out.getvalue().count(b"\n")))
    shim.serve(str(child), "Echo", io.BytesIO(b"".join(json.dumps(r).encode() + b"\n" for r in requests)), out)
    assert synced == [4, 5]  # before the fifth reply (finish's), then once the input had ended


def test_at_end_of_input_the_shim_closes_the_system_and_exits_with_threads_left(tmp_path):
    """Contract (HANDOFF §3.2): when its input ends the shim closes the system, syncs and exits at once, even if the
    system left a non-daemon thread running, so the container ends with the session. Regression caught: PID 1 kept
    alive by a thread, which would cost the harness its whole close grace and a forced removal per instance."""
    proc, replies = run_shim(tmp_path, LINGER, [start(tmp_path), {"op": "observe", "steps": []}], timeout=10)
    assert replies == [{"ok": True}, {"ok": True}]
    assert proc.returncode == 0, proc.stderr.decode()
    assert (tmp_path / "state" / "closed.txt").read_text(encoding="utf-8") == "closed"


def public_api(cls: type) -> dict[str, list[tuple[str, object]]]:
    """A class's own public methods and properties, each with its (parameter name, default) list."""
    api = {}
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(member):
            api[name] = [(p.name, p.default) for p in inspect.signature(member).parameters.values()]
        elif isinstance(member, property):
            api[name] = []
    return api


def test_the_shim_base_mirrors_the_system_api():
    """Contract (plan §2.4): `shim.System`, the standard-library copy a system in the sandbox subclasses, has the same
    public methods and properties as `stream_protocol.System`, each with the same parameter names and defaults (the
    class flags `channels`, `privileged` and `structured` are the harness's, read from the class there). Regression
    caught: an external system that lacks `task` or `finish`, or a base whose `ask` takes other arguments."""
    assert set(public_api(stream_protocol.System)) >= {"start", "task", "observe", "ask", "answer", "finish", "close"}
    assert public_api(shim.System) == public_api(stream_protocol.System)


def test_the_shim_serves_exactly_the_protocols_ops():
    """Contract (plan §2.4): the shim's list of ops, a standard-library copy, is `stream_protocol.OPS`. Regression
    caught: an op added to the protocol that the shim answers as unknown (or the reverse), found only in a container."""
    assert shim.OPS == stream_protocol.OPS


def test_the_shim_base_defaults_behave_like_the_system_base(tmp_path):
    """Contract (plan §2.4): the shim base's defaults are the in-process base's: `ask` maps `answer` over the
    questions, `finish` returns `{}`, `task` does nothing and `start` keeps its three arguments. Regression caught: the
    two copies drifting, so a system behaves differently in the sandbox than in process."""
    class Reverser(shim.System):
        def answer(self, question):
            return {"value": question["qid"][::-1]}

    system = Reverser(tmp_path / "state", tmp_path / "work")
    questions = [{"qid": "ab"}, {"qid": "cd"}]
    assert system.ask("end", {"k": 0, "clock": "c"}, questions) == {"ab": {"value": "ba"}, "cd": {"value": "dc"}}
    assert system.finish() == {} and system.task(0, "08:00", 0.0, "brief") is None
    system.start("0123456789abcdef", {"name": "vacuum"}, "detections")
    assert (system.session, system.robot, system.channel) == ("0123456789abcdef", {"name": "vacuum"}, "detections")
