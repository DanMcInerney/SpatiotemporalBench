"""The sandbox's end of `system.CommandSystem` (VISION rule 6): serve a system class over JSON lines.

Copied into the container as `system_shim.py` and run as `python -I -B system_shim.py FILE CLASS`. Standard library
only: the sandbox has no `stbench`, and must not. FILE's folder is put first on `sys.path`. A system may subclass
`system_shim.System` (the same constructor and methods as `stbench.stream_protocol.System`, a test pins the names) or be
any class with `__init__(state_dir, workdir, **opts)`, `observe(steps)`, `answer(question)` and, optionally, `start`,
`task`, `ask`, `finish` and `close()`.

`serve` speaks the five ops of the stream protocol (`stream_protocol.py` has the table and says what each means). One
UTF-8 JSON object per line, one reply per request:

    {"op": "start", "session", "robot", "channel", "sends", "workdir", "state_dir", "opts"}
                                                       -> {"ok": true}   sets the case's environment (below), builds
                                                                         CLASS(Path, Path, **opts), then its
                                                                         start(session, robot, channel) if it has one
    {"op": "task", "index", "clock", "gap_s", "brief"} -> {"ok": true}   its task(...) if it has one
    {"op": "observe", "steps": [...]}                  -> {"ok": true}
    {"op": "ask", "when", "now", "questions": [...]}   -> {"answers": [{"qid", "answer"}, ...]}   its ask(when, now,
                                                                         questions) if it has one, else answer(question)
                                                                         for each question
    {"op": "finish"}                                   -> {"ok": true, "usage": {...}}   its finish() if it has one,
                                                                         else {}; the disks are synced first
    anything that fails                                -> {"error": "<Type>: <message>"}; the shim keeps serving

At `start`, before FILE is imported, the shim gives the case a home of its own (`case_environment`): the variables
where tools keep their own memory by default (`CASE_ENV`) point at empty folders under `state_dir/home`, and
`STBENCH_CASE_NONCE` is the session token, so whatever a tool the system runs remembers starts empty in every case and
counts in the state size. `main` sets them in its own process, which the system and every process it starts inherit.

At the end of input it closes the system, syncs the disks and exits at once, whatever threads the system left running.
stdout carries only replies: the shim keeps a private copy of fd 1 and points fd 1 at stderr, so a stray print (or a
subprocess's) goes to stderr; fd 0 is pointed at /dev/null, so a stray read sees EOF.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from collections.abc import MutableMapping
from pathlib import Path
from typing import BinaryIO

OPS = ("start", "task", "observe", "ask", "finish")  # the ops of `stbench.stream_protocol.OPS`, a test pins the copy
CASE_HOME = "home"  # the case's home, made inside the state directory at `start`
CASE_ENV = (  # (variable, its folder under the case's home): where tools keep their own memory by default
    ("HOME", ""), ("USERPROFILE", ""), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local"),
    ("XDG_CONFIG_HOME", ".config"), ("XDG_DATA_HOME", ".local/share"), ("XDG_STATE_HOME", ".local/state"),
    ("XDG_CACHE_HOME", ".cache"), ("CLAUDE_CONFIG_DIR", ".claude"), ("CODEX_HOME", ".codex"))
NONCE = "STBENCH_CASE_NONCE"  # the session token, for a system that namespaces a remote memory


class System:
    """Optional base: the same constructor and methods as `stbench.stream_protocol.System` (read its docstring for the
    contract), minus the harness-side class flags `channels`, `privileged` and `structured`."""

    session: str | None = None  # set by `start`
    robot: dict | None = None
    channel: str | None = None

    def __init__(self, state_dir: Path, workdir: Path, **opts):
        self.state_dir, self.workdir, self.opts = Path(state_dir), Path(workdir), opts

    @property
    def public(self) -> Path:
        return self.workdir / "public"

    def start(self, session: str, robot: dict, channel: str) -> None:
        """Called once, before anything else: the run's session token, the robot's `robot.json` and the channel."""
        self.session, self.robot, self.channel = session, robot, channel

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        """A task begins. Default: nothing."""

    def observe(self, steps: list[dict]) -> None:
        raise NotImplementedError

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        """Answers by qid. Default: `answer` for each question."""
        return {q["qid"]: self.answer(q) for q in questions}

    def answer(self, question: dict) -> dict:
        raise NotImplementedError

    def finish(self) -> dict:
        """The run is over: the usage the system reports. Default: none."""
        return {}

    def close(self) -> None:
        """Release what the instance holds; called once, at the end of input."""


def load(file: str, name: str) -> type:
    """Import FILE as module `system_under_test` (its folder first on `sys.path`) and return its attribute NAME."""
    path = Path(file).resolve()
    if sys.path[:1] != [str(path.parent)]:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("system_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def _sync() -> None:
    """Flush the page cache so the host sees every write (an mmap write without msync included)."""
    if hasattr(os, "sync"):
        os.sync()


def case_environment(state_dir: Path, session: str) -> dict[str, str]:
    """The case's variables: each of `CASE_ENV` set to its folder under `state_dir/home` (made here, empty, parents
    included), and `NONCE` set to the session token. The same table on every platform: an unused variable is
    harmless."""
    home = Path(state_dir) / CASE_HOME
    env = {}
    for variable, folder in CASE_ENV:
        path = home / folder
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    env[NONCE] = session
    return env


def serve(file: str, name: str, stdin: BinaryIO, stdout: BinaryIO,
          environ: MutableMapping[str, str] | None = None) -> None:
    """Answer one JSON line per request until stdin ends; then close the system and sync the disks. At `start`, before
    FILE is imported, `environ` (when given) is updated with `case_environment(state_dir, session)` from the message;
    `main` gives `os.environ`, so the system and every process it starts see the case's home. An in-process caller
    that passes nothing changes nothing, not even a folder."""
    system = None

    def handle(msg: dict) -> dict:
        nonlocal system
        op = msg["op"]
        if op not in OPS:
            return {"error": f"unknown op {op!r}"}
        if op == "start":  # the class is loaded here, so an import error is an error reply to `start`
            if system is not None:
                return {"error": "the system is already started"}
            if environ is not None:  # before the import: a module that reads `Path.home()` at import sees the case's
                environ.update(case_environment(Path(msg["state_dir"]), msg["session"]))
            system = load(file, name)(Path(msg["state_dir"]), Path(msg["workdir"]), **msg["opts"])
            if (start := getattr(system, "start", None)) is not None:
                start(msg["session"], msg["robot"], msg["channel"])
            return {"ok": True}
        if system is None:
            return {"error": "the system is not started"}
        if op == "task":
            if (task := getattr(system, "task", None)) is not None:
                task(msg["index"], msg["clock"], msg["gap_s"], msg["brief"])
            return {"ok": True}
        if op == "observe":
            system.observe(msg["steps"])
            return {"ok": True}
        if op == "ask":
            when, now, questions = msg["when"], msg["now"], msg["questions"]
            if (ask := getattr(system, "ask", None)) is not None:
                answers = ask(when, now, questions)
            else:
                answers = {q["qid"]: system.answer(q) for q in questions}
            return {"answers": [{"qid": qid, "answer": answer} for qid, answer in answers.items()]}
        usage = finish() if (finish := getattr(system, "finish", None)) is not None else {}
        _sync()  # before the reply: the harness reads the state directory's size as soon as it hears it
        return {"ok": True, "usage": usage}

    for line in iter(stdin.readline, b""):
        try:
            data = json.dumps(handle(json.loads(line)))  # serialised first: a bad answer becomes an error reply
        except Exception as e:  # noqa: BLE001  (the harness must hear about every failure)
            data = json.dumps({"error": f"{type(e).__name__}: {e}"})
        stdout.write(data.encode("ascii") + b"\n")
        stdout.flush()
    if (close := getattr(system, "close", None)) is not None:
        try:
            close()
        except Exception:  # noqa: BLE001  (nobody awaits a reply now; the harness reads the stderr tail)
            traceback.print_exc()
    _sync()


def main(argv: list[str] | None = None) -> None:
    file, name = (argv or sys.argv)[1:3]
    stdin, stdout = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")  # private, non-inheritable copies
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdin, sys.stdout = open(os.devnull), sys.stderr  # noqa: SIM115  (lives as long as the process)
    sys.modules.setdefault("system_shim", sys.modules[__name__])  # so a system can `from system_shim import System`
    code = 0
    try:
        serve(file, name, stdin, stdout, os.environ)
    except BaseException as e:  # noqa: BLE001  (PID 1 must end whatever happened; the stderr tail says why)
        traceback.print_exc()
        code = e.code if isinstance(e, SystemExit) and isinstance(e.code, int) else 1
    sys.stderr.flush()
    os._exit(code)  # PID 1 ends now: threads or processes the system left behind die with the container


if __name__ == "__main__":
    main()
