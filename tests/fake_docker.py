"""A `CommandSystem` whose container is a Python child and whose Docker is a script that records what the harness asks
of it: no Docker, so the tests that use it (test_command_system_local.py) run everywhere.
Each test file wraps `local` in its own fixture (`yield from fake_docker.local(tmp_path, monkeypatch)`).

A container exists for the scripted Docker while its child runs (`Docker.alive`), so `container ls` lists the children
that have not exited.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from stbench import sandbox
from stbench.system import CommandSystem

SESSION = "0123456789abcdef"  # the `start` message's session and robot, as `Local.make` sends them
ROBOT = {"name": "robot vacuum"}

# The child that stands in for a container. It serves `start`, then does what its mode says: `serve` answers every
# message with a good reply (an `ask` with a question that has `sleep` waits that many seconds first, and one with
# `die` writes a word to stderr and exits with code 7 instead; a `task` whose brief is `error` gets an {"error"} reply,
# and one whose brief is `weird` a reply no op expects); `linger` serves, and when its input ends does not exit (a
# system whose `close()` never returns: it is gone only when it is removed); `garbage` replies with a line that is not
# JSON; `die` writes a word to stderr and exits with code 7; `mortal` does the same once it is sent one more line;
# `partial` reads ten bytes of the next request, replies, and never reads the rest. Its last argument is its
# container's name.
FAKE = r"""
import json, os, sys, time
mode = sys.argv[1]
inp, out = sys.stdin.buffer, sys.stdout.buffer


def reply(obj):
    out.write(json.dumps(obj).encode() + b"\n")
    out.flush()


def serve(msg):
    op = msg["op"]
    if op == "ask":
        for q in msg["questions"]:
            if q.get("die"):
                sys.stderr.write("goodbye\n")
                sys.stderr.flush()
                sys.exit(7)
            time.sleep(q.get("sleep", 0))
        return {"answers": [{"qid": q["qid"], "answer": {"type": "unknown"}} for q in msg["questions"]]}
    if op == "task" and msg["brief"] == "error":
        return {"error": "ValueError: boom"}
    if op == "task" and msg["brief"] == "weird":
        return {"ok": False}
    return {"ok": True, "usage": {"tokens_in": 7}} if op == "finish" else {"ok": True}


inp.readline()
reply({"ok": True})
if mode in ("serve", "linger"):
    for line in inp:
        reply(serve(json.loads(line)))
    if mode == "linger":
        time.sleep(60)
elif mode == "garbage":
    inp.readline()
    out.write(b"not json\n")
    out.flush()
    time.sleep(60)
elif mode in ("die", "mortal"):
    if mode == "mortal":
        inp.readline()
    sys.stderr.write("goodbye\n")
    sys.stderr.flush()
    sys.exit(7)
elif mode == "partial":
    os.read(0, 10)
    reply({"ok": True})
    time.sleep(60)
"""


class Docker:
    """The scripted Docker CLI: `calls` holds (arguments, whether the system's own CLI process was still running) for
    every command; `fail` maps a command's first word to the (returncode, stderr) it answers with instead of success;
    `flaky` maps one to the number of its next calls that fail (stderr "daemon busy") before it succeeds again; `on`
    maps one to a function run when it is called; `inspect` is how many `container inspect` calls find the container
    before it is gone."""

    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], bool]] = []
        self.fail: dict[str, tuple[int, str]] = {}
        self.flaky: dict[str, int] = {}
        self.on: dict[str, Callable[[], object]] = {}
        self.inspect = 0
        self.mode = "serve"
        self.system: CommandSystem | None = None
        self.procs: dict[str, subprocess.Popen] = {}

    def alive(self) -> frozenset[str]:
        """The containers whose child has not exited, by name."""
        return frozenset(name for name, proc in list(self.procs.items()) if proc.poll() is None)

    def run(self, exe, args, timeout):
        proc = self.system._proc if self.system is not None else None
        self.calls.append((tuple(args), proc is not None and proc.poll() is None))
        if args[0] in self.on:
            self.on[args[0]]()
        out = ""
        if args[:2] == ("container", "inspect"):
            seen = sum(1 for a, _ in self.calls if a[:2] == ("container", "inspect"))
            code, err = (0, "") if seen <= self.inspect else (1, "No such container")
        elif args[:2] == ("container", "ls"):
            code, err, out = 0, "", "\n".join(sorted(self.alive())) + "\n"
        elif self.flaky.get(args[0], 0) > 0:
            self.flaky[args[0]] -= 1
            code, err = 1, "daemon busy"
        else:
            code, err = self.fail.get(args[0], (0, ""))
        return subprocess.CompletedProcess([exe, *args], code, out, err)

    def commands(self, word: str) -> list[tuple[str, ...]]:
        return [args for args, _ in self.calls if args[0] == word]


def local(tmp_path: Path, monkeypatch) -> Iterator[object]:
    """`local.make(mode)` is a `CommandSystem` whose container is the FAKE child; `local.docker` is its Docker."""
    docker = Docker()
    made: list[CommandSystem] = []
    monkeypatch.setattr(sandbox, "require", lambda: "docker")
    monkeypatch.setattr(sandbox, "_run", docker.run)

    def container_args(exe, name, state, work, program, **kw):
        return [sys.executable, "-c", FAKE, docker.mode, name]

    monkeypatch.setattr(sandbox, "container_args", container_args)
    real_popen = subprocess.Popen

    def popen(args, *a, **kw):  # a container's child is found by the name it was given
        proc = real_popen(args, *a, **kw)
        if isinstance(args, list) and FAKE in args:
            docker.procs[args[-1]] = proc
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    program = sandbox.Program(tmp_path / "program", "x.py", "X")

    class Local:
        def make(self, mode="serve", *, start=True, **kw) -> CommandSystem:
            """A `CommandSystem` on the FAKE child in `mode`, sent its `start` message unless `start` is false."""
            docker.mode = mode
            cs = CommandSystem(tmp_path / "state", tmp_path / "work", program=program, **kw)
            docker.system = cs
            made.append(cs)
            if start:
                cs.start(SESSION, ROBOT, "rgb_pose")
            return cs

    env = Local()
    env.docker, env.program = docker, program
    yield env
    for cs in made:
        if cs._proc is not None:
            cs._proc.kill()
            cs._proc.wait()
    for proc in docker.procs.values():  # any container's child left running
        if proc.poll() is None:
            proc.kill()
            proc.wait()
