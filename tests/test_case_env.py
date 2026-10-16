"""The case's environment, set by the shim (`stbench/shim.py`, docs/agent.md "Memory"): each case starts with a home of
its own inside its state folder, so the memory a tool keeps there (Claude Code's, Codex's, a library's cache) starts
empty in every case and counts in the state size. The table is spelled out here by hand, not read from the code under
test; the agents are source texts written into temp folders (local_kit.py's pattern), standard library only."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path, PurePosixPath

import pytest

from sandbox_children import ECHO, write_child
from stbench import sandbox, shim
from stbench.local_system import LocalCommandSystem
from stbench.safecopy import tree_bytes
from stbench.stream_protocol import start_message
from stbench.system import CommandSystem, timeouts_of

SESSION, ROBOT = "0123456789abcdef", {"name": "robot vacuum"}
SESSIONS = ("0" * 16, "1" * 16)

# The documented variables and each one's folder under the case's home (docs/agent.md "Memory").
TABLE = {"HOME": "", "USERPROFILE": "", "APPDATA": "AppData/Roaming", "LOCALAPPDATA": "AppData/Local",
         "XDG_CONFIG_HOME": ".config", "XDG_DATA_HOME": ".local/share", "XDG_STATE_HOME": ".local/state",
         "XDG_CACHE_HOME": ".cache", "CLAUDE_CONFIG_DIR": ".claude", "CODEX_HOME": ".codex"}
NONCE = "STBENCH_CASE_NONCE"
# What the agent below writes, as a state folder holds it afterwards: a file a variable, in that variable's folder.
WRITTEN = sorted(["home/HOME.txt", "home/USERPROFILE.txt", "home/AppData/Roaming/APPDATA.txt",
                  "home/AppData/Local/LOCALAPPDATA.txt", "home/.config/XDG_CONFIG_HOME.txt",
                  "home/.local/share/XDG_DATA_HOME.txt", "home/.local/state/XDG_STATE_HOME.txt",
                  "home/.cache/XDG_CACHE_HOME.txt", "home/.claude/CLAUDE_CONFIG_DIR.txt", "home/.codex/CODEX_HOME.txt"])

# At `start`, before it writes anything, it records what it finds: the files under each variable's folder, the
# variables themselves, the nonce, a credential variable of the host's, and where `Path.home()` points, now and when
# its file was imported. Then it writes `<VARIABLE>.txt` into each variable's folder; `finish` returns the record.
RECORD = r'''
import os
from pathlib import Path

VARIABLES = @VARIABLES@
HOME_AT_IMPORT = str(Path.home())


def files_under(variable):
    top = os.environ.get(variable)
    if top is None:
        return None
    return sorted(p.relative_to(top).as_posix() for p in Path(top).rglob("*") if p.is_file())


class Record:
    def __init__(self, state_dir, workdir, **opts):
        self.record = {}

    def start(self, session, robot, channel):
        self.record = {"found": {v: files_under(v) for v in VARIABLES},
                       "env": {v: os.environ.get(v) for v in VARIABLES},
                       "nonce": os.environ.get("STBENCH_CASE_NONCE"),
                       "credential": os.environ.get("STBENCH_TEST_CREDENTIAL"),
                       "home": str(Path.home()), "home_at_import": HOME_AT_IMPORT}
        for v in VARIABLES:
            if v in os.environ:
                (Path(os.environ[v]) / (v + ".txt")).write_text(v)

    def observe(self, steps):
        pass

    def answer(self, question):
        return {"type": "unknown"}

    def finish(self):
        return self.record
'''


def serve_in_process(tmp_path, monkeypatch, name: str, environ) -> tuple[Path, list]:
    """Serve a `start` and a `finish` of the do-nothing system in this process, into a state folder of its own,
    `tmp_path/name/state`; returns that folder with the replies."""
    state, work = tmp_path / name / "state", tmp_path / name / "work"
    state.mkdir(parents=True)
    work.mkdir()
    child = write_child(tmp_path / name / "child", "agent.py", ECHO)
    monkeypatch.setattr(sys, "path", list(sys.path))  # `load` puts the system's folder first
    monkeypatch.setitem(sys.modules, "system_under_test", None)
    requests = [start_message(SESSION, ROBOT, "rgb_pose", str(work), str(state), {}), {"op": "finish"}]
    out = io.BytesIO()
    shim.serve(str(child), "Echo", io.BytesIO(b"".join(json.dumps(r).encode() + b"\n" for r in requests)), out,
               environ)
    return state, [json.loads(line) for line in out.getvalue().decode().splitlines()]


def test_the_shim_changes_only_the_environment_it_is_given(tmp_path, monkeypatch):
    """Contract (docs/agent.md "Memory"): at `start` the shim updates the mapping it was given, and only that, with the
    documented table (each variable at its folder under `state/home`) and the session as the nonce; the folders exist
    and hold nothing; the process's own environment is left alone, and a call with no mapping sets nothing and makes no
    home. Regression caught: a shim test that changes the test worker's `HOME` for every later test, or a table that
    drifts from docs/agent.md."""
    host = dict(os.environ)
    monkeypatch.setattr(os, "environ", dict(host))  # a shim that wrote to it would change this copy, not the worker's
    environ: dict[str, str] = {}
    state, replies = serve_in_process(tmp_path, monkeypatch, "given", environ)
    assert replies == [{"ok": True}, {"ok": True, "usage": {}}]
    home = state / "home"
    assert environ == {**{variable: str(home / folder) for variable, folder in TABLE.items()}, NONCE: SESSION}
    assert all(os.path.isdir(path) for variable, path in environ.items() if variable in TABLE)
    assert [p for p in home.rglob("*") if p.is_file()] == []
    assert os.environ == host

    state, replies = serve_in_process(tmp_path, monkeypatch, "none", None)
    assert replies == [{"ok": True}, {"ok": True, "usage": {}}]
    assert not (state / "home").exists() and os.environ == host


def two_cases(tmp_path, build) -> list[tuple[Path, dict]]:
    """Two cases of the `Record` agent, one after the other, each with state and working folders of its own and its own
    session. `build(state, work)` makes the system; the messages are sent by hand, not through `run_case`, which keeps
    only the usage keys. Returns each case's state folder with the record its `finish` handed back."""
    cases = []
    for n, session in enumerate(SESSIONS):
        state, work = tmp_path / f"state{n}", tmp_path / f"work{n}"
        state.mkdir()
        work.mkdir()
        system = build(state, work)
        try:
            system.start(session, ROBOT, "rgb_pose")
            cases.append((state, system.finish()))
        finally:
            system.close()
    return cases


def check_cases(cases: list[tuple[Path, dict]], state_as_seen, credential: str | None) -> None:
    """What each case's agent found and left: every variable at its folder under its own case's `state/home` (the state
    folder as the agent sees it is `state_as_seen(state)`), `Path.home()` there at import and later, nothing found
    under any of them (in the second case too, though the first wrote under each), the nonce its own session, the
    host's credential variable as given, and, after the case, its ten files in its own state folder, which the state
    size counts."""
    for (state, record), session in zip(cases, SESSIONS, strict=True):
        home = state_as_seen(state) / "home"
        assert record["env"] == {variable: str(home / folder) for variable, folder in TABLE.items()}
        assert record["found"] == {variable: [] for variable in TABLE}
        assert record["home"] == record["home_at_import"] == str(home)
        assert record["nonce"] == session
        assert record["credential"] == credential
        assert sorted(p.relative_to(state).as_posix() for p in state.rglob("*") if p.is_file()) == WRITTEN
        assert tree_bytes(state) == sum(len(variable) for variable in TABLE)  # each file holds its variable's name


def test_a_second_case_finds_nothing_of_the_first(tmp_path, monkeypatch):
    """Contract (docs/agent.md "Memory", with `--local`): each case's agent gets a home of its own, inside its own state
    folder, under each of the ten variables where tools keep their memory, so the second case finds nothing of what
    the first wrote there; the nonce is the case's session, a credential variable of the host's reaches the agent by
    name (a `--local` agent inherits the shell's variables), and what the first case wrote sits in its own state
    folder, where its size is counted. Regression caught: built-in memory (Claude Code's, Codex's) carried from one
    case to the next through the user's home, a home outside the state folder (unmeasured, map HA-H7), or a named
    credential that does not reach a `--local` agent."""
    monkeypatch.setenv("STBENCH_TEST_CREDENTIAL", "not-a-secret")
    file = write_child(tmp_path / "agent", "agent.py", RECORD.replace("@VARIABLES@", repr(tuple(TABLE))))
    cases = two_cases(tmp_path, lambda state, work: LocalCommandSystem(state, work, file, "Record", timeouts_of(60)))
    check_cases(cases, lambda state: state.resolve(), "not-a-secret")  # `start` names the resolved folder


@pytest.mark.docker
def test_a_second_case_finds_nothing_of_the_first_in_docker(tmp_path, monkeypatch):
    """Contract (docs/agent.md "Memory", the verified mode): the same in a container, with container paths
    (`/sandbox/state/home/...`, the state mount), the host's credential variable absent (Docker passes no host
    variable), and the host's state folder holding the first case's ten files afterwards. Regression caught: the
    same isolation lost on Linux, and a host variable leaking into a container."""
    monkeypatch.setenv("STBENCH_TEST_CREDENTIAL", "not-a-secret")
    file = write_child(tmp_path / "agent", "agent.py", RECORD.replace("@VARIABLES@", repr(tuple(TABLE))))
    with sandbox.program(file, "Record") as prog:
        cases = two_cases(tmp_path, lambda state, work: CommandSystem(state, work, program=prog,
                                                                      timeouts=timeouts_of(60)))
    check_cases(cases, lambda state: PurePosixPath("/sandbox/state"), None)
