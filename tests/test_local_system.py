"""`LocalCommandSystem`: an agent as a plain local subprocess through
the shim, speaking the stream protocol. No Docker, so these run everywhere; the agents are small source texts
(local_kit.py and stream_kit.py)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

import local_kit
import stbench
from sandbox_children import BOOM, call_within, write_child
from stbench import local_system
from stbench.local_system import LocalCommandSystem
from stbench.stream import run_case
from stbench.stream_protocol import TransportError
from stbench.system import SystemTimeout, timeouts_of
from stream_kit import check_scripted_run, scripted_case

SESSION, ROBOT = "0123456789abcdef", {"name": "robot vacuum"}
NOW = {"k": 3, "clock": "2026-03-02T08:00:03"}


@pytest.fixture
def folders(tmp_path):
    """`(state, work)`: two empty folders for a system built by hand."""
    state, work = tmp_path / "state", tmp_path / "work"
    state.mkdir()
    work.mkdir()
    return state, work


def question(qid: str = "q1") -> dict:
    return {"qid": qid, "text": "How many?", "answer_type": "count"}


def test_an_agent_that_dies_fails_the_run_with_the_end_of_its_stderr(folders, tmp_path):
    """Contract: a process that exits without answering, in a message or while starting,
    raises `TransportError` (a `RuntimeError`, and the one that ends a case) naming the message, the exit code and the
    end of its stderr, so the user sees why. Regression caught: a bare "pipe closed", or a crash that hangs the run."""
    state, work = folders
    crash = write_child(tmp_path / "crash", "agent.py", local_kit.CRASH)
    system = LocalCommandSystem(state, work, crash, "Crash", timeouts_of(60))
    system.start(SESSION, ROBOT, "rgb_pose")
    with pytest.raises(TransportError, match=r"(?s)\(code 3\) without answering 'observe'.*goodbye from"):
        call_within(60, lambda: system.observe([]))
    system.close()
    boom = write_child(tmp_path / "boom", "agent.py", BOOM)  # dies while it is being imported, at `start`
    dying = LocalCommandSystem(state, work, boom, "Boom", timeouts_of(60))
    with pytest.raises(TransportError, match=r"(?s)'start'.*boom"):
        call_within(60, lambda: dying.start(SESSION, ROBOT, "rgb_pose"))
    dying.close()


def test_a_slow_agent_times_out_and_is_not_left_running(folders, tmp_path):
    """Contract (timeouts as in `CommandSystem`): a message not answered within `timeout`
    raises `SystemTimeout` and kills the agent's whole process tree, so it does not run on after the run has failed.
    The agent ticks into a file from a thread and never answers; once the error is raised the file stops growing.
    Regression caught: a timeout that raises but leaves the agent running (on Windows the virtual environment's
    `python.exe` is a launcher, and killing it alone leaves the real interpreter), and a hang past the deadline."""
    state, work = folders
    beat = write_child(tmp_path / "beat", "agent.py", local_kit.BEAT)
    system = LocalCommandSystem(state, work, beat, "Beat", timeouts_of(60))
    system.start(SESSION, ROBOT, "rgb_pose")
    system.observe([])
    ticks = state / "beat.txt"
    deadline = time.monotonic() + 30
    while (not ticks.exists() or ticks.stat().st_size < 3) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ticks.stat().st_size >= 3  # it is running
    system.timeouts["ask_during"] = 1.0  # after the start, which has its own, larger share of the clock
    with pytest.raises(SystemTimeout, match=r"did not answer 'ask' within 1 s") as late:
        call_within(60, lambda: system.ask("during", NOW, [question()]))
    assert isinstance(late.value, TransportError)
    size = ticks.stat().st_size
    time.sleep(0.6)
    assert ticks.stat().st_size == size  # nothing ticks any more
    system.close()


def test_each_instance_is_its_own_isolated_process_in_a_temp_folder(folders, tmp_path, monkeypatch):
    """Contract: an instance runs as `python -I -B <shim> FILE CLASS` with its working
    directory in a fresh temp folder (not the harness's, not the state's), removed when it closes, and the options the
    constructor was given reach the agent, with the session, robot and channel of `start` and the host paths of its
    folders. Regression caught: an agent that runs in the harness's folder (its relative writes land in the repository
    or the dataset), a folder left behind by a closed instance, options lost, and a `start` that names no session."""
    state, work = folders
    where = write_child(tmp_path / "where", "agent.py", local_kit.WHERE)
    started = []
    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (started.append(a), real(*a, **k))[1])
    live = LocalCommandSystem(state, work, where, "Where", timeouts_of(60), depth=3)
    live.start(SESSION, ROBOT, "rgb_pose")
    first = live.ask("end", NOW, [question()])[0]["answer"]
    assert first["pid"] != os.getpid() and first["opts"] == {"depth": 3} and len(started) == 1
    assert first["started"] == [SESSION, ROBOT, "rgb_pose"]
    assert first["paths"] == [str(state.resolve()), str(work.resolve())]
    assert first["flags"] == [1, 1]  # python -I -B: isolated, no bytecode written next to the agent
    assert started[0][0][:3] == [sys.executable, "-I", "-B"] and Path(started[0][0][3]).name == "shim.py"
    here = Path(first["cwd"]).resolve()
    assert here.is_dir() and here.parent == Path(tempfile.gettempdir()).resolve()
    assert here not in (Path.cwd().resolve(), state.resolve(), work.resolve())

    live.close()
    assert not here.exists() and live._proc.poll() is not None  # the instance and its folder are gone


def test_the_constructor_refuses_what_it_must_before_it_starts_anything(folders, tmp_path, monkeypatch):
    """Contract: a system folder that contains, or sits inside, the dataset, a
    file that is not a Python file, a class name that is not one, and a per-message timeout under `MIN_TIMEOUT` are
    refused before any process starts; a folder beside the dataset is fine. Regression caught: a local agent that can
    walk straight into the dataset (its folder is the dataset's), and a timeout so small that no process can start."""
    state, work = folders
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started"))
    data = tmp_path / "dataset"
    inside = write_child(data / "agent", "agent.py", local_kit.WHERE)
    outer = write_child(tmp_path, "outer.py", local_kit.WHERE)  # its folder, tmp_path, contains the dataset
    for file in (inside, outer):
        with pytest.raises(ValueError, match="must not contain or sit inside the dataset"):
            LocalCommandSystem(state, work, file, "Where", dataset=data)
    beside = write_child(tmp_path / "beside", "agent.py", local_kit.WHERE)
    with pytest.raises(ValueError, match="is not an existing Python file"):
        LocalCommandSystem(state, work, beside.with_name("missing.py"), "Where", dataset=data)
    with pytest.raises(ValueError, match="is not a class name"):
        LocalCommandSystem(state, work, beside, "not a class", dataset=data)
    for timeout in (1, 0, -5, float("nan")):
        with pytest.raises(ValueError, match="at least 5"):
            LocalCommandSystem(state, work, beside, "Where", timeouts_of(timeout), dataset=data)
    monkeypatch.undo()
    ok = LocalCommandSystem(state, work, beside, "Where", dataset=data)  # a folder beside the dataset: no error
    ok.close()


def test_the_shim_is_found_as_package_data_not_through_the_repository_layout(tmp_path):
    """Contract: `LocalCommandSystem` finds `shim.py` with `importlib.resources`, beside its
    own module, so it works from an installed package that has no checkout around it. The whole package is copied to
    `site/stbench` (no `src/`, no repository two levels up) and an agent is run from there by a fresh interpreter: the
    process that answers is running the copy's `shim.py`. Regression caught: the shim looked for through
    `Path(__file__).parents[2] / "src" / ...`, which exists in a checkout and in no wheel."""
    site = tmp_path / "site"
    shutil.copytree(Path(stbench.__file__).parent, site / "stbench", ignore=shutil.ignore_patterns("__pycache__"))
    agent = write_child(tmp_path / "where", "agent.py", local_kit.WHERE)
    state, work = tmp_path / "state", tmp_path / "work"
    state.mkdir()
    work.mkdir()
    script = "\n".join([
        "import json, sys",
        f"sys.path.insert(0, {str(site)!r})",
        "from pathlib import Path",
        "from stbench import local_system",
        f"state, work, agent = Path({str(state)!r}), Path({str(work)!r}), Path({str(agent)!r})",
        "from stbench.system import timeouts_of",
        "s = local_system.LocalCommandSystem(state, work, agent, 'Where', timeouts_of(60))",
        "s.start('0123456789abcdef', {'name': 'robot vacuum'}, 'rgb_pose')",
        "answers = s.ask('end', {'k': 0, 'clock': 'c'}, [{'qid': 'q', 'text': 't', 'answer_type': 'count'}])",
        "print(json.dumps({'module': local_system.__file__, 'answer': answers[0]['answer']}))",
        "s.close()",
    ])
    done = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.splitlines()[-1])
    assert Path(found["module"]).resolve() == (site / "stbench" / "local_system.py").resolve()
    assert Path(found["answer"]["shim"]).resolve() == (site / "stbench" / "shim.py").resolve()
    assert local_system.shim_path().name == "shim.py" and local_system.shim_path().is_file()


def test_a_scripted_system_completes_a_case_with_local(tmp_path):
    """Contract (plan §4.4, B.1's exit check): a scripted standard-library system runs a synthetic case end to end
    through `run_case`, `LocalCommandSystem` and the shim: all five ops reach it, its three answers come back (the
    during one after 13 steps, the end ones after 20), its usage is stamped self-reported, every message is timed and
    its state is sized at the task's end and at finish. Regression caught: an adapter or the shim dropping an op, so
    the case ends or an answer is lost."""
    case, file = scripted_case(tmp_path)
    run = run_case(lambda s, w: LocalCommandSystem(s, w, file, "Scripted", timeouts_of(60), dataset=case.dataset),
                   case)
    check_scripted_run(run)
    assert run.state_bytes["0"] == 1000
