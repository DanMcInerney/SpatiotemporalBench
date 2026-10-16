"""`CommandSystem`'s own mechanics (V0 §9, VISION rule 6), with its container replaced by a Python child and Docker by a
script that records what the harness asks of it: no Docker, so these run everywhere. What a real container does is
tested in test_command_system.py (marker `docker`)."""

from __future__ import annotations

import os
import subprocess

import pytest

import fake_docker
from sandbox_children import call_within
from stbench import sandbox, system
from stbench.stream_protocol import TransportError
from stbench.system import CommandSystem, SystemTimeout, timeouts_of

QUESTION = {"qid": "q1", "text": "How many?", "answer_type": "count"}
NOW = {"k": 0, "clock": "2026-03-02T08:00:00"}


@pytest.fixture
def local(tmp_path, monkeypatch):
    """`local.make(mode)` is a `CommandSystem` whose container is the FAKE child; `local.docker` is its Docker."""
    yield from fake_docker.local(tmp_path, monkeypatch)


def test_a_stopped_instance_removes_its_container_again_once_its_cli_is_gone(local, monkeypatch):
    """Contract (HANDOFF 3.2, only stop what you started): after a failure the instance removes its container, kills
    and reaps its own docker CLI, removes the container once more and waits until Docker no longer lists it; one that
    is still listed after `REMOVE_WAIT` is an error, but when the stop follows another failure that one is raised and
    the removal's is attached to it as a note (a `close` has no other failure to report, so it raises its own).
    Regression caught: a CLI killed in the middle of `create` leaves a Created container that the first `rm -f`, run
    before the kill, never saw, so a failed or timed-out instance leaked a container; a removal failure that replaces
    the error that caused the stop (the run then reports "still there" for a system that broke the protocol)."""
    cs = local.make("garbage")
    local.docker.inspect = 2  # listed for two polls, then gone
    with pytest.raises(RuntimeError, match="broke the protocol"):
        cs.observe([])
    rm = [(args, alive) for args, alive in local.docker.calls if args[0] == "rm"]
    assert rm == [(("rm", "-f", cs.name), True), (("rm", "-f", cs.name), False)]  # the second after the CLI was reaped
    assert len(local.docker.commands("container")) == 3
    assert cs._proc.poll() is not None

    monkeypatch.setattr(system, "REMOVE_WAIT", 0.3)
    stuck = local.make("garbage")
    local.docker.inspect = 10**6  # never gone
    with pytest.raises(RuntimeError, match="broke the protocol") as failed:
        stuck.observe([])
    assert any("still there" in note for note in failed.value.__notes__)
    monkeypatch.setattr(system, "CLOSE_GRACE", 0.2)  # a child that does not end with its input is removed
    with pytest.raises(RuntimeError, match="still there"):
        local.make("garbage").close()


def test_closing_never_waits_for_a_request_the_child_has_not_read(local):
    """Contract (HANDOFF 3.2, deadlines): when the last request is still being written because the child replied
    without reading all of it, `close` removes the container at once instead of ending the input, which would wait for
    that write for good. Regression caught: a harness hung in `close` by a system that stops reading."""
    cs = local.make("partial", timeouts=timeouts_of(30))
    cs.observe([{"blob": "x" * 2**21}])  # far over any pipe buffer: the child answers after ten bytes
    assert cs._pipe.writing
    call_within(30, cs.close)
    assert local.docker.commands("rm")[0] == ("rm", "-f", cs.name) and cs._proc.poll() is not None


def test_the_constructor_refuses_what_it_must_before_it_starts_anything(tmp_path, monkeypatch):
    """Contract (VISION rule 6, HANDOFF 3.2): a state folder that overlaps a path the sandbox must not reveal, and a
    per-message timeout under `MIN_TIMEOUT` (which every start would exceed), are refused before any process starts.
    Regression caught: the mount-source check removed from the constructor, and a timeout so small that no system can
    start within it."""
    monkeypatch.setattr(sandbox, "require", lambda: "docker")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started"))
    data = tmp_path / "data"
    program = sandbox.Program(tmp_path / "program", "x.py", "X", deny=(os.path.normcase(os.path.realpath(data)),))
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="overlaps"):
        CommandSystem(data / "state", work, program=program)
    for timeout in (1, 0, -5, float("nan")):
        with pytest.raises(ValueError, match="at least 5"):
            CommandSystem(tmp_path / "state", work, program=program, timeouts=timeouts_of(timeout))
    with pytest.raises(ValueError, match="must name exactly"):
        CommandSystem(tmp_path / "state", work, program=program, timeouts={"observe": 60.0})


def test_each_op_is_sent_as_its_message_with_its_own_timeout(local):
    """Contract (plan §2.4): `start` sends the table's message with the container's paths, never the host's, and `task`,
    `observe`, `ask` and `finish` send theirs; each waits for its reply at most its own entry of `timeouts`
    (`ask_during` for a during ask, `ask_end` for the end ask); `ask` returns the reply's answers list and `finish` its
    usage; the constructor sent nothing. Hand-written messages. Regression caught: a start that names host paths a
    container cannot see, one timeout for every message (the end ask's 3,600 s cut to the others' 600), and a lost
    session or option."""
    timeouts = {"start": 11.0, "task": 12.0, "observe": 13.0, "ask_during": 14.0, "ask_end": 15.0, "finish": 16.0}
    cs = local.make(start=False, timeouts=timeouts, depth=2)
    sent = []
    real = cs._pipe.request
    cs._pipe.request = lambda message, timeout: (sent.append((message, timeout)), real(message, timeout))[1]
    assert sent == []
    cs.start(fake_docker.SESSION, fake_docker.ROBOT, "detections")
    cs.task(0, "2026-03-02T08:00:00", 0.0, "Clean it.")
    cs.observe([{"k": 0, "clock": "2026-03-02T08:00:00"}])
    assert cs.ask("during", NOW, [QUESTION]) == [{"qid": "q1", "answer": {"type": "unknown"}}]
    cs.ask("end", NOW, [QUESTION])
    assert cs.finish() == {"tokens_in": 7}
    assert sent == [
        ({"op": "start", "session": "0123456789abcdef", "robot": {"name": "robot vacuum"}, "channel": "detections",
          "sends": ["depth", "detections", "pose", "status"], "workdir": "/sandbox/work",
          "state_dir": "/sandbox/state", "opts": {"depth": 2}}, 11.0),
        ({"op": "task", "index": 0, "clock": "2026-03-02T08:00:00", "gap_s": 0.0, "brief": "Clean it."}, 12.0),
        ({"op": "observe", "steps": [{"k": 0, "clock": "2026-03-02T08:00:00"}]}, 13.0),
        ({"op": "ask", "when": "during", "now": NOW, "questions": [QUESTION]}, 14.0),
        ({"op": "ask", "when": "end", "now": NOW, "questions": [QUESTION]}, 15.0),
        ({"op": "finish"}, 16.0)]


def test_a_failed_message_raises_what_broke_and_keeps_the_container_only_if_the_pipe_is_intact(local):
    """Contract (plan §2.4): an `{"error"}` reply and a reply no op expects raise a `RuntimeError` that is no
    `TransportError`, naming the system and what it said, and leave the container serving (the next message is
    answered); a container that exits or breaks the protocol raises a `TransportError`, a timeout a `SystemTimeout` (a
    `TransportError` too), and both remove the container, since the pipe is broken for good. Regression caught: an
    ask error that ends the case like a broken pipe (or a broken pipe that lets the case go on and pairs a late reply
    with a later request), and a timeout that leaves its container running."""
    cs = local.make()
    with pytest.raises(RuntimeError, match=r"system x\.py:X.*'task' failed: ValueError: boom") as error:
        cs.task(0, "c", 0.0, "error")
    assert not isinstance(error.value, TransportError)
    with pytest.raises(RuntimeError, match="unexpected reply to 'task'") as weird:
        cs.task(1, "c", 0.0, "weird")
    assert not isinstance(weird.value, TransportError)
    cs.observe([])  # the pipe is intact: the container still answers
    assert cs._proc.poll() is None and local.docker.commands("rm") == []

    dying = {**QUESTION, "die": True}
    with pytest.raises(TransportError, match=r"(?s)exited without answering 'ask'.*goodbye") as exited:
        call_within(60, lambda: cs.ask("end", NOW, [dying]))
    assert not isinstance(exited.value, SystemTimeout) and cs._proc.poll() is not None
    assert local.docker.commands("rm")[0] == ("rm", "-f", cs.name)

    slow = local.make()
    slow.timeouts["ask_during"] = 0.5  # after the start, which has its own share of the clock
    with pytest.raises(SystemTimeout, match=r"did not answer 'ask' within 0\.5 s; its container was removed") as late:
        call_within(60, lambda: slow.ask("during", NOW, [{**QUESTION, "sleep": 30}]))
    assert isinstance(late.value, TransportError) and slow._proc.poll() is not None
    assert ("rm", "-f", slow.name) in local.docker.commands("rm")
