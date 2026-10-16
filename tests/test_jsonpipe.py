"""The transport of `system.CommandSystem` (HANDOFF 3.2, VISION rule 6): JSON lines to a child, a deadline per message.

Each test runs a tiny stdlib-only child process: no Docker, no network, nothing shared between tests.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from sandbox_children import call_within
from stbench import jsonpipe
from stbench.jsonpipe import JsonPipe, PipeClosed, PipeError, PipeTimeout

# On a Windows virtual environment `sys.executable` is a launcher that starts the real interpreter as a child process:
# killing the launcher would leave that child alive, holding the pipes. The children need nothing but the standard
# library, so they run on the base interpreter, a single process that `kill` ends.
PYTHON = getattr(sys, "_base_executable", sys.executable)

SLEEP = "import time; time.sleep(120)"  # reads nothing, replies nothing
# Fills stderr, ends stdout first (the two pipes need not end together), says its last words late, exits with 3.
NOISY_EXIT = """
import os, sys, time
sys.stderr.write('x' * 100000)
sys.stderr.flush()
os.close(1)
time.sleep(0.3)
sys.stderr.write('boom')
sys.exit(3)
"""
ECHO = """
import json, sys
for line in sys.stdin:
    sys.stdout.write(json.dumps({"echo": json.loads(line)}) + "\\n")
    sys.stdout.flush()
"""


def replies_with(text: str) -> str:
    """A child that reads one request, answers it with exactly `text`, and exits."""
    return f"import sys; sys.stdin.readline(); sys.stdout.write({text!r}); sys.stdout.flush()"


@contextmanager
def running(code: str) -> Iterator[JsonPipe]:
    """A JsonPipe on a child of this test, which is killed and reaped when the block ends."""
    proc = subprocess.Popen(
        [PYTHON, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        yield JsonPipe(proc)
    finally:
        proc.kill()
        proc.wait()
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            with contextlib.suppress(OSError):  # a writer cut off mid-message leaves bytes that cannot be flushed
                pipe.close()


def ask(pipe: JsonPipe, message: dict, timeout: float = 5.0) -> dict:
    return call_within(timeout + 10, lambda: pipe.request(message, timeout))


def test_a_request_times_out_even_when_the_child_reads_nothing():
    """Per-message timeouts (HANDOFF 3.2): a child that never reads cannot hold the harness past the deadline.
    Regression caught: a write on a full pipe, or a read, with no deadline, which hangs the whole benchmark."""
    with running(SLEEP) as pipe:
        started = time.monotonic()
        with pytest.raises(PipeTimeout):
            ask(pipe, {"op": "observe", "steps": ["x" * 2**21]}, timeout=1.0)  # 2 MiB: far over any pipe buffer
        assert time.monotonic() - started >= 1.0  # it waited for the deadline, it did not fail at once
        with pytest.raises(PipeClosed):
            ask(pipe, {"op": "answer"})  # the late reply of a timed-out request must never answer a later one


def test_a_child_that_exits_is_reported_with_the_end_of_its_stderr():
    """An error names what the child said last, and the harness holds only a bounded tail of it.
    Regression caught: a crash that looks like silence (no stderr in the message), or stderr kept without a bound."""
    with running(NOISY_EXIT) as pipe:
        with pytest.raises(PipeClosed):
            ask(pipe, {"op": "start"})
        assert pipe.stderr_tail() == "x" * 1996 + "boom"  # the last 2000 characters, and exactly those
        assert len(pipe.stderr_tail(10**9)) <= jsonpipe.STDERR_KEEP


def test_a_reply_too_long_or_not_a_json_object_is_a_protocol_error(monkeypatch):
    """A reply is one bounded line holding one JSON object; anything else is a PipeError, never an answer.
    Regression caught: memory growing with one huge line, and a list or text passed on as an answer."""
    monkeypatch.setattr(jsonpipe, "MAX_LINE", 1000)
    bad = {
        "longer": "a" * 2000 + "\n",
        "object": "[1, 2]\n",
        "JSON": "not json\n",
    }  # what the message must name: the right check fired, not just any
    for what, text in bad.items():
        with running(replies_with(text)) as pipe, pytest.raises(PipeError, match=what) as caught:
            ask(pipe, {"op": "start"})
        assert type(caught.value) is PipeError  # a protocol error, neither a timeout nor an exit


def test_replies_pair_with_requests_in_order_and_survive_non_ascii_and_large_messages():
    """The wire is one JSON object per line each way, one reply per request (V0 section 9, worker protocol).
    Regression caught: a transport that fails every request, drops non-ASCII text, or mispairs replies."""
    with running(ECHO) as pipe:
        big = {"op": "observe", "steps": ["x" * 2**20, "Zürich — café \U0001f600"]}  # 1 MiB, then non-ASCII
        assert ask(pipe, big) == {"echo": big}
        assert ask(pipe, {"op": "answer", "n": 2}) == {"echo": {"op": "answer", "n": 2}}
