"""Running an external system: `CommandSystem`, a container speaking JSON lines to `system_shim.py` there.

The interface a system implements, `System`, lives in `stream_protocol.py` with the messages, the frame rule, the
timeouts and the failure rules; it is imported here, so `from .system import System` keeps working for the baselines.
A system keeps its memory in the state directory it is given and sees the working folder's `public/` (the robot and
camera files) and `inbox/` (only the files of the frame being fed): the harness empties the inbox before every frame, so
nothing from the future is readable except what the system stored.

External systems run in their own container: `CommandSystem` starts one in the Docker sandbox (`sandbox.py`) and speaks
the stream protocol to `system_shim.py` there, through `PipedSystem`, which `local_system.LocalCommandSystem` shares.
In-process systems are the benchmark's own code: the scripted baselines and the reference LLM system (`codex:`), which a
result stamps `sandbox: in-process`.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time

from . import sandbox
from .jsonpipe import JsonPipe, PipeClosed, PipeError, PipeTimeout
from .stream_protocol import (
    TIMEOUTS,
    System,
    TransportError,
    check_reply,
    start_message,
    timeout_key,
)

UNKNOWN = {"type": "unknown"}
MIN_TIMEOUT = 5.0  # seconds: the least a message timeout may be (a container takes about a second to start)
CLOSE_GRACE = 10.0  # seconds it has to exit after its input ends
REMOVE_WAIT = 30.0  # seconds a removed container has to disappear from Docker's list (the wait ends when it has)


class SystemTimeout(TransportError):
    """An external system did not answer within its message timeout."""


def timeouts_of(seconds: float) -> dict[str, float]:
    """Every message of `TIMEOUTS` given the same `seconds` (`stbench run --timeout`)."""
    return dict.fromkeys(TIMEOUTS, float(seconds))


def check_timeouts(timeouts: dict[str, float]) -> dict[str, float]:
    """A copy of `timeouts` as floats. ValueError unless it names exactly the messages of `TIMEOUTS` and each is at
    least `MIN_TIMEOUT` seconds (a container or process takes about a second to start; NaN is refused too)."""
    if set(timeouts) != set(TIMEOUTS):
        raise ValueError(f"the message timeouts must name exactly {', '.join(TIMEOUTS)}, got {', '.join(timeouts)}")
    for key, value in timeouts.items():
        if not value >= MIN_TIMEOUT:  # also false for NaN
            raise ValueError(f"each message timeout must be at least {MIN_TIMEOUT:g} s (a container takes about a "
                             f"second to start), got {key}: {value!r}")
    return {key: float(value) for key, value in timeouts.items()}


class PipedSystem(System):
    """An external system behind a `JsonPipe`: the five messages of the stream protocol, each sent with its own timeout
    (`timeouts[timeout_key(op, when)]`) and its reply checked by `stream_protocol.check_reply`. A subclass owns the
    child's life: it sets `_pipe` (a `JsonPipe`), says who it is (`_who`), where the system sees its folders (`_paths`)
    and how the child is stopped (`_stopped_by`).

    What a failed message raises: a timeout `SystemTimeout`, a child that exited or broke the protocol
    `TransportError` (both stop the child: the pipe is broken for good, and the case ends); an `{"error"}` reply or a
    reply `check_reply` refuses, `RuntimeError` (the pipe is intact, and the child stays: whether the case goes on is
    the caller's to say). `ask` returns the reply's `answers` list (`stream_protocol.read_answers` reads it) and
    `finish` its `usage`."""

    _gone = "it was stopped"  # how a timeout's message says what became of the child

    def __init__(self, state_dir, workdir, timeouts: dict[str, float], **opts):
        super().__init__(state_dir, workdir, **opts)
        self.timeouts = check_timeouts(timeouts)
        self._pipe: JsonPipe | None = None

    @property
    def _who(self) -> str:
        raise NotImplementedError

    def _paths(self) -> tuple[str, str]:
        """(workdir, state_dir) as the system sees them: the paths its `start` message names."""
        raise NotImplementedError

    def _stopped_by(self, error: BaseException) -> BaseException:
        """Stop the child on behalf of `error`, which the caller raises (and this returns); a failure to stop is added
        to `error` as a note and never raised in its place."""
        raise NotImplementedError

    def _exit_note(self) -> str:
        """What a child that exited without answering adds to the message, such as its exit code. Default: nothing."""
        return ""

    def _send(self, message: dict) -> dict:
        op, who = message["op"], self._who
        timeout = self.timeouts[timeout_key(op, message.get("when"))]
        try:
            reply = self._pipe.request(message, timeout)
        except PipeTimeout:
            raise self._stopped_by(SystemTimeout(f"{who} did not answer {op!r} within {timeout:g} s; "
                                                 f"{self._gone}")) from None
        except PipeClosed:
            raise self._stopped_by(TransportError(f"{who} exited{self._exit_note()} without answering {op!r}: "
                                                  f"{self._pipe.stderr_tail()}")) from None
        except PipeError as e:
            raise self._stopped_by(TransportError(f"{who} broke the protocol on {op!r}: {e}")) from None
        if (reason := check_reply(op, reply)) is not None:  # the child stays: the caller closes it
            raise RuntimeError(f"{who}: {reason}")
        return reply

    def start(self, session: str, robot: dict, channel: str) -> None:
        """Send `start`, with the paths the system sees. A child that fails to start is stopped, never left behind."""
        super().start(session, robot, channel)
        workdir, state_dir = self._paths()
        try:
            self._send(start_message(session, robot, channel, workdir, state_dir, self.opts))
        except BaseException as error:
            self._stopped_by(error)
            raise

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        self._send({"op": "task", "index": index, "clock": clock, "gap_s": gap_s, "brief": brief})

    def observe(self, steps: list[dict]) -> None:
        self._send({"op": "observe", "steps": steps})

    def ask(self, when: str, now: dict, questions: list[dict]) -> list[dict]:  # the wire's answers, not a dict by qid
        return self._send({"op": "ask", "when": when, "now": now, "questions": questions})["answers"]

    def finish(self) -> dict:
        return self._send({"op": "finish"})["usage"]


class CommandSystem(PipedSystem):
    """An external system run in the Docker sandbox (`sandbox.py`): one container, speaking the stream protocol
    through `system_shim.py` (`shim.py` is the other end; the harness sends container paths, never host paths). The
    constructor starts the container and sends nothing: `start` sends the first message.

    The system keeps its memory in its state directory. Each message must be answered within its `timeouts` entry
    (`MIN_TIMEOUT` at least), or the container is removed and `SystemTimeout` raised; a system that dies, breaks the
    protocol or fails to start has its container removed too. Docker is the verified mode: without Docker and the
    sandbox image the constructor raises `sandbox.SandboxUnavailable` (`--local` runs an agent unsandboxed through
    `local_system.LocalCommandSystem` instead). `memory` belongs to the harness: it never reaches the system's `opts`.
    """

    _gone = "its container was removed"

    def __init__(self, state_dir, workdir, program: sandbox.Program, timeouts: dict[str, float] = TIMEOUTS, *,
                 memory: str = sandbox.MEMORY, **opts):
        super().__init__(state_dir, workdir, timeouts, **opts)
        self.program = program
        self.memory = memory
        self._closed = self._stopped = False
        self._life = threading.RLock()  # holds `_proc` and `_stopped` still while a start or a stop is under way
        self._proc = None
        self._docker = sandbox.require()  # SandboxUnavailable, before anything starts
        sandbox.check_sources(program, self.state_dir, self.workdir)
        self.name = sandbox.container_name()
        self._launch()

    @property
    def _who(self) -> str:
        return f"system {self.program.entry}:{self.program.cls} (container {self.name})"

    def _paths(self) -> tuple[str, str]:
        return sandbox.WORK, sandbox.STATE

    def _launch(self) -> None:
        """Start the container; its shim waits for the `start` message."""
        with self._life:
            args = sandbox.container_args(self._docker, self.name, self.state_dir, self.workdir, self.program,
                                          memory=self.memory)
            self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            self._pipe = JsonPipe(self._proc)

    def close(self) -> None:
        """Idempotent. End its input; if it has not exited within `CLOSE_GRACE`, remove its container. If the last
        request is still being written (the child has not read all of it), ending the input would wait for that write
        for good, so the container is removed at once."""
        if self._closed:
            return
        self._closed = True
        if self._pipe.writing:
            self._stop()
            return
        with contextlib.suppress(OSError, ValueError):
            self._proc.stdin.close()
        try:
            self._proc.wait(CLOSE_GRACE)
        except subprocess.TimeoutExpired:
            self._stop()
        else:
            self._release()

    def _stopped_by(self, error: BaseException) -> BaseException:
        """`_stop`, on behalf of `error`, which the caller raises (and this returns): if the removal fails too, that
        failure is added to `error` as a note and never raised in its place: the run reports what went wrong first."""
        try:
            self._stop()
        except Exception as e:  # noqa: BLE001
            error.add_note(f"and then removing its container failed too: {e}")
        return error

    def _stop(self) -> None:
        """Remove this instance's own container by its own name, kill and reap its own docker CLI process, then remove
        the container again: a CLI killed in the middle of `create` leaves a Created container that the first removal
        (before the kill) could not have seen. Never closes the input first: a writer thread stuck on a child that
        reads nothing would hold `close` for good."""
        with self._life:  # a second caller waits for the first to finish, so "returned" means "gone"
            self._closed = True
            if self._stopped:
                return
            self._stopped = True
            try:
                sandbox.remove(self.name)
            finally:
                if self._proc.poll() is None:
                    self._proc.kill()
                self._proc.wait(timeout=30)
                self._release()
            sandbox.remove(self.name)
            deadline = time.monotonic() + REMOVE_WAIT
            while sandbox.exists(self.name):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"container {self.name} is still there {REMOVE_WAIT:g} s after `docker rm -f`")
                time.sleep(0.1)

    def _release(self) -> None:
        """Close this instance's pipe ends once the docker CLI has exited and its pipes have ended."""
        for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            with contextlib.suppress(OSError, ValueError):
                pipe.close()
