"""Running an external system without Docker: the agent as a local subprocess, speaking the shim's JSON lines.

`stbench run FILE.py:Class CASE_DIR --local` uses it (docs/agent.md). The agent runs as
`sys.executable -I -B <shim> FILE CLASS`, the same shim and the same JSON lines as in the sandbox (`shim.py`,
`system.CommandSystem`), so a system that runs here runs in Docker unchanged. There is no second route: no in-process
class, which would be a second contract and lose the shim's stdout discipline.

A local run is NOT sandboxed. The agent is a plain process with the user's permissions, network and files, and nothing
holds it to its working folder or its state directory: it is stamped `sandbox: none`, and its score is not a verified
one (DECISIONS, "Release 0.1"). Honest or buggy systems are the threat model; Docker stays the verified mode. Each case
still starts with an empty home, which the shim sets at `start` as in the sandbox (`shim.case_environment`); the run
stays unsandboxed and inherits the shell's other variables.
"""

from __future__ import annotations

import contextlib
import functools
import os
import signal
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

from .jsonpipe import JsonPipe
from .safecopy import remove_tree
from .stream_protocol import TIMEOUTS
from .system import CLOSE_GRACE, PipedSystem

KILL_WAIT = 30.0  # seconds a killed process has to be reaped
TASKKILL = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "taskkill.exe"


@functools.cache
def shim_path() -> Path:
    """The shim's file, found as package data of `stbench` (`importlib.resources`), never through the repository's
    layout: an installed wheel has it beside this module and no checkout around it."""
    path = Path(str(files(__package__).joinpath("shim.py")))
    if not path.is_file():
        raise RuntimeError(f"the system shim is not a file in the stbench package ({path}); is stbench installed from "
                           "a zip? Install it into a folder")
    return path


def check(file: str | os.PathLike, cls: str, dataset: str | os.PathLike | None = None) -> None:
    """ValueError unless FILE is an existing `.py`, `cls` is a class name, and FILE's folder neither contains nor sits
    inside `dataset` (the cheap part of what `sandbox.program` refuses: the system and the dataset stay apart)."""
    path = Path(file)
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"{file} is not an existing Python file (FILE.py)")
    if not cls.isidentifier():
        raise ValueError(f"{cls!r} is not a class name")
    if dataset is not None:
        folder, data = path.resolve().parent, Path(dataset).resolve()
        if folder == data or folder.is_relative_to(data) or data.is_relative_to(folder):
            raise ValueError(f"the system's folder {folder} must not contain or sit inside the dataset {data}: put "
                             "the system in a folder of its own")


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a running `proc` and what it started. On Windows the interpreter of a virtual environment is a launcher that
    runs the real one as its child, so killing the launcher alone would leave the agent running."""
    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run([str(TASKKILL), "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                           stdin=subprocess.DEVNULL, timeout=KILL_WAIT)
    else:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)  # it leads a session of its own (`start_new_session`)
    with contextlib.suppress(OSError):
        proc.kill()


class LocalCommandSystem(PipedSystem):
    """FILE.py:Class run as `sys.executable -I -B <shim> FILE CLASS` in a fresh temp folder, speaking the stream
    protocol through `JsonPipe` (the messages of `CommandSystem`, with host paths in `start`). The constructor starts
    the process and sends nothing: `start` sends the first message.

    Each message must be answered within its `timeouts` entry (`MIN_TIMEOUT` at least), or the process is killed and
    `SystemTimeout` raised; a system that dies, breaks the protocol or fails to start fails the run with the end of its
    stderr. `close()` ends the input and kills the process if it has not exited within `CLOSE_GRACE`. The constructor
    refuses a system folder that contains, or sits inside, `dataset`.
    """

    _gone = "its process was killed"

    def __init__(self, state_dir, workdir, file: Path, cls: str, timeouts: dict[str, float] = TIMEOUTS, *,
                 dataset: Path | None = None, **opts):
        super().__init__(state_dir, workdir, timeouts, **opts)
        self.file, self.cls = Path(file), cls
        check(self.file, cls, dataset)
        self._closed = self._stopped = False
        self._proc = self._cwd = None
        self._launch()

    @property
    def _who(self) -> str:
        return f"system {self.file.name}:{self.cls}"

    def _paths(self) -> tuple[str, str]:
        return str(self.workdir.resolve()), str(self.state_dir.resolve())

    def _launch(self) -> None:
        """Start the process in a temp folder of its own; its shim waits for the `start` message."""
        self._cwd = Path(tempfile.mkdtemp(prefix="stbench-local-"))
        argv = [sys.executable, "-I", "-B", str(shim_path()), str(self.file.resolve()), self.cls]
        try:
            self._proc = subprocess.Popen(argv, cwd=self._cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, start_new_session=os.name != "nt")
            self._pipe = JsonPipe(self._proc)
        except BaseException:
            self._stop()  # never leave a folder behind a process that did not start
            raise

    def _exit_note(self) -> str:
        with contextlib.suppress(subprocess.TimeoutExpired):
            return f" (code {self._proc.wait(1.0)})"
        return ""

    def close(self) -> None:
        """Idempotent. End its input; if it has not exited within `CLOSE_GRACE`, kill it. If the last request is still
        being written (the child has not read all of it), ending the input would wait for that write for good, so the
        process is killed at once."""
        if self._closed:
            return
        self._closed = True
        if self._proc is not None and not self._pipe.writing:
            with contextlib.suppress(OSError, ValueError):
                self._proc.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(CLOSE_GRACE)
        self._stop()

    def _stopped_by(self, error: BaseException) -> BaseException:
        """`_stop`, on behalf of `error`, which the caller raises (and this returns): a failure to stop is added to it
        as a note and never raised in its place."""
        try:
            self._stop()
        except Exception as e:  # noqa: BLE001
            error.add_note(f"and then stopping its process failed too: {e}")
        return error

    def _stop(self) -> None:
        """Kill the process (and what it started) if it is still running, reap it, and remove its working folder.
        Idempotent. Its stdout and stderr are left to end with it: a reader thread is still on them, and closing them
        here would wait for that thread."""
        self._closed = True
        if self._proc is not None and not self._stopped:
            self._stopped = True
            if self._proc.poll() is None:
                kill_tree(self._proc)
            self._proc.wait(timeout=KILL_WAIT)
            with contextlib.suppress(OSError, ValueError):
                self._proc.stdin.close()
        if self._cwd is not None:
            remove_tree(self._cwd)
            self._cwd = None
