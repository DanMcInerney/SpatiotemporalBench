"""One long `codex exec`, run and read as it goes (`codex_system`).

`Live` starts an exec (`Codex.argv`, so the lockdown has one owner) with the prompt on stdin, and reads its output line
by line in a thread. Every event is checked as it arrives (`codex_exec.check_event`): a tool outside the server's kills
the exec at once, with everything it started (`local_system.kill_tree`), and the exec's `result` raises `ToolEscaped`,
not at its end. Nothing bounds the exec's length but the case: the runner watches `alive`, waits with `wait` and kills
with `kill` when a message's deadline passes (`codex_feed`). `wake` is called when the exec ends, so a waiting runner
looks again.

The tool server it is given is the thin relay (`codex_mcp`): its configuration file holds the feed's address and key
(`server`), nothing else, and lives with the exec's other temp files in a folder `close` removes. An exec that is over
has a `result`, which raises the `CodexError` (or `ContextTooLong`) it ended in, the way `codex_exec.failure` names it,
or is the `Result` of the turn; one the harness killed is a `Result` with `killed` and no usage (a killed turn prints
none, so its tokens are missing and a count that includes it is a lower bound).
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .codex_exec import Codex, CodexError, Result, check_event, digest, failure, json_line
from .codex_log import scrubbed
from .local_system import kill_tree
from .safecopy import remove_tree

KILL_WAIT = 30.0  # seconds for a killed exec to be reaped


class Live:
    """A running `codex exec`. `kind` is `start` or `resume` (of `thread`), `prompt` goes on stdin, `server` is the tool
    server's configuration (`Feed.server_config`), `wake` is called when the exec is over, `attempt` is 2 for the retry
    of a transient failure. Raises CodexMissing/CodexError when the command cannot be started."""

    def __init__(self, codex: Codex, *, kind: str, thread: str | None, prompt: str, server: dict,
                 wake: Callable[[], None] | None = None, attempt: int = 1):
        codex.prepare()
        self.kind, self.attempt, self.events = kind, attempt, []
        self.escaped: CodexError | None = None
        self.killed = False
        self.code: int | None = None
        self._wake, self._done = wake, threading.Event()
        self._work = Path(tempfile.mkdtemp(prefix="stbench-codex-"))
        self._err = None
        try:
            config, empty, cwd = self._work / "server.json", self._work / "empty", self._work / "in"
            config.write_text(json.dumps(server), encoding="utf-8", newline="\n")
            empty.mkdir()
            cwd.mkdir()
            self.argv = codex.argv(kind, thread, config, empty)
            self._err = (self._work / "stderr.txt").open("wb")
            self.proc = codex.popen(self.argv, cwd, self._err)
        except BaseException:
            if self._err is not None:
                self._err.close()
            remove_tree(self._work)
            raise
        self.began = time.monotonic()
        self.ended: float | None = None
        threading.Thread(target=self._write, args=(prompt,), daemon=True, name="stbench-codex-stdin").start()
        self._pump = threading.Thread(target=self._read, daemon=True, name="stbench-codex-live")
        self._pump.start()

    def _write(self, prompt: str) -> None:
        """The prompt, then the end of stdin. A thread of its own: a pipe holds little, and the reader must not wait."""
        with contextlib.suppress(OSError):  # an exec that died before reading its prompt is told by its exit
            self.proc.stdin.write(prompt.encode("utf-8"))
            self.proc.stdin.close()

    def _read(self) -> None:
        try:
            for raw in self.proc.stdout:
                event = json_line(raw.decode("utf-8", "replace"))
                if not isinstance(event, dict):
                    continue
                self.events.append(scrubbed(event))  # an image block as its hash: a long case holds no frames in memory
                try:
                    check_event(event)
                except CodexError as error:
                    self.escaped = error
                    kill_tree(self.proc)
                    break
            self.proc.wait()
        finally:
            self.code, self.ended = self.proc.returncode, time.monotonic()
            with contextlib.suppress(OSError):
                self._err.close()
            self._done.set()
            if self._wake is not None:
                self._wake()

    # -- while it runs ------------------------------------------------------------------------------------------
    def alive(self) -> bool:
        return not self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the exec to end; True when it has."""
        return self._done.wait(timeout)

    def kill(self) -> None:
        """End the exec now, with everything it started. A no-op once it has ended."""
        if not self.alive():
            return
        self.killed = True
        kill_tree(self.proc)
        self._done.wait(KILL_WAIT)

    @property
    def thread(self) -> str:
        """The thread's id, once `thread.started` came (else "")."""
        return digest(self.events)["thread"]

    # -- when it is over ----------------------------------------------------------------------------------------
    def result(self) -> Result:
        """The `Result` of an exec that has ended (call `wait` first). Raises the error it ended in: `ToolEscaped` for a
        tool outside the server's, else the class `codex_exec.failure` names; a killed exec is a `Result` with
        `killed`."""
        self._pump.join(KILL_WAIT)
        if self.escaped is not None:
            raise self.escaped
        read = digest(self.events)
        err = self._stderr()
        if not self.killed:
            if (found := failure(self.code if self.code is not None else -1, self.events, err)) is not None:
                raise found[0](found[1])
            if not read["thread"]:
                raise CodexError("codex exec started no thread")
        return Result(read["thread"], read["messages"], read["usage"], self.events, self.argv,
                      round((self.ended or time.monotonic()) - self.began, 3), self.attempt, read["warnings"], err,
                      completed=read["completed"], killed=self.killed)

    def _stderr(self) -> str:
        """The end of what the exec wrote to stderr (a long exec may write a good deal; only its tail is read)."""
        try:
            with (self._work / "stderr.txt").open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                return f.read().decode("utf-8", "replace").strip()[-2000:]
        except OSError:
            return ""

    def close(self) -> None:
        """Kill the exec if it is running, and remove its temp files."""
        self.kill()
        self._pump.join(KILL_WAIT)
        remove_tree(self._work)
