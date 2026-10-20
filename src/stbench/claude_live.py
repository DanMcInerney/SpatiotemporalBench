"""One long `claude -p` process, run and read as it goes (`claude_system`): the Claude Code twin of `codex_live`.

`ClaudeLive` starts the process (`Claude.argv`, so the lockdown has one owner) with the first user message on stdin and
reads its stream-json output line by line in a thread. Where a Codex exec is one turn, this process holds the whole
case: it stays up after a turn, waiting on stdin, and `send` writes the next user message to it (the fixed resume line,
when the model ended its turn early). A line that is not a JSON object (the notice an unrecognised model id prints, for
one) is kept as a warning of the turn, never an error. Every event is checked as it arrives (`claude_exec.check_event`):
a tool outside the server's kills the process at once, with everything it started (`local_system.kill_tree`), and the
turn's `result` raises `ToolEscaped`, not at its end. Nothing bounds a turn's length but the case: the runner watches
`alive`, waits with `wait` and kills with `kill` when a message's deadline passes (`codex_feed`). `wake` is called
when the turn ends, so a waiting runner looks again.

`alive` means the turn is running: it is false once the turn's `result` came or the process ended. A turn that is over
has a `result`, which raises the `CodexError` (or `ContextTooLong`) it ended in, the way `claude_exec.failure` names it,
or is the `Result` of the turn; one the harness killed is a `Result` with `killed` and no usage (a killed turn prints
none, so its tokens are missing and a count that includes it is a lower bound). The tool server is the thin relay
(`codex_mcp`), configured by a file holding the feed's address and key (`server`) and named in the `--mcp-config` file;
both live with the process's other temp files in a folder `close` removes.

With a memory (`codex_memory`) the process has that memory's server beside the tool server: its calls pass the check,
and each one that completes is handed to `note` as a `memory_call` journal event (`claude_exec.memory_results`), paired
from the `tool_use` of an `assistant` event and the `tool_result` of the `user` event that answers it, since the feed
does not see them.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .claude_exec import Claude, check_event, digest, failure, memory_results, memory_uses
from .codex_exec import CodexError, Result, json_line
from .codex_log import scrubbed
from .local_system import kill_tree
from .safecopy import remove_tree

KILL_WAIT = 30.0  # seconds for a killed process to be reaped
NOISE_LINES = 20  # the most non-JSON lines of a turn that are kept


class ClaudeLive:
    """A running `claude -p`. `prompt` is the first user message, `system` the system prompt (`--system-prompt`),
    `server` the tool server's configuration (`Feed.server_config`), `wake` is called when a turn is over. `memory` is
    a memory's MCP server, `(name, argv)` (`Claude.argv`; None, the reference agent, has none) and `note` receives the
    journal event of each of its calls that completes. Raises CodexMissing/CodexError when the command cannot be
    started. `kind` is the turn's (`start`, then `resume`); `attempt` is always 1 (a failure is never retried: Claude
    Code retries a transient one itself)."""

    attempt = 1

    def __init__(self, claude: Claude, *, system: str, prompt: str, server: dict, wake=None,
                 memory: tuple[str, list[str]] | None = None, note: Callable[[dict], None] | None = None):
        claude.prepare()
        self.kind, self.events, self.noise = "start", [], []
        self.memory, self._note = memory[0] if memory else None, note
        self._calls: dict[str, tuple[str, object]] = {}   # the memory's calls started and not yet answered
        self.escaped: CodexError | None = None
        self.killed = False
        self.code: int | None = None
        self._wake, self._lock = wake, threading.Lock()
        self._over, self._exited = threading.Event(), threading.Event()
        self._work = Path(tempfile.mkdtemp(prefix="stbench-claude-"))
        self._err = None
        try:
            config, mcp, cwd = self._work / "server.json", self._work / "mcp.json", self._work / "in"
            config.write_text(json.dumps(server), encoding="utf-8", newline="\n")
            mcp.write_text(json.dumps(claude.mcp_config(config, memory)), encoding="utf-8", newline="\n")
            cwd.mkdir()
            self.argv = claude.argv(mcp, system, memory=memory)
            self._err = (self._work / "stderr.txt").open("wb")
            self.proc = claude.popen(self.argv, cwd, self._err)
        except BaseException:
            if self._err is not None:
                self._err.close()
            remove_tree(self._work)
            raise
        self.began, self.ended = time.monotonic(), None
        self._pump = threading.Thread(target=self._read, daemon=True, name="stbench-claude-live")
        self._pump.start()
        try:
            self._write(prompt)
        except CodexError:
            self.close()
            raise

    def _write(self, text: str) -> None:
        """One user message on stdin: a line of stream-json. CodexError when the process is gone."""
        line = json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
        try:
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
        except OSError:
            raise CodexError("the claude process is gone: it cannot be sent the next message") from None

    def _read(self) -> None:
        try:
            for raw in self.proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                event = json_line(line)
                if not isinstance(event, dict):
                    with self._lock:
                        if line and len(self.noise) < NOISE_LINES:
                            self.noise.append(line[:300])
                    continue
                with self._lock:
                    self.events.append(scrubbed(event))   # an image block as its hash: a long case holds no frames
                try:
                    check_event(event, self.memory)
                except CodexError as error:
                    self.escaped = error
                    kill_tree(self.proc)
                    break
                if self._note is not None:
                    self._journal(self.events[-1])
                if event.get("type") == "result":
                    self._end_turn()
            self.proc.wait()
        finally:
            self.code = self.proc.returncode
            with contextlib.suppress(OSError):
                self._err.close()
            self._end_turn()
            self._exited.set()

    def _journal(self, event: dict) -> None:
        """The memory's calls `event` starts are remembered, and those it answers are noted, as they arrive."""
        self._calls.update(memory_uses(event, self.memory))
        for entry in memory_results(event, self.memory, self._calls):
            self._note(entry)

    def _end_turn(self) -> None:
        self.ended = self.ended or time.monotonic()
        self._over.set()
        if self._wake is not None:
            self._wake()

    # -- while a turn runs --------------------------------------------------------------------------------------
    def alive(self) -> bool:
        """True while the turn is running."""
        return not self._over.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the turn to end; True when it has."""
        return self._over.wait(timeout)

    def send(self, text: str) -> None:
        """Begin the next turn: write `text` as a user message to the process that finished the last one. CodexError
        when the process is gone."""
        if self._exited.is_set():
            raise CodexError("the claude process is gone: it cannot be sent the next message")
        with self._lock:
            self.kind, self.events, self.noise = "resume", [], []
            self.began, self.ended = time.monotonic(), None
            self._over.clear()
        self._write(text)

    def kill(self) -> None:
        """End the process now, with everything it started (a turn still running is then `killed`). A no-op once it has
        ended."""
        if self._exited.is_set():
            return
        self.killed = self.killed or self.alive()
        kill_tree(self.proc)
        self._exited.wait(KILL_WAIT)

    # -- when a turn is over ------------------------------------------------------------------------------------
    def result(self) -> Result:
        """The `Result` of the turn that is over (call `wait` first). Raises the error it ended in: `ToolEscaped` for a
        tool outside the server's, else the class `claude_exec.failure` names; a killed turn is a `Result` with
        `killed`."""
        if self.killed or self.escaped is not None:
            self._exited.wait(KILL_WAIT)
        if self.escaped is not None:
            raise self.escaped
        with self._lock:
            events, noise = list(self.events), list(self.noise)
        read = digest(events, noise, self.memory)
        err = self._stderr()
        if not self.killed:
            if (found := failure(events, self.code, err)) is not None:
                raise found[0](found[1])
            if not read["thread"]:
                raise CodexError("claude started no session")
        return Result(read["thread"], read["messages"], read["usage"], events, self.argv,
                      round((self.ended or time.monotonic()) - self.began, 3), 1, read["warnings"], err,
                      completed=read["completed"], killed=self.killed)

    def _stderr(self) -> str:
        """The end of what the process wrote to stderr (only its tail is read)."""
        try:
            with (self._work / "stderr.txt").open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                return f.read().decode("utf-8", "replace").strip()[-2000:]
        except OSError:
            return ""

    def close(self) -> None:
        """Kill the process if it is running, and remove its temp files."""
        self.kill()
        self._pump.join(KILL_WAIT)
        with contextlib.suppress(OSError):
            self.proc.stdin.close()
        remove_tree(self._work)
