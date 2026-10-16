"""JSON lines to and from a child process: one reply per request, each within a deadline (`CommandSystem`'s transport).

A writer thread per request and one reader thread for stdout, so neither a child that never reads nor one that never
answers can block the harness past the deadline. The end of stderr is kept, bounded, for error messages. A request that
fails (timeout, exit, bad line) breaks the pipe for good: the harness then stops the child, it does not retry.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time

MAX_LINE = 4 * 2**20  # bytes in one reply line, its newline not counted; a longer line is a protocol error
STDERR_KEEP = 16384  # bytes of the child's stderr kept
STDERR_GRACE = 1.0  # seconds a finished child's stderr gets to drain before an error is raised


class PipeError(RuntimeError):
    """The child broke the protocol: a reply too long, not JSON, or not an object."""


class PipeClosed(PipeError):
    """The child's stdout ended (it exited) before it replied."""


class PipeTimeout(PipeError):
    """No reply within the deadline."""


class JsonPipe:
    def __init__(self, proc: subprocess.Popen) -> None:
        """`proc` was started with binary stdin, stdout and stderr pipes. Starts the stdout and stderr reader threads
        (daemon: they end when the child's pipes do, so the caller must stop the child)."""
        self._proc = proc
        self._lines: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()  # stdout lines; None is the end
        self._broken = False
        self._writer: threading.Thread | None = None  # the writer thread of the latest request
        self._stderr = bytearray()
        self._lock = threading.Lock()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def request(self, message: dict, timeout: float) -> dict:
        """Write `message` as one line (in a daemon writer thread) and return the next reply line as a dict.
        Raises PipeTimeout after `timeout` seconds counted from the call (writing included), PipeClosed at the end of
        the child's stdout, PipeError for a bad line. After any of these the pipe is broken and every later request
        raises PipeClosed."""
        if self._broken:
            raise PipeClosed("an earlier request failed, so the pipe is closed")
        data = json.dumps(message).encode("ascii") + b"\n"
        deadline = time.monotonic() + timeout
        self._writer = threading.Thread(target=self._write, args=(data,), daemon=True)
        self._writer.start()
        try:
            return self._reply(deadline, timeout)
        except PipeError:
            self._broken = True
            raise

    @property
    def writing(self) -> bool:
        """The latest request's writer thread is still writing: the child has not read all of it, so closing the child's
        input would wait for that thread (it holds the pipe's lock). The caller stops the child instead."""
        return self._writer is not None and self._writer.is_alive()

    def stderr_tail(self, n: int = 2000) -> str:
        """The last `n` characters of what the child wrote to stderr (UTF-8, errors replaced)."""
        with self._lock:
            kept = bytes(self._stderr)
        text = kept.decode("utf-8", errors="replace")
        return text[-n:] if n > 0 else ""

    def _reply(self, deadline: float, timeout: float) -> dict:
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
                break
            except queue.Empty:
                if time.monotonic() >= deadline:  # a wait can end a tick early; the deadline is the clock's
                    raise PipeTimeout(f"no reply within {timeout:g} s") from None
        complete = line is not None and line.endswith(b"\n")
        if line is not None and len(line) - complete > MAX_LINE:
            raise PipeError(f"a reply line is longer than {MAX_LINE} bytes")
        if not complete:  # the end of stdout, or a line the child died in the middle of
            self._stderr_thread.join(STDERR_GRACE)  # its last words may still be on their way
            raise PipeClosed("the child's output ended before it replied")
        try:
            reply = json.loads(line)
        except (ValueError, RecursionError) as e:
            raise PipeError(f"a reply is not JSON: {type(e).__name__}") from None
        if not isinstance(reply, dict):
            raise PipeError(f"a reply is not a JSON object but a {type(reply).__name__}")
        return reply

    def _write(self, data: bytes) -> None:
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except (OSError, ValueError):
            pass  # the child closed its input: it has exited (stdout ends, PipeClosed) or the deadline will fire

    def _read_stdout(self) -> None:
        try:
            while True:
                line = self._proc.stdout.readline(MAX_LINE + 1)  # bounded: a line longer than MAX_LINE ends here
                self._lines.put(line or None)
                if not line.endswith(b"\n"):
                    return
        except (OSError, ValueError):
            self._lines.put(None)

    def _read_stderr(self) -> None:
        try:
            while chunk := self._proc.stderr.read1(4096):
                with self._lock:
                    self._stderr += chunk
                    del self._stderr[:-STDERR_KEEP]
        except (OSError, ValueError):
            pass
