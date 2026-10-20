"""The harness end of the pull: what the model's `next_frame` and `submit_answer` calls reach (`codex_mcp` relays them).

The reference agent pulls what the robot senses. The tool server is a child of Codex, so it cannot be called in process:
it relays each call over `multiprocessing.connection` to a `Feed`, which listens on `127.0.0.1` with a random key
(`server_config` is the address and the key, what the server's configuration holds; nothing else of the case reaches
it). The runner still pushes. `CodexSystem` gives the feed one `Item` at a time (a task, a frame, an ask or the end) and
waits for the model to pull again; the model's next `next_frame` means it is done with the item. That call stays open
until the runner supplies the next item, which, the frames being on disk, is at once (but for the frame `CodexSystem`
holds until it knows whether questions go with it).

The runner side: `serve_in_prompt` (the first task goes in the first prompt), `supply` and `wait` (hand an item to the
pending or the next pull, then wait until the model pulls again, the exec ends or the deadline passes), `begin_exec`
(a new exec starts: a pull left by the old one is dropped), `close_ask` and `close`. The model side is `call`, one tool
call at a time under a lock; it never raises for a request, a refusal is `error:` text.

An item is a task, a frame, an ask, the end, or a frame that comes with the questions about it (`Item.with_ask`: kind
`frame`, with `asked`, the text of the questions): one reply to one pull, the frame's text, its image, then the
questions' text, and the ask it opens is any ask's (below). The model looks and decides in one round trip, where a
separate ask item costs it one pull more for every question. Which items are joined is `CodexSystem`'s rule; the feed
serves what it is given.

- `next_frame` waits for the next item and returns it as MCP content (a text block and, for a frame on a camera
  channel, an image block, then the questions' text block of a frame that comes with them). While the ask's labels are
  unsubmitted the first call answers an error naming them (the ask's one nudge) and the second fails them `no_submit`
  and goes through (`codex_asks`). A second call while one is waiting answers `BUSY`. After the end item it answers
  the end again.
- `submit_answer` is the ask table (`codex_asks`) while an ask is pending, else `NO_QUESTION`.
- With `carry` (a memory arm whose SPEC says so) a `next_frame` may hold `memory`, a record for the memory. Once the
  pull is accepted (the run is not over, no other pull is open, no questions are unanswered) the pull is marked open,
  the record is handed to `carry` outside the lock, and only then is the arrival counted, so the runner, which learns
  of a pull by its arrival, never uses the memory's sense pipe while a record is being written to it. The reply
  (`carry`'s dict) is the last text block of the item the pull returns, `"<carry_name>: <compact JSON>"`; the item's
  own blocks are unchanged and in order. A pull refused (`BUSY`, a premature one, one of a closed feed) carries
  nothing. A record that is not carried is journaled as a `memory_dropped` event with a `reason`: `end` (the pull came
  after the end item: it is a call of the model's, answered with the end again, and its `call` event has no
  `memory_reply`) or `abandoned` (the pull's exec ended while its record was carried: it is no call of the model's, and
  the event has the reply, `memory_reply`, which nobody reads).
  The `call` event has the reply as `memory_reply`, and the `served` event of the item has its text.
- At most `max_calls` calls between two items (the cap does not stop `next_frame`: a model told to pull must be able
  to); past it a call answers `MAX_CALLS` and does nothing.

Every item served and every call is handed to `record` (the journal's `note`) with the monotonic and wall time, the
images as their hash and never base64.
"""

from __future__ import annotations

import base64
import contextlib
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener

from .agent_tools import NEXT_NAME, NO_QUESTION, SUBMIT_NAME
from .codex_asks import AskTable
from .codex_log import image_hash

BUSY = "error: another next_frame call is already waiting"
MAX_CALLS = "error: too many tool calls since the last result of next_frame; call next_frame"
CLOSED = "error: the run is over"
WAKE = 0.5  # seconds between a waiting thread's looks at the clock, whatever it was told
WAKE_WAIT = 2.0  # seconds `close` waits for the thread blocked in `accept` to take its last connection


@dataclass(frozen=True)
class Item:
    """One thing the model is shown. `kind` is `task`, `frame`, `ask` or `end`; `image` the frame's camera image (JPEG
    or PNG bytes); `k` a frame's number or an ask's step; `ask` an ask's `when`, `now_k` and `labels` (label -> qid),
    and `types` its labels' answer types. A frame that comes with questions (`with_ask`) also has `asked`, their text,
    which follows the frame's text and image; `ask` and `types` are then the questions'."""

    kind: str
    text: str
    image: bytes | None = None
    k: int | None = None
    ask: dict | None = None
    types: dict[str, str] = field(default_factory=dict)
    asked: str = ""

    def content(self) -> list[dict]:
        """The item as MCP content: its text, then its image when it has one, then the text of the questions that come
        with a frame."""
        blocks = [{"type": "text", "text": self.text}]
        if self.image is not None:
            mime = "image/png" if self.image.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
            blocks.append({"type": "image", "data": base64.b64encode(self.image).decode("ascii"), "mimeType": mime})
        if self.asked:
            blocks.append({"type": "text", "text": self.asked})
        return blocks

    def with_ask(self, ask: Item) -> Item:
        """This frame with the questions of `ask` after it, as one item: the frame's text, image and `k`, the ask's
        text as `asked`, and its `ask` and `types`. ValueError unless this is a frame and `ask` an ask at its step."""
        if self.kind != "frame" or ask.kind != "ask" or ask.k != self.k:
            raise ValueError(f"only an ask at a frame's own step joins it, not a {ask.kind} at k{ask.k} to a "
                             f"{self.kind} at k{self.k}")
        return Item("frame", self.text, self.image, k=self.k, ask=ask.ask, types=ask.types, asked=ask.text)


def _text(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def _reply(name: str, reply: dict) -> str:
    """A carried reply as the model reads it, the last text block of an item: the memory's name, then the reply."""
    return f"{name}: " + json.dumps(reply, ensure_ascii=False, separators=(",", ":"))


class Feed:
    """The pull of one case. Safe to call from any thread."""

    def __init__(self, record: Callable[[dict], None] = lambda event: None, *, max_calls: int = 100,
                 carry: Callable[[dict], dict] | None = None, carry_name: str = "memory"):
        self._record, self._max_calls = record, max_calls
        self._carry, self._carry_name = carry, carry_name
        self._key = secrets.token_bytes(32)
        self._listener = Listener(("127.0.0.1", 0), authkey=self._key)
        self._cond = threading.Condition()
        self._epoch = 0  # the exec the waiting pull belongs to; a new exec drops the old one's
        self._item: Item | None = None  # supplied by the runner, not yet taken by a pull
        self._waiting = False  # a next_frame is open
        self._arrivals = 0  # accepted next_frame calls
        self._taken = 0  # items served so far
        self._at = 0  # the arrivals when the last item was taken
        self._calls = 0  # tool calls since the last item was taken
        self._ask: AskTable | None = None  # the ask the model is answering
        self._asked: AskTable | None = None  # the ask whose pull came, until `close_ask`
        self._final: Item | None = None  # the end item, once served
        self._closed = False
        threading.Thread(target=self._accept, daemon=True, name="stbench-feed").start()

    def server_config(self) -> dict:
        """What the tool server's configuration holds: the address to connect to and the key, in hex."""
        host, port = self._listener.address
        return {"address": [host, port], "key": self._key.hex(), **({"carry": True} if self._carry else {})}

    # -- the connection -----------------------------------------------------------------------------------------
    def _accept(self) -> None:
        """Take each tool server's connection and serve it in a thread of its own. The loop ends on the connection that
        `close` makes after setting `_closed` (so it always accepts once more), or when the listener is closed."""
        while True:
            try:
                conn = self._listener.accept()
            except Exception:  # a client that failed the key or went away mid-handshake, or the listener closed
                if self._closed:
                    return
                time.sleep(0.05)
                continue
            if self._closed:
                conn.close()
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="stbench-feed-call").start()

    def _serve(self, conn) -> None:
        """One tool server's connection: a request a message, until it hangs up."""
        with contextlib.closing(conn):
            while True:
                try:
                    request = json.loads(conn.recv_bytes())
                except (EOFError, OSError):
                    return
                except ValueError:
                    request = None
                if isinstance(request, dict):
                    reply = self.call(request.get("tool"), request.get("arguments"))
                else:
                    reply = {"content": _text("error: not a tool call"), "isError": True}
                try:
                    conn.send_bytes(json.dumps(reply).encode("utf-8"))
                except OSError:
                    return

    def close(self) -> None:
        """Stop listening and let any open pull go. Idempotent."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        waker = threading.Thread(target=self._wake_accept, daemon=True, name="stbench-feed-close")
        waker.start()
        waker.join(WAKE_WAIT)
        with contextlib.suppress(OSError):
            self._listener.close()

    def _wake_accept(self) -> None:
        """Connect once, as a tool server would, so that the thread blocked in `accept` takes the connection, sees that
        the feed is closed and ends. If that thread is already gone nothing answers the handshake: the caller stops
        waiting, and closing the listener ends this one."""
        with contextlib.suppress(Exception):
            Client(self._listener.address, authkey=self._key).close()

    # -- the model's side ---------------------------------------------------------------------------------------
    def call(self, name, arguments) -> dict:
        """The reply to one tool call, `{"content": [blocks], "isError": bool}`, and the call is recorded. Blocks (only
        `next_frame` does) until the runner supplies an item; never raises for a request."""
        at, now = time.monotonic(), time.time()
        with self._cond:
            self._calls += 1
            capped = self._calls > self._max_calls and name != NEXT_NAME
        extra: dict = {}
        if capped:
            content, error = _text(MAX_CALLS), True
        elif name == NEXT_NAME:
            content, error, extra = self._pull(arguments)
        elif name == SUBMIT_NAME:
            with self._cond:
                text = NO_QUESTION if self._ask is None else self._ask.submit(arguments)
            content, error = _text(text), text.startswith("error:")
        else:
            content, error = _text(f"error: unknown tool {str(name)[:60]!r}"), True
        event = {"event": "call", "tool": name if isinstance(name, str) else str(name)[:60], "arguments": arguments,
                 "error": error, "at": round(at, 6), "time": round(now, 6), **extra}
        if "item" not in extra:  # an item is journaled as served
            event["result"] = "\n".join(b["text"] for b in content if b["type"] == "text")
        if not extra.get("abandoned"):  # a pull the run dropped (its exec is gone) is no call of the model's
            self._record(event)
        return {"content": content, "isError": error}

    def _pull(self, arguments=None) -> tuple[list[dict], bool, dict]:
        memory = arguments.get("memory") if isinstance(arguments, dict) and self._carry else None
        with self._cond:
            epoch = self._epoch
            if self._final is not None:
                if memory is not None:
                    self._record({"event": "memory_dropped", "reason": "end", "time": round(time.time(), 6)})
                return self._final.content(), False, {"over": True}
            if self._closed:
                return _text(CLOSED), True, {"abandoned": True}
            if self._waiting:
                return _text(BUSY), True, {}
            extra: dict = {}
            if self._ask is not None:
                before = self._ask.premature
                refusal = self._ask.pull()
                extra = {"premature": True} if self._ask.premature > before else {}
                if refusal is not None:
                    return _text(refusal), True, extra
                self._asked, self._ask = self._ask, None
            carried = None
            if memory is not None:  # the pull is open from now: another is BUSY while the record is written
                self._waiting = True
                self._cond.release()
                try:
                    carried = self._carried(memory)
                finally:
                    self._cond.acquire()
                extra = {**extra, "memory_reply": carried}
                if self._epoch != epoch or self._closed:  # the exec is gone, though the record went
                    self._record({"event": "memory_dropped", "reason": "abandoned", "memory_reply": carried,
                                  "time": round(time.time(), 6)})
                    return _text(CLOSED), True, {"abandoned": True}
            self._arrivals += 1
            self._waiting = True
            self._cond.notify_all()
            while self._item is None and not self._closed and self._epoch == epoch:
                self._cond.wait()
            if self._epoch != epoch or self._closed:
                return _text(CLOSED), True, {"abandoned": True}
            item, self._item = self._item, None
            self._waiting = False
            reply = None if carried is None else _reply(self._carry_name, carried)
            blocks = item.content() + ([] if reply is None else _text(reply))
            return blocks, False, {**extra, "item": self._take(item, memory_reply=reply)}

    def _carried(self, memory) -> dict:
        """The memory's reply to a record: `carry`'s, or an error object when it raised (a pull never raises)."""
        try:
            return self._carry(memory)
        except Exception as error:  # noqa: BLE001  (a request never stops the feed: the model reads the error)
            return {"ok": False, "error": type(error).__name__, "message": str(error)[:200]}

    def _take(self, item: Item, via: str | None = None, memory_reply: str | None = None) -> int:
        """The model has the item: number it, start counting its calls, open its ask, and record it as served."""
        number = self._taken
        self._taken += 1
        self._at, self._calls, self._asked = self._arrivals, 0, None
        self._ask = AskTable(item.types) if item.kind == "ask" or item.asked else None
        if item.kind == "end":
            self._final = item
        event = {"event": "served", "item": number, "kind": item.kind, "k": item.k, "text": item.text,
                 "images": [] if item.image is None else [image_hash(item.image)],
                 "at": round(time.monotonic(), 6), "time": round(time.time(), 6)}
        self._record({**event, **({"ask": item.ask} if item.ask else {}),
                      **({"asked": item.asked} if item.asked else {}), **({"via": via} if via else {}),
                      **({"memory_reply": memory_reply} if memory_reply else {})})
        return number

    # -- the runner's side --------------------------------------------------------------------------------------
    def serve_in_prompt(self, item: Item) -> int:
        """The item goes to the model in its first prompt, not in a reply: count it as served. Its number, the ticket
        for `wait`."""
        with self._cond:
            return self._take(item, via="prompt")

    def supply(self, item: Item) -> int:
        """Hand the next item to the pending or the next `next_frame`. Its number, the ticket for `wait`. RuntimeError
        when the last item has not been taken yet."""
        with self._cond:
            if self._item is not None:
                raise RuntimeError("the last item supplied has not been taken by a next_frame yet")
            self._item = item
            self._cond.notify_all()
            return self._taken

    def wait(self, ticket: int, deadline: float, alive: Callable[[], bool], *, pull: bool = True) -> str:
        """Wait until the item of `ticket` was taken and, with `pull`, the model pulled again (it is done with the
        item): `pulled`; or the exec is over (`alive()` false): `exit`; or `time.monotonic()` passed `deadline`:
        `timeout`. A pull wins over an exit that came with it. Whoever ends the exec calls `wake`."""
        with self._cond:
            while True:
                if self._taken > ticket and (not pull or self._arrivals > self._at):
                    return "pulled"
                if not alive():
                    return "exit"
                left = deadline - time.monotonic()
                if left <= 0:
                    return "timeout"
                self._cond.wait(min(left, WAKE))

    def wake(self) -> None:
        """Something the waiting runner watches changed (an exec ended)."""
        with self._cond:
            self._cond.notify_all()

    def begin_exec(self) -> None:
        """A new exec is about to start: a `next_frame` the old one left open is dropped, and the item waits for the new
        one's pull."""
        with self._cond:
            self._epoch += 1
            self._waiting = False
            self._cond.notify_all()

    def close_ask(self) -> AskTable | None:
        """The ask just answered (or left unfinished), with its table, and no ask is pending after it."""
        with self._cond:
            table = self._asked or self._ask
            self._asked = self._ask = None
            return table
