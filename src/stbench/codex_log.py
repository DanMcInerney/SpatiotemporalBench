"""The log of a Codex run: the events of a case, one record per `codex exec`, and the transcripts the leak scan reads.

`Journal` appends to a JSON-lines file with two kinds of line. An event has an `event` key (`note` writes it):

- `run`: the run's settings (`cmd_codex`);
- `exec_started`: `kind` (`start` or `resume`), `base` (the thread it continues, null for a start), `prompt` (the text
  the exec was sent: for a start the system prompt and the first task, for a resume the fixed line) and `attempt` (2
  for the one retry of a transient failure);
- `served`: one item the model was shown, in order: `item`, `kind` (`task`, `frame`, `ask` or `end`), `k` (a frame's, or
  an ask's step), `text`, `images` (each as `<jpeg N bytes sha256 HEX16>`, or `<png ...>`, never base64), `ask` (an
  ask's `when`, `now_k` and `labels`, the qid each of its labels stands for), `via` (`prompt` for the first task, which
  went in the first prompt, else absent), `at` (the harness's monotonic clock) and `time` (the wall clock). A frame
  that came with questions is one `frame` item that also has `ask` (the questions' `when`, `now_k` and `labels`) and
  `asked`, their text: `text` and `images` are the frame's, and the message the model held was the frame's text, its
  image, then `asked`;
- `call`: one tool call the model made: `tool`, `arguments`, `result` (the reply's text, the images as their hash;
  absent for a `next_frame` that was served an item, whose `item` is its number), `error`, `premature` (a `next_frame`
  made while questions were unanswered), `at` and `time` (when it arrived);
- `answer_outcome` and `exec_failed` (`codex_system`).

An exec record has no `event` key; it is written when an exec ends, whether it completed or was killed by the harness:
`i`, `t`, `session`, `kind`, `base`, `thread` (the thread it ended in), `messages` (the model's replies), `events`
(Codex's JSON lines as printed, any image block as its hash), `argv`, `stderr`, `warnings`, `attempts`, `usage` (what
this exec used: prompt, completion, cached, cache write and reasoning tokens, and a cost of 0), `thread_usage` (the
running total of its thread as Codex printed it), `cost` (0.0 for Codex: the plan, not dollars; for Claude Code the
API-equivalent dollars the exec added, from the CLI's own running `total_cost_usd`), `provider` (`codex` or `claude`),
`latency_s` and `killed`. A turn that was killed prints no usage, so its tokens are missing.

Codex's `turn.completed` usage is the thread's running total, not the turn's (measured: a start 2,256 tokens, its
resume 4,696). `Journal` keeps the total of each thread and writes the difference as `usage`, so that summing the log
counts each token once. The log holds question text, so it never goes under an episode folder or to a system.

A case is one thread, which the model holds as one conversation: `transcripts` puts it together from the events in the
order they were written (the first prompt, each item served as a tool message, the replies to its other calls, the
resume lines and, at the end of an exec, its own words) and gives each ask the conversation as the model held it then,
in the shape `llm_scan` reads. Codex's own compaction may shorten what the model holds; the transcript keeps everything
it was ever shown, so a leak the model has since forgotten is still found. The journal has several writers (the
runner's thread and the feed's), so a line is written under a lock.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from pathlib import Path


def image_hash(data: bytes) -> str:
    """How a log writes an image: its type (`png` by its signature; else `jpeg`), size and the
    first 16 hex characters of its SHA-256."""
    kind = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpeg"
    return f"<{kind} {len(data)} bytes sha256 {hashlib.sha256(data).hexdigest()[:16]}>"


def _bytes(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except ValueError:
        return data.encode("utf-8")


def scrubbed(value):
    """`value` (a Codex or Claude Code event, a reply) with every image written as its hash: an MCP image block
    `{"type": "image", "data": BASE64, ...}` keeps its other keys and gets `hash` in place of `data`, the same as the
    API's `{"type": "image", "source": {"type": "base64", "data": BASE64, ...}}` that Claude Code prints, and a
    `data:image/...;base64,` string is its hash. Nothing else changes."""
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            return {**{k: v for k, v in value.items() if k != "data"}, "hash": image_hash(_bytes(value["data"]))}
        source = value.get("source")
        if value.get("type") == "image" and isinstance(source, dict) and isinstance(source.get("data"), str):
            return {**{k: v for k, v in value.items() if k != "source"},
                    "source": {**{k: v for k, v in source.items() if k != "data"},
                               "hash": image_hash(_bytes(source["data"]))}}
        return {k: scrubbed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrubbed(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        return image_hash(_bytes(value.split(";base64,", 1)[1]))
    return value


def read_log(path: Path) -> list[dict]:
    """Every record of a call log, in order. A line that is not a JSON object (a torn write) raises ValueError."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                record = None
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {n} is not a JSON object (a torn write? delete the line)")
            records.append(record)
    return records


class Journal:
    """The append-only log of one run. `spent` counts what cost the plan something: a tool call of the model and a
    finished exec; a journal with none is the record of nothing, and `cmd_codex` removes it. `provider` is the CLI
    whose execs it records (`codex`, or `claude` for `claude_system`'s turns), written on each exec record."""

    def __init__(self, path: Path, provider: str = "codex"):
        self.path, self.spent, self.execs, self.provider = Path(path), 0, 0, provider
        self.totals: dict[str, dict] = {}  # thread -> its running usage as of its last completed exec
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict, spend: bool = False) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record) + "\n")
            self.spent += spend

    def note(self, event: dict) -> None:
        if "event" not in event:
            raise ValueError("a log event needs an 'event' key")
        self._append(event, event["event"] == "call")

    def exec(self, *, session: str, kind: str, base: str | None, result) -> dict:
        """Append the record of an ended exec (`result` is a `codex_live.Live`'s `codex_exec.Result`, or a turn of
        `claude_live.ClaudeLive`) and return this exec's own usage, the difference from its thread's last total (the
        cost with the tokens: 0 for Codex, whose plan is not dollars; the API-equivalent dollars for Claude Code)."""
        before = {} if kind == "start" else self.totals.get(base, {})
        used = {k: max(0, v - before.get(k, 0)) for k, v in result.usage.items()}
        if result.completed:
            self.totals[result.thread] = result.usage
        record = {"i": self.execs, "t": round(time.time(), 3), "session": session, "kind": kind, "base": base,
                  "thread": result.thread, "messages": result.messages, "events": scrubbed(result.events),
                  "argv": result.argv, "stderr": result.stderr, "warnings": result.warnings,
                  "attempts": result.attempts, "usage": used, "thread_usage": result.usage,
                  "cost": round(float(used.get("cost", 0.0)), 6), "provider": self.provider,
                  "latency_s": result.latency_s, "killed": result.killed}
        self._append(record, spend=True)
        self.execs += 1
        return used


def _tool(content) -> dict:
    return {"role": "tool", "tool_call_id": "", "content": content}


def transcripts(records: list[dict]) -> list[dict]:
    """The journal as `llm_scan` reads calls: `{i, purpose, ask?, request: {messages}}`, one for each ask served (`i`
    its item number, `ask` its `when`, `now_k` and `labels`, `messages` the conversation up to and including it) and a
    last one for the whole case (`purpose` `case`, no `ask`). An ask served with its frame is one message, as the model
    held it: the frame's text, its images, then the questions' text; its transcript is made at that item, so the
    questions' text is first in the conversation there and no frame after its step can be in it. See the module
    docstring. Events other than the ones that make the conversation are skipped; an exec started again (a retry)
    adds nothing."""
    held: list[dict] = []
    found: list[dict] = []
    items = 0
    for r in records:
        event = r.get("event")
        if event == "exec_started" and r.get("attempt", 1) == 1:
            held.append({"role": "user", "content": [{"type": "text", "text": r["prompt"]}]})
        elif event == "served":
            items = r["item"] + 1
            if r.get("via") != "prompt":  # the first task is in the first prompt
                images = [{"type": "image_url", "image_url": {"url": h, "detail": "high"}} for h in r["images"]]
                asked = [{"type": "text", "text": r["asked"]}] if r.get("asked") else []
                held.append(_tool([{"type": "text", "text": r["text"]}, *images, *asked]))
            if r.get("ask"):  # an ask alone, or the questions that came with a frame
                found.append({"i": r["item"], "purpose": "ask", "ask": r["ask"],
                              "request": {"messages": list(held)}})
        elif event == "call" and "result" in r:
            held.append(_tool(str(r["result"])))
        elif event is None:  # an exec record: the model's own words close it
            held += [{"role": "assistant", "content": m} for m in r["messages"]]
    return [*found, {"i": items, "purpose": "case", "request": {"messages": held}}]
