"""Helpers for the tests of a memory beside the Claude Code agent (`test_claude_memory`, `test_claude_carry`): the
synthetic case run through `stbench run claude:sonnet` over the fake command (`claude_cli_fake`) with or without a
memory, and what the run left.

`play` runs the case (six frames, a during ask with frame 3 and the end ask) over a fake with a `plan` (the per-turn
changes of `claude_cli_fake`), with the memory `held` (`codex_kit.memory`; none: the reference agent), and gives back a
namespace of the `fake`, the run folder `out`, the fake's `processes`, the parsed `journal`, the `case` and the
Claude summary. `sent` is everything the model was sent, `made` the calls a server of the fake made, `sensed` the lines
the fake sensor feed wrote in the kept store.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from claude_kit import Fake, claude_run, pulled
from test_cmd_codex import jsonl, small_case

PICTURE = b"\xff\xd8\xff\xe0" + b"memory-picture" + b"\xff\xd9"   # what the fake memory server's `snapshot` shows back
CLAIM = {"server": "mem", "tool": "claim", "arguments": {"text": "frame {k}"}}
VIEW = {"server": "mem", "tool": "view", "arguments": {}}
SNAPSHOT = {"server": "mem", "tool": "snapshot", "arguments": {}}
PLAN = {"memory": [CLAIM, VIEW, SNAPSHOT], "memory_ask": [VIEW]}


def play(tmp_path: Path, monkeypatch, plan: dict | None, held=None, *extra, status: int = 0,
         case=None) -> SimpleNamespace:
    """`stbench run claude:sonnet` on the synthetic case (six frames, a during ask with frame 3, the end ask; `case`, or
    a new one) over a fake with `plan`, with the memory `held` (none: the reference agent), and what it left."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    case, out = case or small_case(tmp_path / "data"), tmp_path / "out"
    fake = Fake(tmp_path, plan, guide=held.guide if held else None)
    given = ["--memory", held.path] if held else []
    assert claude_run(monkeypatch, fake, "claude:sonnet", case.private, "--out", out, *given, *extra) == status
    journal = jsonl(out / "claude.calls.jsonl") if (out / "claude.calls.jsonl").exists() else []
    summary = out / "claude.summary.json"
    return SimpleNamespace(fake=fake, out=out, processes=fake.processes(), journal=journal, case=case,
                           summary=json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None)


def sent(done: SimpleNamespace) -> dict:
    """Everything the model was sent, as the fake and the journal saw it, without the clocks."""
    return {"messages": [p["messages"] for p in done.processes],
            "pulls": [[(c["text"], c["blocks"], c["images"]) for c in pulled(p)] for p in done.processes],
            "journal": [{k: v for k, v in e.items() if k not in ("at", "time")} for e in done.journal
                        if e.get("event") in ("exec_started", "served", "answer_outcome")]}


def made(done: SimpleNamespace, server: str = "mem") -> list[tuple[str, dict]]:
    return [(c["tool"], c["arguments"]) for p in done.processes for c in p["calls"] if c["server"] == server]


def sensed(done: SimpleNamespace, name: str = "mem") -> list[dict]:
    """The lines the fake sensor feed wrote in the case's kept store, in order."""
    path = done.out / f"{name}-store" / "sense.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
