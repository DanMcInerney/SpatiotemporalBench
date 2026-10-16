"""The transport log of a case, for `stbench view`.

An agent system may write its own journal beside a case's `answers.jsonl`: one JSON record a line in
`<name>.calls.jsonl` (the reference Codex agent's `codex.calls.jsonl`, `docs/codex.md`) and a `<name>.summary.json` of
its counts. The viewer knows no system: it takes the first such journal, keeps each record as written and lets the page
show the fields it knows how to show (`i`, `purpose`, `kind`, `event`, `prompt`, `tools`, `messages`, `usage`,
`warnings`, `stderr`, ...) and the rest as JSON. Two bulky fields are left out, the raw `events` (a second copy of
`tools` and `messages`; those that report an error stay, as `event_errors`) and `argv` (the command line, which names
the machine). A string over `MAX_CHARS`, such as a long tool result, is cut and says how much; the page counts them.
A system with no journal has no log.
"""

from __future__ import annotations

import json
from pathlib import Path

from .view_files import json_object, jsonl

LOG_SUFFIX, SUMMARY_SUFFIX = ".calls.jsonl", ".summary.json"
LEFT_OUT = ("events", "argv")
MAX_CHARS = 20_000


def _clip(value, cut: list[int]):
    """VALUE with every string over MAX_CHARS cut there (`cut[0]` counts them)."""
    if isinstance(value, str):
        if len(value) <= MAX_CHARS:
            return value
        cut[0] += 1
        return f"{value[:MAX_CHARS]}\n... [cut: {len(value) - MAX_CHARS:,} more characters]"
    if isinstance(value, list):
        return [_clip(item, cut) for item in value]
    if isinstance(value, dict):
        return {key: _clip(item, cut) for key, item in value.items()}
    return value


def _failure(event) -> bool:
    """Whether a raw event of the journal reports an error: its type says so, or the item it carries has one."""
    if not isinstance(event, dict):
        return False
    item = event.get("item")
    return any(word in str(event.get("type", "")) for word in ("error", "fail")) or (
        isinstance(item, dict) and bool(item.get("error")))


def _record(rec: dict, cut: list[int], left_out: set[str]) -> dict:
    out = {key: value for key, value in rec.items() if key not in LEFT_OUT}
    left_out.update(key for key in LEFT_OUT if key in rec)
    if errors := [event for event in rec.get("events") or [] if _failure(event)]:
        out["event_errors"] = errors
    for call in out.get("tools") or []:
        if isinstance(call, dict) and not isinstance(call.get("result"), str):
            text = json.dumps(call.get("result"), ensure_ascii=False)
            if len(text) > MAX_CHARS:   # a structure too long to show: its text, cut like any other
                call["result"] = text
    return _clip(out, cut)


def read_log(case_dir: Path) -> dict | None:
    """The case's journal: `{"file", "records", "cut" (strings cut at MAX_CHARS), "left_out" (fields left out),
    "problems"}`, or None when the folder holds none."""
    files = sorted(case_dir.glob(f"*{LOG_SUFFIX}"))
    if not files:
        return None
    problems: list[str] = []
    cut, left_out = [0], set()
    records = [_record(rec, cut, left_out) for rec in jsonl(files[0], problems)]
    return {"file": files[0].name, "records": records, "cut": cut[0], "left_out": sorted(left_out),
            "problems": problems}


def read_transport(case_dir: Path) -> dict | None:
    """`{"name", "summary"}` from the first `<name>.summary.json` of the folder (the system's own counts, beside the
    run's `summary.json`), None when there is none."""
    for path in sorted(case_dir.glob(f"*{SUMMARY_SUFFIX}")):
        if (summary := json_object(path)) is not None:
            return {"name": path.name.removesuffix(SUMMARY_SUFFIX), "summary": summary}
    return None
