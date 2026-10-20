"""The summary of a run of the reference agent, counted from its journal (V0 §9).

`summarize` counts a log, not a run. The execs (by kind), tokens, retries, warnings and the time the execs took come
from the exec records; the frames served, the pulls and the per-frame latency from the `served` and `call` events the
feed wrote (`codex_feed`); the answers, the resumes and the failed execs from the events `codex_system` noted; and
whatever the log cannot know (the model, the effort, Codex's version, ...) is passed in as keywords and kept as given.
Tokens are each exec's own share of its thread's running total (`codex_log.Journal`), so a sum counts each token once;
`in` includes the cached ones. A turn the harness killed or that failed printed no usage, so its tokens are missing:
`tokens_lower_bound` says the count is a floor. The cost of a Codex run is its tokens, not dollars (a Claude Code
run's summary adds the CLI's API-equivalent dollars, `cmd_codex.drive`: `cost_usd` and `cost_note`).

- `frames` is the number of frames served, those that came with questions included (`frames_asked` counts these: a
  frame the model was asked about in the same item, `codex_system`'s held frame); `pulls` the `next_frame` calls the
  model made and `premature_pulls` the ones among them made while questions were unanswered; `resumes` the execs
  started to resume a thread whose turn ended early (`stalled` in `exec_failures` when they ran out).
- The per-frame latency is the time from serving a frame to the model's next accepted `next_frame` call: what it took
  to look at the frame and move on (`median_s` and `p90_s` over the frames, `n` of them). It is the plain frames'. A
  frame served with questions is timed to the pull after the answers, which is looking, deciding and answering, so it
  is reported apart (`asked_latency`, the same shape) and not in `frame_latency`. A pull the feed refused (the nudge, a
  second call at once) is no move on and does not count.
- An answer is `submitted`, `repaired` (a rejected try or the nudge came first) or `failed` with a reason (`malformed`,
  `no_submit`). `exec_failures` counts the execs that raised, by `codex_exec.failure_reason` (`stalled` and `timeout`
  among them), and `timed_out` counts the questions whose ask ended on a timeout.
- A run with a memory (`codex_memory`) counts the memory's tool calls by name in `tool_calls` and `tool_errors`, beside
  the tool server's (the journal's `memory_call` events), so the `claim` calls per frame are `tool_calls["claim"]` over
  `frames`. A sensor feed that failed is `memory_failure`, its message cut: a key only such a run has.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

from .agent_tools import NEXT_NAME
from .codex_log import read_log

FAILURES = ("malformed", "no_submit")  # why an answer fails (`codex_system`); each is counted, others as they come
USAGE = {"in": "prompt_tokens", "cached_in": "cached_tokens", "out": "completion_tokens",
         "reasoning": "reasoning_tokens"}


def latency(events: list[dict], *, asked: bool = False) -> dict:
    """Per-frame latency: for each frame served, the seconds to the model's next `next_frame` call that the feed
    accepted, which is the pull the next item went to (one pull waits at a time and a refused one takes nothing). It is
    found by the item's number, not by comparing times: a coarse clock (Windows' monotonic one ticks every 15.6 ms
    before Python 3.13) gives a quick pull its frame's own time. Plain frames, or with `asked` the frames that came
    with questions: their pull comes after the answers. `n` is how many frames have one; `median_s` and `p90_s`
    (nearest rank) are None for none."""
    took = {e["item"]: e["at"] for e in events if e["event"] == "call" and e["tool"] == NEXT_NAME and "item" in e}
    waits = [took[e["item"] + 1] - e["at"] for e in events
             if e["event"] == "served" and e["kind"] == "frame" and bool(e.get("asked")) == asked
             and e["item"] + 1 in took]
    if not waits:
        return {"n": 0, "median_s": None, "p90_s": None}
    ranked = sorted(waits)
    return {"n": len(waits), "median_s": round(statistics.median(ranked), 3),
            "p90_s": round(ranked[math.ceil(0.9 * len(ranked)) - 1], 3)}


MESSAGE_CHARS = 300  # the most of a failed exec's message the summary keeps


def summarize(log: Path, **meta) -> dict:
    """The summary of a journal: `meta` as given, then the counts. A log that does not exist is an empty run."""
    records = read_log(log) if Path(log).exists() else []
    execs = [r for r in records if "event" not in r]
    events = [r for r in records if "event" in r]
    outcomes = [e for e in events if e["event"] == "answer_outcome"]
    broken = [e for e in events if e["event"] == "exec_failed"]
    calls = [e for e in events if e["event"] == "call"]
    memory = [e for e in events if e["event"] == "memory_call"]
    unfed = [e for e in events if e["event"] == "memory_failed"]
    failed = Counter(e.get("reason", "unspecified") for e in outcomes if e["outcome"] == "failed")
    pulls = [c for c in calls if c["tool"] == NEXT_NAME]
    return {
        **meta,
        "execs": {"total": len(execs), "by_kind": dict(Counter(r["kind"] for r in execs))},
        "frames": sum(e["event"] == "served" and e["kind"] == "frame" for e in events),
        "frames_asked": sum(e["event"] == "served" and e["kind"] == "frame" and bool(e.get("asked")) for e in events),
        "pulls": len(pulls),
        "premature_pulls": sum(bool(c.get("premature")) for c in pulls),
        "resumes": sum(e["event"] == "exec_started" and e["kind"] == "resume" and e["attempt"] == 1 for e in events),
        "frame_latency": latency(events),
        "asked_latency": latency(events, asked=True),
        "tokens": {name: sum(r["usage"][key] for r in execs) for name, key in USAGE.items()},
        "tokens_lower_bound": bool(broken) or any(r["killed"] for r in execs),
        "tool_calls": dict(Counter(c["tool"] for c in [*calls, *memory])),
        "tool_errors": sum(bool(c["error"]) for c in [*calls, *memory]),
        "retries": sum(r["attempts"] - 1 for r in execs),
        "warnings": sum(len(r["warnings"]) for r in execs),
        "exec_wall_s": round(sum(r["latency_s"] for r in execs), 3),
        "answers": {"submitted": sum(e["outcome"] == "submitted" for e in outcomes),
                    "repaired": sum(e["outcome"] == "repaired" for e in outcomes),
                    "failed": {reason: failed[reason] for reason in FAILURES}
                    | {reason: n for reason, n in failed.items() if reason not in FAILURES},
                    "timed_out": sum(e.get("questions", 0) for e in broken if e["reason"] == "timeout")},
        "exec_failures": dict(Counter(e["reason"] for e in broken)),
        # the last failed exec's own message (cut), kept here because a run whose first exec failed spent nothing and
        # its journal is removed (`cmd_codex._unran`): without it an outage reads only "error" (FINDINGS 2026-10-14)
        "exec_failure_message": str(broken[-1].get("message") or "")[:MESSAGE_CHARS] if broken else None,
        **({"memory_failure": str(unfed[-1].get("error") or "")[:MESSAGE_CHARS]} if unfed else {}),
    }


def _seen(lat: dict) -> str:
    return f"median {lat['median_s']:g} s, p90 {lat['p90_s']:g} s" if lat["n"] else "none"


def format_summary(s: dict) -> list[str]:
    """The lines `stbench run` prints for a summary."""
    t, a, e = s["tokens"], s["answers"], s["execs"]
    asked = (f"; {s['frames_asked']} came with questions, timed to the pull after the answers: "
             f"{_seen(s['asked_latency'])}" if s.get("frames_asked") else "")
    return [
        f"execs {e['total']} {e['by_kind']}, {s['resumes']} resumed, {s['retries']} retried, {s['warnings']} warnings, "
        f"wall {s['exec_wall_s']:g} s",
        f"frames {s['frames']}, pulls {s['pulls']} ({s['premature_pulls']} premature), per-frame latency "
        f"{_seen(s['frame_latency'])}{asked}",
        f"tokens: in {t['in']} ({t['cached_in']} cached), out {t['out']}, reasoning {t['reasoning']}"
        + (" (a lower bound: a turn was killed or failed)" if s["tokens_lower_bound"] else ""),
        *([f"cost: {s['cost_usd']:g} USD, {s['cost_note']}"] if s.get("cost_note") else []),
        f"tool calls: {sum(s['tool_calls'].values())} {s['tool_calls']}, {s['tool_errors']} errors",
        f"answers: {a['submitted']} submitted, {a['repaired']} repaired, {sum(a['failed'].values())} failed "
        f"{a['failed']}, {a['timed_out']} timed out; execs that failed: {s['exec_failures'] or 'none'}",
        *([f"last failed exec: {s['exec_failure_message']}"] if s.get("exec_failure_message") else []),
        *([f"the memory's sensor feed failed: {s['memory_failure']}"] if s.get("memory_failure") else []),
    ]
