"""The journal of the reference agent, read back: its transcripts (`codex_log`), its summary (`llm_summary`) and its
leak scan (`llm_scan`), on hand-written logs. A log here is a list of records as `codex_log.Journal` and the feed write
them (`exec_started`, `served`, `call`, the exec records and the events `codex_system` notes), so every expected number
and line is worked out by hand. The same journal, written by a real case over the fake command, is
`test_codex_pull.py`'s.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_kit import ROBOT
from stbench import llm_render
from stbench.codex_log import Journal, read_log, scrubbed, transcripts
from stbench.llm_scan import HEADER, scan_log
from stbench.llm_summary import format_summary, latency, summarize

USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0,
         "reasoning_tokens": 0, "cost": 0.0}
QID_1, QID_2, QID_3 = "a1b2c3d4e5f60718", "0f1e2d3c4b5a6978", "1122334455667788"
EPISODE = "99aabbccddeeff00"
QUESTIONS = [
    {"qid": QID_1, "text": "How many red mugs did you see on your first visit?"},
    {"qid": QID_2, "text": "Where was the lamp last?"},
    {"qid": QID_3, "text": "How many red mugs"},  # its text is the start of the first question's
]
JPEG = "<jpeg 3 bytes sha256 aa>"


def write_log(path: Path, lines: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    return path


def started(prompt: str, kind: str = "start", attempt: int = 1) -> dict:
    return {"event": "exec_started", "kind": kind, "base": None if kind == "start" else "T", "prompt": prompt,
            "attempt": attempt}


def served(item: int, kind: str, text: str, *, k=None, images=(), ask=None, via=None, at: float = 0.0,
           asked: str | None = None) -> dict:
    """A `served` event as the feed writes it (`asked`: the questions' text of a frame that came with them)."""
    return {"event": "served", "item": item, "kind": kind, "k": k, "text": text, "images": list(images), "at": at,
            "time": at, **({"ask": ask} if ask else {}), **({"asked": asked} if asked else {}),
            **({"via": via} if via else {})}


def called(tool: str, result: str | None = None, *, item: int | None = None, at: float = 0.0, premature=False,
           error: bool = False) -> dict:
    """A `call` event: a pull that was served an item names it and has no result, any other call has its reply."""
    return {"event": "call", "tool": tool, "arguments": {}, "error": error, "at": at, "time": at,
            **({"item": item} if item is not None else {"result": result}),
            **({"premature": True} if premature else {})}


def execrec(i: int = 0, *, kind: str = "start", messages=(), usage=None, killed=False, latency_s=0.0, attempts=1,
            warnings=()) -> dict:
    """An exec record as `Journal.exec` writes it (the keys the readers use)."""
    return {"i": i, "t": 0.0, "session": "s", "kind": kind, "base": None if kind == "start" else "T", "thread": "T",
            "messages": list(messages), "events": [], "argv": [], "stderr": "", "warnings": list(warnings),
            "attempts": attempts, "usage": usage or USAGE, "thread_usage": USAGE, "cost": 0.0, "provider": "codex",
            "latency_s": latency_s, "killed": killed}


def ask_of(when: str, now_k: int, **labels: str) -> dict:
    return {"when": when, "now_k": now_k, "labels": labels}


# -- the transcripts ---------------------------------------------------------------------------------------------
def test_a_transcript_is_the_conversation_the_model_held_at_each_ask_and_a_last_one_for_the_case():
    """Contract (`codex_log` docstring): a case is one thread, and `transcripts` rebuilds what the model held from the
    events in the order they were written: the first prompt (the first task, served in it, adds nothing more), each item
    served as a tool message (its text, then its images as the hash the log keeps), the replies to the model's other
    calls (a pull that was served an item adds nothing: the item is already there), a resume's line as a user message,
    and at the end of an exec the model's own words. Each ask gets the conversation up to and including it, with the
    ask's labels; the case gets one last transcript with everything. A retried exec's prompt (attempt 2) is not added
    again. Regression caught: a transcript that gives an ask only its own item (the scan would never see a frame served
    before), one that lists a prompt twice, or one that loses the order of the model's calls and the items."""
    log = [served(0, "task", "T0", via="prompt"), started("first"), started("first", attempt=2),
           served(1, "frame", "[k0 c] h", k=0, images=[JPEG]), called("next_frame", item=1),
           served(2, "ask", "Q1: ?", k=0, ask=ask_of("end", 0, Q1="x1")), called("next_frame", item=2),
           called("submit_answer", "ok"), called("next_frame", "error: you have not answered Q2", premature=True),
           execrec(messages=["done"]), started("Call next_frame to continue.", kind="resume"),
           served(3, "end", "over"), execrec(1, kind="resume", messages=["bye"])]
    first = [{"role": "user", "content": [{"type": "text", "text": "first"}]},
             {"role": "tool", "tool_call_id": "", "content": [{"type": "text", "text": "[k0 c] h"},
                                                              {"type": "image_url",
                                                               "image_url": {"url": JPEG, "detail": "high"}}]},
             {"role": "tool", "tool_call_id": "", "content": [{"type": "text", "text": "Q1: ?"}]}]
    rest = [{"role": "tool", "tool_call_id": "", "content": "ok"},
            {"role": "tool", "tool_call_id": "", "content": "error: you have not answered Q2"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": [{"type": "text", "text": "Call next_frame to continue."}]},
            {"role": "tool", "tool_call_id": "", "content": [{"type": "text", "text": "over"}]},
            {"role": "assistant", "content": "bye"}]
    found = transcripts(log)
    assert [(c["i"], c["purpose"]) for c in found] == [(2, "ask"), (4, "case")]
    assert found[0]["request"]["messages"] == first and found[0]["ask"] == ask_of("end", 0, Q1="x1")
    assert found[1]["request"]["messages"] == first + rest and "ask" not in found[1]
    assert transcripts([])[0]["request"]["messages"] == [] and transcripts([])[0]["i"] == 0


def test_a_frame_that_came_with_questions_is_one_message_and_its_ask_ends_a_transcript_there():
    """Contract (`codex_log` docstring; FD-questions-0.10 §1.5): a frame served with the questions about it is one
    `frame` item with `ask` and `asked`, and the model held it as one tool message: the frame's text, its image, then
    the questions' text. Its ask gets a transcript that ends with that message (its item number, its `when`, `now_k` and
    labels); the frame before it, which had no questions, makes none. Regression caught: a transcript that drops the
    questions' text (the scan would never see a question shown), one that puts it before the image, or an ask that
    makes no transcript because its item is a frame."""
    ask = ask_of("during", 1, Q1="x1")
    log = [served(0, "task", "T0", via="prompt"), started("first"),
           served(1, "frame", "[k0 c] h", k=0, images=[JPEG]), called("next_frame", item=1),
           served(2, "frame", "[k1 c] h", k=1, images=[JPEG], ask=ask, asked="Questions during the job.\nQ1: ?"),
           called("next_frame", item=2), called("submit_answer", "ok")]
    found = transcripts(log)
    assert [(c["i"], c["purpose"]) for c in found] == [(2, "ask"), (3, "case")]
    joined = {"role": "tool", "tool_call_id": "", "content": [
        {"type": "text", "text": "[k1 c] h"}, {"type": "image_url", "image_url": {"url": JPEG, "detail": "high"}},
        {"type": "text", "text": "Questions during the job.\nQ1: ?"}]}
    assert found[0]["ask"] == ask and len(found[0]["request"]["messages"]) == 3
    assert found[0]["request"]["messages"][-1] == joined
    assert found[1]["request"]["messages"][-2:] == [joined, {"role": "tool", "tool_call_id": "", "content": "ok"}]


def test_the_journal_scrubs_images_to_hashes_and_counts_what_cost_the_plan_something(tmp_path):
    """Contract (FD-frame-pull §3.3, "never base64 in the journal"): an MCP image block in an event is written with its
    `hash` (`<jpeg N bytes sha256 HEX16>`) in place of its `data`, a `data:image/...;base64,` string is its hash, other
    keys and values are untouched; an exec's own usage is the difference from its thread's last running total (a
    killed exec, which prints none, leaves the total as it was); `spent` counts the model's calls and the exec records,
    not the other events. Hand-written hashes of known bytes. Regression caught: base64 in the log, a killed exec that
    resets its thread's total (the next exec's share would be the whole total), or a journal of nothing kept as a
    run."""
    data = b"\xff\xd8\xff\xe0frame-7\xff\xd9"
    tag = "<jpeg 13 bytes sha256 c0eb4e1e112bf53f>"  # sha256 of `data`, first 16 hex characters
    event = {"type": "item.completed", "item": {"result": {"content": [
        {"type": "text", "text": "[k7 c] h"}, {"type": "image", "data": base64.b64encode(data).decode(),
                                              "mimeType": "image/jpeg"}]}, "other": [1, "x"]},
        "uri": "data:image/jpeg;base64," + base64.b64encode(data).decode()}
    assert scrubbed(event) == {"type": "item.completed", "item": {"result": {"content": [
        {"type": "text", "text": "[k7 c] h"}, {"type": "image", "mimeType": "image/jpeg", "hash": tag}]},
        "other": [1, "x"]}, "uri": tag}
    journal = Journal(tmp_path / "j.jsonl")
    assert journal.spent == 0
    journal.note({"event": "served", "item": 0})
    assert journal.spent == 0
    journal.note({"event": "call", "tool": "next_frame"})
    assert journal.spent == 1
    with pytest.raises(ValueError, match="needs an 'event' key"):
        journal.note({"item": 1})
    total = {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 40, "cache_write_tokens": 0,
             "reasoning_tokens": 0, "cost": 0.0}
    done = SimpleNamespace(thread="T", messages=[], usage=total, events=[event], argv=[], latency_s=1.0, attempts=1,
                           warnings=[], stderr="", completed=True, killed=False)
    assert journal.exec(session="s", kind="start", base=None, result=done)["prompt_tokens"] == 100
    cut = SimpleNamespace(**{**vars(done), "usage": dict(USAGE), "completed": False, "killed": True})
    assert journal.exec(session="s", kind="resume", base="T", result=cut)["prompt_tokens"] == 0
    again = SimpleNamespace(**{**vars(done), "usage": {**total, "prompt_tokens": 250}})
    assert journal.exec(session="s", kind="resume", base="T", result=again)["prompt_tokens"] == 150
    records = read_log(journal.path)
    assert [r.get("killed") for r in records if "event" not in r] == [False, True, False] and journal.spent == 4
    assert tag in journal.path.read_text(encoding="utf-8") and base64.b64encode(data).decode() not in (
        journal.path.read_text(encoding="utf-8"))


# -- the summary -------------------------------------------------------------------------------------------------
def test_the_summary_counts_a_log_by_cause(tmp_path):
    """Contract (plan 4.9; FD-frame-pull §3.4): `summarize` counts a journal, not a run: execs by kind and each exec's
    own share of the tokens (the cached ones inside the prompt tokens), frames served, pulls and the premature ones,
    resumes (not the retry of a start), tool calls by name and the replies that were errors, the retries the transport
    made, Codex's warnings and the time the execs took; the answers by outcome with the failures by cause (a cause the
    system adds later is counted, not dropped); the execs that failed by reason, the questions whose ask ended on a
    timeout, and whether a turn that was killed or failed makes the token count a floor; the extra keywords are kept as
    given; a log that does not exist is an empty run. Regression caught: a failure cause that vanishes from the
    totals, a timed-out ask scored as nothing, double-counted tokens, a retry counted as a resume, or a crash on an
    empty run."""
    usage = USAGE | {"prompt_tokens": 100, "cached_tokens": 40, "completion_tokens": 10, "reasoning_tokens": 2}
    log = write_log(tmp_path / "codex.calls.jsonl", [
        started("p"), served(0, "task", "T", via="prompt"),
        served(1, "frame", "[k0 c] h", k=0, images=[JPEG], at=10.0), called("next_frame", item=1, at=9.5),
        served(2, "ask", "Q1:", k=0, ask=ask_of("during", 0, Q1="x"), at=10.5), called("next_frame", item=2, at=10.4),
        called("next_frame", "error: you have not answered Q1", premature=True, error=True, at=10.6),
        called("submit_answer", "ok", at=10.7), called("submit_answer", "error: the answer was already submitted",
                                                       error=True, at=10.8),
        called("next_frame", "error: another next_frame call is already waiting", error=True, at=10.9),
        served(3, "frame", "[k1 c] h", k=1, images=[JPEG], at=20.0), called("next_frame", item=3, at=10.95),
        called("next_frame", item=4, at=20.5),
        execrec(0, usage=usage, latency_s=1.5, warnings=["w"]),
        started("Call next_frame to continue.", kind="resume"), started("Call next_frame to continue.", kind="resume",
                                                                         attempt=2),
        execrec(1, kind="resume", usage=usage | {"prompt_tokens": 300, "cached_tokens": 200}, latency_s=2.0,
                attempts=2),
        {"event": "answer_outcome", "qid": "q1", "label": "Q1", "outcome": "submitted"},
        {"event": "answer_outcome", "qid": "q2", "label": "Q2", "outcome": "repaired"},
        {"event": "answer_outcome", "qid": "q3", "label": "Q3", "outcome": "failed", "reason": "malformed"},
        {"event": "answer_outcome", "qid": "q4", "label": "Q4", "outcome": "failed", "reason": "no_submit"},
        {"event": "answer_outcome", "qid": "q5", "label": "Q5", "outcome": "failed", "reason": "no_submit"},
        {"event": "answer_outcome", "qid": "q6", "label": "Q6", "outcome": "failed", "reason": "oddity"},
        {"event": "exec_failed", "purpose": "ask", "kind": "resume", "reason": "timeout", "questions": 3},
        {"event": "exec_failed", "purpose": "observe", "kind": "resume", "reason": "stalled"},
        {"event": "exec_failed", "purpose": "ask", "kind": "resume", "reason": "context", "questions": 2,
         "message": "ran out of room in the model's context window"},
    ])
    summary = summarize(log, model="m", case="c")
    assert summary == {
        "model": "m", "case": "c",
        "execs": {"total": 2, "by_kind": {"start": 1, "resume": 1}},
        "frames": 2, "frames_asked": 0, "pulls": 6, "premature_pulls": 1, "resumes": 1,
        "frame_latency": {"n": 2, "median_s": 0.45, "p90_s": 0.5},
        "asked_latency": {"n": 0, "median_s": None, "p90_s": None},
        "tokens": {"in": 400, "cached_in": 240, "out": 20, "reasoning": 4}, "tokens_lower_bound": True,
        "tool_calls": {"next_frame": 6, "submit_answer": 2}, "tool_errors": 3, "retries": 1, "warnings": 1,
        "exec_wall_s": 3.5,
        "answers": {"submitted": 1, "repaired": 1, "failed": {"malformed": 1, "no_submit": 2, "oddity": 1},
                    "timed_out": 3},
        "exec_failures": {"timeout": 1, "stalled": 1, "context": 1},
        "exec_failure_message": "ran out of room in the model's context window"}   # the last failed exec's own words
    lines = format_summary(summary)
    assert lines[1] == "frames 2, pulls 6 (1 premature), per-frame latency median 0.45 s, p90 0.5 s"
    assert lines[2] == "tokens: in 400 (240 cached), out 20, reasoning 4 (a lower bound: a turn was killed or failed)"
    empty = summarize(tmp_path / "missing.jsonl")
    assert empty["execs"] == {"total": 0, "by_kind": {}} and empty["tokens"]["in"] == 0 and empty["frames"] == 0
    assert empty["answers"] == {"submitted": 0, "repaired": 0, "failed": {"malformed": 0, "no_submit": 0},
                                "timed_out": 0}
    assert empty["exec_failures"] == {} and empty["exec_wall_s"] == 0 and empty["tokens_lower_bound"] is False
    assert empty["frame_latency"] == {"n": 0, "median_s": None, "p90_s": None} == empty["asked_latency"]
    assert format_summary(empty)[1].endswith("per-frame latency none") and empty["frames_asked"] == 0


def test_the_per_frame_latency_is_the_time_from_serving_a_frame_to_the_next_pull():
    """Contract (FD-frame-pull §3.4): the latency of a frame is the seconds from the moment it was served to the arrival
    of the model's next `next_frame` call, whatever else the model called in between; the summary gives the median
    (of an even count the mean of the middle two) and the nearest-rank 90th percentile. Frames served at 10, 20, 30 and
    40 s, with pulls arriving at 9.5 (the one that took the first), 10.4, 20.9, 31.2 and 40.2: waits of 0.4, 0.9, 1.2
    and 0.2 s, median 0.65 and p90 1.2. A frame with no pull after it is left out. The next pull is the one the next
    item went to, whatever the clock says: a coarse clock (15.6 ms on Windows before Python 3.13) stamps the pull that
    took a frame, the frame and the quick pull after it with one time, and that frame waited 0 s. Regression caught: a
    latency counted from the pull that took the frame (it would be the harness's own time), a frame dropped or timed
    to a later pull because its pull came in the same tick of the clock (CI on Windows, run 38669003003), or a
    percentile read from the wrong end."""
    events = [served(n, "frame", "[k c] h", k=n, at=at) for n, at in enumerate((10.0, 20.0, 30.0, 40.0), 1)]
    events += [called("submit_answer", "ok", at=15.0)]
    events += [called("next_frame", item=n, at=at) for n, at in enumerate((9.5, 10.4, 20.9, 31.2, 40.2), 1)]
    assert latency(events) == {"n": 4, "median_s": 0.65, "p90_s": 1.2}
    assert latency(events[:1])["n"] == 0  # no pull at all after the frame
    assert latency([served(1, "frame", "[k c] h", k=0, at=1.0), called("next_frame", item=1, at=1.0)])["n"] == 0
    tick = [called("next_frame", item=n, at=at) for n, at in enumerate((1.0, 1.0, 2.0, 2.0), 1)]
    tick += [served(n, "frame", "[k c] h", k=n, at=at) for n, at in enumerate((1.0, 1.0, 2.0), 1)]
    assert latency(tick) == {"n": 3, "median_s": 0.0, "p90_s": 1.0}  # waits of 0, 1 and 0 s


def test_a_frame_served_with_questions_is_timed_to_the_pull_after_the_answers_and_apart_from_plain_frames(tmp_path):
    """Contract (FD-questions-0.10 §1.5): a frame that came with questions is timed from serving to the first
    `next_frame` the feed accepted, so a pull it refused (the nudge, an error) does not end it: that is looking,
    deciding and answering, and it is reported apart (`asked_latency`) and not among the plain frames'
    (`frame_latency`), though `frames` counts it and `frames_asked` counts it again. Plain frames served at 10.0 and
    13.0, pulled at 10.5
    and 13.2 (0.5 and 0.2 s); frames with questions served at 10.5 and 13.2, the first with a refused pull at 11.0 and
    both accepted at 13.0 and 15.2 (2.5 and 2.0 s). Regression caught: the two kinds of wait in one median (a model that
    answers is slower than one that looks, and the per-frame figure would say the model slowed down whenever a point
    asked), or a frame with questions timed to the nudge."""
    ask = ask_of("during", 1, Q1="x1")
    log = write_log(tmp_path / "codex.calls.jsonl", [
        started("p"), served(0, "task", "T", via="prompt"),
        served(1, "frame", "[k0 c] h", k=0, images=[JPEG], at=10.0), called("next_frame", item=2, at=10.5),
        served(2, "frame", "[k1 c] h", k=1, images=[JPEG], at=10.5, ask=ask, asked="Questions\nQ1: ?"),
        called("next_frame", "error: you have not answered Q1", premature=True, error=True, at=11.0),
        called("submit_answer", "ok", at=11.5), called("next_frame", item=3, at=13.0),
        served(3, "frame", "[k2 c] h", k=2, images=[JPEG], at=13.0), called("next_frame", item=4, at=13.2),
        served(4, "frame", "[k3 c] h", k=3, images=[JPEG], at=13.2, ask=ask, asked="Questions\nQ1: ?"),
        called("next_frame", item=5, at=15.2)])
    events = [r for r in read_log(log) if "event" in r]
    assert latency(events) == {"n": 2, "median_s": 0.35, "p90_s": 0.5}
    assert latency(events, asked=True) == {"n": 2, "median_s": 2.25, "p90_s": 2.5}
    summary = summarize(log)
    assert (summary["frames"], summary["frames_asked"]) == (4, 2) and summary["premature_pulls"] == 1
    assert summary["frame_latency"] == latency(events) and summary["asked_latency"] == latency(events, asked=True)
    assert format_summary(summary)[1] == (
        "frames 4, pulls 5 (1 premature), per-frame latency median 0.35 s, p90 0.5 s; "
        "2 came with questions, timed to the pull after the answers: median 2.25 s, p90 2.5 s")


# -- the scan ----------------------------------------------------------------------------------------------------
def test_the_scan_reads_what_the_harness_wrote_and_each_message_once(tmp_path):
    """Contract (plan §2): `scan_log` applies the leak scan's markers (an entity id, a region id, the word truth in any
    case) and the given map names, case-insensitively, to every message the harness made (the prompt and the items and
    replies it served: system, user and tool), and to nothing else: the model's own messages are not the harness's. A
    message that a later transcript repeats is reported once, at the first one; a clean log gives []. Regression
    caught: a scan of the model's own words (false alarms that teach people to ignore it), a scan that skips the items
    the model is served because they are tool messages, or a report per repeated transcript."""
    leaky = write_log(tmp_path / "leaky.jsonl", [
        served(0, "task", "T0", via="prompt"), started("Mission: tidy up. E0042 is a mug."),
        served(1, "frame", "[k0 c] You saw R3 in Apartment_A. The TRUTH is out there.", k=0),
        served(2, "ask", "Q1: ?", k=0, ask=ask_of("end", 0, Q1=QID_1)),
        called("submit_answer", "truth R9 apartment_a E0099"), execrec(messages=["I recall E0099 and R7."])])
    assert scan_log(leaky, ["apartment_a", ""]) == [
        "call 2, message 0 (user): entity_id 'E0042'",
        "call 2, message 1 (tool): region_id 'R3'",
        "call 2, message 1 (tool): truth 'TRUTH'",
        "call 2, message 1 (tool): map_name 'apartment_a'",
        "call 3, message 3 (tool): entity_id 'E0099'",
        "call 3, message 3 (tool): region_id 'R9'",
        "call 3, message 3 (tool): truth 'truth'",
        "call 3, message 3 (tool): map_name 'apartment_a'",
    ]
    clean = write_log(tmp_path / "clean.jsonl", [
        served(0, "task", "T0", via="prompt"), started("Mission: tidy up."),
        served(1, "frame", "Frames k0 to k1", k=0), called("submit_answer", "error: unknown tool"),
        execrec(messages=["R9 E0007 truth"])])
    assert scan_log(clean, ["apartment_a"]) == []


def test_the_scan_catches_the_future(tmp_path):
    """Contract (plan 4.9; map HA-H14): beside the markers, `scan_log` reports, in the messages the harness made (not
    the model's own words): a question id or the case's id as a whole word; the dataset's absolute path; the text of a
    question before the ask that puts it (the questions asked so far may appear later in the thread, and a question
    whose text an asked one contains is not another); and, in an ask's transcript, a frame header whose `k` is above the
    ask's step (k2 is fine at an ask at step 2, k3 is not). Each hit names the call (the ask's item, or the number of
    items for the whole case), message and role. Without the run's questions, ids and path only the markers and the
    frame rule are scanned. Regression caught: a leak the markers cannot see (an id, a path, a later question reaching
    the stream, a frame from after an ask's step in its transcript)."""
    dataset = tmp_path / "data"
    log = write_log(tmp_path / "scan.jsonl", [
        served(0, "task", "T0", via="prompt"), started("Task 0 begins at c.\nClean."),
        served(1, "frame", "Frames\n[k0 t]\n[k1 t]\n[k2 t]\n[k3 t]\nWhere was the lamp last? ok", k=0, images=[JPEG]),
        served(2, "ask", "Questions during the job.\nQ1: " + QUESTIONS[0]["text"], k=2,
               ask=ask_of("during", 2, Q1=QID_1)),
        called("submit_answer", f"saved to {dataset.resolve().as_posix()}/episodes"),
        served(3, "ask", f"Questions after the job, see {EPISODE}.\nQ2: {QUESTIONS[1]['text']}\nQ3: "
               f"{QUESTIONS[2]['text']}", k=9, ask=ask_of("end", 9, Q2=QID_2, Q3=QID_3)),
        called("submit_answer", f"read {QID_1}"), execrec(messages=[f"noted {QID_2} [k99 x]"])])
    hits = scan_log(log, questions=QUESTIONS, episode=EPISODE, dataset=dataset)
    assert hits == [
        f"call 2, message 1 (tool): the text of question {QID_2} before its ask",
        "call 2, message 1 (tool): frame k3 after the ask's step k2",
        f"call 3, message 3 (tool): dataset path '{dataset.resolve().as_posix().lower()}'",
        f"call 3, message 4 (tool): episode_id '{EPISODE}'",
        f"call 4, message 5 (tool): qid '{QID_1}'",
    ]
    assert scan_log(log) == ["call 2, message 1 (tool): frame k3 after the ask's step k2"]


def test_a_question_shown_in_any_message_before_its_own_ask_item_is_found(tmp_path):
    """Contract (V0 §9; the final review's S1): the ask's own questions count as asked only in the ask's item, the last
    message of its transcript (an ask alone, or a frame item that carries it); every message before it is read against
    the asks before it. So a question's text in a frame served before its own ask is a hit, whether that frame is the
    first of the case, the one right before the ask or one between two asks (where the earlier ask's question may be
    shown and the next one's may not; the frame there shows both, and only the next one's is the hit), and in a joined
    item the frame part may hold no question but the item's own. The controls are clean: the ask's item showing its
    question and a joined item with its questions. Hand-written logs; QID_1 and QID_2 are asked in turn. Regression
    caught: the questions of the transcript's ask taken as asked for all its messages (the scan then misses a question
    put in the stream ahead of its ask, the one leak it exists to find)."""
    q1, q2 = (q["text"] for q in QUESTIONS[:2])
    one, two = ask_of("during", 1, Q1=QID_1), ask_of("end", 2, Q2=QID_2)
    head = [served(0, "task", "T0", via="prompt"), started("Task 0 begins at c.\nClean.")]
    joined = served(2, "frame", "[k1 t]", k=1, ask=one, asked="Questions during the job.\nQ1: " + q1)

    def hit(call: int, at: int, qid: str) -> list[str]:
        return [f"call {call}, message {at} (tool): the text of question {qid} before its ask"]

    cases = {
        "ahead of the ask": ([*head, served(1, "frame", "[k0 t] " + q1, k=0), served(2, "frame", "[k1 t]", k=1),
                              served(3, "ask", "Q1: " + q1, k=1, ask=one)],
                             hit(3, 1, QID_1)),
        "right before the ask": ([*head, served(1, "frame", "[k0 t]", k=0), served(2, "frame", "[k1 t] " + q1, k=1),
                                  served(3, "ask", "Q1: " + q1, k=1, ask=one)],
                                 hit(3, 2, QID_1)),
        "ahead of a joined item": ([*head, served(1, "frame", "[k0 t] " + q1, k=0), joined],
                                   hit(2, 1, QID_1)),
        "in the frame part of a joined item": ([*head, served(1, "frame", "[k0 t]", k=0),
                                                {**joined, "text": "[k1 t] " + q2}],
                                               hit(2, 2, QID_2)),
        "between two asks": ([*head, served(1, "frame", "[k0 t]", k=0), served(2, "ask", "Q1: " + q1, k=0, ask=one),
                              called("submit_answer", "ok"), served(3, "frame", f"[k1 t] {q1} {q2}", k=1),
                              served(4, "ask", "Q2: " + q2, k=2, ask=two)],
                             hit(4, 4, QID_2)),
        "the item alone": ([*head, served(1, "frame", "[k0 t]", k=0), served(2, "ask", "Q1: " + q1, k=0, ask=one)], []),
        "a joined item with its questions": ([*head, served(1, "frame", "[k0 t]", k=0), joined], []),
    }
    for name, (lines, hits) in cases.items():
        log = write_log(tmp_path / f"{name.replace(' ', '-')}.jsonl", lines)
        assert scan_log(log, questions=QUESTIONS[:2]) == hits, name


def test_what_is_served_after_the_last_ask_is_scanned_to_the_end_of_the_case(tmp_path):
    """Contract (V0 §9; the final review's S1): the frames served after the last ask are in the case's own transcript
    (the last one, which has no ask) and are read there against all the asks: they may show a question that was asked
    and not a question that never was, and a marker in them is a hit. The frame rule does not apply to them (the stream
    has moved on past the last ask's step, and a later frame is expected). By hand: the frame at item 3 shows the first
    question (asked: allowed), the second (never asked: a hit) and a region id. Regression caught: a scan that reads
    only the ask transcripts, so that what the model is shown after its last ask is never checked."""
    q1, q2 = (q["text"] for q in QUESTIONS[:2])
    log = write_log(tmp_path / "after.jsonl", [
        served(0, "task", "T0", via="prompt"), started("Task 0 begins at c.\nClean."),
        served(1, "frame", "[k0 t]", k=0), served(2, "ask", "Q1: " + q1, k=0, ask=ask_of("end", 0, Q1=QID_1)),
        called("submit_answer", "ok"), served(3, "frame", f"[k5 t] {q1} {q2} R7", k=5), served(4, "end", "over")])
    assert scan_log(log, questions=QUESTIONS[:2]) == [
        "call 5, message 4 (tool): region_id 'R7'",
        f"call 5, message 4 (tool): the text of question {QID_2} before its ask"]


def test_the_scan_is_silent_on_a_clean_run_and_on_the_model_repeating_its_own_question(tmp_path):
    """Contract (plan 4.9): a run whose items show a question only from its ask on (the first question's text holds the
    third's, so an ask of the first shows it too), whose asks hold only frames up to their step, whose model quotes a
    question it has not been asked yet in its own words, and whose replies hold nothing foreign, gives no hit.
    Regression caught: a scan that cries wolf on every ask (the first false alarm teaches people to ignore it)."""
    log = write_log(tmp_path / "clean.jsonl", [
        served(0, "task", "T0", via="prompt"), started("Task 0 begins at c.\nClean."),
        served(1, "frame", "[k0 t]", k=0), served(2, "frame", "[k1 t]", k=1),
        served(3, "ask", f"Q1: {QUESTIONS[0]['text']}", k=1, ask=ask_of("during", 1, Q1=QID_1)),
        called("next_frame", item=3), execrec(messages=[QUESTIONS[1]["text"], f"{QID_2}"]),
        served(4, "frame", "[k2 t]", k=2),
        served(5, "ask", f"Q2: {QUESTIONS[1]['text']}\nQ3: {QUESTIONS[2]['text']}", k=2,
               ask=ask_of("end", 2, Q2=QID_2, Q3=QID_3)), called("submit_answer", "ok")])
    assert scan_log(log, ["apartment_a"], questions=QUESTIONS, episode=EPISODE, dataset=tmp_path / "data") == []


def test_the_scans_frame_pattern_matches_what_the_renderer_writes(tmp_path):
    """Contract: the frame rule reads a frame's header line as `llm_render.frame_message` writes it, so frames 7 and 8
    have headers the scan finds; a frame mentioned mid-line is not a header; and the system prompt, which the first
    message carries, starts no line like one. Regression caught: a renderer change that the scan does not follow, which
    would turn the check for frames after an ask's step into one that silently finds nothing, or a prompt example that
    the scan reads as a frame."""
    steps = [{"k": k, "clock": f"2026-03-02T08:00:0{k}", "rgb": f"inbox/rgb/{k:06d}.jpg",
              "depth": f"inbox/depth/{k:06d}.png", "pose": {"pos": [2.0, 1.5, 0.0], "yaw_deg": 0.0},
              "camera": {"pitch_deg": 5.0}, "status": {"phase": "cleaning", "task": 0}} for k in (7, 8)]
    assert [HEADER.findall(llm_render.frame_message(step, "rgb_pose")[0]) for step in steps] == [["7"], ["8"]]
    assert HEADER.findall("see [k9 t] in the middle of a line") == []
    folder = tmp_path / "public"
    folder.mkdir()
    (folder / "camera.json").write_text(json.dumps({"width": 640, "height": 480, "hfov_deg": 80.0, "near_m": 0.01,
                                                    "far_m": 4.0}), encoding="utf-8")
    (folder / "robot.json").write_text(json.dumps(ROBOT), encoding="utf-8")
    for channel in ("rgb_pose", "detections"):
        assert HEADER.findall(llm_render.system_prompt(folder, channel)) == []


def test_a_torn_log_line_is_a_loud_error(tmp_path):
    """Contract: a log line that is not a JSON object stops `read_log`, naming the line. Regression: a silent skip that
    drops a paid call from replay and from the cost total."""
    log = tmp_path / "calls.jsonl"
    log.write_text('{"event": "x"}\n{"i": 0, "key": "abc\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_log(log)
