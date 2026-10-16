"""The pull (`codex_system`, `codex_feed`, `codex_live`): the reference agent calls `next_frame` inside one long exec,
and the harness answers with one item at a time. Over the fake command (`codex_fake`), whose "model" pulls, answers
the asks and, by plan, ends its turn early, hangs, pulls too early, pulls twice at once or uses a tool it should not.
No model, no `codex`, no network.

What is pinned here is the order in which the model is shown things (a frame is held until the next message shows what
goes with it: `test_codex_held_frame.py`), what happens when it does not pull, and what the journal and the leak scan
make of it; the rules of an ask's table are `test_codex_asks.py`'s and the feed's own `test_codex_feed.py`'s. Expected
values are written by hand from the fake's rules (`codex_fake`'s docstring). `Run.stream` leaves the last frame held:
`Run.serve` serves it, as the next message would.
"""

from __future__ import annotations

import json
import time

import pytest

from codex_kit import Run, jpeg_hash, pulled
from stbench import codex_exec
from stbench.codex_exec import CodexError, CodexTimeout, ToolEscaped
from stbench.codex_feed import BUSY
from stbench.codex_log import read_log
from stbench.codex_system import Settings
from stbench.llm_scan import scan_log
from stbench.llm_summary import summarize
from stbench.stream_protocol import TIMEOUTS, TransportError

QID_1, QID_2 = "a1b2c3d4e5f60718", "0f1e2d3c4b5a6978"
COUNT = {"qid": QID_1, "text": "How many frames have you seen?", "answer_type": "count"}
POSITION = {"qid": QID_2, "text": "Where is the robot now?", "answer_type": "position"}
BRIEF = "Your assignment is to clean up."
FIRST = "2026-03-02T08:00:00"


def kinds(record: dict) -> list[str]:
    """What an exec's calls were, in order, by the first words of each reply: the items (`task`, `frame`, `frame+ask`
    for a frame that comes with questions, `ask`, `end`), `nudge` (the error a premature `next_frame` gets), `busy`,
    and for `submit_answer` `ok`, `again` (the label was done) or `error`."""
    found = []
    for call in record["calls"]:
        text = call["text"]
        if call["tool"] == "submit_answer":
            found.append("ok" if text == "ok" else "again" if "already" in text else "error")
        elif text.startswith("[k") and "\nQuestions " in text:
            found.append("frame+ask")
        else:
            found.append(next((name for name, start in (("frame", "[k"), ("ask", "Questions"), ("task", "Task"),
                                                        ("end", "The job is over"),
                                                        ("nudge", "error: you have not answered"),
                                                        ("busy", BUSY)) if text.startswith(start)), "other"))
    return found


def frames(record: dict) -> list[str]:
    """The headers' `[kN` of the frames an exec has been shown, in order."""
    return [c["text"].split(" ")[0] for c in pulled(record) if c["text"].startswith("[k")]


def test_the_model_is_never_shown_frame_k_plus_one_before_it_pulls_again(tmp_path):
    """Contract (FD-frame-pull §3.3; FD-questions-0.10 §1.5): after `observe` of frame k has returned, the model has
    been shown frames 0..k-1 and no more (frame k is held until the next message shows what goes with it), with its
    next pull open and the exec still running; the stream does not run ahead of the model. Serving the held frame
    shows frame k and nothing later. Regression caught: a feed that serves the next frame before the model has asked
    for it (frames queued behind a pull), which would show the model frames at the speed of the harness and not its
    own."""
    with Run(tmp_path) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        for k in range(4):
            run.system.observe(run.batch([k]))
            (live,) = run.fake.execs()
            assert frames(live) == [f"[k{n}" for n in range(k)] and not live["finished"]
            assert not (run.work / "inbox" / "rgb" / f"{k + 1:06d}.jpg").exists()  # the next frame is not even on disk
        run.serve()
        (live,) = run.fake.execs()
        assert frames(live) == [f"[k{n}" for n in range(4)] and not live["finished"]


ASKS = [
    pytest.param({}, ["frame+ask", "ok", "again", "ok", "again", "frame", "end"], 0,
                 [("submitted", None), ("submitted", None)], id="answered"),
    pytest.param({"premature": 1}, ["frame+ask", "nudge", "ok", "again", "ok", "again", "frame", "end"], 1,
                 [("repaired", None), ("repaired", None)], id="one early pull, then answered"),
    pytest.param({"premature": 2}, ["frame+ask", "nudge", "frame", "end"], 2,
                 [("failed", "no_submit"), ("failed", "no_submit")], id="two early pulls"),
    pytest.param({"submit": {"Q2": "none"}}, ["frame+ask", "ok", "again", "nudge", "frame", "end"], 2,
                 [("submitted", None), ("failed", "no_submit")], id="one never answered"),
    pytest.param({"submit": {"Q2": "late"}}, ["frame+ask", "ok", "again", "nudge", "ok", "again", "frame", "end"], 1,
                 [("submitted", None), ("repaired", None)], id="one answered after the nudge"),
]


@pytest.mark.parametrize(("plan", "sequence", "premature", "outcomes"), ASKS)
def test_an_asks_questions_come_with_its_frame_and_no_frame_follows_until_it_is_done(
        tmp_path, plan, sequence, premature, outcomes):
    """Contract (FD-questions-0.10 §1.5): the questions of an ask at the held frame's step are in the same item as that
    frame (k1 here: its header line and image, then the ask's text); then no frame comes until every label is answered,
    or the second premature pull fails what is left `no_submit` (the first is refused with an error naming the labels
    left); the answers come back by qid, with the outcome the journal notes (`repaired` when a nudge came first). The
    frame after the ask (k2) is held through all of it and is served only by the message that follows (`finish`), the
    end item after it. Hand-written sequences of the model's calls for five plans, from the frame with the questions
    on. Regression caught: questions in an item of their own (the model pays a pull for each ask), questions leaking
    into the stream before their frame, a stream that skips an ask whose questions were not answered, a nudge that
    never ends, or the next frame served before the questions are done."""
    with Run(tmp_path, {0: plan}) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0, 1])
        got = run.system.ask("during", {"k": 1, "clock": "2026-03-02T08:00:01"}, [COUNT, POSITION])
        run.stream([2])
        assert frames(run.fake.execs()[0]) == ["[k0", "[k1"]  # frame 2 is held, not shown, though the ask is over
        run.system.finish()
        (live,) = run.fake.execs()
    assert kinds(live) == ["frame", *sequence]
    asked = next(c for c in live["calls"] if "\nQuestions " in c["text"])
    header, ask = asked["text"].split("\n", 1)
    assert header.startswith("[k1 ") and asked["blocks"] == ["text", "image", "text"]
    assert asked["images"] == [jpeg_hash(1)]
    assert ask.startswith("Questions during the job, at the robot's latest frame (k1, ")
    expected = {QID_1: {"type": "count", "value": 2}, QID_2: {"type": "position", "xy": [2.0, 0.0]}}
    failed = {"type": "failed", "reason": "no_submit"}
    assert got == {qid: expected[qid] if outcomes[n][0] != "failed" else failed for n, qid in enumerate(expected)}
    log = read_log(run.log)
    assert [(e["outcome"], e.get("reason")) for e in log if e.get("event") == "answer_outcome"] == outcomes
    summary = summarize(run.log)
    assert (summary["frames"], summary["frames_asked"], summary["premature_pulls"]) == (3, 1, premature)


def test_an_early_turn_end_is_resumed_with_the_fixed_line_on_the_same_thread(tmp_path):
    """Contract (FD-frame-pull §3.3): when the exec's turn ends before the end item while the system waits for a pull,
    the thread is resumed with one fixed line, "Call next_frame to continue.", the model pulls again and the stream
    goes on: two execs, the second a resume of the first's thread, no other prompt; the frames the model counted
    include those it saw before (the thread keeps them); the tokens are differenced across the resume (the fake prints
    100 in, 40 cached and 10 out for the first exec and a running total of 200, 80 and 20 for the second), so the
    case's tokens are 200, 80 and 20 and not 300 (the totals added). Regression caught: a silent hang on a model that
    stops pulling, an exec per message coming back, or tokens counted twice across a resume."""
    with Run(tmp_path, {0: {"pulls": 2}}) as run:  # the first exec ends after frame 1
        system = run.system
        system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0, 1, 2])
        got = system.ask("end", {"k": 2, "clock": "2026-03-02T08:00:02"}, [COUNT])
        usage = system.finish()
    first, second = run.fake.execs()
    assert (first["sub"], second["sub"], second["base"], second["thread"]) == (None, "resume", "thread-0000",
                                                                               "thread-0000")
    assert second["prompt"] == "Call next_frame to continue." and frames(first) == ["[k0", "[k1"]
    assert frames(second) == ["[k2"] and first["finished"] and second["finished"]
    assert got == {QID_1: {"type": "count", "value": 3}}
    records = [r for r in read_log(run.log) if "event" not in r]
    assert [(r["kind"], r["base"], r["usage"]["prompt_tokens"], r["thread_usage"]["prompt_tokens"])
            for r in records] == [("start", None, 100, 100), ("resume", "thread-0000", 100, 200)]
    assert usage["models"]["fake-model"] == {"tokens_in": 200, "tokens_cached_in": 80, "tokens_cache_write_in": 0,
                                             "tokens_out": 20}
    summary = summarize(run.log)
    assert summary["resumes"] == 1 and summary["execs"]["by_kind"] == {"start": 1, "resume": 1}


@pytest.mark.parametrize(("limit", "execs"), [(3, 4), (1, 2)])
def test_the_fourth_early_end_in_a_row_is_stalled(tmp_path, limit, execs):
    """Contract (FD-frame-pull §3.3): resumes are counted, and an exec that ends again after `max_resumes` (3) in a row
    without a pull in between ends the case as `TransportError("stalled")` (so four execs, and two when the limit is 1),
    journaled as a failed exec with that word, with every exec read and journaled. Regression caught: a loop with no
    bound on a model that never pulls, or a case that gives up after the first stop."""
    plan = {0: {"pulls": 1}, 1: {"pulls": 0}, 2: {"pulls": 0}, 3: {"pulls": 0}}
    with Run(tmp_path, plan, settings=Settings(max_resumes=limit)) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.system.observe(run.batch([0]))  # held: the next message serves it
        with pytest.raises(TransportError, match="^stalled$") as stalled:
            run.serve()
    cause = stalled.value.__cause__
    assert isinstance(cause, CodexError) and "without calling next_frame" in str(cause)
    assert [e["sub"] for e in run.fake.execs()] == [None] + ["resume"] * (execs - 1)
    log = read_log(run.log)
    assert [(e["purpose"], e["reason"]) for e in log if e.get("event") == "exec_failed"] == [("observe", "stalled")]
    assert len([r for r in log if "event" not in r]) == execs
    assert summarize(run.log)["exec_failures"] == {"stalled": 1}


def test_a_pull_that_does_not_come_in_time_kills_the_exec_as_a_timeout(tmp_path):
    """Contract (FD-frame-pull §3.3): from serving an item to the model's next pull the wait is bounded by the message's
    timeout (here 3 s for a frame); past it the exec is killed with everything it started and the case ends as
    `TransportError("timeout")`, chained to a `CodexTimeout`; the exec's record is journaled as killed, with no usage,
    and the failure is counted. Regression caught: a case that never ends (a model that stalls inside a frame), or a
    kill that leaves the exec running."""
    with Run(tmp_path, {0: {"hang": 1}}, timeouts={**TIMEOUTS, "observe": 3.0}) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.system.observe(run.batch([0]))  # held: the wait is from serving it
        began = time.monotonic()
        with pytest.raises(TransportError, match="^timeout$") as timeout:
            run.serve()
    assert 2.5 < time.monotonic() - began < 40 and isinstance(timeout.value.__cause__, CodexTimeout)
    (live,) = run.fake.execs()
    assert frames(live) == ["[k0"] and not live["finished"]
    log = read_log(run.log)
    (record,) = [r for r in log if "event" not in r]
    assert record["killed"] is True and record["thread_usage"]["prompt_tokens"] == 0
    assert [(e["purpose"], e["reason"]) for e in log if e.get("event") == "exec_failed"] == [("observe", "timeout")]
    assert summarize(run.log)["tokens_lower_bound"] is True


def test_an_ask_that_times_out_fails_its_questions_with_the_exec(tmp_path):
    """Contract (FD-frame-pull §3.3): an ask the model answers but never moves on from (it submitted, then hung) ends
    the case at the ask's timeout, even though a label was answered: the ask is over only at the pull. The exec is
    journaled failed as a `timeout` with the number of questions the ask took, and the summary counts those questions
    as timed out. Regression caught: an ask taken as complete because labels were submitted (a hung exec would run to
    the end of the case's twelve hours), or a timeout that does not say which questions it took."""
    with Run(tmp_path, {0: {"hang": 1}}, timeouts={**TIMEOUTS, "ask_during": 3.0}) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        with pytest.raises(TransportError, match="^timeout$"):
            run.system.ask("during", {"k": 0, "clock": FIRST}, [COUNT, POSITION])
    (live,) = run.fake.execs()
    assert kinds(live) == ["ask", "ok", "again", "ok", "again"] and not live["finished"]
    failed = [e for e in read_log(run.log) if e.get("event") == "exec_failed"]
    assert [(e["purpose"], e["reason"], e["questions"]) for e in failed] == [("ask", "timeout", 2)]
    assert summarize(run.log)["answers"]["timed_out"] == 2


def test_an_escaped_tool_event_kills_the_exec_when_it_arrives_and_not_at_the_end(tmp_path):
    """Contract (FD-frame-pull §3.3): every event is checked as it arrives: a command item that comes after frame 0,
    when the exec would then sleep a minute, kills the exec at once and ends the case as `TransportError("escaped")`,
    chained to `ToolEscaped`, journaled so that the run is invalid. Regression caught: an escape found only when the
    exec ends, which with one exec for a whole case lets a model use a tool for the rest of the case."""
    plan = {0: {"escape": "command_execution", "escape_at": 1, "hang": 1, "hang_s": 60}}
    with Run(tmp_path, plan) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.system.observe(run.batch([0]))  # held: the next message serves it
        began = time.monotonic()
        with pytest.raises(TransportError, match="^escaped$") as escaped:
            run.serve()
    assert time.monotonic() - began < 40 and isinstance(escaped.value.__cause__, ToolEscaped)
    (live,) = run.fake.execs()
    assert not live["finished"]
    failed = [e for e in read_log(run.log) if e.get("event") == "exec_failed"]
    assert [(e["purpose"], e["reason"]) for e in failed] == [("observe", "escaped")]
    assert "command_execution" in failed[0]["message"]


def test_a_second_pull_at_once_is_refused_and_the_stream_goes_on(tmp_path):
    """Contract (FD-frame-pull §3.3): the feed serves one `next_frame` at a time; when the model makes two at once (here
    through two tool servers) exactly one is refused with the error and the other waits for the frame, which is then
    delivered once. Regression caught: two pulls that both take items (a frame lost or shown twice), or a refused call
    that disturbs the pull that waits."""
    with Run(tmp_path, {0: {"double": True}}) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        limit = time.monotonic() + 30
        while time.monotonic() < limit and "busy" not in kinds(run.fake.execs()[0]):
            time.sleep(0.05)  # both calls are made once the refusal is logged; the frame is not supplied before
        run.stream([0])
        run.serve()
        (live,) = run.fake.execs()
    assert sorted(kinds(live)[:2]) == ["busy", "frame"] and frames(live) == ["[k0"]
    assert next(c["text"] for c in live["calls"] if c["text"].startswith("error")) == BUSY


def test_one_transient_failure_before_the_model_acts_is_retried_once_after_a_wait(tmp_path):
    """Contract (plan 6: one bounded retry with a backoff): a 5xx before the model did anything is tried once more after
    `RETRY_WAIT` seconds (the transport's sleep, injected here), the retried exec is the case's start (a new thread),
    the record says it took 2 attempts, and the journal's second `exec_started` is marked attempt 2 and adds nothing
    to the conversation; a failure after the model made a call is not retried. Regression caught: a retry loop, no
    wait before the retry, or a retry that repeats a model's calls."""
    for name in ("early", "late"):
        (tmp_path / name).mkdir()
    with Run(tmp_path / "early", {0: {"fail": "flaky"}}) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0])
        run.serve()
    assert run.fake.waits == [codex_exec.RETRY_WAIT] and [e["n"] for e in run.fake.execs()] == [0, 1]
    records = [r for r in read_log(run.log) if "event" not in r]
    assert [(r["kind"], r["attempts"]) for r in records] == [("start", 2)]  # closed with the case, unfinished
    started = [(e["kind"], e["attempt"]) for e in read_log(run.log) if e.get("event") == "exec_started"]
    assert started == [("start", 1), ("start", 2)] and frames(run.fake.execs()[1]) == ["[k0"]
    with Run(tmp_path / "late", {0: {"pulls": 1, "fail_late": "flaky"}}) as other:
        other.system.task(0, FIRST, 0.0, BRIEF)
        other.system.observe(other.batch([0]))
        with pytest.raises(TransportError, match="^error$"):
            other.serve()
    assert other.fake.waits == [] and [e["n"] for e in other.fake.execs()] == [0]


def test_a_turn_that_does_not_complete_after_the_end_is_killed_and_the_count_is_a_lower_bound(tmp_path):
    """Contract (FD-frame-pull §3.3): `finish` gives the end item and then the turn up to `finish_wait_s` to complete
    (2 s here, 60 s by default) and kills it otherwise; the case's answers stand and `finish` does not raise; the killed
    turn printed no usage, so its tokens are missing and the summary says the count is a lower bound. Regression caught:
    a case that waits for ever on a turn that will not end, or a missing count passed off as a complete one."""
    with Run(tmp_path, {0: {"linger": 60}}, settings=Settings(finish_wait_s=2.0)) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0])
        began = time.monotonic()
        usage = run.system.finish()
    assert 1.5 < time.monotonic() - began < 40
    assert usage["models"]["fake-model"] == {"tokens_in": 0, "tokens_cached_in": 0, "tokens_cache_write_in": 0,
                                             "tokens_out": 0}
    (record,) = [r for r in read_log(run.log) if "event" not in r]
    assert record["killed"] is True and not run.fake.execs()[0]["finished"]
    assert kinds(run.fake.execs()[0]) == ["frame", "end"]
    assert summarize(run.log)["tokens_lower_bound"] is True


def test_with_text_only_the_frames_are_headers_and_no_image_block_goes_out(tmp_path):
    """Contract (FD-frame-pull §3.3; plan A8): with `text_only` on `rgb_pose` each frame reply is the header line alone,
    though the stream delivered the JPEG, and the first message tells the model it is not sent images; the same case
    without the option sends each frame's JPEG as an image block, byte for byte. Regression caught: an image on a text
    arm (the zero-shot check would measure the pictures), or no image on the picture arm."""
    seen = {}
    for name, opts in (("text", {"text_only": True}), ("camera", {})):
        (tmp_path / name).mkdir()
        with Run(tmp_path / name, **opts) as run:
            run.system.task(0, FIRST, 0.0, BRIEF)
            run.stream([0, 1])
            run.serve()
            (live,) = run.fake.execs()
        seen[name] = live
    assert [c["blocks"] for c in pulled(seen["text"])] == [["text"], ["text"]]
    assert [c["images"] for c in pulled(seen["text"])] == [[], []]
    assert "In this run you are not sent the images: a frame is its header line alone." in seen["text"]["prompt"]
    assert [c["images"] for c in pulled(seen["camera"])] == [[jpeg_hash(0)], [jpeg_hash(1)]]
    assert [c["text"] for c in pulled(seen["text"])] == [c["text"] for c in pulled(seen["camera"])]


def test_the_journal_of_a_clean_case_scans_clean_and_a_planted_future_frame_is_found(tmp_path):
    """Contract (FD-frame-pull §3.4; FD-questions-0.10 §1.5): the leak scan reads the conversation rebuilt from the
    served items, and an ask now starts inside a frame item (the frame, then the questions, one message): a clean case's
    journal has no hit; a frame header above an ask's step planted in an item served before that ask is found, in the
    transcript of each ask that holds it (here k9 in the first frame, found for the ask at k1 and for the end ask at
    k2), and so is one planted in the frame item that carries the ask (its own message, in its own transcript); the text
    of a question not yet asked planted in that frame item is found as shown before its ask, while the item's own
    questions in it are not. The summary counts the two frames that came with questions and times them apart from the
    plain one. Regression caught: a scan that reads only the old exec records and so finds nothing in a log of pulls, a
    scan blind to the frame part of a joined item, or one that cries wolf on the questions joined to a frame."""
    with Run(tmp_path) as run:
        system = run.system
        system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0, 1])
        system.ask("during", {"k": 1, "clock": "2026-03-02T08:00:01"}, [COUNT])
        run.stream([2])
        system.ask("end", {"k": 2, "clock": "2026-03-02T08:00:02"}, [POSITION])
        system.finish()
    questions = [COUNT, POSITION]
    assert scan_log(run.log, ["apartment_a"], questions=questions, dataset=run.work.parent) == []
    summary = summarize(run.log)
    assert (summary["frames"], summary["frames_asked"], summary["pulls"]) == (3, 2, 4)
    assert (summary["premature_pulls"], summary["resumes"]) == (0, 0)
    assert (summary["frame_latency"]["n"], summary["asked_latency"]["n"]) == (1, 2)
    assert summary["tool_calls"] == {"next_frame": 4, "submit_answer": 4}
    served = [r for r in read_log(run.log) if r.get("event") == "served" and r["kind"] == "frame"]
    assert [bool(r.get("asked")) for r in served] == [False, True, True]
    assert served[1]["ask"] == {"when": "during", "now_k": 1, "labels": {"Q1": QID_1}}
    assert served[1]["text"].startswith("[k1 ") and served[1]["asked"].startswith("Questions during")
    header = "\n[k9 2026-03-02T08:00:09] base at (0.00, 0.00, 0.00), planted"
    for name, which, plant, hits in (
            ("plain", 0, header, ["call 2, message 1 (tool): frame k9 after the ask's step k1",
                                  "call 3, message 1 (tool): frame k9 after the ask's step k2"]),
            ("joined", 1, header, ["call 2, message 2 (tool): frame k9 after the ask's step k1",
                                   "call 3, message 2 (tool): frame k9 after the ask's step k2"]),
            ("question", 1, "\n" + POSITION["text"],
             [f"call 2, message 2 (tool): the text of question {QID_2} before its ask"])):
        records = read_log(run.log)
        row = next(r for r in records if r.get("event") == "served" and r["kind"] == "frame" and r["k"] == which)
        row["text"] += plant
        planted = tmp_path / f"planted-{name}.jsonl"
        planted.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        assert scan_log(planted, [], questions=questions) == hits, name
