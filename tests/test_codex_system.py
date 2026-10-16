"""`CodexSystem`, the reference agent on the stream protocol (`codex_system.py`), over the fake command (`codex_fake`):
one Codex thread a case, one long exec in which the model pulls what the robot senses, driven by the calls the harness
makes (`task`, `observe`, `ask`, `finish`). No model, no `codex`, no network, and no `stream.run_case` (the end-to-end
run is `test_cmd_codex.py`; the pull's own contracts, early ends and kills are `test_codex_pull.py`).

The scenario fixture runs one case, a clean one, and the tests below read what it left (the fake's record of what the
model was shown and did, the journal) without changing it; the contracts that need a failure or a plan of their own run
their own short cases. Expected values are hand-written: the fake counts the keyframes it is shown and answers a count
or a position from how many it counted.
"""

from __future__ import annotations

import base64
import re
import time
from types import SimpleNamespace

import pytest

from codex_kit import CREDIT, SESSION, Run, jpeg, jpeg_hash, pulled
from stbench import stream_protocol
from stbench.agent_tools import ContextTooLong
from stbench.answers import PROMPT_SHAPES
from stbench.codex_exec import CodexLimit
from stbench.codex_feed import Feed
from stbench.codex_log import read_log
from stbench.codex_system import CodexSystem
from stbench.llm_scan import scan_log
from stbench.stream_protocol import TIMEOUTS, TransportError

QID_COUNT, QID_POS, QID_NUMBER = "a1b2c3d4e5f60718", "0f1e2d3c4b5a6978", "1122334455667788"
COUNT = {"qid": QID_COUNT, "text": "How many frames have you seen?", "answer_type": "count"}
POSITION = {"qid": QID_POS, "text": "Where is the robot now?", "answer_type": "position"}
NUMBER = {"qid": QID_NUMBER, "text": "How far have you been?", "answer_type": "number"}
BRIEF = "Your assignment is to clean up."
CASE_ID = "99aabbccddeeff00"
HEADER_K1 = ("[k1 2026-03-02T08:00:01] base at (5.60, 4.95, 0.00), heading -90.0 deg, camera pitch 5 deg; "
             "status: phase cleaning, task 0, battery_pct 97, brush true")
scenario = pytest.mark.xdist_group("codex-system-scenario")  # one worker builds the scenario once


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """One case on one thread: task 0, frames 0 and 1, a during ask of two questions, task 1, frame 2 and an end ask of
    one question; then `finish`. It runs on the default channel, the camera alone (the order of the images is
    `test_codex_pull.py`'s)."""
    with Run(tmp_path_factory.mktemp("case")) as run:
        system = run.system
        system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        run.stream([0, 1])
        during = system.ask("during", {"k": 1, "clock": "2026-03-02T08:00:01"}, [COUNT, POSITION])
        system.task(1, "2026-03-02T08:20:31", 612.0, "Next: carry on.")
        run.stream([2])
        end = system.ask("end", {"k": 2, "clock": "2026-03-02T08:00:02"}, [NUMBER])
        usage = system.finish()
    return SimpleNamespace(run=run, during=during, end=end, usage=usage, execs=run.fake.execs(),
                           records=[r for r in read_log(run.log) if "event" not in r],
                           events=[r for r in read_log(run.log) if "event" in r])


@scenario
def test_one_thread_per_case(case):
    """Contract (plan §4.9; recommendations §2.4): a case is one `start` exec, on one thread, and nothing is resumed
    when the model does not stop early; no exec is a fork, none is ephemeral, and the journal says the same.
    Regression caught: the retired fork route returning (design §12), a reset that starts a second thread, or an exec
    for every message coming back."""
    (only,) = case.execs
    assert only["sub"] is None and only["base"] is None and only["thread"] == "thread-0000" and only["finished"]
    assert "--ephemeral" not in only["argv"] and "fork" not in only["argv"]
    assert [(r["kind"], r["base"], r["thread"]) for r in case.records] == [("start", None, "thread-0000")]
    assert {r["session"] for r in case.records} == {SESSION} and not case.records[0]["killed"]
    assert [e["kind"] for e in case.events if e["event"] == "exec_started"] == ["start"]


@scenario
def test_the_first_message_carries_the_prompt_and_every_other_item_is_the_reply_to_a_pull(case):
    """Contract (plan §4.9; DECISIONS 2026-10-10; FD-frame-pull §3.3): the exec's message is the system prompt and the
    first task's message, nothing more (no memory folder listing, no cue), so the brief reaches the model in the task
    and never in the system prompt; everything after it is the reply to a `next_frame` call, in stream order: each
    frame's header, the questions of an ask in the same reply as the frame they are about (FD-questions-0.10 §1.5: the
    during ask after frame 1, the end ask after frame 2), the later task (its clock and the unobserved gap), the end.
    Regression caught: a system prompt sent again with each item, a first message that lists a memory folder the agent
    no longer has, an item that arrives in another order than the stream's, questions in an item of their own, or a
    cue that coaches it to keep notes."""
    (only,) = case.execs
    head, task = only["prompt"].split("\n\nTask 0 begins", 1)
    assert only["prompt"].startswith("You are the brain of a robot that is doing a job. ")
    assert task == " at 2026-03-02T08:00:00.\nYour assignment is to clean up." and BRIEF not in head
    replies = [c["text"] for c in pulled(only)]
    assert len(replies) == 5 and replies[0].startswith("[k0 2026-03-") and "Questions" not in replies[0]
    assert replies[1].splitlines()[0] == HEADER_K1 and replies[1].splitlines()[1].startswith(
        "Questions during the job, at the robot's latest frame (k1, 2026-03-02T08:00:01)")
    assert replies[2] == ("Task 1 begins at 2026-03-02T08:20:31.\n"
                          "The robot went unobserved for 612 s since the last task's final frame.\nNext: carry on.")
    assert replies[3].startswith("[k2 ") and replies[3].splitlines()[1].startswith("Questions after the job")
    assert replies[4] == "The job is over: there are no more frames."
    assert not [r for r in replies if "You are the brain of a robot" in r]


@scenario
def test_asks_use_labels_and_map_back(case):
    """Contract (plan §4.9; map HA-H14): an ask labels its questions Q1..Qn in the item and gives the tool server only
    the labels (it relays them), the model submits by label, and the answers come back by qid: the fake counted the
    frames it was shown (2 at the during ask, 3 at the end ask) and answered from them. No qid, case id or dataset path
    is in anything the model was sent, and the ask's served record names the labels' qids and the step. Regression
    caught: qids reaching the model (map HA-H14), or answers mapped to the wrong question."""
    assert case.during == {QID_COUNT: {"type": "count", "value": 2}, QID_POS: {"type": "position", "xy": [2.0, 0.0]}}
    assert case.end == {QID_NUMBER: {"type": "number", "value": 3}}
    (only,) = case.execs
    asks = [c["text"] for c in pulled(only) if "\nQuestions " in c["text"]]  # the questions come after their frame
    assert "Q1: How many frames have you seen?" in asks[0] and "Q2: Where is the robot now?" in asks[0]
    assert "Q1: How far have you been?" in asks[1]
    sent = "\n".join([only["prompt"], *(c["text"] for c in only["calls"])])
    secrets = (QID_COUNT, QID_POS, QID_NUMBER, CASE_ID, str(case.run.work.parent))  # ids, case, the case's folder
    assert not [token for token in secrets if token in sent]
    served = [e for e in case.events if e["event"] == "served" and e.get("ask")]
    assert [e["ask"] for e in served] == [{"when": "during", "now_k": 1, "labels": {"Q1": QID_COUNT, "Q2": QID_POS}},
                                          {"when": "end", "now_k": 2, "labels": {"Q1": QID_NUMBER}}]
    assert [(e["kind"], e["k"]) for e in served] == [("frame", 1), ("frame", 2)]  # each with the frame it is about
    outcomes = [(e["label"], e["outcome"]) for e in case.events if e["event"] == "answer_outcome"]
    assert outcomes == [("Q1", "submitted"), ("Q2", "submitted"), ("Q1", "submitted")]


@scenario
def test_finish_reports_the_cases_tokens_under_the_models_id(case):
    """Contract (plan §4.9): the case's tokens are reported under the model's id (the fake: one exec, 100 tokens in,
    40 of them cached and 10 out), with no peak context and no remote store; its keys are the stream's usage keys, and
    the exec's journal record holds the same share. Regression caught: a usage nobody can read, or tokens lost on the
    way from the exec to the report."""
    assert case.usage == {"models": {"fake-model": {"tokens_in": 100, "tokens_cached_in": 40,
                                                    "tokens_cache_write_in": 0, "tokens_out": 10}},
                          "peak_context": None, "remote_store_bytes": None}
    assert set(case.usage) <= set(stream_protocol.USAGE_KEYS)
    assert case.records[0]["usage"]["prompt_tokens"] == 100 and case.records[0]["messages"] == ["noted"]


@scenario
def test_the_journal_of_a_clean_case_scans_clean(case):
    """Contract (plan §4.9): the leak scan over the journal of a case the system ran without a fault gives no hit, with
    the case's questions, id and folder switched on. Regression caught: a scan that disagrees with what the system
    writes (the label lines, the frame headers), which would make every run invalid."""
    run = case.run
    questions = [COUNT, POSITION, NUMBER]
    assert scan_log(run.log, ["apartment_a"], questions=questions, episode=CASE_ID, dataset=run.work.parent) == []


@scenario
def test_each_frame_reaches_the_model_with_its_own_image_and_the_journal_holds_hashes(case):
    """Contract (FD-frame-pull §3.3; FD-questions-0.10 §1.5): on `rgb_pose` each frame's reply is a text block (its
    header) and one image block that is the step's JPEG, byte for byte (the fake names it by hash), and then, for a
    frame that comes with questions, a text block of them after the image; no other reply holds an image; the journal's
    served record of the frame holds the hash `<jpeg N bytes sha256 HEX16>` and the file holds no base64. Regression
    caught: base64 in the log, a frame without its picture, questions before the picture, or a picture on a task or an
    ask."""
    (only,) = case.execs
    frames = [c for c in pulled(only) if c["text"].startswith("[k")]
    assert [c["blocks"] for c in frames] == [["text", "image"], ["text", "image", "text"], ["text", "image", "text"]]
    assert [c["images"] for c in frames] == [[jpeg_hash(k)] for k in (0, 1, 2)]
    assert not [c for c in pulled(only) if c["images"] and not c["text"].startswith("[k")]
    served = [e for e in case.events if e["event"] == "served" and e["kind"] == "frame"]
    assert [e["images"][0].split(" ")[:3] for e in served] == [["<jpeg", "13", "bytes"]] * 3
    log = case.run.log.read_text(encoding="utf-8")
    assert not [k for k in (0, 1, 2) if base64.b64encode(jpeg(k)).decode() in log]


MUG = {"d": "9f3a01bc", "kind": "mug", "attributes": {"color": "red"}, "bbox_px": [120, 200, 180, 260],
       "center_cam": [0.1, -0.3, 1.5], "center_map": [2.0, 1.0, 0.9], "extent_m": [0.1, 0.1, 0.12]}


def test_on_detections_a_case_runs_on_text_alone_with_no_image(tmp_path):
    """Contract (the detections arm): started on `detections`, the system sends a frame as text, the header and its
    detections as lines, with no image block, depth image included, though the stream delivers the depth file; its first
    message tells the model the images are not sent and has no word of depth; the fake counts the frames from their
    header lines, so the answers are the same flow as on pictures; and the journal scans clean, the row's key `d` not
    among what the model was sent. Regression caught: a detections case that still sends images or reads the inbox, a
    prompt line about images the channel does not send, or the answer flow breaking for want of images."""
    with Run(tmp_path, channel="detections") as run:
        system = run.system
        system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        system.observe(run.batch([0], detections={0: [MUG]}))
        system.observe(run.batch([1]))
        got = system.ask("end", {"k": 1, "clock": "2026-03-02T08:00:01"}, [COUNT])
        system.finish()
    (only,) = run.fake.execs()
    replies = [c for c in pulled(only)]
    assert got == {QID_COUNT: {"type": "count", "value": 2}}
    assert [c["blocks"] for c in replies] == [["text"], ["text", "text"], ["text"]]  # frame 1 comes with the end ask
    assert [c["images"] for c in replies] == [[]] * 3
    assert replies[0]["text"].splitlines()[1] == "Detections in frame k0 (1):"
    assert replies[0]["text"].splitlines()[2].startswith(
        "  - mug (color red): map (2.00, 1.00, 0.90), camera (0.10, -0.30, 1.50), ")
    assert replies[1]["text"].splitlines()[1] == "Detections in frame k1: none"
    assert replies[1]["text"].splitlines()[2].startswith("Questions after the job")
    assert "9f3a01bc" not in replies[0]["text"]
    assert "You are not sent the images, only the detections in them" in only["prompt"]
    assert "grayscale" not in only["prompt"] and "depth" not in only["prompt"].lower()
    assert scan_log(run.log, ["apartment_a"], questions=[COUNT], episode=CASE_ID, dataset=run.work.parent) == []


def test_a_failed_exec_is_journaled_and_raised_as_a_transport_failure(tmp_path):
    """Contract (plan 4.9, 6.3): an exec that fails (a usage limit before its first pull, a full context in the middle
    of an ask) is noted in the journal with the word that counts it, its purpose and, for an ask, how many questions it
    took, and raised as `TransportError(reason)` chained to Codex's own error, which is what ends a case whatever
    message it was (`run_case` lets any other error of an ask fail only that ask). Regression caught: a failed ask
    that leaves the case running (the quota is spent on it), a stopped run that leaves no trace of why, or a context
    overflow counted as a generic error."""
    for name in ("limit", "context"):
        (tmp_path / name).mkdir()
    refused = pytest.raises(TransportError, match="^limit$")
    with Run(tmp_path / "limit", {0: {"fail": "limit"}}) as early, refused as limit:
        early.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
    assert isinstance(limit.value.__cause__, CodexLimit) and "usage limit" in str(limit.value.__cause__)
    with Run(tmp_path / "context", {0: {"pulls": 1, "fail_late": "context"}}) as late:
        late.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        with pytest.raises(TransportError, match="^context$") as context:
            late.system.ask("during", {"k": 1, "clock": "2026-03-02T08:00:01"}, [COUNT, POSITION])
    assert isinstance(context.value.__cause__, ContextTooLong)
    for run, want in ((early, ("task", "start", "limit", None)), (late, ("ask", "start", "context", 2))):
        failures = [e for e in read_log(run.log) if e.get("event") == "exec_failed"]
        assert [(e["purpose"], e["kind"], e["reason"], e.get("questions")) for e in failures] == [want]


@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_the_headline_agent_has_next_frame_and_submit_answer_and_no_other_tool_and_writes_no_file(tmp_path, channel):
    """Contract (DECISIONS 2026-10-10: the agent is offered no tool to help it in its task): on `rgb_pose` and
    `detections` the tool server the agent reaches lists exactly `next_frame` and `submit_answer`; the server is given
    the harness's address and key and nothing of a memory folder, a camera, a working folder or the case; and the case
    writes no file into its state folder. Regression caught: a memory tool or a perception tool given to the headline
    agent on any channel, or a state folder the agent is expected to fill."""
    with Run(tmp_path, channel=channel, asks_during=True, **CREDIT) as run:
        run.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        run.system.observe(run.batch([0], detections={0: [MUG]}))
        run.system.ask("during", {"k": 0, "clock": "2026-03-02T08:00:00"}, [COUNT])
        run.system.finish()
    (only,) = run.fake.execs()
    assert only["tools"] == ["next_frame", "submit_answer"] and set(only["config"]) == {"address", "key"}
    assert [p for p in run.state.rglob("*") if p.is_file()] == []


COACHING = re.compile(r"memory|remember|locate|keep|notes|shortened|map position", re.IGNORECASE)


def coaching(text: str) -> list[str]:
    """The words of `COACHING` in `text`, once the answer shapes are taken out: they are the parser's format (some say
    "a map position in metres" of an answer's field), not an instruction to the agent."""
    for shape in PROMPT_SHAPES.values():
        text = text.replace(shape, "")
    return [hit.group().lower() for hit in COACHING.finditer(text)]


@pytest.mark.parametrize("asks_during", [True, False])
@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_nothing_the_agent_is_sent_speaks_of_memory_or_coaches_what_to_keep(tmp_path, channel, asks_during):
    """Contract (DECISIONS 2026-10-10: the reference agent remembers with its own context; no instruction on what or
    when to remember; FD-frame-pull §3.3): nothing the model is sent in a case, the first message (system prompt and
    task), a later task, a frame, both asks, the nudge, the line that resumes a thread and the end, on every channel and
    for a job with and without during questions, holds "memory", "remember", "locate", "keep", "notes", "shortened" or
    "map position" in any case, bar the answer shapes (the brief is the benchmark's and shared by every system, so the
    case is given a neutral one), and the prompt has no word of a batch. Positive control: the same scan flags the
    retired scaffold's cue, its instruction on what to keep and its `locate` text. Regression caught: the memory-folder
    listing, the "Update your memory now." cue or a prompt bullet on what to keep coming back to the agent's inputs, or
    the retired batch wording."""
    for retired in ("Update your memory now.", "Your conversation may be shortened at any time to make room",
                    "locate things as you see them and keep their map positions", "Your memory folder holds:"):
        assert coaching(retired), retired
    plan = {0: {"submit": {"Q1": "none"}, "pulls": 3}}  # the ask is nudged, and the turn ends before the end item
    with Run(tmp_path, plan, channel=channel, asks_during=asks_during, **CREDIT) as run:
        run.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        run.system.observe(run.batch([0], detections={0: [MUG]}))
        run.system.ask("during", {"k": 0, "clock": "2026-03-02T08:00:00"}, [COUNT])
        run.system.task(1, "2026-03-02T08:20:31", 612.0, "Next: carry on.")
        run.system.finish()
    first, resumed = run.fake.execs()
    assert resumed["prompt"] == "Call next_frame to continue."
    texts = [first["prompt"], resumed["prompt"], *(c["text"] for e in (first, resumed) for c in e["calls"])]
    assert any(t.startswith("error: you have not answered Q1.") for t in texts)  # the nudge was sent
    assert any(t.startswith("Task 1 begins") for t in texts) and any(t.startswith("The job is over") for t in texts)
    assert [(n, found) for n, text in enumerate(texts) if (found := coaching(text))] == []
    assert not [n for n, text in enumerate(texts) if re.search(r"batch", text, re.IGNORECASE)]
    assert ("at some points during the job and after it" in first["prompt"]) == asks_during
    assert "decision point" not in first["prompt"] and "Credit is four quarters that add up." in first["prompt"]


def test_each_message_has_its_own_timeout_and_the_end_its_own_wait(tmp_path, monkeypatch):
    """Contract (plan 4.9; `stream_protocol.TIMEOUTS`; FD-frame-pull §3.3; FD-questions-0.10 §1.5): the model has the
    timeout the stream protocol allows the message to pull again: 600 s after a task, a frame served alone and a
    during ask (the frame it comes with is served with the ask and has the ask's timeout, and no wait of its own),
    3,600 s after the end ask, and after the end item `finish_wait_s` (60 s) for the turn to complete. Here (the frame
    and the during-ask timeouts set apart, 100 s and 200 s): the task, frame 0 served alone by frame 1, frame 1 with the
    during ask, and the end ask (alone: its frame was served by then). Regression caught: the end ask killed by the
    600 s that fits a frame (a long end ask is where the case's answers are), a frame with questions timed as a plain
    frame, or one timeout for every message."""
    waits: list[tuple[int, bool]] = []
    real = Feed.wait

    def spy(self, ticket, deadline, alive, *, pull=True):
        waits.append((round(deadline - time.monotonic()), pull))
        return real(self, ticket, deadline, alive, pull=pull)

    monkeypatch.setattr(Feed, "wait", spy)
    with Run(tmp_path, timeouts={**TIMEOUTS, "observe": 100.0, "ask_during": 200.0}) as run:
        system = run.system
        assert TIMEOUTS["ask_end"] == 3600.0 and TIMEOUTS["task"] == 600.0  # the numbers below are the protocol's
        system.task(0, "c", 0.0, BRIEF)
        run.stream([0, 1])
        system.ask("during", {"k": 1, "clock": "c"}, [COUNT])
        system.ask("end", {"k": 1, "clock": "c"}, [COUNT])
        system.finish()
    assert waits == [(600, True), (100, True), (200, True), (3600, True), (60, False)]
    assert isinstance(run.system, CodexSystem) and run.system.channels == ("rgb_pose", "detections")


def test_the_stream_sends_one_frame_an_observe_and_a_longer_list_is_refused(tmp_path):
    """Contract (FD-frame-pull §3.1): the reference agent is sent one step an `observe`; a list of two is a ValueError
    before anything is sent to the model. Regression caught: batches coming back into the agent, which would show it two
    frames in one reply."""
    with Run(tmp_path) as run:
        run.system.task(0, "c", 0.0, BRIEF)
        with pytest.raises(ValueError, match="one frame at a time, not 2"):
            run.system.observe(run.batch([0, 1]))
        (only,) = run.fake.execs()
        assert not [c for c in pulled(only) if c["text"].startswith("[k")]
