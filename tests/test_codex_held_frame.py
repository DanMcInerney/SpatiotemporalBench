"""The held frame (`codex_system`, FD-questions-0.10 §1.5): `observe` holds the frame and the harness's next message
shows what goes with it. An ask at the frame's step is served in the frame's item; an `observe`, a `task` or `finish`
serves the frame alone first. Over the fake command (`codex_fake`), no model, no `codex`, no network. The model's view
of an ask that joins a frame (the nudge, the second premature pull, the answers) is `test_codex_pull.py`'s `ASKS`; the
timeouts of each kind of item are `test_codex_system.py`'s. Expected values are written by hand from the fake's rules.
"""

from __future__ import annotations

import time

import pytest

from codex_kit import Run, jpeg_hash
from stbench.codex_log import read_log
from stbench.llm_summary import summarize
from stbench.stream_protocol import TIMEOUTS, TransportError
from test_codex_pull import BRIEF, COUNT, FIRST, POSITION, QID_1, QID_2, frames, kinds

CLOCK_1 = "2026-03-02T08:00:01"


def served(run: Run) -> list[dict]:
    return [e for e in read_log(run.log) if e.get("event") == "served"]


def test_a_frame_followed_by_anything_but_its_ask_is_served_alone_once_and_in_order(tmp_path):
    """Contract (FD-questions-0.10 §1.5): a frame the harness does not ask about is served alone, as it was before
    frames were held: by the next `observe`, by a `task` (before the task message), or by `finish` (before the end
    item). Every frame is served exactly once, in the stream's order, and none is lost or late: here frames 0 and 1 by
    the next frames, frame 2 by the task, frame 3 by `finish`, so the model is shown frame, frame, frame, task, frame,
    end; none carries questions, and each is timed to its pull. Regression caught: a frame held for ever (lost at a
    task or at the end), one served after the message that followed it (the model sees the new task before the frame
    it ends), or a frame served twice."""
    with Run(tmp_path) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0, 1, 2])
        assert frames(run.fake.execs()[0]) == ["[k0", "[k1"]  # frame 2 is held
        run.system.task(1, "2026-03-02T08:20:31", 612.0, "Next: carry on.")
        assert frames(run.fake.execs()[0]) == ["[k0", "[k1", "[k2"]
        run.stream([3])
        assert frames(run.fake.execs()[0]) == ["[k0", "[k1", "[k2"]  # frame 3 is held
        run.system.finish()
        (live,) = run.fake.execs()
    assert kinds(live) == ["frame", "frame", "frame", "task", "frame", "end"]
    assert [(e["kind"], e["k"], "asked" in e) for e in served(run)] == [
        ("task", None, False), ("frame", 0, False), ("frame", 1, False), ("frame", 2, False), ("task", None, False),
        ("frame", 3, False), ("end", None, False)]
    summary = summarize(run.log)
    assert (summary["frames"], summary["frames_asked"], summary["frame_latency"]["n"]) == (4, 0, 4)


def test_the_end_ask_joins_the_last_frame_when_it_is_held(tmp_path):
    """Contract (FD-questions-0.10 §1.5): the end ask, sent after the last frame, is served with that frame when the
    frame is still held: one item, the frame's header and image and then the end questions; the model has been shown
    the frame once. Regression caught: the end ask in an item of its own after the last frame (a pull more at the end of
    every case), or the last frame served alone first and again with the ask."""
    with Run(tmp_path) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0, 1])
        got = run.system.ask("end", {"k": 1, "clock": CLOCK_1}, [COUNT])
        run.system.finish()
        (live,) = run.fake.execs()
    assert kinds(live) == ["frame", "frame+ask", "ok", "again", "end"] and frames(live) == ["[k0", "[k1"]
    joined = live["calls"][1]
    assert joined["text"].splitlines()[1].startswith("Questions after the job, at the robot's last frame (k1, ")
    assert joined["blocks"] == ["text", "image", "text"] and joined["images"] == [jpeg_hash(1)]
    assert got == {QID_1: {"type": "count", "value": 2}}
    assert [(e["kind"], e["k"], e.get("ask", {}).get("when")) for e in served(run)][1:] == [
        ("frame", 0, None), ("frame", 1, "end"), ("end", None, None)]


def test_a_second_ask_at_the_same_step_is_an_item_of_its_own(tmp_path):
    """Contract (FD-questions-0.10 §1.5): the stream sends a point's asks one after another, each returning when the
    model has answered, so each `ask` after a held frame joins that frame if it is still held and is otherwise its own
    item: the first of two asks at one step comes with the frame, the second after it, alone (the model has already seen
    the frame, so its count of frames has not grown). Regression caught: the second ask joined to a frame already
    served (the frame shown twice), or lost for want of a frame to join."""
    with Run(tmp_path) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0, 1])
        first = run.system.ask("during", {"k": 1, "clock": CLOCK_1}, [COUNT])
        second = run.system.ask("during", {"k": 1, "clock": CLOCK_1}, [POSITION])
        run.system.finish()
        (live,) = run.fake.execs()
    assert kinds(live) == ["frame", "frame+ask", "ok", "again", "ask", "ok", "again", "end"]
    assert frames(live) == ["[k0", "[k1"] and live["calls"][4]["blocks"] == ["text"]
    assert first == {QID_1: {"type": "count", "value": 2}}
    assert second == {QID_2: {"type": "position", "xy": [2.0, 0.0]}}
    assert [(e["kind"], bool(e.get("asked"))) for e in served(run)][1:] == [
        ("frame", False), ("frame", True), ("ask", False), ("end", False)]
    assert summarize(run.log)["frames_asked"] == 1


def test_an_ask_at_another_step_than_the_held_frames_is_not_joined_to_it(tmp_path):
    """Contract (FD-questions-0.10 §1.5): an ask joins the frame it is about, the one at its own step (`now["k"]`); an
    ask at another step serves the held frame alone first and is then its own item, so a frame is never labelled with
    questions about another. Regression caught: questions glued to the nearest frame whatever their step, whose text
    says "at the robot's latest frame (k0)" under a header that says k1."""
    with Run(tmp_path) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0, 1])
        run.system.ask("during", {"k": 0, "clock": FIRST}, [COUNT])
        (live,) = run.fake.execs()
    assert kinds(live) == ["frame", "frame", "ask", "ok", "again"]
    assert not [e for e in served(run) if e["kind"] == "frame" and e.get("asked")]


@pytest.mark.parametrize("quick", [True, False])
def test_a_held_frames_deadline_starts_when_it_is_served_not_when_it_was_held(tmp_path, quick):
    """Contract (FD-questions-0.10 §1.5): the `observe` timeout counts from serving the frame. With a 2 s timeout, a
    frame held for 3 s and then served to a model that pulls at once is served without a timeout; and one served to a
    model that hangs gets its whole 2 s from serving (more than 1.8 s after the serving message began), not what was
    left of 2 s after the 1.2 s it was held. Regression caught: a deadline taken at `observe`, which would kill the
    case whenever the harness is slow between messages (an ask that takes long to build, the stream's own copying), or
    one that cuts the model's time short by the wait."""
    plan = {} if quick else {0: {"hang": 1, "hang_s": 60}}
    with Run(tmp_path, plan, timeouts={**TIMEOUTS, "observe": 2.0}) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.system.observe(run.batch([0]))
        time.sleep(3.0 if quick else 1.2)  # the frame is held
        began = time.monotonic()
        if quick:
            run.serve()
            assert time.monotonic() - began < 2.0
            assert frames(run.fake.execs()[0]) == ["[k0"]
        else:
            with pytest.raises(TransportError, match="^timeout$"):
                run.serve()
            assert 1.8 < time.monotonic() - began < 40


def test_the_questions_of_a_frame_that_timed_out_with_its_item_are_counted_as_timed_out(tmp_path):
    """Contract (FD-questions-0.10 §1.5): the item that carries a frame and its ask is the ask's, so when the model
    answers and then never pulls, the case ends at the ask's timeout (not the frame's), the exec failure is journaled
    for the ask with the number of its questions, and the summary counts those questions as timed out. Regression
    caught: a joined item timed or journaled as a plain frame (its questions would be lost from `timed_out`, and the
    frame's shorter timeout would apply to a model that is answering)."""
    with Run(tmp_path, {0: {"hang": 1}}, timeouts={**TIMEOUTS, "observe": 60.0, "ask_during": 3.0}) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0])
        began = time.monotonic()
        with pytest.raises(TransportError, match="^timeout$"):
            run.system.ask("during", {"k": 0, "clock": FIRST}, [COUNT, POSITION])
    assert 2.5 < time.monotonic() - began < 40
    (live,) = run.fake.execs()
    assert kinds(live) == ["frame+ask", "ok", "again", "ok", "again"] and not live["finished"]
    failed = [e for e in read_log(run.log) if e.get("event") == "exec_failed"]
    assert [(e["purpose"], e["reason"], e["questions"]) for e in failed] == [("ask", "timeout", 2)]
    assert summarize(run.log)["answers"]["timed_out"] == 2
