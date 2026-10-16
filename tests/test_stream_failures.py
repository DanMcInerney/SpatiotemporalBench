"""What `stream.run_case` does when a system or a case goes wrong (plan §2.4's table, one scenario per row), the case
cap, what a case refuses before anything is sent, and the run folder's removal. Each scenario runs the 20-step hand
case (`stream_kit.hand20`: one during question at step 12 and two end questions; twenty frames) on a `Recorder` that
fails at one message, and checks the status, which questions were answered and which failed (and why), and that
nothing was sent after the case ended."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stbench import stream
from stbench.stream import run_case
from stbench.stream_protocol import TransportError
from stream_kit import (
    QUESTIONS,
    UNKNOWN,
    Recorder,
    case_on,
    during_q,
    end_q,
    hand20,
    ops,
    play,
    qid,
)

Q1, Q2, Q3 = qid(1), qid(2), qid(3)


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    """The 20-step hand case, written once: the tests below only read it."""
    return case_on(tmp_path_factory.mktemp("hand"), hand20(), QUESTIONS)
FULL = ["start", "task", *["observe"] * 13, "ask", *["observe"] * 7, "ask", "finish"]   # a complete run's ops
DURING = 16   # the ops up to and including the during ask (after the 13th frame, step 12)


def failing(op: str, error: BaseException, *, at: int = 1, when: str | None = None) -> type[Recorder]:
    """A `Recorder` whose `at`-th message of kind `op` (an ask of kind `when`), counting from 1, raises `error`."""

    class Failing(Recorder):
        def _note(self, name, *args):
            super()._note(name, *args)
            if name == op and (when is None or args[0] == when) and sum(
                    e[0] == op and (when is None or e[1] == when) for e in self.log) == at:
                raise error

    return Failing


def replying(reply) -> type[Recorder]:
    """A `Recorder` whose end ask answers `reply` (whatever it is: a dict, a list, a wrong shape) and whose during ask
    answers properly."""

    class Replying(Recorder):
        def ask(self, when, now, questions):
            answers = super().ask(when, now, questions)
            return reply if when == "end" else answers

    return Replying


def settled(status: str, answers: dict, failures: dict) -> dict:
    """What a run holds: the status's head, each question's answer type or failure reason (the three questions are
    answered `unknown` unless `answers` says otherwise), and the failure counts."""
    return {"status": status, "answers": {Q1: "unknown", Q2: "unknown", Q3: "unknown", **answers}, "failures": failures}


def held(run) -> dict:
    """The same of a run: "ended: observe" for "ended: observe: RuntimeError: ...", each answer by type or reason."""
    status = run.status if run.status == "complete" else ": ".join(run.status.split(": ")[:2])
    answers = {q: a["reason"] if a["type"] == "failed" else a["type"] for q, a in run.answers.items()}
    return {"status": status, "answers": answers, "failures": run.failures}


# (what fails, the system, what the run holds after, the messages that were sent): plan §2.4's table, a row each
SCENARIOS = {
    "start raises": (failing("start", RuntimeError("no memory")),
                     settled("ended: start", {q: "case ended: start" for q in (Q1, Q2, Q3)}, {"case ended: start": 3}),
                     ["start"]),
    "task raises": (failing("task", ValueError("bad brief")),
                    settled("ended: task", {q: "case ended: task" for q in (Q1, Q2, Q3)}, {"case ended: task": 3}),
                    ["start", "task"]),
    "observe raises after the during ask": (
        failing("observe", RuntimeError("out of memory"), at=14),
        settled("ended: observe", {Q2: "case ended: observe", Q3: "case ended: observe"}, {"case ended: observe": 2}),
        FULL[: DURING + 1]),
    "observe raises a TransportError": (
        failing("observe", TransportError("pipe broke"), at=1),
        settled("ended: observe", {q: "case ended: observe" for q in (Q1, Q2, Q3)}, {"case ended: observe": 3}),
        FULL[:3]),
    "an ask raises a TransportError": (
        failing("ask", TransportError("timed out"), when="during"),
        settled("ended: ask", {q: "case ended: ask" for q in (Q1, Q2, Q3)}, {"case ended: ask": 3}),
        FULL[:DURING]),
    "the end ask raises a TransportError": (
        failing("ask", TransportError("pipe broke"), when="end"),
        settled("ended: ask", {Q2: "case ended: ask", Q3: "case ended: ask"}, {"case ended: ask": 2}),
        FULL[:-1]),
    "an ask raises anything else": (
        failing("ask", RuntimeError("system said no"), when="during"),
        settled("complete", {Q1: "ask error"}, {"ask error": 1}), FULL),
    "the end ask raises anything else": (
        failing("ask", ValueError("bad"), when="end"),
        settled("complete", {Q2: "ask error", Q3: "ask error"}, {"ask error": 2}), FULL),
    "an ask answers in the wrong shape": (
        replying(None), settled("complete", {Q2: "ask error", Q3: "ask error"}, {"ask error": 2}), FULL),
    "an ask leaves a question out": (
        replying({Q2: UNKNOWN}), settled("complete", {Q3: "missing"}, {"missing": 1}), FULL),
    "an ask answers a question twice": (
        replying([{"qid": Q2, "answer": UNKNOWN}, {"qid": Q2, "answer": {"type": "count", "value": 1}},
                  {"qid": Q3, "answer": UNKNOWN}]),
        settled("complete", {Q2: "answered twice"}, {"answered twice": 1}), FULL),
    "an ask answers with something that is no object": (
        replying({Q2: UNKNOWN, Q3: "unknown"}), settled("complete", {Q3: "not an object"}, {"not an object": 1}),
        FULL),
    "finish raises": (failing("finish", RuntimeError("no usage")), settled("ended: finish", {}, {}), FULL),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_failure_semantics(hand, name):
    """Contract (plan §2.4's table): each failure ends the case or fails its questions exactly as the table says, one
    scenario a row: a start, task or observe that raises (a `TransportError` or anything else) ends the case and fails
    every question not yet answered ("case ended: <message>") while an earlier answer stands; an ask that raises a
    `TransportError` ends it, failing this ask's questions and every later one; an ask that raises anything else, or
    answers in the wrong shape, fails its own questions ("ask error") and the case goes on to finish; a question the
    reply leaves out, repeats or answers with a non-object fails alone ("missing", "answered twice", "not an object");
    a `finish` that raises leaves every answer standing and no usage. After the case ended nothing more is sent.
    Regression caught: a late reply paired with a later question, a failure scored as an answer, an ask error that
    ends the case like a broken pipe, or a system asked again after the case is over."""
    system, expected, sent = SCENARIOS[name]
    run, log = play(hand, system)
    assert held(run) == expected
    assert ops(log) == sent
    assert (run.usage is not None) == (sent[-1] == "finish" and name != "finish raises")


def test_make_that_raises_fails_the_case_and_a_late_finish_failure_keeps_the_answers(hand):
    """Contract (plan §2.4): a system that cannot be built ends its case before `start` (every question fails, no
    message is sent, nothing is timed), and a system whose `close` raises after a good run keeps its answers, the
    status naming what did not close. Regression caught: a constructor failure that crashes the harness instead of
    failing its case, or a cleanup error that throws away the answers of a finished case."""
    case = hand

    def broken(state, work):
        raise RuntimeError("cannot start the container")

    run = run_case(broken, case)
    assert run.status == "ended: start: RuntimeError: cannot start the container"
    assert {a["reason"] for a in run.answers.values()} == {"case ended: start"} and run.failures == {
        "case ended: start": 3}
    assert run.wall_s == {} and run.usage is None and run.state_bytes == {}

    class Stubborn(Recorder):
        def close(self):
            raise RuntimeError("container still there")

    run, _ = play(case, Stubborn)
    assert run.status == "ended: close: RuntimeError: container still there" and run.failures == {}
    assert run.answers == {q: UNKNOWN for q in (Q1, Q2, Q3)} and run.usage is not None


class Clock:
    """The harness's clock for a test: it moves only when a system moves it, so no test sleeps or races."""

    now = 0.0

    def monotonic(self) -> float:
        return self.now


def test_the_case_cap_ends_the_case(hand, monkeypatch):
    """Contract (plan §2.4): the cap is checked before each message; past it no message is sent and every question not
    yet answered fails with "case cap". With `cap_s` 100 and a system that takes 1,000 s (on the harness's clock) over
    its first observe, the second observe is never sent and all three questions fail; one that is slow only in its end
    ask answers its questions, but `finish` is not sent (no usage) and the status names the cap. A fast system under
    the same cap completes. Regression caught: an agent that never ends a case, or a cap that throws away answers it
    already has."""
    clock = Clock()
    monkeypatch.setattr(stream, "time", SimpleNamespace(monotonic=clock.monotonic))
    case = hand

    class SlowObserve(Recorder):
        def observe(self, steps):
            super().observe(steps)
            clock.now += 1000

    run, log = play(case, SlowObserve, cap_s=100)
    assert run.status == "ended: case cap" and ops(log) == ["start", "task", "observe"]
    assert run.failures == {"case cap": 3} and {a["reason"] for a in run.answers.values()} == {"case cap"}
    assert run.usage is None and run.wall_s["observe"] == [1000]

    class SlowEnd(Recorder):
        def ask(self, when, now, questions):
            if when == "end":
                clock.now += 1000
            return super().ask(when, now, questions)

    clock.now = 0.0
    run, log = play(case, SlowEnd, cap_s=100)
    assert run.status == "ended: case cap" and ops(log)[-1] == "ask" and run.failures == {}
    assert run.answers == {q: UNKNOWN for q in (Q1, Q2, Q3)} and run.usage is None

    clock.now = 0.0
    run, _ = play(case, Recorder, cap_s=100)
    assert run.status == "complete" and run.usage is not None


def test_a_case_refuses_what_it_cannot_run_before_anything_is_sent(hand, tmp_path):
    """Contract (plan §4.4): a channel that is no channel, a channel that answers none of the case's questions, one that
    the system's class does not list, a question to be asked at a step outside the window, a picture missing from the
    script and an invalid question row each raise `ValueError` (or `FileNotFoundError`) before the first message: no
    `start` is sent, and a system already built is closed. Regression caught: a run started, and paid for, that must
    fail at its first ask; a stale `--channel rgb` mapped to some channel; a system left running."""
    case = hand
    built: list[Recorder] = []

    def maker(cls=Recorder):
        def make(state, work):
            built.append(cls(state, work, log=[]))
            return built[-1]
        return make

    class Detections(Recorder):
        channels = ("detections",)

    with pytest.raises(ValueError, match="unknown channel"):
        run_case(maker(), case, channel="rgb")
    only_detections = case_on(tmp_path / "d", hand20(), [end_q(1, needs=("detections",))])
    with pytest.raises(ValueError, match="no question to ask on the rgb_pose channel"):
        run_case(maker(), only_detections)
    bad = case_on(tmp_path / "b", hand20(), [end_q(1, needs=("now",))])
    with pytest.raises(ValueError, match="not a valid question"):
        run_case(maker(), bad)
    twice = case_on(tmp_path / "t", hand20(), [end_q(1), end_q(1)])
    with pytest.raises(ValueError, match="one qid twice"):
        run_case(maker(), twice)
    assert built == []  # refused before any system was made

    with pytest.raises(ValueError, match="runs on the detections channel only, not 'rgb_pose'"):
        run_case(maker(Detections), case)
    past_the_end = case_on(tmp_path / "m", hand20(), [during_q(1, 25), end_q(2)])
    with pytest.raises(ValueError, match="outside the window"):
        run_case(maker(), past_the_end)
    gone = case_on(tmp_path / "g", hand20(), QUESTIONS)
    gone.script.rgb(7).unlink()
    with pytest.raises(FileNotFoundError, match="1 image files"):
        run_case(maker(), gone)
    assert len(built) == 3 and all(s.closed and s.log == [] for s in built)


def test_the_run_folder_is_removed(hand):
    """Contract (plan §4.4): the run folder is removed whatever the system left in it, a read-only file and a path past
    the platform's limit included, after a complete run and after one that failed. Regression caught: state leaking
    into the next case (a stale run folder), or a cleanup that stops at the first file it cannot remove."""
    roots: set[Path] = set()
    prefix = "\\\\?\\" if os.name == "nt" else ""

    class Awkward(Recorder):
        fail_at = 0   # the observe (counting from 1) that raises; 0: none

        def observe(self, steps):
            super().observe(steps)
            deep = self.state_dir.joinpath(*["d" * 60] * 5)
            os.makedirs(prefix + str(deep), exist_ok=True)
            with open(prefix + str(deep / "leaf"), "wb") as f:
                f.write(b"x")
            readonly = self.state_dir / "ro.txt"
            if not readonly.exists():
                readonly.write_bytes(b"x")
                readonly.chmod(0o400)
            roots.add(self.root)
            if sum(e[0] == "observe" for e in self.log) == self.fail_at:
                raise RuntimeError("this system fails in its third frame")

    class Failing(Awkward):
        fail_at = 3

    case = hand
    for system, status in ((Awkward, "complete"), (Failing, "ended: observe")):
        roots.clear()
        run, _ = play(case, system)
        assert run.status.startswith(status)
        assert len(roots) == 1 and not roots.pop().exists()
