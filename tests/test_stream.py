"""`stream.run_case` (the stream protocol's harness side) on hand-built synthetic cases,
with systems that record every call: what is delivered (numbers, files, keys) and when it is asked. Failures, refusals
and the run folder are in test_stream_failures.py; the adapters' messages in test_command_system_local.py and
test_local_system.py, and a whole case through them in the last test of each of those and of
test_command_system.py."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import pytest

from stbench import case as case_mod
from stbench import episode as ep
from stbench import safecopy, sandbox
from stbench.case import read_case
from stbench.records import check_gold, check_question
from stream_kit import (
    QUESTIONS,
    K,
    L,
    Recorder,
    add_case,
    case_on,
    clock,
    during_q,
    end_q,
    gold_row,
    hand20,
    messages,
    observed,
    ops,
    play,
    qid,
)
from synth import Synth

CID, CID2 = "0123456789abcdef", "fedcba9876543210"


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    """The 20-step hand case, written once: the tests that take it only read it."""
    return case_on(tmp_path_factory.mktemp("hand"), hand20(), QUESTIONS)


def force_case_id(monkeypatch, cid: str) -> None:
    """Every case written from now on claims `cid` (script ids stay random: `episode` claims its own)."""
    monkeypatch.setattr(case_mod, "claim_id", lambda dataset, entry: cid)


def world_messages(log: list) -> str:
    """What a system was told of the world, as JSON: its task and observe messages."""
    return messages([e for e in log if e[0] in ("observe", "task")])


def frame_keys(log: list) -> list[list[str]]:
    """The detection keys of every delivered step, in order."""
    return [[d["d"] for d in step["detections"]] for entry in log if entry[0] == "observe" for step in entry[1]]


def key_set(cid: str, kc: int, n: int) -> set[str]:
    """The keys a frame of `n` detections gets in case `cid` at case step `kc`: the contract's formula, by hand."""
    return {hashlib.sha256(f"{cid}\x1fd\x1f{kc}\x1f{i}".encode()).hexdigest()[:8] for i in range(n)}


# -- numbers, keys and files never name the window's place -----------------------------------------------------------
def window_script(before: list[list[tuple]]) -> Synth:
    """`before` tasks, then the window's task: six keyframes in the kitchen with a red mug and a blue book in view
    (and, before the window only, a green ball), whatever came before it."""
    s = Synth(tasks=[*before, [(K, 6)]])
    k0 = s.task_starts()[-1]
    s.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    s.item("E02", "book", {"color": "blue"}, [(0, None, (2.5, 1.0, 0.9))])
    s.item("E03", "ball", {"color": "green"}, [(0, None, (1.0, 2.0, 0.9))])
    s.see(range(k0, k0 + 6), "E01")
    s.see(range(k0 + 2, k0 + 6), "E02")
    s.see(range(0, k0), "E03")
    return s


@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_a_window_streams_the_same_from_any_script(tmp_path, monkeypatch, channel):
    """Contract (map X-H4, VISION rule 6): two scripts that differ only before the window, their cases sharing one id,
    give a system the same messages and the same inbox file names, on the agents' channel and on the gate channel.
    Script B has more and longer tasks before the window, so its window starts at another keyframe and script time;
    its oracle file holds each frame's detections in the other order with other keys. Regression caught: a window
    revealing its place (a case k, clock, task index or file name that carries the script's numbering, or a detection
    key or its order that depends on the script keyframe: the oracle's keys come from the script's inputs and its
    keyframe)."""
    force_case_id(monkeypatch, CID)
    a = case_on(tmp_path / "a", window_script([[(K, 4)]]), [end_q(1)], a=1)
    b = case_on(tmp_path / "b", window_script([[(K, 9)], [(L, 2)]]), [end_q(1)], a=2)
    assert (read_case(a).k0, read_case(b).k0) == (4, 11)  # the windows start at different keyframes of their scripts
    rows = [json.loads(line) for line in b.script.oracle.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["detections"].reverse()
    ep.write_jsonl(b.script.oracle, rows)
    (run_a, log_a), (run_b, log_b) = play(a, channel=channel), play(b, channel=channel)
    assert run_a.status == run_b.status == "complete"
    assert messages(log_a) == messages(log_b)
    assert [e[-1]["inbox"] for e in log_a] == [e[-1]["inbox"] for e in log_b]
    first = next(e for e in log_a if e[0] == "observe")[1][0]
    assert first["k"] == 0 and first["status"]["task"] == 0 and first["clock"] == read_case(a).start_clock + ".0"


def test_detections_are_rekeyed_from_the_case(tmp_path, monkeypatch):
    """Contract (the review's V-2): every detection a system gets is keyed from the case's id and the case's own step,
    `d` = the first 8 hex characters of SHA-256 over (case id, "d", case k, index in the frame) joined by `\\x1f`, in a
    canonical order of the frame's detections; so two cases of one window with different ids get different keys, the
    same case run twice the same keys, and the keys are distinct within a frame. A privileged system's `identity`
    maps each new key to the entity of the detection with that key (the red mug E01 or the blue book E02). Regression
    caught: the oracle's script-keyframe keys reaching a system, from which the seed and the window's place could be
    searched, and an `identity` left keyed by the old keys."""
    force_case_id(monkeypatch, CID)
    one = case_on(tmp_path, hand20(), QUESTIONS)
    force_case_id(monkeypatch, CID2)
    two = add_case(one, QUESTIONS)

    class Privileged(Recorder):
        privileged = True

    log1, log1b, log2 = (play(case, Privileged, channel="detections")[1] for case in (one, one, two))
    assert frame_keys(log1) == frame_keys(log1b)
    assert all(x != y for x, y in zip(frame_keys(log1), frame_keys(log2), strict=True))
    for cid, log in ((CID, log1), (CID2, log2)):
        for entry in (e for e in log if e[0] == "observe"):
            for step in entry[1]:
                keys = [d["d"] for d in step["detections"]]
                assert len(set(keys)) == len(keys) == (1 if step["k"] < 5 else 2)
                assert set(keys) == key_set(cid, step["k"], len(keys))
                for det in step["detections"]:
                    assert step["identity"][det["d"]] == ("E01" if det["kind"] == "mug" else "E02")
                assert set(step["identity"]) == set(keys)


def test_each_ask_carries_its_questions_and_the_run_is_recorded(hand):
    """Contract (plan §2.4, map D-20): on the hand case of 20 steps (decision points at case k 3 and 12, one task) the
    during ask carries the during question with `now` the step it follows (step 12, 18 s into the case), the end ask
    every end question in file order with `now` the last step (19, 28.5 s: a step is 1.5 s after the last and the clock
    is delivered to the tenth); every question is answered, the run is complete, each message is timed (one observe a
    frame: 20), and the usage is self-reported. Where the asks fall among the frames is `test_stream_frames.py`'s.
    Regression caught: an ask that names the wrong step or clock, questions out of file order, or a message left
    untimed."""
    case = hand
    run, log = play(case)
    during, end = (e for e in log if e[0] == "ask")
    assert during[1:3] == ("during", {"k": 12, "clock": clock(case, 18)})
    assert [q["qid"] for q in during[3]] == [qid(1)]
    assert end[1:3] == ("end", {"k": 19, "clock": clock(case, 28.5)})
    assert [q["qid"] for q in end[3]] == [qid(2), qid(3)]
    assert run.status == "complete" and run.failures == {}
    assert run.answers == {qid(n): {"type": "unknown"} for n in (1, 2, 3)}
    assert {key: len(times) for key, times in run.wall_s.items()} == {
        "start": 1, "task": 1, "observe": 20, "ask_during": 1, "ask_end": 1, "finish": 1}
    assert run.usage is not None and run.usage["self_reported"] is True and run.usage["models"] is None


def test_files_are_plain_copies(tmp_path):
    """Contract (plan §2.3): the inbox's files are plain copies, never `copy2`, so a file's time is the time of the run
    and not its source's. The sources' times are set to 2001. Regression caught: a file time carrying the script's
    history (the render's time order, or the keyframe a frame came from)."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    for k in range(20):
        for source in (case.script.rgb(k), case.script.depth(k)):
            os.utime(source, (978307200, 978307200))  # 2001-01-01
    times = []
    for channel in ("rgb_pose", "detections"):  # a JPEG a step on the one, a depth PNG a step on the other
        _, log = play(case, channel=channel)
        times += [m for e in log if e[0] == "observe" for m in e[-1]["mtimes"].values()]
    assert len(times) == 40
    assert all(m > time.time() - 3600 for m in times)


def test_a_system_that_opens_no_picture_is_given_none(hand):
    """Contract: a system whose class sets `reads_files` False gets the same messages and the same steps (which still
    name their pictures) as one that reads them, and an inbox that stays empty; one that does not say gets the files.
    Hand case: the 20-step case on the `detections` channel with a `Recorder`, then with a subclass that sets the flag:
    the logs are equal but for what the inbox held, which is empty at every message for the second and the frame's
    depth PNGs for the first. Regression caught: the flag dropped (every baseline copies pictures it never opens), or
    honoured by skipping the steps' own contents."""
    class Blind(Recorder):
        reads_files = False

    reading, seen = play(hand, channel="detections")[1], play(hand, Blind, channel="detections")[1]
    assert any(e[-1]["inbox"] for e in reading) and not any(e[-1]["inbox"] for e in seen)
    assert [e[:-1] for e in seen if e[0] != "start"] == [e[:-1] for e in reading if e[0] != "start"]   # not the token


def truth_in(names: list[str]) -> list[str]:
    """The names among `names` (POSIX paths) whose last part is one the sandbox keeps from a system."""
    return [n for n in names if n.rsplit("/", 1)[-1].lower() in sandbox.TRUTH_FILES]


def test_no_truth_in_the_working_folder(hand):
    """Contract (VISION rule 6): at every message the run folder holds no name of `sandbox.TRUTH_FILES` and nothing
    but the state folder and the working folder's `public/robot.json`, `public/camera.json` and the current inbox.
    Regression caught: a truth file (gold, questions, the timeline, the visibility log) handed to a system in its
    folder. The checker is run on a planted name first, so it is not passing vacuously."""
    assert truth_in(["work/public/gold.jsonl", "work/public/robot.json"]) == ["work/public/gold.jsonl"]
    case = hand
    _, log = play(case)
    assert len(log) == 25   # start, task, 20 observes, 2 asks, finish
    folders = {"state", "work", "work/public", "work/public/robot.json", "work/public/camera.json", "work/inbox"}
    for entry in log:
        names = entry[-1]["all"]
        assert truth_in(names) == []
        assert folders <= set(names)
        assert all(n in folders or n.startswith("work/inbox/") for n in names)


def test_identity_and_structure_only_when_declared(hand):
    """Contract (plan §2.4): a step carries `identity` only for a class that sets `privileged`, and a question carries
    `template` and `slots` only for a class that sets `structured`; every question is exactly its `qid`, `text` and
    `answer_type` otherwise, and never a gold-side key (`when`, `decision_k`, `needs`). Regression caught: perfect
    identity or the question's structure reaching an agent."""
    case = hand

    class Privileged(Recorder):
        privileged = True

    class Structured(Recorder):
        structured = True

    def steps(log):
        return [s for e in log if e[0] == "observe" for s in e[1]]

    def asked(log):
        return [q for e in log if e[0] == "ask" for q in e[3]]

    _, plain = play(case, channel="detections")
    assert all("identity" not in s for s in steps(plain))
    assert all(set(q) == {"qid", "text", "answer_type"} for q in asked(plain)) and len(asked(plain)) == 3
    _, privileged = play(case, Privileged, channel="detections")
    assert all(isinstance(s["identity"], dict) and s["identity"] for s in steps(privileged))
    _, pictures = play(case, Privileged)  # no detections on the default channel: nothing to identify
    assert all(s["identity"] == {} for s in steps(pictures))
    _, structured = play(case, Structured)
    assert [set(q) for q in asked(structured)] == [{"qid", "text", "answer_type", "template", "slots"}] * 3
    assert asked(structured)[0]["template"] == "path_ahead" and asked(structured)[0]["slots"] == {"cap": 2}


def test_during_off_streams_identical_frames(hand):
    """Contract (plan §2.4): with `during` off the observe messages and the task messages are byte-equal to
    the run with it on, no during ask is sent, and the during question is not part of the run (no answer, no failure).
    Regression caught: a leak-check arm that differs in more than the asks, so the measure of the asks is not one."""
    case = hand
    on, log_on = play(case)
    off, log_off = play(case, during=False)
    assert world_messages(log_on) == world_messages(log_off)
    assert observed(log_off) == [[k] for k in range(20)]
    assert ops(log_off) == ["start", "task", *["observe"] * 20, "ask", "finish"]
    assert [e[1] for e in log_off if e[0] == "ask"] == ["end"]
    assert set(on.answers) == {qid(1), qid(2), qid(3)} and set(off.answers) == {qid(2), qid(3)}
    assert off.status == "complete" and off.failures == {}


def test_tasks_arrive_with_their_clock_gap_and_brief(tmp_path):
    """Contract (plan §2.4): each task of a window is announced before its first frame with its case index, the clock of
    its first step, the unobserved seconds since the last task ended (0 for the first) and its brief: the first brief
    is the case paragraph with the first assignment, a later one is its assignment alone. Hand values: tasks start
    7200 s apart and a step is 1.5 s after the last, so a 4-step task leaves 7195.5 s unobserved, and the second task
    starts two hours into the case. Regression caught: a task message that is missing, early, offset by the window's
    place in its script, or that carries every assignment at once."""
    synth = Synth(tasks=[[(K, 4)], [(L, 3)]], decisions=[(3, "before_doorway"), (5, "before_doorway")])
    case = case_on(tmp_path, synth, [end_q(1)], b=1, length="medium")
    _, log = play(case)
    assert ops(log) == ["start", "task", *["observe"] * 4, "task", *["observe"] * 3, "ask", "finish"]
    assert observed(log) == [[k] for k in range(7)]
    first, second = (e for e in log if e[0] == "task")
    assert first[1:4] == (0, clock(case, 0), 0.0) and second[1:4] == (1, clock(case, 7200), 7195.5)
    assert first[4].startswith("You are the onboard assistant of a robot vacuum working on one floor of a furnished "
                               "home. Its current assignment is to clean the kitchen. ")
    assert "living room" not in first[4]
    assert second[4] == "Its current assignment is to clean the living room."
    steps = [s for e in log if e[0] == "observe" for s in e[1]]
    assert [s["status"]["task"] for s in steps] == [0, 0, 0, 0, 1, 1, 1]


def test_state_bytes_need_no_copy(tmp_path, monkeypatch):
    """Contract (plan §4.4): the state folder's size is counted in place at each task's end and at finish, with nothing
    copied. Hand count: a system that writes 1,000 bytes in its first observe, 1,000 in the end ask and 1,500 in finish
    has `{"0": 1000, "finish": 3500}`; over a window of two tasks (4 and 3 steps) that writes 100 bytes in each observe,
    one a frame, the sizes are 400 at the first task's end, 700 at the second's and 700 at finish. Regression caught:
    the old checkpoint copy returning (a copy of the state at every task end), or a size taken before the task's last
    frame."""
    monkeypatch.setattr(safecopy, "copy_tree", lambda *a, **k: pytest.fail("the state was copied"))
    monkeypatch.setattr(shutil, "copytree", lambda *a, **k: pytest.fail("the state was copied"))

    class Writer(Recorder):
        def put(self, n: int) -> None:
            with open(self.state_dir / "memory.bin", "ab") as f:
                f.write(b"x" * n)

        def observe(self, steps):
            super().observe(steps)
            if not any(e[0] == "observe" for e in self.log[:-1]):
                self.put(1000)

        def ask(self, when, now, questions):
            if when == "end":
                self.put(1000)
            return super().ask(when, now, questions)

        def finish(self):
            self.put(1500)
            return super().finish()

    run, _ = play(case_on(tmp_path / "one", hand20(), QUESTIONS), Writer)
    assert run.state_bytes == {"0": 1000, "finish": 3500}

    class Steady(Recorder):
        def observe(self, steps):
            super().observe(steps)
            with open(self.state_dir / "memory.bin", "ab") as f:
                f.write(b"x" * 100)

    synth = Synth(tasks=[[(K, 4)], [(L, 3)]], decisions=[(3, "before_doorway"), (5, "before_doorway")])
    run, _ = play(case_on(tmp_path / "two", synth, [end_q(1)], b=1, length="medium"), Steady)
    assert run.state_bytes == {"0": 400, "1": 700, "finish": 700}


def test_a_case_fed_another_cases_stream(tmp_path, monkeypatch):
    """Contract (plan §4.4, `stream_from`): the steps, cuts and tasks are the other case's, its detection keys drawn
    from this case's id, and this case's during question is asked after the other stream's step `ask_map[decision_k]`,
    which must be a step of that stream (any step: a system gets one frame at a time); the questions asked are this
    case's. Hand case: X has 8 steps and a decision at 3, Y 12 steps with decisions at 5 and 9 (decision windows
    [0, 6), [6, 10), [10, 12)); X's question at 3 mapped to Y's 9 is asked right after Y's frame 9, with `now` that
    step, and mapped to Y's 8, which ends no decision window, right after frame 8. A map to step 12 (past Y's last) or
    a missing entry is refused. Regression caught: a baseline fed another stream that is asked at the wrong step, keyed
    from the other case, a missing map accepted, or a step the stream does not deliver."""
    force_case_id(monkeypatch, CID)
    x = case_on(tmp_path, Synth(tasks=[[(K, 8)]], decisions=[(3, "before_doorway")]), [during_q(1, 3), end_q(2)])
    force_case_id(monkeypatch, CID2)
    sy = Synth(tasks=[[(K, 12)]], seed=2, decisions=[(5, "before_doorway"), (9, "before_doorway")])
    sy.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    sy.see(range(12), "E01")
    y = case_on(tmp_path, sy, [end_q(9)])
    run, log = play(x, stream_from=y, ask_map={3: 9}, channel="detections")
    assert observed(log) == [[k] for k in range(12)]
    assert ops(log) == ["start", "task", *["observe"] * 10, "ask", "observe", "observe", "ask", "finish"]
    during, end = (e for e in log if e[0] == "ask")
    assert during[1:3] == ("during", {"k": 9, "clock": clock(y, 13.5)}) and [q["qid"] for q in during[3]] == [qid(1)]
    assert end[1:3] == ("end", {"k": 11, "clock": clock(y, 16.5)}) and [q["qid"] for q in end[3]] == [qid(2)]
    assert set(run.answers) == {qid(1), qid(2)} and run.status == "complete"
    assert all(set(keys) == key_set(CID, k, 1) for k, keys in enumerate(frame_keys(log)))  # keyed from X, not Y
    with pytest.raises(ValueError, match="ask_map"):
        play(x, stream_from=y)
    _, mid = play(x, stream_from=y, ask_map={3: 8}, channel="detections")
    assert next(e for e in mid if e[0] == "ask")[2]["k"] == 8
    with pytest.raises(ValueError, match="outside the window"):
        play(x, stream_from=y, ask_map={3: 12})
    with pytest.raises(ValueError, match="outside the window"):
        play(x, stream_from=y, ask_map={})


def test_a_question_that_needs_depth_is_not_asked_on_the_camera_alone_channel(tmp_path):
    """Contract (2026-10-10: no depth image for the agent; VISION rule 9, a run includes only the questions its
    sensors can answer): on the default channel `rgb_pose` a question whose `needs` name `depth` is not part of the run
    (no ask, no answer, no failure) and the case's other questions are asked as before; on `detections`, which sends
    depth, the same case asks all three. Regression caught: a depth-needing question put to an agent that is sent no
    depth image (it would be scored on a sensor it was never given), or one that fails the case instead of being left
    out."""
    case = case_on(tmp_path, hand20(), [during_q(1, 12, needs=("depth", "pose")), end_q(2), end_q(3)])
    run, log = play(case)
    assert set(run.answers) == {qid(2), qid(3)} and run.status == "complete" and run.failures == {}
    assert [e[1] for e in log if e[0] == "ask"] == ["end"]
    run, log = play(case, channel="detections")
    assert set(run.answers) == {qid(1), qid(2), qid(3)} and [e[1] for e in log if e[0] == "ask"] == ["during", "end"]


def test_the_kits_rows_pass_the_record_checks():
    """Contract (the test kit, which B.2, E1 and E2 build on): the hand-written during and end question rows pass
    `records.check_question` and their gold rows `records.check_gold`. Regression caught: a kit row the validators
    refuse, which every test built on it would then mistake for a failure of the code it tests."""
    for row in (during_q(1, 12), end_q(2), end_q(3, needs=("detections", "pose"))):
        assert check_question(row) == []
        assert check_gold(gold_row(row)) == []
