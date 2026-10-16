"""One frame per `observe` (FD-frame-pull, `stream._stream`): every system, scripted or not, is sent the window one step
at a time, in order; the inbox holds only the current frame's files while its `observe` and the ask after it run; and a
during ask comes right after the frame at its step. The compiler's look windows (`look_windows`) are never visible in
the messages. Hand cases only: `stream_kit.hand20` (20 steps, one task, decision points at step 3 and 12) and a window
of two tasks (4 and 3 steps)."""

from __future__ import annotations

import pytest

from stream_kit import QUESTIONS, K, L, case_on, clock, during_q, end_q, hand20, observed, ops, play, qid
from synth import Synth


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    """The 20-step hand case, written once: the tests below only read it."""
    return case_on(tmp_path_factory.mktemp("hand"), hand20(), QUESTIONS)


def two_tasks(root):
    """A window of two tasks (4 and 3 steps) in the home's two rooms, decisions at steps 3 and 5, one end question."""
    synth = Synth(tasks=[[(K, 4)], [(L, 3)]], decisions=[(3, "before_doorway"), (5, "before_doorway")])
    return case_on(root, synth, [end_q(1)], b=1, length="medium")


@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_an_observe_carries_exactly_one_step_and_every_step_comes_once_in_order(hand, tmp_path, channel):
    """Contract (FD-frame-pull 3.1): each `observe` message holds a list of exactly one step, and the window's steps
    arrive one each, in order, none twice and none left out, whatever the look windows around them (hand20's windows
    are [0, 4), [4, 12), [12, 13) and [13, 20), a decision point at 3 and 12; the two-task window has a task end at
    3 and 6). By hand: 20 observes of [0], [1] ... [19] for the one, 7 of [0] ... [6] for the other. Regression
    caught: batches coming back (an observe of a look window's 4, 8 or 7 steps), a step sent twice, or one dropped."""
    for case, n in ((hand, 20), (two_tasks(tmp_path), 7)):
        _, log = play(case, channel=channel)
        sent = [entry[1] for entry in log if entry[0] == "observe"]
        assert all(len(steps) == 1 for steps in sent)
        assert observed(log) == [[k] for k in range(n)]
        assert [steps[0]["k"] for steps in sent] == list(range(n))


@pytest.mark.parametrize(("channel", "kind", "ext"), [("rgb_pose", "rgb", "jpg"), ("detections", "depth", "png")])
def test_the_inbox_holds_exactly_the_current_frames_files_during_its_observe_and_the_ask_after_it(
        hand, tmp_path, channel, kind, ext):
    """Contract (FD-frame-pull 3.1, VISION rule 6; 2026-10-10): while a frame's `observe` runs and during the ask
    that follows it the inbox holds that frame's file alone, named by case k, and only the image its channel sends (a
    JPEG on the default `rgb_pose`, a depth PNG on `detections`). Hand names: the during ask after step 12 sees
    `inbox/<kind>/000012.<ext>` and no other, the end ask only step 19's, and in the two-task window the first frame
    of the second task (step 4) arrives in an inbox that holds step 4 alone and not step 3's. Regression caught: an
    earlier frame left in the inbox (it was not emptied, or a look window's files put in), or a depth file in the
    camera-alone channel's inbox."""
    def files(k: int) -> list[str]:
        return [f"inbox/{kind}/{k:06d}.{ext}"]

    _, log = play(hand, channel=channel)
    frame = None
    for entry in log:
        if entry[0] == "observe":
            frame = entry[1][0]["k"]
            assert entry[-1]["inbox"] == files(frame)
        elif entry[0] == "ask":
            assert entry[-1]["inbox"] == files(frame) and entry[2]["k"] == frame
    assert [e[2]["k"] for e in log if e[0] == "ask"] == [12, 19]
    assert all(e[-1]["inbox"] == [] for e in log if e[0] in ("start", "task"))

    _, log = play(two_tasks(tmp_path), channel=channel)
    first_of_second = next(e for e in log if e[0] == "observe" and e[1][0]["k"] == 4)
    assert first_of_second[-1]["inbox"] == files(4)


def test_a_during_ask_comes_right_after_the_frame_at_its_step(tmp_path):
    """Contract (FD-frame-pull 3.1): a during ask is sent right after the `observe` of the frame it is about, before the
    next frame, with `now` that frame's step and clock. Hand case: hand20 with a during question at each of its two
    decision points (steps 3 and 12): after 4 observes the first ask (`now` step 3), after 9 more the second (step
    12), then 7 observes, the end ask (step 19) and `finish`. Regression caught: an ask landing a frame late (after the
    next observe), early, or at the end of a look window instead of at its step."""
    case = case_on(tmp_path, hand20(), [during_q(1, 3), during_q(2, 12), end_q(3)])
    run, log = play(case)
    assert run.status == "complete" and run.failures == {}
    assert ops(log) == ["start", "task", *["observe"] * 4, "ask", *["observe"] * 9, "ask", *["observe"] * 7, "ask",
                        "finish"]
    previous = None
    for entry in log:
        if entry[0] == "observe":
            previous = entry[1][0]
        elif entry[0] == "ask":
            assert entry[2] == {"k": previous["k"], "clock": previous["clock"]}
    asks = [(e[1], e[2]["k"], [q["qid"] for q in e[3]]) for e in log if e[0] == "ask"]
    assert asks == [("during", 3, [qid(1)]), ("during", 12, [qid(2)]), ("end", 19, [qid(3)])]
    assert [e[2]["clock"] for e in log if e[0] == "ask"][1] == clock(case, 18)   # step 12 is 12 x 1.5 s in


def test_a_case_fed_another_cases_stream_is_sent_one_frame_at_a_time_too(tmp_path):
    """Contract (FD-frame-pull 3.1, `stream_from`): the other case's stream is delivered one frame per `observe` as
    well, and a during question mapped onto it is asked right after the frame it maps to. Hand case: the question at
    step 3 of an 8-step case, mapped onto step 12 of hand20's stream (a decision point there), is asked right after
    the 13th observe. Regression caught: a baseline fed another stream that is still sent look windows, or asked a
    frame late."""
    asked = case_on(tmp_path / "x", Synth(tasks=[[(K, 8)]], decisions=[(3, "before_doorway")]),
                    [during_q(1, 3), end_q(2)])
    stream = case_on(tmp_path / "y", hand20(), QUESTIONS)
    _, log = play(asked, stream_from=stream, ask_map={3: 12})
    assert observed(log) == [[k] for k in range(20)]
    assert ops(log) == ["start", "task", *["observe"] * 13, "ask", *["observe"] * 7, "ask", "finish"]
