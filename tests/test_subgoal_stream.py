"""The subgoal on the stream: which steps carry it, on which channels and under which brief (`stream._window`), and how
the viewer shows it (`view_frames`). A job that states a subgoal is made with `subgoal_kit.with_subgoals`."""

from __future__ import annotations

import pytest

from stbench.stream_protocol import CHANNELS
from stbench.view_frames import Streams
from stream_kit import QUESTIONS, case_on, hand20, play
from subgoal_kit import with_subgoals

# hand20's script: 20 keyframes of one task, the subgoal of keyframe k reads stage k mod 4 (`subgoal_kit.stages`)
WANT = [f"cleaning the kitchen, stage {k % 4}" for k in range(20)]


def delivered(log: list) -> list[dict]:
    return [step for entry in log if entry[0] == "observe" for step in entry[1]]


@pytest.mark.parametrize("channel", sorted(CHANNELS))
def test_steps_carry_the_subgoal_on_every_channel(tmp_path, monkeypatch, channel):
    """Contract (FD-0.5.0-control-impl §2): a job's subgoal reaches a system in every step, on each channel (the camera
    alone, the camera with depth, the detections), as `step["subgoal"]`, the job's text for that keyframe (the case is
    the whole script here, so the step numbers are the keyframes'). Regression caught: a subgoal sent on some
    channels only (copied inside a branch of one channel), or sent off by one keyframe."""
    with_subgoals(monkeypatch)
    case = case_on(tmp_path, hand20(), QUESTIONS)
    run, log = play(case, channel=channel)
    assert run.status == "complete"
    steps = delivered(log)
    assert [s["k"] for s in steps] == list(range(20)) and [s["subgoal"] for s in steps] == WANT


def test_a_job_without_subgoals_sends_no_key(tmp_path):
    """Contract: a step of a job that states no subgoal has no `subgoal` key (not an empty one), so the header of such
    a job is the one it had. Regression caught: a default text sent for every job."""
    run, log = play(case_on(tmp_path, hand20(), QUESTIONS))
    assert run.status == "complete" and all("subgoal" not in s for s in delivered(log))


def test_the_neutral_brief_withholds_the_subgoal(tmp_path, monkeypatch):
    """Contract (FD-0.5.0-control-impl §15 item 7, the coordinator's default): the `neutral` brief states no assignment,
    and a subgoal that names the task's own things would state it again, so that variant sends steps without the key,
    on every channel, while `correct` sends it. Regression caught: a neutral arm that still carries the task in every
    header, which measures nothing about the assignment."""
    with_subgoals(monkeypatch)
    case = case_on(tmp_path, hand20(), QUESTIONS)
    for channel in sorted(CHANNELS):
        steps = delivered(play(case, channel=channel, brief="neutral")[1])
        assert len(steps) == 20 and all("subgoal" not in s for s in steps), channel
        assert [s["subgoal"] for s in delivered(play(case, channel=channel, brief="correct")[1])] == WANT, channel


def test_the_swapped_brief_is_refused_for_a_job_with_subgoals(tmp_path, monkeypatch):
    """Contract: `run_case` refuses (ValueError, before a message is sent) the `swapped` brief for a job that states a
    subgoal, saying why it is retired: the subgoal says the true task in every frame and the brief would say
    another. A job with none is not refused here. Regression caught: a swapped run whose header and brief disagree."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    with_subgoals(monkeypatch)
    with pytest.raises(ValueError, match="a subgoal says the true task in every frame"):
        play(case, brief="swapped")


def test_the_viewer_shows_the_subgoal_of_each_step(tmp_path, monkeypatch):
    """Contract: the frame-by-frame view packs the delivered steps' subgoals as a list with one text per step (the
    script's, here 20), and `null` for a job that states none, so the page shows a "Subgoal" fact only where there is
    one. Regression caught: a viewer that shows the steps a system was sent and not the words it was told."""
    plain = case_on(tmp_path / "plain", hand20(), QUESTIONS)
    with_subgoals(monkeypatch)
    stated = case_on(tmp_path / "stated", hand20(), QUESTIONS)
    for case, want in ((stated, WANT), (plain, None)):
        problems: list[str] = []
        streams = Streams(case.dataset)
        key = streams.get(case.id, "rgb_pose", problems)
        assert problems == [] and streams.items[key]["subgoal"] == want
