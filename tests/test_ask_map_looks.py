"""`suite_priors.ask_map` for look questions (0.6.0 lane A2, 0.10.0): a look question has no decision point, so it goes
to the frame of the other stream nearest in steps into the task; a look point is any frame, so every frame of the own
window is a valid look and every target is a frame of the other stream."""

from __future__ import annotations

import pytest

from baseline_kit import play
from family_kit import fake_spec, register
from stbench import suite_priors
from stbench.baselines import AllUnknown
from stream_kit import case_on, during_q, end_q, qid
from synth import HOME, Synth

BD = "before_doorway"
ROOM = HOME.rooms[0]


@pytest.fixture
def look_family(monkeypatch):
    """A look family `fake_look` in the registry (`questions_text.LOOKS`, which `ask_map` reads)."""
    register(monkeypatch, fake_spec("fake_look", "during", "now"), lambda w, k: None, look=True)


def look_q(n: int, k: int) -> dict:
    return during_q(n, k, template="fake_look", slots={"n": n})


def test_a_look_goes_to_the_other_streams_frame_nearest_in_steps_into_the_task(tmp_path, look_family):
    """Contract (plan §4 A2, 0.10.0): a look question at (task i, s steps in) is asked at the frame of the other stream,
    in its task `task_map[i]`, s steps in, else the last frame of that task when it is shorter (any frame may be
    asked). Hand-worked. Own: the second task of a script (steps 20-39), a look question at 23, 31, 32, 36 and 39, that
    is 3, 11, 12, 16 and 19 steps into the task (31 lies inside a decision window, which ends at 32). Other: one task of
    14 steps (frames 0-13) with a decision at 8. Mapped: 3 -> 3, 11 -> 11, 12 -> 12, 16 -> 13 (the last frame, 3 away),
    19 -> 13. The look at keyframe 23, which is also own's decision point, goes by the look rule (to 3), not as the
    first `before_doorway` (which would be 8). The other stream, run with every question answered, asks each.
    Regression caught: a look mapped as a decision point (a crash on the first look question of `other_case`), a target
    that is no frame of the other stream, or a look moved to a decision window's end."""
    own = case_on(tmp_path, Synth(tasks=[[(ROOM, 20)], [(ROOM, 20)]], seed=1,
                                  decisions=[(3, BD), (12, BD), (23, BD), (32, BD), (36, "room_done")]),
                  [look_q(n, k) for n, k in enumerate((23, 31, 32, 36, 39), 1)] + [end_q(9)], a=1)
    other = case_on(tmp_path, Synth(tasks=[[(ROOM, 14)]], seed=2, decisions=[(8, BD)]), [during_q(1, 8)])
    got = suite_priors.ask_map(own, other)
    assert got == {23: 3, 31: 11, 32: 12, 36: 13, 39: 13}
    run, _ = play(own, AllUnknown, stream_from=other, ask_map=got)
    assert run.status == "complete" and run.failures == {}
    assert set(run.answers) == {qid(n) for n in (1, 2, 3, 4, 5, 9)}


def test_a_look_in_a_later_task_goes_to_the_last_task_the_other_holds(tmp_path, look_family):
    """Contract: the task a look is asked in goes through `task_map`: the same case task of the other window, or its
    last when it holds fewer. Hand-worked. Own: two tasks of 20 steps, cut at its decision 36 (16 steps into the second
    task) and every eighth step and task end: a look at 36 and a decision question at 12 (a `before_doorway`, own's
    first). Other: one task of 20 steps (decision at 8). The look in own's task 1 maps to the other's task 0 (its
    last): 16 steps in -> 16; the decision question goes as before, to the other's first `before_doorway`, 8.
    Regression caught: a look sent to a task the other lacks (a KeyError) or past its last step, and a decision
    question disturbed by the look rule."""
    own = case_on(tmp_path, Synth(tasks=[[(ROOM, 20)], [(ROOM, 20)]], seed=1,
                                  decisions=[(3, BD), (12, BD), (36, "room_done")]),
                  [during_q(1, 12), look_q(2, 36), end_q(3)], a=0, b=1, length="medium")
    other = case_on(tmp_path, Synth(tasks=[[(ROOM, 20)]], seed=2, decisions=[(8, BD)]), [during_q(1, 8)])
    got = suite_priors.ask_map(own, other)
    assert got == {12: 8, 36: 16}
    run, _ = play(own, AllUnknown, stream_from=other, ask_map=got)
    assert run.status == "complete" and run.failures == {}


def test_a_look_outside_its_own_window_is_refused(tmp_path, look_family):
    """Contract: a look question whose keyframe is no frame of its own window is a bug of the compile, refused with the
    keyframe named (not mapped to a guess); one at any frame inside it maps, though it ends no decision window.
    Hand-worked: own is the first task (steps 0-19), cut after 12 and 19 only; a look at 5 maps to the other's step 5
    and a look at 25 (the second task, not in the window) is refused. Regression caught: a mapping that hides a point
    the harness would have refused, or a look still demanded at a decision window's end."""
    own = case_on(tmp_path / "in", Synth(tasks=[[(ROOM, 20)], [(ROOM, 20)]], seed=1, decisions=[(12, BD)]),
                  [look_q(1, 5), end_q(2)])
    stray = case_on(tmp_path / "out", Synth(tasks=[[(ROOM, 20)], [(ROOM, 20)]], seed=1, decisions=[(12, BD)]),
                    [look_q(1, 25), end_q(2)])
    other = case_on(tmp_path / "other", Synth(tasks=[[(ROOM, 20)]], seed=2, decisions=[(8, BD)]), [during_q(1, 8)])
    assert suite_priors.ask_map(own, other) == {5: 5}
    with pytest.raises(ValueError, match="no frame of case"):
        suite_priors.ask_map(stray, other)


def test_two_questions_at_one_keyframe_that_map_differently_are_refused(tmp_path, look_family):
    """Contract (A2's review): a keyframe maps to one keyframe of the other stream, so a look question and a decision
    question asked at the same keyframe whose rules give different targets are refused, naming the keyframe, rather
    than the later silently replacing the earlier. Hand-worked on the first test's own and other streams: a look at
    keyframe 23 goes to 3 (the same steps in), a `before_doorway` question at 23 is own's third such point and goes to
    8 (the other's last), so asking both raises; the look alone, and the decision question alone, map as before.
    Regression caught: `other_case` asked a question at the wrong point because its map kept only the last target."""
    own_synth = Synth(tasks=[[(ROOM, 20)], [(ROOM, 20)]], seed=1, decisions=[(3, BD), (12, BD), (23, BD), (32, BD)])
    other_synth = Synth(tasks=[[(ROOM, 20)]], seed=2, decisions=[(8, BD)])
    other = case_on(tmp_path / "other", other_synth, [during_q(1, 8)])
    both = case_on(tmp_path / "both", own_synth, [look_q(1, 23), during_q(2, 23)], a=1)
    with pytest.raises(ValueError, match="keyframe 23"):
        suite_priors.ask_map(both, other)
    look = case_on(tmp_path / "look", own_synth, [look_q(1, 23)], a=1)
    decision = case_on(tmp_path / "decision", own_synth, [during_q(2, 23)], a=1)
    assert suite_priors.ask_map(look, other) == {23: 3} and suite_priors.ask_map(decision, other) == {23: 8}
