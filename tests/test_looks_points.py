"""The look-point schedule (`questions_looks`, 0.10.0, FD-questions-0.10 §3): which frames of a window a look family is
asked at. A point is any frame at least 4 into its task and not its last, at least 4 from every point kept, up to
`ceil(frames / 6)` of them, the decision frames of the job's priority kinds drawn first. Worked by hand on the
vacuum's task of 28 frames (candidates 4 to 26: `looks_kit`) and on long tasks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from looks_kit import FRAMES, VACUUM_CAP, hand_points, long_task, look_job, vacuum_run, world
from space_kit import JOB, inputs
from stbench import case as case_mod
from stbench import questions_looks as ql
from stbench import stream
from stbench.questions_looks import LOOK_GAP_STEPS, LOOK_MIN_STEPS, LOOK_PER_FRAMES, candidate_points, points
from stbench.questions_world import World
from stbench.seeding import stable_hex
from stream_kit import case_on, end_q

DOORWAYS = {"before_doorway": (), "room_done": ()}   # the decision kinds of the priority tests, asking nothing


def ids(tag: str, n: int) -> list[str]:
    return [stable_hex(tag, i, n=16) for i in range(n)]


def test_the_constants_and_the_cap_are_the_plan():
    """Contract (FD-questions-0.10 §3): a point lies at least 4 steps into its task and 4 from every other, and a case
    asks up to one point for 6 frames, rounded up (6, from the replayed draw on the dev root: 8 left a point once in 7.8
    frames, 6 gives one in 5.9, and below 5.5 the gap is the limit). Hand-worked: 1 frame and 6 frames give 1 point, 7
    frames 2, 28 frames 5, 90 frames 15, 130 frames 22. Regression caught: a minimum, a gap or a rate changed without
    the plan (the density the redesign targets rests on them), or a cap that rounds down."""
    assert (LOOK_MIN_STEPS, LOOK_GAP_STEPS, LOOK_PER_FRAMES) == (4, 4, 6)
    assert [ql.points_cap(n) for n in (1, 6, 7, 28, 90, 130)] == [1, 1, 2, 5, 15, 22]


def test_the_candidates_are_every_frame_four_steps_into_the_task_but_its_last():
    """Contract: a look point's candidates are every frame of the window at least `LOOK_MIN_STEPS` into the task it
    belongs to, except the task's last frame, wherever the stream's decision windows end. Hand-worked: the vacuum's task
    of 28 frames (decision windows end after 5, 13, 18, 21 and 27) has the candidates 4 to 26, among them frames inside
    a decision window such as 9; a task of 30 frames with decisions at 2, 3, 4 and 10 has 4 to 28. In a window of two
    tasks the four steps count from the task a frame belongs to: the second task begins at 28, so its candidates are 32
    to 54 and the frames 28 to 31 are none, though they are 28 or more into the window, and each task's last frame (27,
    55) is none. A job with no look family has no candidate. Regression caught: only the decision windows' ends
    asked (a point on every eighth step), a minimum counted from the window's start, a task's last frame asked."""
    assert candidate_points(world(vacuum_run(), job=look_job())) == FRAMES and 9 in FRAMES
    assert candidate_points(world(vacuum_run(), job=JOB)) == []
    early = long_task(30, decisions=[2, 3, 4, 10])
    assert candidate_points(world(early, job=look_job())) == list(range(4, 29))
    two = vacuum_run(2)
    assert candidate_points(world(two, length="medium", b=1, job=look_job())) == [*range(4, 27), *range(32, 55)]


@pytest.mark.parametrize("frames", [28, 90, 130])
def test_a_case_asks_one_point_for_every_six_frames_four_steps_apart(frames):
    """Contract: the points asked are `ceil(frames / 6)` of the candidates when there are enough, pairwise 4 or more
    steps apart, in keyframe order, taken in `stable_hash(case id, k, "ask")` order. Hand-worked on tasks of 28, 90 and
    130 frames with no decision: 5, 15 and 22 points over the candidates 4 to frames - 2, for 12 case ids each, equal to
    the hand draw. The 90 and 130 frames fill the cap; the 28 frames' 23 candidates hold five points four apart only
    in some orders of the draw, so a case there asks four or five. Regression caught: a cap taken from the length
    instead of the frames, points bunched closer than the gap, or a draw that is not the hash order."""
    script = vacuum_run() if frames == 28 else long_task(frames)
    cap = {28: 5, 90: 15, 130: 22}[frames]
    for cid in ids(f"cap{frames}", 12):
        got = points(world(script, cid, job=look_job()))
        assert got == hand_points(cid, list(range(4, frames - 1)), cap)
        assert len(got) == cap if frames > 28 else cap - 1 <= len(got) <= cap
        assert all(b - a >= LOOK_GAP_STEPS for a, b in zip(got, got[1:], strict=False))
    assert VACUUM_CAP == 5


def test_a_point_may_fall_inside_a_decision_window():
    """Contract: a look point is no longer the end of a decision window, so the draw reaches frames the stream's cuts
    do not end. Hand-worked: the vacuum's decision windows end after frames 5, 13, 18, 21 and 27; over 30 case ids
    some point falls on another frame. Regression caught: the draw restricted to the window ends again (the questions
    would sit on the same few frames in every case)."""
    ends = {5, 13, 18, 21, 27}
    drawn = {k for cid in ids("inside", 30) for k in points(world(vacuum_run(), cid, job=look_job()))}
    assert drawn - ends and drawn <= set(FRAMES)


def test_the_decision_frames_of_the_priority_kinds_are_drawn_first():
    """Contract (FD-questions-0.10 §3): the frames of the kinds in `JobSpec.look_decisions` come first in the draw, then
    the rest in hash order, so a case asks its decision frames before any other. Hand-worked on a task of 30 frames
    (5 points) with `before_doorway` decisions at 2, 6, 12, 18, 24, 28 and 29 and `room_done` decisions at 9 and 15,
    the job naming `before_doorway`: 2 is under 4 steps in and 29 is the last frame, so neither is a candidate, and
    the other five, 6, 12, 18, 24 and 28, are 4 or more apart and fill the cap: every case asks exactly them. Without
    the priority the same cases draw by hash alone, and some differ. Regression caught: a decision frame drawn only
    when its hash comes early, a decision frame that is no candidate forced in, or `room_done` taken as a priority."""
    plan = [2, 6, 12, 18, 24, 28, 29, (9, "room_done"), (15, "room_done")]
    script, want = long_task(30, decisions=plan), [6, 12, 18, 24, 28]
    first = look_job(decisions=DOORWAYS, look_decisions=("before_doorway",))
    plain = look_job(decisions=DOORWAYS)
    differ = 0
    for cid in ids("first", 20):
        assert points(world(script, cid, job=first)) == want
        got = points(world(script, cid, job=plain))
        assert got == hand_points(cid, list(range(4, 29)), 5)
        differ += got != want
    assert differ > 0


def test_priority_frames_closer_than_the_gap_keep_the_hash_first_and_the_rest_fill_after():
    """Contract: the priority frames are subject to the same gap, and once they are kept the rest of the cap is filled
    from the other candidates in hash order. Hand-worked on a task of 40 frames (7 points) with `before_doorway`
    decisions at 20 and 22 (2 apart) and 30: the hash-first of 20 and 22 is kept with 30, never both of the pair, and
    the other points come from the remaining candidates by the hand draw (at least one more: the gap may stop the draw
    short of seven). Regression caught: both of a close pair kept, or the priority frames kept and nothing drawn after
    them."""
    script = long_task(40, decisions=[20, 22, 30])
    job = look_job(decisions=DOORWAYS, look_decisions=("before_doorway",))
    for cid in ids("close", 20):
        got = points(world(script, cid, job=job))
        assert got == hand_points(cid, list(range(4, 39)), 7, first={20, 22, 30})
        assert 30 in got and len({20, 22} & set(got)) == 1 and 3 < len(got) <= 7


def test_the_draw_reads_the_case_id_and_the_plan_never_the_world():
    """Contract (design §8.5): which points are asked depends on the case id and the plan's decision points alone, so it
    reveals nothing about the world. Hand-worked: the vacuum's run with thirty things added (ten in view at keyframe
    13, ten at 21, one elsewhere), another seed and different depth draws the same points as the bare run for every id,
    with the decision frames first as well. Regression caught: a point chosen where a thing is in view."""
    plan = [(5, "before_room"), (18, "room_done"), (21, "before_doorway")]
    job = look_job(decisions={"before_room": (), "room_done": (), "before_doorway": ()},
                   look_decisions=("room_done", "before_doorway"))
    bare = vacuum_run(decisions=plan)
    busy = vacuum_run(seed=9, boxes=[], frame_boxes={19: []}, decisions=plan)
    for i in range(30):
        busy.item(f"E{i:04d}", "bag", {"color": "red"}, [(0, None, (0.5 + i * 0.1, 1.0, 0.0))],
                  size=(0.374, 0.304, 0.314))
        busy.see((13, 21, i % 28)[i % 3], f"E{i:04d}", 1500)   # ten things in view at 13, ten at 21, one elsewhere
    for cid in ids("world", 20):
        got = points(world(bare, cid, job=job))
        assert got == points(world(busy, cid, job=job)) == hand_points(cid, first={18, 21})


def test_every_point_is_a_frame_the_delivered_stream_can_be_asked_after(tmp_path):
    """Contract (the harness's own rule, `stream._during_at`): a during question is asked after any step of the window,
    and after none outside it. Written to disk as a case of two tasks and drawn for 20 case ids over its window: each
    point is a step of the delivered stream, `_during_at` puts a question at it, and over the ids some point falls on a
    step that ends no decision window. A question asked at the step after the last, or before the first, raises.
    Regression caught: `_during_at` still demanding a window end (the harness would stop on the first look question
    drawn inside a window), or accepting a step the stream does not deliver."""
    s = vacuum_run(2)
    case = case_on(tmp_path, s, [end_q(1)], a=0, b=1, length="medium")
    rec = case_mod.read_case(case)
    delivered = stream.delivered(case)
    ends = {hi - 1 for _, hi in delivered.windows}
    inside = 0
    for cid in ids("stream", 20):
        w = World(inputs(s, look_job()), replace(rec, case=cid))
        got = points(w)
        assert got
        asked = [{"qid": f"{n:016x}", "when": "during", "decision_k": k} for n, k in enumerate(got)]
        by_step = stream._during_at(asked, delivered, rec, None)
        assert sorted(by_step) == sorted(case_mod.case_k(rec, k) for k in got)
        inside += sum(case_mod.case_k(rec, k) not in ends for k in got)
    assert inside > 0
    for step in (-1, len(delivered.steps)):
        outside = {"qid": "0" * 16, "when": "during", "decision_k": rec.k0 + step}
        with pytest.raises(ValueError, match="outside the window"):
            stream._during_at([outside], delivered, rec, None)


def test_a_job_without_looks_draws_no_point():
    """Contract: `points` is empty for a job that names no look family, however long the window. Regression caught: a
    job drawing look points it never maps (a stray question for every job, the old datasets recompiling with new
    rows)."""
    assert points(world(long_task(130), "c9", "long", job=JOB)) == [] == ql.candidate_points(world(vacuum_run()))
