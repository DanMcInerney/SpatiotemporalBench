"""The evidence age on every gold row (`questions_age`, 0.10.0, FD-questions-0.10 §4.5, review S5): `age_s`, the clock
seconds from the gold's oldest evidence keyframe to the ask, and `age_band`, hand-checked on a two-task synthetic
vacuum (28 keyframes a task, 1.5 s apart within a task, the second task beginning at t = 7200 s)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from family_kit import fake_cand, fake_spec, multi_run, register, world
from stbench import questions, questions_age
from stbench.questions_select import Cand
from stbench.records import AGE_BANDS, check_gold


@pytest.fixture
def two_tasks(monkeypatch):
    """The window of both tasks (keyframes 0 to 55, the last at t = 7240.5 s) with a during family asked of the
    memory (`fake_d`) and of the present view (`fake_n`) and an end family (`fake_e`)."""
    register(monkeypatch, fake_spec("fake_d", "during"), lambda w, k: None)
    register(monkeypatch, fake_spec("fake_n", "during", scope="now"), lambda w, k: None)
    register(monkeypatch, fake_spec("fake_e", "end"), lambda w: [])
    return world(multi_run(2), 0, 1, length="medium")


def cand(family: str, evidence: tuple[int, ...], decision_k: int | None = None, *, unknown: bool = False) -> Cand:
    return replace(fake_cand(family, 1, decision_k=decision_k, unknown=unknown), evidence_k=evidence)


def aged(w, *cands: Cand) -> list[tuple[float | None, str | None]]:
    qs, golds = questions.records(list(cands), w)
    assert all(check_gold(g) == [] for g in golds)
    return [(g["age_s"], g["age_band"]) for g in golds]


def test_a_during_question_is_aged_from_the_oldest_evidence_keyframe_to_its_decision_point(two_tasks):
    """Contract (§4.5): `age_s` is the timeline's time of the ask (a during question's `decision_k`) less that of the
    gold's OLDEST evidence keyframe, and the band is `recent` up to 10 s, `task` when the evidence is in the task the
    ask falls in, `earlier` when it is in a task before. Ask at keyframe 21 (t = 31.5 s): evidence 15 (22.5 s) is 9.0 s
    old, recent; evidence 14 (21.0 s) 10.5 s old, just past the bar, so `task`; evidence 12 (18.0 s) 13.5 s, `task`;
    the pair (20, 12) is 13.5 s old by its oldest. Ask at 33 (t = 7207.5 s, the second task): evidence 12 is 7189.5 s
    old, in the first task, `earlier`; 29 (7201.5 s) 6.0 s old, recent; and at 40 (7218.0 s) the same keyframe is 16.5
    s old, `task`. Regression caught: the newest keyframe taken for the oldest, the bar read as exclusive or in frames,
    or a band that ignores the task boundary."""
    w = two_tasks
    assert aged(w, cand("fake_d", (15,), 21), cand("fake_d", (14,), 21), cand("fake_d", (12,), 21),
                cand("fake_d", (12, 20), 21), cand("fake_d", (12,), 33), cand("fake_d", (29,), 33),
                cand("fake_d", (29,), 40)) == [
        (9.0, "recent"), (10.5, "task"), (13.5, "task"), (13.5, "task"), (7189.5, "earlier"), (6.0, "recent"),
        (16.5, "task")]


def test_an_end_question_is_aged_to_the_windows_last_frame(two_tasks):
    """Contract (§4.5): an end question is asked after the window's last frame (keyframe 55, t = 7240.5 s). Evidence 50
    (7233.0 s) is 7.5 s old, `recent`; evidence 30 (7203.0 s) 37.5 s old and in the last task, `task`; evidence 3
    (4.5 s) 7236.0 s old and in the first, `earlier`. Regression caught: an end question aged to a decision point (its
    `decision_k` is null) or to the first frame."""
    assert aged(two_tasks, cand("fake_e", (50,)), cand("fake_e", (30,)), cand("fake_e", (3,))) == [
        (7.5, "recent"), (37.5, "task"), (7236.0, "earlier")]


def test_the_band_is_now_exactly_when_the_rows_scope_is_now_and_its_age_is_zero(two_tasks):
    """Contract (§4.5; final review S6): the band is `now` exactly when the scope is `now`, and a `now` row is answered
    by the latest frame, so its age is 0.0 s whatever evidence keyframe the candidate names (`open_heading` and
    `clear_path` `now` rows cite an earlier frame whose cells alone give the gold: 35 and 9 rows of the 0.9.0 data had
    ages above 0); a `memory` row is never `now`, even with evidence in the frame asked at (age 0.0 s, `recent`).
    Hand-worked: ask at keyframe 21 (31.5 s), a `now` candidate citing 21, 12 (18.0 s: 13.5 s before the ask) and 3
    (4.5 s) gets (0.0, `now`) in all three, an end `now` row too; the memory row citing 21 is (0.0, `recent`) and the
    one citing 12 (13.5, `task`). Regression caught: an age counted from the evidence keyframe of a `now` row, a band
    computed from the age alone (a present-view row called `recent`, a zero-age memory row `now`)."""
    now = [cand("fake_n", (21,), 21), cand("fake_n", (12,), 21), cand("fake_n", (3,), 21)]
    assert aged(two_tasks, *now, cand("fake_d", (21,), 21), cand("fake_d", (12,), 21)) == [
        (0.0, "now")] * 3 + [(0.0, "recent"), (13.5, "task")]
    assert questions_age.age(two_tasks, cand("fake_n", (3,), 21), "during", "now") == (0.0, "now")
    assert questions_age.age(two_tasks, cand("fake_n", (3,)), "end", "now") == (0.0, "now")


def test_an_unknown_row_and_a_candidate_with_no_evidence_have_no_age(two_tasks):
    """Contract (§4.5): both keys are null on an `unknown` row (it names no evidence), even one that does name a
    keyframe, and, since there is no keyframe to age, on a candidate that names none (`records` would refuse the row, an
    answerable gold needing evidence). Regression caught: a made-up age on a row that cites nothing."""
    assert aged(two_tasks, cand("fake_d", (), 21, unknown=True), cand("fake_e", (), unknown=True),
                cand("fake_d", (12,), 21, unknown=True)) == [(None, None)] * 3
    assert questions_age.age(two_tasks, cand("fake_e", ()), "end", "memory") == (None, None)


def test_a_candidate_compiled_without_a_world_has_a_null_age(monkeypatch):
    """Contract: `questions.records` without the world (a hand-built candidate in a test) writes both keys, null, so the
    row passes `check_gold` and keeps the key order. Regression caught: the keys missing from such a row."""
    register(monkeypatch, fake_spec("fake_e", "end"), lambda w: [])
    _, (g,) = questions.records([cand("fake_e", (3,))])
    assert (g["age_s"], g["age_band"]) == (None, None) and check_gold(g) == []
    assert AGE_BANDS == ("now", "recent", "task", "earlier")
