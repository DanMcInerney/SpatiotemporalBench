"""`person_mem` (look, `memory`): where the nearest part of the one person the robot saw is from the camera now, as last
seen. Hand-worked on `person_kit`'s scene: the look point is keyframe 13 (the look window 6-13), where the camera is at
(3.0, 1.13, 0.08) heading north, so a point (x, y) is `y - 1.13` ahead and `3.0 - x` to the left. A person is a 0.34 x
0.56 m footprint; the job's clearance is 2.0 m (rungs 1.0, 2.0, 4.0)."""

from __future__ import annotations

from dataclasses import replace

import pytest

import person_kit as pk
from ego_kit import K
from stbench import person_seen
from stbench.questionnaire import conflicts
from stbench.questions_people import person_mem

JOB = pk.one_person()


def asked(s, k=K, **kw):
    """(the answerable candidates, the unknown ones) of the look point `k` of the window ended there."""
    m = person_mem(pk.cut(s, k, JOB) if not kw else pk.world(s, JOB, **kw), k)
    return None if m is None else (m.answerable, m.unknown)


def test_the_gold_is_the_last_sightings_nearest_point_in_the_camera_frame_now():
    """Contract (FD-task-questions review, section 3.4): the person's last evidence sighting lies before the look window
    and nothing has been seen of them since, so the gold is that sighting's hull, its point nearest the camera's floor
    point at the look point, in the camera's frame there. By hand: seen at keyframe 2 at (2.0, 3.63) the hull is x
    1.83-2.17, y 3.35-3.91, and the point nearest the camera (3.0, 1.13) is the corner (2.17, 3.35): 2.22 ahead, 0.83
    left; the range is the slant to it at the floor, sqrt(0.83^2 + 2.22^2 + 0.08^2) = 2.3714. The evidence is that
    keyframe, the need C (2.0) with the radial harm and the resolution of that sighting's camera, 0.0982 (the resolution
    tests work it out), the class `answerable`, the subject `w:2` (the sighting read). Regression caught: the sighting's
    own camera used for the offset (the robot moved: (2.0, 1.63) would give 1.72 ahead), left taken as right, or the
    centre of the person in place of the nearest part."""
    s = pk.scene()
    pk.put_person(s, "A", (2.0, 3.63), [2])
    (answerable, unknown) = asked(s)
    (c,) = answerable
    assert unknown == [] and c.family == "person_mem" and c.slots == {} and c.decision_k == K
    assert c.gold == {"type": "offset", "ahead_m": 2.22, "left_m": 0.83}
    assert c.evidence_k == (2,) and c.subjects == ("w:2",) and c.entities == () and c.answer_class == "answerable"
    assert c.tolerances == {"resolution_m": pytest.approx(0.0982, abs=1e-4), "need_m": 2.0, "harm": "radial"}
    assert c.range_m == pytest.approx(2.3714, abs=1e-4)


def test_the_gold_reads_the_last_sighting_alone():
    """Contract: people shift on the spot, so no union over sightings: the gold is the latest evidence sighting's hull
    only. By hand: the person stands at (2.0, 2.63) until 4 s and at (2.0, 3.63) after; seen at keyframe 2 (3.0 s) at
    2.63 and at keyframe 4 (6.0 s) at 3.63, the gold is that of keyframe 4's hull, corner (2.17, 3.35): 2.22 ahead,
    0.83 left (a union with the first sighting would reach (2.17, 2.35): 1.22 ahead). Regression caught: a blurred
    place from the union of sightings, or the first sighting read."""
    s = pk.scene()
    pk.put_person(s, "A", [(0, 4.0, (2.0, 2.63, 0.0)), (4.0, None, (2.0, 3.63, 0.0))], [2, 4])
    ((c,), _) = asked(s)
    assert c.gold == {"type": "offset", "ahead_m": 2.22, "left_m": 0.83} and c.evidence_k == (4,)


@pytest.mark.parametrize("seen, k_row", [([2, 8], None), ([2], 9), ([2], K), ([2, K], None), ([2, 6], None)])
def test_a_point_is_asked_only_if_nothing_of_the_person_was_seen_since_the_last_sighting(seen, k_row):
    """Contract: answerable only when the latest evidence sighting lies before the look window (keyframes 6-13) and no
    row of the person, at any pixel count, falls after it up to the point. By hand: an evidence sighting inside the look
    window (8, 6 or 13) puts the latest sighting in the current view; a 300 px glimpse at keyframe 9 or 13 after the
    sighting at 2 shows the person may have moved. All leave the point unasked (not unknown: the person was seen).
    Regression caught: a stale place asked about a person who was seen walking off, or the current view answering a
    memory question."""
    s = pk.scene()
    pk.put_person(s, "A", (2.0, 3.63), seen)
    if k_row is not None:
        s.see([k_row], "A", 300)
    assert asked(s) is None


def test_unknown_only_when_the_window_up_to_the_point_holds_no_row_of_any_person():
    """Contract (FD-task-questions final review, B1): `unknown` exactly when the window up to the look point holds no
    row of any person, at any pixel count: nobody at all, a row only after the point (keyframe 20; the point is 13), or
    one in the task before a window that starts at the second task. The candidate is in the moment's `unknown` pool
    with class `unknown` and no evidence or subject. Regression caught: the whole script's record read (a place first
    seen later answers a question asked earlier)."""
    nobody = pk.scene()
    later = pk.scene()
    pk.put_person(later, "A", (2.0, 3.63), [20])
    for s in (nobody, later):
        answerable, unknown = asked(s, K, a=0, b=0) if s is later else asked(s)
        (c,) = unknown
        assert answerable == [] and c.gold == {"type": "unknown"} and c.slice == "unknown"
        assert c.answer_class == "unknown" and c.evidence_k == () and c.subjects == () and c.decision_k == K
    before = pk.scene(2)
    pk.put_person(before, "A", (2.0, 3.63), [2])
    (c,) = person_mem(pk.world(before, JOB, a=1, b=1), K + 28).unknown
    assert c.gold == {"type": "unknown"}


# The helper's clearance (0.15 m, C/2 = 7.5 cm) with the resident 1.0 m ahead of the camera's floor point: the contact
# pixel reads to 3.3 cm at one pixel there, so the tie fails only if the feet lie more than 7.5 cm behind the hull's
# nearest point. A standing person's belly or back projects about 10 cm beyond their heels (review B1), so the upper
# body's points reach y = 2.13 m (the nearest point, 1.0 m ahead of the camera at y = 1.13) while the feet start at
# 2.23.
HELPER_JOB = replace(JOB, clearance=0.15)
BELLY_M = 0.10


def belly_pts(center_y: float = 2.41):
    """A person 1.0 m ahead whose upper body is `BELLY_M` nearer the camera than their feet."""
    pts = pk.body((3.0, center_y))
    near = center_y - pk.HALF_Y
    return pts[(pts[:, 2] > 0.0) | (pts[:, 1] >= near + BELLY_M - 1e-9)]


def test_the_belly_sighting_the_rule_rejects_is_a_sighting_of_the_person():
    """Contract (FD-task-questions final review, B1; case 494174b100946ed8 keyframe 42): the person plainly seen, in
    full view with their feet on the floor, but at a sighting the evidence rule rejects (the hull's nearest point has
    no foot pixel within C/2 = 7.5 cm, their belly 10 cm before their heels) is a person the robot saw: the point asks
    nothing, where `unknown` would score -1 for the reader who answers an offset. By hand: the feet ring starts 10 cm
    behind the footprint's near edge, the upper rings do not, so the tie fails with a clear box, 1,500 px and a depth
    (checked against `person_seen.hull` itself, and passing once the feet reach the near edge). Rows of every kind
    leave the point unasked: the belly sighting, a 300 px glimpse, and both together. Regression caught: `unknown` read
    from the absence of an evidence sighting rather than of a row (46% of the helper's clear sightings failed this
    rule)."""
    s = pk.scene()
    pk.put_person(s, "A", (3.0, 2.41), [2], pts=belly_pts(), depth=1.0)
    world = pk.cut(s, K, HELPER_JOB)
    assert person_seen.hull(person_seen.frame_of(world, "A", 2), "A", HELPER_JOB.clearance_m) is None
    assert person_mem(world, K) is None
    glimpse = pk.scene()
    pk.put_person(glimpse, "A", (3.0, 2.41), [2], px=300)
    assert person_mem(pk.cut(glimpse, K, HELPER_JOB), K) is None
    both = pk.scene()
    pk.put_person(both, "A", (3.0, 2.41), [2, 4], pts=belly_pts(), depth=1.0)
    both.see([5], "A", 300)
    assert person_mem(pk.cut(both, K, HELPER_JOB), K) is None
    feet = pk.scene()
    pk.put_person(feet, "A", (3.0, 2.41), [2], depth=1.0)
    (c,) = person_mem(pk.cut(feet, K, HELPER_JOB), K).answerable
    assert c.evidence_k == (2,)


def test_a_passing_sighting_answers_unless_a_row_follows_it():
    """Contract: the answerable side reads the latest *evidence* sighting, and a rejected sighting before it does not
    displace it, while a row at any pixel count after it leaves the point unasked. By hand: evidence at keyframe 2 and a
    rejected belly sighting at keyframe 1 (before it) answer from keyframe 2, whose hull's nearest point (3.0, 2.13) is
    1.0 ahead and 0.0 left of a camera at (3.0, 1.13); the same belly sighting at keyframe 4 (after it) leaves the point
    unasked. Regression caught: a rejected sighting treated as the person's latest place, or one after the evidence
    ignored."""
    s = pk.scene()
    pk.put_person(s, "A", (3.0, 2.41), [2], depth=1.0)
    s.see([1], "A", 1500, depth=1.0)
    s.parts[1, "A"] = belly_pts()
    (c,) = person_mem(pk.cut(s, K, HELPER_JOB), K).answerable
    assert c.evidence_k == (2,) and c.gold == {"type": "offset", "ahead_m": 1.0, "left_m": 0.0}
    t = pk.scene()
    pk.put_person(t, "A", (3.0, 2.41), [2], depth=1.0)
    t.see([4], "A", 1500, depth=1.0)
    t.parts[4, "A"] = belly_pts()
    assert person_mem(pk.cut(t, K, HELPER_JOB), K) is None


def test_a_case_asks_one_sighting_once():
    """Contract (FD-task-questions final review, S4): the answerable candidate's subject names the sighting it reads, so
    the case's subject rule (`questionnaire.conflicts`) lets a case ask each sighting at most once however many look
    points follow it. By hand: the person seen at keyframe 2 only is the sighting that points 13 and 18 (the recent
    views 6-13 and 11-18) both read, so their subjects conflict; seen again at keyframe 15, the point 24 (the recent
    view 17-24) reads that sighting instead, which does not conflict with the first. Regression caught: no subject
    (one sighting asked six times in case 7347a70e0feea32f, 27 rows from 14 sightings)."""
    once = pk.scene()
    pk.put_person(once, "A", (2.0, 3.63), [2])
    first, again = (person_mem(pk.cut(once, k, JOB), k).answerable[0] for k in (K, 18))
    assert first.evidence_k == again.evidence_k == (2,) and first.subjects
    assert conflicts(first.subjects, again.subjects)
    moved = pk.scene()
    pk.put_person(moved, "A", [(0, 8.0, (2.0, 3.63, 0.0)), (8.0, None, (2.0, 3.0, 0.0))], [2, 15])
    other = person_mem(pk.cut(moved, 24, JOB), 24).answerable[0]
    assert other.evidence_k == (15,) and not conflicts(first.subjects, other.subjects)


@pytest.mark.parametrize("k, asked_at", [(13, False), (17, False), (18, True), (19, True), (27, True)])
def test_the_recent_view_slides_with_the_point_and_a_point_need_end_no_window(k, asked_at):
    """Contract (0.10.0, `look_windows.recent`): the person counts as seen "in the recent view" when the sighting lies
    in the last 8 frames of the task up to the point, whatever look window the stream cuts, so the point is asked once
    the sighting has slid out of it and not before. By hand: the person seen at keyframe 10 (evidence). The recent view
    at 13 is 6-13, at 17 it is 10-17: the sighting is in it and the point asks nothing; at 18 it is 11-18, at 19 12-19,
    at 27 20-27: the sighting is before it and the person's place as last seen is asked, evidence (10,), subject w:10,
    at the points 19 and 27, which end no decision window as well as at 18, which ends one. Regression caught: the
    window of the stream still deciding (a sighting 9 frames back called recent at a window's end, or asked at only the
    window ends), or a view that does not slide."""
    s = pk.scene()
    pk.put_person(s, "A", (2.0, 3.63), [10])
    got = asked(s, k)
    if not asked_at:
        assert got is None
        return
    (c,), unknown = got
    assert unknown == [] and (c.evidence_k, c.subjects, c.decision_k) == ((10,), ("w:10",), k)


def test_it_is_asked_only_of_a_job_with_one_person_and_refuses_a_world_with_two():
    """Contract: 'the person' names the helper's one resident, so a job that does not declare one person asks nothing,
    and a one-person job whose script holds two people is refused, naming the job and the case. Regression caught: a
    'the person' question in a world of pickers, with two candidates for one name."""
    s = pk.scene()
    pk.put_person(s, "A", (2.0, 3.63), [2])
    assert person_mem(pk.cut(s, K), K) is None
    pk.put_person(s, "B", (4.0, 3.63), [2])
    with pytest.raises(ValueError, match="declares one person.*holds 2"):
        person_mem(pk.cut(s, K, JOB), K)
