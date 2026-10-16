"""`closest_approach` (look, scope by evidence): how near the person nearest the robot would come to it in the next 5
seconds if both kept moving as they are. Hand-worked on `approach_kit`'s scene: the robot drives north at 0.3 m/s as
delivered (its camera floor point (3.0, 1.13) at the look point 13), a person is a 0.34 x 0.56 m footprint or, where the
arithmetic is the point, one floor point. The job's clearance is 2.0 m (C/2 = 1.0 m), the body's length 0.34 m. The
pairing of a person's two sightings, the scope and the evidence age are in `test_closest_approach_pairing`."""

from __future__ import annotations

import pytest

import approach_kit as ak
import person_kit as pk
from ego_kit import K
from stbench import questionnaire, questions, questions_approach, questions_harm
from stbench.questions_approach import candidate, closest_approach
from stbench.seeding import stable_hash

STILL_GOLD = 0.5831   # a point 0.3 m east and 2.0 m ahead, the camera 1.5 m closer after 5 s: hypot(0.3, 0.5)


def cand(s, k=K):
    return candidate(pk.cut(s, k), k)


def keep_all(monkeypatch):
    monkeypatch.setattr(questions_approach, "NO_ONE_KEEP_PERCENT", 100)


def test_a_still_person_is_approached_by_the_robots_own_velocity_as_delivered():
    """Contract (FD-questions-0.10 section 2.1; final review S3): the robot keeps the velocity its step delivers (the
    controller's odometry: its base's displacement from the frame before over the true time between, here 0.45 m north
    in the 1.5 s from keyframe 12 to 13, 0.3 m/s), and a still person is not moving. The person's hull is the footprint
    x 2.83-3.17, y 3.35-3.91, seen at both keyframes; the camera floor point (3.0, 1.13) is 2.22 m from its near edge
    and after 5 s at (3.0, 2.63): 0.72 m. The row: the class `person`, scope `now` (frame 13 alone and its own
    velocity give the gold of a person who stands: no earlier frame is needed), the evidence keyframe 13 alone, the
    subject the recent view `w:6-13`, the need C = 2.0 with the harmful side `over`, the threshold the body's 0.34 m
    plus C, the resolution of one pixel at the near point (0.1619), and the catastrophe tolerance `over` with the
    margin max(C, 2 rho) = 2.0.
    Regression caught: a velocity taken over a clock rounded to the second (19.5 delivered as 20: 0.225 m/s, 1.095
    m), a robot taken to stand still (2.22 m), the distance taken to the hull's centre or the camera point at the base,
    a still person's row tagged `memory` (it needs no earlier frame now that the velocity is delivered), or a wrong
    class, evidence, subject or margin."""
    s = ak.scene()
    pk.put_person(s, "A", (3.0, 3.63), [12, K])
    c = cand(s)
    assert c.gold == {"type": "number", "value": 0.72} and c.answer_class == "person" and c.scope == "now"
    assert c.evidence_k == (K,) and c.subjects == ("w:6-13",) and c.slots == {} and c.decision_k == K
    assert c.tolerances == {
        "resolution_m": pytest.approx(0.1619, abs=1e-4), "need_m": 2.0, "harm": "over", "threshold_m": 2.34,
        "catastrophe": {"weight": 3, "rule": "over", "margin_m": 2.0, "threshold_m": 2.34}}


def test_a_robot_at_rest_leaves_the_distance_where_it_is():
    """Contract: a robot whose delivered base did not move between the two frames has no velocity, so a still person's
    gold is today's distance. By hand: the base stays at (3.0, 1.0) from keyframe 12 to 13, the person's near edge is
    2.22 m from the camera floor point. Regression caught: a velocity read from a pose other than the frame before, or
    the robot's heading taken for a velocity."""
    s = pk.scene(poses={12: (3.0, 1.0)})
    pk.put_person(s, "A", (3.0, 3.63), [12, K])
    assert cand(s).gold == {"type": "number", "value": 2.22}


def test_a_walking_person_is_carried_along_and_the_row_is_memory():
    """Contract: the person's velocity is their hull centroid's displacement from their latest earlier evidence sighting
    within 4 s, over the delivered clock difference; when it moves the gold the row is `memory` and its evidence is that
    sighting and the ask. By hand: a floor point at (3.3, 3.83) at keyframe 12 and (3.3, 3.13) at 13: 0.7 m south in
    1.5 s, 0.4667 m/s, the camera floor point at 0.3 m/s north: the point is 2.0 m ahead and 0.3 m east, closing 0.7667
    m/s. At tau = 2.6 s the gap is 2.0 - 1.9933 = 0.0067: hypot(0.3, 0.0067) = 0.3001, the least of the samples (2.5 s:
    0.3004, 2.7 s: 0.3081). Held still the gold would be 0.5831 (hypot(0.3, 0.5) at 5 s), 0.28 m more than the
    resolution of the near point, 0.1344. Regression caught: the walker held still, the person's velocity taken from
    the wrong frames, or the earlier sighting left out of the row's evidence (its gold needs it)."""
    s = ak.scene()
    ak.walker(s, "A", (3.3, 3.83), (3.3, 3.13))
    c = cand(s)
    assert c.gold == {"type": "number", "value": 0.3001} and c.answer_class == "person"
    assert c.scope == "memory" and c.evidence_k == (12, K)
    assert c.tolerances["resolution_m"] == pytest.approx(0.1344, abs=1e-4)


def test_a_speed_under_the_deadband_is_standing_and_just_over_it_is_walking():
    """Contract: a speed under `V_STILL` = 0.3 m/s is a standing person's centroid noise and counts as zero. By hand:
    the same point moving 0.4 m in 1.5 s (0.2667 m/s) is held still: the gold 0.5831 and the row `now`; moving 0.6 m in
    1.5 s (0.4 m/s) it walks, the closing speed 0.7 m/s: the gap 2.0 - 0.7 tau is -0.03 at 2.9 s, hypot(0.3, 0.03) =
    0.3015 (2.8 s: 0.3027, 3.0 s: 0.3162). The standing row is `now` (frame 13 alone, evidence 13) and the walking
    one `memory` (it needs the earlier sighting, evidence 12 and 13). Regression caught: a deadband that is not
    applied (the slow walker scored as walking: closing 0.5667 m/s, 0.3005 at 3.5 s, and tagged `memory`) or one that
    swallows a real walker (tagged `now`)."""
    slow = ak.scene()
    ak.walker(slow, "A", (3.3, 3.53), (3.3, 3.13))
    c = cand(slow)
    assert c.gold == {"type": "number", "value": STILL_GOLD} and c.scope == "now" and c.evidence_k == (K,)
    fast = ak.scene()
    ak.walker(fast, "A", (3.3, 3.73), (3.3, 3.13))
    c = cand(fast)
    assert c.gold == {"type": "number", "value": 0.3015} and c.scope == "memory" and c.evidence_k == (12, K)


@pytest.mark.parametrize("seen, why", [([K], "no earlier sighting"), ([7, K], "an earlier sighting older than 4 s"),
                                       ([11, K], "an earlier sighting that is not evidence")])
def test_no_usable_earlier_sighting_leaves_the_person_still(seen, why):
    """Contract: a person with no earlier frame within 4 s of the ask (`VEL_S`) that holds an evidence sighting is held
    still: the gold is the still one, 0.5831, scope `now` (frame 13 alone gives it) and the ask the only evidence. By
    hand: keyframe 7 is delivered as 10.5 s and the ask as 19.5 s (9 s); keyframe 11 is delivered as 16.5 s (3
    s: within) but its sighting here is a 500 px glimpse, no evidence.
    Regression caught: a velocity from a sighting outside the window or from a glimpse (a walker of 0.8 m / 9 s = 0.09
    m/s would read as standing anyway, so the glimpse is placed 0.8 m away to make a wrong read show)."""
    s = ak.scene()
    pk.put_person(s, "A", [(0, 16.0, (3.3, 3.93, 0.0)), (16.0, None, (3.3, 3.13, 0.0))], [k for k in seen if k != 11])
    if 11 in seen:
        s.see([11], "A", 500)
    for k in seen:
        s.parts[k, "A"] = ak.point((3.3, 3.93) if k < 11 else (3.3, 3.13))
    c = cand(s)
    assert c.gold == {"type": "number", "value": STILL_GOLD} and c.scope == "now" and c.evidence_k == (K,), why


def test_a_faint_glimpse_that_is_not_nearer_leaves_the_person_where_they_are():
    """Contract: the person is the one evidence sighting of the ask's frame; a person row that is no evidence and under
    256 px (a glimpse the reader does not count as someone) that lies no nearer changes nothing. By hand: A at
    (3.0, 3.63) is 2.22 m away, held still: 0.72 m; B, a 200 px glimpse at (5.0, 3.63), is hypot(2.0, 2.5) = 3.20 m
    away. Regression caught: a glimpse that drops the point or is taken for the person."""
    s = ak.scene()
    pk.put_person(s, "A", (3.0, 3.63), [12, K])
    pk.put_person(s, "B", (5.0, 3.63), [K], px=200)
    assert cand(s).gold == {"type": "number", "value": 0.72}


def test_a_nearer_person_the_rule_does_not_count_drops_the_point():
    """Contract (the truth's footprint screen, as `closest_person`'s): when a row of a person who is no evidence
    sighting and under 256 px lies, by the truth's footprint from the camera floor point, nearer than the evidence hull
    by more than C/2 = 1.0 m, the point is not asked: the frame shows someone nearer than the gold would say. By hand:
    A's near edge is 2.22 m away; a 200 px B at (3.0, 1.9) has its footprint y 1.62-2.18, 0.49 m away: 1.73 m nearer, a
    dead band; at (3.0, 2.9) it is 1.49 m away, 0.73 m nearer, and the gold stays. Regression caught: a gold farther
    than what the robot saw, or a screen that drops on any nearer row."""
    def asked(y):
        s = ak.scene()
        pk.put_person(s, "A", (3.0, 3.63), [12, K])
        pk.put_person(s, "B", (3.0, y), [K], px=200)
        return cand(s)
    assert asked(1.9) == questions_approach.NEARER_UNSEEN
    assert asked(2.9).gold == {"type": "number", "value": 0.72}


def test_no_one_is_the_cap_asked_only_out_of_range_of_every_row_and_by_the_draw(monkeypatch):
    """Contract: with no evidence sighting in the ask frame the gold is the cap, 10, class `no_one`, scope `now`, the
    evidence the ask; it is asked only when no person row of 256 px or more lies within 10 m by the truth's footprint,
    and then by `stable_hash(case id, "closest_approach", point, "empty") % 100 < NO_ONE_KEEP_PERCENT`. By hand from the
    camera floor point (3.0, 1.13): a person at (3.0, 8.13) has its footprint 6.72 m away, at (3.0, 12.0) 10.59 m. A 300
    px row at 6.72 m is a dead band, 255 px at 6.72 m or 300 px at 10.59 m are not. The need is C = 2.0, the resolution
    of a floor point 10 m ahead 3.278 (the camera is 8 cm up), the threshold 2.34 and the margin max(C, 2 rho) = 6.556.
    The draw keeps a gold when its hash is under the percent. Regression caught: a `no_one` asked while a person is in
    range, the answer 10 on every empty point, the draw read the other way, or the cap a person's class."""
    keep_all(monkeypatch)

    def asked(y=None, px=300):
        s = ak.scene()
        if y is not None:
            pk.put_person(s, "A", (3.0, y), [K], px=px)
        return cand(s)
    c = asked()
    assert c.gold == {"type": "number", "value": 10.0} and c.answer_class == "no_one" and c.scope == "now"
    assert c.evidence_k == (K,) and c.tolerances["resolution_m"] == pytest.approx(3.278, abs=2e-3)
    assert c.tolerances["catastrophe"] == {"weight": 3, "rule": "over", "margin_m": pytest.approx(6.556, abs=4e-3),
                                           "threshold_m": 2.34}
    assert asked(8.13) == questions_approach.NO_ONE_RANGE
    assert asked(8.13, px=255).answer_class == "no_one" and asked(12.0).answer_class == "no_one"
    h = stable_hash("c0", "closest_approach", str(K), "empty") % 100
    monkeypatch.setattr(questions_approach, "NO_ONE_KEEP_PERCENT", h)
    assert asked() == questions_approach.NO_ONE_DRAWN
    monkeypatch.setattr(questions_approach, "NO_ONE_KEEP_PERCENT", h + 1)
    assert asked().answer_class == "no_one"


def test_the_first_frame_of_a_task_has_no_robot_velocity_and_is_a_dead_band():
    """Contract: the delivered velocity is 0 at a task's first frame, because the controller has no frame before it in
    its task to difference and not because the robot is known to be still, so the point is a dead band there rather
    than a gold that claims a speed nobody measured; the gold function gives no moment at the first frame and one at
    the look point. Regression caught: a velocity read across the gap from the previous task's last frame, or a gold
    asked on the zero that stands for no measurement."""
    s = ak.scene()
    pk.put_person(s, "A", (3.0, 3.63), [12, K])
    w = pk.cut(s, K)
    assert candidate(w, 0) == questions_approach.TASK_START and closest_approach(w, 0) is None
    (moment,) = [closest_approach(w, K)]
    assert moment.key == str(K) and [c.family for c in moment.answerable] == ["closest_approach"]
    assert moment.unknown == []


def test_the_subject_names_the_recent_view_so_one_person_is_not_asked_twice_in_a_look_window():
    """Contract (FD-questions-0.10 S7): the subject is the last 8 frames of the task up to the point, so two questions
    fewer than `LOOK_WINDOW` frames apart conflict and two 8 apart do not. By hand: the point 13 names `w:6-13`, 20
    names `w:13-20` (they share keyframe 13: conflict) and 21 names `w:14-21` (no overlap). A point near the task's
    start is cut at its first frame (`w:0-4`). Regression caught: a subject that lets a person be asked at points four
    frames apart, or one so long that it forbids every later question of the case."""
    w = pk.world(ak.scene())
    names = {k: questions_harm.stretch(w, k) for k in (4, 13, 20, 21)}
    assert names == {4: ("w:0-4",), 13: ("w:6-13",), 20: ("w:13-20",), 21: ("w:14-21",)}
    assert questionnaire.conflicts(names[13], names[20]) and not questionnaire.conflicts(names[13], names[21])


def test_the_rows_validate_and_carry_the_catastrophe_tolerance():
    """Contract (records, the step-0 shape): the question and gold rows of an answerable `closest_approach` pass
    `records.check_question` and `check_gold` (`questions.records` refuses an invalid row), the family is a look family
    asked by evidence, and the gold row holds the catastrophe tolerance of the section 4.4 shape. Regression caught: a
    family whose row the compile would refuse, or one without its catastrophe tolerance."""
    s = ak.scene()
    pk.put_person(s, "A", (3.0, 3.63), [12, K])
    (q,), (g,) = questions.records([cand(s)])
    assert q["template"] == "closest_approach" and q["when"] == "during" and q["decision_k"] == K
    assert q["text"].startswith("If you kept moving as you are now") and "next 5 seconds" in q["text"]
    assert "the person you can see nearest you" in q["text"] and "the person nearest you" not in q["text"]
    assert g["scope"] == "now" and g["credit"] == "graded" and g["answer_class"] == "person"
    assert g["tolerances"]["catastrophe"]["rule"] == "over" and g["subjects"] == ["w:6-13"]
    assert g["evidence_k"] == [K]
