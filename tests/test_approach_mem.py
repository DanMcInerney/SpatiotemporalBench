"""`approach_mem` (look, `memory`, 0.11.0): how close the robot, kept moving as it is, would come in the next 5 seconds
to the person it saw last, where it last saw them: `closest_approach` for the person now out of view. Hand-worked on
`approach_kit`'s scene: the robot drives north at 0.3 m/s as delivered (camera floor point (3.0, 1.13) at the look point
13), the person's one sighting is keyframe 10 (15.0 s, 4.5 s before the ask) and they stand at (3.5, 2.0), a 0.34 x 0.56
m footprint (x 3.33-3.67, y 1.72-2.28). The job's clearance is 2.0 m (C/2 = 1.0 m), the body's length 0.34 m."""

from __future__ import annotations

import pytest

import approach_kit as ak
import person_kit as pk
from ego_kit import K
from stbench import approach_rule as rule
from stbench import brief, job_registry, questions, questions_approach, questions_looks
from stbench.questions_approach import approach_mem
from stbench.records import BALANCE_POOLS, pool_of
from stbench.seeding import stable_hash

GOLD = 0.33   # the camera floor point (3.0, 1.13 + 0.3 tau) passes 0.33 m west of the footprint from 1.97 s to 3.83 s


def stand(s, pid="A", at=ak.AT, seen=(ak.J,), px=1500):
    """The person `pid` standing at `at`, sighted at the keyframes `seen`."""
    return pk.put_person(s, pid, at, list(seen), px=px)


def test_the_gold_is_the_last_sighting_carried_along_the_robots_own_motion():
    """Contract (FD-0.11-hard 3.2 step 8): the person is the one of the latest evidence sighting before the ask, taken
    to stand where they were seen, and the gold is `closest_m` from the camera floor point at the ask, moved at the
    delivered velocity, to that hull. By hand: the camera is at (3.0, 1.13 + 0.3 tau); the footprint of the sighting
    (3.5, 2.0) is x 3.33-3.67, y 1.72-2.28, so for tau from 1.97 s to 3.83 s the camera is level with it and 0.33 m to
    its west: 0.33, the least of the samples (at 5 s the camera is 0.35 m past the corner: hypot(0.33, 0.35) = 0.48).
    Out of view the person walks 0.5 m east, to (4.0, 2.0) at 17.0 s, under C/2 = 1.0 m, so the row is asked and the
    gold stays the sighting's: the world at the ask would give 0.83. Row: class `person`, scope `memory`, evidence the
    sighting and the ask, need C = 2.0, `over`, threshold 0.34 + 2.0, the resolution of one pixel at the sighting's
    near point from the camera of keyframe 10 (2.24 m away: (2.24^2 + 0.08^2) / (0.08 * 381.4) = 0.165), and the
    catastrophe margin max(C, 2 rho) = 2.0.
    Regression caught: the truth read at the ask (0.83), the robot taken at rest (0.676), `closest_approach`'s cap when
    no one is in view (10), the resolution taken from the ask's camera (0.015) or a wrong class, scope or evidence."""
    s = ak.scene()
    pk.put_person(s, "A", [(0, 17.0, (*ak.AT, 0.0)), (17.0, None, (4.0, 2.0, 0.0))], [ak.J])
    c = ak.mem(s)
    assert c.gold == {"type": "number", "value": GOLD} and c.answer_class == "person" and c.scope == "memory"
    assert c.evidence_k == (ak.J, K) and c.decision_k == K and c.slots == {} and c.entities == ()
    assert c.tolerances == {
        "resolution_m": pytest.approx(0.165, abs=3e-3), "need_m": 2.0, "harm": "over", "threshold_m": 2.34,
        "catastrophe": {"weight": 3, "rule": "over", "margin_m": 2.0, "threshold_m": 2.34}}


def test_the_row_names_the_sighting_and_validates():
    """Contract (the record rules of 0.11.0): the subject is the sighting, `a:10` for keyframe 10, not the recent view,
    so a case asks one sighting once and the question does not conflict with a `closest_approach` stretch; the row
    passes `records.check_question` and `check_gold` (`questions.records` refuses an invalid one, so `a:<k>` is a
    valid subject); the age counts from the sighting, 19.5 - 15.0 = 4.5 s, `recent`; and the text is the twin's with
    the person last seen. Regression caught: a subject the validator refuses (the family could not be compiled), the
    recent view as the subject (two sightings asked as one), an age counted from the ask, or the text drifting from
    the design."""
    s = ak.scene()
    stand(s)
    w = pk.cut(s, K)
    (q,), (g,) = questions.records([ak.mem(s)], w)
    assert q["template"] == "approach_mem" and q["when"] == "during" and q["decision_k"] == K
    assert q["text"] == ("If you kept moving as you are now, how close would you come in the next 5 seconds to the "
                         "person you saw last, where you last saw them? Answer the smallest distance in metres along "
                         "the floor from the point below your camera to that person; answer 10 if you would not "
                         "come within 10 m.")
    assert g["subjects"] == ["a:10"] and g["evidence_k"] == [ak.J, K] and g["scope"] == "memory"
    assert g["age_s"] == 4.5 and g["age_band"] == "recent" and g["credit"] == "graded" and g["answer_class"] == "person"


def _in_view():
    s = ak.scene()
    stand(s, seen=(ak.J, K))
    return s


def _none_seen():
    s = ak.scene()
    stand(s, px=500)   # a 500 px glimpse: shown, not placed
    return s


def _two_people():
    s = ak.scene()
    stand(s)
    stand(s, "B", (6.0, 5.0))
    return s


def _unplaced_then():
    s = ak.scene()
    stand(s)
    stand(s, "B", (6.0, 5.0), px=300)
    return s


def _unplaced_since():
    s = ak.scene()
    stand(s)
    stand(s, "B", (6.0, 5.0), seen=(12,), px=300)
    return s


def _walking(*, ambiguous=False):
    s = ak.scene()
    pk.put_person(s, "A", [(0, 14.0, (3.5, 2.9, 0.0)), (14.0, None, (*ak.AT, 0.0))], [9, ak.J])
    if ambiguous:
        stand(s, "B", (6.0, 5.0), seen=(9,))
    return s


def _moved(then):
    s = ak.scene()
    placements = [(0, 17.0, (*ak.AT, 0.0))] + ([] if then is None else [(17.0, None, (*then, 0.0))])
    pk.put_person(s, "A", placements, [ak.J])
    return s


def _glimpse(seen, px):
    """The base world and a row of B (a floor-hull-less glimpse at `px`) at the keyframes `seen`."""
    s = ak.scene()
    stand(s)
    stand(s, "B", (6.0, 5.0), seen=seen, px=px)
    return s


def _slow_walker():
    """A seen at keyframe 9 at (3.5, 2.4) and at keyframe 10 at (3.5, 2.0): 0.4 m in 1.5 s."""
    s = ak.scene()
    pk.put_person(s, "A", [(0, 14.0, (3.5, 2.4, 0.0)), (14.0, None, (*ak.AT, 0.0))], [9, ak.J])
    return s


@pytest.mark.parametrize("make, band", [
    (_in_view, questions_approach.IN_VIEW),
    (_none_seen, questions_approach.NONE_SEEN),
    (ak.scene, questions_approach.NONE_SEEN),
    (_two_people, questions_approach.TWO_PEOPLE),
    (_unplaced_then, questions_approach.UNPLACED),
    (_unplaced_since, questions_approach.UNPLACED_SINCE),
    (_walking, questions_approach.WALKING),
    (lambda: _walking(ambiguous=True), questions_approach.WALKING),
    (lambda: _moved((3.5, 3.5)), questions_approach.MOVED_UNSEEN),
    (lambda: _moved(None), questions_approach.MOVED_UNSEEN),
])
def test_the_dead_bands_each_return_their_reason(make, band):
    """Contract (FD-0.11-hard 3.2 steps 2-7): a point is asked only when the sensors say whom the robot saw last and
    that they stand, so each case below returns its reason and no moment. By hand, on the base world (the person seen
    only at keyframe 10 at (3.5, 2.0)): `IN_VIEW` - the person is also in view at the ask (a row of 256 px or more is
    `closest_approach`'s point); `NONE_SEEN` - the only sighting is a 500 px glimpse, which places no one, or there is
    no person; `TWO_PEOPLE` - B is an evidence sighting at keyframe 10 too, so the latest sighting names no one;
    `UNPLACED` - B is a 300 px row at keyframe 10, shown and not placed; `UNPLACED_SINCE` - the same row at keyframe
    12, after the sighting; `WALKING` - A is also seen at keyframe 9 at (3.5, 2.9), 0.9 m in 1.5 s = 0.6 m/s over
    `V_STILL` = 0.3, or B is there too (two hulls at the earlier frame: the pairing is ambiguous); `MOVED_UNSEEN` -
    the truth puts A 1.5 m from the sighting at the ask (C/2 = 1.0), or nowhere (gone at 17.0 s).
    Regression caught: a gold that claims what the sensors lack (two people paired by the truth, a walker's place, a
    person shown and not placed) or the truth's world now where the sighting is wrong."""
    s = make()
    assert ak.mem(s) == band
    assert approach_mem(pk.cut(s, K), K) is None


@pytest.mark.parametrize("make, why", [
    (lambda: _moved((4.0, 2.0)), "moved 0.5 m out of view, under C/2 = 1.0"),
    (lambda: _glimpse((12,), 255), "a 255 px row since the sighting is not a person shown"),
    (lambda: _glimpse((9,), 300), "a 300 px row before the sighting is none of the pairing's frames"),
    (lambda: _glimpse((K,), 255), "a 255 px row at the ask is not in view"),
    (_slow_walker, "0.4 m in 1.5 s = 0.267 m/s is under V_STILL"),
])
def test_the_points_just_inside_the_dead_bands_are_asked(make, why):
    """Contract: each dead band stops where the rule says. By hand, with the base world's gold 0.33: A moved 0.5 m
    while out of view (under C/2 = 1.0); a glimpse of B at 255 px (under `NO_ONE_PX` = 256) since the sighting or at
    the ask; a 300 px glimpse of B at keyframe 9, before the sighting (the rule reads the frames from the sighting to
    the ask, and the sighting's own); A seen at keyframe 9 at (3.5, 2.4) and keyframe 10 at (3.5, 2.0), 0.4 m in 1.5 s
    = 0.267 m/s, which is standing. Each is asked with the gold 0.33. Regression caught: a band that swallows the row
    beside it (a threshold off by one pixel, a frame read that is not the rule's)."""
    c = ak.mem(make())
    assert not isinstance(c, str), why
    assert c.gold == {"type": "number", "value": GOLD}, why


def test_a_task_start_has_no_velocity_and_the_kit_world_has_its_dead_bands_where_they_are():
    """Contract: the first frame of a task is a dead band (`TASK_START`: its delivered velocity is 0 because nothing
    differences there), and over the look points 4 to 26 of the base world the point is a dead band exactly where the
    rule says. By hand: the person is seen only at keyframe 10, so keyframes 4-9 have seen no one (`NONE_SEEN`, 6),
    keyframe 10 has them in view (`IN_VIEW`, 1), and keyframes 11-26 (16) are asked: nothing is shown since, they stood,
    and the sighting is the latest. Regression caught: a velocity taken across the gap from another task, a sighting
    read from a frame after the ask, or a point asked before anyone was seen."""
    s = ak.scene()
    stand(s)
    assert questions_approach.mem_candidate(pk.world(s), 0) == questions_approach.TASK_START
    bands = {k: ak.mem(s, k) for k in range(4, 27)}
    names = [b if isinstance(b, str) else "asked" for b in bands.values()]
    assert names.count(questions_approach.NONE_SEEN) == 6 and names.count(questions_approach.IN_VIEW) == 1
    assert names.count("asked") == 16 and all(bands[k] == questions_approach.NONE_SEEN for k in range(4, 10))
    assert bands[10] == questions_approach.IN_VIEW and all(not isinstance(bands[k], str) for k in range(11, 27))


def test_a_person_the_robot_leaves_far_behind_is_the_cap_and_a_no_one_kept_by_the_draw(monkeypatch):
    """Contract (FD-0.11-hard 3.2, the no-one keep): a last-seen person who would not come within 10 m has the gold
    10, the class `no_one`, and is asked when `stable_hash(case id, "approach_mem", point, "empty") % 100` is under
    `APPROACH_MEM_NO_ONE_KEEP_PERCENT`, a draw of its own family. By hand: the base is put at (13.0, 1.5) at keyframe
    10 and the person at (16.0, 1.5): 2.8 m from that camera, and from the camera floor point (3.0, 1.13) at the ask,
    moving 1.5 m north, 12.8 m or more away. The need is C = 2.0, the margin max(C, 2 rho) with rho one pixel at
    2.8 m. The draw keeps a gold when its hash is under the percent. Regression caught: the cap a person's class, the
    draw read the other way, or one family's draw shared with `closest_approach`'s."""
    monkeypatch.setattr(questions_approach, "APPROACH_MEM_NO_ONE_KEEP_PERCENT", 100)
    s = ak.scene(poses={ak.J: (13.0, 1.5)})
    stand(s, at=(16.0, 1.5))
    c = ak.mem(s)
    assert c.gold == {"type": "number", "value": 10.0} and c.answer_class == "no_one" and c.scope == "memory"
    assert c.evidence_k == (ak.J, K) and c.tolerances["catastrophe"]["margin_m"] == 2.0
    h = stable_hash("c0", "approach_mem", str(K), "empty") % 100
    monkeypatch.setattr(questions_approach, "APPROACH_MEM_NO_ONE_KEEP_PERCENT", h)
    assert ak.mem(s) == questions_approach.NO_ONE_DRAWN
    monkeypatch.setattr(questions_approach, "APPROACH_MEM_NO_ONE_KEEP_PERCENT", h + 1)
    assert ak.mem(s).answer_class == "no_one"
    assert stable_hash("c0", "approach_mem", str(K), "empty") != stable_hash("c0", "closest_approach", str(K), "empty")


@pytest.mark.parametrize("job_id", ["amr/1", "helper/1"])
def test_a_case_draws_it_in_the_third_slot_at_its_declared_share(job_id):
    """Contract (FD-0.11-hard 3.2, "Who asks it"): over 1,000 look points a case's third slot (`slot_family`'s hash of
    case, point and slot 2) draws `approach_mem` at about the share the job declares (50% on the AMR, 30% on the helper;
    the job's own table is read, not retyped), the family has the balance pools `person` and `no_one`, each class in
    its own, and its phrase in the brief is `closest_approach`'s, said once. The slot tables, the looks, the hazard
    wiring, the ability and the kind are pinned where they are owned: `test_jobs_decisions.TABLES`,
    `test_jobs_ask_060.EXPECTED_LOOKS`, `test_job_look_tables.HAZARDS` and `test_abilities*`.
    Regression caught: a share declared and not drawn, a family with no balance pools (the balance check would read
    nothing of it), or a phrase said twice to a job that asks both `closest_approach` and `approach_mem`."""
    job = job_registry.get(job_id)
    slot = job.look_slots[2]
    drawn = sum(questions_looks.slot_family("c0", k, 2, slot) == "approach_mem" for k in range(1000))
    assert abs(drawn - 10 * dict(slot)["approach_mem"]) <= 40
    assert set(BALANCE_POOLS["approach_mem"]) == {"person", "no_one"}
    assert pool_of("approach_mem", "person") == "person" and pool_of("approach_mem", "no_one") == "no_one"
    assert brief.during_sentence(job).count("how close something would come to the robot") == 1


def test_last_seen_reads_hulls_by_step_and_names_no_one_when_the_latest_holds_two():
    """Contract (`approach_rule.last_seen`, FD-0.11-hard 3.2): the person last seen before step k is the one hull of
    the latest step of the task before k that holds a hull; it is None when no step does, and when the latest holds
    two (the sensors give no identity, so a reader cannot say which was seen last, and it does not fall back to an
    earlier step). By hand: hulls A at steps 3 and 9, B at 5, A and B together at 7. Before k = 4: step 3, A. Before
    6: step 5, B. Before 8: step 7 holds two: None (not B, not step 5's). Before 10: step 9, A. Before 3: none. With
    the task's steps 4 to 7 only, step 3 is outside it: k = 4 has none and k = 6 has step 5's B. A step with an empty
    list holds none. The arithmetic of the design's example is the same rule: a robot at (0, 0) moving +x at 1 m/s
    and a 0.4 m square centred at (4, 0.5) come within 0.3 m (tau 3.8 s to 4.2 s), and at rest hypot(3.8, 0.3) =
    3.8118 away. Regression caught: a track's identity used for the person, a fall back to an older sighting when the
    latest is ambiguous, or a step of another task read."""
    a = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    b = [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]]
    held = {3: [a], 5: [b], 7: [a, b], 9: [a], 11: []}
    steps = range(0, 12)
    assert rule.last_seen(held, 4, steps) == (3, a) and rule.last_seen(held, 6, steps) == (5, b)
    assert rule.last_seen(held, 8, steps) is None and rule.last_seen(held, 10, steps) == (9, a)
    assert rule.last_seen(held, 3, steps) is None and rule.last_seen(held, 12, steps) == (9, a)
    assert rule.last_seen(held, 4, range(4, 8)) is None and rule.last_seen(held, 6, range(4, 8)) == (5, b)
    square = [[3.8, 0.3], [4.2, 0.3], [4.2, 0.7], [3.8, 0.7]]
    assert rule.closest_m((0.0, 0.0), (1.0, 0.0), square, (0.0, 0.0)) == pytest.approx(0.3, abs=1e-9)
    assert rule.closest_m((0.0, 0.0), (0.0, 0.0), square, (0.0, 0.0)) == pytest.approx(3.8118, abs=1e-4)   # at rest
