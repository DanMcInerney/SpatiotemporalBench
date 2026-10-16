"""`drop_zone_people` (look, scope `now`): which people stand within 4 m of the point below the drone. Hand-worked on
`hazard_kit`'s drone: it hovers over (2.0, 1.5), its camera 11.9 m up and looking straight down, in the phase `lower`; a
person is a 0.34 x 0.56 m footprint, so a person at (x, y) has its floor hull x - 0.17 to x + 0.17, y - 0.28 to y + 0.28
and its centroid at (x, y). The job's clearance is 2.4 m and its matching radius 2.0 m."""

from __future__ import annotations

import pytest

import hazard_kit as hk
import person_kit as pk
from stbench import questionnaire, questions, questions_hazard
from stbench.questions_harm import stretch
from stbench.questions_hazard import drop_zone_people, zone_candidate
from stbench.records import check_gold
from stbench.robots import ROBOTS
from stbench.seeding import stable_hash

K = hk.LOOK


def cand(s, k=K):
    return zone_candidate(hk.zone_world(s, k), k)


def keep_all(monkeypatch):
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "drop_zone_people", 100)


def test_the_gold_is_the_centroids_of_the_evidence_people_whose_hull_is_within_4_m():
    """Contract (FD-questions-0.10 section 2.2): the people with an evidence sighting in the ask frame whose hull has a
    point within 4 m of the camera's floor point, each at the hull's floor centroid. By hand from (2.0, 1.5): A at (3.5,
    1.5) has its hull's nearest point (3.33, 1.5), 1.33 m; R at (2.0, 5.0) has (2.0, 4.72), 3.22 m; Q at (6.5, 1.5) has
    (6.33, 1.5), 4.33 m, outside. The gold lists R and A, sorted; the row: the class `items`, scope `now`, the evidence
    keyframe 8, the subject `w:1-8`, the matching radius 2.0 m, the half weight of a false item, one resolution a gold
    item (one pixel from 11.9 m: 0.047 m at 3.5 m, 0.0447 m at 1.5 m) and the catastrophe tolerance `miss` with the
    people within C = 2.4 m of the point: A alone (3.22 m is farther), the margin max(C, 2 rho) = 2.4. Regression
    caught: a person outside the zone listed, the hull's corner in place of its centroid, a critical person that is not
    within C, or the items in an order the resolutions do not follow."""
    s = hk.drone()
    for pid, xy in (("A", (3.5, 1.5)), ("R", (2.0, 5.0)), ("Q", (6.5, 1.5))):
        pk.put_person(s, pid, xy, [K])
    c = cand(s)
    assert c.gold == {"type": "positions", "items": [{"xy": [2.0, 5.0]}, {"xy": [3.5, 1.5]}]}
    assert c.answer_class == "items" and c.scope == "now" and c.evidence_k == (K,) and c.subjects == ("w:1-8",)
    assert c.tolerances == {"match_m": 2.0, "resolutions_m": [pytest.approx(0.047, abs=1e-3),
                                                               pytest.approx(0.0447, abs=1e-3)], "fp_weight": 0.5,
                            "catastrophe": {"weight": 3, "rule": "miss", "margin_m": 2.4, "critical": [[3.5, 1.5]]}}
    assert c.decision_k == K and c.slots == {} and c.entities == ()


def test_a_person_with_pixels_and_no_evidence_is_a_dont_care_circle_not_an_item(monkeypatch):
    """Contract (review B2): a person with pixels in the ask frame and no evidence sighting (a 500 px glimpse) cannot be
    placed to the credit's precision, so they are a don't-care circle: the footprint's circumscribed radius plus the
    match radius, when their footprint is within 4 m plus the match radius of the camera's floor point. By hand: B (500
    px) at (3.0, 2.0) is in the zone; its circle is hypot(0.34, 0.56) / 2 + 2.0 = 0.3276 + 2.0 = 2.3276 m about (3.0,
    2.0). D (500 px) at (9.5, 1.5) has its footprint 7.33 m away, past 6 m: no circle. With nobody else the gold is the
    empty list with the one circle. Regression caught: a glimpse of a person in the zone scored as a false item, a miss,
    or a circle drawn for every person seen."""
    keep_all(monkeypatch)
    s = hk.drone()
    pk.put_person(s, "B", (3.0, 2.0), [K], px=500)
    pk.put_person(s, "D", (9.5, 1.5), [K], px=500)
    c = cand(s)
    assert c.gold == {"type": "positions", "items": [],
                     "dont_care": [{"xy": [3.0, 2.0], "radius": pytest.approx(2.3276, abs=1e-4)}]}
    assert c.answer_class == "empty" and c.tolerances["resolutions_m"] == []
    assert c.tolerances["catastrophe"] == {"weight": 3, "rule": "miss", "margin_m": 2.4, "critical": []}


def test_the_empty_gold_is_kept_by_the_balance_draw(monkeypatch):
    """Contract: an empty gold (nobody within 4 m) is asked when `stable_hash(case id, "drop_zone_people", point,
    "empty") % 100` is under the family's keep percent, so the empty list is not the common answer; one with a person
    never reads the draw. Regression caught: the draw read the other way round, or applied to a gold with items."""
    h = stable_hash("c0", "drop_zone_people", str(K), "empty") % 100
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "drop_zone_people", h)
    assert cand(hk.drone()) == questions_hazard.EMPTY_DRAWN
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "drop_zone_people", h + 1)
    assert cand(hk.drone()).answer_class == "empty"
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "drop_zone_people", 0)
    s = hk.drone()
    pk.put_person(s, "A", (3.5, 1.5), [K])
    assert cand(s).answer_class == "items"


@pytest.mark.parametrize("scene, reason", [
    (lambda: hk.drone(base_z=4.0), "the camera 3.9 m up covers +-2.45 m north-south: the 4 m disc is not in view"),
    (lambda: hk.drone(pitch=-0.6109), "pitched 35 degrees down the point below the camera is 55 degrees off the axis"),
])
def test_the_point_is_asked_only_when_the_disc_is_wholly_in_the_frame(monkeypatch, scene, reason):
    """Contract (review B2): the question is asked only when the disc of radius 4 m around the camera's floor point on
    the ground lies wholly inside the frame, from the delivered camera pose and the camera's focal lengths, so an empty
    list is a reading of what the frame shows. By hand: at 11.9 m looking down the view covers +-9.98 m by +-7.49 m, the
    disc fits; at 3.9 m it covers +-2.45 m north-south; at a pitch of -35 degrees the nadir is 55 degrees off the axis,
    the vertical half-angle being 32 degrees. Regression caught: a question asked of a view that does not hold the zone
    (an empty list would then be unreadable) or a view test that never fails."""
    keep_all(monkeypatch)
    assert cand(hk.drone()).answer_class == "empty"
    assert cand(scene()) == questions_hazard.NOT_IN_VIEW, reason


def test_the_phases_are_the_drones_over_the_drop_point(monkeypatch):
    """Contract: the question is asked in `inspect`, `descend` and `lower`, the phases in which the drone hovers over
    the drop point and lowers, and nowhere else; the three are phases of the drone's status vocabulary. Regression
    caught: a question about the drop zone asked in cruise, 40 m up, or a phase that is not the drone's (the phase test
    would then silently ask nothing)."""
    keep_all(monkeypatch)
    assert hk.ZONE_JOB.look_phases["drop_zone_people"] <= set(ROBOTS["drone"].status.phases)
    for phase in ("approach", "return"):
        assert cand(hk.drone(phase=phase)) == questions_hazard.PHASE
    for phase in sorted(hk.ZONE_JOB.look_phases["drop_zone_people"]):
        assert cand(hk.drone(phase=phase)).answer_class == "empty", phase


def test_the_moment_and_the_rows(monkeypatch):
    """Contract (records, the step-0 shape): a dead band gives no moment; an answerable point gives one `Moment` keyed
    by the point; the question and gold rows pass `records.check_question` and `check_gold` with the catastrophe
    tolerance and the don't-care circles (`questions.records` refuses an invalid row); and the subjects of two points
    fewer than 8 frames apart conflict. Regression caught: a gold the compile would refuse, or a subject that lets a
    person be asked at points four frames apart."""
    keep_all(monkeypatch)
    s = hk.drone()
    pk.put_person(s, "A", (3.5, 1.5), [K])
    pk.put_person(s, "B", (3.0, 2.0), [K], px=500)
    w = hk.zone_world(s)
    (moment,) = [drop_zone_people(w, K)]
    assert moment.key == str(K) and [c.family for c in moment.answerable] == ["drop_zone_people"]
    assert drop_zone_people(hk.zone_world(hk.drone(phase="approach")), K) is None
    (q,), (g,) = questions.records(moment.answerable)
    assert q["text"].startswith("Before you lower the package: where are the people within 4 m of the point below you?")
    assert g["scope"] == "now" and g["credit"] == "graded_set" and g["answer_class"] == "items"
    assert g["gold"]["dont_care"] == [{"xy": [3.0, 2.0], "radius": pytest.approx(2.3276, abs=1e-4)}]
    assert check_gold(g) == []
    wide = hk.zone_world(s, 11)
    assert questionnaire.conflicts(stretch(wide, 8), stretch(wide, 11))        # w:1-8 and w:4-11
    assert not questionnaire.conflicts(stretch(wide, 3), stretch(wide, 11))    # w:0-3 and w:4-11
