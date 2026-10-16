"""`tangle_ahead` (look, scope by evidence): the things on the floor within 1 m of the vacuum, ahead or to its sides,
that its brush could catch. Hand-worked on `ego_kit`'s scene: the look point is keyframe 13, where the base is at (3.0,
1.0) heading north, so the camera floor point is (3.0, 1.13) and the front edge (3.0, 1.17); the phase is `cleaning`. A
cord is 0.5 x 0.18 x 0.03 m, a floor point is a thing of 0.02 m. The job's clearance is 2.0 m and its matching radius
0.5 m."""

from __future__ import annotations

import pytest

import hazard_kit as hk
from ego_kit import K, put, scene
from ego_kit import cut as ego_cut
from stbench import questionnaire, questions, questions_hazard
from stbench.questions_harm import stretch
from stbench.questions_hazard import tangle_ahead, tangle_candidate
from stbench.records import check_gold
from stbench.robots import ROBOTS
from stbench.seeding import stable_hash


def cut_world(s, k=K):
    return hk.tangle_world(s, k)


def world(s):
    return hk.tangle_world(s)


def cut_world_without_phases(s):
    """The window as the plain vacuum job reads it: it declares no phases."""
    return ego_cut(s)


CORD = (0.5, 0.18, 0.03)
DOT = (0.02, 0.02, 0.03)   # a thing of one floor point, for the arc's edge
CAM = (3.0, 1.13)


def cord(s, eid, xy, kind="extension_cord", seen=(K,), size=CORD, **kw):
    return put(s, eid, kind, "black", xy, seen, size=size, mobility="clutter", **kw)


def covered(s, cells_at=None, k=K):
    """The window ended at `k` whose grid observed the disc of 1.2 m around the camera floor point at every keyframe up
    to `k`, or the cells `cells_at(keyframe)`."""
    return hk.covering(cut_world(s, k), cells_at or (lambda j: hk.disc(CAM, 1.2)))


def keep_all(monkeypatch):
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "tangle_ahead", 100)


def test_the_gold_lists_the_things_in_reach_ahead_or_to_the_sides_at_their_middles():
    """Contract (FD-questions-0.10 section 2.3): the things of a tangle kind on the floor, recognised up to the point,
    whose floor hull has its nearest point within 1 m of the camera's floor point and at most 100 degrees from the
    heading, each at its middle. By hand from (3.0, 1.13) heading north: C1 at (3.0, 1.8) has its hull's nearest point
    (3.0, 1.71), 0.58 m ahead; C2 at (3.0, 1.45) (3.0, 1.36), 0.23 m; C3 at (2.2, 1.13) (2.45, 1.13), 0.55 m to the
    left, 90 degrees; C4 at (3.0, 2.6) is 1.38 m away; C5 at (3.0, 0.2) has (3.0, 0.29), 0.84 m but behind, 180 degrees.
    The gold lists C3, C2 and C1 in order of place. The row: the class `items`, scope `now` (every item is in this
    frame), the evidence keyframe 13, the subject `w:6-13`, the matching radius 0.5 m (a cord's widening is 0), the
    resolutions of one pixel at each middle in that order (0.0213 for C3, the floor of 0.01 for C2, 0.0150 for C1), and
    the catastrophe tolerance `miss` with the one item within 0.3 m of the front edge (3.0, 1.17): C2, 0.19 m (C1 and C3
    are 0.54 and 0.55 m), the margin max(C, 2 rho) = 2.0. Regression caught: a thing behind or out of reach listed, the
    nearest point measured from the base or the front edge, a critical item that is not within 0.3 m of the front edge,
    or the resolutions out of the gold's order."""
    s = scene()
    places = {"C1": (3.0, 1.8), "C2": (3.0, 1.45), "C3": (2.2, 1.13), "C4": (3.0, 2.6), "C5": (3.0, 0.2)}
    for eid, xy in places.items():
        cord(s, eid, xy)
    c = tangle_candidate(cut_world(s), K)
    assert c.gold == {"type": "positions", "items": [{"xy": [2.2, 1.13]}, {"xy": [3.0, 1.45]}, {"xy": [3.0, 1.8]}]}
    assert c.answer_class == "items" and c.scope == "now" and c.evidence_k == (K,) and c.subjects == ("w:6-13",)
    assert c.tolerances == {"match_m": 0.5, "fp_weight": 0.5,
                            "resolutions_m": [pytest.approx(0.0213, abs=1e-4), 0.01, pytest.approx(0.015, abs=1e-4)],
                            "catastrophe": {"weight": 3, "rule": "miss", "margin_m": 2.0, "critical": [[3.0, 1.45]]}}
    assert c.decision_k == K and c.slots == {} and c.entities == ()


def test_the_arc_ends_at_100_degrees_from_the_heading():
    """Contract: "ahead or to your sides" is at most 100 degrees from the heading (`hazard_rule.ARC_DEG`). By hand: a
    floor point 0.5 m from (3.0, 1.13) at 95 degrees to the left of north is at world angle 185 degrees: (2.5019,
    1.0864); at 105 degrees (195 degrees) (2.5170, 1.0006). The first is listed, the second is not (and, being out of
    reach by the truth too, no circle). Regression caught: an arc of 90 degrees (the first dropped) or of 110 degrees
    (the second listed), or a bearing taken from the robot's base."""
    s = scene()
    cord(s, "P1", (2.5019, 1.0864), "cable_bundle", size=DOT)
    cord(s, "P2", (2.5170, 1.0006), "cable_bundle", size=DOT)
    assert tangle_candidate(cut_world(s), K).gold == {"type": "positions", "items": [{"xy": [2.5019, 1.0864]}]}


def test_the_kinds_that_recur_at_one_spot_and_the_unrecognised_are_dont_care_circles(monkeypatch):
    """Contract (review S4): a thing of a kind that recurs at one spot from script to script (a bath mat) is never an
    item, nor is a thing of a tangle kind that the robot never recognised, but each in reach is a don't-care circle, so
    a reader that lists it is not charged and one that does not is not either. A thing off the floor (a towel on a shelf
    at 0.5 m) and a thing of another kind (a bag) are neither. By hand from (3.0, 1.13): the mat M (0.4 x 0.6) at (3.0,
    1.6) has its hull's nearest point 0.17 m ahead, its circle radius 0.5 + (0.6 / 2 - 0.25) = 0.55 m; the cord U (never
    seen, 0.5 x 0.18) at (2.4, 1.5) has its nearest point (2.65, 1.41), 0.45 m, its circle 0.5 m; the towel T at (3.0,
    1.5) on a shelf and the bag B at (3.2, 1.4) are in neither. The gold is the empty list with the two circles.
    Regression caught: a layout prior asked for as an item, a hidden cord charged as a false item, or every thing in
    reach given a circle."""
    keep_all(monkeypatch)
    s = scene()
    put(s, "M", "bath_mat", "beige", (3.0, 1.6), [K], size=(0.4, 0.6, 0.015), mobility="clutter")
    cord(s, "U", (2.4, 1.5), seen=())
    put(s, "T", "towel", "blue", (3.0, 1.5), [K], where=[(0, None, (3.0, 1.5, 0.5))], size=(0.356, 0.263, 0.088))
    put(s, "B", "bag", "red", (3.2, 1.4), [K], size=(0.3, 0.3, 0.3))
    c = tangle_candidate(covered(s), K)
    assert c.gold["items"] == [] and c.answer_class == "empty"
    assert c.gold["dont_care"] == [{"xy": [2.4, 1.5], "radius": 0.5}, {"xy": [3.0, 1.6], "radius": pytest.approx(0.55)}]
    assert c.tolerances["resolutions_m"] == [] and c.tolerances["catastrophe"]["critical"] == []


def test_a_thing_that_moved_since_it_was_seen_is_no_item_and_both_places_are_circles(monkeypatch):
    """Contract: a thing seen in reach and moved since (its placement at the point 0.15 m or more from the place it was
    seen at) is no item, since the frames now show another floor there; the place it was seen at and the place it is at,
    if in reach, are don't-care circles. By hand: the cord seen at keyframe 9 (13.5 s) at (3.0, 1.8) is moved at 17 s to
    (3.0, 1.5), 0.3 m; both are in reach of (3.0, 1.13). The gold is empty with the two circles. Regression caught: the
    old place listed (the reader who sees an empty floor is charged), or the new place listed (nobody saw it there)."""
    keep_all(monkeypatch)
    s = scene()
    cord(s, "U", (3.0, 1.8), seen=(9,), where=[(0, 17.0, (3.0, 1.8, 0.0)), (17.0, None, (3.0, 1.5, 0.0))])
    c = tangle_candidate(covered(s), K)
    assert c.gold["items"] == [] and [d["xy"] for d in c.gold["dont_care"]] == [[3.0, 1.5], [3.0, 1.8]]


def test_an_empty_gold_needs_the_reach_region_observed_and_is_kept_by_the_draw(monkeypatch):
    """Contract (review S4): an empty gold needs at least 80% of the reach region's fine cells to have been observed in
    any state by the point (the free-space grid of every frame sent, as `clear_path`'s coverage), else it is a dead
    band, since nothing says the region is clear; scope `now` when frame 13 alone observed 80% of them, else `memory`,
    with the keyframes that last observed the cells as its evidence. By hand: a grid that observed a disc of 1.2 m
    around the camera floor point at 13: covered, `now`, evidence (13,); only the cells east of x = 3.0 at 13: about
    half of the region, a dead band; the cells west of 3.0 at keyframe 11 and the east ones at 13: covered, `memory`,
    evidence (11, 13). The kept empty gold is then asked by `stable_hash(case id, "tangle_ahead", point, "empty") % 100
    < percent`. Regression caught: an empty list asked of a floor nobody looked at, the scope of a gold that needs the
    earlier frames called `now`, or the draw read the other way."""
    keep_all(monkeypatch)
    s = scene()
    c = tangle_candidate(covered(s), K)
    assert c.answer_class == "empty" and c.scope == "now" and c.evidence_k == (K,) and c.gold["items"] == []
    east = {c for c in hk.disc(CAM, 1.2) if (c[0] + 0.5) * 0.1 >= 3.0}
    west = hk.disc(CAM, 1.2) - east
    assert tangle_candidate(covered(s, lambda j: east), K) == questions_hazard.UNCOVERED
    both = tangle_candidate(covered(s, lambda j: west if j == 11 else east if j == K else set()), K)
    assert both.scope == "memory" and both.evidence_k == (11, K)
    h = stable_hash("c0", "tangle_ahead", str(K), "empty") % 100
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "tangle_ahead", h)
    assert tangle_candidate(covered(s), K) == questions_hazard.EMPTY_DRAWN
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "tangle_ahead", h + 1)
    assert tangle_candidate(covered(s), K).answer_class == "empty"


def test_cells_observed_after_the_point_do_not_count_for_an_empty_gold(monkeypatch):
    """Contract (truth rule 2, a gold reads no frame after the point): an empty gold counts the cells observed up to the
    point and no later frame. By hand: a grid that observed the whole disc of 1.2 m around the camera floor point only
    at keyframe 20, in the full window of the script, leaves the point 13 a dead band (nothing was seen by then);
    observed at keyframe 11 it is covered, with the evidence (11,). Regression caught: a coverage read from the whole
    window (a floor seen later called clear earlier)."""
    keep_all(monkeypatch)
    s = scene()
    later = hk.covering(world(s), lambda j: hk.disc(CAM, 1.2) if j == 20 else set())
    assert tangle_candidate(later, K) == questions_hazard.UNCOVERED
    earlier = hk.covering(world(s), lambda j: hk.disc(CAM, 1.2) if j == 11 else set())
    assert tangle_candidate(earlier, K).evidence_k == (11,)


def test_a_gold_with_items_needs_no_coverage_and_is_memory_when_an_item_was_seen_earlier(monkeypatch):
    """Contract: a gold with items does not read the free-space grid, and its scope is `now` exactly when every item's
    last recognisable sighting is the point's frame. By hand: a cord seen at keyframe 11 only (not at 13) at (3.0, 1.8)
    is an item with evidence (11,), `memory`; with a second seen at 13 the evidence is (11, 13), `memory`; seen at 13
    alone, `now`. Regression caught: an item dropped for want of coverage, or a stale sighting called `now`."""
    monkeypatch.setitem(questions_hazard.EMPTY_KEEP_PERCENT, "tangle_ahead", 0)
    def asked(*pairs):
        s = scene()
        for eid, xy, seen in pairs:
            cord(s, eid, xy, seen=seen)
        return tangle_candidate(cut_world(s), K)
    old = asked(("A", (3.0, 1.8), (11,)))
    assert old.evidence_k == (11,) and old.scope == "memory" and old.gold["items"] == [{"xy": [3.0, 1.8]}]
    mixed = asked(("A", (3.0, 1.8), (11,)), ("B", (2.2, 1.13), (K,)))
    assert mixed.evidence_k == (11, K) and mixed.scope == "memory"
    assert asked(("B", (2.2, 1.13), (K,))).scope == "now"


def test_the_phases_are_the_jobs_to_declare_and_a_job_that_declares_none_raises():
    """Contract: the question is asked in the status phases the job declares for it (`JobSpec.look_phases`): the
    vacuum's are `cleaning` and `transit` and nowhere else (not while docked, undocking or docking, where the status
    already says what the brush does), phases of its status vocabulary; a job that asks it and declares none raises,
    naming itself, so that no phase is asked that nobody chose. By hand: keyframe 13 is `cleaning`, 5 `transit`, 2
    `undocking` and 24 `docking`. Regression caught: a question about the floor ahead asked on the dock, a phase that is
    not the vacuum's, or a phase named in the compiler (`test_events_resolve_from_the_job` holds that out)."""
    assert hk.TANGLE_JOB.look_phases["tangle_ahead"] <= set(ROBOTS["vacuum"].status.phases)
    with pytest.raises(ValueError, match="declares no `look_phases`"):
        tangle_candidate(cut_world_without_phases(scene()), K)
    s = scene()
    cord(s, "A", (3.0, 1.8), seen=(2, 5, 24, K))
    for k, ok in ((K, True), (5, True), (2, False), (24, False)):
        found = tangle_candidate(cut_world(s, k), k)
        assert (found == questions_hazard.PHASE) is not ok, k


def test_the_rows_validate_and_the_moment_is_none_in_a_dead_band(monkeypatch):
    """Contract (records, the step-0 shape): a dead band gives no moment; an answerable point gives one `Moment` keyed
    by the point; the question and gold rows, an items one and an empty one with circles, pass `records.check_question`
    and `check_gold` (`questions.records` refuses an invalid row) with the catastrophe tolerance and the circles.
    Regression caught: a gold the compile would refuse, or a moment for a dead band."""
    keep_all(monkeypatch)
    s = scene()
    cord(s, "C1", (3.0, 1.45))
    w = cut_world(s)
    (moment,) = [tangle_ahead(w, K)]
    assert moment.key == str(K) and [c.family for c in moment.answerable] == ["tangle_ahead"] and moment.unknown == []
    assert tangle_ahead(cut_world(s, 2), 2) is None
    (q,), (g,) = questions.records(moment.answerable)
    assert q["text"].startswith("List anything lying on the floor within 1 m of you, ahead or to your sides")
    assert g["scope"] == "now" and g["credit"] == "graded_set" and g["answer_class"] == "items" and check_gold(g) == []
    empty = scene()
    put(empty, "M", "bath_mat", "beige", (3.0, 1.6), [K], size=(0.4, 0.6, 0.015), mobility="clutter")
    (g,) = questions.records(tangle_ahead(covered(empty), K).answerable)[1]
    assert g["answer_class"] == "empty" and g["gold"]["dont_care"] and check_gold(g) == []


def test_two_points_in_one_look_window_conflict_so_one_thing_is_not_asked_twice():
    """Contract (FD-questions-0.10 S7): the subject is the recent view, so two points fewer than 8 frames apart name
    overlapping stretches. By hand: 13 names `w:6-13`, 20 names `w:13-20` (shared keyframe 13), 21 `w:14-21`.
    Regression caught: a cable asked at points four frames apart."""
    w = cut_world(scene(), 21)
    assert questionnaire.conflicts(stretch(w, 13), stretch(w, 20))
    assert not questionnaire.conflicts(stretch(w, 13), stretch(w, 21))
