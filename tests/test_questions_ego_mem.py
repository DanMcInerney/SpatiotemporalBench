"""`ego_mem` (`questions_ego`): the gold, which things are answerable and which unknown, and that nothing after the look
point decides a question. The scene is `ego_kit`'s: things seen at keyframes 1 and 2 from the kitchen's centre, the look
point 13 ending the look window 6-13, the camera then at (3.0, 1.13, 0.08) heading north."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import family_kit as fk
from ego_kit import EGO_JOB, SMALL, K, corners, cut, named, put, scene, told, world
from space_kit import record
from stbench.questions_ego import ego_mem
from stbench.questions_text import FAMILIES
from stbench.questions_world import World
from synth import RECOG

DARK_GREY = (85, 85, 85)


def pools(s, w=None):
    moment = ego_mem(w or world(s), K)
    return (named(moment), moment.unknown) if moment else ({}, [])


def test_the_gold_is_the_hull_of_every_sighting_at_the_place_in_the_camera_frame_now():
    """Contract (FD-0.7.0 review, change 7): the gold is the point nearest the camera's floor point now of the union of
    the floor hulls of every recognisable sighting before the look window at the thing's place, put in the camera frame
    at the look point; a side no sighting showed does not count, and the last sighting alone is not the gold. By hand: a
    red bag at (5.0, 3.13) seen at keyframes 1, 2 and 3 as three faces of its 0.1 m footprint: the west face (4.95,
    3.08) to (4.95, 3.18) at 1, the south face (4.95, 3.08) to (5.05, 3.08) at 2 and the north face (4.95, 3.18) to
    (5.05, 3.18) at 3. Their hulls' union is the whole square, whose nearest point to the camera floor point (3.0, 1.13)
    is the corner (4.95, 3.08): 1.95 ahead and 1.95 to the right, slant sqrt(1.95^2 + 1.95^2 + 0.08^2) = 2.7589; the
    last sighting alone would put it at (4.95, 3.18), 2.05 ahead. A blue bag at (1.0, 1.13) seen at 2 has its east face
    nearest, (1.05, 1.13): 0.0 ahead and 1.95 left, slant 1.9516. The need is the job's C (2.0); the resolution is one
    pixel at the gold point from the sighting whose contact read best: keyframe 2 (camera (2.0, 1.63, 0.08), the cameras
    of 1 and 3 being (1.0919, 1.0919) and (1.13, 1.5), farther from the south face), 3.2871 m from (4.95, 3.08): 0.3545
    (keyframes 1 and 3 would give 0.6177 and 0.5604), and for the blue bag, 1.0735 m from keyframe 2's camera, 0.0381.
    The evidence is all three sightings. Regression caught: the last sighting alone, the first, the middle of the thing,
    the hull's vertex in place of its edge, a need that depends on the range, or a resolution read from another
    sighting's camera."""
    s = scene(poses={1: (1.0, 1.0), 3: (1.0, 1.5)}, yaws={1: math.pi / 4, 3: 0.0})
    west, south, north = ([(4.95, 3.08, 0.0), (4.95, 3.18, 0.0)], [(4.95, 3.08, 0.0), (5.05, 3.08, 0.0)],
                          [(4.95, 3.18, 0.0), (5.05, 3.18, 0.0)])
    put(s, "A", "bag", "red", (5.0, 3.13), [1, 2, 3], pts=lambda k: {1: west, 2: south, 3: north}[k])
    put(s, "B", "bag", "blue", (1.0, 1.13), [2])
    answerable, _ = pools(s)
    assert list(answerable) == ["A", "B"]
    a, b = answerable["A"], answerable["B"]
    assert a.gold == {"type": "offset", "ahead_m": pytest.approx(1.95, abs=1e-4),
                      "left_m": pytest.approx(-1.95, abs=1e-4)}
    assert b.gold == {"type": "offset", "ahead_m": pytest.approx(0.0, abs=1e-4),
                      "left_m": pytest.approx(1.95, abs=1e-4)}
    assert a.range_m == pytest.approx(2.7589, abs=1e-4) and b.range_m == pytest.approx(1.9516, abs=1e-4)
    assert a.tolerances == {"resolution_m": pytest.approx(0.3545, abs=1e-4), "need_m": 2.0, "harm": "radial"}
    assert b.tolerances == {"resolution_m": pytest.approx(0.0381, abs=1e-4), "need_m": 2.0, "harm": "radial"}
    assert (a.evidence_k, a.entities, a.subjects) == ((1, 2, 3), ("A", "A", "A"), ("e:A",))
    assert (a.slice, a.answer_class, a.decision_k) == ("answerable", "answerable", K)


def test_only_the_sightings_at_the_gold_place_set_the_gold():
    """Contract (FD-0.7.0 review, change 7): the union is of the sightings at the thing's current place; one from
    before a move says where the thing was. By hand: a red bag stood at (2.0, 2.2) until 3.75 s and stands at
    (5.0, 3.13) after; seen at keyframe 1 and at keyframe 3. Its corners at the new place put the nearest point at
    (4.95, 3.08): 1.95 ahead, 1.95 right; the old place's corner (2.05, 2.15), 1.02 ahead and 0.95 left, is not in the
    union. Evidence is the sighting at the new place, 3. Regression caught: a gold from where the thing no longer is."""
    s = scene()
    put(s, "A", "bag", "red", (0, 0), [1, 3], where=[(0, 3.75, (2.0, 2.2, 0.0)), (3.75, None, (5.0, 3.13, 0.0))])
    a = pools(s)[0]["A"]
    assert a.gold == {"type": "offset", "ahead_m": pytest.approx(1.95, abs=1e-4),
                      "left_m": pytest.approx(-1.95, abs=1e-4)}
    assert a.evidence_k == (3,)


def test_the_gold_reads_no_frame_from_the_look_window_on():
    """Contract (plan 4 A3, causality): the gold reads the recognisable sightings before the look window (keyframes up
    to 5) and no later frame, so what the thing looked like at keyframe 20, after the look point, cannot move it. By
    hand: a red bag seen at 2 and again at 20 with its points elsewhere; a recording `points_of` is asked for keyframe 2
    only, and the gold is the keyframe-2 corner (4.95, 3.08), 1.95 ahead and 1.95 right. Regression caught: a later
    sighting in the union (the gold moves with the future)."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2, 20],
        pts=lambda k: corners((5.0, 3.13), SMALL) if k == 2 else [(1.0, 1.0, 0.0)])
    inp, asked = fk.inputs(s, EGO_JOB), []
    w = World(replace(inp, points_of=lambda k, e: (asked.append(k), inp.points_of(k, e))[1]), record(s))
    a = named(ego_mem(w, K))["A"]
    assert set(asked) == {2} and a.evidence_k == (2,)
    assert a.gold == {"type": "offset", "ahead_m": pytest.approx(1.95, abs=1e-4),
                      "left_m": pytest.approx(-1.95, abs=1e-4)}


def test_the_screen_reads_the_camera_that_saw_the_point_not_the_one_asking_now():
    """Contract (FD-0.7.0 review, change 5; the coordinator's test gap): the one-pixel error is read from the sighting
    camera, so a thing seen from near passes at a tight clearance though it is far from the camera now, and a thing seen
    from afar fails though it is near now. By hand (C = 0.10, so C/2 = 0.05; the camera 0.08 m up, fy = 381.36, so
    0.08 * fy = 30.5088; the camera now at (3.0, 1.13), keyframe 2's at (2.0, 1.63), keyframe 1's at (1.0919, 1.0919)):
    - A, a 0.1 m floor box at (2.0, 2.63) seen at keyframe 2: its point nearest now is the corner (2.05, 2.58), 0.9513 m
      from keyframe 2's camera, (0.9050 + 0.0064) / 30.5088 = 0.0299, asked; from the camera now it is 1.7335 m,
      (3.0050 + 0.0064) / 30.5088 = 0.0987, which would drop it;
    - B, a 0.1 m floor box at (3.0, 2.0) seen at keyframe 1: its point nearest now lies on the edge between the corners
      (2.95, 1.95) and (3.05, 1.95), 1.9 m from keyframe 1's camera, (4.189 + 0.0064) / 30.5088 = 0.1375, dropped; from
      the camera now it is 0.8215 m, 0.0223, which would ask it.
    Regression caught: the screen read from the camera at the look point (the reverse of both answers)."""
    s = scene(poses={1: (1.0, 1.0)}, yaws={1: math.pi / 4})
    for eid, colour, xy, seen in (("A", "red", (2.0, 2.63), [2]), ("B", "blue", (3.0, 2.0), [1])):
        floor = [(xy[0] + dx, xy[1] + dy, 0.0) for dx in (-0.05, 0.05) for dy in (-0.05, 0.05)]
        put(s, eid, "bag", colour, xy, seen, pts=floor)
    tight = replace(EGO_JOB, clearance=0.10)
    assert set(named(ego_mem(fk.world(s, job=tight), K))) == {"A"}


def test_a_thing_whose_sightings_at_its_place_span_two_placements_is_not_asked():
    """Contract (`questions_ego._remembered`): the table unions hulls by the centres of the sightings and the gold by
    the timeline's placements; a thing whose sightings at its place (within a centimetre) lie in more than one placement
    of the timeline is not asked, so the two cannot differ. By hand: a red bag seen at keyframes 2 (3 s) and 5 (7.5 s)
    at (5.0, 3.13) in one placement is asked; a blue one that was re-placed 5 mm along at 4.5 s, between the sightings,
    is not (it is in two placements, 5 mm apart). Regression caught: a question whose gold unions what the timeline
    calls two placements."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2, 5])
    put(s, "B", "bag", "blue", (0, 0), [2, 5], where=[(0, 4.5, (1.0, 1.13, 0.0)), (4.5, None, (1.005, 1.13, 0.0))])
    assert set(pools(s)[0]) == {"A"}


def test_a_thing_with_a_pixel_that_has_no_depth_is_not_asked():
    """Contract (`questions_ego._cand`): the hull is of the pixels that have a depth, so a thing any of whose evidence
    pixels has none may have a part the hull misses; it is not asked, in either family. By hand: two bags seen at
    keyframe 2, one with a pixel of no depth there (`Inputs.gaps_of` says 1) and one without: only the second is named.
    Regression caught: a gold nearest a point of a hull that is missing part of the thing."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2])
    put(s, "B", "bag", "blue", (1.0, 1.13), [2])
    s.gaps[2, "A"] = 1
    assert set(pools(s)[0]) == {"B"}


def test_the_moment_is_the_look_point_and_the_text_pure():
    """Contract (plan 3, FD-0.7.0 review change 7): a look family's moment is keyed `str(k)` and the text is the render
    of the slots. 'As you last saw it, where is the point of the red bag you saw that is nearest to your camera now,
    measured along the floor? Answer in metres ahead of the camera and to its left.' Regression caught: a moment keyed
    on a thing (one point drawing its pools per thing), or a text that leaves 'nearest part' open to a reading (the
    side that faced the robot, the middle)."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2])
    assert ego_mem(world(s), K).key == str(K)
    text = FAMILIES["ego_mem"].render({"kind": "bag", "attributes": {"color": "red"}})
    assert text == ("As you last saw it, where is the point of the red bag you saw that is nearest to your camera now, "
                    "measured along the floor? Answer in metres ahead of the camera and to its left.")


def test_a_thing_with_any_row_in_the_look_window_is_never_asked():
    """Contract (brief 2.2): the thing has no row in the look window, at any pixel count, since the current view would
    answer. By hand: bags seen recognisably at 2 are asked, except one also seen at 1,500 px at keyframe 10 and one with
    a 40 px sliver at 12, both inside the look window 6-13. Regression caught: a thing in view asked as one out of
    view."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2])
    put(s, "B", "bag", "blue", (1.0, 1.13), [2, 10])
    put(s, "C", "bag", "green", (6.0, 2.0), [2])
    s.see(12, "C", 40)
    assert list(pools(s)[0]) == ["A"]


@pytest.mark.parametrize("k, want", [(13, set()), (16, set()), (17, {"A"}), (18, {"A"})])
def test_the_recent_view_slides_and_a_point_need_end_no_window(k, want):
    """Contract (0.10.0, `look_windows.recent`): "seen before the recent view and not in it" reads the last 8 frames of
    the task up to the point, whatever look window the stream cuts, so a thing seen in them is never asked and one seen
    only before them is, at a point that ends no decision window as at one that does. By hand (camera at (3.0, 1.13)
    heading north at 9, 13, 16, 17 and 18 alike): a red bag A seen at keyframe 9 and a blue bag B seen at 3. The recent
    view at 13 is 6-13 and at 16 is 9-16: A is in it, B is not, so only B is asked; at 17 it is 10-17 and at 18 11-18:
    A has slid out and both are asked. 16 and 17 end no window (the stream's end at 13 and 18). Regression caught: the
    window of the stream deciding (a sighting 8 frames back called recent at a window's end, or the points confined to
    the window ends), or a view that does not slide."""
    ks = (3, 9, 13, 16, 17, 18)
    s = scene(poses={j: (3.0, 1.0) for j in ks if j != 13}, yaws={j: math.pi / 2 for j in ks if j != 13})
    put(s, "A", "bag", "red", (5.0, 3.13), [9])
    put(s, "B", "bag", "blue", (1.0, 1.13), [3])
    got = set(named(ego_mem(world(s), k)))
    assert got == want | {"B"}


def test_a_thing_must_be_unchanged_within_a_centimetre_from_its_last_recognisable_sighting():
    """Contract (FD-0.7.0 review, change 7, `World.steady`): every sighting from the last recognisable one to the look
    point lies within 1 cm of that place, since 0.15 m is over C/2 for a vacuum, so the table, which unions the hulls
    at one place, holds the thing where the gold does. By hand: a bag seen recognisably at (5.0, 3.13) and glimpsed
    (300 px) at keyframe 5 after moving 0.5 m is not asked; one that moved 0.1 m (asked under the old 0.15 m) is not;
    one that moved 5 mm is, and its gold is the recognisable place. Regression caught: a gold at a place the table has
    already replaced, by up to 0.15 m."""
    s = scene()
    for eid, colour, x0, dx in (("A", "red", 5.0, 0.5), ("B", "blue", 1.0, 0.1), ("C", "green", 6.0, 0.005)):
        put(s, eid, "bag", colour, (0, 0), [2, 5], where=[(0, 5.0, (x0, 3.13, 0.0)), (5.0, None, (x0 + dx, 3.13, 0.0))])
        s.see(5, eid, 300)
    answerable, _ = pools(s)
    assert list(answerable) == ["C"]


def test_a_thing_that_moved_unseen_after_its_last_sighting_is_not_asked():
    """Contract (`questions_ego._remembered`, DECISIONS 0.7.0): the placement is unchanged within 1 cm from the last
    sighting to the ask, by the truth at the look point and not only by later sightings, since the prompt says the thing
    is where the robot saw it and the gold is that place. By hand: bags seen recognisably at (5.0, 3.13) at keyframe 2
    and never again; one moves 0.5 m at t=5 and one moves 5 mm: only the 5 mm one is asked. Regression caught: a move
    nobody saw (a fan 5 m, a drone car 60 m) asked, with a gold at the old place."""
    s = scene()
    for eid, colour, x0, dx in (("A", "red", 5.0, 0.5), ("B", "blue", 1.0, 0.005)):
        put(s, eid, "bag", colour, (0, 0), [2], where=[(0, 5.0, (x0, 3.13, 0.0)), (5.0, None, (x0 + dx, 3.13, 0.0))])
    answerable, _ = pools(s)
    assert list(answerable) == ["B"]


@pytest.mark.parametrize("twin_frames, twin_kind, asked", [
    ([4], "bag", set()),         # a twin sighted before the look point
    ([10], "bag", set()),        # a twin sighted in the look window itself: it is in view now
    ([20], "bag", {"A"}),        # a twin first sighted after the look point does not exist yet
    ([], "bag", {"A"}),          # a twin the robot never saw cannot confuse it
    ([4], "box", set()),         # a look-alike kind (CONFUSABLE): a red box could be a red bag
])
def test_a_description_must_be_unique_among_the_things_sighted_so_far(twin_frames, twin_kind, asked):
    """Contract (brief 2.2): the name is unique among the things sighted in the window up to the look point, so a twin
    seen so far hides the thing and one first seen later, or never, does not. By hand: a red bag "A" seen at 2 with a
    red bag or box "T" seen at the frames given. Regression caught: world-unique naming (the supply of 12 on the helper)
    or a future twin deciding a question."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2])
    put(s, "T", twin_kind, "red", (6.5, 2.0), twin_frames)
    assert set(pools(s)[0]) - {"T"} == asked


@pytest.mark.parametrize("before, after, asked", [
    (None, DARK_GREY, True),     # black in every frame up to k; grey only afterwards
    (DARK_GREY, None, False),    # grey up to k; black only afterwards
])
def test_the_colour_is_read_from_the_frames_up_to_the_look_point(before, after, asked):
    """Contract (brief 2.2): the colour word must be borne out by the frames up to the look point alone. By hand: a bag
    labelled black seen at 2 and 3, rendering black (faithful) or grey (85, 85, 85) there, and at 20 to 24 rendering the
    other; only the first is asked. Regression caught: the colour of the whole window deciding, so that a later frame
    names or vetoes a question."""
    s = scene()
    put(s, "A", "bag", "black", (5.0, 3.13), [2, 3], rgb=before)
    s.see([20, 21, 22, 23, 24], "A", RECOG, rgb=after)
    assert ("A" in pools(s)[0]) is asked


@pytest.mark.parametrize("drum_px, asked", [(RECOG, {"B", "D"}), (100, set())])
def test_a_drum_hides_a_bin_only_when_it_was_not_plainly_made_out(drum_px, asked):
    """Contract: a bin and a drum of one colour are told apart by shape once the drum has been plainly
    seen, so a recognisable one hides neither; one only glimpsed (100 px: sighted, not recognisable) could be the bin,
    so the bin is not asked (and the sliver is no question). By hand: a red bin seen at 2 and a red drum at 3.
    Regression caught: the unseen-only groups applied to a seen drum (supply lost), or not at all (the bin named beside
    a speck)."""
    s = scene()
    put(s, "B", "bin", "red", (5.0, 3.13), [2])
    put(s, "D", "drum", "red", (6.5, 2.0), [3], px=drum_px)
    assert set(pools(s)[0]) == asked


def test_a_clutter_thing_is_named_by_its_kind_and_hidden_by_another_of_it():
    """Contract (brief 2.2): clutter is named by kind alone, so a second chair (light furniture too) sighted so far
    hides it. By hand: one chair seen at 2 is asked as 'the chair'; with another seen at 3 it is not. Regression caught:
    a chair asked with two in the home (the table would answer with either)."""
    s = scene()
    put(s, "C1", "chair", "oak", (5.0, 3.13), [2], size=(0.4, 0.4, 0.9), mobility="clutter")
    answerable, _ = pools(s)
    assert answerable["C1"].slots == {"kind": "chair", "attributes": {}}
    put(s, "C2", "chair", "oak", (6.0, 2.0), [3], size=(0.4, 0.4, 0.9), mobility="light")
    assert pools(s)[0] == {}


def test_the_answerable_pool_does_not_read_a_frame_after_the_look_point():
    """Contract (plan 4 A3): naming and gold read no frame after k, so cutting the window at k leaves the answerable
    pool. By hand, after keyframe 13: a twin of the red bag at 20, the black bag's frames turning grey at 20 to 24, and
    a blue bag "X", seen blue at 3, whose frames 20 to 24 render red (so a look at the whole window would take it for
    a red bag and hide "the red bag"). Regression caught: a future frame deciding which question is asked, or the
    look-alike screen reading renders past the look point."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2])
    put(s, "T", "bag", "red", (6.5, 2.0), [20])
    put(s, "B", "bag", "black", (1.0, 1.13), [2, 3])
    s.see([20, 21, 22, 23, 24], "B", RECOG, rgb=DARK_GREY)
    put(s, "X", "bag", "blue", (6.0, 3.0), [3])
    s.see([20, 21, 22, 23, 24], "X", RECOG, rgb=(200, 30, 30))
    full, early = pools(s)[0], pools(s, cut(s))[0]
    assert set(full) == {"A", "B", "X"}
    assert {k: (c.key, c.gold, c.tolerances, c.range_m) for k, c in full.items()} == {
        k: (c.key, c.gold, c.tolerances, c.range_m) for k, c in early.items()}


def test_the_unknown_pool_holds_things_no_frame_shows_and_never_one_met_later():
    """Contract (brief 2.2, plan 4 A3): the unknown slice names real things with no pixel anywhere in the window, so
    none is met later in the case. By hand: a yellow bag with no pixel is in the pool; a green one first seen at
    keyframe 20 (after the look point) is not, though at the look point it was still unseen; with the window cut at 13
    it is. A told thing, and a thing that does not exist yet (from t = 300 s, after keyframe 13) are out. Regression
    caught: an unknown question whose referent the robot meets later in the case (it would then have a right answer)."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2])
    put(s, "U", "bag", "yellow", (6.0, 2.0))
    put(s, "L", "bag", "green", (6.5, 2.5), [20])
    put(s, "T", "bag", "orange", (6.5, 1.0))
    put(s, "N", "bag", "pink", (0, 0), where=[(300.0, None, (6.0, 1.0, 0.0))])
    real = lambda cands: {c.subjects[0] for c in cands if c.subjects[0].startswith("e:")}   # noqa: E731
    w = told(world(s), "T")
    assert real(ego_mem(w, K).unknown) == {"e:U"}
    assert real(ego_mem(told(cut(s), "T"), K).unknown) == {"e:U", "e:L"}
    u = next(c for c in ego_mem(w, K).unknown if c.subjects == ("e:U",))
    assert (u.gold, u.slice, u.answer_class, u.range_m, u.evidence_k, u.entities, u.decision_k) == (
        {"type": "unknown"}, "unknown", "unknown", None, (), (), K)
    assert u.slots == {"kind": "bag", "attributes": {"color": "yellow"}}


def test_a_look_alike_sighted_anywhere_in_the_window_hides_an_unseen_thing():
    """Contract (brief 2.2): an unseen thing is not named when a look-alike of its colour is sighted in
    the window, even a drum plainly seen for an unseen bin (nothing seen says which is meant), and even after the look
    point. By hand: an unseen yellow box with a yellow bag seen at 2, an unseen red bin with a red drum seen at 3, and
    an unseen blue box with a blue bag first seen at 20 are all out; an unseen purple box beside the yellow bag is in.
    Regression caught: the model pointing at the drum for a bin it never saw."""
    s = scene()
    put(s, "A", "bag", "yellow", (5.0, 3.13), [2])
    put(s, "Y", "box", "yellow", (6.0, 2.0))
    put(s, "D", "drum", "red", (6.5, 1.0), [3])
    put(s, "R", "bin", "red", (7.0, 1.0))
    put(s, "Z", "bag", "blue", (7.0, 2.0), [20])
    put(s, "X", "box", "blue", (7.5, 2.0))
    put(s, "P", "box", "purple", (7.5, 1.0))
    real = {c.subjects[0] for c in ego_mem(world(s), K).unknown if c.subjects[0].startswith("e:")}
    assert real == {"e:P"}


def test_absent_descriptions_are_of_kinds_already_sighted_and_match_nothing():
    """Contract (brief 2.2): the unknown pool's absent descriptions (`never_seen`) are of a kind sighted by the look
    point, and none matches a thing of the world. By hand: with only a bag sighted by 13 and a box first seen at 20, the
    decoys are all bags, though `never_seen` over the whole window lists boxes too; each is named `d:<kind>/<attrs>`.
    Regression caught: a decoy whose kind the robot has not met, which the kind alone gives away."""
    from stbench.descriptions import desc_key
    from stbench.questions_where import never_seen
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2])
    put(s, "X", "box", "green", (6.0, 2.0), [20])
    w = world(s)
    decoys = [c for c in ego_mem(w, K).unknown if c.subjects[0].startswith("d:")]
    assert decoys and {c.slots["kind"] for c in decoys} == {"bag"}
    assert "box" in {key[0] for key in never_seen(w)}
    assert all(c.subjects[0].startswith(f"d:{c.slots['kind']}/") and c.gold == {"type": "unknown"} for c in decoys)
    assert not {desc_key(c.slots["kind"], c.slots["attributes"]) for c in decoys} & set(w.groups)


def test_nothing_answerable_is_a_dead_band():
    """Contract: the draw needs both pools, so a point with nothing answerable is a dead band (None). Regression
    caught: an unknown pool offered alone, which would make every such point an unknown question and the slice
    predictable from whether anything was seen."""
    s = scene()
    put(s, "U", "bag", "yellow", (6.0, 2.0))
    assert ego_mem(world(s), K) is None
