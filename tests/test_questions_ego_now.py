"""`ego_now` (`questions_ego`): the gold, its rungs and which things it may name. The scene is `ego_kit`'s: the camera
at (3.0, 1.13, 0.08) heading north at the look point 13, every point and offset worked out by hand."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import family_kit as fk
from ego_kit import CAMERA, CLEARANCE_M, EGO_JOB, SMALL, K, cut, named, put, scene, told, world
from stbench.questions_ego import ego_now
from stbench.questions_text import FAMILIES
from synth import RECOG

FY = 381.36   # the vacuum's camera, pixels (`sensors/camera.json`)


def ask(s, **kw):
    return named(ego_now(world(s, **kw), K))


def test_the_gold_is_the_floor_hull_point_nearest_the_camera_in_the_cameras_frame():
    """Contract (FD-0.7.0 review, changes 6-7): the gold is the point of the convex hull of the thing's visible pixels,
    dropped on the floor, that lies nearest the camera's floor point, put in the delivered camera's frame (ahead, then
    left), and it may lie on a hull edge. The need is the job's clearance C (here 2.0), whatever the range, the harmful
    side radial, and `resolution_m` one pixel at the gold point, (d^2 + 0.0064) / (0.08 * 381.36) radially and
    sqrt(d^2 + 0.0064) / 381.36 sideways, together 0.2573, 0.1250, 0.0804 and 0.0423 for d = 2.8, 1.95, 1.5628 and
    1.1314 (the resolution tests work these out in full); `range_m` is the slant from the camera to that point at the
    height of the visible point nearest it. By hand
    (camera at (3.0, 1.13, 0.08), north): a red bag seen as the corners of a 0.2 x 0.4 face at y 3.93 and 4.33 has its
    nearest hull point (3.0, 3.93) in the middle of an edge, 2.8 ahead and 0.0 left, slant sqrt(2.8^2 + 0.08^2) =
    2.8011; a blue bag at (1.0, 1.13) has it on its east face, (1.05, 1.13): 0.0 ahead, 1.95 left (west of a robot
    facing north), slant 1.9516; a 0.9 x 0.6 x 0.5 box at (5.0, 1.63) has it at its corner (4.55, 1.33), 0.2 ahead and
    1.55 to the right, slant 1.5648; a 0.4 x 0.4 x 0.9 chair at (4.0, 2.13) at (3.8, 1.93), 0.8 ahead and 0.8 right,
    slant 1.1342, named by its kind alone. Regression caught: the middle asked in place of the near face (the old gold),
    the nearest vertex in place of the nearest point of an edge, the base used as the camera, left taken as right, a
    3-D nearest in place of a floor one, or a need that grows with the range."""
    s = scene()
    face = [(x, y, z) for x in (2.9, 3.1) for y in (3.93, 4.33) for z in (0.0, 0.3)]
    put(s, "T", "bag", "red", (3.0, 4.13), [K], pts=face)
    put(s, "W", "bag", "blue", (1.0, 1.13), [K])
    put(s, "X", "box", "green", (5.0, 1.63), [K], size=(0.9, 0.6, 0.5))
    put(s, "C", "chair", "oak", (4.0, 2.13), [K], size=(0.4, 0.4, 0.9), mobility="clutter")
    found = ask(s)
    assert list(found) == ["T", "W", "X", "C"]
    expected = {"T": ((2.8, 0.0), 2.8011, 0.2573), "W": ((0.0, 1.95), 1.9516, 0.1250),
                "X": ((0.2, -1.55), 1.5648, 0.0804), "C": ((0.8, -0.8), 1.1342, 0.0423)}
    for eid, ((ahead, left), r, rho) in expected.items():
        c = found[eid]
        assert c.gold == {"type": "offset", "ahead_m": pytest.approx(ahead, abs=1e-4),
                          "left_m": pytest.approx(left, abs=1e-4)}
        assert c.range_m == pytest.approx(r, abs=1e-4) and c.slice == "answerable" and c.answer_class is None
        assert c.evidence_k == (K,) and c.entities == (eid,) and c.subjects == (f"e:{eid}",) and c.decision_k == K
        assert c.tolerances == {"resolution_m": pytest.approx(rho, abs=1e-4), "need_m": 2.0, "harm": "radial"}
    assert found["T"].slots == {"kind": "bag", "attributes": {"color": "red"}}
    assert found["C"].slots == {"kind": "chair", "attributes": {}}
    assert CAMERA[:2] == (3.0, 1.13) and CLEARANCE_M == 2.0


def test_a_camera_over_the_thing_has_the_zero_offset_as_its_gold():
    """Contract (FD-0.7.0 review, 3e): when the camera's floor point lies inside the floor hull (a drone lowering
    over the thing) the nearest point is the camera's own floor point, an offset of (0, 0), a valid answer. By hand:
    a 1.2 x 1.2 x 0.5 box centred under the camera at (3.0, 1.13) is 0.0 ahead and 0.0 left, slant 0.08 (the visible
    point nearest it is a base corner, 0.08 m below the camera); its base corners, 0.85 m from the gold point, are floor
    pixels within C/2 = 1.0 and read there to (0.72 + 0.0064) / (0.08 * 381.36) = 0.024. Regression caught: the hull
    boundary used when the ground point is inside (an answer of a metre for a thing the camera is over)."""
    s = scene()
    put(s, "O", "box", "green", (3.0, 1.13), [K], size=(1.2, 1.2, 0.5))
    o = ask(s)["O"]
    assert o.gold == {"type": "offset", "ahead_m": pytest.approx(0.0, abs=1e-4), "left_m": pytest.approx(0.0, abs=1e-4)}
    assert o.range_m == pytest.approx(0.08, abs=1e-4)


def test_the_text_is_pure_and_asks_for_the_point_nearest_the_camera_along_the_floor():
    """Contract (FD-0.7.0 review, change 6): the text is the render of the slots alone and names the point and the
    frame in words. A red bag reads 'Where is the point of the red bag you can see now that is nearest to your camera,
    measured along the floor? Answer in metres ahead of the camera and to its left.', a chair 'the chair'. Regression
    caught: a wording that says 'middle' or 'nearest part' (visible or true, 3-D or floor left open) or drops the frame
    (an answer in the wrong axes), or a slot the gold needs but the text hides."""
    render = FAMILIES["ego_now"].render
    tail = " Answer in metres ahead of the camera and to its left."
    assert render({"kind": "bag", "attributes": {"color": "red"}}) == (
        "Where is the point of the red bag you can see now that is nearest to your camera, measured along the floor?"
        + tail)
    assert render({"kind": "chair", "attributes": {}}) == (
        "Where is the point of the chair you can see now that is nearest to your camera, measured along the floor?"
        + tail)


@pytest.mark.parametrize("clearance, z, asked", [
    (0.52, 0.0, True),    # the floor pixel reads to 0.2575 m, under C/2 = 0.26
    (0.50, 0.0, False),   # ... and over C/2 = 0.25
    (2.0, 0.02, True),    # a point 2 cm up is still a floor pixel
    (2.0, 0.03, False),   # 3 cm up is not
    (2.0, 0.11, False),   # 3 cm above the camera: no floor pixel, whatever its own height would read
])
def test_a_thing_is_asked_only_where_a_floor_pixel_reads_within_half_the_clearance(clearance, z, asked):
    """Contract (the floor read, `nearest_part.floor_read_error` and `resolvable`): ask only where a pixel of the thing
    at most 2 cm above the floor lies within C/2 of the gold point and one pixel there, read on the floor plane, moves
    it by at most C/2: (d^2 + H^2) / (H fy) with d the floor distance from the camera and H its height. By hand: the
    camera at (3.0, 1.13, 0.08) and a bag whose points are (2.9, 3.93), (3.1, 3.93) and (3.0, 4.33) at height z, so
    the gold point is (3.0, 3.93), 0.1 m from the nearest two: at z 0 each has d^2 = 2.8^2 + 0.1^2 = 7.85 and e =
    (7.85 + 0.0064) / (0.08 * 381.36) = 0.2575, asked at C/2 = 0.26 and not at 0.25; z = 0.02 is a floor pixel, 0.03
    and 0.11 are not (the old rule read the point at its own height, 0.11 as (7.85 + 0.0009) / (0.03 * 381.36) = 0.69,
    and asked it). Regression caught: a screen that reads the thing's size, the job or the range (it reads only the
    camera, the pose, the points and C), a point read at its own height, or a floor tolerance of the wrong size."""
    s = scene()
    put(s, "T", "bag", "red", (3.0, 4.13), [K], pts=[(2.9, 3.93, z), (3.1, 3.93, z), (3.0, 4.33, z)])
    job = replace(EGO_JOB, clearance=clearance)
    assert (set(named(ego_now(fk.world(s, job=job), K))) == {"T"}) is asked


def test_a_job_that_declares_the_fallback_keeps_a_screened_question_with_the_pixel_as_its_resolution():
    """Contract (FD-0.7.0 review, change 5, the declared fallback; 0.9.0): on a job whose `screen_fallback` names the
    family a question the screen drops is kept, its resolution the pixel the contact reads to, wider than the need (the
    kernel widens the need to twice it, so the full quarter is what one pixel could miss); a job that names another
    family, or none, drops it; a thing with no floor pixel near the point stays dropped. By hand (C = 0.1; the camera
    0.08 m above the floor, fy = fx = 381.36115): a bag whose nearest points are (2.9, 3.93), (3.0, 3.93) and (3.1,
    3.93) on the floor (the gold point (3.0, 3.93) is one of them) has a contact error of (2.8^2 + 0.0064) / (0.08 *
    381.36115) = 0.2572 radially, and the point 0.2573 with the sideways pixel 0.0073: that is the resolution, against
    the need 0.1; the same bag 0.5 m up has no floor pixel and no error. Regression caught: a fallback that is always on
    (a point no camera could place kept for every job), one that ignores which family the job named, one that keeps a
    thing no floor pixel ties to the floor (an unbounded error kept), or a kept row whose resolution is the floor of 1
    cm."""
    for z, resolution in ((0.0, 0.2573), (0.5, None)):
        s = scene()
        put(s, "T", "bag", "red", (3.0, 4.13), [K], pts=[(2.9, 3.93, z), (3.0, 3.93, z), (3.1, 3.93, z),
                                                          (3.0, 4.33, z)])
        for declared, kept_expected in ((frozenset(), False), (frozenset({"ego_mem"}), False),
                                        (frozenset({"ego_now"}), resolution is not None)):
            job = replace(EGO_JOB, clearance=0.1, screen_fallback=declared)
            kept = named(ego_now(fk.world(s, job=job), K))
            assert (list(kept) == ["T"]) is kept_expected
            if kept_expected:
                assert kept["T"].tolerances == {"resolution_m": pytest.approx(resolution, abs=1e-4), "need_m": 0.1,
                                                "harm": "radial"}


def test_every_asked_gold_passes_the_screen_and_a_farther_thing_does_not():
    """Contract (FD-0.7.0 review, change 5): every ego gold that is asked has one pixel at its point within C/2, and a
    thing past that range is screened out, the range being the only thing that differs. By hand (C = 2.0, so C/2 = 1.0;
    the camera 0.08 m above the floor, fy = 381.36, a bag seen as four points on the floor; a tall one would also show a
    point high up, with a smaller error): its near face 3.0, 5.0 and 5.6 m ahead gives (d^2 + 0.0064) / (0.08 * 381.36)
    = 0.295, 0.820 and 1.028: the first two are asked and the third is not. Regression caught: a screen that is not
    applied (the far bag asked), or one that follows the thing's size or its colour in place of the camera, the pose and
    the point."""
    s = scene()
    for eid, colour, ahead in (("N", "red", 3.0), ("M", "blue", 5.0), ("F", "green", 5.6)):
        y = 1.13 + ahead
        floor = [(x, y + dy, 0.0) for x in (2.95, 3.05) for dy in (0, 0.1)]
        put(s, eid, "bag", colour, (3.0, y + 0.05), [K], pts=floor)
    asked = ask(s)
    assert set(asked) == {"N", "M"}
    for c in asked.values():
        d = math.hypot(c.gold["ahead_m"], c.gold["left_m"])
        assert (d * d + 0.08 ** 2) / (0.08 * FY) <= CLEARANCE_M / 2


def test_only_a_thing_recognisable_in_the_look_frame_is_named():
    """Contract (brief 2.1): the certificate is recognisable in frame k, and no other frame stands in. By hand: a bag
    seen at 1,500 px at keyframe 13 is asked; one seen so only at 12, one at 500 px at 13 (a glimpse above the dead band
    but under the recognisable floor), and one at 1,500 px at 13 too dark to name (lum 0.005) are not. Regression
    caught: a thing seen a frame earlier asked as if it were in view now, or a speck named."""
    s = scene()
    put(s, "A", "bag", "red", (3.0, 4.13), [K])
    put(s, "B", "bag", "blue", (1.0, 1.13), [K - 1])
    put(s, "C", "bag", "green", (5.0, 1.63), [K], px=500)
    put(s, "D", "bag", "orange", (2.0, 3.0), [K])
    s.see(K, "D", RECOG, lum=0.005)
    assert list(ask(s)) == ["A"]


def test_a_thing_the_robots_log_tells_about_is_never_named():
    """Contract (brief 2.1): told things are never named, since the log, not the picture, would answer. By hand: the red
    bag is the third target of the task (the helper's carried thing), so only the blue bag is asked. Regression caught:
    a question answered from the subgoal text."""
    s = scene()
    put(s, "T", "bag", "red", (3.0, 4.13), [K])
    put(s, "W", "bag", "blue", (1.0, 1.13), [K])
    assert list(named(ego_now(told(world(s), "T"), K))) == ["W"]


@pytest.mark.parametrize("twin_kind, twin_colour, twin_px, asked", [
    ("bag", "red", RECOG, set()),            # two red bags: neither is "the red bag"
    ("bag", "red", 40, set()),               # a twin that is only a few pixels in frame k is still a twin
    ("box", "red", RECOG, set()),            # a box can be taken for a bag (CONFUSABLE), both ways
    ("bag", "blue", RECOG, {"A", "B"}),      # another colour is told apart
    ("mug", "red", RECOG, {"A", "B"}),       # a kind it cannot be mistaken for
])
def test_a_description_another_thing_in_view_could_fit_is_not_asked(twin_kind, twin_colour, twin_px, asked):
    """Contract (brief 2.1): nothing else in frame k of a confusable kind (its own included) may be taken for the
    description; a row at any pixel count counts. By hand: a red bag "A" beside a "B" at (1.0, 1.13) is asked, with B,
    only when B cannot be taken for it: a blue bag or a red mug (a mug is no bag, box, tote or bin); beside a red bag
    (even a 40 px sliver of one) or a red box neither is. Regression caught: 'the red bag' asked with two in view (an
    answer neither the table nor the model can tell apart)."""
    s = scene()
    put(s, "A", "bag", "red", (3.0, 4.13), [K])
    put(s, "B", twin_kind, twin_colour, (1.0, 1.13), [K], px=twin_px)
    assert set(ask(s)) == asked


@pytest.mark.parametrize("drum_px, asked", [(RECOG, {"B", "D"}), (100, set())])
def test_a_drum_in_view_hides_a_bin_only_when_it_is_not_plainly_made_out(drum_px, asked):
    """Contract (DECISIONS 'A never-seen bin or sack ...'): a bin and a drum are told apart by their shape
    when the drum is plainly in view, so a recognisable red drum beside a red bin hides neither, but a drum that is a
    sliver of the frame (100 px, not recognisable) could be taken for the bin, so the bin is not asked (and the sliver
    is no question). By hand: the same red bin and red drum, the drum at 1,500 px then at 100. Regression caught: the
    unseen-only groups applied to everything (a drum plainly in view hiding the bin, the supply gone), or to nothing
    (the bin asked beside a sliver that is a drum)."""
    s = scene()
    put(s, "B", "bin", "red", (3.0, 4.13), [K])
    put(s, "D", "drum", "red", (1.0, 1.13), [K], px=drum_px)
    assert set(ask(s)) == asked


def test_a_label_the_picture_does_not_bear_out_is_not_a_name():
    """Contract (DECISIONS 'A question names a thing only by a colour its render shows'): a carried thing is named
    only when the colour frame k's `rgb_med` confidently shows is its label and the frames up to k bear it out. By
    hand: a bag labelled black that renders silver (85, 85, 85 reads grey) in frame 13, one labelled purple that
    renders next to a boundary (180, 60, 170 reads pink or purple), one labelled green that renders green at 13 but
    orange in four earlier frames (the colour most of its frames show), and one labelled blue that renders blue in four
    earlier frames and next to a boundary at 13 (30, 125, 140 reads blue or green) are not asked; a red bag rendering
    red is. Regression caught: the catalog's word used where the picture says another (the model is asked for a black
    bag that looks grey)."""
    s = scene()
    put(s, "A", "bag", "red", (3.0, 4.13), [K])
    put(s, "B", "bag", "black", (1.0, 1.13), [K], rgb=(85, 85, 85))
    put(s, "C", "bag", "purple", (5.0, 1.63), [K], rgb=(180, 60, 170))
    put(s, "D", "bag", "green", (2.0, 3.0), [K])
    s.see([8, 9, 10, 11], "D", RECOG, rgb=(240, 130, 20))
    put(s, "E", "bag", "blue", (6.5, 3.0), [8, 9, 10, 11])
    s.see(K, "E", RECOG, rgb=(30, 125, 140))
    assert list(ask(s)) == ["A"]


def test_the_pool_at_k_does_not_read_a_frame_after_k():
    """Contract (plan 4 A3): naming and gold read no frame after the look point, so the pool is the same whether the
    window goes on or is cut at k. By hand, after the look point 13: a twin of the red bag shows at 20, the green bag
    renders orange in five frames (20 to 24) that would outvote its three green ones, and a blue bag "X" in view at 13
    renders red in those five frames (so a look at the whole window would take it for a red bag and hide "the red
    bag"). Regression caught: a future frame deciding which question is asked (a twin that does not exist yet hiding the
    thing, a later render vetoing a name or turning a look-alike red)."""
    s = scene()
    put(s, "A", "bag", "red", (3.0, 4.13), [K])
    put(s, "A2", "bag", "red", (6.0, 1.63), [20])
    put(s, "B", "bag", "green", (1.0, 1.13), [K, 9, 10])
    s.see([20, 21, 22, 23, 24], "B", RECOG, rgb=(240, 130, 20))
    put(s, "X", "bag", "blue", (5.0, 3.0), [K])
    s.see([20, 21, 22, 23, 24], "X", RECOG, rgb=(200, 30, 30))
    full, early = ego_now(world(s), K), ego_now(cut(s), K)
    assert [(c.key, c.gold, c.tolerances) for c in full.answerable] == [
        (c.key, c.gold, c.tolerances) for c in early.answerable] and set(named(full)) == {"A", "B", "X"}


def test_a_look_point_may_be_any_frame_and_reads_that_frame_alone():
    """Contract (0.10.0, one frame per `observe`): a look point is any frame of the window, and `ego_now` reads the
    frame it is asked at alone. By hand: the camera at keyframe 12 is where it is at 13, (3.0, 1.13) heading north; a
    red bag 0.1 m across at (3.0, 4.13) is seen at 12 and 13; keyframe 12 ends no decision window (the window 6-13 ends
    at 13), and the gold at 12 is the bag's near face in frame 12, (3.0, 4.08): 2.95 ahead and 0.0 left, evidence (12,),
    decision point 12; frame 11 shows no bag and asks nothing. Regression caught: a gold refused at a mid-window frame,
    or one read from the frame after it."""
    s = scene(poses={12: (3.0, 1.0)}, yaws={12: math.pi / 2})
    put(s, "A", "bag", "red", (3.0, 4.13), [12, K])
    a = named(ego_now(world(s), 12))["A"]
    assert (a.evidence_k, a.decision_k) == ((12,), 12)
    assert a.gold == {"type": "offset", "ahead_m": pytest.approx(2.95, abs=1e-4),
                      "left_m": pytest.approx(0.0, abs=1e-4)}
    assert ego_now(world(s), 11) is None


def test_no_pool_is_a_dead_band():
    """Contract: with nothing nameable in frame k the point is a dead band (None), not an empty moment. Regression
    caught: an empty pool handed to the draw, which would count the point as asked."""
    assert ego_now(world(scene()), K) is None
    assert SMALL[2] == 0.4


def test_a_kind_only_name_is_hidden_by_any_look_alike_kind_in_view():
    """Contract (brief 2.1): clutter is named by kind alone, so a chair is not 'the chair' when a stool, a kind
    it can be mistaken for (`CONFUSABLE`), is in the frame, whatever the stool's colour, and neither is when a second
    chair is; a mug in view changes nothing. By hand: one chair at 1,500 px beside each. Regression caught: a kind-only
    name asked beside a stool (the table's `matching` by kind would find both)."""
    for other, mobility, asked in (("stool", "light", set()), ("chair", "light", set()),
                                   ("mug", "carried", {"C", "O"})):
        s = scene()
        put(s, "C", "chair", "oak", (4.0, 2.13), [K], size=(0.4, 0.4, 0.9), mobility="clutter")
        put(s, "O", other, "oak" if mobility == "light" else "red", (2.0, 2.13), [K], size=(0.3, 0.3, 0.5),
            mobility=mobility)
        assert set(ask(s)) == asked, other
