"""The shared rules of what the robot can know: the "in view" test over planar depth, and a room's change set."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seen_bounds import use
from stbench import visibility
from stbench.camera import Camera
from stbench.episode import write_jsonl
from stbench.geometry import quat_yaw_pitch
from stbench.render_encode import write_png16
from stbench.rules import (
    Item,
    Viewer,
    coarse_depth,
    glimpse_px,
    gold_seen_px,
    load_depth,
    min_cover_px,
    room_events,
    seen_px,
    would_cover_px,
)
from stbench.sightings import Sightings, kinds_of
from synth import DEPTH_MM, camera, depth_with_wall, three_tasks

CAM = camera()
CM_CAM = replace(CAM, depth_unit="cm")
POS = (0.0, 0.0, 1.5)
FACING_X = tuple(quat_yaw_pitch(0.0, 0.0))
MUG = (0.1, 0.1, 0.1)


def viewer(depth=None):
    img = np.full((480, 640), DEPTH_MM, dtype=np.uint16) if depth is None else depth
    return Viewer(CAM, {0: (POS, FACING_X)}, lambda k: img)


def test_the_seen_threshold_has_one_owner():
    """Contract: the 215 px that "in view" requires and the threshold that makes an entity "seen" are one function,
    owned by rules.py and imported by visibility.py, and the compiler's `gold_seen_px` is derived from it. Regression
    caught: two copies drifting apart, so a spot counts as in view where nothing could be seen."""
    assert visibility.seen_px is seen_px and min_cover_px(CAM) == seen_px(CAM) == 215


def test_the_compiler_counts_a_sighting_as_seen_only_from_269_px():
    """Contract (DECISIONS, a sighting decides a gold only from 269 px): the compiler's sightings are seen from
    ceil(1.25 x 215) = 269 px; 1..268 px is its dead band, so a 215..268 px record that the oracle channel detects
    decides no gold; 0 px (no record) is neither. Regression caught: the compiler and the oracle channel sharing one
    threshold again, under which a detector at 269 px got 35 of 610 golds wrong on the 24 episodes (changed 17 of
    79)."""
    assert seen_px(CAM) == 215 and gold_seen_px(CAM) == 269
    s = Sightings([{"k": 0, "entities": [{"id": "A", "px": 268}, {"id": "B", "px": 269}, {"id": "C", "px": 215},
                                         {"id": "D", "px": 1}]},
                   {"k": 1, "entities": [{"id": "A", "px": 300}]}], CAM)
    assert (s.seen("A"), s.band("A")) == ([1], [0])
    assert (s.seen("B"), s.band("B")) == ([0], [])
    assert (s.seen("C"), s.band("C")) == ([], [0])
    assert (s.seen("D"), s.band("D")) == ([], [0])
    assert (s.seen("E"), s.band("E")) == ([], [])


def test_every_threshold_is_the_kinds_and_a_kind_without_an_entry_is_global(monkeypatch):
    """Contract (V0 section 6, per kind): with mug = (lo 20, hi 200 px) and glasses = (30, 400) at 640x480, a kind's
    seen threshold is its hi (200, 400), its glimpse bound its lo (20, 30), the compiler's sure-seen bound 1.25 x hi
    rounded up (250, 500: the margin rule of the global 269 = ceil(1.25 x 215)) and the in-view cover its hi. A kind
    with no entry (a bowl, fixed furniture, no kind at all) keeps 215, 1 px, 269 and 215.
    Regression caught: a lookup that ignores the kind (so a table changes nothing), a margin dropped from the kind's
    sure-seen bound, or a kind's entry leaking into another kind."""
    use(monkeypatch, mug=(20, 200), glasses=(30, 400))
    got = {k: (seen_px(CAM, k), glimpse_px(CAM, k), gold_seen_px(CAM, k), min_cover_px(CAM, k))
           for k in ("mug", "glasses", "bowl", "door", None)}
    assert got == {"mug": (200, 20, 250, 200), "glasses": (400, 30, 500, 400), "bowl": (215, 1, 269, 215),
                   "door": (215, 1, 269, 215), None: (215, 1, 269, 215)}
    assert (seen_px(CAM), glimpse_px(CAM), gold_seen_px(CAM), min_cover_px(CAM)) == (215, 1, 269, 215)


def test_the_bounds_scale_with_the_image(monkeypatch):
    """Contract: the table holds fractions of the image, so at 320x240 (a quarter of the pixels) mug = (20, 200) is
    (5, 50) px with the sure-seen bound ceil(1.25 x 50) = 63, and the global bounds are round(0.0007 x 76800) = 54, 1
    and ceil(1.25 x 54) = 68. Regression caught: bounds kept as pixel counts of 640x480."""
    use(monkeypatch, mug=(20, 200))
    small = Camera(320, 240, 190.0, 190.0, 160.0, 120.0)
    assert (glimpse_px(small, "mug"), seen_px(small, "mug"), gold_seen_px(small, "mug")) == (5, 50, 63)
    assert (glimpse_px(small), seen_px(small), gold_seen_px(small)) == (1, 54, 68)


def test_a_kinds_bounds_decide_its_sightings_and_nothing_under_lo_is_a_glimpse(monkeypatch):
    """Contract (V0 section 7 per kind): with mug = (20, 200) and glasses = (30, 400), a mug is a glimpse (the dead
    band) from 20 px up to 249 and seen from 250 (its 240 px record is band here: the oracle channel detects it, the
    compiler needs 250); glasses are band from 30 to 499 and seen from 500; a bowl keeps 1..268 band and 269 seen. A
    record under lo (19 px of a mug, 29 of glasses) is in neither list: unseen, not band. With no `kind_of` every
    entity has the global bounds. Frames: k0 19, 29, 1 px; k1 20, 30, 268; k2 249, 499, 269; k3 250, 500; k4 240 (mug).
    Regression caught: a kind's bounds ignored, a below-lo glimpse still counted as the dead band, an off-by-one at
    lo (`>` for `>=`), or the sure-seen margin missing."""
    use(monkeypatch, mug=(20, 200), glasses=(30, 400))
    rows = [{"k": 0, "entities": [{"id": "M", "px": 19}, {"id": "G", "px": 29}, {"id": "B", "px": 1}]},
            {"k": 1, "entities": [{"id": "M", "px": 20}, {"id": "G", "px": 30}, {"id": "B", "px": 268}]},
            {"k": 2, "entities": [{"id": "M", "px": 249}, {"id": "G", "px": 499}, {"id": "B", "px": 269}]},
            {"k": 3, "entities": [{"id": "M", "px": 250}, {"id": "G", "px": 500}]},
            {"k": 4, "entities": [{"id": "M", "px": 240}]}]
    s = Sightings(rows, CAM, {"M": "mug", "G": "glasses", "B": "bowl"}.get)
    assert (s.seen("M"), s.band("M")) == ([3], [1, 2, 4])
    assert (s.seen("G"), s.band("G")) == ([3], [1, 2])
    assert (s.seen("B"), s.band("B")) == ([2], [0, 1])
    plain = Sightings(rows, CAM)
    assert (plain.seen("M"), plain.band("M")) == ([], [0, 1, 2, 3, 4])
    assert (plain.seen("G"), plain.band("G")) == ([2, 3], [0, 1])  # 499 and 500 px clear 269


def test_load_reads_the_log_with_the_kinds_bounds(monkeypatch, tmp_path):
    """Contract: `Sightings.load(path, cam, kind_of)` (what the compiler's `load_inputs` will call) classifies the
    written visibility log by the kind's bounds like the constructor does: with mug = (20, 200) a mug record of 19 px
    is nowhere, 20 px is band and 250 px is seen, and with no `kind_of` all three are band. Regression caught: `load`
    dropping `kind_of`, so the compiler on files ignores a table the compiler on rows follows."""
    use(monkeypatch, mug=(20, 200))
    path = tmp_path / "visibility.jsonl"
    write_jsonl(path, [{"k": k, "entities": [{"id": "M", "px": px}]} for k, px in enumerate([19, 20, 250])])
    s = Sightings.load(path, CAM, {"M": "mug"}.get)
    assert (s.seen("M"), s.band("M")) == ([2], [1])
    plain = Sightings.load(path, CAM)
    assert (plain.seen("M"), plain.band("M")) == ([], [0, 1, 2])


def test_kinds_of_names_the_kind_of_each_entity_of_a_timeline():
    """Contract: `kinds_of(timeline)` is the `kind_of` the compiler's sightings take: an entity id to its kind, None for
    an id the timeline does not hold (that entity then has the global bounds). Regression caught: a mapping by index or
    one that raises on an unknown id."""
    s = three_tasks()
    s.item("M1", "mug", {"color": "red"}, [(0, None, (1, 1, 0.9))])
    s.item("G1", "glasses", {"color": "black"}, [(0, None, (2, 1, 0.9))])
    kind_of = kinds_of(s.timeline())
    assert (kind_of("M1"), kind_of("G1"), kind_of("nobody")) == ("mug", "glasses", None)


def test_coarse_depth_keeps_the_nearest_surface_of_each_block_and_ignores_missing_depth():
    """Contract: `coarse_depth` reduces each 4x4 block to its nearest nonzero depth (0 only if the whole block
    has none), so a thin occluder is not averaged away and a hole in the depth image does not block. A 4x8 image with
    one 1000 mm pixel in the first block and a block of zeros in the second gives [[1000, 0]].
    Regression caught: a mean or a plain subsample, which would make the in-view test depend on the sampled pixel."""
    img = np.full((4, 8), 6000, dtype=np.uint16)
    img[2, 1] = 1000
    img[:, 4:] = 0
    assert coarse_depth(img).tolist() == [[1000, 0]]
    assert coarse_depth(np.zeros((8, 8), dtype=np.uint16)).shape == (2, 2)


def test_a_depth_png_is_read_coarse_in_the_files_own_unit_and_a_missing_one_is_none(tmp_path):
    """Contract (`rules.load_depth`; plan-city §2: a map's depth PNGs are in the unit its camera file names): a 16-bit
    depth PNG is read as `coarse_depth` of its pixels, counts as the file holds them (a centimetre image is not scaled
    to millimetres here: every reader scales by the camera file's unit), and a file that is not there is None. A 4 x 8
    image of 600 with a 100 in its first block reads [[100, 600]]. Regression caught: a loader that scales the counts
    or reads a block's first pixel instead of its nearest, or one that raises where the baselines expect None."""
    img = np.full((4, 8), 600, dtype=np.uint16)
    img[1, 2] = 100
    write_png16(tmp_path / "depth.png", img)
    assert load_depth(tmp_path / "depth.png").tolist() == [[100, 600]]
    assert load_depth(tmp_path / "missing.png") is None


def test_in_view_works_on_a_coarse_depth_image():
    """Contract: the viewer accepts the coarse image the table keeps in its state, so the compiler (full files,
    coarsened) and a system (its stored summaries) agree. A wall 1.8 m away blocks a point 2 m ahead at either
    resolution. Regression caught: coordinates not scaled for a smaller depth image."""
    wall = coarse_depth(depth_with_wall(1.8))
    assert wall.shape == (120, 160)
    assert not viewer(wall).in_view([0], (2.0, 0.0, 1.5), MUG)
    assert viewer(coarse_depth(depth_with_wall(1.95))).in_view([0], (2.0, 0.0, 1.5), MUG)


def test_perturbed_thresholds_can_flip_a_borderline_view():
    """Contract: `cover_scale` and `depth_tol` move the two thresholds. A mug at 2.4 m covers 252 px: in view at
    the rule (215), not with the in-view cover threshold +25% (269); a wall 0.12 m in front of a point is in the way at
    the rule's 0.1 m tolerance and not at 0.15 m. This is what lets the compiler drop events the thresholds could
    flip. Regression caught: perturbations that do nothing, so every fragile event looks robust."""
    img = np.full((480, 640), DEPTH_MM, dtype=np.uint16)
    p = (2.4, 0.0, 1.5)
    assert would_cover_px(MUG, 2.4, CAM) == pytest.approx(252.4, abs=0.5)
    assert Viewer(CAM, {0: (POS, FACING_X)}, lambda k: img).in_view([0], p, MUG)
    assert not Viewer(CAM, {0: (POS, FACING_X)}, lambda k: img, cover_scale=1.25).in_view([0], p, MUG)
    near = depth_with_wall(2.28)
    assert not Viewer(CAM, {0: (POS, FACING_X)}, lambda k: near).in_view([0], p, MUG)
    assert Viewer(CAM, {0: (POS, FACING_X)}, lambda k: near, depth_tol=0.15).in_view([0], p, MUG)


def test_in_view_needs_the_cover_of_the_kinds_seen_bound_and_the_band_is_the_kinds_too(monkeypatch):
    """Contract: "in view" asks a thing to cover its KIND's seen bound, and the compiler's +-25% cover variants scale
    that bound. A mug-sized thing 2.4 m ahead covers 252 px: in view for a kind with no entry (215) and for mug =
    (20, 200), not for glasses = (30, 400); 3.0 m ahead it covers 162: in view for toy = (20, 100), not for the global
    215. At 2.4 m (252 px) a toy with hi 200 stays in view when the cover threshold is raised 25% (250), one with hi 210
    does not (262.5). Regression caught: a viewer that ignores the kind (the compiler and the baseline would agree with
    each other and disagree with the oracle channel), or a variant that scales the global threshold."""
    use(monkeypatch, mug=(20, 200), glasses=(30, 400), toy=(20, 100))
    img = np.full((480, 640), DEPTH_MM, dtype=np.uint16)
    near, far = (2.4, 0.0, 1.5), (3.0, 0.0, 1.5)
    v = Viewer(CAM, {0: (POS, FACING_X)}, lambda k: img)
    assert [v.in_view([0], near, MUG, kind) for kind in (None, "bowl", "mug", "glasses")] == [True, True, True, False]
    assert [v.in_view([0], far, MUG, kind) for kind in (None, "toy")] == [False, True]
    wide = Viewer(CAM, {0: (POS, FACING_X)}, lambda k: img, cover_scale=1.25)
    use(monkeypatch, toy=(20, 200))
    assert wide.in_view([0], near, MUG, "toy")
    use(monkeypatch, toy=(20, 210))
    assert v.in_view([0], near, MUG, "toy") and not wide.in_view([0], near, MUG, "toy")


def test_a_point_ahead_in_clear_view_is_in_view():
    """Contract: a mug-sized thing 2 m dead ahead covers 381.36^2 * 0.01 / 4 = 363 px >= 215 and nothing is nearer,
    so it is in view. Regression caught: a projection or depth-unit slip that rejects every clear point."""
    assert would_cover_px(MUG, 2.0, CAM) == pytest.approx(363.6, abs=0.2)
    assert viewer().in_view([0], (2.0, 0.0, 1.5), MUG)


def test_a_point_too_far_to_cover_enough_pixels_is_not_in_view():
    """Contract: at 3 m a mug-sized thing would cover 161 px < 215, so an empty spot there proves nothing.
    Regression caught: declaring things removed from places the camera could not have resolved them."""
    assert would_cover_px(MUG, 3.0, CAM) == pytest.approx(161.6, abs=0.2)
    assert not viewer().in_view([0], (3.0, 0.0, 1.5), MUG)


@pytest.mark.parametrize(("point", "why"), [((-1.0, 0.0, 1.5), "behind the camera"), ((0.2, 0.0, 1.5), "z below 0.3"),
                                           ((2.0, 3.0, 1.5), "off the left of the image"),
                                           ((2.0, 0.0, 3.5), "above the top of the image")])
def test_points_outside_the_view_volume_are_not_in_view(point, why):
    """Contract: a point must project inside the 640x480 image with camera z in (0.3, 150). Regression caught: a
    missing bounds check treating off-screen places as observed."""
    assert not viewer().in_view([0], point, MUG), why


FORKLIFT = (2.6, 1.1, 2.1)  # a forklift-sized box, cross-section 3.304 m2
BLOCK = (12.0, 12.0, 12.0)  # a box that covers 215 px or more far beyond 150 m, so only the far bound can decide


@pytest.mark.parametrize(("x", "in_view"), [(9.0, True), (31.0, True), (140.0, True), (149.9, True), (150.1, False),
                                           (160.0, False)])
def test_a_clear_point_is_in_view_out_to_150_m_and_not_beyond(x, in_view):
    """Contract (plan-city §5.4): the in-view rule's far bound is 150 m, past the distance at which a drone
    still sees the largest carried kind (a box truck covers 0.75 of its seen bound out to 116 m) and past the downtown
    map's longest in-map distance (139 m), where it was 30 m and 8 m before. A box of 12 x 12 x 12 m
    (cross-section 144 m2) covers 381.36^2 x 144 / z^2 = 818 px at 160 m, over the 215 px bound, and the depth image (in
    centimetres, as the downtown map's are, and holding 65535 = 655.35 m, the far scenery: nothing nearer than the
    point) shows no surface, so only the far bound decides: in view at 9, 31, 140 and 149.9 m, not at 150.1 and 160 m.
    Regression caught: the old 30 m bound (the drone's removals at 40 m neither events nor don't-care regions), or a
    bound that lets a point past the range count as seen."""
    assert would_cover_px(BLOCK, 160.0, CM_CAM) == pytest.approx(818.0, abs=0.5)
    assert would_cover_px(BLOCK, 160.0, CM_CAM) > min_cover_px(CM_CAM)
    far = np.full((480, 640), 65535, dtype=np.uint16)
    v = Viewer(CM_CAM, {0: (POS, FACING_X)}, lambda k: far)
    assert v.in_view([0], (x, 0.0, 1.5), BLOCK) is in_view


@pytest.mark.parametrize(("unit", "x", "in_view"), [("mm", 9.95, True), ("mm", 10.5, False), ("mm", 99.95, False),
                                                    ("cm", 99.95, True), ("cm", 100.5, False), ("cm", 10.5, True)])
def test_a_depth_image_is_read_in_the_unit_of_its_cameras_file(unit, x, in_view):
    """Contract (plan-city §2, `camera.json`'s `depth_unit`): the same image is a different surface under a different
    unit. A flat image of 10000 is a wall 10 m away to a camera whose unit is mm and one 100 m away to a camera whose
    unit is cm. A point 9.95 m ahead (within the 0.1 m tolerance of the 10 m wall) is in view and one at 10.5 or 99.95
    m is behind it; under cm the wall is at 100 m, so 99.95 m is in view, 100.5 m is behind it, and 10.5 m (in front)
    is in view. Regression caught: the depth read as millimetres on a centimetre map (every point past 10 m blocked by
    a wall that is 100 m away), or as centimetres on a millimetre one (the apartment's walls no longer blocking)."""
    wall = np.full((480, 640), 10000, dtype=np.uint16)
    v = Viewer(replace(CAM, depth_unit=unit), {0: (POS, FACING_X)}, lambda k: wall)
    assert v.in_view([0], (x, 0.0, 1.5), BLOCK) is in_view


def test_a_removal_the_robot_could_see_from_12_m_is_a_removed_event():
    """Contract (plan-warehouse §5.2, VISION rule 2): a thing seen in the earlier visit whose clear place, 12 m from the
    camera, is in view in the later one and is gone is `removed`, reported at the old place, with no don't-care region
    (at the old 8 m bound it was neither, so a correct report of it was charged). Regression caught: the far bound
    cutting a visible removal out of the gold."""
    clear = viewer(np.zeros((480, 640), dtype=np.uint16))
    gone = Item("a", FORKLIFT, a=(12.0, 0.0, 1.5))
    events, dont_care = room_events("hall", lambda xy: "hall", [gone], lambda c, s, kind: False,
                                    lambda c, s, kind: clear.in_view([0], c, s, kind))
    assert events == [("removed", (12.0, 0.0))] and dont_care == []


def test_a_nearer_surface_blocks_the_view_only_beyond_the_depth_tolerance():
    """Contract: the planar depth at the point's pixel must be at least z - 0.1 m. A wall at 1.95 m does not block a
    point at 2.0 m (within 0.1), a wall at 1.8 m does. Regression caught: seeing through walls, or blocking on
    the entity's own front surface."""
    p = (2.0, 0.0, 1.5)
    assert viewer(depth_with_wall(1.95)).in_view([0], p, MUG)
    assert not viewer(depth_with_wall(1.8)).in_view([0], p, MUG)


def test_missing_depth_means_nothing_is_in_view_and_zero_depth_means_nothing_blocks():
    """Contract: with no depth image, in view is False (occlusion cannot be ruled out); a depth of 0 (no surface
    recorded) blocks nothing. Regression caught: a run without depth files silently producing `removed` events."""
    p = (2.0, 0.0, 1.5)
    assert not Viewer(CAM, {0: (POS, FACING_X)}, lambda k: None).in_view([0], p, MUG)
    assert viewer(np.zeros((480, 640), dtype=np.uint16)).in_view([0], p, MUG)


def test_any_keyframe_of_the_visit_may_provide_the_view():
    """Contract: `in_view(ks, ...)` is True if at least one listed keyframe sees the point. Regression caught: only
    the first or last keyframe of a visit being consulted."""
    behind = tuple(quat_yaw_pitch(np.pi, 0.0))
    v = Viewer(CAM, {0: (POS, behind), 1: (POS, FACING_X)}, lambda k: np.full((480, 640), DEPTH_MM, dtype=np.uint16))
    assert not v.in_view([0], (2.0, 0.0, 1.5), MUG)
    assert v.in_view([0, 1], (2.0, 0.0, 1.5), MUG)


# -- room_events ---------------------------------------------------------------------------------------------------
def room_of(xy):
    return "kitchen" if xy[0] < 4 else "living"


def at(x, y):
    return (x, y, 0.95)


def events(items, seen_i=True, seen_j=True):
    return room_events("kitchen", room_of, items, lambda c, s, kind: seen_i, lambda c, s, kind: seen_j)


def test_moved_is_reported_at_the_new_place_and_only_from_0_45_m():
    """Contract: seen in both visits inside the room, 0.5 m apart: moved, reported at its visit-j place;
    under 0.15 m it is the same place and nothing is reported; in between nothing is reported but both places become
    don't-care regions. Regression caught: a threshold off by a rung, or reporting the old place."""
    far = Item("a", MUG, at(1.0, 1.0), at(1.0, 1.5))
    same = Item("b", MUG, at(2.0, 1.0), at(2.0, 1.1))
    mid = Item("c", MUG, at(3.0, 1.0), at(3.0, 1.3))
    ev, care = events([far, same, mid])
    assert ev == [("moved", (1.0, 1.5))]
    assert care == [(3.0, 1.0), (3.0, 1.3)]


def test_removed_needs_the_old_place_in_view_and_added_needs_the_new_place_in_view_before():
    """Contract: seen only in visit i and then the old place in clear view of visit j: removed at the old place;
    seen only in visit j and the new place in clear view of visit i: added at the new place; without a clear view
    neither is reported. Regression caught: reporting what the robot could not have checked."""
    gone = Item("a", MUG, at(1.0, 1.0), None)
    new = Item("b", MUG, None, at(3.0, 2.0))
    assert events([gone, new]) == ([("added", (3.0, 2.0)), ("removed", (1.0, 1.0))], [])
    assert events([gone, new], seen_i=False, seen_j=False) == ([], [])


def test_leaving_or_entering_the_room_is_removed_or_added_without_a_view_test():
    """Contract: seen in the kitchen in visit i and elsewhere in visit j: removed at the old place; seen elsewhere in
    visit i and in the kitchen in visit j: added at the new place. Regression caught: a mug carried to the next room
    being called moved in one room and missing from the other."""
    leaves = Item("a", MUG, at(1.0, 1.0), at(6.0, 1.0))
    enters = Item("b", MUG, at(6.5, 1.0), at(2.0, 2.0))
    elsewhere = Item("c", MUG, at(5.0, 1.0), at(6.0, 2.0))
    ev, care = events([leaves, enters, elsewhere], seen_i=False, seen_j=False)
    assert ev == [("added", (2.0, 2.0)), ("removed", (1.0, 1.0))] and care == []


def test_the_in_view_tests_of_an_event_get_the_items_kind():
    """Contract: `room_events` asks "was its place in view" with the item's kind, so the cover threshold is the kind's.
    A mug seen only in visit i is checked against visit j with kind "mug", a glasses item seen only in visit j against
    visit i with kind "glasses", and the size and place are the item's. Regression caught: the kind dropped on the way
    (every event then judged by the global bound)."""
    asked = []

    def view(visit):
        return lambda c, s, kind: asked.append((visit, c, s, kind)) or True

    gone = Item("a", MUG, at(1.0, 1.0), None, kind="mug")
    new = Item("b", MUG, None, at(3.0, 2.0), kind="glasses")
    unkinded = Item("c", MUG, at(2.0, 2.0), None)
    room_events("kitchen", room_of, [gone, new, unkinded], view("i"), view("j"))
    assert asked == [("j", at(1.0, 1.0), MUG, "mug"), ("i", at(3.0, 2.0), MUG, "glasses"),
                     ("j", at(2.0, 2.0), MUG, None)]


def test_unsure_items_yield_only_dont_care_regions():
    """Contract: an item with unsure places yields no event, only a don't-care region at each place inside the room
    and at each unsure place. Regression caught: a dead-band glimpse turning into a confident removal."""
    glimpsed = Item("a", MUG, at(1.0, 1.0), None, unsure=((1.0, 1.0),))
    ev, care = events([glimpsed])
    assert ev == [] and sorted(care) == [(1.0, 1.0), (1.0, 1.0)]

