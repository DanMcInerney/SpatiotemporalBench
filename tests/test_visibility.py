"""Visibility: pixel counts, boxes, depth, luminance and centroids computed from synthetic images with known answers.

The images here are hand-built so every expectation is arithmetic done on paper; nothing is computed by the code under
test. The rendered end-to-end checks live in test_render_blender.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from builders import minimal_timeline
from seen_bounds import use
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera
from stbench.render_encode import write_png16

CAM = Camera(64, 48, 50.0, 50.0, 32.0, 24.0)  # a small pinhole: principal point (32, 24), focal length 50 px
POS, ROT = [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0]  # camera at (1, 2, 3) looking along +x, body frame = map frame


def block(index_value: int, rows: slice, cols: slice, depth_mm: int, rgb=(255, 0, 0)):
    """A 48x64 frame holding one rectangle of `index_value`, all at `depth_mm`, in one colour."""
    idx = np.zeros((48, 64), dtype=np.uint16)
    idx[rows, cols] = index_value
    depth = np.where(idx > 0, depth_mm, 0).astype(np.uint16)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[rows, cols] = rgb
    return idx, depth, img


def test_seen_px_is_215_at_640x480_and_scales_with_the_image():
    """The seen threshold is 0.07% of the image: 215 px at 640x480 (V0 section 6). A change to the fraction or the
    rounding moves every question that hinges on the dead band."""
    assert V.seen_px(Camera(640, 480, 381.0, 381.0, 320.0, 240.0)) == 215
    assert V.seen_px(Camera(100, 100, 50.0, 50.0, 50.0, 50.0)) == 7  # 0.0007 * 10000 = 7


@pytest.mark.parametrize("px, expected", [(0, "unseen"), (1, "band"), (214, "band"), (215, "seen"), (216, "seen"),
                                          (5000, "seen")])
def test_status_thresholds_and_the_dead_band(px, expected):
    """0 px is unseen, 1..214 px is the dead band, 215 px and up is seen (at 640x480): a regression moving the edge by
    one pixel changes which questions are skipped."""
    assert V.status(px, Camera(640, 480, 381.0, 381.0, 320.0, 240.0)) == expected


@pytest.mark.parametrize("kind, px, expected", [
    ("mug", 19, "unseen"), ("mug", 20, "band"), ("mug", 199, "band"), ("mug", 200, "seen"), ("mug", 210, "seen"),
    ("mug", 240, "seen"),
    ("glasses", 29, "unseen"), ("glasses", 30, "band"), ("glasses", 300, "band"), ("glasses", 399, "band"),
    ("glasses", 400, "seen"),
    ("bowl", 0, "unseen"), ("bowl", 1, "band"), ("bowl", 214, "band"), ("bowl", 215, "seen"), (None, 1, "band"),
    (None, 215, "seen")])
def test_status_uses_the_kinds_bounds(monkeypatch, kind, px, expected):
    """Contract (V0 section 6, per kind): with mug = (lo 20, hi 200 px) and glasses = (30, 400), the oracle channel's
    status is "unseen" under lo, "band" from lo to hi - 1 and "seen" from hi, for that kind only: a mug at 210 or 240 px
    is seen (the global 215 would call 210 band), glasses at 300 px are band (the global 215 would call them seen), a
    mug at 19 px is unseen (the global rule would call it band). A bowl and no kind keep 0 unseen, 1..214 band, 215
    seen. Regression caught: a caller that drops `kind` and silently uses 215, or an lo that is ignored."""
    use(monkeypatch, mug=(20, 200), glasses=(30, 400))
    assert V.status(px, Camera(640, 480, 381.0, 381.0, 320.0, 240.0), kind) == expected


def test_count_bbox_and_centroid_of_one_entity_by_hand():
    """A 5-row x 4-col block at rows 10..14, cols 20..23 and depth 2.0 m: 20 px, bbox [20, 10, 24, 15] (end exclusive),
    centroid in the camera frame ((22 - 32) * 2 / 50, (12.5 - 24) * 2 / 50, 2) = (-0.4, -0.46, 2), and in the map
    frame (x_body = z_cv, y_body = -x_cv, z_body = -y_cv) = (1 + 2, 2 + 0.4, 3 + 0.46) = (3, 2.4, 3.46). Pixel centres
    are at (j + 0.5, i + 0.5): using corners instead would shift the centroid by half a pixel."""
    idx, depth, rgb = block(7, slice(10, 15), slice(20, 24), 2000)
    (rec,) = V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {7: "E0007"})
    assert rec["id"] == "E0007" and rec["px"] == 20 and rec["bbox"] == [20, 10, 24, 15]
    assert rec["depth_med_m"] == 2.0
    assert rec["centroid_cam"] == pytest.approx([-0.4, -0.46, 2.0], abs=1e-4)
    assert rec["centroid_map"] == pytest.approx([3.0, 2.4, 3.46], abs=1e-4)


def test_depth_is_the_median_not_the_mean():
    """19 px at 2.0 m and one at 40.0 m: the median is 2.0 (a mean would be 3.9). A far pixel at an edge must not drag
    the depth that the compiler reads."""
    idx, depth, rgb = block(3, slice(10, 15), slice(20, 24), 2000)
    depth[10, 20] = 40000
    (rec,) = V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {3: "E0003"})
    assert rec["depth_med_m"] == 2.0


def test_luminance_is_linear_light_with_rec709_weights():
    """A pure-red block has relative luminance 0.2126, pure green 0.7152, white 1.0, black 0; a half-red half-green
    entity averages the two. Averaging the encoded values instead would give 0.33 for mid-grey input."""
    expect = {(255, 0, 0): 0.2126, (0, 255, 0): 0.7152, (255, 255, 255): 1.0, (0, 0, 0): 0.0}
    for colour, lum in expect.items():
        idx, depth, rgb = block(1, slice(10, 14), slice(10, 14), 1000, colour)
        (rec,) = V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {1: "E0001"})
        assert rec["lum"] == pytest.approx(lum, abs=1e-4)
    idx, depth, rgb = block(1, slice(10, 14), slice(10, 14), 1000, (255, 0, 0))
    rgb[10:12, 10:14] = (0, 255, 0)
    (rec,) = V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {1: "E0001"})
    assert rec["lum"] == pytest.approx((0.2126 + 0.7152) / 2, abs=1e-4)


def test_rgb_med_is_the_median_of_each_channel_over_the_entitys_pixels_only():
    """Contract: `rgb_med` is the median of each sRGB channel over the entity's own mask pixels, as whole
    numbers, so a glossy highlight or an edge pixel does not move the colour the picture shows it in, and the pixels
    around it never do. Hand values on a 5 x 4 block of 20 px: 17 of (200, 10, 10) and 3 specular ones of (255, 255,
    255) give [200, 10, 10] (the mean would be [208, 47, 47]); a second entity of 4 px, two of (10, 20, 30) and two of
    (20, 40, 50), has its channels' medians at the midpoints, [15, 30, 40]; a bright frame (240, 240, 240) around
    them changes neither. Regression caught: the mean (a highlight turns a red thing pink), the frame's pixels read
    instead of the mask's, or a median left in floating point (the log holds whole numbers)."""
    idx, depth, rgb = block(7, slice(10, 15), slice(20, 24), 2000, (200, 10, 10))
    rgb[10, 20:23] = (255, 255, 255)
    idx[30:32, 40:42], depth[30:32, 40:42] = 5, 900
    rgb[30, 40:42], rgb[31, 40:42] = (10, 20, 30), (20, 40, 50)
    surround = (idx == 0)
    rgb[surround] = (240, 240, 240)
    recs = {r["id"]: r for r in V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {7: "E0007", 5: "E0005"})}
    assert recs["E0007"]["rgb_med"] == [200, 10, 10] and recs["E0005"]["rgb_med"] == [15, 30, 40]
    assert all(type(x) is int for r in recs.values() for x in r["rgb_med"])


def test_every_entity_with_one_pixel_is_listed_sorted_and_background_is_not():
    """Visibility is unthresholded: a single pixel is listed (the compiler applies thresholds), records are sorted by
    id, and pass_index 0 (structure, sky) is never an entity."""
    idx, depth, rgb = block(9, slice(0, 4), slice(0, 4), 1500)
    idx[20, 30], depth[20, 30] = 2, 800
    idx[30:33, 40:42], depth[30:33, 40:42] = 5, 900
    rows = V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {2: "E0030", 5: "E0010", 9: "E0020"})
    assert [(r["id"], r["px"]) for r in rows] == [("E0010", 6), ("E0020", 16), ("E0030", 1)]
    assert rows[2]["bbox"] == [30, 20, 31, 21]


def test_pixels_without_depth_count_but_do_not_place_the_entity():
    """An entity whose pixels all have depth 0 is still counted and boxed, with null depth and centroids (nothing to
    unproject); mixed pixels place the entity from the pixels that have depth only."""
    idx, depth, rgb = block(4, slice(5, 8), slice(5, 8), 0)
    (rec,) = V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {4: "E0004"})
    assert rec["px"] == 9 and rec["bbox"] == [5, 5, 8, 8]
    assert rec["depth_med_m"] is None and rec["centroid_cam"] is None and rec["centroid_map"] is None
    idx, depth, rgb = block(4, slice(5, 8), slice(5, 8), 3000)
    depth[5, 5] = 0
    (rec,) = V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {4: "E0004"})
    assert rec["px"] == 9 and rec["depth_med_m"] == 3.0


def test_an_index_missing_from_the_instance_map_is_an_error():
    """A pass_index the map does not list means the render and the map disagree; guessing would mislabel entities."""
    idx, depth, rgb = block(8, slice(0, 3), slice(0, 3), 1000)
    with pytest.raises(ValueError, match="pass_index 8"):
        V.frame_entities(idx, depth, rgb, CAM, POS, ROT, {7: "E0007"})


def test_mismatched_image_sizes_are_rejected():
    """Instances, depth and RGB must be the camera's size; a half-size depth image would silently shift every pixel."""
    idx, depth, rgb = block(1, slice(0, 3), slice(0, 3), 1000)
    with pytest.raises(ValueError, match="sizes disagree"):
        V.frame_entities(idx, depth[:24], rgb, CAM, POS, ROT, {1: "E0001"})


def test_build_reads_the_rendered_files_and_writes_the_log(tmp_path):
    """From PNG, JPEG and JSON files on disk to private/visibility.jsonl: one line per keyframe, `k` and its entities,
    with the entity id taken from instance_map.json and the pose of that keyframe (here, a 100x50 block at row 100,
    col 200, 3.0 m ahead of the helper's camera at (3.8, 1.0, 1.3) facing +x on keyframe 0) and the median sRGB of its
    pixels in the JPEG file (a flat grey 200 frame, so [200, 200, 200] through the JPEG). Regression caught: an entity
    named by the wrong pass_index, a keyframe's pose not used (one map point for every keyframe), a colour read from
    the wrong file, or a log row with a key besides `k` and `entities`."""
    tl = minimal_timeline()
    paths = ep.write_script(tmp_path, tl)
    idx = np.zeros((480, 640), dtype=np.uint16)
    idx[100:150, 200:300] = 3
    depth = np.where(idx > 0, 3000, 0).astype(np.uint16)
    for k in range(len(tl.ego.keyframes)):
        write_png16(paths.instances(k), idx)
        write_png16(paths.depth(k), depth)
        Image.fromarray(np.full((480, 640, 3), 200, dtype=np.uint8)).save(paths.rgb(k), quality=95)
    paths.instance_map.write_text(json.dumps({"3": "E0003"}), encoding="utf-8")
    assert V.write(paths) == 6
    rows = V.read(paths.visibility)
    assert [r["k"] for r in rows] == list(range(6)) and all(set(r) == {"k", "entities"} for r in rows)
    rec = rows[0]["entities"][0]
    assert rec["id"] == "E0003" and rec["px"] == 5000 and rec["bbox"] == [200, 100, 300, 150]
    assert rec["rgb_med"] == [200, 200, 200]
    # keyframe 0 faces yaw 0 with pitch -15 degrees, so the camera-frame centroid is as unprojected; its map point
    # is what the pose says, and keyframes with another yaw (k = 1) give another map point for the same pixels.
    assert rec["centroid_cam"][2] == pytest.approx(3.0, abs=1e-3)
    assert rows[1]["entities"][0]["centroid_map"] != rec["centroid_map"]


def test_build_names_the_missing_files(tmp_path):
    """Contract: perceiving before rendering fails with a message that says what to run (`stbench render`), not a
    bare traceback. Regression caught: a missing render file reported as a plain FileNotFoundError that leaves the
    reader to find out what produces it."""
    paths = ep.write_script(tmp_path, minimal_timeline())
    paths.instance_map.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="stbench render"):
        V.build(paths)
