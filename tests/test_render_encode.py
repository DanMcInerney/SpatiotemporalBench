"""Encoding Blender's raw float passes into the dataset's 16-bit PNGs, and keeping metadata out of images."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from builders import minimal_timeline
from conftest import script_on_map
from render_kit import synthetic_passes, vacuum_script, write_passes
from stbench import episode as ep
from stbench import render_cli as R
from stbench.render_encode import (
    SELF_ID,
    cut_depth,
    depth_to_mm,
    depth_to_units,
    index_to_u16,
    read_png16,
    rgb_clip_m,
    self_filter,
    strip_jpeg,
    write_png16,
)
from stbench.robots import ROBOTS


def test_depth_metres_become_rounded_millimetres_and_nothing_else_becomes_zero():
    """Contract (the apartment's and warehouse's depth images, kept byte for byte): 1.2344 m -> 1234 mm and 1.2346 m ->
    1235 mm (round to nearest); 65.535 m is the last representable depth and anything beyond, sub-millimetre, negative,
    infinite, NaN or Cycles' background value (1e10) is 0 = no depth. `depth_to_mm` and `depth_to_units(.., "mm")` are
    the one encoding. Truncating instead of rounding would bias every depth 0.5 mm short; wrapping past 65535 would
    invent near surfaces; a unit-aware encoder that changed the millimetre case would change every kept depth image."""
    m = np.array([[1.2344, 1.2346, 65.535, 65.5361], [0.0004, -2.0, np.inf, np.nan], [1e10, 0.001, 0.0, 5.0]])
    for got in (depth_to_mm(m), depth_to_units(m, "mm")):
        assert got.dtype == np.uint16
        assert got.tolist() == [[1234, 1235, 65535, 0], [0, 0, 0, 0], [0, 1, 0, 5000]]


def test_millimetres_are_byte_for_byte_what_the_kept_depth_images_were_encoded_with():
    """Contract (plan-city §2): `depth_to_units(.., "mm")` gives exactly the arrays the encoder gave the 150 kept
    episodes before units existed. The reference is that encoder's formula, copied here (round to nearest, 1..65535
    kept, the rest 0), over 10,000 random depths of 0-70 m and of 0-2 m and the values that mean "no depth".
    Regression caught: a unit scale computed as 1 / 0.001 (float noise that moves a rounding), or a range test moved by
    one step."""
    def kept(depth_m):
        d = np.asarray(depth_m, dtype=np.float64)
        ok = np.isfinite(d) & (d > 0)
        mm = np.zeros(d.shape, dtype=np.uint16)
        scaled = np.rint(np.where(ok, d, 0.0) * 1000.0)
        keep = ok & (scaled >= 1) & (scaled <= 65535)
        mm[keep] = scaled[keep].astype(np.uint16)
        return mm

    rng = np.random.default_rng(7)
    m = np.concatenate([rng.uniform(0, 70, 5000), rng.uniform(0, 2, 5000), [np.inf, np.nan, 1e10, -1.0, 0.0, 65.535]])
    assert np.array_equal(depth_to_units(m, "mm"), kept(m)) and np.array_equal(depth_to_mm(m), kept(m))


def test_depth_metres_become_rounded_centimetres_to_655_35_m_and_nothing_else_becomes_zero():
    """Contract (plan-city §2, §5.4: the downtown map's depth images are in centimetres, which reach 655.35 m): 1.234 m
    -> 123 cm and 1.236 m -> 124 cm (round to nearest); 655.35 m is the last representable depth, 655.36 m and
    anything beyond, under half a centimetre, negative, infinite, NaN or Cycles' background value (1e10) is 0 = no
    depth, and 1 cm is the nearest. Regression caught: the millimetre scale left in the centimetre case (1.234 m stored
    as 1234, read back as 12.34 m), truncation instead of rounding, or a depth past the range wrapping into a near
    surface."""
    m = np.array([[1.234, 1.236, 655.35, 655.36], [0.004, -2.0, np.inf, np.nan], [1e10, 0.01, 0.0, 65.0]])
    got = depth_to_units(m, "cm")
    assert got.dtype == np.uint16
    assert got.tolist() == [[123, 124, 65535, 0], [0, 0, 0, 0], [0, 1, 0, 6500]]


def test_depth_metres_in_metres_and_an_unknown_unit_is_refused():
    """Contract (`camera.DEPTH_UNITS`: a map may name mm, cm or m): in metres 12.4 m is 12 and 65535.4 m is the last
    depth; a unit the table does not hold is a ValueError naming the known ones, not a silent millimetre encoding.
    Regression caught: an unknown unit encoded as some other (a map's `depth_unit` typo giving depth in the wrong
    scale for a whole render)."""
    assert depth_to_units(np.array([12.4, 65535.4, 65535.6, 0.4]), "m").tolist() == [12, 65535, 0, 0]
    with pytest.raises(ValueError, match="unknown depth unit 'dm'; known: mm, cm, m"):
        depth_to_units(np.ones((2, 2)), "dm")


@pytest.mark.parametrize(("unit", "expected"), [("mm", [[0, 2500], [2500, 2500]]), ("cm", [[7000, 250], [250, 250]]),
                                                 (None, [[0, 2500], [2500, 2500]])])
def test_encode_frames_writes_depth_in_the_unit_the_episodes_camera_file_names(tmp_path, unit, expected):
    """Contract (plan-city §2: `encode_frames` reads the unit from `sensors/camera.json`, absent meaning mm): the same
    raw depth pass (2.5 m, and 70 m at one pixel) is stored as 2500 and 0 (70 m is past the helper's 10 m range) under
    an episode whose camera file says "mm" or has no `depth_unit` (one made before the key existed), and as 250 and 7000
    under one that says "cm" (the drone sees to 600 m); the instance pass is the same either way. Regression caught: the
    encoder writing millimetres for a centimetre map (every depth past 65 m lost, the rest ten times too large to
    read)."""
    raw = np.full((4, 4), 2.5)
    raw[0, 0] = 70.0
    tmp = tmp_path / "raw"
    write_passes(tmp, 0, raw, np.full((4, 4), 3.0), np.full((480, 640, 3), 0.1))
    tl = script_on_map("downtown_a", 1, "drone", [(40.0, 13.5, 12.0, 0.0)]) if unit == "cm" else minimal_timeline()
    paths = ep.write_script(tmp_path / "ds", tl)  # the camera file is the robot's: the helper's says mm, the drone's cm
    if unit is None:
        camera = json.loads(paths.camera.read_text(encoding="utf-8"))
        paths.camera.write_text(json.dumps({k: v for k, v in camera.items() if k != "depth_unit"}), encoding="utf-8")
    R.encode_frames(paths, tmp, (0, 1))
    assert read_png16(paths.depth(0))[:2, :2].tolist() == expected
    assert read_png16(paths.instances(0)).tolist() == [[3] * 4] * 4


def test_depth_is_cut_and_self_filtered(tmp_path):
    """Contract (`cut_depth`, `self_filter`, `encode_frames`): on a synthetic frame under the vacuum's camera (depth
    from 0.01 m to 4.0 m), depth nearer than near_m or farther than far_m is 0 and depth at either bound is kept (0.009,
    0.01, 2.0, 4.0 and 4.001 m read 0, 10, 2000, 4000 and 0 mm); pixels whose index is `SELF_ID` get depth 0 and index 0
    while the others keep theirs; and the JPEG is the same whatever the passes hold (the RGB still shows the robot).
    Regression caught: the robot's own body certified as an obstacle, depth the robot cannot have, or a truth pass
    reaching the picture."""
    assert cut_depth(np.array([0.009, 0.01, 2.0, 4.0, 4.001]), 0.01, 4.0).tolist() == [0.0, 0.01, 2.0, 4.0, 0.0]
    depth, index = self_filter(np.array([[5, 6], [7, 8]], dtype=np.uint16), np.array([[1, SELF_ID], [SELF_ID, 3]],
                                                                                     dtype=np.uint16))
    assert depth.tolist() == [[5, 0], [0, 8]] and index.tolist() == [[1, 0], [0, 3]]
    paths = ep.write_script(tmp_path / "ds", vacuum_script(1))
    linear = np.full((480, 640, 3), 0.1)
    metres = np.array([[0.009, 0.01, 2.0, 4.0], [4.001, 3.0, 3.0, 3.0]])
    index = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, SELF_ID, 1.0, 1.0]])
    for name, passes in (("a", (metres, index)), ("b", (np.full((2, 4), 2.0), np.full((2, 4), 5.0)))):
        write_passes(tmp_path / name, 0, *passes, linear)
        R.encode_frames(paths, tmp_path / name, (0, 1))
        if name == "a":
            assert read_png16(paths.depth(0)).tolist() == [[0, 10, 2000, 4000], [0, 0, 3000, 3000]]
            assert read_png16(paths.instances(0)).tolist() == [[1, 1, 1, 1], [1, 0, 1, 1]]
            jpeg = paths.rgb(0).read_bytes()
    assert paths.rgb(0).read_bytes() == jpeg


def test_truth_passes_ignore_the_camera_model(tmp_path):
    """Contract (D9: noise, vignetting and white balance are in the RGB only): `encode_frames` over the same synthetic
    passes with two `IspSpec`s (the robot's, and one with no vignette, no noise and no sharpening) writes depth and
    instance PNGs that are identical byte for byte and JPEGs that differ (the test is not vacuous). Regression caught: a
    truth pass changed by a look."""
    paths = ep.write_script(tmp_path / "ds", vacuum_script(1))
    cam = ROBOTS["vacuum"].camera
    write_passes(tmp_path / "raw", 0, *synthetic_passes(cam.width, cam.height))
    plain = replace(cam.isp, vignette=0.0, noise_a=0.0, noise_b=0.0, sharpen=0.0)
    out = []
    for isp in (None, plain):
        R.encode_frames(paths, tmp_path / "raw", (0, 1), isp=isp)
        out.append([p.read_bytes() for p in (paths.depth(0), paths.instances(0), paths.rgb(0))])
    assert out[0][:2] == out[1][:2] and out[0][2] != out[1][2]


def test_frames_are_developed_at_the_size_the_camera_file_says(tmp_path):
    """Contract (`encode_frames`): Blender renders at the size of the script's own `sensors/camera.json`, so the JPEG is
    developed at that size, from the robot's camera model: a script whose camera file says 320 x 240 (a test or a
    look-dev render that is small) gives 320 x 240 JPEGs and depth, not an error about the robot's 640 x 480. Regression
    caught: the camera model refusing a render that is smaller than the robot's own sensor."""
    paths = ep.write_script(tmp_path / "ds", vacuum_script(1))
    camera = json.loads(paths.camera.read_text(encoding="utf-8"))
    paths.camera.write_text(json.dumps({**camera, "width": 320, "height": 240}), encoding="utf-8")
    write_passes(tmp_path / "raw", 0, *synthetic_passes(320, 240))
    (log,) = R.encode_frames(paths, tmp_path / "raw", (0, 1))
    with Image.open(paths.rgb(0)) as im:
        assert im.size == (320, 240)
    assert read_png16(paths.depth(0)).shape == (240, 320) and log["k"] == 0


def test_the_rgb_clip_is_nearer_than_every_depth_range():
    """Contract (`rgb_clip_m`): for every robot the renderer's near clip is above 0 and at most the nearest depth the
    robot reports, so no surface depth reads is cut out of the picture (and the vacuum's, 2 cm under a shelf, is a
    5 mm clip). Regression caught: an RGB clip that cuts away what depth reports (review round 3: today's 0.05 m clip
    showed the inside of a shelf from 2 cm)."""
    for robot in ROBOTS.values():
        assert 0.0 < rgb_clip_m(robot.depth.near_m) <= robot.depth.near_m, robot.id
    assert rgb_clip_m(ROBOTS["vacuum"].depth.near_m) == 0.005 and rgb_clip_m(0.2) == 0.005


def test_index_pass_converts_to_exact_pass_indices():
    """Integral floats map to the same integers (0 stays 0, 65535 survives); the conversion must be exact because the
    index is an identity, not a measurement."""
    got = index_to_u16(np.array([[0.0, 1.0, 3.0000002], [65535.0, 41.0, 7.0]], dtype=np.float32))
    assert got.dtype == np.uint16 and got.tolist() == [[0, 1, 3], [65535, 41, 7]]


@pytest.mark.parametrize("bad", [2.5, -1.0, np.nan, np.inf, 70000.0])
def test_index_pass_rejects_values_that_are_not_ids(bad):
    """A blended edge (2.5), a negative, a non-finite or an out-of-range value means the renderer, not the data, is
    wrong: failing loudly beats rounding it into some other entity's id."""
    with pytest.raises(ValueError, match="not integers"):
        index_to_u16(np.array([[0.0, bad]]))


def test_png16_round_trip_is_exact_and_carries_no_chunks_but_the_image(tmp_path):
    """Every uint16 value (0, 1, 255, 256, 40000, 65535 included) survives a write and a read, and the file holds only
    IHDR, IDAT and IEND: no text chunks, timestamps or software tags that could carry metadata."""
    arr = np.array([[0, 1, 255, 256], [40000, 65535, 12345, 7]], dtype=np.uint16)
    path = tmp_path / "x.png"
    write_png16(path, arr)
    back = read_png16(path)
    assert back.dtype == np.uint16 and np.array_equal(back, arr)
    data, i, kinds = path.read_bytes(), 8, []
    while i < len(data):
        n = int.from_bytes(data[i:i + 4], "big")
        kinds.append(data[i + 4:i + 8])
        i += 12 + n
    assert set(kinds) == {b"IHDR", b"IDAT", b"IEND"}


def test_png16_rejects_the_wrong_dtype(tmp_path):
    """An 8-bit or float array written as a "16-bit" PNG would silently lose depth precision."""
    with pytest.raises(ValueError, match="uint16"):
        write_png16(tmp_path / "x.png", np.zeros((2, 2), dtype=np.uint8))


def test_strip_jpeg_removes_comments_and_app_segments_without_touching_pixels(tmp_path):
    """Blender stamps JPEGs with comment segments (render times) and an application segment. After stripping, no COM or
    APP1..APP15 segment remains, the JFIF header stays, and the decoded pixels are bit-identical."""
    buf = io.BytesIO()
    img = Image.fromarray((np.arange(64 * 64 * 3) % 251).astype(np.uint8).reshape(64, 64, 3))
    img.save(buf, format="JPEG", quality=92, comment=b"Blender:cycles.total_time:00:01.5", exif=b"Exif\x00\x00II*\x00")
    path = tmp_path / "a.jpg"
    path.write_bytes(buf.getvalue())
    before = np.asarray(Image.open(path)).copy()
    assert b"Blender:cycles" in path.read_bytes() and b"Exif" in path.read_bytes()
    strip_jpeg(path)
    data = path.read_bytes()
    markers, i = [], 2
    while data[i] == 0xFF and data[i + 1] != 0xDA:
        markers.append(data[i + 1])
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    assert 0xFE not in markers and not any(0xE1 <= m <= 0xEF for m in markers) and 0xE0 in markers
    assert b"Blender" not in data and b"Exif" not in data
    assert np.array_equal(np.asarray(Image.open(path)), before)
    strip_jpeg(path)  # idempotent
    assert path.read_bytes() == data
