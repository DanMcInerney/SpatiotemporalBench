"""The pointing tool of the reference agent (`llm_locate.py`): a pixel of a frame of the current batch
becomes a map point. Expected values are worked out by hand, from the camera model's
definition (docs/format.md: the pose quaternion maps the body frame x forward, y left, z up to the map; the camera
looks along the body's x, its image x is the body's -y and its image y the body's -z; pixel column j spans u in
[j, j + 1)), never taken from `camera.py`.

The synthetic frames are flat walls at a known distance seen by a camera of 400 px focal length. A pose facing +y has
the quaternion [cos 45, 0, 0, sin 45]; one looking 30 degrees down is a rotation of +30 degrees about the body's y axis,
the quaternion [cos 15, 0, sin 15, 0] (pitch is positive looking up, so down is a positive turn about y).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from stbench.camera import Camera
from stbench.llm_locate import batch_of, locate

CAMERA = Camera.load({"width": 640, "height": 480, "fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0})
CENTRED = Camera.load({"width": 640, "height": 480, "fx": 400.0, "fy": 400.0, "cx": 320.5, "cy": 240.5})
FACING_Y = [0.70710678, 0.0, 0.0, 0.70710678]
LOOKING_DOWN_30 = [0.9659258, 0.0, 0.2588190, 0.0]
POS = [1.0, 2.0, 1.5]


BASE = {"pos": [50.0, 50.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0], "yaw_deg": 0.0}  # nowhere near any camera below


def frame(folder, k: int, millimetres, rot=FACING_Y) -> dict:
    """A step for `batch_of` whose depth image is written under `folder`: a number (a flat wall) or an array. Its
    camera is at `POS` with `rot`; its base, which `locate` must never read, is far away and unrotated."""
    depth = np.full((480, 640), millimetres, dtype=np.uint16) if np.isscalar(millimetres) else millimetres
    path = folder / "inbox" / "depth" / f"{k:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth).save(path)
    return {"k": k, "depth": f"inbox/depth/{k:06d}.png", "pose": BASE, "camera": {"pos": POS, "rot": rot}}


# -- the arithmetic: hand-computed map points ------------------------------------------------------------------------
@pytest.mark.parametrize(("camera", "rot", "wall_mm", "pixel", "expected"), [
    # Facing +y (the body's x is the map's +y, the body's y the map's -x), the wall 3 m away. Pixel (420, 300) has
    # its centre at (420.5, 300.5): image x = 100.5 * 3 / 400 = 0.75375 right, image y = 60.5 * 3 / 400 = 0.45375
    # down. Right of a robot facing +y is +x, so the point is (1 + 0.75375, 2 + 3, 1.5 - 0.45375) = (1.75375, 5,
    # 1.04625).
    (CAMERA, FACING_Y, 3000, (420, 300), "frame k7, pixel (420, 300): map position (1.75, 5.00, 1.05) m, "
                                          "3.00 m from the camera"),
    # (320, 240): centre (320.5, 240.5), 0.5 px off the principal point: 0.00375 right and down: (1.00375, 5, 1.49625).
    (CAMERA, FACING_Y, 3000, (320, 240), "frame k7, pixel (320, 240): map position (1.00, 5.00, 1.50) m, "
                                          "3.00 m from the camera"),
    # The top-left pixel, centre (0.5, 0.5): 319.5 * 3 / 400 = 2.39625 left, 239.5 * 3 / 400 = 1.79625 up:
    # (1 - 2.39625, 5, 1.5 + 1.79625) = (-1.39625, 5, 3.29625).
    (CAMERA, FACING_Y, 3000, (0, 0), "frame k7, pixel (0, 0): map position (-1.40, 5.00, 3.30) m, "
                                      "3.00 m from the camera"),
    # The last pixel, reached by 639.9 and 479.9: centre (639.5, 479.5), 2.39625 right and 1.79625 down:
    # (3.39625, 5, -0.29625).
    (CAMERA, FACING_Y, 3000, (639.9, 479.9), "frame k7, pixel (639.9, 479.9): map position (3.40, 5.00, -0.30) m, "
                                              "3.00 m from the camera"),
    # Looking 30 degrees down from (1, 2, 1.5) with the principal point at a pixel centre, the wall 2 m away: pixel
    # (320, 240) is the optical axis, (cos 30, 0, -sin 30) * 2 = (1.73205, 0, -1) from the camera: (2.73205, 2, 0.5).
    (CENTRED, LOOKING_DOWN_30, 2000, (320, 240), "frame k7, pixel (320, 240): map position (2.73, 2.00, 0.50) m, "
                                                  "2.00 m from the camera"),
    # 40 px to the right is 0.2 m to the right = -y_body, and the map's y is the body's y here: (2.73, 1.8, 0.5).
    (CENTRED, LOOKING_DOWN_30, 2000, (360, 240), "frame k7, pixel (360, 240): map position (2.73, 1.80, 0.50) m, "
                                                  "2.00 m from the camera"),
    # 100 px down is 0.5 m down = -z_body, body (2, 0, -0.5); a turn of 30 degrees about y takes x_body to
    # (0.86603, 0, -0.5) and z_body to (0.5, 0, 0.86603): (1.73205 - 0.25, 0, -1 - 0.43301) = (1.48205, 0, -1.43301),
    # so (2.48205, 2, 0.06699).
    (CENTRED, LOOKING_DOWN_30, 2000, (320, 340), "frame k7, pixel (320, 340): map position (2.48, 2.00, 0.07) m, "
                                                  "2.00 m from the camera"),
])
def test_a_pixel_of_a_flat_wall_gives_the_hand_computed_map_point(tmp_path, camera, rot, wall_mm, pixel, expected):
    """Contract (plan: `locate` back-projects the depth at the pixel with the camera's intrinsics and moves it by the
    frame's pose): for a wall at a known distance, a camera of known intrinsics and a pose facing +y or looking 30
    degrees down, the reply is the point worked out by hand in the comments above, to the centimetre, with the planar
    distance of the wall; the centre of the pixel is used (so (0, 0) is 0.5 px in), the last pixel is (639.9, 479.9),
    and a point's z is its height in the map frame. Regression caught: a sign swapped between the camera's right and the
    body's left, down taken as up, pitch turned the wrong way, the corner of a pixel used for its centre, the depth
    taken as the distance along the ray instead of the camera z, or the pose applied backwards (a camera-to-map matrix
    used as map-to-camera)."""
    batch = batch_of([frame(tmp_path, 7, wall_mm, rot)])
    assert locate({"k": 7, "u": pixel[0], "v": pixel[1]}, batch, tmp_path, camera) == expected


def test_a_centimetre_depth_image_locates_the_point_a_millimetre_one_does(tmp_path):
    """Contract (plan-city §2: a map's depth images are in the unit its `camera.json` names, and `locate` scales the
    depth by it): a wall 3 m away is 3000 in a millimetre image and 300 in a centimetre one, and the camera that goes
    with each gives the same reply, the hand-computed point of the first case above; a wall 80 m away, which a
    millimetre image cannot hold (65.535 m at most), is 8000 in a centimetre one, and the point is (1 + 100.5 x 80 /
    400, 2 + 80, 1.5 - 60.5 x 80 / 400) = (21.1, 82, -10.6). Regression caught: the depth read as millimetres whatever
    the camera says (a downtown frame located ten times too near), or the unit applied twice."""
    mm, cm = CAMERA, replace(CAMERA, depth_unit="cm")
    same = "frame k7, pixel (420, 300): map position (1.75, 5.00, 1.05) m, 3.00 m from the camera"
    at = {"k": 7, "u": 420, "v": 300}
    assert locate(at, batch_of([frame(tmp_path / "mm", 7, 3000)]), tmp_path / "mm", mm) == same
    assert locate(at, batch_of([frame(tmp_path / "cm", 7, 300)]), tmp_path / "cm", cm) == same
    far = batch_of([frame(tmp_path / "far", 7, 8000)])
    assert locate(at, far, tmp_path / "far", cm) == (
        "frame k7, pixel (420, 300): map position (21.10, 82.00, -10.60) m, 80.00 m from the camera")


def test_a_pixel_is_the_cell_it_falls_in_and_zero_depth_is_no_depth(tmp_path):
    """Contract: a depth image holding one pixel's depth (column 420, row 300, 3000 mm) and zeros elsewhere (no depth)
    gives that pixel's point for every (u, v) in its cell, from (420, 300) to (420.99, 300.99), and an error for the
    cell next to it, the sky of a pixel with no depth being an error the model can act on (try a nearby pixel).
    Regression caught: coordinates rounded to the nearest pixel (so 420.99 would read column 421), a zero depth
    back-projected to the camera's own position, or the error left out."""
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[300, 420] = 3000
    batch = batch_of([frame(tmp_path, 7, depth)])
    point = "map position (1.75, 5.00, 1.05) m, 3.00 m from the camera"
    for u, v in ((420, 300), (420.5, 300.5), (420.99, 300.99)):
        assert locate({"k": 7, "u": u, "v": v}, batch, tmp_path, CAMERA).endswith(point)
    for u, v in ((421, 300), (420, 301), (419.99, 300), (100, 100)):
        reply = locate({"k": 7, "u": u, "v": v}, batch, tmp_path, CAMERA)
        assert reply.startswith(f"error: no depth at pixel ({u:g}, {v:g}) of frame k7: ") and "try a pixel" in reply


# -- what it refuses -------------------------------------------------------------------------------------------------
def test_a_frame_outside_the_batch_and_a_pixel_off_the_image_are_errors_the_model_can_read(tmp_path):
    """Contract (plan: errors for a frame not in the current batch and a pixel out of bounds): a keyframe that is not in
    the batch (an earlier one, whose depth is gone, or a later one) is refused naming the frames that can be located; an
    empty batch says there is none; u must be at least 0 and below the width, v the same for the height (640 and 480 are
    outside, 639.99 and 479.99 are not); the replies start with `error:` as the memory tool's do, and none names a host
    path. Regression caught: a lookup of a frame whose file was emptied from the inbox (an OSError text with the host
    path in it), a pixel past the edge wrapping to the next row, or a negative one indexing from the end."""
    steps = [frame(tmp_path, k, 3000) for k in (5, 6, 7)]
    batch = batch_of(steps)
    for k in (4, 8, 0, 99):
        assert locate({"k": k, "u": 10, "v": 10}, batch, tmp_path, CAMERA) == (
            f"error: no frame k{k} in this batch: you can locate in frames k5 to k7 only (the depth of earlier frames "
            "is gone)")
    assert locate({"k": 5, "u": 10, "v": 10}, {}, tmp_path, CAMERA) == (
        "error: no frame k5: there is no batch of frames to locate in")
    assert locate({"k": 7, "u": 10, "v": 10}, batch_of(steps[2:]), tmp_path, CAMERA).startswith("frame k7, pixel")
    assert "in frame k7 only" in locate({"k": 8, "u": 1, "v": 1}, batch_of(steps[2:]), tmp_path, CAMERA)
    off = "is outside the 640 x 480 image: u must be at least 0 and below 640, and v at least 0 and below 480"
    for u, v in ((640, 10), (-1, 10), (10, 480), (10, -0.5), (700, 700)):
        assert locate({"k": 6, "u": u, "v": v}, batch, tmp_path, CAMERA) == f"error: pixel ({u:g}, {v:g}) {off}"
    assert locate({"k": 6, "u": 639.99, "v": 479.99}, batch, tmp_path, CAMERA).startswith("frame k6, pixel (639.99")
    (tmp_path / "inbox" / "depth" / "000006.png").unlink()
    reply = locate({"k": 6, "u": 10, "v": 10}, batch, tmp_path, CAMERA)
    assert reply == "error: the depth image of frame k6 could not be read" and str(tmp_path) not in reply
    Image.fromarray(np.full((240, 320), 3000, dtype=np.uint16)).save(tmp_path / "inbox" / "depth" / "000006.png")
    assert locate({"k": 6, "u": 10, "v": 10}, batch, tmp_path, CAMERA) == (
        "error: the depth image of frame k6 is not 640 x 480")


def test_bad_arguments_are_errors_and_json_text_is_read_as_a_model_sends_it(tmp_path):
    """Contract (the memory tool's argument rules, here for k, u and v): the arguments may be an object or the JSON text
    of one; a missing or unknown argument, a value that is not a number (a string, a bool, null, nan), a k that is
    negative or not whole, and text that is not a JSON object are each an `error:` naming the problem, never an
    exception; a whole float for k (7.0) is accepted. Regression caught: a tool that raises on what a model sends it,
    which would stop a paid run, or accepts `true` as the pixel 1."""
    batch = batch_of([frame(tmp_path, 7, 3000)])
    ok = locate({"k": 7, "u": 420, "v": 300}, batch, tmp_path, CAMERA)
    assert ok.startswith("frame k7, pixel (420, 300): map position (")
    assert locate('{"k": 7, "u": 420, "v": 300}', batch, tmp_path, CAMERA) == ok
    assert locate({"k": 7.0, "u": 420.0, "v": 300.0}, batch, tmp_path, CAMERA) == ok
    cases = [
        ('{"k": 7, "u": 420}', "error: missing required argument: v"),
        ("{}", "error: missing required argument: k"),
        ("", "error: missing required argument: k"),
        (None, "error: missing required argument: k"),
        ({"k": 7, "u": 1, "v": 2, "w": 3}, "error: unexpected argument: w (locate takes k, u, v)"),
        ({"k": 7, "u": "420", "v": 300}, "error: u must be a number"),
        ({"k": 7, "u": True, "v": 300}, "error: u must be a number"),
        ({"k": 7, "u": 1, "v": None}, "error: v must be a number"),
        ({"k": 7, "u": float("nan"), "v": 1}, "error: u must be a number"),
        ({"k": -1, "u": 1, "v": 1}, "error: k must be an integer of at least 0"),
        ({"k": 7.5, "u": 1, "v": 1}, "error: k must be an integer of at least 0"),
        ({"k": "7", "u": 1, "v": 1}, "error: k must be a number"),
        ("[1, 2]", "error: the arguments must be a JSON object"),
    ]
    for arguments, reply in cases:
        assert locate(arguments, batch, tmp_path, CAMERA) == reply, arguments
    assert locate("{not json", batch, tmp_path, CAMERA).startswith("error: the arguments are not valid JSON (")


def test_a_batch_holds_the_frames_with_a_depth_path_and_a_camera_rotation_only():
    """Contract: `batch_of` keeps, by keyframe, the depth path and the camera's pose of the steps that have both a depth
    image and a camera rotation, and leaves the others out (a step with none cannot be located in). Regression caught:
    a batch that holds a frame it has no depth for, which `locate` would then fail on with an exception."""
    steps = [{"k": 1, "depth": "inbox/depth/000001.png", "camera": {"pos": POS, "rot": FACING_Y}},
             {"k": 2, "camera": {"pos": POS, "rot": FACING_Y}},
             {"k": 3, "depth": "inbox/depth/000003.png", "camera": {"pos": POS}},
             {"k": 4, "depth": "inbox/depth/000004.png", "pose": {"pos": POS, "rot": FACING_Y}}]
    assert batch_of(steps) == {1: {"depth": "inbox/depth/000001.png", "pos": POS, "rot": FACING_Y}}


def test_locate_reads_the_camera(tmp_path):
    """Contract (plan §4.9; the stream protocol's step has a `pose`, the base, and a `camera`): `locate` projects from
    the camera block, not the base. The base stands at (1, 2, 0) heading +y; the camera is mounted 0.13 m ahead of it
    and 0.08 m up (the vacuum's mount) and pitched 5 degrees up, so at (1.0, 2.13, 0.08) looking along (0, cos 5, sin 5)
    = (0, 0.99619, 0.08716). The optical axis (pixel (320, 240) of a camera whose principal point is at its centre)
    meets a wall 3 m away at (1, 2.13 + 3 x 0.99619, 0.08 + 3 x 0.08716) = (1.00, 5.12, 0.34); read from the base it
    would be (1.00, 5.00, 0.00). The camera's rotation is yaw 90 then pitch 5 as a quaternion, [cos 45 cos 2.5, sin 45
    sin 2.5, -cos 45 sin 2.5, sin 45 cos 2.5]. Regression caught: points misplaced by the mount (13 cm) and the pitch
    (26 cm at 3 m), as they would be if `batch_of` read `pose`."""
    base = {"pos": [1.0, 2.0, 0.0], "rot": FACING_Y, "yaw_deg": 90.0}
    camera = {"pos": [1.0, 2.13, 0.08], "rot": [0.70643375, 0.03084322, -0.03084322, 0.70643375], "pitch_deg": 5.0}
    step = {**frame(tmp_path, 7, 3000), "pose": base, "camera": camera}
    assert locate({"k": 7, "u": 320, "v": 240}, batch_of([step]), tmp_path, CENTRED) == (
        "frame k7, pixel (320, 240): map position (1.00, 5.12, 0.34) m, 3.00 m from the camera")
