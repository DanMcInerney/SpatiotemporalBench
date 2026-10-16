"""The surface tie (`nearest_part.contact_error`, `contact_pixel_error`, `surface_tie`, `resolvable`,
`fallback_ladder`): a point is asked only where a camera-only reader could place it, by a pixel where the thing meets
the surface it rests on (the contact tie) or by the near edge of a high support (the edge tie, `test_ego_support.py`).
Hand-worked: a pinhole with fy = 400, a camera 0.5 m up at the origin looking along +x unless said, C = 0.1 so C/2 =
0.05. One pixel at a point on the base plane d from the camera's foot, the camera H above that plane, moves it by (d^2 +
H^2) / (H fy): at H = 0.5 and d = 3.0 that is 0.04625, at 3.1 0.0493, at 3.2 0.05245, at 2.2 0.02545."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stbench.nearest_part import (
    CONTACT_M,
    contact_error,
    contact_pixel_error,
    fallback_ladder,
    nearest_part,
    resolvable,
    surface_tie,
    visible_points,
)
from test_nearest_part import CAM, LEVEL, images

FY, C = 400.0, 0.1
CAMERA = (0.0, 0.0, 0.5)


def face(x: float, zs=(0.0, 0.25, 0.5), ys=(-0.1, 0.0, 0.1)) -> np.ndarray:
    """The points of a box's near face at floor position x: a grid over `ys` and `zs`."""
    return np.array([(x, y, z) for y in ys for z in zs])


def read(pts, xy, base=0.0, cam=CAMERA, fy=FY, c_m=C):
    """The contact tie's error of the gold point `xy` from one sighting of `pts` resting on `base`."""
    return contact_error([(cam, np.asarray(pts, dtype=float), base)], xy, fy, c_m)


def test_a_box_on_the_floor_is_asked_through_the_contact_of_its_near_face():
    """Contract (`contact_error`): a thing whose near face meets the floor is asked: its lowest pixels are contact
    pixels within C/2 of the gold point and one pixel reads there to at most C/2. By hand: a face at x = 3.0 from y -0.1
    to 0.1 and z 0 to 0.5 (plus the top's far edge at x = 3.4) has its nearest hull point (3.0, 0.0); the contact pixel
    (3.0, 0.0, 0) is at d = 3.0 and H = 0.5, so e = (9 + 0.25) / (0.5 * 400) = 0.04625 <= 0.05, and the two at y = +-0.1
    are 0.1 from the gold point, past C/2, and are not read. Regression caught: a contact past C/2 of the gold point
    used (the box's far corner would tie the gold to the wrong floor place), or a box on the floor not asked."""
    pts = np.vstack([face(3.0), [[3.4, -0.1, 0.5], [3.4, 0.1, 0.5]]])
    part = nearest_part([(CAMERA, pts)], (0.0, 0.0))
    assert part.xy == (3.0, 0.0) and part.z == 0.0
    e = read(pts, part.xy)
    assert e == pytest.approx(0.04625, abs=1e-9) and resolvable(e, C) and surface_tie(e, False, C) == "contact"
    assert contact_pixel_error(CAMERA, (3.0, 0.1), 0.0, FY) == pytest.approx((9.01 + 0.25) / 200, abs=1e-12)
    assert contact_pixel_error((0.0, 0.0, 0.01), (3.0, 0.0), 0.0, FY) is None   # a camera in the plane reads nothing


def test_a_floor_lamp_whose_shade_overhangs_a_hidden_base_is_not_asked():
    """Contract (`contact_error`, `surface_tie`): a thing whose nearest point overhangs its contact by more than C/2, or
    whose contact is hidden, has no contact pixel within C/2 of the gold point and, with no edge tie, is not asked. By
    hand: a lamp shade 1.2 m up with its near rim at x = 3.0 (points (3.0, 0, 1.2), (3.3, 0, 1.2), (3.15, +-0.15, 1.2)
    and the top at 1.4) and a base pole at x = 3.15 whose floor point (3.15, 0, 0) is 0.15 from the gold point (3.0,
    0.0), past C/2: no contact, so no error and no tie. Regression caught: a point read at its own height (a reader
    without depth does not know it), or a contact taken from anywhere on the thing."""
    shade = [(3.0, 0.0, 1.2), (3.3, 0.0, 1.2), (3.15, 0.15, 1.2), (3.15, -0.15, 1.2), (3.15, 0.0, 1.4)]
    pole = [(3.15, 0.0, 0.0), (3.15, 0.0, 0.5)]
    part = nearest_part([(CAMERA, np.array(shade + pole))], (0.0, 0.0))
    assert part.xy == (3.0, 0.0) and part.z == 1.2
    assert read(shade + pole, part.xy) is None and read(shade, part.xy) is None   # the base hidden altogether
    assert surface_tie(None, False, C) is None


def test_a_contact_beyond_the_range_of_one_pixel_is_not_asked_unless_another_camera_reads_it():
    """Contract (`contact_error`, `resolvable`): the contact pixel must read to at most C/2, from the camera that saw
    it, and the error is the smallest over the sightings, each from its own camera. By hand: a face whose floor point is
    3.1 from the camera reads (9.61 + 0.25) / 200 = 0.0493, asked; at 3.2 it reads (10.24 + 0.25) / 200 = 0.05245, not;
    a second sighting from a camera at (1, 0, 0.5), 2.2 from the floor point, reads (4.84 + 0.25) / 200 = 0.02545, which
    is the error and asks it. Regression caught: the error read from one camera for every sighting, the largest taken
    in place of the smallest, or a range one pixel cannot read asked."""
    near, far = face(3.1), face(3.2)
    assert read(near, (3.1, 0.0)) == pytest.approx(0.0493, abs=1e-9) and resolvable(read(near, (3.1, 0.0)), C)
    one = read(far, (3.2, 0.0))
    assert one == pytest.approx(0.05245, abs=1e-9) and not resolvable(one, C)
    two = contact_error([(CAMERA, far, 0.0), ((1.0, 0.0, 0.5), far, 0.0)], (3.2, 0.0), FY, C)
    assert two == pytest.approx(0.02545, abs=1e-9) and resolvable(two, C)


@pytest.mark.parametrize("offset, counts", [(-0.01, True), (0.0, True), (CONTACT_M, True), (CONTACT_M + 0.0002, False),
                                            (0.03, False)])
def test_a_pixel_is_a_contact_within_two_centimetres_of_the_base_at_close_range(offset, counts):
    """Contract (`CONTACT_M`): at close range a visible point at most 2 cm from the base z (above it, or below it by a
    rounding of the depth) is a contact pixel; one further is not. By hand, a support top at z 0.3 and a lone face
    point at (3.0, 0, 0.3 + offset), the camera 0.5 up (H = 0.2, one pixel 3.0 / 400 = 0.0075 m, under the 2 cm): it
    reads (9 + 0.04) / (0.2 * 400) = 0.113 when it counts and nothing when it does not. Regression caught: a tolerance
    that ignores a point on the surface or admits a point that is off it."""
    e = read([(3.0, 0.0, 0.3 + offset)], (3.0, 0.0), base=0.3, c_m=0.5)
    assert (e is not None) is counts
    if counts:
        assert e == pytest.approx(0.113, abs=1e-9)


def test_a_thing_on_a_slab_at_drone_range_is_asked_through_a_contact_the_size_of_one_pixel():
    """Contract (`contact_error`, the pixel tolerance): at long range a pixel is taller than 2 cm, so a contact pixel is
    within max(2 cm, slant / fy) of the base. By hand (fy = 400, C = 2.4, the camera 15 m up at the origin): a car on a
    0.15 m slab whose face is at x = 24 has its lowest visible centre at z 0.20; its slant is sqrt(24^2 + 14.8^2) =
    28.197, one pixel 28.197 / 400 = 0.0705 m, so 0.05 above the base is a contact; H = 15 - 0.15 = 14.85 and e = (576
    + 14.85^2) / (14.85 * 400) = 796.5225 / 5940 = 0.1341, well under C/2 = 1.2. A centre at z 0.25 (0.10 above) is
    past that pixel and is none. Regression caught: a fixed 2 cm tolerance (every drone thing on a slab dropped), or one
    that grows without the slant."""
    car = [(24.0, 0.0, 0.20), (24.0, 0.0, 1.0)]
    drone = (0.0, 0.0, 15.0)
    e = read(car, (24.0, 0.0), base=0.15, cam=drone, c_m=2.4)
    assert e == pytest.approx(0.134095, abs=1e-6) and resolvable(e, 2.4)
    assert read([(24.0, 0.0, 0.25), (24.0, 0.0, 1.0)], (24.0, 0.0), base=0.15, cam=drone, c_m=2.4) is None


def test_a_mug_on_a_low_table_seen_from_above_is_asked_through_the_table_top():
    """Contract (`contact_error`, the base is the support's top): the contact is where the thing meets the surface it
    rests on, which is the table top, not the floor. By hand (fy = 400, C = 0.1, the camera 1.3 m up): a mug whose near
    face is at x = 2.0 on a table top at 0.42 has its contact pixel (2.0, 0, 0.42), H = 0.88, d = 2.0, so e = (4 +
    0.7744) / (0.88 * 400) = 0.01356 <= 0.05; read at the floor it has no pixel. A camera 1 cm above the top reads
    nothing. Regression caught: the floor assumed for every thing, so every raised thing is dropped, or a camera in the
    surface's plane read at an unbounded error."""
    mug = [(2.0, 0.0, 0.42), (2.0, 0.0, 0.5), (2.0, 0.0, 0.6)]
    eye = (0.0, 0.0, 1.3)
    e = read(mug, (2.0, 0.0), base=0.42, cam=eye)
    assert e == pytest.approx(0.0135636, abs=1e-7) and surface_tie(e, False, C) == "contact"
    assert read(mug, (2.0, 0.0), base=0.0, cam=eye) is None
    assert read(mug, (2.0, 0.0), base=0.42, cam=(0.0, 0.0, 0.43)) is None


def test_a_thing_hanging_on_a_face_has_no_base_and_so_no_contact():
    """Contract (`contact_error`, a base of None): a thing with no base to rest on (a coat hanging on a wardrobe's side)
    has no contact tie even though its lowest pixels are at the z its placement says. By hand: the face at 3.0 above
    reads 0.04625 with a base of 0.0, and nothing with a base of None. Regression caught: a hanging coat tied by its
    hem."""
    pts = face(3.0)
    assert read(pts, (3.0, 0.0), base=0.0) == pytest.approx(0.04625, abs=1e-9)
    assert contact_error([(CAMERA, pts, None)], (3.0, 0.0), FY, C) is None


def test_the_tie_is_the_contacts_first_then_the_edge_and_the_fallback_takes_the_contacts_error():
    """Contract (`surface_tie`, `fallback_ladder`): a contact within C/2 asks the question, else the edge tie does, else
    it is dropped; a dropped question the job keeps has the rungs max(C m, 2 m e) for m in 0.5, 1 and 2 with e the
    contact's error, so the full rung is what one pixel could miss, and none when there is no contact. By hand (C =
    0.1): the face at 3.2 has e = 0.05245, so the rungs are max(0.05, 0.05245) = 0.05245, max(0.1, 0.1049) = 0.1049 and
    max(0.2, 0.2098) = 0.2098; with the edge tie it is asked ("edge"); the lamp above has no contact and no fallback.
    Regression caught: the fallback keeping a question on rungs read from the point's own height, or an unreadable
    contact beating an edge that holds."""
    e = read(face(3.2), (3.2, 0.0))
    assert surface_tie(e, False, C) is None and surface_tie(e, True, C) == "edge"
    assert surface_tie(0.04, True, C) == "contact" and surface_tie(None, True, C) == "edge"
    assert fallback_ladder(e, C) == pytest.approx([0.05245, 0.1049, 0.2098], abs=1e-9)
    assert fallback_ladder(None, C) is None


def reader(index, depth, pass_index, pos, gold, c_m, base=0.0):
    """The independent reader: from the pinhole formula alone, the smallest error at the base plane over the thing's
    pixels that sit within max(2 cm, one pixel) of it and within C/2 of `gold` (None when there is none)."""
    best = None
    for i, j in zip(*np.nonzero((index == pass_index) & (depth > 0)), strict=True):
        d = depth[i, j] / 1000.0
        x, y, z = pos[0] + d, pos[1] - (j + 0.5 - 320) * d / 400, pos[2] - (i + 0.5 - 240) * d / 400
        slant = math.dist((x, y, z), pos)
        if abs(z - base) <= max(0.02, slant / 400) and math.hypot(x - gold[0], y - gold[1]) <= c_m / 2:
            h = abs(pos[2] - base)
            e = (math.hypot(x - pos[0], y - pos[1]) ** 2 + h * h) / (h * 400.0)
            best = e if best is None else min(best, e)
    return best


def test_every_asked_gold_passes_reading_only_the_camera_the_pose_the_depth_and_the_mask():
    """Contract (VISION rule 5, the gold is what a perfect reading of the delivered frames gives): a reader that sees
    only the camera, the pose, the depth and the instance mask finds the contact pixel that makes a gold askable, and
    finds none where the gold is not asked. By hand, a camera at (0, 0, 0.5) looking along +x, fy = 400, a pixel 0.0075
    m tall at 3.0 m: a box face (instance 7, 3000 mm, rows 200-306, columns 300-339, the lowest centre 0.00125 m up) has
    its gold point (3.0, 0), and the contact pixels within 0.05 of it are the rows 304-306 of columns 314-325; the
    nearest to the camera's foot is at y = 0.00375, e = (9.0000141 + 0.25) / 200 = 0.04625. A lamp (instance 5: a shade
    at 3000 mm, rows 147-173, columns 300-339, and a pole at 3150 mm, rows 174-306, columns 318-321) whose base an
    occluder (instance 9, 2500 mm, rows 231-306) hides has no contact pixel in view, and the reader finds none.
    Regression caught: a gold the pictures do not tie to a surface asked, or one they do tie not asked."""
    box = images({7: (slice(200, 307), slice(300, 340), 3000)})
    lamp = images({5: (slice(147, 174), slice(300, 340), 3000)})
    for blocks in ((slice(174, 307), slice(318, 322), 3150, 5), (slice(231, 307), slice(280, 360), 2500, 9)):
        rows, cols, mm, pass_index = blocks   # the pole, then the occluder in front of its foot
        lamp[0][rows, cols], lamp[1][rows, cols] = pass_index, mm
    for name, (idx, dep), pass_index, want in (("box", box, 7, 0.04625), ("lamp", lamp, 5, None)):
        points = visible_points(idx, dep, CAM, CAMERA, LEVEL, pass_index)
        part = nearest_part([(CAMERA, points)], (0.0, 0.0))
        assert part.xy == pytest.approx((3.0, 0.0), abs=1e-9), name
        found, e = reader(idx, dep, pass_index, CAMERA, part.xy, C), read(points, part.xy)
        if want is None:
            assert found is None and e is None and surface_tie(e, False, C) is None, name
        else:
            assert found == pytest.approx(want, abs=1e-6) and e == pytest.approx(found, abs=1e-12), name
            assert surface_tie(e, False, C) == "contact", name
