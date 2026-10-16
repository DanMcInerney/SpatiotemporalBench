"""The nearest visible part (`nearest_part`, `pixels`): hand-made masks and depth, worked out pixel by pixel.

The camera is a hand-made pinhole: 640 x 480, fx = fy = 400, principal point (320, 240), depth in millimetres. Pixel
(row i, column j) at planar depth d has the camera-frame point ((j + 0.5 - 320) d / 400, (i + 0.5 - 240) d / 400, d),
which a camera at `pos` looking along +x (identity rotation) puts in the map at (pos_x + d, pos_y - x_cv, pos_z - y_cv).
The expected values below use that formula and a brute-force reader written here, never the code under test."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from stbench import pixels
from stbench.camera import Camera
from stbench.case import ScriptPaths
from stbench.nearest_part import (
    floor_hull,
    nearest_on_hull,
    nearest_part,
    nearest_point,
    union_hull,
    visible_points,
)
from stbench.render_encode import write_png16

CAM = Camera.load({"width": 640, "height": 480, "fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0,
                   "depth_unit": "mm"})
LEVEL = [1.0, 0.0, 0.0, 0.0]                              # looking along +x
DOWN = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]   # looking straight down


def images(blocks: dict[int, tuple[slice, slice, object]]):
    """An instance image and a depth image (mm): each block is (rows, cols, depth in mm or an array of it)."""
    index, depth = np.zeros((480, 640), dtype=np.uint16), np.zeros((480, 640), dtype=np.uint16)
    for pass_index, (rows, cols, mm) in blocks.items():
        index[rows, cols] = pass_index
        depth[rows, cols] = mm
    return index, depth


def read_level(index, depth, pass_index, pos=(0.0, 0.0, 1.0)):
    """The independent reader: every pixel of the instance with a depth, by the pinhole formula above, camera level."""
    out = []
    for i, j in zip(*np.nonzero((index == pass_index) & (depth > 0)), strict=True):
        d = depth[i, j] / 1000.0
        out.append((pos[0] + d, pos[1] - (j + 0.5 - 320) * d / 400, pos[2] - (i + 0.5 - 240) * d / 400))
    return out


def brute_nearest(points_xy, q):
    """The point of the convex hull nearest q (q outside it) by brute force: the nearest point of any segment between
    two of the points, which holds every edge of the hull and nothing outside it."""
    best = None
    for a, b in itertools.combinations(sorted(set(points_xy)), 2):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy)))
        p = (a[0] + t * dx, a[1] + t * dy)
        if best is None or math.dist(p, q) < math.dist(best, q) - 1e-12:
            best = p
    return best


def test_a_things_points_are_its_own_pixels_with_a_depth_at_the_pixel_centres():
    """Contract (`visible_points`): the points are the pixels of the thing's instance that have a depth, unprojected at
    the pixel centres through the planar depth and moved by the camera's pose; a pixel of another instance or of depth
    0 adds none. By hand: a 2 x 2 block at rows 240-241, columns 320-321 at 2000 mm from a camera at (0, 0, 1.0): the
    centres are 0.5 and 1.5 pixels off the principal point, so x_cv = y_cv = 0.0025 and 0.0075 (0.5 * 2 / 400 and
    1.5 * 2 / 400); the points are (2, -x_cv, 1 - y_cv). A pixel of depth 0 inside the block, and a block of another
    instance beside it, add nothing. Regression caught: the pixel corner for its centre (a half-pixel shift), a point
    for a pixel without a depth (a 0 m point at the camera), or another thing's pixels."""
    index, depth = images({7: (slice(240, 242), slice(320, 322), 2000), 9: (slice(240, 242), slice(322, 324), 1500)})
    depth[241, 321] = 0
    got = sorted(map(tuple, visible_points(index, depth, CAM, (0.0, 0.0, 1.0), LEVEL, 7).round(9).tolist()))
    assert got == sorted([(2.0, -0.0025, 0.9975), (2.0, -0.0075, 0.9975), (2.0, -0.0025, 0.9925)])
    assert visible_points(index, depth, CAM, (0.0, 0.0, 1.0), LEVEL, 5).shape == (0, 3)
    with pytest.raises(ValueError, match="differ"):
        visible_points(index[:10], depth, CAM, (0.0, 0.0, 1.0), LEVEL, 7)


def test_a_pixel_perfect_reader_reproduces_the_gold_exactly():
    """Contract (FD-0.7.0 review, 3c): the gold is exactly what a reader of the same instance image and depth gives.
    By hand: a 6 x 8 block (rows 250-255, columns 300-307) whose depth grows 20 mm a column and 10 mm a row from
    1500 mm, seen from a camera at (0, 0, 1) looking along +x, read by the brute-force reader above (every pixel by the
    pinhole formula, the points to the millimetre, the nearest point of the hull of any two of them). From the camera's
    floor point (0, 0) and from (1.8, 0.1), a point beside the thing's face, `nearest_part` gives that point to a
    nanometre, and the height of the visible point nearest it. Regression caught: a nearest taken over
    the pixels' points in place of their hull (a point of the face between two pixels is nearer than either), a hull
    built from unrounded points, or a pose applied twice."""
    rows, cols = np.arange(6)[:, None], np.arange(8)[None, :]
    index, depth = images({4: (slice(250, 256), slice(300, 308), 1500 + 10 * rows + 20 * cols)})
    pts = read_level(index, depth, 4)
    mine = visible_points(index, depth, CAM, (0.0, 0.0, 1.0), LEVEL, 4)
    for ground in ((0.0, 0.0), (1.8, 0.1)):
        want = brute_nearest({(round(x, 3), round(y, 3)) for x, y, _ in pts}, ground)
        part = nearest_part([((0.0, 0.0, 1.0), mine)], ground)
        assert part.xy == pytest.approx(want, abs=1e-9)
        near = min(pts, key=lambda p: math.dist(p[:2], want))
        assert part.z == pytest.approx(near[2], abs=1e-9)


def test_the_union_of_two_sightings_has_the_nearest_point_of_either():
    """Contract (FD-0.7.0 review, change 7): `ego_mem`'s gold takes the hull of the union of the sightings, so a point
    of the thing that one camera showed counts after another has moved. By hand: the block of the test above seen from
    (0, 0, 1) and again from a camera at (0.5, 0.4, 1.4) looking the same way (the points shift by (0.5, 0.4, 0.4)); the
    brute-force reader over both sets of points gives the union's nearest point to the floor point (3.0, 0.0), which is
    not the first sighting's alone. Regression caught: the last sighting alone, or the first alone."""
    rows, cols = np.arange(6)[:, None], np.arange(8)[None, :]
    index, depth = images({4: (slice(250, 256), slice(300, 308), 1500 + 10 * rows + 20 * cols)})
    first, second = (0.0, 0.0, 1.0), (0.5, 0.4, 1.4)
    pts = read_level(index, depth, 4) + read_level(index, depth, 4, second)
    want = brute_nearest({(round(x, 3), round(y, 3)) for x, y, _ in pts}, (3.0, 0.0))
    seen = [(first, visible_points(index, depth, CAM, first, LEVEL, 4)),
            (second, visible_points(index, depth, CAM, second, LEVEL, 4))]
    part = nearest_part(seen, (3.0, 0.0))
    assert part.xy == pytest.approx(want, abs=1e-9)
    assert nearest_part(seen[:1], (3.0, 0.0)).xy != pytest.approx(part.xy, abs=1e-3)


def test_a_thing_with_no_pixel_or_no_depth_has_no_part_and_one_cut_by_the_edge_keeps_its_visible_part():
    """Contract (FD-0.7.0 review, 3e): a thing with no pixel in the frame, or none that has a depth, has no part
    (None, so the question is not asked); a thing cut by the image's edge still has its visible part. By hand: instance
    3 is absent, instance 6's block has depth 0 everywhere; instance 8 fills columns 0-3 of rows 100-103 (16 pixels at
    1000 mm), cut by the left edge, and all 16 are points: the leftmost column's centre is 0.5 - 320 = -319.5 pixels
    off the axis, 1000 * -319.5 / 400 mm = -0.79875 m, so its map y is +0.79875. Regression caught: a thing at the edge
    dropped (a visible part is a part), a depth-less pixel read as a point at the camera, or an empty thing answered."""
    index, depth = images({6: (slice(10, 20), slice(10, 20), 0), 8: (slice(100, 104), slice(0, 4), 1000)})
    for absent in (3, 6):
        none = visible_points(index, depth, CAM, (0, 0, 1), LEVEL, absent)
        assert none.shape == (0, 3) and nearest_part([((0.0, 0.0, 1.0), none)], (0, 0)) is None
    edge = visible_points(index, depth, CAM, (0.0, 0.0, 1.0), LEVEL, 8)
    assert len(edge) == 16 and edge[:, 1].max() == pytest.approx(0.79875, abs=1e-9)
    assert nearest_part([], (0.0, 0.0)) is None


def test_from_above_the_nearest_part_is_the_nearest_edge_on_the_floor_whatever_the_height():
    """Contract (FD-0.7.0 review, 3a, 3e): a camera looking straight down (a drone) has the same floor plane, and the
    part is the nearest point along the floor, not in 3-D. By hand: a camera at (10, 5, 15) pitched straight down puts
    the pixel (row i, column j) at depth 15000 mm at the map point (10 - y_cv, 5 - x_cv, 0) (body x is down, body y
    stays, body up is map +x). Pixels in rows 240-243 and columns 330-333 are 10.5 to 13.5 pixels right and 0.5 to 3.5
    down of the principal point: x_cv = 0.39375 to 0.50625 and y_cv = 0.01875 to 0.13125 for 10.5, 11.5, 12.5, 13.5 and
    0.5 to 3.5 pixels; so the points lie at map x from 9.86875 to 9.98125 and y from 4.49375 to 4.60625, on the floor.
    The nearest to the camera's floor point (10, 5) is the corner (9.98125, 4.60625), (9.981, 4.606) to the millimetre
    as a hull vertex; the part is at height 0. Regression caught: the 3-D nearest (the
    same here but not on a roof), a height read as a distance, or the pose of a downward camera applied as a level
    one."""
    index, depth = images({2: (slice(240, 244), slice(330, 334), 15000)})
    pos = (10.0, 5.0, 15.0)
    pts = visible_points(index, depth, CAM, pos, DOWN, 2)
    assert pts[:, 2] == pytest.approx(0.0, abs=1e-9)
    part = nearest_part([(pos, pts)], (10.0, 5.0))
    assert part.xy == pytest.approx((9.981, 4.606), abs=1e-9) and part.z == pytest.approx(0.0, abs=1e-9)


def test_the_floor_hull_is_counter_clockwise_to_the_millimetre_and_without_collinear_vertices():
    """Contract (`floor_hull`, the oracle channel's `floor_hull`): the convex hull of the (x, y) of the points, counter-
    clockwise from the lowest (smallest x, then y) vertex, every vertex to the millimetre. By hand: the corners of the
    unit square, its centre, a point on an edge, a duplicate of a corner and a corner given as (0.0004, 0.0004) give the
    four corners [0, 0], [1, 0], [1, 1], [0, 1]; one point gives itself, two or more collinear the two ends, none an
    empty list. Regression caught: clockwise vertices (a different polygon to every reader), collinear vertices, or
    unrounded vertices (a table and the gold taking different polygons)."""
    pts = [(1, 1, 0), (0.0004, 0.0004, 5), (1, 0, 0), (0, 1, 0), (0.5, 0.5, 0), (0.5, 0, 0), (1, 1, 2)]
    assert floor_hull(pts) == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert floor_hull([(2, 3, 0)]) == [[2.0, 3.0]]
    assert floor_hull([(0, 0, 0), (1, 1, 0), (2, 2, 0), (0.5, 0.5, 0)]) == [[0.0, 0.0], [2.0, 2.0]]
    assert floor_hull([]) == []
    assert union_hull([[[0, 0], [1, 0]], [[0, 1], [1, 1]], [[0.5, 0.5]]]) == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                                                                                [0.0, 1.0]]


def test_the_nearest_point_of_a_hull_may_lie_on_an_edge_and_is_the_ground_point_inside():
    """Contract (`nearest_on_hull`): the nearest point of the filled hull to a ground point. By hand, the unit square:
    from (0.5, -2) it is (0.5, 0) on the bottom edge, not a vertex; from (3, 3) the corner (1, 1); from (0.5, 0.5),
    inside, and from (1, 0.5), on its boundary, the point itself; from the camera's floor point above a lone point, that
    point. A two-point hull is a segment: from (1, 1) over (0, 0)-(2, 0) it is (1, 0). Regression caught: the nearest
    vertex in place of the nearest point (the answer 0.5 m out for a face seen straight on), or a point inside answered
    with the boundary (a drone over the thing)."""
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert nearest_on_hull(square, (0.5, -2.0)) == (0.5, 0.0)
    assert nearest_on_hull(square, (3.0, 3.0)) == (1.0, 1.0)
    assert nearest_on_hull(square, (0.5, 0.5)) == (0.5, 0.5) and nearest_on_hull(square, (1.0, 0.5)) == (1.0, 0.5)
    assert nearest_on_hull([[0.0, 0.0], [2.0, 0.0]], (1.0, 1.0)) == (1.0, 0.0)
    assert nearest_on_hull([[4.0, 4.0]], (0.0, 0.0)) == (4.0, 4.0) and nearest_on_hull([], (0.0, 0.0)) is None


def test_a_tie_for_nearest_goes_to_the_smaller_x_then_y_then_z():
    """Contract (`nearest_point`): of equally near points the smaller x, then y, then z, so float noise never picks the
    point. By hand: (1, 1, 5), (1, -1, 2) and (1, -1, 1) are all sqrt 2 from the origin; the answer is (1, -1, 1).
    Regression caught: an arbitrary choice that differs between the gold and a reader of the same pixels."""
    pts = np.array([[1.0, 1.0, 5.0], [1.0, -1.0, 2.0], [1.0, -1.0, 1.0]])
    assert nearest_point(pts, (0.0, 0.0)) == (1.0, -1.0, 1.0) and nearest_point(pts[:0], (0.0, 0.0)) is None


def test_the_script_reader_reads_one_keyframe_of_its_own_files_and_nothing_else(tmp_path, monkeypatch):
    """Contract (`pixels.ScriptPixels`): `points(k, id)` is the thing's pixels of keyframe `k` of the script's files,
    through that keyframe's depth and the pose row's camera, and reads no other keyframe, so a gold that names frame `j`
    cannot see a later one; `gaps` counts its pixels with no depth. By hand: a script of frames 3 and 7, thing E0001
    (pass index 5) a 1 x 2 block at row 240, columns 320-321 at 2000 mm, the second pixel of depth 0 at frame 7, the
    camera level at (0, 0, 1) at frame 3 and at (1, 0, 1) at frame 7: the frame-3 points are (2, -0.0025, 0.9975) and
    (2, -0.0075, 0.9975), the frame-7 one is (3, -0.0025, 0.9975) with 1 gap. Asked for frame 3 it opens frame 3's
    instance and depth images only; an id that is no instance raises KeyError; a thing with no pixel in the frame has
    the empty set; a script with no image at all is fine until it is asked. Regression caught: a gold that reads the
    whole script (an ego question that sees the future), a reader that fails at load for a script that is never asked,
    or a depth-less pixel counted as a point (the hull then misses nothing and says so)."""
    paths = ScriptPaths(tmp_path, "s1")
    assert pixels.ScriptPixels(paths) is not None   # nothing is read yet
    for folder in (paths.sensors, paths.instances_dir, paths.depth_dir):
        folder.mkdir(parents=True)
    paths.camera.write_text(json.dumps({"width": 640, "height": 480, "fx": 400.0, "fy": 400.0, "cx": 320.0,
                                        "cy": 240.0, "depth_unit": "mm", "near_m": 0.3, "far_m": 150.0}),
                            encoding="utf-8")
    paths.pose.write_text("\n".join(json.dumps({"k": k, "camera": {"pos": [x, 0.0, 1.0], "rot": LEVEL}})
                                    for k, x in ((3, 0.0), (7, 1.0))), encoding="utf-8")
    paths.instance_map.write_text(json.dumps({"5": "E0001", "6": "E0002"}), encoding="utf-8")
    for k in (3, 7):
        index, depth = images({5: (slice(240, 241), slice(320, 322), 2000)})
        depth[240, 321] = 0 if k == 7 else 2000
        write_png16(paths.instances(k), index)
        write_png16(paths.depth(k), depth)
    opened = []
    real = pixels.read_png16
    monkeypatch.setattr(pixels, "read_png16", lambda p: (opened.append(p.name), real(p))[1])
    reader = pixels.ScriptPixels(paths)
    assert reader.points(3, "E0001").round(9).tolist() == [[2.0, -0.0025, 0.9975], [2.0, -0.0075, 0.9975]]
    assert opened == ["000003.png", "000003.png"] and reader.points(3, "E0002").shape == (0, 3)
    assert reader.gaps(3, "E0001") == 0
    with pytest.raises(KeyError, match="E0099"):
        reader.points(3, "E0099")
    assert reader.points(7, "E0001").round(9).tolist() == [[3.0, -0.0025, 0.9975]] and reader.gaps(7, "E0001") == 1
