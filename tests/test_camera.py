"""Camera geometry: the one transform every reader of poses and detections shares."""

import math

import numpy as np
import pytest

from stbench.camera import Camera, cam_to_map, in_image, map_to_cam, project, rot_matrix, unproject

CAM = Camera(640, 480, 381.36115, 381.36115, 320.0, 240.0)
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def yaw_quat(deg: float) -> list[float]:
    a = math.radians(deg) / 2
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def test_axes_follow_the_documented_body_to_opencv_mapping():
    """Contract: with an identity pose, map +x is camera forward (+z_cv), map +y (left) is -x_cv, map +z is -y_cv.
    Regression: a sign or axis swap in CV_FROM_BODY puts every detection on the wrong side of the robot."""
    pos = [0.0, 0.0, 0.0]
    np.testing.assert_allclose(map_to_cam([2.0, 0.0, 0.0], pos, IDENTITY), [0.0, 0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(map_to_cam([0.0, 1.0, 0.0], pos, IDENTITY), [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(map_to_cam([0.0, 0.0, 1.0], pos, IDENTITY), [0.0, -1.0, 0.0], atol=1e-12)


def test_yaw_turns_the_view_counter_clockwise():
    """Contract: a camera at (1, 1, 1.5) yawed +90 deg looks along map +y, so a point 3 m north of it is dead ahead.
    Regression: transposed rotation (map->body applied as body->map) mirrors the scene for any non-zero yaw."""
    np.testing.assert_allclose(map_to_cam([1.0, 4.0, 1.5], [1.0, 1.0, 1.5], yaw_quat(90)), [0.0, 0.0, 3.0],
                               atol=1e-9)


def test_round_trip_map_cam_map_is_identity_for_arbitrary_pose():
    """Contract: cam_to_map inverts map_to_cam for any unit quaternion and batch of points.
    Regression: an inverse that drops the translation or the CV permutation breaks memory baselines that store
    detections in the map frame."""
    rng = np.random.default_rng(7)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    pos = [0.3, -2.0, 1.5]
    pts = rng.uniform(-5, 5, size=(20, 3))
    np.testing.assert_allclose(cam_to_map(map_to_cam(pts, pos, q), pos, q), pts, atol=1e-9)


def test_projection_hits_the_principal_point_and_pinhole_offsets():
    """Contract: a point on the optical axis projects to (cx, cy); 1 m right at 2 m depth lands fx/2 px right.
    Regression: swapped fx/fy or cx/cy, or dividing by range instead of planar depth."""
    assert project([0.0, 0.0, 2.0], CAM) == (320.0, 240.0)
    u, v = project([1.0, -0.5, 2.0], CAM)
    assert u == pytest.approx(320.0 + 381.36115 / 2)
    assert v == pytest.approx(240.0 - 381.36115 / 4)
    assert project([0.0, 0.0, -1.0], CAM) is None


def test_unproject_inverts_project_with_planar_depth():
    """Contract: unproject(project(p), z) returns p when z is the planar depth (camera-frame z).
    Regression: treating depth as ray length shrinks off-axis points toward the camera (0.69x at the image corner)."""
    p = np.array([1.7, -0.9, 3.2])
    u, v = project(p, CAM)
    np.testing.assert_allclose(unproject(u, v, 3.2, CAM), p, atol=1e-9)


def test_in_image_bounds_are_half_open():
    """Contract: u in [0, width), v in [0, height); points behind the camera are never in the image.
    Regression: an inclusive upper bound counts an off-image point as in view in the `changed` rules."""
    assert in_image((0.0, 0.0), CAM)
    assert not in_image((640.0, 10.0), CAM)
    assert not in_image((10.0, 480.0), CAM)
    assert not in_image(None, CAM)


def test_non_unit_quaternion_is_rejected():
    """Contract: a malformed rotation raises instead of silently scaling the scene.
    Regression: a pose written as degrees or an unnormalised quaternion would distort every transform."""
    with pytest.raises(ValueError):
        rot_matrix([2.0, 0.0, 0.0, 0.0])
