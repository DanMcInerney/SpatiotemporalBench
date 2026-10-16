"""The rules a gold row of the graded credit is built from (`rules.resolution_m`, `rules.graded_tolerances`, 0.9.0):
the camera's one-pixel error at a floor point and the `tolerances` dict, checked against `records.check_gold`."""

from __future__ import annotations

import math

import pytest

from stbench.rules import HARMS, graded_tolerances, resolution_m


def test_the_resolution_is_one_pixel_radially_and_laterally_combined():
    """Contract (brief §3.7): a floor point at floor distance d from a camera h above the floor is one pixel off by
    (d^2 + h^2) / (h fy) along the ray's footprint and sqrt(d^2 + h^2) / fx across it, combined as a hypotenuse. Hand
    case, a point 1 m from a camera 1.3 m up, fx = fy = 100 px: radial 2.69 / 130 = 0.020692, lateral 1.640122 / 100 =
    0.016401, together 0.026405. Regression caught: a resolution from the wrong axis, or one that forgets the lateral
    pixel."""
    assert resolution_m((0.0, 0.0, 1.3), (1.0, 0.0), 0.0, 100.0, 100.0) == pytest.approx(0.026405, abs=1e-6)
    assert resolution_m((0.0, 0.0, 1.3), (0.0, 1.0), 0.0, 100.0, 100.0) == pytest.approx(0.026405, abs=1e-6)
    # a wider horizontal focal length narrows the lateral pixel only: fx 400 gives lateral 0.004100, together 0.021095
    assert resolution_m((0.0, 0.0, 1.3), (1.0, 0.0), 0.0, 400.0, 100.0) == pytest.approx(0.021095, abs=1e-6)


def test_the_resolution_is_read_from_the_floor_the_point_stands_on():
    """Contract: h is the camera's height above the point's own floor (`base_z`), not above z = 0. Hand case: the same
    point with the floor 0.5 m up has h = 0.8: radial (1 + 0.64) / 80 = 0.0205, lateral sqrt(1.64) / 100 = 0.012806,
    together 0.024171. Regression caught: a camera height taken from the world's origin on a raised floor."""
    assert resolution_m((0.0, 0.0, 1.3), (1.0, 0.0), 0.5, 100.0, 100.0) == pytest.approx(0.024171, abs=1e-6)


def test_the_resolution_is_at_least_a_centimetre_and_none_along_the_floor():
    """Contract: a resolution under `MIN_RHO_M` (1 cm) is 1 cm (a camera 1.3 m straight above, 1,000 px: 0.0018 m), and
    a camera within `CONTACT_M` (2 cm) of the floor's plane has no bound, None. Regression caught: a resolution under
    the pose's own rounding, or a division by a height of 0."""
    assert resolution_m((0.0, 0.0, 1.3), (0.0, 0.0), 0.0, 1000.0, 1000.0) == 0.01
    assert resolution_m((0.0, 0.0, 0.02), (1.0, 0.0), 0.0, 100.0, 100.0) is None
    assert resolution_m((0.0, 0.0, 0.0), (1.0, 0.0), 0.0, 100.0, 100.0) is None


def test_the_tolerances_are_the_rows_keys_for_each_gold_type():
    """Contract: `graded_tolerances` builds an offset's row (resolution, need, harm), a number's (and the threshold)
    and a position's (the same three keys as an offset's: a place holds no room), to the micrometre, and `rho` is at
    least 1 cm. Regression caught: a key the row validator expects missing, or a position row that carries a room the
    scorer no longer reads."""
    assert graded_tolerances(0.0526314, 0.15, "radial") == {"resolution_m": 0.052631, "need_m": 0.15, "harm": "radial"}
    assert graded_tolerances(0.02, 0.15, "over", threshold=0.49) == {"resolution_m": 0.02, "need_m": 0.15,
                                                                      "harm": "over", "threshold_m": 0.49}
    assert graded_tolerances(0.001, 0.5, None) == {"resolution_m": 0.01, "need_m": 0.5, "harm": None}
    assert HARMS == ("over", "radial", None)


@pytest.mark.parametrize(("args", "match"), [
    ((0.02, 0.0, "over"), "need"), ((0.02, -0.1, "over"), "need"), ((math.nan, 0.15, "over"), "rho"),
    ((0.0, 0.15, "over"), "rho"), ((0.02, 0.15, "under"), "harm"), ((0.02, 0.15, "over", 0.0), "threshold")])
def test_a_bad_resolution_need_harm_or_threshold_is_refused(args, match):
    """Contract: a resolution or a need that is not a finite number above 0, a harm outside `HARMS` and a threshold at
    or under 0 are refused, naming the argument. Regression caught: a row the validator would then refuse at compile,
    far from the family that built it."""
    with pytest.raises(ValueError, match=match):
        graded_tolerances(*args)
