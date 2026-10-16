"""The relative rungs of a distance asked of the camera (`rules.relative_ladder`, 0.6.0), by hand."""

from __future__ import annotations

import math

import pytest

from stbench.rules import REL_FLOORS_M, REL_SHARES, relative_ladder


@pytest.mark.parametrize("r, rungs", [
    (0.3, [0.05, 0.10, 0.20]),      # every share is under its floor: 0.03, 0.075 and 0.15 m
    (1.0, [0.10, 0.25, 0.50]),      # 10%, 25% and 50% of 1 m, all over their floors
    (5.0, [0.50, 1.25, 2.50]),
    (20.0, [2.0, 5.0, 10.0]),
])
def test_the_rungs_are_a_share_of_the_range_with_a_floor(r, rungs):
    """Contract: rung i is `max(floor_i, share_i * r)` with shares 10%, 25%, 50% and floors 5, 10, 20 cm, so a thing
    0.3 m away has the floors, one 1 m away the shares, and farther things proportionally wider rungs. By hand at r =
    0.3, 1, 5 and 20 m. Regression caught: a floor dropped (a near thing's rungs at 3 mm, ungradeable) or the shares
    swapped with the floors."""
    assert relative_ladder(r) == pytest.approx(rungs)


def test_a_widening_is_added_to_every_rung_and_the_rungs_always_rise():
    """Contract: a large thing's widening adds to each rung (after the max), the rungs are rounded to the micrometre,
    and they are strictly increasing for every range from 0 up, so each is a valid `tolerances.ladder`. By hand: r = 1
    with a 0.3 m widening is 0.40, 0.55, 0.80; at r = 0 the floors plus the widening. Regression caught: the widening
    taken inside the max (a floor swallowing it), or a range at which two rungs meet and a rung met would be worth
    nothing."""
    assert relative_ladder(1.0, 0.3) == pytest.approx([0.40, 0.55, 0.80])
    assert relative_ladder(0.0, 0.1) == pytest.approx([0.15, 0.20, 0.30])
    assert relative_ladder(1.23456789) == [0.123457, 0.308642, 0.617284]   # to the micrometre
    for r in (0.0, 0.05, 0.2, 0.4, 0.5, 1.0, 2.0, 18.0278, 100.0):
        rungs = relative_ladder(r, 0.05)
        assert all(a < b for a, b in zip(rungs, rungs[1:], strict=False)), r


def test_the_range_is_the_slant_not_the_horizontal_distance():
    """Contract: the rungs scale with the 3-D slant range, so a drone 15 m up looking at a thing 10 m along the floor
    gets rungs for 18.03 m, not 10 m. By hand: hypot(10, 15) = 18.0278, rungs 1.80278, 4.50694 and 9.0139 against the
    horizontal 1.0, 2.5 and 5.0. Regression caught: the horizontal range feeding the ladder, which gave the drone's
    near things rungs under 0.1 m (the planner's measurement: 'the drop point' scored -0.72)."""
    slant = math.hypot(10.0, 15.0)
    assert relative_ladder(slant) == pytest.approx([1.80278, 4.50694, 9.0139], abs=1e-4)
    assert relative_ladder(10.0) == pytest.approx([1.0, 2.5, 5.0])


def test_the_constants_and_the_refusals():
    """Contract: `rules` is the one owner of the numbers (the prompt reads them): three shares and three floors, in
    rising order, and a range that is negative, not finite or not a number (or a negative widening) is refused.
    Regression caught: a copy of the numbers in the prompt that drifts from the scorer's, or a NaN range yielding rungs
    that every error meets."""
    assert REL_SHARES == (0.10, 0.25, 0.50) and REL_FLOORS_M == (0.05, 0.10, 0.20)
    for bad in (-1.0, float("nan"), float("inf"), "3", None, True):
        with pytest.raises(ValueError, match="r_m"):
            relative_ladder(bad)
    with pytest.raises(ValueError, match="widen_m"):
        relative_ladder(1.0, -0.1)
