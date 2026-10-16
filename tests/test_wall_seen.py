"""The wall record and the corridor's hit (0.8.0 lane L2: `wall_seen`), worked by hand on `fly_kit`'s two walls.

`WE` runs from (20, -10) to (20, 10), 0.6 m thick, so its box is x 19.7..20.3, y -10.3..10.3; a point's bin along
it is `floor(y + 10)`, its distance across the wall `20 - x` (a point inside the box is within 0.3, the record
takes 0.35 more). A point is on a wall only more than 0.5 m up.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from fly_kit import MAP, WE, WW, face, flight, fly_inputs
from stbench import wall_seen
from stbench.robots import ROBOTS

DRONE = ROBOTS["drone"]


def test_a_depth_point_is_on_a_wall_by_height_distance_and_end():
    """Contract (plan L2, `wall_seen.bins_of`): a point is on a wall when it is more than 0.5 m above the floor,
    within 0.35 m of the wall's box across it (0.3 + 0.35 = 0.65 m of the wall's line) and within 0.5 m of its
    ends, and the bin is `floor(along / 1 m)` from the wall's `a` end. Hand-worked on `WE` (along = y + 10,
    across = 20 - x): the point (19.7, 0, 5) is at along 10, bin 10; (19.7, 0.99, 5) is still bin 10 and (19.7,
    1.0, 5) is bin 11; z 0.5 is not above 0.5 and z 0.51 is; x 19.5 (0.5 across) is on, x 19.0 (1.0 across) is
    not; y -10.4 (along -0.4) is bin -1 and y 10.4 (along 20.4) bin 20, y -10.6 and y 10.6 (0.6 past the ends)
    are not. Regression caught: the street seen from above counted as a wall (the floor test lost), a record
    that takes the whole facing street, or a bin that drifts from the wall's end."""
    pts = np.array([(19.7, 0.0, 5.0), (19.7, 0.99, 5.0), (19.7, 1.0, 5.0), (19.7, 5.5, 0.5), (19.7, 6.5, 0.51),
                    (19.5, -3.0, 5.0), (19.0, -3.0, 5.0), (19.7, -10.4, 5.0), (19.7, 10.4, 5.0), (19.7, -10.6, 5.0),
                    (19.7, 10.6, 5.0)])
    assert wall_seen.bins_of(pts, [WE]) == {("WE", 10), ("WE", 11), ("WE", 16), ("WE", 7), ("WE", -1), ("WE", 20)}


def test_each_wall_has_its_own_bins_by_its_own_direction():
    """Contract: bins are kept per wall, along the wall's own direction from its `a` end, so two walls never share a
    bin. Hand-worked: (19.7, 0, 5) is on `WE` (bin 10), (-14.7, 3, 5) is on `WW` (along 13, bin 13) and (-14.7,
    3, 5) is nowhere near `WE`; a point between the walls (0, 0, 5) is on neither. Regression caught: bins
    keyed by position alone (a wall's neighbour stretch read as the other wall's)."""
    pts = np.array([(19.7, 0.0, 5.0), (-14.7, 3.0, 5.0), (0.0, 0.0, 5.0)])
    assert wall_seen.bins_of(pts, [WE, WW]) == {("WE", 10), ("WW", 13)}
    assert wall_seen.bins_of(np.empty((0, 3)), [WE, WW]) == frozenset()


def test_the_corridor_meets_the_near_face_from_the_edge_of_the_body():
    """Contract (plan L2): the corridor starts at the body's edge, the base plus the radius (0.65 m) along the heading
    or its opposite, and the distance is to the wall box's near face, with the id of the wall there.
    Hand-worked from the base (0, 0): ahead (+x) the edge is at x 0.65 and `WE`'s face at 19.7 is 19.05 m on; back
    the edge is at x -0.65 and `WW`'s face at -14.7 is 14.05 m on; a heading of pi turns ahead into back; from (0,
    3.5) ahead is still `WE`. Regression caught: a distance from the base or the camera (19.7), or the wrong wall
    named."""
    assert wall_seen.hit(MAP, DRONE, (0.0, 0.0), 0.0, "ahead", 60.0) == (19.05, "WE")
    assert wall_seen.hit(MAP, DRONE, (0.0, 0.0), 0.0, "back", 60.0) == (14.05, "WW")
    assert wall_seen.hit(MAP, DRONE, (0.0, 0.0), math.pi, "ahead", 60.0) == (14.05, "WW")
    assert wall_seen.hit(MAP, DRONE, (0.0, 3.5), 0.0, "ahead", 60.0) == (19.05, "WE")


def test_a_corridor_that_runs_on_is_open_air_and_a_clipped_end_still_meets_the_wall():
    """Contract: nothing within the cap is None (open air is never certified); a corridor that clips the wall's end
    meets that wall. Hand-worked: facing +y from (0, 0) nothing stands in 60 m; the wall 19.05 m on is out of a
    10 m cap; from (0, 10.6) the 1.3 m corridor (y 9.95..11.25) clips `WE`'s end (its box ends at y 10.3) and the
    centre line is past the wall's end (along 20.6), so the hit is `WE`. The body is a disc (radius 0.65) and meets the
    end's corner at the sub-cell centred (19.71, 10.29), 19.06 on and 0.31 to the side, after
    19.06 + 0.65 - sqrt(0.65^2 - 0.31^2) = 19.139, less 0.01: 19.129 (a strip would say 19.05). Regression caught: an
    uncertified 60, the wall of a hit past its end not found, or the disc read as a strip."""
    assert wall_seen.hit(MAP, DRONE, (0.0, 0.0), math.pi / 2, "ahead", 60.0) is None
    assert wall_seen.hit(MAP, DRONE, (0.0, 0.0), 0.0, "ahead", 10.0) is None
    assert wall_seen.hit(MAP, DRONE, (0.0, 10.6), 0.0, "ahead", 60.0) == (19.129, "WE")


def test_a_frame_saw_the_hit_wall_by_any_bin_of_it():
    """Contract: a keyframe saw the wall the body meets when its record holds some bin of that wall, however far
    along it from the stretch met (a building face is one flat plane), and the keyframes are given back in the order
    asked. Hand-worked: records 0 {WE 9}, 1 {WE 15}, 2 {WE 0, WW 10}, 3 {WW 10}, 4 {} against the wall `WE`: 0, 1
    and 2 saw it (bins 9, 15 and 0), 3 saw another wall's bin 10, 4 saw nothing; `WW` is seen by 2 and 3; a wall
    nobody saw gives none. Regression caught: the neighbour rule kept (a wall seen five metres along answered
    `unknown`), or another wall's bins counted."""
    record = {0: frozenset({("WE", 9)}), 1: frozenset({("WE", 15)}), 2: frozenset({("WE", 0), ("WW", 10)}),
              3: frozenset({("WW", 10)}), 4: frozenset()}
    assert wall_seen.saw(record, [0, 1, 2, 3, 4], "WE") == [0, 1, 2]
    assert wall_seen.saw(record, [4, 3, 2, 1, 0], "WE") == [2, 1, 0]
    assert wall_seen.saw(record, [0, 1, 2, 3, 4], "WW") == [2, 3]
    assert wall_seen.saw(record, [0, 1, 2, 3, 4], "WX") == []
    assert wall_seen.saw(record, [0, 9], "WE") == [0]   # a keyframe with no record saw nothing


def test_a_keyframe_is_read_once_per_inputs_and_a_new_inputs_reads_again():
    """Contract (plan L2): the record is cached per `Inputs`, as `states_of` is: asking for keyframes twice, and for
    an overlapping set, reads each image once; another `Inputs` of the same script reads its own. Counted by
    hand: asking [0, 1], then [1, 2, 2], then [0] reads keyframes 0, 1, 2 once each; a second `Inputs` asked
    [0] reads 0 again. Regression caught: a depth image read for every question of a case (minutes a script)."""
    script = flight(tasks=1)
    read: list[int] = []
    inp = fly_inputs(script, {0: face(19.7, [0.0])}, counter=read)
    assert wall_seen.bins(inp, [0, 1]) == {0: frozenset({("WE", 10)}), 1: frozenset()}
    wall_seen.bins(inp, [1, 2, 2])
    assert wall_seen.bins(inp, [0]) == {0: frozenset({("WE", 10)})}
    assert read == [0, 1, 2]
    other: list[int] = []
    wall_seen.bins(fly_inputs(script, {}, counter=other), [0])
    assert other == [0]


def test_inputs_without_depth_points_are_refused_by_name():
    """Contract: a compile whose inputs carry no `depth_points_of` raises, naming it, rather than reading no wall.
    Regression caught: a flier's question answered `unknown` everywhere for want of a reader."""
    inp = fly_inputs(flight(tasks=1))
    with pytest.raises(ValueError, match="depth_points_of"):
        wall_seen.bins(replace(inp, depth_points_of=None), [0])
