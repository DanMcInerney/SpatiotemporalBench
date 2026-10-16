"""The free-space rule in `grid.py` (plan §4.5): fine-cell states from points, the latest state of each cell, the
`path_ahead` answer and a footprint's swath, every expected value worked out by hand."""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle

import numpy as np
import pytest

from stbench import grid
from stbench.robots import ROBOTS, blind_m

VACUUM = ROBOTS["vacuum"]
# the vacuum's size as a box: the rectangle rule's hand cases (a disc's are `test_grid_disc.py`)
BOX = dataclasses.replace(VACUUM, body=dataclasses.replace(VACUUM.body, shape="rect"))
FRONT_EDGE_AT_ZERO = (-0.17, 0.0)  # the vacuum's base 0.17 m (half its 0.34 m length) behind x = 0


def test_band_by_hand():
    """Contract: `band(robot)` is (what it drives over plus the depth quantum, the body's height plus half its
    under-margin). Hand values: the vacuum 0.02 + 0.003 = 0.023 and 0.09 + 0.01 / 2 = 0.095; the AMR 0.01 + 0.003 =
    0.013 and 0.65 + 0.05 / 2 = 0.675. Regression caught: a band topped at the passing height itself (0.10), which
    lets the vacuum's 0.10 m shelf, a surface the planner treats as overhead, mark cells occupied whatever the depth
    image's millimetre rounding; a floor that ends at the climb, so a rug as high as the climb stops the robot
    (`test_band_quantum.py`)."""
    assert grid.band(VACUUM) == (0.023, 0.095)
    assert grid.band(ROBOTS["amr"]) == (0.013, 0.675)


def test_observe_by_hand():
    """Contract: a fine cell is occupied when any point's height above the floor is over the climb and at most the
    top, free when at least four points are at or under the climb and none is in the band, and absent otherwise; points
    above the band are ignored. Hand-worked on the vacuum's band (0.02, 0.095) over a floor at z 1.0, with a point's
    cell the 0.1 m cell holding its (x, y), a coordinate on a boundary opening the cell above (x = 0.3 is cell 3):
    cell (3, 4) holds three floor points (z 1.0, 1.01 and 0.999): absent; a fourth, at exactly the climb (1.02), makes
    it free; cell (5, 5) holds ten floor points and one at 1.05: occupied; cell (6, 6) one at exactly the top (1.095):
    occupied; cell (7, 7) one at 1.0951 and the floor four times over: overhead ignored, so free; cell (8, 8) only the
    point at 1.0951: absent. An occupied cell also names the 0.02 m sub-cells (a, b) = floor(x / 0.02), floor(y / 0.02)
    that hold its in-band points: (5, 5)'s point (0.53, 0.53) is in (26, 26) and (6, 6)'s (0.63, 0.63) in (31, 31).
    Regression caught: a floor read as free from fewer than four points, a band edge off by one side, or an overhead
    surface marking cells."""
    def at(u, v, *heights):
        return [(0.1 * u + 0.03, 0.1 * v + 0.03, z) for z in heights]

    pts = np.array(at(3, 4, 1.0, 1.01, 0.999) + at(5, 5, *[1.0] * 10, 1.05) + at(6, 6, 1.095)
                   + at(7, 7, 1.0951, *[1.0] * 4) + at(8, 8, 1.0951))
    states = grid.observe(pts, 1.0, (0.02, 0.095))
    assert states == {(5, 5): "occupied", (6, 6): "occupied", (7, 7): "free"}
    assert states[(5, 5)].subs == {(26, 26)} and states[(6, 6)].subs == {(31, 31)}  # (0.53, 0.53), (0.63, 0.63)
    states = grid.observe(np.array(at(3, 4, 1.0, 1.01, 0.999, 1.02)), 1.0, (0.02, 0.095))
    assert states == {(3, 4): "free"}
    assert grid.observe(np.array([(0.3, 0.3, 1.0)] * 4), 1.0, (0.02, 0.095)) == {(3, 3): "free"}
    assert grid.observe(np.zeros((0, 3)), 0.0, (0.02, 0.095)) == {}


def test_an_occupied_cell_names_the_sub_cells_of_its_points():
    """Contract: an occupied state is the string "occupied" that also holds the 0.02 m sub-cells of the cell's in-band
    points (floor points name none), and a point's fine cell is its sub-cell's, so a decimal on a boundary opens the
    same cell in both. Hand-worked on the vacuum's band over a floor at z 1.0: points (0.21, 0.31) and (0.29, 0.39) at
    1.05 are in sub-cells (10, 15) and (14, 19) of fine cell (2, 3), with a floor point (0.25, 0.35) beside them that
    adds none; (0.3, 0.3) at 1.05 is in sub-cell (15, 15), so in fine cell (3, 3). The state survives pickle and copy
    (states are kept per keyframe and handed between processes). Regression caught: a cell that says it is occupied
    and not where, so the corridor's width is judged to 0.1 m, or states that fall back to bare text over a process
    boundary."""
    pts = np.array([(0.21, 0.31, 1.05), (0.29, 0.39, 1.05), (0.25, 0.35, 1.0), (0.3, 0.3, 1.05)])
    states = grid.observe(pts, 1.0, (0.02, 0.095))
    assert states == {(2, 3): "occupied", (3, 3): "occupied"}
    assert states[(2, 3)].subs == {(10, 15), (14, 19)} and states[(3, 3)].subs == {(15, 15)}
    for same in (pickle.loads(pickle.dumps(states[(2, 3)])), copy.deepcopy(states[(2, 3)])):
        assert same == "occupied" and same.subs == {(10, 15), (14, 19)}


def test_latest_keeps_the_last_state_and_its_keyframe():
    """Contract: `latest` gives each fine cell the state and the keyframe of the highest k that observed it, whatever
    order the keyframes come in. Hand values: cell (1, 1) is free at k 2 and occupied at k 5, so (occupied, 5); cell
    (2, 2) is occupied at k 2 and free at k 5 (a thing that left), so (free, 5); (3, 3) is seen only at k 2. Regression
    caught: the first state kept, or a state with no keyframe, so the evidence of a gold names the wrong frame."""
    seen = {5: {(1, 1): "occupied", (2, 2): "free"}, 2: {(1, 1): "free", (2, 2): "occupied", (3, 3): "free"}}
    assert grid.latest(seen) == {(1, 1): ("occupied", 5), (2, 2): ("free", 5), (3, 3): ("free", 2)}
    assert grid.latest({}) == {}


def corridor_states(free_us, k=3, v_range=range(-2, 2), cells=None):
    """The latest states of a corridor from the origin along +x (u counts the 0.1 m cells, v -2 to 1 across it): `free`
    at every u in `free_us`, observed at keyframe k."""
    return {**{(u, v): ("free", k) for u in free_us for v in v_range}, **(cells or {})}


def test_path_ahead_gold_by_hand():
    """Contract: the corridor starts at the body's front edge, is the body's width (0.34 m, v -2 to 1) and `cap` long;
    the answer is the near edge of the first occupied cell, `cap` when there is none, and `unknown` when fewer than 80%
    of the cells beyond `blind_m` and before that cell are free. Hand values for the vacuum as a box (`BOX`; base x
    -0.17, front edge x 0, so a cell u is 0.1 u to 0.1 u + 0.1 and its centre is 0.05 + 0.1 u from the edge; blind_m
    is 0.1158):
    - occupied at u 14 (centre 1.45) with u 1 to 13 free (52 cells): 1.45 - 0.05 = 1.40, evidence the free cells' k 3
      and the occupied cell's k 5; a corridor that began at the camera (x -0.04) would give 1.44;
    - nothing occupied and u 1 to 29 free (116 cells): 3.0; u 0, nearer than blind_m, may be unobserved;
    - cap 1.0, so 36 counted cells (u 1 to 9): 29 free is 80.5% and answers 1.0, 28 free is 77.8% and answers unknown,
      22 free is 61% and answers unknown;
    - an occupied cell at u 0 (centre 0.05, inside the blind distance): 0.0, with nothing counted before it;
    - the same corridor turned to +y from base (0, -0.17) with the cell at (0, 14): 1.4.
    The vacuum's blind distance is 0.1158 m, so u 0 (0.05) is inside it and u 1 (0.15) is not.
    Regression caught: the corridor started at the camera (map X-H20), the 80% rule dropped or counted from the blind
    band, or the near edge read as the cell's centre (1.45)."""
    assert blind_m(VACUUM) == pytest.approx(0.1158, abs=1e-4)
    answer, evidence = grid.path_ahead(corridor_states(range(1, 14), 3, cells={(14, 0): ("occupied", 5)}),
                                       BOX, FRONT_EDGE_AT_ZERO, 0.0, 3.0)
    assert (answer, evidence) == ({"type": "number", "value": 1.4}, [3, 5])
    assert grid.path_ahead(corridor_states(range(1, 30)), BOX, FRONT_EDGE_AT_ZERO, 0.0, 3.0) == (
        {"type": "number", "value": 3.0}, [3])
    free_29 = {(u, v): ("free", 3) for u in range(1, 10) for v in range(-2, 2)}
    assert grid.path_ahead(free_29, BOX, FRONT_EDGE_AT_ZERO, 0.0, 1.0)[0] == {"type": "number", "value": 1.0}
    for dropped, expected in ((7, "number"), (8, "unknown"), (14, "unknown")):
        states = dict(free_29)
        for cell in list(states)[:dropped]:
            del states[cell]
        assert grid.path_ahead(states, BOX, FRONT_EDGE_AT_ZERO, 0.0, 1.0)[0]["type"] == expected
    near = grid.path_ahead({(0, 0): ("occupied", 4)}, BOX, FRONT_EDGE_AT_ZERO, 0.0, 3.0)
    assert near == ({"type": "number", "value": 0.0}, [4])
    north = {(u, v): ("free", 3) for v in range(1, 14) for u in range(-2, 2)} | {(0, 14): ("occupied", 5)}
    assert grid.path_ahead(north, BOX, (0.0, -0.17), math.pi / 2, 3.0) == ({"type": "number", "value": 1.4}, [3, 5])


def test_path_ahead_counts_what_it_saw_and_nothing_else():
    """Contract: an absent cell is unobserved, not free, and an occupied cell beyond the first is never read. Hand
    values (cap 3): the first occupied cell at u 20 (centre 2.05) after 76 counted cells (u 1 to 19): all free gives
    2.0; 60 free of the 76 is 78.9% and gives unknown; of two occupied cells at u 5 (centre 0.55) the one the corridor
    meets first is the cell with the lower v, so the answer is 0.5 and the evidence its k, not the other's. An empty
    state map has every cell of the corridor counted and none seen: unknown, with no evidence. Regression caught: a
    corridor read as empty because nothing in it was seen (map X-H5), or the scan taking the nearest occupied cell of
    the wrong row."""
    wall = {(20, 0): ("occupied", 6)}
    assert grid.path_ahead(corridor_states(range(1, 20), 2, cells=wall), BOX, FRONT_EDGE_AT_ZERO, 0.0, 3.0) == (
        {"type": "number", "value": 2.0}, [2, 6])
    sparse = corridor_states(range(1, 20), 2, cells=wall)
    for cell in [(u, 1) for u in range(10, 20)] + [(u, -2) for u in range(14, 20)]:
        del sparse[cell]
    answer, evidence = grid.path_ahead(sparse, BOX, FRONT_EDGE_AT_ZERO, 0.0, 3.0)
    assert answer == {"type": "unknown"} and evidence == [2, 6]
    two = corridor_states(range(1, 5), 2, cells={(5, -2): ("occupied", 7), (5, 1): ("occupied", 8)})
    assert grid.path_ahead(two, BOX, FRONT_EDGE_AT_ZERO, 0.0, 3.0) == ({"type": "number", "value": 0.5}, [2, 7])
    assert grid.path_ahead({}, BOX, FRONT_EDGE_AT_ZERO, 0.0, 3.0) == ({"type": "unknown"}, [])


def sub_cells(u, v, rows=None, cols=None):
    """The sub-cells of fine cell (u, v) in the given rows (b offsets 0-4) and columns (a offsets); all by default."""
    return {(5 * u + da, 5 * v + db) for db in (rows if rows is not None else range(5))
            for da in (cols if cols is not None else range(5))}


def test_path_ahead_stops_only_for_what_stands_in_the_bodys_width():
    """Contract (F1 of M1's review): what stops the corridor is an obstacle sub-cell (0.02 m) whose centre lies inside
    the body's width (the vacuum's box, 0.34 m, half-width 0.17 from the axis), not a fine cell whose 0.1 m centre
    does: a door frame beside a doorway, 0.19 m off the axis, is outside, and the occupied cell that holds it neither
    stops the body nor counts against the 80% rule. Hand values, cap 2 (so cells u 1 to 19, v -2 to 1, are counted: 76
    of them, free at keyframe 3), the front edge at x 0, an obstacle in cell (10, 1), x 1.0 to 1.1 and y 0.1 to 0.2,
    whose sub-cells (a, b) have b 5 to 9 and centres y = 0.02 b + 0.01 = 0.11, 0.13, 0.15, 0.17 and 0.19:
    - on b 9 alone (0.19, 0.02 outside): no stop, the answer is 2.0, with only the free cells' keyframe 3 as evidence;
    - on b 8 (0.17, on the edge, which is inside) or b 7 (0.15): the stop, 1.0 (the cell's centre 1.05 less 0.05), with
      evidence 3 and the cell's keyframe 5;
    - the mirror, cell (10, -2) (y -0.2 to -0.1) with b -10 (-0.19) outside and b -9 (-0.17) inside, behaves the same,
      and so does the corridor turned to +y from (0, -0.17) with cell (1, 10) (x 0.1 to 0.2): a 9 (0.19) outside, a 8
      (0.17) inside; and cell (-2, 10): a -10 (-0.19) outside, a -9 (-0.17) inside;
    - the cell at the far side of the doorway having a bare "occupied" state is the whole cell, so it stops;
    - sixteen cells of row v 1 (u 2 to 17) holding only an obstacle at b 9 leave 60 cells counted, all free: 2.0, where
      counting them as not seen would give 60 of 76 = 78.9% and unknown; and one at u 5 (keyframe 4) ahead of a bare
      cell at u 8, v 0 (keyframe 6) is not counted, so the evidence is 3 and 6, not 4: the stop is 0.8.
    Regression caught: the corridor judged by fine cell centres (the old rule, 0.05 m too wide), which gave a doorway
    the robot drives through a gold of 0.255 m against 3 m and scored an agent that saw it open as wrong."""
    free = corridor_states(range(1, 20), 3)

    def ahead(cells):
        return grid.path_ahead({**free, **cells}, BOX, FRONT_EDGE_AT_ZERO, 0.0, 2.0)

    def standing(u, v, **where):
        return {(u, v): (grid.Occupied(sub_cells(u, v, **where)), 5)}

    assert ahead(standing(10, 1, rows=[4])) == ({"type": "number", "value": 2.0}, [3])
    for b in (3, 2):
        assert ahead(standing(10, 1, rows=[b])) == ({"type": "number", "value": 1.0}, [3, 5])
    assert ahead(standing(10, -2, rows=[0])) == ({"type": "number", "value": 2.0}, [3])
    assert ahead(standing(10, -2, rows=[1])) == ({"type": "number", "value": 1.0}, [3, 5])
    north_free = {(u, v): ("free", 3) for v in range(1, 20) for u in range(-2, 2)}
    for u, a, stops in ((1, 4, False), (1, 3, True), (-2, 0, False), (-2, 1, True)):
        got = grid.path_ahead({**north_free, **standing(u, 10, cols=[a])}, BOX, (0.0, -0.17), math.pi / 2, 2.0)
        assert got == ({"type": "number", "value": 1.0 if stops else 2.0}, [3, 5] if stops else [3]), (u, a)
    assert ahead({(10, 1): ("occupied", 5)}) == ({"type": "number", "value": 1.0}, [3, 5])
    frame = {(u, 1): (grid.Occupied(sub_cells(u, 1, rows=[4])), 4) for u in range(2, 18)}
    assert ahead(frame) == ({"type": "number", "value": 2.0}, [3])
    beyond = {(5, 1): frame[(5, 1)], (8, 0): ("occupied", 6)}
    assert ahead(beyond) == ({"type": "number", "value": 0.8}, [3, 6])


def test_path_ahead_of_a_flier_is_refused():
    """Contract: a robot that flies has no floor in view, so `path_ahead` raises (`robots.blind_m`) rather than answer.
    Regression caught: a ground rule applied to the drone."""
    with pytest.raises(ValueError, match="flies"):
        grid.path_ahead({}, ROBOTS["drone"], (0.0, 0.0), 0.0, 5.0)


DISC_ALONG_X = {(u, v) for u in range(10) for v in range(-2, 2)} | {
    (-1, v) for v in range(-2, 2)} | {(-2, -1), (-2, 0)} | {(10, v) for v in range(-2, 2)} | {(11, -1), (11, 0)}


def test_swath_by_hand():
    """Contract: a disc covers the fine cells whose centres lie within half its length of a segment between poses; a
    turn in place adds nothing; a rectangle covers the hull of its footprints at both ends when the yaw is unchanged
    and either end's footprint when it changes; one pose covers its own footprint; a centre on the boundary is in.
    Hand-worked (centres at 0.05 + 0.1 u):
    - a 0.34 m disc from (0, 0) to (1, 0): v -2 to 1 (|y| 0.05 or 0.15 is within 0.17) for u 0 to 9 (40 cells), the
      start's cap u -1 (four) and u -2 (v -1, 0: (0.15, 0.15) is 0.212 away), and the same at the end: 52 cells;
    - the same disc turned in place at the origin through three headings: the 12 cells of one pose (four at 0.071, eight
      at 0.158);
    - a 0.4 x 0.2 m rectangle at 45 degrees: centres with |x + y| at most 0.283 and |y - x| at most 0.141, which are
      (-1, -1), (0, 0), (-2, -1), (-1, 0), (0, 1), (-1, -2), (0, -1) and (1, 0);
    - that rectangle at yaw 0 from (0, 0) to (1, 0): x from -0.2 to 1.2 and y within 0.1, so u -2 to 11 by v -1, 0;
    - and with the heading turned to 90 degrees at the far end: the start footprint (u -2 to 1, v -1, 0) and the end's
      (u 9 and 10, v -2 to 1), 16 cells, nothing between.
    Regression caught: a swath that cuts the corners of a disc's caps, a turn in place that sweeps, or a rectangle that
    sweeps a turn it did not make, so `path_log` claims cells its body never covered or misses ones it did."""
    assert grid.swath([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], "disc", 0.34, 0.34) == DISC_ALONG_X
    assert len(DISC_ALONG_X) == 52
    turn = grid.swath([(0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2), (0.0, 0.0, math.pi)], "disc", 0.34, 0.34)
    assert turn == grid.swath([(0.0, 0.0, 0.0)], "disc", 0.34, 0.34) and len(turn) == 12
    diagonal = {(-1, -1), (0, 0), (-2, -1), (-1, 0), (0, 1), (-1, -2), (0, -1), (1, 0)}
    assert grid.swath([(0.0, 0.0, math.pi / 4)], "rect", 0.4, 0.2) == diagonal
    straight = {(u, v) for u in range(-2, 12) for v in (-1, 0)}
    assert grid.swath([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], "rect", 0.4, 0.2) == straight
    turned = grid.swath([(0.0, 0.0, 0.0), (1.0, 0.0, math.pi / 2)], "rect", 0.4, 0.2)
    assert turned == {(u, v) for u in range(-2, 2) for v in (-1, 0)} | {(u, v) for u in (9, 10) for v in range(-2, 2)}


def test_swath_edges():
    """Contract: no poses cover nothing; a shape other than disc or rect is refused; a cell whose centre lies on the
    footprint's boundary is covered. Hand values: a 0.1 x 0.1 m rectangle at yaw 0 centred on (0.05, 0.05) spans x and
    y from 0 to 0.1 and covers exactly fine cell (0, 0) (its neighbours' centres are 0.1 away); centred on (0.1, 0.1) it
    spans 0.05 to 0.15 and the four cell centres (0.05, 0.05), (0.15, 0.05), (0.05, 0.15) and (0.15, 0.15) sit on its
    corners, so all four are covered. Regression caught: an exact float comparison dropping the cells on a boundary,
    differently in the gold's swath and the mapper's."""
    assert grid.swath([], "disc", 0.34, 0.34) == set()
    with pytest.raises(ValueError, match="'disc' or a 'rect'"):
        grid.swath([(0.0, 0.0, 0.0)], "ellipse", 0.34, 0.34)
    assert grid.swath([(0.05, 0.05, 0.0)], "rect", 0.1, 0.1) == {(0, 0)}
    assert grid.swath([(0.1, 0.1, 0.0)], "rect", 0.1, 0.1) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_corridor_clear_runs_either_way():
    """Contract: `corridor_clear` runs the path rule from any start along any heading, so a rear corridor is the same
    rule from the rear edge, and `path_ahead` is that rule from the front edge. Hand-worked on the vacuum at base
    (0.17, 0) heading 0 (rear edge x 0): fine cells u -1 to -10 free across the corridor (v -2 to 1) and u -11 occupied
    across it give 1.0 (the near edge of u -11, 1.05 less half a cell) from `corridor_clear(front_edge(base, pi), pi,
    3, 0)`, with the keyframes of the free cells and the occupied one as evidence; started at the base it would give
    1.17. And `path_ahead` is the wrapper: the same answer from the front edge on each of three hand scenes, and 1.2
    ahead of a free floor with one occupied cell at u 12. Regression caught: the rear corridor started at the base or
    the camera, or `path_ahead` and `corridor_clear` drifting into two rules."""
    base = (0.17, 0.0)
    rear = grid.front_edge(VACUUM, base, math.pi)
    assert rear == pytest.approx((0.0, 0.0)) and grid.front_edge(VACUUM, base, 0.0) == pytest.approx((0.34, 0.0))
    behind = {(u, v): ("free", 1) for u in range(-10, 0) for v in range(-3, 3)}
    behind |= {(-11, v): ("occupied", 2) for v in range(-3, 3)}
    assert grid.corridor_clear(behind, VACUUM, rear, math.pi, 3.0, 0.0) == ({"type": "number", "value": 1.0}, [1, 2])
    assert grid.corridor_clear(behind, VACUUM, base, math.pi, 3.0, 0.0)[0] == {"type": "number", "value": 1.17}
    ahead = {(u, v): ("free", 3) for u in range(-3, 30) for v in range(-3, 3)} | {(12, 0): ("occupied", 4)}
    for scene, (x, y, yaw) in ((behind, (0.17, 0.0, math.pi)), (ahead, (*FRONT_EDGE_AT_ZERO, 0.0)),
                               ({}, (0.0, 0.0, 1.0))):
        assert grid.path_ahead(scene, VACUUM, (x, y), yaw, 3.0) == grid.corridor_clear(
            scene, VACUUM, grid.front_edge(VACUUM, (x, y), yaw), yaw, 3.0, blind_m(VACUUM))
    assert grid.path_ahead(ahead, VACUUM, FRONT_EDGE_AT_ZERO, 0.0, 3.0) == ({"type": "number", "value": 1.2}, [3, 4])


def test_place_free_by_hand():
    """Contract: a place is free when, of the fine cells whose centres lie within the radius of its centre, at least 80%
    are `free` and none is `occupied`; a cell nobody observed counts against it, one outside the radius not at all.
    Hand-worked for the disc of radius 0.2 m at the origin, whose 12 cells are u -2 to 1 by v -2 to 1 less the four
    corners (a corner's centre is 0.212 away): 10 free and 2 absent is free (0.83); one occupied is not; 9 free and 3
    absent is not (0.75), unless the share asked is 0.75; an occupied corner outside the disc changes nothing; a place
    with no cell centre in reach, or none observed, is not free. Regression caught: a place called empty that was not
    seen empty (truth rule 2), or an obstacle outside the radius making one occupied."""
    corners = {(-2, -2), (-2, 1), (1, -2), (1, 1)}
    cells = [(u, v) for u in range(-2, 2) for v in range(-2, 2) if (u, v) not in corners]
    assert len(cells) == 12

    def states(free, occupied=0, extra=()):
        out = {c: ("free", 1) for c in cells[:free]}
        out |= {c: ("occupied", 1) for c in cells[free:free + occupied]}
        return out | dict(extra)

    here = (0.0, 0.0)
    assert grid.place_free(states(12), here, 0.2) and grid.place_free(states(10), here, 0.2)
    assert not grid.place_free(states(10, occupied=1), here, 0.2)
    assert not grid.place_free(states(9), here, 0.2) and grid.place_free(states(9), here, 0.2, share=0.75)
    assert grid.place_free(states(12, extra={(-2, -2): ("occupied", 1)}), here, 0.2)
    assert not grid.place_free({}, here, 0.2) and not grid.place_free(states(12), (50.0, 50.0), 0.2)
    assert not grid.place_free(states(12), here, 0.02)
