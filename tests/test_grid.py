"""The answer grid's conventions (`grid.py`, plan §2.8): cells, fine cells and the corridor, all by hand arithmetic."""

from __future__ import annotations

import math

import pytest

from stbench.grid import cell, corridor, fine_cell, fine_cells_of


def test_cells_are_half_open_squares():
    """Contract: cell (i, j) is [0.5i, 0.5i + 0.5) x [0.5j, 0.5j + 0.5) and fine cell (u, v) is [0.1u, 0.1u + 0.1) x
    [0.1v, 0.1v + 0.1), negative coordinates included; a cell holds exactly 25 fine cells, row by row from its
    south-west corner. Hand-worked: (0.49, 0.5) is cell (0, 1); (-0.01, 0) is (-1, 0); (0.3, 0.7) is fine cell (3, 7);
    (-0.3, 0) is (-3, 0); cell (-1, 0) spans x -0.5 to 0, so u -5 to -1, and y 0 to 0.5, so v 0 to 4. Regression
    caught: cells shifted at negative coordinates (the downtown's are negative), by truncating toward zero instead of
    flooring."""
    assert cell(0.49, 0.5) == (0, 1)
    assert cell(-0.01, 0) == (-1, 0)
    assert cell(-0.5, -0.5) == (-1, -1)
    assert fine_cell(0.3, 0.7) == (3, 7)
    assert fine_cell(-0.3, 0) == (-3, 0)
    assert fine_cell(-0.05, 0.05) == (-1, 0)
    fine = fine_cells_of((-1, 0))
    assert len(fine) == 25 and set(fine) == {(u, v) for u in range(-5, 0) for v in range(0, 5)}
    assert fine_cells_of((0, 0))[:6] == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1)]
    assert fine_cells_of((2, 3))[-1] == (14, 19)


def test_a_decimal_on_a_boundary_lands_in_the_cell_it_opens():
    """Contract: a decimal that sits on a cell edge opens the cell above it, whatever the float division gives: 0.3 is
    fine cell 3 (0.3 / 0.1 is 2.9999999999999996 in floats), 0.6 is 6, and every decimal k / 10 from -3.0 to 3.0 is fine
    cell k. Regression caught: `floor(x / 0.1)` without the tolerance, which puts 0.3, 0.6 and a dozen other decimals in
    the cell below, so the gold and a hand-drawn answer disagree on the very walls rooms are built on."""
    assert fine_cell(0.3, 0.6) == (3, 6)
    off = [k for k in range(-30, 31) if fine_cell(float(f"{k}e-1"), float(f"{k}e-1")) != (k, k)]
    assert off == []
    assert [cell(0.5 * k, 0.5 * k) for k in (-2, -1, 0, 1, 2)] == [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]


def test_corridor_by_hand():
    """Contract: `corridor(start, yaw, width, length)` lists the fine cells whose centres lie in the rectangle running
    `length` from `start` along `yaw` and `width` across it, each with its centre's distance along the axis, sorted by
    that distance, then u, then v. Hand-worked from the origin, width 0.34 (half 0.17), length 1.0: at yaw 0 the centres
    x = 0.1u + 0.05 run u 0 to 9 (0.05 to 0.95) and y = 0.1v + 0.05 is within 0.17 for v -2 to 1 (-0.15 to 0.15), so 40
    cells at distance 0.05 + 0.1u; at yaw pi/2 the axis turns to +y, so u -2 to 1 by v 0 to 9 at 0.05 + 0.1v; at yaw pi
    it points to -x, so u -10 to -1 by v -2 to 1 at -(0.1u + 0.05), nearest first. Regression caught: a corridor that
    disagrees between the gold and the baselines' mapper, or one whose cells swap order when the yaw's float noise
    moves two equal distances apart."""
    east = corridor((0, 0), 0, 0.34, 1.0)
    assert [(u, v) for u, v, _ in east] == [(u, v) for u in range(10) for v in range(-2, 2)]
    assert [d for *_, d in east] == pytest.approx([0.05 + 0.1 * u for u in range(10) for _ in range(4)])
    north = corridor((0, 0), math.pi / 2, 0.34, 1.0)
    assert [(u, v) for u, v, _ in north] == [(u, v) for v in range(10) for u in range(-2, 2)]
    assert [d for *_, d in north] == pytest.approx([0.05 + 0.1 * v for v in range(10) for _ in range(4)])
    west = corridor((0, 0), math.pi, 0.34, 1.0)
    assert [(u, v) for u, v, _ in west] == [(u, v) for u in range(-1, -11, -1) for v in range(-2, 2)]
    assert [d for *_, d in west] == pytest.approx([-(0.1 * u + 0.05) for u in range(-1, -11, -1) for _ in range(4)])


def test_a_diagonal_corridor_and_a_moved_start_by_hand():
    """Contract: the corridor turns with the yaw and moves with its start. Hand-worked at yaw pi/4, width 0.1, length
    0.5 from the origin: along = (cx + cy) / sqrt 2 and across = (cy - cx) / sqrt 2 for a centre (cx, cy); across
    within 0.05 forces cx = cy (a neighbour is 0.1 / sqrt 2 = 0.0707 off), and along at most 0.5 allows cx = 0.05, 0.15,
    0.25, 0.35 (0.35 sqrt 2 = 0.495), so fine cells (0, 0), (1, 1), (2, 2), (3, 3) at distance sqrt 2 times each cx;
    from (1.0, 2.0) the same cells shift by (10, 20). Regression caught: a rotation that is right on the axes and
    wrong between them, or a start that is ignored."""
    distances = [math.sqrt(2) * cx for cx in (0.05, 0.15, 0.25, 0.35)]
    at_origin = corridor((0, 0), math.pi / 4, 0.1, 0.5)
    assert [(u, v) for u, v, _ in at_origin] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert [d for *_, d in at_origin] == pytest.approx(distances)
    moved = corridor((1.0, 2.0), math.pi / 4, 0.1, 0.5)
    assert [(u, v) for u, v, _ in moved] == [(10, 20), (11, 21), (12, 22), (13, 23)]
    assert [d for *_, d in moved] == pytest.approx(distances)


def test_a_centre_on_an_edge_is_inside_and_one_just_past_it_is_not():
    """Contract: a cell whose centre lies on the rectangle's edge, to 1e-9, is in the corridor. Hand-worked from the
    origin at yaw 0, width 0.1: centres y = -0.05 and 0.05 (v -1 and 0) sit on the side edges, and length 0.95 puts the
    centre x = 0.95 of u = 9 on the end edge, so u 0 to 9, 20 cells; length 0.9499 drops u = 9, leaving 18. Regression
    caught: an exact float comparison, which drops the edge cells at 0.9500000000000001 and so shortens every corridor
    whose length is a multiple of 0.05, differently on each side of the gold and the mapper."""
    on_edge = corridor((0, 0), 0, 0.1, 0.95)
    assert [(u, v) for u, v, _ in on_edge] == [(u, v) for u in range(10) for v in (-1, 0)]
    assert [(u, v) for u, v, _ in corridor((0, 0), 0, 0.1, 0.9499)] == [(u, v) for u in range(9) for v in (-1, 0)]
