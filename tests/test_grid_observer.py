"""The free-space rule over a depth image's points, `grid_observer.Observer` (what `grid.observe` and the mapper call),
against the rule worked out from its definition, a point at a time, in plain Python."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stbench import grid, grid_observer
from stbench.grid_observer import CellFilter, Observer

BAND = (0.023, 0.095)   # the vacuum's, as `grid.band` gives it


def by_definition(pts, floor_z, band):
    """`grid.observe`'s rule, a point at a time: each point above the floor by at most the band's top joins its fine
    cell (the one holding its 0.02 m sub-cell, `floor(x / 0.02 + 1e-9)` along each axis); a point at most the climb
    above the floor is a floor point, any other an obstacle on its sub-cell; a cell with an obstacle is occupied
    (naming its sub-cells), else free with `MIN_FLOOR_PX` floor points, else absent. The cells in order by (u, v)."""
    climb, top = band
    found: dict[tuple[int, int], list] = {}
    for x, y, z in pts.tolist():
        height = z - floor_z
        if height > top + grid.EPS:
            continue
        a, b = math.floor(x / grid.SUB_M + grid.EPS), math.floor(y / grid.SUB_M + grid.EPS)
        entry = found.setdefault((a // grid.SUB_PER_FINE, b // grid.SUB_PER_FINE), [0, set()])
        if height <= climb + grid.EPS:
            entry[0] += 1
        else:
            entry[1].add((a, b))
    out = {}
    for c in sorted(found):
        floor_points, subs = found[c]
        if subs:
            out[c] = grid.Occupied(subs)
        elif floor_points >= grid.MIN_FLOOR_PX:
            out[c] = "free"
    return out


def same(got, want):
    """Equal as dicts, in the same order, with the same sub-cells in each occupied cell."""
    assert list(got) == list(want)
    assert all(got[c] == want[c] and type(got[c]) is type(want[c]) for c in want)
    assert all(got[c].subs == want[c].subs for c in want if isinstance(want[c], grid.Occupied))


def cloud(seed: int, n: int) -> np.ndarray:
    """`n` points over a few metres, negative coordinates included, on a grid of centimetres and millimetres so that
    many sit on a cell, sub-cell or band edge (x = 0.3 is on a cell's edge, z = 0.023 on the climb's)."""
    rng = np.random.default_rng(seed)
    x, y = np.round(rng.uniform(-1.2, 1.2, (2, n)), 2)
    z = np.round(rng.choice([0.0, 0.023, 0.095, 0.096, 0.2], n) + rng.integers(-1, 2, n) * 0.001, 3)
    return np.column_stack([x, y, z])


@pytest.mark.parametrize("seed", range(8))
def test_the_observer_gives_the_definitions_cells(seed):
    """Contract: `Observer.observe` (and `grid.observe` over it) is the rule of `grid.observe`'s definition: the same
    cells with the same states, the occupied ones naming the same sub-cells, in order by (u, v), whatever the cloud and
    the floor, with the band's edges, the cells' edges and the sub-cells' on points. Regression caught: a speed-up of
    the rule (counting floor points in an array, working a cloud's cells out once for several bands) that moves a
    point to the next cell, drops a cell with a lone obstacle point, or changes the order the dict lists them in."""
    pts = cloud(seed, [1, 7, 60, 400, 3000][seed % 5])
    for floor_z in (0.0, 0.002, -0.1):
        want = by_definition(pts, floor_z, BAND)
        same(Observer(pts).observe(floor_z, BAND), want)
        same(grid.observe(pts, floor_z, BAND), want)


def test_a_cloud_with_nothing_in_the_band_has_no_cells():
    """Contract: no points, or only points over the band, observe nothing (`{}`), and a cloud of obstacle points alone
    observes occupied cells and no free one. Regression caught: a reduction over an empty selection raising, or a cell
    with no floor point read as free."""
    assert Observer(np.zeros((0, 3))).observe(0.0, BAND) == {}
    assert Observer(np.array([(0.5, 0.5, 0.3)] * 9)).observe(0.0, BAND) == {}
    only = Observer(np.array([(0.5, 0.5, 0.05)])).observe(0.0, BAND)
    assert list(only) == [(5, 5)] and only[5, 5] == "occupied" and only[5, 5].subs == {(25, 25)}


def test_keeping_cells_gives_the_same_states_as_the_whole_and_filtering_it():
    """Contract: `observe(keep=...)`, and `within(...)` the cells of a larger set then `observe(keep=...)`, give the
    whole dict's cells that are in `keep`, with the whole dict's states, in its order: a cell's state is a function of
    its own points. Cells of `keep` that no point reaches are absent, and a `keep` of no cell gives nothing. Regression
    caught: a zone band that sees a cell differently from the whole image (a point lost at the edge of the bounding
    box of the cells, a filter that tests the wrong axis)."""
    pts = cloud(3, 3000)
    whole = by_definition(pts, 0.0, BAND)
    rng = np.random.default_rng(0)
    cells = list(whole) + [(int(a), int(b)) for a, b in rng.integers(-30, 30, (40, 2))]
    keep = frozenset(c for c in cells if rng.random() < 0.4)
    want = {c: s for c, s in whole.items() if c in keep}
    assert want and len(want) < len(whole)
    seen = Observer(pts)
    same(seen.observe(0.0, BAND, keep=CellFilter(keep)), want)
    same(seen.within(CellFilter(keep | frozenset(list(whole)[::2]))).observe(0.0, BAND, keep=CellFilter(keep)), want)
    assert seen.observe(0.0, BAND, keep=CellFilter(())) == {}
    assert seen.within(CellFilter(())).observe(0.0, BAND) == {}


@pytest.mark.parametrize("limit", [0, grid_observer.DENSE_MAX])
def test_counting_in_an_array_and_by_sorting_agree(monkeypatch, limit):
    """Contract: `_distinct` counts a small bounding box of cells into an array and sorts a large one; both give the
    distinct pairs in order with the same counts, and so the same cells. Hand case: the pairs (3, -2) twice, (-1, 5),
    (3, -3), (-1, 5), (-1, 5): (-1, 5) three times, (3, -3) once, (3, -2) twice, in that order. Regression caught:
    one of the two paths mis-decoding a pair (the array's key, or negative cells)."""
    monkeypatch.setattr(grid_observer, "DENSE_MAX", limit)
    u, v, n = grid_observer._distinct(np.array([3, -1, 3, -1, 3, -1]), np.array([-2, 5, -3, 5, -2, 5]))
    assert list(zip(u.tolist(), v.tolist(), n.tolist(), strict=True)) == [(-1, 5, 3), (3, -3, 1), (3, -2, 2)]
    pts = cloud(5, 2000)
    same(Observer(pts).observe(0.0, BAND), by_definition(pts, 0.0, BAND))
