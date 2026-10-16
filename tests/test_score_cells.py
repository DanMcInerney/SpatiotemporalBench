"""Scoring by cell (FD-task-questions review §5): a column is the equal mean of its cells, a cell an (ability, scope)
pair, so clearance, asked in both scopes (the corridor from the present view, a person from memory), has a cell
in each column. Every expected value is worked by hand from the scores written in the test."""

from __future__ import annotations

import pytest

from ability_kit import scored
from stbench.score_stats import aggregate


def rows() -> list[dict]:
    """`ego_now` [1, 0, 0, 0] (distance, `now`); `clear_path` [1, 1] (clearance, `now`); `closest_person` [1, 0, 0]
    (clearance, `memory`); `object_where` [1] x 6 (memory, `memory`)."""
    return [*scored("ego_now", [1, 0, 0, 0], scope="now"), *scored("clear_path", [1, 1], scope="now"),
            *scored("closest_person", [1, 0, 0], scope="memory"), *scored("object_where", [1] * 6, scope="memory")]


def test_an_ability_with_rows_in_both_scopes_counts_in_each_column_as_its_own_cell():
    """Contract: clearance's corridor rows are a cell of the `now` column and its person rows a cell of the
    `memory` column, each pooled over its own rows and each weighing as much as any other cell of its column, whatever
    its count. By hand: `now` holds distance 1/4 (4 rows) and clearance 1 (2 rows), their equal mean 5/8; `memory`
    holds memory 1 (6 rows) and clearance 1/3 (3 rows), their equal mean 2/3; the headline is (5/8 + 2/3)/2 = 31/48 =
    0.646. The four cells weigh 1/4 each. Under a rule that counted an ability once, in `memory`, pooled
    over its five rows (3/5), `now` would be 1/4 and `memory` (1 + 3/5)/2 = 4/5, the headline 21/40 = 0.525; pooling the
    rows of each column would give `now` (1 + 2)/6 = 1/2. Regression caught: clearance's corridor rows moved out of the
    `now` column, a cell weighed by its count, or the two cells of an ability pooled."""
    agg = aggregate(rows(), n_boot=50)
    assert agg["headline"]["mean"] == pytest.approx(31 / 48)
    assert {k: (c["mean"], c["n"]) for k, c in agg["columns"].items()} == pytest.approx(
        {"now": (5 / 8, 6), "memory": (2 / 3, 9)})
    assert agg["abilities_mean"]["mean"] == pytest.approx(31 / 48) and agg["pooled"]["mean"] == pytest.approx(2 / 3)
    clearance = agg["by"]["ability"]["clearance"]
    assert clearance["scope"] == "memory/now" and clearance["n"] == 5
    assert {s: (c["n"], c["counted"], c["mean"], c["weight"]) for s, c in clearance["cells"].items()} == pytest.approx(
        {"now": (2, 2, 1.0, 1 / 4), "memory": (3, 3, 1 / 3, 1 / 4)})
    assert agg["by"]["ability"]["distance"]["cells"]["now"]["weight"] == pytest.approx(1 / 4)


def test_a_thin_cell_leaves_its_column_and_the_weights_follow():
    """Contract (the 20-line of 0.6.0 to 0.10.0, per (job, ability, column), as a set of cells each weighing 0): a cell
    in `thin` is shown and counted in no mean, and its ability's other cell still counts. By hand (`rows`, the helper's
    clearance corridor cell thin): `now` is distance alone, 1/4; `memory` is unchanged, 2/3; the headline (1/4 + 2/3)/2
    = 11/24. The weights: distance carries all of `now`, 1/2; the two `memory` cells 1/4 each; the thin cell 0, its mean
    and count still reported (1, 2 rows, none counted). A thin entry of two names is thin in both columns: distance thin
    that way leaves `now` with clearance alone, 1, and the headline (1 + 2/3)/2 = 5/6. Regression caught: a thin cell
    counted through its ability's other cell, a weight that ignores the cell left out, or a two-name entry that is no
    longer read."""
    agg = aggregate(rows(), n_boot=50, thin=frozenset({("helper/1", "clearance", "now")}))
    assert agg["headline"]["mean"] == pytest.approx(11 / 24)
    assert agg["thin"] == [["helper/1", "clearance", "now", 0.0]]
    cells = agg["by"]["ability"]["clearance"]["cells"]
    assert (cells["now"]["n"], cells["now"]["counted"], cells["now"]["diagnostic"], cells["now"]["weight"]) == (
        2, 0, True, 0.0)
    assert cells["now"]["mean"] == pytest.approx(1.0) and cells["memory"]["weight"] == pytest.approx(1 / 4)
    assert agg["by"]["ability"]["distance"]["cells"]["now"]["weight"] == pytest.approx(1 / 2)
    older = aggregate(rows(), n_boot=50, thin=frozenset({("helper/1", "distance")}))
    assert older["headline"]["mean"] == pytest.approx(5 / 6)


def test_a_cell_of_part_weight_pools_its_rows_by_weight_and_weighs_that_in_its_column():
    """Contract (FD-0.11 §3.7): a thin cell given a weight below 1 stays in the means at that weight, between its
    two ends (weight 0, as when it was dropped, and weight 1, whole). By hand (`rows`, the helper's clearance corridor
    cell at 1/2): `now` holds distance 1/4 (weight 1) and clearance 1 (weight 1/2), mean (1/4 + 1/2) / (3/2) = 1/2;
    `memory` is unchanged, 2/3; the headline (1/2 + 2/3) / 2 = 7/12, between 11/24 (weight 0) and 31/48 (weight 1).
    The weights in the headline: `now` and `memory` weigh half each (their cells' largest weight is 1), so distance
    1/2 * 2/3 = 1/3, the thin cell 1/2 * 1/3 = 1/6 and the `memory` cells 1/4 each. The mean of the cells is
    (1/4 + 1/2 + 1 + 1/3) / (7/2) = 25/42 and the counted rows pooled by weight (1 + 1 + 1 + 6) / (4 + 1 + 3 + 6) =
    9/14. At weight 1/100 the headline is 0.462, next to the 11/24 = 0.458 of weight 0. Regression caught: a part-weight
    cell counted whole or dropped, weighed by its question count, or a headline that jumps at the end of the range."""
    thin = {("helper/1", "clearance", "now"): 0.5}
    agg = aggregate(rows(), n_boot=50, thin=thin)
    assert agg["headline"]["mean"] == pytest.approx(7 / 12)
    assert {k: c["mean"] for k, c in agg["columns"].items()} == pytest.approx({"now": 1 / 2, "memory": 2 / 3})
    assert agg["abilities_mean"]["mean"] == pytest.approx(25 / 42) and agg["pooled"]["mean"] == pytest.approx(9 / 14)
    assert agg["thin"] == [["helper/1", "clearance", "now", 0.5]]
    cells = agg["by"]["ability"]["clearance"]["cells"]
    assert (cells["now"]["counted"], cells["now"]["diagnostic"]) == (2, False)
    assert cells["now"]["weight"] == pytest.approx(1 / 6) and cells["memory"]["weight"] == pytest.approx(1 / 4)
    assert agg["by"]["ability"]["distance"]["cells"]["now"]["weight"] == pytest.approx(1 / 3)
    near = aggregate(rows(), n_boot=50, thin={("helper/1", "clearance", "now"): 0.01})["headline"]["mean"]
    assert near == pytest.approx(0.4620, abs=1e-4)


def test_a_column_of_thin_cells_alone_weighs_the_largest_of_them_in_the_headline():
    """Contract (FD-0.11 §3.7, read for a column): a column weighs in the headline the largest weight of its cells, so
    a column whose cells are all thin is thin, as the cell is in its column, and the headline moves continuously where
    the column once vanished. By hand (`rows`, both `now` cells at 1/2): `now` is (1/2 * 1/4 + 1/2 * 1) / 1 = 5/8 and
    weighs 1/2; `memory` is 2/3 and weighs 1; the headline (1/2 * 5/8 + 2/3) / (3/2) = 47/72. The `now` cells weigh
    1/6 each in it and the `memory` ones 1/3 each. Regression caught: a lone thin cell making its column half the
    headline whole (it would be (5/8 + 2/3) / 2 = 0.646), or a column dropped when its cells are thin."""
    thin = {("helper/1", "clearance", "now"): 0.5, ("helper/1", "distance", "now"): 0.5}
    agg = aggregate(rows(), n_boot=50, thin=thin)
    assert agg["headline"]["mean"] == pytest.approx(47 / 72)
    assert agg["by"]["ability"]["clearance"]["cells"]["now"]["weight"] == pytest.approx(1 / 6)
    assert agg["by"]["ability"]["distance"]["cells"]["now"]["weight"] == pytest.approx(1 / 6)
    assert agg["by"]["ability"]["clearance"]["cells"]["memory"]["weight"] == pytest.approx(1 / 3)
