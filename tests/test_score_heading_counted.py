"""`open_heading` counts in the clearance ability (FD-0.11 §3.3; before 0.11.0 it was the diagnostic ability `heading`,
kept out of every mean while its credit was a cliff). Hand rows; every number is worked from the scores written here,
never from the code under test."""

from __future__ import annotations

import pytest

from ability_kit import r
from stbench.score_stats import aggregate


def rows(*, with_heading: bool) -> list[dict]:
    """`clear_path` (clearance, `now`) scoring 1.0 and 0.5, and `open_heading` scoring -1.0 and -1.0 in `now` and -0.5
    in `memory` (it is asked from the present view and from memory)."""
    found = [r("c1", "q1", 1.0, "clear_path", scope="now", script="s1"),
             r("c1", "q2", 0.5, "clear_path", scope="now", script="s1")]
    if with_heading:
        found += [r("c1", "q3", -1.0, "open_heading", scope="now", script="s2"),
                  r("c1", "q4", -1.0, "open_heading", scope="now", script="s2"),
                  r("c1", "q5", -0.5, "open_heading", scope="memory", script="s2")]
    return found


def test_open_heading_rows_are_clearance_rows_in_the_cell_of_their_scope():
    """Contract (FD-0.11 §3.3): rows of `open_heading` are rows of the clearance ability, pooled with `clear_path`'s in
    the `now` cell and opening the `memory` cell, and they move the headline. By hand, `clear_path` alone is a `now`
    clearance cell of (1.0 + 0.5) / 2 = 0.75, the only column, so the headline is 0.75; with `open_heading` (-1.0, -1.0
    in `now`, -0.5 in `memory`) the `now` cell is (1.0 + 0.5 - 1.0 - 1.0) / 4 = -0.125 over 4 rows, the `memory` cell is
    -0.5 over 1, and the headline is (-0.125 - 0.5) / 2 = -0.3125. Regression caught: `open_heading` back in a
    diagnostic ability of its own (the headline then stays 0.75 and the heading rows weigh nothing), or kept out of
    clearance's `memory` cell."""
    alone = aggregate(rows(with_heading=False), n_boot=20)
    both = aggregate(rows(with_heading=True), n_boot=20)
    assert alone["headline"]["mean"] == pytest.approx(0.75)
    assert both["headline"]["mean"] == pytest.approx(-0.3125)
    assert {k: c["mean"] for k, c in both["columns"].items()} == pytest.approx({"now": -0.125, "memory": -0.5})
    assert {k: c["n"] for k, c in both["columns"].items()} == {"now": 4, "memory": 1}
    assert "heading" not in both["by"]["ability"] and both["by"]["ability"]["clearance"]["n"] == 5


def test_the_help_kind_reports_open_heading_s_mean_and_the_means_count_it():
    """Contract: `by.kind` pools every row of a kind, as `by.family` does, so `help` (the `open_heading` family alone)
    reports its mean, -5/6 over 3 rows, and says that the means count all of them (`counted` 3, not `diagnostic`),
    while `harm` (`clear_path`) counts both of its rows. Regression caught: the help kind flagged as counted in no
    mean while its rows are in the clearance cells, or dropped from the kind split."""
    kinds = aggregate(rows(with_heading=True), n_boot=20)["by"]["kind"]
    assert kinds["help"]["mean"] == pytest.approx(-5 / 6) and kinds["help"]["n"] == 3
    assert (kinds["help"]["counted"], kinds["help"]["diagnostic"]) == (3, False)
    assert (kinds["harm"]["n"], kinds["harm"]["counted"], kinds["harm"]["diagnostic"]) == (2, 2, False)
