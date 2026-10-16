"""The `kind` split (brief §5): the reports split their rows by the vision's four kinds (harm, help,
situation, reuse) beside the ability split, each pooled within itself, and the headline's cells do not move."""

from __future__ import annotations

import pytest

from score_kit import result_row
from stbench.score_stats import aggregate, paired_difference

# family, scope, penalised score: the harm kind holds clear_path and closest_person, help open_heading (clearance since
# 0.11.0), situation ego_now and ego_mem, reuse object_where and task_things
ROWS = [("clear_path", "now", 0.5), ("closest_person", "now", 1.0), ("open_heading", "memory", -0.5),
        ("ego_now", "now", 1.0), ("ego_mem", "memory", 0.0), ("object_where", "memory", -1.0),
        ("task_things", "memory", 0.0), ("person_mem", "memory", 0.5)]


def rows():
    return [result_row(f"c{i}", f"q{i}", score, "partial", script=f"s{i}", family=family, scope=scope,
                       job="helper/1") for i, (family, scope, score) in enumerate(ROWS)]


def test_by_kind_pools_its_own_rows_by_the_visions_four_kinds():
    """Contract (§5): `by.kind` has a group a kind, each the penalised score pooled over its rows with the count. By
    hand: harm = clear_path 0.5, closest_person 1.0 and person_mem 0.5, mean 2/3 over 3; help = open_heading -0.5 over
    1; situation = ego_now 1.0 and ego_mem 0.0, mean 0.5 over 2; reuse = object_where -1.0 and task_things 0.0, mean
    -0.5 over 2. Regression caught: a family in the wrong kind (person_mem is harm, a person to avoid), or a kind
    pooled with another's rows."""
    by = aggregate(rows(), n_boot=20)["by"]["kind"]
    assert set(by) == {"harm", "help", "situation", "reuse"}
    assert {k: (v["mean"], v["n"]) for k, v in by.items()} == {
        "harm": (pytest.approx(2 / 3), 3), "help": (pytest.approx(-0.5), 1), "situation": (pytest.approx(0.5), 2),
        "reuse": (pytest.approx(-0.5), 2)}


def test_a_kind_whose_rows_the_means_do_not_count_reports_its_mean_flagged_diagnostic():
    """Contract (0.9.0 lane H2; `open_heading` counts since 0.11.0): a kind none of whose rows the means count keeps its
    mean over all its rows and says so, as `by.ability` does: `counted` 0 and `diagnostic` true; the other kinds count
    every row. By hand (`rows`, the cells `memory` and `relevance` of the helper given as thin, each weighing 0):
    reuse is object_where -1.0 and task_things 0.0, mean -0.5 over 2, counted 0; help is open_heading -0.5 over 1,
    counted 1 now it is clearance; harm counts 3 and situation 1 (`ego_mem` is a thin `memory` row).
    Regression caught: a kind losing its mean with its counted rows (a report that hides the family), or a kind
    reported as counted when its rows are in no mean."""
    thin = frozenset({("helper/1", "memory", "memory"), ("helper/1", "relevance", "memory")})
    by = aggregate(rows(), n_boot=20, thin=thin)["by"]["kind"]
    assert by["reuse"]["mean"] == pytest.approx(-0.5)
    assert {k: (v["counted"], v["diagnostic"]) for k, v in by.items()} == {
        "harm": (3, False), "help": (1, False), "situation": (1, False), "reuse": (0, True)}


def test_the_kind_split_moves_no_headline_cell():
    """Contract (§5): the headline stays the mean of the `now` and `memory` columns, each the equal mean of its
    (ability, scope) cells, with open_heading in clearance (since 0.11.0). By hand: now = clearance (clear_path 0.5,
    closest_person 1.0: 0.75) and distance (ego_now 1.0), mean 0.875; memory = clearance (person_mem 0.5, open_heading
    -0.5: 0), memory (ego_mem 0.0, object_where -1.0: -0.5) and relevance (task_things 0.0), mean -1/6; the headline
    (0.875 - 1/6) / 2 = 17/48. The `kind` groups are not cells: the abilities' cells are the same, and help's one row is
    no cell. Regression caught: a kind that becomes a cell, so that help (one row) weighs as an ability, or
    an ability that loses a cell."""
    agg = aggregate(rows(), n_boot=20)
    assert agg["headline"]["mean"] == pytest.approx(17 / 48)
    assert agg["columns"]["now"]["mean"] == pytest.approx(0.875)
    assert agg["columns"]["memory"]["mean"] == pytest.approx(-1 / 6)
    cells = agg["by"]["ability"]["clearance"]["cells"]
    assert cells["now"]["mean"] == pytest.approx(0.75) and cells["memory"]["mean"] == pytest.approx(0.0)


def test_a_paired_difference_can_be_cut_by_kind():
    """Contract (`paired_difference`'s `where`): the derived split `kind` filters rows like `ability` does. By hand, two
    systems over the same eight rows, B scoring 0.25 less on each: kept to `harm` the pooled difference is 0.25 over 3
    questions. Regression caught: a derived split read from a row key that does not exist."""
    a = rows()
    b = [{**r, "score": r["score"] - 0.25} for r in a]
    got = paired_difference(a, b, where={"kind": "harm"}, n_boot=20)
    assert (got["mean"], got["n"]) == (pytest.approx(0.25), 3)
