"""The `thin` cells in the forms they travel in (`cell_weights`, FD-0.11 §3.7): as a score takes them, as a results
document stores them and as a document of an earlier build wrote them. Hand tables."""

from __future__ import annotations

import pytest

from ability_kit import scored
from stbench.cell_weights import thin_from, thin_weights, weight_of
from stbench.score_stats import aggregate

CELL = ("helper/1", "clearance", "now")


def test_a_set_of_cells_weighs_zero_each_and_a_mapping_is_taken_as_given():
    """Contract: a score takes `thin` as `{cell: weight}`, or as a set of cells, which is the rule of 0.6.0 to 0.10.0
    (counted in no mean) and so weighs 0 each. Hand cases: a set of two cells is two zeros; a mapping with 1/2 and 3/20
    stays as it is. Regression caught: a set read as full weight, which would count every cell the old rule dropped."""
    other = ("amr/1", "memory", "memory")
    assert thin_weights(frozenset({CELL, other})) == {CELL: 0.0, other: 0.0}
    assert thin_weights({CELL: 0.5, ("amr/1", "memory"): 0.15}) == {CELL: 0.5, ("amr/1", "memory"): 0.15}


def test_a_cells_weight_is_its_entry_then_its_two_name_entry_then_one():
    """Contract: the weight of a (job, ability, scope) cell is its own entry, else the (job, ability) entry that is thin
    in both columns, else 1; the own entry wins. Hand cases: the cell at 1/2 with a pair at 1/4 is 1/2; the same job and
    ability in the other column is 1/4; another ability is 1. Regression caught: a pair entry overriding a cell's own,
    or a cell with no entry weighing 0."""
    thin = {CELL: 0.5, ("helper/1", "clearance"): 0.25}
    assert weight_of(thin, "helper/1", "clearance", "now") == 0.5
    assert weight_of(thin, "helper/1", "clearance", "memory") == 0.25
    assert weight_of(thin, "helper/1", "distance", "now") == 1.0 and weight_of(thin, "amr/1", "clearance", "now") == 1.0


def test_the_thin_list_a_results_document_stores_reads_back_and_an_old_one_weighs_zero():
    """Contract: an aggregate lists its thin cells as `[job, ability, scope, weight]`, `thin_from` reads that back to
    what was given, and a list of an earlier build, which names the cell alone (three names, or two for both columns),
    reads as weight 0, the rule it was made under. Hand cases: 1/2 and 3/20 round-trip through `aggregate`; `["h", "a",
    "now"]` is 0 and `["h", "a"]` is 0. Regression caught: a stored weight lost on the way to `eval compare` (the
    comparison would re-score on other cells than the run), or an old document read at full weight."""
    given = {CELL: 0.5, ("amr/1", "change", "memory"): 0.15}
    rows = [*scored("clear_path", [1, 0], scope="now"), *scored("object_was", [1], job="amr/1")]
    stored = aggregate(rows, n_boot=20, thin=given)["thin"]
    assert stored == [["amr/1", "change", "memory", 0.15], ["helper/1", "clearance", "now", 0.5]]
    assert thin_from(stored) == pytest.approx(given)
    assert thin_from([["h", "a", "now"], ["h", "b"]]) == {("h", "a", "now"): 0.0, ("h", "b"): 0.0}
