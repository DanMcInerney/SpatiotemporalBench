"""Scoring by ability (0.6.0, `score_stats`): the headline weighs the abilities in each column equally, a diagnostic
ability and a (job, ability, column) cell given as thin (a set of cells, each weighing 0: the rule of 0.6.0 to 0.10.0;
`test_score_cells.py` has the part weights) are shown and counted in no mean, and one resample serves every interval.
Expected values are worked by hand in each test."""

from __future__ import annotations

import pytest

from ability_kit import r, results_doc, scored
from stbench import eval_results
from stbench.score_stats import OPTIONAL_ROW_KEYS, ROW_KEYS, aggregate, paired_difference

THIN = frozenset({("amr/1", "change", "memory")})


def mixed() -> list[dict]:
    """Seven abilities-worth of rows: `ego_now` [1, 1] (distance, now); `mover_next` [-1, -1] (prediction, a diagnostic,
    now); `object_where` [1, 0, 0] (memory); `object_was` on the helper [1, 1] and on the AMR [-1] (change)."""
    return [*scored("ego_now", [1, 1]), *scored("mover_next", [-1, -1]), *scored("object_where", [1, 0, 0]),
            *scored("object_was", [1, 1]), *scored("object_was", [-1], job="amr/1")]


def test_the_headline_weighs_abilities_equally_in_a_column_and_the_columns_equally():
    """Contract (VISION rule 4, FD-0.6.0 §9): `now` and `memory` are equal; inside a column each ability weighs the same
    whatever its questions, each pooled over its own rows. By hand: `ego_now` [1, 0, 0, 0] is distance, 1/4;
    `object_where`
    [1, 1, 1] with `ego_mem` [0] is the one ability memory, 3/4; `object_was` [1] is change, 1; `task_things`
    [0, 0, 0, 0] is relevance, 0. The `memory` column is (3/4 + 1 + 0)/3 = 7/12 and the headline (1/4 + 7/12)/2 = 5/12;
    beside it the abilities' equal mean (1/4 + 3/4 + 1 + 0)/4 = 1/2 and the 13 counted rows pooled, (1 + 3 + 1)/13 =
    5/13.
    Regression caught: a headline that pools each column's questions (it gives (1/4 + 4/9)/2 = 0.347), a
    mean of family means, or the equal mean standing where the headline should."""
    rows = [*scored("ego_now", [1, 0, 0, 0]), *scored("object_where", [1, 1, 1]), *scored("ego_mem", [0]),
            *scored("object_was", [1]), *scored("task_things", [0, 0, 0, 0])]
    agg = aggregate(rows, n_boot=50)
    assert agg["headline"]["mean"] == pytest.approx(5 / 12)
    assert {k: c["mean"] for k, c in agg["columns"].items()} == pytest.approx({"now": 1 / 4, "memory": 7 / 12})
    assert {k: c["n"] for k, c in agg["columns"].items()} == {"now": 4, "memory": 9}
    assert agg["abilities_mean"]["mean"] == pytest.approx(1 / 2) and agg["pooled"]["mean"] == pytest.approx(5 / 13)
    assert {k: v["mean"] for k, v in agg["by"]["ability"].items()} == pytest.approx(
        {"distance": 1 / 4, "memory": 3 / 4, "change": 1.0, "relevance": 0.0})


def test_a_diagnostic_and_an_under_20_pair_are_listed_and_counted_in_no_mean():
    """Contract (FD-0.6.0 §10): `mover_next` (prediction, a diagnostic) and the AMR's change questions (the pair is
    thin) appear in `by.ability` with their counts and flags and are in no mean; the helper's change questions count. By
    hand (`mixed`, thin = the AMR's change): distance 1; memory 1/3; change [1, 1] = 1, so the `memory` column is
    (1/3 + 1)/2 = 2/3 and the headline (1 + 2/3)/2 = 5/6; the equal mean of the three counted abilities (1 + 1/3 +
    1)/3 =
    7/9; the 7 counted rows pooled (2 + 1 + 2)/7 = 5/7. In `by.ability` prediction has 2 questions, none counted,
    mean -1; change 3, 2 counted, mean (1 + 1 - 1)/3 = 1/3. Without the thin pair, change is 1/3 and the headline
    (1 + 1/3)/2 = 2/3. Regression caught: a thin ability or a diagnostic moving the headline, or the pair dropped for
    every job instead of the one."""
    agg = aggregate(mixed(), n_boot=50, thin=THIN)
    assert agg["headline"]["mean"] == pytest.approx(5 / 6)
    assert agg["abilities_mean"]["mean"] == pytest.approx(7 / 9) and agg["pooled"]["mean"] == pytest.approx(5 / 7)
    assert {k: c["n"] for k, c in agg["columns"].items()} == {"now": 2, "memory": 5}
    ability = agg["by"]["ability"]
    assert (ability["prediction"]["n"], ability["prediction"]["counted"], ability["prediction"]["diagnostic"]) == (
        2, 0, True)
    assert ability["prediction"]["mean"] == pytest.approx(-1)
    assert (ability["change"]["n"], ability["change"]["counted"], ability["change"]["diagnostic"]) == (3, 2, False)
    assert ability["change"]["mean"] == pytest.approx(1 / 3)
    assert ability["memory"]["scope"] == "memory" and ability["distance"]["scope"] == "now"
    assert agg["thin"] == [["amr/1", "change", "memory", 0.0]] and agg["thin_ignored"] is False
    assert aggregate(mixed(), n_boot=50)["headline"]["mean"] == pytest.approx(2 / 3)


def test_rows_that_are_all_diagnostic_give_a_null_headline_not_a_relaxed_rule():
    """Contract: whether the under-20 rule applies at all is the dataset's decision (`ability_counts.thin_rule`), never
    the rows': a run whose every row is in a thin pair has no counted ability, so its headline and columns are null and
    its abilities are all diagnostics, while the same pairs passed with `thin_ignored` (the dataset is too small and
    the caller applies no pair) score the rows. By hand: `ego_now` [1, 0] (1/2) and `object_where` [1, 1] (1) with both
    pairs thin: no headline; with no pair applied: (1/2 + 1)/2 = 3/4 and `thin_ignored` recorded. Regression caught:
    a rule relaxed from a subset's rows (a vacuum-only run and the full run then disagree), or a crash on null."""
    rows = [*scored("ego_now", [1, 0]), *scored("object_where", [1, 1])]
    thin = frozenset({("helper/1", "distance", "now"), ("helper/1", "memory", "memory")})
    agg = aggregate(rows, n_boot=50, thin=thin)
    assert agg["headline"] == {"mean": None, "ci": None} and agg["columns"] == {} and agg["above_floor"]["mean"] is None
    assert all(v["diagnostic"] and v["counted"] == 0 for v in agg["by"]["ability"].values())
    relaxed = aggregate(rows, n_boot=50, thin_ignored=True)
    assert relaxed["headline"]["mean"] == pytest.approx(3 / 4) and relaxed["thin_ignored"] is True
    assert relaxed["thin"] == []


def test_an_abilitys_counted_mean_leaves_out_its_thin_rows():
    """Contract: `by.ability` gives the mean over every row of the ability and, beside it, `counted_mean` over the rows
    the means count. By hand (`mixed`, the AMR's change thin): change has [1, 1] on the helper and [-1] on the AMR, so
    the mean is 1/3 over 3 and the counted mean 1 over 2; prediction, a diagnostic, has none counted and no counted
    mean.
    Regression caught: a thin row's score shown in an ability's line as if it were in the means."""
    agg = aggregate(mixed(), n_boot=50, thin=THIN)["by"]["ability"]
    assert (agg["change"]["mean"], agg["change"]["counted_mean"]) == pytest.approx((1 / 3, 1.0))
    assert agg["prediction"]["counted_mean"] is None and agg["prediction"]["counted"] == 0
    assert agg["distance"]["counted_mean"] == pytest.approx(agg["distance"]["mean"])


def test_one_resample_serves_the_headline_and_the_columns():
    """Contract: every interval comes from the same script weights. Two scripts, anti-correlated: `ego_now` scores 1 on
    script s0 and -1 on s1, `object_where` -1 on s0 and 1 on s1. In a resample with weights (w0, w1) the `now` column
    is (w0 - w1)/(w0 + w1) and the `memory` column its negative, so the headline is 0 in every resample: interval
    (0, 0), while each column's is (-1, 1) (the draws (2, 0) and (0, 2) each have probability 1/4). Regression caught: a
    column resampled apart from the other, whose mean would then move the headline's interval."""
    rows = [*scored("ego_now", [1], case="c0", script="s0"), *scored("ego_now", [-1], case="c1", script="s1"),
            *scored("object_where", [-1], case="c0", script="s0"), *scored("object_where", [1], case="c1", script="s1")]
    agg = aggregate(rows)
    assert agg["headline"]["mean"] == 0 and agg["headline"]["ci"] == (0.0, 0.0)
    assert agg["columns"]["now"]["ci"] == pytest.approx((-1.0, 1.0))
    assert agg["columns"]["memory"]["ci"] == pytest.approx((-1.0, 1.0))


def test_a_resample_that_lacks_an_ability_leaves_it_out_of_its_column():
    """Contract: an ability with no row in a resample is dropped from its column there; the others stand. `object_where`
    has one row on script s0 (score 1); `object_was` has s0 [0] and s1 [-1]. Weights (1, 1) (the point): (1 +
    (-1/2))/2 =
    1/4; (2, 0): (1 + 0)/2 = 1/2; (0, 2): `object_where` is absent and the column is `object_was`, -1. With
    probabilities 1/4, 1/2, 1/4 the interval is (-1, 1/2). Regression caught: a resample without an ability dropped
    whole (which gives (1/4, 1/2)), or the missing ability counted as 0."""
    rows = [*scored("object_where", [1], case="c0", script="s0"), *scored("object_was", [0], case="c0", script="s0"),
            *scored("object_was", [-1], case="c1", script="s1")]
    agg = aggregate(rows)
    assert agg["headline"]["mean"] == pytest.approx(1 / 4)
    assert agg["headline"]["ci"] == pytest.approx((-1.0, 0.5))


def test_the_paired_headline_difference_is_the_difference_of_the_two_headlines():
    """Contract: `paired_difference(headline=True)` is A's headline less B's over the shared questions, with the same
    thin pairs. By hand (`mixed`, thin the AMR's change questions) A is 5/6; B scores `ego_now` [0, 1], `object_where`
    [1, 1, 0], the helper's change [1, 0] and the AMR's [-1]: `now` 1/2, `memory` (2/3 + 1/2)/2 = 7/12, headline 13/24.
    The difference is 20/24 - 13/24 = 7/24. Without `thin`, the AMR's question joins the change ability and the number
    moves. By ability, `memory` alone: A [1, 0, 0] - B [1, 1, 0] pools to -1/3. Regression caught: a paired headline
    that
    forgets the under-20 rule, or one that is not the difference of the headlines."""
    other = [*scored("ego_now", [0, 1]), *scored("mover_next", [-1, -1]), *scored("object_where", [1, 1, 0]),
             *scored("object_was", [1, 0]), *scored("object_was", [-1], job="amr/1")]
    kw = {"headline": True, "n_boot": 50}
    assert paired_difference(mixed(), other, thin=THIN, **kw)["mean"] == pytest.approx(7 / 24)
    assert paired_difference(mixed(), other, **kw)["mean"] != pytest.approx(7 / 24)
    assert aggregate(other, n_boot=50, thin=THIN)["headline"]["mean"] == pytest.approx(13 / 24)
    d = paired_difference(mixed(), other, where={"ability": "memory"}, n_boot=50)
    assert d["mean"] == pytest.approx(-1 / 3) and d["n"] == 3


def test_the_answer_class_is_a_split_and_a_missing_class_reads_none():
    """Contract: `by.answer_class` pools each declared class apart and files a gold that declares none under `none`. By
    hand: `yes` [1, 0] 1/2, `moved` [-1] -1, no class [1, 1] 1. Regression caught: a class split that raises on null, or
    pools the unclassed rows into a class."""
    rows = [*scored("same_place", [1, 0], answer_class="yes"), *scored("same_place", [-1], case="c2",
    answer_class="moved"),
            *scored("object_where", [1, 1], case="c3")]
    by = aggregate(rows, n_boot=20)["by"]["answer_class"]
    assert {k: v["mean"] for k, v in by.items()} == pytest.approx({"yes": 1 / 2, "moved": -1.0, "none": 1.0})


def test_a_row_of_an_older_run_without_range_m_reads_null_and_scores_the_same():
    """Contract (0.6.0 rows): `range_m` is a row key, optional like `answer_class`, so a row of a run before 0.6.0 reads
    null and the aggregate of such rows is the aggregate of the same rows with the key null; a results document
    holds the key as a number or null and refuses another type. Regression caught: a run kept from `m8` becoming
    unreadable, or a document that accepts a range given as text."""
    assert "range_m" in ROW_KEYS and "range_m" in OPTIONAL_ROW_KEYS
    rows = [r("c1", "a", 1.0, "ego_now", range_m=4.5), r("c2", "b", 0.0, "ego_now"), r("c3", "c", 1.0, "object_where")]
    older = [{k: v for k, v in x.items() if k != "range_m"} for x in rows]
    assert aggregate(older, n_boot=20)["headline"]["mean"] == aggregate(rows, n_boot=20)["headline"]["mean"]
    assert eval_results.problems(results_doc(rows)) == []
    bad = results_doc([{**rows[0], "range_m": "far"}, *rows[1:]])
    assert any("range_m" in p for p in eval_results.problems(bad))
