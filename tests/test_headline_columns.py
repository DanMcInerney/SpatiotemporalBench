"""The headline weighs the `now` and `memory` columns equally (VISION rule 4): the mean of the two columns,
each the penalised score pooled over its own rows, whatever their counts; one column when the rows hold one scope; the
interval is the mean's own, from one resample of scripts for both columns; `above_floor` and `paired_difference(
headline=True)` take the same statistic. The splits (`by`) and `pooled` stay pooled, and the gates read those. Rows are
`score_kit.result_row`s, which default to scope `memory` and a script per case; every expected value is worked out by
hand in the test."""

from __future__ import annotations

import pytest

from eval_kit import cli, hand_dataset, results
from score_kit import result_row
from stbench.eval_report import HEADLINE_RULE, render
from stbench.score_stats import aggregate, paired_difference, pooled


def make(case, qid, score, **kw):
    """A row; the outcome follows the score (1 correct, 0 abstained, below 0 wrong), as the credit does not matter. A
    `now` row is of the family `pa` and a `memory` row of `where_last` unless the test names one, so each column is one
    ability (an ability in both columns counts once, in `memory`: `test_score_by_ability`)."""
    kw.setdefault("family", "pa" if kw.get("scope") == "now" else "where_last")
    return result_row(case, qid, score, "correct" if score > 0 else "abstained" if score == 0 else "wrong", **kw)


def test_now_and_memory_count_equally_whatever_their_counts():
    """Contract (VISION rule 4): the headline is the mean of the two columns, not of the questions. One `now` row at +1
    and nine `memory` rows at 0, a case each: the columns are 1.0 (n 1) and 0.0 (n 9), so the headline is 0.5, while
    the pooled mean the gates read is 1/10 = 0.1. Mirrored (nine `now` rows at +1, one `memory` row at 0) it is 0.5
    again, while the pooled mean is 9/10 = 0.9. Regression caught: a pooled headline, which the larger column decides
    (0.1 and 0.9 here)."""
    rows = [make("n0", "q", 1.0, scope="now")] + [make(f"m{i}", "q", 0.0) for i in range(9)]
    rep = aggregate(rows, n_boot=50)
    assert rep["headline"]["mean"] == pytest.approx(0.5)
    assert {k: (v["mean"], v["n"]) for k, v in rep["by"]["scope"].items()} == {"now": (1.0, 1), "memory": (0.0, 9)}
    assert pooled(rows, n_boot=50)["mean"] == pytest.approx(0.1)
    mirror = [make(f"n{i}", "q", 1.0, scope="now") for i in range(9)] + [make("m0", "q", 0.0)]
    assert aggregate(mirror, n_boot=50)["headline"]["mean"] == pytest.approx(0.5)
    assert pooled(mirror, n_boot=50)["mean"] == pytest.approx(0.9)


def test_rows_of_one_scope_have_that_column_as_their_headline():
    """Contract: a column with no rows is left out, so over rows of one scope (a during-off arm, a family, a job with no
    `now` question) the headline is that column, with the pooled interval. `memory` rows 1, 0, 0, 1, -1 (a script each)
    give 0.2, equal to `by.scope.memory` and to `pooled`; `now` rows 1 and 0.5 give 0.75. Regression caught: an empty
    column counted as 0 (the headline halves to 0.1 and 0.375) or a null headline for such rows."""
    memory = [make(f"c{i}", "q", s) for i, s in enumerate((1.0, 0.0, 0.0, 1.0, -1.0))]
    rep = aggregate(memory, n_boot=200)
    assert rep["headline"]["mean"] == pytest.approx(0.2) and rep["by"]["scope"]["memory"]["mean"] == pytest.approx(0.2)
    assert rep["headline"] == {k: pooled(memory, n_boot=200)[k] for k in ("mean", "ci")}
    assert rep["headline"]["ci"] is not None
    now = [make("c0", "q", 1.0, scope="now"), make("c1", "q", 0.5, scope="now")]
    assert aggregate(now, n_boot=50)["headline"]["mean"] == pytest.approx(0.75)


def floor_rows():
    """Six rows of two cases. `now` (case c1): an unknown gold at +1 and an answerable one at +1. `memory` (case c2): an
    unknown at +1 and three answerable at 0."""
    return [make("c1", "u", 1.0, scope="now", slice_="unknown"), make("c1", "a", 1.0, scope="now"),
            make("c2", "u", 1.0, slice_="unknown")] + [make("c2", f"a{i}", 0.0) for i in range(3)]


def test_above_floor_weighs_the_unknown_share_as_the_headline_does():
    """Contract: `above_floor` is the headline less the unknown share, both weighted by column. Headline (1 + 1/4)/2 =
    0.625; the unknown share is 1/2 of the `now` rows and 1/4 of the `memory` rows, so (0.5 + 0.25)/2 = 0.375 and
    `above_floor` = 0.25. A system answering `unknown` throughout scores 1 on the two unknown golds and 0 on the rest:
    columns 1/2 and 1/4, headline 0.375, `above_floor` exactly 0. Regression caught: the pooled share 2/6 subtracted
    from an equal-weight headline (0.292), which also leaves the floor at +0.042 instead of 0."""
    rep = aggregate(floor_rows(), n_boot=50)
    assert rep["headline"]["mean"] == pytest.approx(0.625) and rep["above_floor"]["mean"] == pytest.approx(0.25)
    floor = [{**r, "score": 1.0 if r["slice"] == "unknown" else 0.0} for r in floor_rows()]
    rep = aggregate(floor, n_boot=50)
    assert rep["headline"]["mean"] == pytest.approx(0.375) and rep["above_floor"]["mean"] == pytest.approx(0.0)


def test_the_interval_is_of_the_mean_resampled_jointly():
    """Contract: both columns and their mean are computed on every resample of scripts, from one draw. Four scripts,
    each with a `now` and a `memory` row: even scripts +1 and -1, odd ones -1 and +1. Each column alone is the
    alternating +-1 of `test_score_stats`'s four scripts, so its interval is (-1, 1) at seed 1; but the `memory` mean
    is minus the `now` mean in every resample, so the headline is exactly 0 in every resample and its interval is
    (0, 0). Regression caught: an interval built from the columns' intervals ((-1, 1) averaged is (-1, 1)), or the
    columns drawn separately (which would spread the headline)."""
    rows = []
    for i in range(4):
        sign = 1.0 if i % 2 == 0 else -1.0
        rows += [make(f"c{i}", "n", sign, scope="now", script=f"s{i}"),
                 make(f"c{i}", "m", -sign, script=f"s{i}")]
    rep = aggregate(rows, seed=1)
    assert rep["by"]["scope"]["now"]["ci"] == (-1.0, 1.0) and rep["by"]["scope"]["memory"]["ci"] == (-1.0, 1.0)
    assert rep["headline"]["mean"] == pytest.approx(0.0) and rep["headline"]["ci"] == pytest.approx((0.0, 0.0))


def test_a_resample_without_a_column_takes_the_other_alone():
    """Contract: a resample that lacks a column is the other column alone, in the interval as in the point. Four
    scripts: s0 has a `now` row at 0 and a `memory` row at +1, s1-s3 only a `memory` row at +1. The point is
    (0 + 1)/2 = 0.5. A resample misses s0 with probability (3/4)^4 = 81/256 (31.6%, above 2.5%) and scores 1.0; any
    other holds `now` 0 and `memory` 1 and scores 0.5, so the interval is (0.5, 1.0). Regression caught: a missing
    column counted as 0 (every resample 0.5) or the resample dropped (the same), both an interval of (0.5, 0.5)."""
    rows = [make("c0", "n", 0.0, scope="now", script="s0"), make("c0", "m", 1.0, script="s0")]
    rows += [make(f"c{i}", "m", 1.0, script=f"s{i}") for i in (1, 2, 3)]
    rep = aggregate(rows, seed=0)
    assert rep["headline"]["mean"] == pytest.approx(0.5) and rep["headline"]["ci"] == pytest.approx((0.5, 1.0))


def test_the_headline_difference_is_the_difference_of_the_headlines():
    """Contract: with `headline=True` `paired_difference` is A's headline less B's over the shared questions; by default
    (what the gates and the splits read) it pools. A and B share one `now` question X (A +1, B 0, difference 1) and
    three `memory` questions Y (both 0): the columns' differences are 1 and 0, so the headline difference is 0.5, which
    is A's headline 0.5 less B's 0.0; pooled it is 1/4. With `where` family Y only the `memory` column is left: 0.0, and
    with family X 1.0. Regression caught: `headline=True` ignored (0.25), or compare's headline difference not agreeing
    with its two headlines."""
    a = [make("c1", "x1", 1.0, scope="now", family="X")] + [make("c1", f"y{i}", 0.0, family="Y") for i in range(3)]
    b = [make("c1", "x1", 0.0, scope="now", family="X")] + [make("c1", f"y{i}", 0.0, family="Y") for i in range(3)]
    got = paired_difference(a, b, headline=True, n_boot=20)
    assert got["mean"] == pytest.approx(0.5) and got["n"] == 4
    assert aggregate(a, n_boot=20)["headline"]["mean"] == pytest.approx(0.5)
    assert aggregate(b, n_boot=20)["headline"]["mean"] == pytest.approx(0.0)
    assert paired_difference(a, b, n_boot=20)["mean"] == pytest.approx(0.25)
    assert paired_difference(a, b, where={"family": "Y"}, headline=True, n_boot=20)["mean"] == pytest.approx(0.0)
    assert paired_difference(a, b, where={"family": "X"}, headline=True, n_boot=20)["mean"] == pytest.approx(1.0)


def test_a_row_of_another_scope_is_refused():
    """Contract: a result row's scope is `now` or `memory` (`records.SCOPES`; `evidence` is a family's, resolved before
    a question is asked), so a row of another scope raises ValueError naming it. Regression caught: a row left out of
    both columns, which would drop its question from the headline without a word."""
    rows = [make("c1", "a", 1.0, scope="evidence")]
    for call in (aggregate, pooled):
        with pytest.raises(ValueError, match="scope.*evidence"):
            call(rows)


def test_results_md_prints_the_floor_the_headline_subtracts(tmp_path):
    """Contract: `results.md` prints the unknown share the headline subtracts, not the pooled one. On the document of
    `all_unknown` over `hand_dataset` with its layout replaced by the aggregate of `floor_rows` (6 questions): headline
    +0.625, the unknown share 0.375 in the above-the-floor row's name and +0.250 as its mean, and the rule that the
    headline is the mean of the columns. Regression caught: the pooled share 0.333 printed beside a headline of
    +0.625, so the two rows would not add up, or the rule left unsaid."""
    hand_dataset(tmp_path / "data")
    assert cli("all_unknown", "--data", tmp_path / "data", "--out", tmp_path / "run") == 0
    doc = results(tmp_path / "run")
    md = render({**doc, "rows": floor_rows(), "layouts": {"apartment_a": aggregate(floor_rows(), n_boot=50)}})
    assert "| headline | 6 | +0.625 | [" in md
    assert "| above the floor (the headline less the unknown share, 0.375) | 6 | +0.250 | [" in md
    assert HEADLINE_RULE in md
