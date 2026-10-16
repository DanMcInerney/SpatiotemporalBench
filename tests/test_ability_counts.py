"""The dataset-wide count that weighs each (job, ability, column) cell (`ability_counts`: `min(1, n / 20)`, FD-0.11
§3.7), and its use by `eval` (`eval_results.build`)."""

from __future__ import annotations

import json

import pytest

from eval_kit import cli, compiled, gold, hand_script, results
from run_kit import golds
from stbench import case as case_mod
from stbench.ability_counts import counts, thin_abilities, thin_rule, weights
from stbench.cli import main
from stream_kit import during_q, end_q, gold_row


def dataset(root, n_end: int, *, unknown: tuple[int, ...] = ()):
    """One case asked a during question (`path_ahead`) and `n_end` end ones (`object_where`, qids 2..); the golds at the
    indices `unknown` of the question list are unknown."""
    compiled(root, hand_script(1), [during_q(1, 12), *[end_q(n) for n in range(2, 2 + n_end)]], unknown=unknown)
    (job,) = {e["job"] for e in (case_mod.read_entry(root, i) for i in case_mod.read_index(root)) if "job" in e}
    return job


@pytest.mark.parametrize("n_end, thin", [(19, True), (20, False)])
def test_a_pair_with_fewer_than_twenty_questions_in_the_dataset_is_thin(tmp_path, n_end, thin):
    """Contract (FD-0.6.0 §10; per column): `thin_abilities` names the (job, ability, column) cells the
    whole dataset asks fewer than 20 questions of; 20 is enough. By hand: one case with `n_end` end questions of
    `object_where` (the ability `memory`, scope `memory`) and one `path_ahead` question (an ability of its own, scope
    `now`, always 1): the `memory` cell is thin at 19 and not at 20; the `path_ahead` cell is thin at both. Regression
    caught: the boundary off by one, or an ability counted by its family instead of pooled."""
    job = dataset(tmp_path, n_end)
    assert counts(tmp_path) == {(job, "memory", "memory"): n_end, (job, "path_ahead", "now"): 1}
    assert thin_abilities(tmp_path) == frozenset(
        {(job, "path_ahead", "now")} | ({(job, "memory", "memory")} if thin else set()))


@pytest.mark.parametrize("n_end", [0, 10, 19, 20, 21, 35])
def test_a_cells_weight_is_n_over_twenty_up_to_one_and_moves_continuously(tmp_path, n_end):
    """Contract (FD-0.11 §3.7): a cell of `n` questions weighs `min(1, n / 20)`, so it moves by 1/20 a question and
    reaches 1 at 20 and stays there (the 0.6.0 to 0.10.0 rule dropped a cell of 19 to nothing and kept one of 20 whole).
    By hand: one `path_ahead` question (weight 1/20) and `n_end` end questions of `object_where` (the ability `memory`):
    0.5 at 10, 0.95 at 19, 1 at 20, 21 and 35; no `memory` cell at all at 0. `thin_rule` lists the cells under 1 with
    their weights and applies them, except that it is relaxed (no cell applied) when every cell is thin, as at 0, 10 and
    19. Regression caught: a cell vanishing at 19 questions, a weight past 1, or a rule that disagrees with the
    weights."""
    job = dataset(tmp_path, n_end)
    want = {(job, "path_ahead", "now"): 1 / 20, **({(job, "memory", "memory"): min(1.0, n_end / 20)} if n_end else {})}
    assert weights(tmp_path) == pytest.approx(want)
    thin = {cell: w for cell, w in want.items() if w < 1}
    got, relaxed = thin_rule(tmp_path)
    if len(thin) == len(want):   # every cell is thin: the dataset is too small for the weights
        assert (got, relaxed) == ({}, True)
    else:
        assert got == pytest.approx(thin) and relaxed is False


def test_eval_weighs_the_cells_by_the_whole_dataset_and_a_thin_one_weighs_its_share_of_the_headline(tmp_path):
    """Contract: the weights read the dataset, not the run: a run of `all_unknown` over one case still carries the cell
    thin. By hand, 20 end questions of which 5 have an unknown gold and one answerable `path_ahead`: `all_unknown`
    scores `memory` 5/20 = 1/4 and `path_ahead` 0; the `path_ahead` cell has 1 question in the dataset, so it weighs
    1/20, and its column (`now`, that cell alone) weighs 1/20 against the `memory` column's 1. The headline is (1 * 1/4
    + 1/20 * 0) / (1 + 1/20) = 5/21 (equal columns would give 1/8, and the old rule, which dropped the cell and its
    column, 1/4); the cells weigh 20/21 and 1/21 in it, and the aggregate lists the cell with its weight. Regression
    caught: a thin cell counted whole, a thin cell dropped, or the weights computed from the rows of the run (a pilot
    would then weigh every cell by its handful of questions)."""
    job = dataset(tmp_path, 20, unknown=(1, 2, 3, 4, 5))
    assert cli("all_unknown", "--data", tmp_path, "--out", tmp_path / "run") == 0
    agg = results(tmp_path / "run")["layouts"]["apartment_a"]
    assert agg["headline"]["mean"] == pytest.approx(5 / 21) and list(agg["columns"]) == ["now", "memory"]
    assert agg["thin"] == [[job, "path_ahead", "now", pytest.approx(1 / 20)]] and agg["thin_ignored"] is False
    cell = agg["by"]["ability"]["path_ahead"]["cells"]["now"]
    assert (cell["diagnostic"], cell["counted"], cell["weight"]) == (False, 1, pytest.approx(1 / 21))
    assert agg["by"]["ability"]["memory"]["cells"]["memory"]["weight"] == pytest.approx(20 / 21)


NOTE = ("every ability is under 20 questions on its job in this dataset; the rule is not applied, so this headline is "
        "not the release statistic")


def test_the_rule_is_relaxed_once_for_the_dataset_and_a_subset_run_agrees_with_the_full_one(tmp_path):
    """Contract (review of lane A1): the dataset decides whether the weights apply at all: not when every
    non-diagnostic (job, ability) pair of the whole dataset is thin (`thin_rule`), whatever a run selects. By hand: one
    case of a `path_ahead` question and 5 end questions (3 of them with unknown golds) and a second of 5 end questions
    (2 unknown): `memory` has 10 questions and `path_ahead` 1, both thin, so no pair applies and the rule is relaxed.
    `all_unknown` over both cases scores `memory` 5/10 and `path_ahead` 0, headline (0 + 1/2)/2 = 1/4; over the second
    case alone, `memory` 2/5, the headline 2/5 (not null, and the same rule as the full run's). Both results say
    `thin_ignored` and carry the note that the headline is not the release statistic. At 20 end questions it is not
    relaxed (the previous test). Regression caught: a rule relaxed from the rows of a run, so that a subset and the
    full run disagree on what counts, or a relaxed headline given as a release one."""
    job = dataset(tmp_path, 5, unknown=(1, 2, 3))
    other = compiled(tmp_path, hand_script(2), [end_q(n) for n in range(2, 7)], unknown=(0, 1))
    assert counts(tmp_path) == {(job, "memory", "memory"): 10, (job, "path_ahead", "now"): 1}
    assert thin_rule(tmp_path) == ({}, True) and len(thin_abilities(tmp_path)) == 2
    assert cli("all_unknown", "--data", tmp_path, "--out", tmp_path / "full") == 0
    assert cli("all_unknown", "--data", tmp_path, "--out", tmp_path / "part", "--cases", other.id) == 0
    full, part = (results(tmp_path / name) for name in ("full", "part"))
    assert full["layouts"]["apartment_a"]["headline"]["mean"] == pytest.approx(1 / 4)
    assert part["layouts"]["apartment_a"]["headline"]["mean"] == pytest.approx(2 / 5)
    for doc in (full, part):
        assert doc["layouts"]["apartment_a"]["thin_ignored"] is True and doc["layouts"]["apartment_a"]["thin"] == []
        assert NOTE in doc["notes"]


def test_a_single_run_scores_with_the_datasets_weights_and_counts_only_the_counted_questions(tmp_path, capsys):
    """Contract (`stbench run`): the summary's scores use the whole dataset's weights, count their questions from the
    columns (diagnostic rows are not in them) and say so when every row is a diagnostic. By hand, a case of 1
    `mover_next` question (the diagnostic ability `prediction`) and a case of 20 end ones (`memory`, 5 with unknown
    golds): `memory` weighs 1. `all_unknown` on the second case scores `memory` 5/20, headline 1/4 over 20 questions; on
    the first it scores one diagnostic row: no headline and no columns, and the line says every one of 1 questions is a
    diagnostic. Regression caught: a headline from an unrestricted aggregate that disagrees with `eval`'s on the same
    rows, a count that includes diagnostics, or `None` formatted as a number."""
    during = compiled(tmp_path, hand_script(1), [during_q(1, 12)])
    golds(during, rows=[gold_row(during_q(1, 12), family="mover_next")])
    ends = compiled(tmp_path, hand_script(2), [end_q(n) for n in range(2, 22)], unknown=tuple(range(5)))
    assert main(["run", "all_unknown", str(ends.private)]) == 0
    assert "scores: headline 0.250 over 20 questions" in capsys.readouterr().out
    summary = json.loads((tmp_path / "runs" / f"all_unknown-{ends.id}" / "summary.json").read_text(encoding="utf-8"))
    assert summary["scores"]["columns"]["memory"]["n"] == 20 and "now" not in summary["scores"]["columns"]
    assert main(["run", "all_unknown", str(during.private)]) == 0
    assert "scores: every one of 1 questions is a diagnostic here, so there is no headline" in capsys.readouterr().out
    summary = json.loads((tmp_path / "runs" / f"all_unknown-{during.id}" / "summary.json").read_text(encoding="utf-8"))
    assert summary["scores"]["headline"]["mean"] is None and summary["scores"]["columns"] == {}


def test_a_cell_under_twenty_weighs_its_own_count_while_its_abilitys_other_cell_weighs_one(tmp_path):
    """Contract (FD-task-questions review §5; FD-0.11 §3.7): the weight is per (job, ability, column), so an ability
    asked in both scopes has a cell in each and a thin cell does not ride on the other's count. By hand: one case of 35
    end questions of `object_where` (the ability `memory`), the first 15 with scope `now` (3 of them unknown) and the
    other 20 with scope `memory` (10 unknown). The counts are 15 and 20, so the `now` cell weighs 15/20 = 3/4 and the
    `memory` cell 1. `all_unknown` scores 1 on an unknown gold and 0 on any other: the `memory` cell 10/20 = 1/2, the
    `now` cell 3/15 = 1/5. Each column holds one cell, so the `now` column weighs 3/4 and the `memory` column 1: the
    headline is (3/4 * 1/5 + 1 * 1/2) / (3/4 + 1) = 13/35 (equal columns, as when it counted whole, give 7/20, and when
    it was dropped, 1/2), and the cells weigh 3/7 and 4/7 in it. Regression caught: 15 noisy rows weighing half a
    headline, the cell dropped at 15 where it should weigh 3/4, or the line applied per (job, ability) so that a cell
    of 15 weighed 1 for riding on 35."""
    job = dataset(tmp_path, 0)
    asked = [end_q(n) for n in range(10, 45)]
    case = compiled(tmp_path, hand_script(2), asked)
    rows = [gold(q, unknown=i in (0, 1, 2, *range(15, 25))) for i, q in enumerate(asked)]
    golds(case, rows=[{**row, "scope": "now" if i < 15 else "memory"} for i, row in enumerate(rows)])
    assert counts(tmp_path) == {(job, "path_ahead", "now"): 1, (job, "memory", "now"): 15,
                                (job, "memory", "memory"): 20}
    assert thin_abilities(tmp_path) == frozenset({(job, "path_ahead", "now"), (job, "memory", "now")})
    assert cli("all_unknown", "--data", tmp_path, "--out", tmp_path / "run", "--cases", case.id) == 0
    agg = results(tmp_path / "run")["layouts"]["apartment_a"]
    assert agg["headline"]["mean"] == pytest.approx(13 / 35) and list(agg["columns"]) == ["now", "memory"]
    cells = agg["by"]["ability"]["memory"]["cells"]
    assert (cells["now"]["n"], cells["now"]["counted"], cells["now"]["diagnostic"]) == (15, 15, False)
    assert cells["now"]["weight"] == pytest.approx(3 / 7)
    assert (cells["memory"]["n"], cells["memory"]["counted"]) == (20, 20)
    assert cells["memory"]["weight"] == pytest.approx(4 / 7)
    assert cells["now"]["mean"] == pytest.approx(1 / 5) and cells["memory"]["mean"] == pytest.approx(1 / 2)
