"""The reports of the evidence age and of catastrophes (`score_stats.aggregate`, `eval_results`, `eval_report`;
0.10.0, FD-questions-0.10 §4.5-§4.6): `by.age_band`, `job_age_band` and `catastrophe_rate` on hand rows, the headline
staying the mean of the two columns on the reported score, and the results of an earlier build still loading and
rendering."""

from __future__ import annotations

import pytest

from eval_kit import cli, hand_dataset, results
from score_kit import result_row
from stbench import eval_report, eval_results
from stbench.score_stats import aggregate

# (case, family, job, scope, age_band, reported score, gate score, outcome, catastrophe, slice): nine rows on two jobs.
# harm families (`clear_path`, `closest_person`) answered: r1, r2, r3, r7, r8; r4 is `unknown` on an answerable gold
# and r6 `unknown` on an unknown one (neither is an answered row); r5 and r9 are `ego_now` (a situation family).
ROWS = [
    ("r1", "clear_path", "vacuum/1", "memory", "recent", -2.0, -0.05, "wrong", True, "answerable"),
    ("r2", "clear_path", "vacuum/1", "memory", "recent", 1.0, 1.0, "correct", False, "answerable"),
    ("r3", "clear_path", "vacuum/1", "memory", "task", 0.5, 0.5, "partial", False, "answerable"),
    ("r4", "clear_path", "vacuum/1", "memory", "task", 0.0, 0.0, "abstained", False, "answerable"),
    ("r5", "ego_now", "vacuum/1", "now", "now", 1.0, 1.0, "correct", False, "answerable"),
    ("r6", "clear_path", "vacuum/1", "memory", None, 1.0, 1.0, "correct", False, "unknown"),
    ("r7", "closest_person", "amr/1", "memory", "earlier", -3.0, -1.0, "wrong", True, "answerable"),
    ("r8", "closest_person", "amr/1", "memory", "earlier", 1.0, 1.0, "correct", False, "answerable"),
    ("r9", "ego_now", "amr/1", "now", "now", 1.0, 1.0, "correct", False, "answerable"),
]


def rows() -> list[dict]:
    return [result_row(case, f"q{case}", score, outcome, family=family, job=job, scope=scope, age_band=band,
                       score_gate=gate, catastrophe=cata, slice_=slice_, script=f"s{case}")
            for case, family, job, scope, band, score, gate, outcome, cata, slice_ in ROWS]


def test_the_reported_score_is_split_by_the_age_of_the_evidence_overall_and_per_job():
    """Contract (§4.6): `by.age_band` pools each band's reported scores and counts its rows, `none` for a row with no
    band, and `job_age_band` does so within each job, bands in `records.AGE_BANDS` order. By hand over the nine rows:
    `now` 1.0 and 1.0 (2 rows, mean 1.0); `recent` -2.0 and 1.0 (-0.5); `task` 0.5 and 0.0 (0.25); `earlier` -3.0 and
    1.0 (-1.0); `none` 1.0 (1). The vacuum has now 1 (1.0), recent 2 (-0.5), task 2 (0.25), none 1 (1.0); the AMR
    earlier 2 (-1.0), now 1 (1.0). Regression caught: a band pooled over the gate score, a job's rows in another job's
    band, or a band the rows hold dropped."""
    agg = aggregate(rows(), n_boot=50)
    assert {band: (stat["n"], stat["mean"]) for band, stat in agg["by"]["age_band"].items()} == {
        "now": (2, 1.0), "recent": (2, -0.5), "task": (2, 0.25), "earlier": (2, -1.0), "none": (1, 1.0)}
    jobs = agg["job_age_band"]
    assert list(jobs["vacuum/1"]) == ["now", "recent", "task", "none"] and list(jobs["amr/1"]) == ["now", "earlier"]
    assert {band: (stat["n"], stat["mean"]) for band, stat in jobs["vacuum/1"].items()} == {
        "now": (1, 1.0), "recent": (2, -0.5), "task": (2, 0.25), "none": (1, 1.0)}
    assert {band: (stat["n"], stat["mean"]) for band, stat in jobs["amr/1"].items()} == {
        "now": (1, 1.0), "earlier": (2, -1.0)}


def test_the_catastrophe_rate_is_catastrophes_over_the_harm_rows_answered_with_the_counts():
    """Contract (§4.6): `catastrophe_rate` is the rows whose answer was a catastrophe over the rows of a harm family
    that were answered (not `unknown`, which an abstention on an answerable gold and a right `unknown` on an unknown
    one both are), overall and per job, with both counts. By hand: five harm rows are answered (r1, r2, r3, r7, r8),
    two are catastrophes (r1, r7): 0.4, 2 of 5; the vacuum 1 of 3, the AMR 1 of 2; the situation rows (r5, r9) are in
    no count. Regression caught: an `unknown` answer counted as answered (diluting the rate), a situation family
    counted as harm, or a job's rows counted in another's."""
    found = aggregate(rows(), n_boot=50)["catastrophe_rate"]
    assert (found["rate"], found["n"], found["of"]) == (pytest.approx(0.4), 2, 5)
    assert {job: (stat["rate"], stat["n"], stat["of"]) for job, stat in found["jobs"].items()} == {
        "vacuum/1": (pytest.approx(1 / 3), 1, 3), "amr/1": (pytest.approx(0.5), 1, 2)}


def test_the_headline_keeps_its_form_on_the_reported_scores():
    """Contract (§4.6): the headline is the mean of the `now` and `memory` columns of counted abilities, each the equal
    mean of its cells, on the reported `score`. Here `now` is the distance cell, 1.0 and 1.0: 1.0; `memory` is the
    clearance cell over seven rows, (-2 + 1 + 0.5 + 0 + 1 - 3 + 1) / 7 = -1.5 / 7; the headline (1 - 1.5 / 7) / 2 =
    0.392857. On the gate scores the memory cell would be (-0.05 + 1 + 0.5 + 0 + 1 - 1 + 1) / 7 = 2.45 / 7. Regression
    caught: the headline read off the gate score, so that catastrophes never reach a report."""
    agg = aggregate(rows(), n_boot=50)
    assert agg["headline"]["mean"] == pytest.approx((1.0 + (-1.5 / 7)) / 2)
    assert agg["columns"]["now"]["mean"] == pytest.approx(1.0)
    assert agg["columns"]["memory"]["mean"] == pytest.approx(-1.5 / 7)


def test_rows_of_an_earlier_build_aggregate_with_no_age_and_no_catastrophe():
    """Contract (older results stay readable): a row without `age_band`, `score_gate` or `catastrophe` aggregates: its
    band reads `none`, its gate score is its `score` and no catastrophe is counted, so the same nine rows with those
    keys removed give the same headline and a catastrophe rate of 0 over the five harm rows answered. Regression caught:
    a KeyError on a result written before 0.10.0."""
    old = [{k: v for k, v in r.items() if k not in ("age_band", "score_gate", "catastrophe")} for r in rows()]
    agg, now = aggregate(old, n_boot=50), aggregate(rows(), n_boot=50)
    assert agg["headline"]["mean"] == pytest.approx(now["headline"]["mean"])
    assert {band: stat["n"] for band, stat in agg["by"]["age_band"].items()} == {"none": 9}
    assert (agg["catastrophe_rate"]["n"], agg["catastrophe_rate"]["of"]) == (0, 5)


def test_the_report_prints_the_age_table_and_the_catastrophe_rate():
    """Contract (§4.6): `results.md` has a table of the evidence age (a line for each band, then for each job's bands,
    jobs in name order) and the catastrophe rate with its counts, from the aggregate. Hand lines for the nine rows
    above. Regression caught: a report that hides the harm term or the age split it was added to show."""
    agg = aggregate(rows(), n_boot=50)
    table = eval_report._age_rows(agg)
    assert [r[:3] for r in table[:5]] == [["now", "2", "+1.000"], ["recent", "2", "-0.500"], ["task", "2", "+0.250"],
                                          ["earlier", "2", "-1.000"], ["none", "1", "+1.000"]]
    assert [r[0] for r in table[5:]] == ["amr/1: now", "amr/1: earlier", "vacuum/1: now", "vacuum/1: recent",
                                         "vacuum/1: task", "vacuum/1: none"]
    head, space, overall, amr, vacuum, blank = eval_report._catastrophe_lines(agg)
    assert head == "### Catastrophes" and space == blank == ""
    assert overall.startswith("- all jobs: 0.400 [") and overall.endswith("], 2 catastrophes of 5 harm rows answered")
    assert vacuum.startswith("- vacuum/1: 0.333 [") and vacuum.endswith("], 1 catastrophe of 3 harm rows answered")
    assert amr.startswith("- amr/1: 0.500 [") and amr.endswith("], 1 catastrophe of 2 harm rows answered")


def test_a_results_document_of_an_earlier_build_loads_and_renders_without_the_new_sections(tmp_path):
    """Contract (older results stay readable): a document with none of the 0.10.0 keys (no `age_band`, `score_gate` or
    `catastrophe` in a row; no `by.age_band`, `job_age_band` or `catastrophe_rate` in an aggregate) passes
    `eval_results.problems` and renders, with neither the age table nor the catastrophe section; the same run as written
    now has both. Regression caught: a results.json already on disk that the new schema or report refuses."""
    hand_dataset(tmp_path)
    assert cli("all_unknown", "--data", tmp_path, "--out", tmp_path / "run") == 0
    doc = results(tmp_path / "run")
    assert eval_results.problems(doc) == []
    text = eval_report.render(doc)
    assert "### Per age of the evidence" in text and "### Catastrophes" in text
    old = {**doc, "rows": [{k: v for k, v in r.items() if k not in ("age_band", "score_gate", "catastrophe")}
                           for r in doc["rows"]],
           "layouts": {name: {**{k: v for k, v in agg.items() if k not in ("job_age_band", "catastrophe_rate")},
                              "by": {k: v for k, v in agg["by"].items() if k != "age_band"}}
                       for name, agg in doc["layouts"].items()}}
    assert eval_results.problems(old) == []
    older = eval_report.render(old)
    assert "### Per age of the evidence" not in older and "### Catastrophes" not in older
    assert "### Per kind" in older
