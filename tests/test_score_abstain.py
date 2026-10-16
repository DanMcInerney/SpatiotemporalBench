"""The abstention report (`score_abstain`, and its section of `results.md`, `eval_report._abstention`): recall,
precision and the `unknown` rate on answerable questions, counted from result rows by hand. The rows are
`score_kit.result_row`'s.

Twelve questions in all (the hand case): helper/1 `object_where` has three `unknown` golds (answered `unknown`,
`unknown`, a place) and three answerable (abstained, partial, correct); helper/1 `ego_mem` one `unknown` gold (answered
`unknown`) and three answerable (abstained, wrong, partial); vacuum/2 `ego_mem` one `unknown` gold (a place) and one
answerable (correct)."""

from __future__ import annotations

import pytest

from score_kit import result_row
from stbench.eval_report import render
from stbench.score_abstain import Share, groups, said_unknown, summary, wilson
from stbench.score_stats import aggregate

SPEC = [
    ("helper/1", "object_where", "unknown", "correct"), ("helper/1", "object_where", "unknown", "correct"),
    ("helper/1", "object_where", "unknown", "wrong"), ("helper/1", "object_where", "answerable", "abstained"),
    ("helper/1", "object_where", "answerable", "partial"), ("helper/1", "object_where", "answerable", "correct"),
    ("helper/1", "ego_mem", "unknown", "correct"), ("helper/1", "ego_mem", "answerable", "abstained"),
    ("helper/1", "ego_mem", "answerable", "wrong"), ("helper/1", "ego_mem", "answerable", "partial"),
    ("vacuum/2", "ego_mem", "unknown", "wrong"), ("vacuum/2", "ego_mem", "answerable", "correct"),
]


def rows() -> list[dict]:
    """The hand case as result rows; a row's score is the penalised one (1 right `unknown`, 0 an abstention, -1 a
    wrong answer, 0.5 or 1 otherwise), which no abstention number reads."""
    score = {"correct": 1.0, "partial": 0.5, "wrong": -1.0, "abstained": 0.0}
    return [result_row(f"c{i}", f"q{i}", score[outcome], outcome, family=family, job=job, slice_=slice_,
                       robot=job.split("/")[0], when="end" if family == "object_where" else "during")
            for i, (job, family, slice_, outcome) in enumerate(SPEC)]


def test_an_unknown_answer_is_an_abstention_on_an_answerable_gold_or_a_correct_answer_on_an_unknown_one():
    """Contract: the report counts a row as answered `unknown` exactly when its outcome is `abstained` (an answerable
    gold) or `correct` on an `unknown` gold; a wrong answer to an `unknown` gold, and every answer to an answerable
    one that is not an abstention, is not. By hand over the outcomes each slice can have (an `unknown` gold is
    answered right or wrong). Regression caught: a wrong place on an `unknown` gold counted as an abstention
    (inflating precision), or a right `unknown` missed (deflating recall)."""
    outcomes = {"answerable": ("correct", "partial", "wrong", "abstained"), "unknown": ("correct", "wrong")}
    said = {(slice_, outcome): said_unknown(result_row("c", "q", 0.0, outcome, slice_=slice_))
            for slice_, found in outcomes.items() for outcome in found}
    assert {k for k, v in said.items() if v} == {("answerable", "abstained"), ("unknown", "correct")}


def test_recall_precision_and_the_answerable_rate_count_the_hand_case_overall_by_job_and_by_family():
    """Contract (0.11.0, FD-0.11-design 3.8 and 4.5): recall is the `unknown` answers over the `unknown` golds,
    precision the same over all `unknown` answers, and the over-abstention rate the `unknown` answers on answerable
    golds over the answerable golds, overall, per job and per (job, family), in that order with jobs and families
    sorted. By hand: overall 5 `unknown` golds with 3 answered `unknown` (3/5), 7 answerable of which 2 abstained
    (2/7), 5 `unknown` answers of which 3 on `unknown` golds (3/5). helper/1: 4 golds, 3 hits (3/4); 6 answerable, 2
    abstentions (2/6); 5 answers, 3 right (3/5). helper/1 object_where: 2/3, precision 2/3, 1/3. helper/1 ego_mem:
    1/1, 1/2, 1/3. vacuum/2 (and its only family): 0 of 1 golds, no `unknown` answer so precision is 0 of 0, 0 of 1
    answerable. Regression caught: the three shares over each other's denominators, the groups out of order, or a
    family pooled across jobs."""
    found = groups(rows())
    assert [label for label, _ in found] == ["all", "helper/1", "helper/1 ego_mem", "helper/1 object_where", "vacuum/2",
                                             "vacuum/2 ego_mem"]
    by = dict(found)
    assert by["all"] == (Share(3, 5), Share(3, 5), Share(2, 7))
    assert by["helper/1"] == (Share(3, 4), Share(3, 5), Share(2, 6))
    assert by["helper/1 object_where"] == (Share(2, 3), Share(2, 3), Share(1, 3))
    assert by["helper/1 ego_mem"] == (Share(1, 1), Share(1, 2), Share(1, 3))
    assert by["vacuum/2"] == by["vacuum/2 ego_mem"] == (Share(0, 1), Share(0, 0), Share(0, 1))
    assert summary(rows()) == by["all"]
    assert Share(0, 0).rate is None and Share(0, 0).ci is None and Share(3, 5).rate == 0.6


@pytest.mark.parametrize("k, n, lo, hi", [(3, 4, 0.3006, 0.9544), (3, 5, 0.2307, 0.8824), (2, 7, 0.0822, 0.6411),
                                          (0, 10, 0.0, 0.2775), (1, 1, 0.2065, 1.0)])
def test_the_interval_is_the_wilson_score_interval_at_95_percent_and_stays_inside_zero_and_one(k, n, lo, hi):
    """Contract: `wilson(k, n)` is the 95% Wilson score interval, `(p + z^2/2n -+ z sqrt(p(1-p)/n + z^2/4n^2)) / (1 +
    z^2/n)`, clipped to [0, 1], and None for no trials. The expected ends are worked out by hand with z = 1.96 (3 of
    4: 0.3006 to 0.9544; 0 of 10: 0 to 0.2775, the textbook values). Regression caught: the Wald interval (which has
    zero width at 0 of 10 and leaves [0, 1] at 1 of 1), or a wrong z."""
    got = wilson(k, n)
    assert got == pytest.approx((lo, hi), abs=2e-4)
    assert wilson(0, 0) is None


def test_results_md_has_an_abstention_section_by_job_and_family_with_the_counts():
    """Contract (0.11.0): `results.md` carries a `### Abstention` section after the outcomes, a line for the whole
    layout, each job and each (job, family), with recall, precision and the `unknown` rate on answerable questions as
    a rate, its Wilson interval and its counts. By hand on the hand case (all in one layout): all 0.600 [0.23, 0.88]
    (3/5), 0.600 [0.23, 0.88] (3/5), 0.286 [0.08, 0.64] (2/7); helper/1 0.750 [0.30, 0.95] (3/4), 0.600 [0.23, 0.88]
    (3/5), 0.333 [0.10, 0.70] (2/6); vacuum/2 0.000 [0.00, 0.79] (0/1), n/a (0/0), 0.000 [0.00, 0.79] (0/1).
    Regression caught: the section missing from the report, or a share shown without its denominator (an interval
    over three questions read as a rate over a hundred)."""
    data = rows()
    doc = {"system": "hand", "channel": "rgb_pose", "brief": "correct", "during": True, "sandbox": "in-process",
           "reasoning": None, "text_only": False, "verified": False, "partial": False, "max_usd": None,
           "cost_usd": None, "notes": [], "rows": data, "layouts": {"apartment_a": aggregate(data, n_boot=20)}}
    doc["cases"] = [{"layout": "apartment_a", "id": r["case"], "status": "complete", "error": None,
                     "tokens": {k: None for k in ("tokens_in", "tokens_cached_in", "tokens_cache_write_in",
                                                  "tokens_out")}, "cost_usd": None, "wall_s": {"run": 1.0}}
                    for r in data]
    lines = render(doc).splitlines()
    start = lines.index("### Abstention")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("### "))
    table = lines[start:end]
    assert "| all | 0.600 [0.23, 0.88] (3/5) | 0.600 [0.23, 0.88] (3/5) | 0.286 [0.08, 0.64] (2/7) |" in table
    assert "| helper/1 | 0.750 [0.30, 0.95] (3/4) | 0.600 [0.23, 0.88] (3/5) | 0.333 [0.10, 0.70] (2/6) |" in table
    assert "| vacuum/2 | 0.000 [0.00, 0.79] (0/1) | n/a (0/0) | 0.000 [0.00, 0.79] (0/1) |" in table
    assert any(line.startswith("| helper/1 object_where | 0.667 [0.21, 0.94] (2/3) |") for line in table)
    assert lines.index("### Outcomes") < start < lines.index("### Credit bands by ability")
