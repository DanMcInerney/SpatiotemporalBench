"""The headline rule a results document prints (`eval_report.HEADLINE_RULE`, final review S3) against what
`score_stats.aggregate` does with the weights: the words and the arithmetic are held to one hand case, so the rule
cannot go on stating an earlier build's."""

from __future__ import annotations

import pytest

from ability_kit import scored
from stbench.abilities import MIN_ABILITY_N
from stbench.eval_report import HEADLINE_RULE
from stbench.score_stats import aggregate

# the dataset's thin cells: the helper's `change` cell in `memory` holds 10 of the 20 questions a full cell needs and
# its `clearance` cell in `now` 5; the `where_last` cell holds 30 and weighs 1 (so it is no entry)
THIN = {("helper/1", "change", "memory"): 10 / MIN_ABILITY_N, ("helper/1", "clearance", "now"): 5 / MIN_ABILITY_N}


def test_the_headline_rule_states_the_weights_the_aggregate_applies():
    """Contract (final review S3; FD-0.11 §3.7): a cell weighs min(1, n / 20), a column is the mean of its cells by
    their weights, and the headline is the mean of the columns, each weighing the largest weight of its cells; the
    printed rule says so, with the 20 read from `MIN_ABILITY_N`. By hand: the `memory` column holds `change` at weight
    1/2 (four answers of +1: 1.0) and `where_last` at weight 1 (two of 0: 0.0), so it is (0.5 * 1 + 1 * 0) / 1.5 = 1/3
    and weighs 1; the `now` column holds `clearance` at weight 1/4 (two of +0.6: 0.6), so it is 0.6 and weighs 1/4; the
    headline is (1 * 1/3 + 1/4 * 0.6) / (1 + 1/4) = (0.33333 + 0.15) / 1.25 = 0.38667. The 0.10.0 rule (cells equal
    inside a column, the columns equal) gives 0.5 and 0.6 and a headline of 0.55; counting the thin cells in no mean,
    0.0 and none, 0.0. Regression caught: the printed rule or the aggregate going back to the equal or the dropped
    rule, or a column weighing 1 whatever its cells."""
    rows = [*scored("object_was", [1, 1, 1, 1]), *scored("where_last", [0, 0], case="c2"),
            *scored("clear_path", [0.6, 0.6], case="c3", scope="now")]
    found = aggregate(rows, n_boot=20, thin=THIN)
    assert found["columns"]["memory"]["mean"] == pytest.approx(1 / 3)
    assert found["columns"]["now"]["mean"] == pytest.approx(0.6)
    assert found["headline"]["mean"] == pytest.approx(0.38667, abs=1e-5)
    assert MIN_ABILITY_N == 20
    for said in ("each weighing the largest weight of its cells", "A column is the mean of its cells by their weights",
                 "A cell's weight is min(1, n / 20)", "a thinner one by its share"):
        assert said in HEADLINE_RULE
    for gone in ("its cells count equally", "fewer than 20 questions"):
        assert gone not in HEADLINE_RULE
