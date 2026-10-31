"""Aggregation over result rows (plan §2.6): the headline (the mean of the `now` and `memory` columns, each the equal
mean of its abilities; the abilities' own contracts are in `test_score_by_ability`), the splits reported beside it,
outcome counts, abstention, the bootstrap over scripts, paired differences and gap closed. Expected
values are worked out by hand in each test."""

from __future__ import annotations

import pytest

from score_kit import UNKNOWN, pos, rec, result_row
from stbench.score import aggregate, gap_closed, paired_difference, row
from stbench.score_stats import DIMS, pooled


def make(case, qid, score, outcome_, **kw):
    return result_row(case, qid, score, outcome_, **kw)


# Twelve rows in three cases of three scripts. `ow` is object_where (end, memory), `pa` path_ahead (during, now), `own`
# a family of the robot's own path. The first eight rows are short cases of the vacuum on apartment_a, the last four a
# long case of another robot and job on the warehouse; rows a and e carry the critical tag; j and k have unknown golds.
FIXTURE = [
    make("c1", "a", 1.0, "correct", family="ow", critical=True),
    make("c1", "b", -1.0, "wrong", family="ow"),
    make("c1", "c", 0.5, "partial", family="pa", when="during", scope="now"),
    make("c1", "d", 0.5, "partial", family="pa", when="during", scope="now"),
    make("c2", "e", 1.0, "correct", family="ow", critical=True),
    make("c2", "f", 0.0, "abstained", family="ow"),
    make("c2", "g", 1.0, "correct", family="pa", when="during", scope="now"),
    make("c2", "h", 0.0, "abstained", family="pa", when="during", scope="now"),
    make("c3", "i", 1.0, "correct", family="pa", when="during", scope="now", layout="warehouse_a", robot="amr",
         job="pick/1", length="long"),
    make("c3", "j", 1.0, "correct", family="ow", slice_="unknown", layout="warehouse_a", robot="amr", job="pick/1",
         length="long"),
    make("c3", "k", -1.0, "wrong", family="ow", slice_="unknown", layout="warehouse_a", robot="amr", job="pick/1",
         length="long"),
    make("c3", "l", 0.0, "abstained", family="own", when="during", reference="path_log", layout="warehouse_a",
         robot="amr", job="pick/1", length="long"),
]


def test_aggregate_reports_the_headline_the_splits_counts_wrong_rate_and_abstention():
    """Contract (plan §2.6, VISION rule 4): the headline is the mean of the `now` and `memory` columns, each the equal
    mean of its abilities (`ow`, `pa` and `own` are three abilities of their own: unlisted families); the splits
    (`DIMS`) pool within each group; outcome counts; confidently wrong
    rate = wrong over all questions; abstention precision = unknown answers on unknown golds over unknown answers,
    recall = of unknown golds. Worked on FIXTURE (scores 1, -1, .5, .5, 1, 0, 1, 0, 1, 1, -1, 0 sum to 4): pooled 4/12;
    the `now` column is `pa` alone, 0.6, and the `memory` column (1/6 + 0)/2 = 1/12, so the headline is
    (3/5 + 1/12) / 2 = 41/120. By family ow (1-1+1+0+1-1)/6 = 1/6, pa
    (.5+.5+1+0+1)/5 = 0.6, own 0. By scope (pooled) now 0.6, memory (1 + 0)/7 = 1/7; own path "True" 0, "False" 4/11;
    when during
    (3 + 0)/6 = 0.5, end 1/6; length short 3/8 and long 1/4, which are also the layout, robot and job splits here;
    critical "True" (1 + 1)/2 = 1, "False" 2/10. Counts: correct 5, partial 2, wrong 2, abstained 3. Wrong 2/12. Unknown
    answers: 1 correct unknown and 3 abstentions = 4, 1 of them on an unknown gold: precision 1/4; 2 unknown golds, 1
    answered: recall 1/2.

    Regression caught: a headline pooled within each column (13/35) or over both (4/12), a split
    pooled across groups, or wrong conflated with abstained."""
    rep = aggregate(FIXTURE, n_boot=50)
    assert (rep["n"], rep["scripts"], rep["cases"]) == (12, 3, 3)
    assert rep["headline"]["mean"] == pytest.approx(41 / 120)
    assert {k: c["mean"] for k, c in rep["columns"].items()} == pytest.approx({"now": 0.6, "memory": 1 / 12})
    assert {k: c["n"] for k, c in rep["columns"].items()} == {"now": 5, "memory": 7}
    by = rep["by"]
    assert set(by) == set(DIMS)
    assert {k: v["mean"] for k, v in by["family"].items()} == pytest.approx({"ow": 1 / 6, "pa": 0.6, "own": 0.0})
    assert {k: v["n"] for k, v in by["family"].items()} == {"ow": 6, "pa": 5, "own": 1}
    assert {k: v["mean"] for k, v in by["scope"].items()} == pytest.approx({"now": 0.6, "memory": 1 / 7})
    assert {k: v["mean"] for k, v in by["own_path"].items()} == pytest.approx({"True": 0.0, "False": 4 / 11})
    assert {k: v["mean"] for k, v in by["when"].items()} == pytest.approx({"during": 0.5, "end": 1 / 6})
    for dim, short, long in (("length", "short", "long"), ("layout", "apartment_a", "warehouse_a"),
                             ("robot", "vacuum", "amr"), ("job", "vacuum/1", "pick/1")):
        assert {k: v["mean"] for k, v in by[dim].items()} == pytest.approx({short: 3 / 8, long: 1 / 4}), dim
    assert {k: v["mean"] for k, v in by["critical"].items()} == pytest.approx({"True": 1.0, "False": 0.2})
    assert {k: v["n"] for k, v in by["critical"].items()} == {"True": 2, "False": 10}
    assert rep["counts"] == {"correct": 5, "partial": 2, "wrong": 2, "abstained": 3}
    assert rep["confidently_wrong_rate"]["rate"] == pytest.approx(2 / 12)
    ab = rep["abstention"]
    assert ab["precision"] == pytest.approx(1 / 4) and ab["recall"] == pytest.approx(1 / 2)
    assert (ab["answered_unknown"], ab["unknown_gold"]) == (4, 2)
    assert pooled(FIXTURE, n_boot=50)["mean"] == pytest.approx(4 / 12)


def test_aggregate_of_no_rows_is_empty_not_an_error():
    """Contract: `aggregate([])` is `n` 0 with null means, no splits and zero counts (a system that answered nothing
    in a slice still reports). Regression caught: a report that raises on an empty group."""
    rep = aggregate([])
    assert (rep["n"], rep["scripts"], rep["cases"]) == (0, 0, 0)
    assert rep["headline"] == {"mean": None, "ci": None} and rep["counts"] == dict.fromkeys(rep["counts"], 0)
    assert rep["by"] == {d: {} for d in DIMS} and rep["confidently_wrong_rate"]["rate"] is None
    assert rep["abstention"]["precision"] is None and pooled([])["mean"] is None


def test_abstention_is_none_when_nothing_to_divide_by():
    """Contract: precision is None with no unknown answers, recall None with no unknown golds. Regression caught: 0/0
    reported as 0 or crashing a report on a system that never abstains."""
    rows = [make("c1", "a", 1.0, "correct"), make("c2", "b", -1.0, "wrong")]
    ab = aggregate(rows, n_boot=20)["abstention"]
    assert ab["precision"] is None and ab["recall"] is None
    assert ab["ci_precision"] is None and ab["ci_recall"] is None


@pytest.mark.parametrize(
    "rows",
    [[make("c1", "a", 1.0, "correct")] * 2, [{**make("c1", "a", 1.0, "correct"), "outcome": "great"}],
     [{k: v for k, v in make("c1", "a", 1.0, "correct").items() if k != "script"}],
     [{k: v for k, v in make("c1", "a", 1.0, "correct").items() if k != "critical"}]],
    ids=["duplicate", "bad-outcome", "missing-script", "missing-critical"],
)
def test_aggregate_refuses_malformed_input(rows):
    """Contract: a repeated (case, qid), an unknown outcome or a missing key raises ValueError. Regression caught: a
    double-counted question or a typo'd outcome skewing a report without any error."""
    with pytest.raises(ValueError):
        aggregate(rows, n_boot=10)


def test_the_same_qid_in_two_cases_is_two_questions():
    """Contract: rows are keyed by (case, qid), so two cases may hold a qid and both count. Regression caught: a key on
    the qid alone, refusing or merging the questions of different cases."""
    rows = [make("c1", "a", 1.0, "correct"), make("c2", "a", 0.0, "abstained")]
    rep = aggregate(rows, n_boot=10)
    assert rep["n"] == 2 and rep["headline"]["mean"] == 0.5


def scripts(n_scripts, cases_each=1, questions_each=1):
    """Script i scores +1 on all its questions if i is even, else -1: `cases_each` cases of `questions_each` each."""
    return [make(f"s{i}c{c}", f"q{j}", 1.0 if i % 2 == 0 else -1.0, "correct" if i % 2 == 0 else "wrong",
                 script=f"s{i}")
            for i in range(n_scripts) for c in range(cases_each) for j in range(questions_each)]


def width(report):
    lo, hi = report["headline"]["ci"]
    return hi - lo


def test_bootstrap_is_deterministic_and_independent_of_row_order():
    """Contract: the same rows and seed give identical results, whatever the row order; another seed gives other
    resamples. Regression caught: an unseeded generator (flaky gates) or results depending on how rows were sorted."""
    rows = [make(f"c{i}", f"q{j}", i / 10, "correct") for i in range(12) for j in range(3)]
    a = aggregate(rows, seed=7)
    assert a == aggregate(rows, seed=7)
    assert a == aggregate(list(reversed(rows)), seed=7)
    assert a["headline"]["ci"] != aggregate(rows, seed=8)["headline"]["ci"]


def test_ci_widens_with_fewer_scripts():
    """Contract: with script scores alternating +1/-1 (mean 0), 4 scripts give CI [-1, 1] (a resample is all one sign
    with probability 2/16 = 12.5%, above 2.5%, so both percentiles sit at the extremes) while 40 scripts give a standard
    error of 1/sqrt(40) = 0.16 and a width near 0.62, under 0.9. Regression caught: a CI that does not shrink with
    sample size, for example one computed on the wrong axis."""
    few, many = aggregate(scripts(4), seed=1), aggregate(scripts(40), seed=1)
    assert few["headline"]["ci"] == (-1.0, 1.0) and width(few) == 2.0
    assert 0.3 < width(many) < 0.9


def test_cases_of_one_script_move_together():
    """Contract (design §9): the bootstrap resamples scripts, so the two cases of a script are in every resample
    together. Four scripts of two cases each, script scores alternating +1/-1, resampled by script give [-1, 1] (a
    resample is all one sign with probability 1/16 each way, above 2.5%). Resampled by case (8 cases, 4 of each sign) a
    resample would be all one sign with probability 1/256 each way, under 2.5%, and the interval would sit inside
    [-1, 1]. Regression caught: intervals too narrow for correlated windows of one world and path."""
    rep = aggregate(scripts(4, cases_each=2), seed=3)
    assert rep["headline"]["ci"] == (-1.0, 1.0) and (rep["scripts"], rep["cases"]) == (4, 8)


def test_ci_resamples_scripts_not_questions():
    """Contract: questions of one script move together. Four scripts with 1 question each and with 50 each (same
    per-script scores) have identical CIs; resampling questions would shrink the second by about sqrt(50). Regression
    caught: an i.i.d. question bootstrap, which makes CIs falsely tight because questions share a world."""
    one, many = aggregate(scripts(4, 1, 1), seed=3), aggregate(scripts(4, 1, 50), seed=3)
    assert one["headline"]["ci"] == many["headline"]["ci"] == (-1.0, 1.0)


def test_a_single_script_has_no_ci():
    """Contract: with one script the CI is None (no between-script variation to resample), the mean is still given.
    Regression caught: a degenerate zero-width interval read as certainty."""
    rep = aggregate(scripts(1, 2, 3), seed=1)
    assert rep["headline"]["ci"] is None and rep["headline"]["mean"] == 1.0
    assert rep["by"]["family"]["where_last"]["ci"] is None


def paired(scores_a, scores_b, **kw):
    """One question per script i: A scores scores_a[i], B scores_b[i]."""

    def mk(scores):
        return [make(f"c{i}", f"q{i}", s, "correct", **kw) for i, s in enumerate(scores)]

    return mk(scores_a), mk(scores_b)


def test_paired_difference_is_a_minus_b_on_shared_questions():
    """Contract: the mean difference is A minus B (positive when A is better), over the questions both answered only.
    Regression caught: the sign flipped, which would turn the gates' 'oracle - memory_table <= 0.02' and 'memory_table
    - shortcut > 0.30' into their opposites; or unshared questions entering the mean."""
    a, b = paired([1.0, 1.0, 1.0, 1.0], [0.0, 0.5, 0.25, 0.25])
    d = paired_difference(a, b, seed=1)
    assert d["mean"] == pytest.approx(0.75)
    assert paired_difference(b, a, seed=1)["mean"] == pytest.approx(-0.75)
    extra = a + [make("c9", "only_a", -1.0, "wrong")]
    d2 = paired_difference(extra, b, seed=1)
    assert (d2["n"], d2["scripts"], d2["mean"]) == (4, 4, pytest.approx(0.75))


def test_paired_difference_ci_bounds_come_from_script_resampling():
    """Contract: with per-script differences (0, 2, 0, 2) the mean is 1 and the 95% CI is [0, 2] (a resample is all-0
    or all-2 with probability 1/16 each, above 2.5%); constant differences give a zero-width CI at the constant.
    Regression caught: a CI around the wrong centre, or the lower bound reported where the upper is needed (the gates
    test 'upper CI' for knowable and 'lower CI' for separation)."""
    a, b = paired([2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 0.0])
    d = paired_difference(a, b, seed=5)
    assert d["mean"] == pytest.approx(1.0) and d["ci"] == pytest.approx((0.0, 2.0))
    a, b = paired([1.0] * 4, [0.75] * 4)
    assert paired_difference(a, b, seed=5)["ci"] == pytest.approx((0.25, 0.25))


def test_paired_difference_pools_questions_and_filters_with_where():
    """Contract: the statistic is the pooled mean of the differences (family X: 1 question, difference 1; family Y: 3
    questions, difference 0 give 1/4, not a mean of family means), and `where` keeps the rows of A whose keys equal its
    values: family Y alone 0, family X alone 1, scope `now` (none here) is no shared question. A key that is not a row
    key is refused. Regression caught: family means letting a small family outweigh its questions, or a filter that
    silently keeps everything."""
    a = [make("c1", "x1", 1.0, "correct", family="X", scope="now")]
    a += [make("c1", f"y{i}", 0.0, "abstained", family="Y") for i in range(3)]
    b = [make("c1", "x1", 0.0, "abstained", family="X", scope="now")]
    b += [make("c1", f"y{i}", 0.0, "abstained", family="Y") for i in range(3)]
    assert paired_difference(a, b, n_boot=20)["mean"] == pytest.approx(0.25)
    assert paired_difference(a, b, where={"family": "Y"}, n_boot=20)["mean"] == 0.0
    assert paired_difference(a, b, where={"family": "X"}, n_boot=20)["mean"] == 1.0
    assert paired_difference(a, b, where={"scope": "now"}, n_boot=20)["n"] == 1
    with pytest.raises(ValueError, match="share no questions"):
        paired_difference(a, b, where={"scope": "evidence"}, n_boot=20)
    with pytest.raises(ValueError, match="row keys"):
        paired_difference(a, b, where={"famly": "X"}, n_boot=20)


def test_paired_difference_refuses_disjoint_or_inconsistent_sets():
    """Contract: no shared questions, or one (case, qid) filed under another script or family, raises ValueError.
    Regression caught: comparing two systems that answered different cases and reporting a meaningless number."""
    a = [make("c1", "q", 1.0, "correct")]
    with pytest.raises(ValueError, match="share no questions"):
        paired_difference(a, [make("c2", "q", 1.0, "correct")], n_boot=10)
    with pytest.raises(ValueError, match="different script or family"):
        paired_difference(a, [make("c1", "q", 1.0, "correct", family="count")], n_boot=10)
    with pytest.raises(ValueError, match="different script or family"):
        paired_difference(a, [make("c1", "q", 1.0, "correct", script="other")], n_boot=10)


def test_gap_closed():
    """Contract: (S - floor) / (oracle - floor): 0 at the floor, 1 at the oracle, linear between; undefined when the
    oracle does not exceed the floor. Regression caught: the denominator or numerator reversed, or a division by zero
    on a degenerate gate input."""
    assert gap_closed(0.6, 0.2, 1.0) == pytest.approx(0.5)
    assert gap_closed(0.2, 0.2, 1.0) == 0.0
    assert gap_closed(1.0, 0.2, 1.0) == 1.0
    assert gap_closed(0.0, 0.2, 1.0) == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="exceed"):
        gap_closed(0.5, 0.5, 0.5)


def test_all_unknown_on_a_mixed_set_scores_exactly_the_unknown_gold_share():
    """Contract: the all-unknown system scores 0 on answerable questions and 1 on unknown golds, so its headline is
    exactly the unknown share: 2 unknown of 10 gives 0.2, with no wrong answers and recall 1. Regression caught: any
    change to the table that lets the floor drift from the unknown share, which would break the shortcut gates'
    'unknown share + 0.10' reference."""
    golds = [{**rec(pos(0, 0), qid=f"q{i}"), "scope": "memory", "reference": None, "critical": False}
             for i in range(8)]
    golds += [{**rec(UNKNOWN, slice="unknown", qid=f"u{i}"), "scope": "memory", "reference": None, "critical": False}
              for i in range(2)]
    info = {"script": "s1", "layout": "l", "robot": "r", "job": "j", "length": "short"}
    rows = [row(UNKNOWN, g, when="end", info={**info, "case": "c1" if i < 5 else "c2"}) for i, g in enumerate(golds)]
    rep = aggregate(rows, n_boot=20)
    assert rep["headline"]["mean"] == pytest.approx(0.2)
    assert rep["counts"] == {"correct": 2, "partial": 0, "wrong": 0, "abstained": 8}
    assert rep["confidently_wrong_rate"]["rate"] == 0.0 and rep["abstention"]["recall"] == 1.0
    assert rep["abstention"]["precision"] == pytest.approx(0.2)
    assert rep["above_floor"]["mean"] == pytest.approx(0.0)   # nothing above the floor: it is the floor


def test_slice_relevance_and_above_floor_by_hand():
    """Contract (FD-0.5.0-impl §2, "Reporting"): `by.slice` pools the answerable and the unknown rows apart;
    `by.relevance` splits the answerable rows into `critical` and `situational` by their tag and files every row of an
    unknown gold under `unknown`, tagged or not; `above_floor` is the headline less the unknown share. Worked on
    FIXTURE (12 rows, scores summing to 4, golds j and k unknown with scores 1 and -1): answerable (4 - 0)/10 = 0.4,
    unknown 0/2 = 0.0; critical rows a and e (1 + 1)/2 = 1.0, situational the other eight 2.0/8 = 0.25. The columns
    are now 3/5 and memory 1/12 (headline 41/120); the unknown shares are now 0 and memory (`ow`: 2/6, `own`: 0) 1/6, so
    the weighted share is 1/12 and the score above the floor 41/120 - 10/120 = 31/120. Regression caught: the unknown
    slice hidden in the headline (answering `unknown` throughout scores the weighted unknown share, 1/12 here, and only
    what is above it is skill), the pooled share 2/12 subtracted, or the critical tag read on an unknown gold."""
    rep = aggregate(FIXTURE, n_boot=50)
    by = rep["by"]
    assert {k: v["mean"] for k, v in by["slice"].items()} == pytest.approx({"answerable": 0.4, "unknown": 0.0})
    assert {k: v["n"] for k, v in by["slice"].items()} == {"answerable": 10, "unknown": 2}
    assert {k: v["mean"] for k, v in by["relevance"].items()} == pytest.approx(
        {"critical": 1.0, "situational": 0.25, "unknown": 0.0})
    assert {k: v["n"] for k, v in by["relevance"].items()} == {"critical": 2, "situational": 8, "unknown": 2}
    assert rep["above_floor"]["mean"] == pytest.approx(31 / 120)
    tagged = [{**r, "critical": True} if r["slice"] == "unknown" else r for r in FIXTURE]
    assert {k: v["n"] for k, v in aggregate(tagged, n_boot=50)["by"]["relevance"].items()} == {
        "critical": 2, "situational": 8, "unknown": 2}   # the tag does not move an unknown row
    assert aggregate([], n_boot=5)["above_floor"] == {"mean": None, "ci": None}


def test_above_floor_is_resampled_by_script_not_shifted():
    """Contract: the interval of `above_floor` comes from the bootstrap over scripts of the mean less the unknown
    share, so it moves with the share. Four scripts of one question each, every score 1: the headline is 1 in every
    resample (interval [1, 1]), while the score above the floor of a resample is its answerable share, 0 to 1 (a
    resample is all unknown with probability 1/16, above 2.5%, and all answerable the same), so [0, 1]. Regression
    caught: an interval for the score above the floor made by subtracting the point share from the headline's."""
    rows = [make(f"c{i}", f"q{i}", 1.0, "correct", script=f"s{i}", slice_="unknown" if i % 2 else "answerable")
            for i in range(4)]
    rep = aggregate(rows, seed=1)
    assert rep["headline"]["ci"] == (1.0, 1.0) and rep["above_floor"]["mean"] == pytest.approx(0.5)
    assert rep["above_floor"]["ci"] == pytest.approx((0.0, 1.0))


def test_paired_difference_reads_the_relevance_class():
    """Contract: `where={"relevance": ...}` keeps the rows of A of that class, an unknown gold under `unknown` whatever
    its tag. By hand, five questions over two scripts, A minus B: two critical answerable +1 and +1, a situational
    answerable -1, two unknown golds (the second tagged critical) +2 and 0: critical 2/2 = 1.0, situational -1.0,
    unknown 2/2 = 1.0, and all five 3/5 = 0.6. Regression caught: a critical difference that counts the unknown rows
    (the headline's difference stands in for it) or is taken over the tag alone."""
    questions = [(1.0, 0.0, {"critical": True}), (1.0, 0.0, {"critical": True}), (0.0, 1.0, {}),
                 (1.0, -1.0, {"slice_": "unknown"}), (0.0, 0.0, {"slice_": "unknown", "critical": True})]
    a = [make(f"c{i % 2}", f"q{i}", sa, "correct", script=f"s{i % 2}", **kw) for i, (sa, _, kw) in enumerate(questions)]
    b = [make(f"c{i % 2}", f"q{i}", sb, "correct", script=f"s{i % 2}", **kw) for i, (_, sb, kw) in enumerate(questions)]
    got = {c: paired_difference(a, b, where={"relevance": c}, n_boot=20)["mean"]
           for c in ("critical", "situational", "unknown")}
    assert got == pytest.approx({"critical": 1.0, "situational": -1.0, "unknown": 1.0})
    assert paired_difference(a, b, n_boot=20)["mean"] == pytest.approx(0.6)
