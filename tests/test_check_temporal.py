"""`stbench check` and the temporal families (FD-0.5.0-impl §2, §5): the share of each job's and length's answerable
end questions that are temporal, how a failed `time_needed` is named, and the whole command over a synthetic dataset
with one family made temporal, which must run both notebooks and hand the registry's temporal set to the gates."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import replace

import pytest

from check_kit import hand_synth, synthetic_dataset
from run_kit import golds
from stbench import check, cli, gates, suite_report
from stbench.case import CasePaths
from stbench.questions_text import FAMILIES
from stream_kit import UNKNOWN, case_on, during_q, end_q, gold_row
from test_gates import N_BOOT
from test_gates_time_needed import FAMILY, FLOOR, TEMPORAL, with_notebooks

CHANGE = "change_x"   # a family these tests call temporal


def asked(root, seed: int, ends: list[tuple[str, bool]], *, tasks: int = 1, length: str = "short") -> CasePaths:
    """A case over a hand-made script of `tasks` tasks, asking a during `path_ahead` and, at the end, one question of
    each `(family, unknown)` of `ends`, with hand-written golds: answerable, or `unknown` where it says so."""
    qs = [during_q(1, 12), *(end_q(n, template=family, slots={"n": n}) for n, (family, _) in enumerate(ends, 2))]
    case = case_on(root, hand_synth(seed, tasks=tasks), qs, b=tasks - 1, length=length)
    hidden = {"gold": UNKNOWN, "slice": "unknown"}
    golds(case, rows=[gold_row(qs[0]),
                      *(gold_row(q, **(hidden if unknown else {}))
                        for q, (_, unknown) in zip(qs[1:], ends, strict=True))])
    return case


def test_the_temporal_share_is_read_per_length_from_answerable_end_questions(tmp_path):
    """Contract (FD-0.5.0-impl §5: change over time is claimed only where temporal families carry at least 20% of a
    job-and-length cell's answerable end questions and at least one a case): `temporal_shares` counts, for each length
    in the order short, medium, long, the cases, their end questions with an answerable gold and those of a temporal
    family, the share and the temporal ones a case. Hand cases: two short, the first asking a plain and a temporal
    question, the second a plain one and a temporal one with an unknown gold (not answerable), and one medium case of
    two temporal and one plain; every case also asks a during question, which is no end question. Short: 3 answerable,
    1 temporal, a third, half a case; medium: 3 and 2, two thirds, 2 a case. With no temporal family the counts are 0;
    a length with no answerable end question has no share. Regression caught: the during or the unknown questions
    counted, the share taken over every length at once, or a gate gone unread for want of the number it is judged by."""
    a = asked(tmp_path, 1, [("object_where", False), (CHANGE, False)])
    b = asked(tmp_path, 2, [("object_where", False), (CHANGE, True)])
    c = asked(tmp_path, 3, [(CHANGE, False), (CHANGE, False), ("object_where", False)], tasks=2, length="medium")
    shares = suite_report.temporal_shares([c, a, b], frozenset({CHANGE}))
    assert list(shares) == ["short", "medium"]
    assert shares["short"] == {"cases": 2, "answerable": 3, "temporal": 1, "share": pytest.approx(1 / 3),
                               "per_case": 0.5}
    assert shares["medium"] == {"cases": 1, "answerable": 3, "temporal": 2, "share": pytest.approx(2 / 3),
                                "per_case": 2.0}
    assert suite_report.temporal_shares([a, b, c], frozenset())["short"]["temporal"] == 0
    hidden = asked(tmp_path, 4, [(CHANGE, True)])
    assert suite_report.temporal_shares([hidden], frozenset({CHANGE})) == {
        "short": {"cases": 1, "answerable": 0, "temporal": 0, "share": None, "per_case": 0.0}}


@pytest.mark.parametrize("notebook", gates.NOTEBOOKS)
def test_a_failed_time_needed_is_named_with_the_notebook_and_its_margin(notebook):
    """Contract (VISION rule 5, plan §4.14): `check` says which gate failed, by how much and which notebook did it: a
    temporal family that `latest_table` (or `first_table`) answers 0.20 over the floor is named with that notebook, that
    number and the 0.1 the gate allows. Hand rows as the gate's own test. Regression caught: a `time_needed` failure
    with no line (a family that fails but is not named, so the verdict fails without a reason), a line that names the
    wrong notebook (the other one scored the floor), or a template that reads another gate's keys."""
    latest = [0.45] * 4 if notebook == gates.LATEST else FLOOR
    first = [0.45] * 4 if notebook == gates.FIRST else FLOOR
    report = gates.evaluate(with_notebooks(latest, first), temporal=TEMPORAL, n_boot=N_BOOT)
    assert check.outcome(report, "point") == (False, [
        f"FAIL apartment_a vacuum/1 {FAMILY} time_needed: {notebook}, a notebook that keeps one place for each thing, "
        "beats all_unknown by +0.200; the gate allows at most 0.1."])


@pytest.fixture(scope="module")
def checked(tmp_path_factory):
    """`stbench check --verdict point` over four synthetic scripts compiled with the registry as it is, then checked
    with `object_where` made temporal (the synthetic dataset has no change family): (the dataset's cases, the report,
    what it printed, what it printed on stderr)."""
    root = tmp_path_factory.mktemp("temporal")
    cases = synthetic_dataset(root, 4)
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        patch.setitem(FAMILIES, "object_where", replace(FAMILIES["object_where"], temporal=True))
        code = cli.main(["check", "--data", str(root), "--verdict", "point", "--workers", "1"])
    assert code == 1   # the worlds are alike: every prior memorises them and the gates fail
    report = json.loads((root / "check_report.json").read_text(encoding="utf-8"))
    return cases, report, out.getvalue(), err.getvalue()


def test_check_runs_both_notebooks_and_hands_the_temporal_set_to_the_gates(checked):
    """Contract (DECISIONS, "0.5.0's code review"): `check` runs `latest_table` and `first_table` over every group
    beside the other systems, and passes the registry's temporal families (`FamilySpec.temporal`) to the gates, so
    `time_needed` reads the notebooks' rows on exactly those families: with `object_where` temporal, its
    `time_needed` is the larger of the notebooks' scores minus `all_unknown`'s, judged against 0.1 (the worlds do not
    change, so each notebook answers it like the oracle and it fails, the notebook named on stderr), and
    `path_ahead`'s is `n/a`.
    Regression caught: a gate gone `n/a` for want of rows (a notebook not run, or the temporal set not passed), a
    notebook run on some families only, or a gate that reads one of the two."""
    _, report, _, err = checked
    by_family = {g["family"]: g for g in report["gates"]}
    assert all(name in g["scores"] for g in report["gates"] for name in gates.NOTEBOOKS)
    needed = by_family["object_where"]["gates"]["time_needed"]
    scores = by_family["object_where"]["scores"]
    assert needed["worst"] in gates.NOTEBOOKS and needed["threshold"] == 0.1
    assert needed["value"] == pytest.approx(max(scores[n]["mean"] for n in gates.NOTEBOOKS)
                                            - scores["all_unknown"]["mean"])
    assert needed["pass"] is False and needed["point"] is False and needed["value"] > 0.1
    assert by_family["path_ahead"]["gates"]["time_needed"]["pass"] is None
    assert f"FAIL apartment_a vacuum/1 object_where time_needed: {needed['worst']}, a notebook that keeps" in err


def test_check_prints_and_reports_the_temporal_share_of_each_length(checked):
    """Contract (FD-0.5.0-impl §2): `check` prints, for each job, each length's temporal share of the answerable end
    questions (with the count and the temporal ones a case) and puts the same numbers in the report's group. With
    `object_where` temporal it is every end question the synthetic dataset asks, so the share is 1.00 and the count is
    the dataset's own, counted here from its question and gold files. Regression caught: a share computed over the wrong
    questions (during ones, unknown golds) or printed for no length."""
    cases, report, printed, _ = checked
    answerable = 0
    for case in cases:
        when = {q["qid"]: q["when"] for q in map(json.loads, case.questions.read_text().splitlines())}
        answerable += sum(when[g["qid"]] == "end" and g["slice"] != "unknown"
                          for g in map(json.loads, case.gold.read_text().splitlines()))
    (group,) = report["groups"]
    short = group["temporal"]["short"]
    assert list(group["temporal"]) == ["short"] and answerable > 0
    assert (short["cases"], short["answerable"], short["temporal"], short["share"]) == (4, answerable, answerable, 1.0)
    assert short["per_case"] == answerable / 4
    assert f"temporal share of answerable end questions: short 1.00 ({answerable} of {answerable}, " in printed
