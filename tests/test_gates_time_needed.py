"""The `time_needed` gate (DECISIONS, "0.5.0's code review"): on a temporal family the larger score of the two
time-blind notebooks, `latest_table` and `first_table`, may not beat `all_unknown` by more than 0.10, and both are
among the shortcuts `separation` compares the table against. Hand rows, the family a synthetic one named in `temporal`
(the real change families are another lane's)."""

from __future__ import annotations

import pytest

from stbench.gates import BLIND, FIRST, GATES, LATEST, NOTEBOOKS, SHORTCUTS, evaluate, failing, format_table
from test_gates import N_BOOT, gates_of, rows, system_set

FAMILY = "change_where"
TEMPORAL = frozenset({FAMILY})
FLOOR = [0.25] * 4   # a notebook exactly at `all_unknown`'s mean on `system_set` (four scripts, one unknown in four)


def with_notebooks(latest, first=FLOOR, family=FAMILY, **options):
    """`system_set` of the family (four scripts of four questions, the last unknown: all_unknown 0.25) and the rows of
    `latest_table` and `first_table` with the given scores."""
    return system_set(family, **options) | {LATEST: rows(family, latest), FIRST: rows(family, first)}


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_either_notebook_over_the_floor_fails_time_needed_and_is_named(notebook):
    """Contract (DECISIONS, "0.5.0's code review"): on a family in `temporal`, `latest_table` and `first_table` must
    each be within 0.10 of `all_unknown`, the shortcut margin, because a family that asks a moved thing at one state
    only is answered by one of them. Hand rows, the unknown share 0.25: the notebook at 0.45 is 0.20 over and the gate
    fails on both verdicts and names that notebook (the other, at the floor, is not the worst), while the other gates,
    which a notebook of one place a thing does not touch, hold (the table still beats it by 0.55); at 0.35, exactly 0.10
    over, it passes. Regression caught: a temporal family that a one-sided notebook answers (it measures where things
    first or last were, not what changed), a gate that reads only one of the two, or a margin that is exclusive or read
    against zero."""
    latest = [0.45] * 4 if notebook == LATEST else FLOOR
    first = [0.45] * 4 if notebook == FIRST else FLOOR
    report = evaluate(with_notebooks(latest, first), temporal=TEMPORAL, n_boot=N_BOOT)
    g = report["rows"][0]["gates"]["time_needed"]
    assert g["pass"] is False and g["point"] is False and g["worst"] == notebook
    assert g["value"] == pytest.approx(0.20) and g["threshold"] == 0.10 and g["unknown_share"] == 0.25
    assert g["scores"] == {LATEST: pytest.approx(sum(latest) / 4), FIRST: pytest.approx(sum(first) / 4)}
    assert list(failing(report)) == [("apartment_a", "vacuum/1", FAMILY, "time_needed")]
    assert not report["passed"] and not report["passed_point"]


def test_both_notebooks_within_the_margin_pass_and_the_larger_one_is_read():
    """Contract: `time_needed` passes when the larger of the two notebooks' scores is at most `all_unknown` + 0.10: both
    at 0.35 (exactly 0.10 over) and one at 0.35 with the other far under it pass, and neither makes the other's score
    count; when both are over it, the larger is the value and the worst (latest 0.30 against first 0.45 is 0.20, not the
    mean 0.125 and not the 0.05 of the other). A notebook below the floor never offsets the other. Regression caught: a
    mean or a sum of the two notebooks, a gate that reads only the first it finds, or a notebook that scores under the
    floor (a wrong answer costs more than an abstention) hiding the other's excess."""
    both = evaluate(with_notebooks([0.35] * 4, [0.35] * 4), temporal=TEMPORAL, n_boot=N_BOOT)
    assert both["rows"][0]["gates"]["time_needed"]["pass"] is True and both["passed"] and list(failing(both)) == []
    one = gates_of(with_notebooks([0.35] * 4, [-0.9] * 4), temporal=TEMPORAL)["time_needed"]
    assert one["pass"] is True and one["worst"] == LATEST and one["value"] == pytest.approx(0.10)
    larger = gates_of(with_notebooks([0.30] * 4, [0.45] * 4), temporal=TEMPORAL)["time_needed"]
    assert larger["pass"] is False and larger["worst"] == FIRST and larger["value"] == pytest.approx(0.20)
    under = gates_of(with_notebooks([-0.9] * 4, [0.45] * 4), temporal=TEMPORAL)["time_needed"]
    assert under["pass"] is False and under["worst"] == FIRST and under["value"] == pytest.approx(0.20)
    mixed = gates_of(with_notebooks([0.3, 0.3, 0.3, 1.0]), temporal=TEMPORAL)["time_needed"]   # (0.9 + 1) / 4 = 0.475
    assert mixed["value"] == pytest.approx(0.225) and mixed["pass"] is False and mixed["worst"] == LATEST


def test_time_needed_reads_temporal_families_only_and_leaves_shortcuts_alone():
    """Contract: `time_needed` is `n/a` (neither a pass nor a fail, no note) for a family that is not in `temporal`,
    whatever the notebooks score there; neither is a shortcut of the `shortcuts` gate on any family (they are read by
    `time_needed` and `separation` alone). The rows that fail on the temporal family above pass the same family when
    it is not named temporal, and a temporal family and a plain one in one report are gated on their own. Regression
    caught: the gate applied to every family (a notebook that rightly answers a `now` family failing it), or a notebook
    counted among the learned shortcuts."""
    rows_ = with_notebooks([0.45] * 4, [0.45] * 4)
    g = gates_of(rows_, temporal=frozenset())
    assert g["time_needed"] == {"pass": None, "point": None, "evidence": None, "value": None, "threshold": 0.10}
    assert evaluate(rows_, n_boot=N_BOOT)["passed"]
    assert set(gates_of(rows_, temporal=TEMPORAL)["shortcuts"]["scores"]) == set(SHORTCUTS)
    assert not set(NOTEBOOKS) & set(SHORTCUTS)
    plain_rows = with_notebooks([0.45] * 4, [0.45] * 4, "plain")
    both = {n: [*r, *plain_rows[n]] for n, r in rows_.items()}
    report = evaluate(both, temporal=TEMPORAL, n_boot=N_BOOT)
    verdicts = {r["family"]: r["gates"]["time_needed"]["pass"] for r in report["rows"]}
    assert verdicts == {FAMILY: False, "plain": None}


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_each_notebook_is_a_shortcut_of_separation_on_temporal_families_only(notebook):
    """Contract: on a temporal family `separation` compares the table against the best of the shortcuts and the two
    notebooks; on any other family against the shortcuts alone. A notebook at 0.70 (blind at 0.30 is the best shortcut,
    the other notebook at the floor) leaves the table 0.30 above it, under the 0.40 it wants: separation fails on the
    temporal family and names that notebook, and passes on the same rows of a plain family against `blind` (1.0 - 0.3 =
    0.7). Regression caught: a temporal family that the table does not beat a notebook on, passing for want of that
    notebook among its rivals (the first as well as the latest)."""
    latest = [0.7] * 4 if notebook == LATEST else FLOOR
    first = [0.7] * 4 if notebook == FIRST else FLOOR
    g = gates_of(with_notebooks(latest, first), temporal=TEMPORAL)["separation"]
    assert g["pass"] is False and g["point"] is False and g["best"] == notebook and g["value"] == pytest.approx(0.30)
    plain = gates_of(with_notebooks(latest, first), temporal=frozenset())["separation"]
    assert plain["pass"] is True and plain["best"] == BLIND and plain["value"] == pytest.approx(0.70)


def test_a_temporal_family_without_a_notebooks_rows_is_na_with_a_note_not_a_pass():
    """Contract (the gates' rule for a system with no rows): a temporal family whose notebooks did not run leaves
    `time_needed` and `separation` `n/a` with a note naming what is missing (a pass proven on part of the systems is not
    a pass), and lists them as missing; so does one notebook's absence when the other holds the margin, but a notebook
    that fails stands whatever is missing. On a family that is not temporal the same absence is no one's business (no
    `missing` entry, no note), as it is for the reports made before the notebooks. Regression caught: a gate that passes
    because the system that would fail it left no rows, or a plain family reported as missing a system no gate reads."""
    results = system_set(FAMILY)
    report = evaluate(results, temporal=TEMPORAL, n_boot=N_BOOT)
    (row,) = report["rows"]
    for name in ("time_needed", "separation"):
        assert row["gates"][name]["pass"] is None
    assert row["gates"]["time_needed"]["note"] == f"no rows for {LATEST}, {FIRST}"
    assert row["gates"]["separation"]["note"] == f"no rows for {LATEST}, {FIRST}"
    assert row["missing"] == [LATEST, FIRST] and report["passed"]
    plain = evaluate(results, n_boot=N_BOOT)["rows"][0]
    assert plain["missing"] == [] and "note" not in plain["gates"]["time_needed"]
    assert "note" not in plain["gates"]["separation"] and plain["gates"]["separation"]["pass"] is True
    only_latest = results | {LATEST: rows(FAMILY, FLOOR)}
    (partial,) = evaluate(only_latest, temporal=TEMPORAL, n_boot=N_BOOT)["rows"]
    assert partial["gates"]["time_needed"]["pass"] is None and partial["missing"] == [FIRST]
    assert partial["gates"]["time_needed"]["note"] == f"no rows for {FIRST}"
    failing_alone = results | {FIRST: rows(FAMILY, [0.45] * 4)}
    (failed,) = evaluate(failing_alone, temporal=TEMPORAL, n_boot=N_BOOT)["rows"]
    assert failed["gates"]["time_needed"]["pass"] is False and failed["gates"]["time_needed"]["worst"] == FIRST


def test_the_table_prints_the_time_needed_column():
    """Contract: `time_needed` is the last gate column of `format_table` (before `evid` and `verdict`) and of `GATES`,
    printing a verdict, value and threshold as the others do: `FAIL +0.20 vs 0.1` for a notebook 0.20 over, `n/a` for a
    plain family. Regression caught: a gate in the report that the printed table leaves out, or a header out of step
    with the rows."""
    assert GATES[-1] == "time_needed"
    lines = format_table(evaluate(with_notebooks([0.45] * 4), temporal=TEMPORAL, n_boot=N_BOOT)).splitlines()
    assert lines[0].split()[-3:] == ["time_needed", "evid", "verdict"]
    assert "FAIL  +0.20 vs 0.1" in lines[1] and lines[1].index("FAIL  +0.20") == lines[0].index("time_needed")
    assert lines[-1] == "GATES FAIL"
    plain = format_table(evaluate(with_notebooks([0.45] * 4), n_boot=N_BOOT)).splitlines()
    assert plain[1].split()[-3] == "n/a"
