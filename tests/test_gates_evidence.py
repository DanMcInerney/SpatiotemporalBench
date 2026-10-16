"""The evidence verdict (`gates_evidence`, FD-0.5.0-control §8) on hand-made result rows: a shortcut-type gate fails
only where a system's paired gain over `all_unknown` is over the margin with an interval above 0, read in the whole
group, in each length tier and in each slice; the point and interval verdicts are read as they were; BROKEN needs the
same evidence; and `stbench check --verdict evidence` follows it."""

from __future__ import annotations

import contextlib
import io
import json
import re

import pytest

from check_kit import synthetic_dataset
from score_kit import result_row
from stbench import case, check, cli, gates, gates_evidence, suite_balance, suite_report
from stbench.gates import BLIND, CURRENT, CURRENT_PRIOR, FIRST, FORM, LATEST, ORACLE, TABLE, evaluate
from test_gates import N_BOOT, gates_of

LAYOUT, JOB, FAMILY = "downtown_a", "drone/1", "zone_clear"
NAMES = (*gates.BASELINE_NAMES, FORM)
COIN = [7, 2, 6, 3, 5, 4, 6, 2, 7, 3, 5, 3]   # of 8 questions a script, how many a blind guesser has right: 53 of 96


def rows(score, *, scripts, per=4, unknown=(), scopes=None, lengths=None, family=FAMILY) -> list[dict]:
    """Rows of one system: `scripts` scripts of one case each, `per` questions a case; `score(e, q)` is the penalised
    score of question q of script e, `unknown` the questions with an unknown gold, `scopes[q]` a question's scope (all
    `now` by default) and `lengths[e]` a case's length (all `short` by default)."""
    out = []
    for e in range(scripts):
        for q in range(per):
            s = score(e, q)
            outcome = "correct" if s >= 1 else "wrong" if s < 0 else "abstained"
            out.append(result_row(f"{family}-c{e}", f"q{q}", s, outcome, script=f"{family}-s{e}", family=family,
                                  when="during", scope=(scopes or ["now"] * per)[q],
                                  slice_="unknown" if q in unknown else "answerable", layout=LAYOUT, job=JOB,
                                  length=(lengths or ["short"] * scripts)[e]))
    return out


def world(systems: dict | None = None, **options) -> dict[str, list[dict]]:
    """A healthy family, every system of the gates over the same questions: the oracle and `memory_table` exact, the
    rest at `all_unknown`'s level (0 on an answerable gold, 1 on an unknown one); `systems` replaces some of them by
    their `score(e, q)`. The options are `rows`'."""
    unknown = options.get("unknown", ())
    base = {n: (lambda e, q: 1 if q in unknown else 0) for n in NAMES}
    base |= {ORACLE: lambda e, q: 1, TABLE: lambda e, q: 1}
    return {n: rows(score, **options) for n, score in (base | (systems or {})).items()}


def shortcuts_of(results, **kwargs) -> dict:
    return gates_of(results, **kwargs)["shortcuts"]


def test_a_tilt_that_chance_explains_passes_the_evidence_verdict_and_fails_the_point_one():
    """Contract (the drone's `blind` on `zone_clear`, 2026-10-08): a shortcut at +0.104 over `all_unknown` on a
    balanced coin of 96 questions is not shown. Twelve scripts of eight yes/no questions, `blind` right on 53 of the 96
    (score +1, else -1, so a mean of (53 - 43) / 96 = 0.104, past the 0.10 margin), the right ones spread unevenly over
    the scripts: the point verdict and the interval verdict fail, naming `blind`, and the evidence verdict passes, as
    the paired interval reaches below 0 in every cell. Regression caught: a family dropped for a tilt that a fair coin
    makes one time in six (the coin must be balanced, the gate must not fail it), or an evidence reading that follows
    the point estimate."""
    report = evaluate(world({BLIND: lambda e, q: 1 if q < COIN[e] else -1}, scripts=12, per=8), n_boot=N_BOOT)
    g = report["rows"][0]["gates"]["shortcuts"]
    assert g["value"] == pytest.approx(10 / 96) and g["worst"] == BLIND
    assert g["pass"] is False and g["point"] is False and g["evidence"] is True
    assert not report["passed"] and not report["passed_point"] and report["passed_evidence"]
    assert check.outcome(report, "evidence") == (True, [])
    assert [cell["cell"] for cell in g["cells"]] == [["all"], ["length", "short"], ["slice", "answerable"]]
    for cell in g["cells"]:
        assert cell["mean"] == pytest.approx(10 / 96) and cell["scripts"] == 12 and cell["n"] == 96
        assert cell["ci"][0] < 0 < cell["ci"][1] and cell["counts"] is False


def test_the_length_cells_follow_the_order_of_the_case_lengths():
    """Contract: the gate modules sit on the baseline side of the import fences and cannot read `case`, so
    `gates_evidence.LENGTHS` is the order of `case.LENGTHS` written out, and the cells of a report run short, medium,
    long whatever the order of the rows. Regression caught: a length added or renamed in `case` that the evidence cells
    then list last or not at all."""
    assert tuple(case.LENGTHS) == gates_evidence.LENGTHS
    cells = gates_of(world(scripts=4, lengths=["long", "short", "medium", "short"]))["shortcuts"]["cells"]
    assert [tuple(c["cell"]) for c in cells] == [
        ("all",), ("length", "short"), ("length", "medium"), ("length", "long"), ("slice", "answerable")]


def test_a_gain_on_the_answerable_slice_that_the_pooled_mean_hides_fails_the_evidence_verdict():
    """Contract (the helper's short `distance_between`, +0.375 on its answerable questions): a constant gain on the
    answerable slice, with an interval above 0, fails the evidence verdict though the pooled reading passes. Twelve
    scripts of four questions, the first answerable and the other three unknown (`all_unknown` scores 0, 1, 1, 1);
    `blind` scores 0.375 on the answerable one and 1 on the unknown ones, as `all_unknown` does: over the whole group
    its gain is 0.375 / 4 = 0.094, under the 0.10 margin, so the point and interval verdicts of `shortcuts` hold, and
    on the answerable slice it is +0.375 in every script, so the slice cell counts and names `blind`. The unknown
    slice, where `all_unknown` scores 1, and the group as a whole do not count. Regression caught: a shortcut that
    answers the few questions that have an answer and abstains elsewhere, hidden by the unknown share around it."""
    unknown = (1, 2, 3)
    results = world({BLIND: lambda e, q: 0.375 if q == 0 else 1}, scripts=12, unknown=unknown)
    g = shortcuts_of(results)
    assert g["pass"] is True and g["point"] is True and g["value"] == pytest.approx(0.375 / 4)
    assert g["evidence"] is False
    counting = {tuple(c["cell"]): c for c in g["cells"] if c["counts"]}
    assert set(counting) == {("slice", "answerable")}
    cell = counting["slice", "answerable"]
    assert cell["worst"] == BLIND and cell["mean"] == pytest.approx(0.375) and cell["ci"] == pytest.approx((0.375,) * 2)
    unknown_slice = next(c for c in g["cells"] if c["cell"] == ["slice", "unknown"])
    assert unknown_slice["mean"] == 0 and unknown_slice["counts"] is False
    report = evaluate(results, n_boot=N_BOOT)
    assert report["rows"][0]["gates"]["shortcuts"]["evidence"] is False and not report["passed_evidence"]
    assert check.outcome(report, "evidence")[0] is False


def test_a_gain_in_one_length_tier_is_named_by_its_cell():
    """Contract: the evidence reading is per length tier, and `check` names the failing cell, the system and its gain
    with the interval. Twelve scripts of four answerable questions, four of each length; `blind` scores 0.15, 0.35,
    0.15 and 0.35 (a script's questions alike) in the short ones and nothing in the others, so its gain is 0.25 in the
    short cell (the 6.25% of resamples that draw only the lower, or only the higher, bound the 95% interval at 0.15
    and 0.35) and 0.25 / 3 = 0.083 over the group, under the margin: the point reading passes. The medium and long
    cells hold no gain. Regression caught: a tier's shortcut averaged away by the other two."""
    lengths = ["short"] * 4 + ["medium"] * 4 + ["long"] * 4
    gain = [0.15, 0.35, 0.15, 0.35] + [0] * 8
    results = world({BLIND: lambda e, q: gain[e]}, scripts=12, lengths=lengths)
    report = evaluate(results, n_boot=N_BOOT)
    g = report["rows"][0]["gates"]["shortcuts"]
    assert g["point"] is True and g["evidence"] is False
    assert {tuple(c["cell"]): c["counts"] for c in g["cells"]} == {
        ("all",): False, ("length", "short"): True, ("length", "medium"): False, ("length", "long"): False,
        ("slice", "answerable"): False}
    passed, lines = check.outcome(report, "evidence")
    assert passed is False and check.outcome(report, "point") == (True, [])
    (line,) = lines
    assert line == ("FAIL downtown_a drone/1 zone_clear shortcuts [length short]: blind beats all_unknown by +0.250 "
                    "(95% interval +0.150 to +0.350); the gate counts a gain over 0.1 whose interval is above 0.")
    table = gates.format_table(report).splitlines()
    assert table[-3] == ("EVID downtown_a drone/1 zone_clear shortcuts [length short]: blind beats all_unknown by "
                         "+0.250 (95% interval +0.150 to +0.350)")
    assert table[-2:] == ["EVIDENCE VERDICT FAILS", "ALL GATES PASS"]   # the interval reading passes it


def test_a_cell_of_one_script_cannot_count_and_is_noted():
    """Contract: a cell of fewer than two scripts has no interval, so no system counts in it, and the gate says which
    cells those are. Six scripts of four answerable questions, five short and one long; `blind` scores 0.5 in the long
    script: +0.5 in the long cell, but of one script (`ci` None), and 0.5 / 6 = 0.083 over the group, so the evidence
    verdict passes with a note naming `length long`. With two long scripts, both at 0.5, the same cell has an interval
    (0.5 to 0.5) and counts; the group, whose resamples miss both long scripts one time in eleven, does not.
    Regression caught: a count on a cell with no interval (a lone script's luck), or a cell that silently cannot
    fail."""
    one = shortcuts_of(world({BLIND: lambda e, q: 0.5 if e == 5 else 0}, scripts=6, lengths=["short"] * 5 + ["long"]))
    long_cell = next(c for c in one["cells"] if c["cell"] == ["length", "long"])
    assert long_cell["mean"] == pytest.approx(0.5) and long_cell["ci"] is None and long_cell["scripts"] == 1
    assert long_cell["counts"] is False and one["evidence"] is True
    assert "length long" in one["evidence_note"] and "cannot count" in one["evidence_note"]
    assert "length short" not in one["evidence_note"]
    both = ["short"] * 4 + ["long"] * 2
    two = shortcuts_of(world({BLIND: lambda e, q: 0.5 if e >= 4 else 0}, scripts=6, lengths=both))
    assert two["evidence"] is False and "evidence_note" not in two
    assert {tuple(c["cell"]) for c in two["cells"] if c["counts"]} == {("length", "long")}


def test_a_dataset_of_one_script_has_no_cell_that_counts():
    """Contract: with one script in all, no cell has an interval, so the evidence verdict of the shortcut gates holds
    whatever the shortcut scores, while the point verdict (a pooled mean over the margin) fails; each cell says it has
    one script. Regression caught: M1-style one-script groups failing an evidence reading that needs the spread
    between scripts."""
    g = shortcuts_of(world({BLIND: lambda e, q: 0.9}, scripts=1))
    assert g["point"] is False and g["evidence"] is True
    assert all(c["ci"] is None and c["scripts"] == 1 and not c["counts"] for c in g["cells"])
    assert g["evidence_note"].startswith("all rows") and "cannot count" in g["evidence_note"]


def test_memory_needed_is_read_on_memory_rows_only_and_is_na_on_a_family_of_now_rows():
    """Contract: `memory_needed`'s evidence reading, like its point verdict, counts the `memory` rows alone. Twelve
    scripts of four questions, two `now` and two `memory` (the last unknown): `current_view`, the reference of the
    `now` rows, scores 1 on them and so beats `all_unknown` by 1 there, and it is not read; `current_view_prior` gains
    0.3 on the answerable memory question, 0.15 over the memory rows, past the 0.05 margin with an interval above 0:
    the gate fails by evidence naming it, in the cells of its memory rows (all of them, and the answerable ones). A
    family of `now` rows only has no memory row: `n/a`, with an `evidence` of None. Regression caught: the reference
    of a `now` family failing the memory gate, or a family with nothing to remember counted as a pass."""
    scopes, unknown = ["now", "now", "memory", "memory"], (3,)
    systems = {CURRENT: lambda e, q: 1 if q < 2 else 0 if q == 2 else 1}
    ok = gates_of(world(systems, scripts=12, scopes=scopes, unknown=unknown))["memory_needed"]
    assert ok["pass"] is True and ok["evidence"] is True and ok["value"] == 0
    failing = gates_of(world(systems | {CURRENT_PRIOR: lambda e, q: 0.3 if q == 2 else 1 if q == 3 else 0},
                             scripts=12, scopes=scopes, unknown=unknown))["memory_needed"]
    assert failing["evidence"] is False and failing["worst"] == CURRENT_PRIOR
    counting = {tuple(c["cell"]): c for c in failing["cells"] if c["counts"]}
    assert set(counting) == {("all",), ("length", "short"), ("slice", "answerable")}   # the memory rows, as read
    assert counting["all",]["mean"] == pytest.approx(0.15) and counting["all",]["n"] == 24
    assert counting["slice", "answerable"]["mean"] == pytest.approx(0.3) and counting["slice", "answerable"]["n"] == 12
    assert all(c["worst"] == CURRENT_PRIOR and c["ci"][0] > 0 for c in counting.values())
    (everything,) = [r for r in evaluate(world({CURRENT_PRIOR: lambda e, q: 0.3 if q == 2 else 1 if q == 3 else 0},
                                               scripts=12, scopes=scopes, unknown=unknown),
                                         n_boot=N_BOOT)["rows"]]
    assert everything["scope"] == "mixed" and everything["evidence"] is False
    passed, lines = check.outcome({"rows": [everything], "passed_evidence": False}, "evidence")
    assert passed is False and len(lines) == 3
    assert lines[0].startswith("FAIL downtown_a drone/1 zone_clear memory_needed [all rows]: current_view_prior beats "
                               "all_unknown by +0.150 (95% interval +0.150 to +0.150) on memory rows; the gate counts "
                               "a gain over 0.05")
    assert "[slice answerable]: current_view_prior beats all_unknown by +0.300" in lines[2]
    now = gates_of(world({CURRENT: lambda e, q: 1}, scripts=12))["memory_needed"]
    assert now["pass"] is None and now["point"] is None and now["evidence"] is None and "cells" not in now


def test_time_needed_counts_a_notebook_by_the_same_evidence():
    """Contract: `time_needed` is read as `shortcuts` is, the two notebooks against `all_unknown`: `first_table` scoring
    0.45 on every answerable question, 0.2 over the margin with a constant interval, fails by evidence and by the point
    verdict, naming the notebook in the line; at the floor it holds. A family that is not temporal is `n/a`. Regression
    caught: the notebooks left out of the evidence reading, which would let a time-blind answer pass when the shortcuts
    gate is relaxed."""
    temporal = frozenset({FAMILY})
    g = gates_of(world({FIRST: lambda e, q: 0.45, LATEST: lambda e, q: 0}, scripts=6), temporal=temporal)["time_needed"]
    assert g["point"] is False and g["evidence"] is False and g["worst"] == FIRST
    assert [(c["cell"], c["worst"]) for c in g["cells"] if c["counts"]] == [
        (["all"], FIRST), (["length", "short"], FIRST), (["slice", "answerable"], FIRST)]
    assert gates_of(world(scripts=6), temporal=temporal)["time_needed"]["evidence"] is True
    assert gates_of(world({FIRST: lambda e, q: 0.45}, scripts=6))["time_needed"]["evidence"] is None


def test_the_other_gates_keep_their_point_verdict_as_their_evidence():
    """Contract (§8): `ceiling`, `knowable` and `separation` have no evidence reading of their own: their `evidence` is
    their point verdict, interval or not. A table 0.04 under the oracle in alternating scripts has a point gap of 0.02
    (held) and an upper bound of 0.04 (the interval verdict fails): `knowable` is `pass` False, `point` True,
    `evidence` True; one script has no interval at all, and `knowable` and `separation` read the same; an oracle at 0.94
    fails `ceiling` on all three, and `check` names it under `--verdict evidence`. Regression caught: an interval that
    the evidence verdict inherits (a one-script group could never pass it) or a point failure that it forgets."""
    split = gates_of(world({TABLE: lambda e, q: 1 if e % 2 == 0 else 0.96}, scripts=4))["knowable"]
    assert (split["pass"], split["point"], split["evidence"]) == (False, True, True)
    one = gates_of(world(scripts=1))
    for name in ("knowable", "separation"):
        assert (one[name]["pass"], one[name]["point"], one[name]["evidence"]) == (False, True, True)
    low = world({ORACLE: lambda e, q: 0.94}, scripts=4)
    bad = gates_of(low)["ceiling"]
    assert (bad["pass"], bad["point"], bad["evidence"]) == (False, False, False)
    passed, lines = check.outcome(evaluate(low, n_boot=N_BOOT), "evidence")
    assert passed is False and len(lines) == 1 and "ceiling: the oracle scores 0.940" in lines[0]


def test_the_point_and_interval_verdicts_stay_on_the_pooled_means_whatever_the_evidence_says():
    """Contract: the evidence keys are additions. A gate's `pass` and `point` of the hand rows of tests/test_gates.py
    are what they were (those tests assert them); here three families whose evidence differs from their pooled reading
    are held to the hand values: the coin (pooled 0.104: both False), the slice gain (pooled 0.094: both True) and a
    shortcut at the floor (True), and `check.outcome` follows each verdict on its own. Regression caught: `evidence`
    replacing `point` in `passed_point`, or the point and interval readings looking into the cells."""
    coin = evaluate(world({BLIND: lambda e, q: 1 if q < COIN[e] else -1}, scripts=12, per=8), n_boot=N_BOOT)
    gain = evaluate(world({BLIND: lambda e, q: 0.375 if q == 0 else 1}, scripts=12, unknown=(1, 2, 3)), n_boot=N_BOOT)
    quiet = evaluate(world(scripts=12), n_boot=N_BOOT)
    assert [(r["rows"][0]["gates"]["shortcuts"]["pass"], r["rows"][0]["gates"]["shortcuts"]["point"])
            for r in (coin, gain, quiet)] == [(False, False), (True, True), (True, True)]
    assert quiet["passed"] and quiet["passed_point"] and quiet["passed_evidence"]
    assert check.outcome(coin, "point")[0] is False and check.outcome(coin, "interval")[0] is False
    assert check.outcome(coin, "evidence")[0] is True


def test_the_table_shows_each_groups_scope_and_its_evidence_verdict():
    """Contract: `format_table` has a `scope` column read from the oracle's rows (`now`, `memory` or `mixed`), so a
    family of `now` questions is in a column of its own, and an `evid` column, `ok` or `FAIL`, before `verdict`.
    Regression caught: a `now` family indistinguishable from a memory one in the gate table, or an evidence verdict that
    the table leaves out."""
    parts = [world(family="now_family", scopes=["now"] * 4, scripts=4),
             world(family="mixed_family", scopes=["now", "now", "memory", "memory"], scripts=4),
             world(family="memory_family", scopes=["memory"] * 4, scripts=4)]
    results = {n: [r for part in parts for r in part[n]] for n in NAMES}
    table = gates.format_table(evaluate(results, n_boot=N_BOOT)).splitlines()
    head = table[0].split()
    assert head[3] == "scope" and head[-2:] == ["evid", "verdict"]
    scopes_by_family = {line.split()[2]: line.split()[3] for line in table[1:4]}
    assert scopes_by_family == {"memory_family": "memory", "mixed_family": "mixed", "now_family": "now"}
    assert all(line.split()[-2] == "ok" for line in table[1:4]) and "EVIDENCE VERDICT PASSES" in table


def reach(rate: float = 0.5) -> dict:
    return {(LAYOUT, JOB): {"families": {FAMILY: {
        "hit_rates": dict.fromkeys(("blind", "path_log", "current_view_prior"), rate),
        "reach_rates": dict.fromkeys(("blind", "path_log", "current_view_prior"), rate)}}}}


def test_a_learned_shortcut_far_under_the_floor_is_broken_only_with_evidence():
    """Contract: BROKEN (a learned shortcut that does worse than abstaining) needs the evidence the shortcut verdict
    needs: its mean more than the margin under `all_unknown` and its interval under it too. Six scripts of two
    questions, the second unknown (`all_unknown` 0.5). `blind` scoring -0.24 on the answerable one is 0.38, 0.12 under:
    BROKEN when every script does alike (interval 0.38 to 0.38), not when scripts differ so that the interval, 0.02 to
    0.75, reaches over the floor, and not with one script, which has no interval. Regression caught: a learned
    shortcut called broken on a bad draw of scripts, or on a group too small to have an interval."""
    steady = world({BLIND: lambda e, q: 1 if q else -0.24}, scripts=6, per=2, unknown=(1,))
    wide = [1, -1, 1, -1, -1, -0.44]   # mean -0.24, spread over scripts
    varied = world({BLIND: lambda e, q: 1 if q else wide[e]}, scripts=6, per=2, unknown=(1,))
    alone = world({BLIND: lambda e, q: 1 if q else -0.24}, scripts=1, per=2, unknown=(1,))
    (line,) = suite_report.broken_instruments(evaluate(steady, n_boot=N_BOOT), reach())
    assert line.startswith("BROKEN downtown_a drone/1 zone_clear: blind scores 0.380 (95% interval 0.380 to 0.380), "
                           "0.120 under all_unknown (0.500)")
    report = evaluate(varied, n_boot=N_BOOT)
    own = report["rows"][0]["gates"]["shortcuts"]["pooled"][BLIND]
    assert own["mean"] == pytest.approx(0.38) and own["ci"][1] > 0.5   # the floor is inside the interval
    assert suite_report.broken_instruments(report, reach()) == []
    only = evaluate(alone, n_boot=N_BOOT)
    assert only["rows"][0]["gates"]["shortcuts"]["pooled"][BLIND]["ci"] is None
    assert suite_report.broken_instruments(only, reach()) == []


@pytest.fixture(scope="module")
def checked(tmp_path_factory):
    """A synthetic dataset of four compiled cases and the real `check` over it with the runs cached: the instrument runs
    once, however many verdicts a test asks of it (`check._check_runs` is the only call that runs systems)."""
    root, cache, real = tmp_path_factory.mktemp("evidence"), {}, check._check_runs
    synthetic_dataset(root, 4)

    def once(cases, args, log):
        key = tuple(c.id for c in cases)
        if key not in cache:
            cache[key] = real(cases, args, log)
        return cache[key]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(check, "_check_runs", once)
        yield root


def run_check(root, verdict, out, monkeypatch, **passed) -> tuple[int, dict, str]:
    """`stbench check --verdict V` with the gates' three overall verdicts set to `passed` (`passed`, `passed_point`,
    `passed_evidence`; the rest of `gates.evaluate`'s report as it is): (the exit code, the report, what it printed)."""
    real = gates.evaluate

    def doctored(*args, **kwargs):
        return {**real(*args, **kwargs), **passed} if passed else real(*args, **kwargs)

    monkeypatch.setattr(gates, "evaluate", doctored)
    if passed:   # the instruments' faults and the balance rule are other contracts: none here
        monkeypatch.setattr(suite_report, "broken_instruments", lambda *args: [])
        monkeypatch.setattr(suite_balance, "read", lambda *args, **kwargs: [])
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = cli.main(["check", "--data", str(root), "--verdict", verdict, "--workers", "1", "--out", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8")), printed.getvalue()


@pytest.mark.parametrize("verdict, key", [("interval", "passed"), ("point", "passed_point"),
                                          ("evidence", "passed_evidence")])
def test_the_verdict_drives_the_exit_code_and_evidence_is_one_of_them(checked, tmp_path, monkeypatch, verdict, key):
    """Contract: `stbench check --verdict V` exits 0 when the gates' overall verdict of V passes and 1 when it fails,
    whatever the other two say, and the report names the verdict and holds all three. The synthetic dataset is
    checked with each of the three overall verdicts set alone to true and alone to false (the instruments not broken);
    `--verdict evidence` follows `passed_evidence` only. Regression caught: `evidence` refused by the parser, an exit
    that follows `passed`, or a report that lacks `passed_evidence`."""
    for follows in (True, False):
        flags = dict.fromkeys(("passed", "passed_point", "passed_evidence"), not follows) | {key: follows}
        code, report, _ = run_check(checked, verdict, tmp_path / f"{verdict}-{follows}.json", monkeypatch, **flags)
        assert report["verdict"] == verdict and report["broken"] == []
        assert code == (0 if follows else 1) and report["passed"] is follows   # `passed` is the verdict asked for
        assert report["passed_interval"] is flags["passed"] and report["passed_point"] is flags["passed_point"]
        assert report["passed_evidence"] is flags["passed_evidence"]


def test_check_prints_the_evidence_columns_and_names_a_failing_cell_on_stderr(checked, tmp_path, monkeypatch, capsys):
    """Contract: the gate table `check` prints has the `scope` and `evid` columns and the evidence verdict line, and the
    report's gates carry `evidence`, `cells` and `pooled` for the shortcut-type gates; under `--verdict evidence` a
    failing cell is named on stderr. The synthetic dataset's worlds are alike, so its priors memorise them and
    `shortcuts` fails everywhere by every reading, in the cell `all rows` among others. Regression caught: the
    evidence reading computed and then dropped on the way to the report or the printout."""
    code, report, printed = run_check(checked, "evidence", tmp_path / "r.json", monkeypatch)
    assert code == 1   # the real evaluation: memorising priors beat all_unknown in every cell
    header = next(line for line in printed.splitlines() if line.startswith("layout"))
    assert header.split()[3] == "scope" and header.split()[-2:] == ["evid", "verdict"]
    assert "EVIDENCE VERDICT FAILS" in printed
    shortcuts = next(g for g in report["gates"] if g["family"] == "object_where")["gates"]["shortcuts"]
    assert {"evidence", "cells", "pooled"} <= set(shortcuts) and shortcuts["evidence"] is False
    err = capsys.readouterr().err
    n = r"\+\d\.\d{3}"
    assert re.search(rf"^FAIL apartment_a vacuum/1 object_where shortcuts \[all rows\]: \S+ beats all_unknown by {n} "
                     rf"\(95% interval {n} to {n}\); the gate counts a gain over 0\.1 whose interval is above 0\.$",
                     err, re.M), err
