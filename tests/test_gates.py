"""The validity gates (plan §4.5, §4.6) on hand-made result rows: each threshold and inequality, the point and interval
verdicts, the own-path exemption and the layout-constant one (held-out layouts only) read from the rows, `memory_needed`
on memory rows only, `n/a` and missing systems, and the table."""

from __future__ import annotations

import pytest

from score_kit import result_row
from stbench.gates import (
    BASELINE_NAMES,
    BLIND,
    CURRENT,
    CURRENT_PRIOR,
    FORM,
    GATES,
    ORACLE,
    OTHER,
    PATH_LOG,
    QBLIND,
    SHORTCUTS,
    TABLE,
    UNKNOWN,
    evaluate,
    failing,
    format_table,
    in_pool_name,
)
from stbench.map_files import HELD_OUT

N_BOOT = 500   # enough that a 1/16 resample is past the 2.5% tails of the bootstrap
BASE = {ORACLE: [1, 1, 1, 1], TABLE: [1, 1, 1, 1], CURRENT: [0, 0, 0, 1], CURRENT_PRIOR: [0, 0, 0, 1],
        PATH_LOG: [-0.2] * 4, BLIND: [0.3] * 4, OTHER: [0.0] * 4, QBLIND: [0.0] * 4, UNKNOWN: [0, 0, 0, 1],
        FORM: [0, 0, 0, 1]}


def rows(family, per_question, *, scripts=4, unknown=(3,), scopes=None, reference=None, layout="apartment_a",
         job="vacuum/1"):
    """Rows of one system: each of `scripts` scripts has one case that asks the family's questions. `per_question` is
    the penalised score of each (or a function of the script's index), `unknown` the indices with an unknown gold,
    `scopes` the scope of each question (all memory by default)."""
    out = []
    for e in range(scripts):
        scores = per_question(e) if callable(per_question) else per_question
        for q, score in enumerate(scores):
            out.append(result_row(f"{layout}-{job}-c{e}", f"{family}{q}", score,
                                  "correct" if score >= 1 else "wrong" if score < 0 else "abstained",
                                  script=f"{layout}-{job}-s{e}", family=family, when="end",
                                  scope=(scopes or ["memory"] * len(scores))[q], reference=reference,
                                  slice_="unknown" if q in unknown else "answerable", layout=layout, job=job))
    return out


def system_set(family="where_last", **options):
    """Four questions per script, the last one with an unknown gold (so the unknown share is 0.25), for four scripts: a
    healthy family. Override a system's scores by name (the rest of `options` goes to `rows`)."""
    scores = {n: options.pop(n) for n in list(options) if n in BASE}
    return {n: rows(family, scores.get(n, v), **options) for n, v in BASE.items()}


def merged(*sets):
    return {n: [r for s in sets for r in s[n]] for n in sets[0]}


def gates_of(results, **kwargs):
    (row,) = evaluate(results, n_boot=N_BOOT, **kwargs)["rows"]
    return row["gates"]


def test_a_healthy_family_passes_every_gate_with_the_numbers_behind_them():
    """Contract (plan §4.5): ceiling oracle >= 0.95; knowable upper bound of oracle - memory_table <= 0.02;
    memory_needed current_view and current_view_prior each <= all_unknown + 0.05; shortcuts each <= all_unknown + 0.10
    (unknown share 0.25 here, blind at 0.30 is 0.05 over); separation memory_table - best shortcut >= 0.40 with lower
    bound > 0.30 (1.0 - 0.3). Both verdicts, interval and point, hold, and the row counts its scripts and questions.
    Regression caught: a gate computed on the wrong rows or with the wrong inequality."""
    report = evaluate(system_set(), n_boot=N_BOOT)
    (row,) = report["rows"]
    g = row["gates"]
    assert report["passed"] and report["passed_point"] and row["pass"] and row["point"] and row["missing"] == []
    assert (row["layout"], row["job"], row["family"], row["scripts"], row["n"]) == (
        "apartment_a", "vacuum/1", "where_last", 4, 16)
    assert set(g) == set(GATES) and all(v["pass"] and v["point"] for k, v in g.items() if k != "time_needed")
    assert g["time_needed"]["pass"] is None   # `where_last` is not a temporal family (tests/test_gates_time_needed.py)
    assert g["ceiling"]["value"] == 1.0
    assert g["knowable"]["value"] == 0.0 and g["knowable"]["ci"] == (0.0, 0.0)
    assert g["memory_needed"]["value"] == 0.0 and g["memory_needed"]["unknown_share"] == 0.25
    assert g["shortcuts"]["value"] == pytest.approx(0.05) and g["shortcuts"]["worst"] == BLIND
    assert set(g["shortcuts"]["scores"]) == set(SHORTCUTS)
    assert g["separation"]["value"] == pytest.approx(0.7) and g["separation"]["best"] == BLIND
    assert row["scores"][ORACLE]["mean"] == 1.0 and row["scores"][UNKNOWN]["mean"] == 0.25
    assert list(failing(report)) == []


def test_the_ceiling_is_the_oracles_mean_and_inclusive():
    """Contract: ceiling is the oracle's pooled mean against 0.95, met at 0.95 itself. Oracle (1 + 1 + 1 + 0.76) / 4 =
    0.94 fails it, and nothing else; (1 + 1 + 1 + 0.8) / 4 = 0.95 passes. Regression caught: gating a single question
    or the tier mean, or an exclusive threshold."""
    report = evaluate(system_set(oracle=[1, 1, 1, 0.76]), n_boot=N_BOOT)
    assert list(failing(report)) == [("apartment_a", "vacuum/1", "where_last", "ceiling")]
    assert not report["passed"] and not report["passed_point"]
    assert evaluate(system_set(oracle=[1, 1, 1, 0.8]), n_boot=N_BOOT)["passed"]


def test_knowable_is_the_upper_ci_of_the_paired_gap_not_its_mean():
    """Contract: oracle - memory_table of 0.05 on every question has CI (0.05, 0.05): the upper bound 0.05 > 0.02 fails
    `knowable` on both verdicts. Gaps of 0 and 0.04 by script alternating have mean 0.02, which holds the point
    estimate, and an upper bound of 0.04, which fails the interval. Regression caught: gating the mean gap alone, or
    only the table's own score."""
    g = gates_of(system_set(memory_table=[0.95] * 4))["knowable"]
    assert not g["pass"] and not g["point"] and g["value"] == pytest.approx(0.05)
    assert g["ci"] == pytest.approx((0.05, 0.05))
    g = gates_of(system_set(memory_table=lambda e: [1, 1, 1, 1] if e % 2 == 0 else [0.96] * 4))["knowable"]
    assert g["point"] is True and g["pass"] is False and g["value"] == pytest.approx(0.02)
    assert g["ci"][1] == pytest.approx(0.04)


@pytest.mark.parametrize("shortcut", SHORTCUTS)
def test_a_shortcut_more_than_0_10_above_the_unknown_share_fails_and_is_named(shortcut):
    """Contract: each of blind, path_log, other_case, questionnaire_blind and the `form` adversary at 0.40, with unknown
    share 0.25, is 0.15 over the allowed 0.10: shortcuts fails (both verdicts) and names it. It is also the best
    shortcut, so separation measures the table's 1.0 against its 0.40 (0.60, which passes). At 0.35, 0.10 over, it
    passes. Regression caught: comparing to zero instead of the unknown share, which would pass a family that
    all-`unknown` already scores 0.25 on; or leaving one of the five out of the gate or of the best shortcut."""
    g = gates_of(system_set(**{shortcut: [0.4] * 4}))
    assert g["shortcuts"]["pass"] is False and g["shortcuts"]["point"] is False and g["shortcuts"]["worst"] == shortcut
    assert g["shortcuts"]["value"] == pytest.approx(0.15) and shortcut in g["shortcuts"]["scores"]
    assert g["separation"]["pass"] and g["separation"]["best"] == shortcut and g["separation"]["value"] == 0.6
    assert gates_of(system_set(**{shortcut: [0.35] * 4}))["shortcuts"]["pass"]


def test_an_own_path_family_is_exempt_from_path_log_as_the_rows_say():
    """Contract: when every oracle row of a group has `reference == "path_log"`, path_log is the family's reference: it
    is left out of the shortcut gate and of the best shortcut, read from the rows and never from the family's name. The
    same rows with no reference gate path_log (it scores 1, so shortcuts fails and names it); a group where only some
    rows carry the reference is not exempt; other shortcuts are gated all the same. Regression caught: failing an
    own-path family because its own path answers it, excusing path_log everywhere or by name, or letting one tagged row
    excuse a whole group."""
    own = system_set("any_name", reference="path_log", **{PATH_LOG: [1] * 4})
    g = gates_of(own)
    assert g["shortcuts"]["pass"] and PATH_LOG not in g["shortcuts"]["scores"]
    assert g["separation"]["pass"] and g["separation"]["best"] == BLIND
    plain = gates_of(system_set("any_name", **{PATH_LOG: [1] * 4}))
    assert plain["shortcuts"]["pass"] is False and plain["shortcuts"]["worst"] == PATH_LOG
    tagged = system_set("any_name", **{PATH_LOG: [1] * 4})
    tagged[ORACLE] = [{**r, "reference": "path_log"} if r["qid"] == "any_name0" else r for r in tagged[ORACLE]]
    assert gates_of(tagged)["shortcuts"]["pass"] is False
    gated = gates_of(system_set("any_name", reference="path_log", **{PATH_LOG: [1] * 4, BLIND: [0.5] * 4}))
    assert gated["shortcuts"]["pass"] is False and gated["shortcuts"]["worst"] == BLIND


def test_a_layout_constant_family_is_exempt_from_other_case_on_a_held_out_layout_only():
    """Contract (VISION rule 5, the held-out clause; M4 call 16): when every oracle row of a group on a layout of
    `map_files.HELD_OUT` has `reference == "other_case"`, other_case is left out of the shortcut gate and of the best
    shortcut. The same rows on a public layout keep the gate: other_case at 1.0 against an unknown share of 0.25 is
    0.75 over the allowed 0.10, shortcuts fails and names it, and it is the best shortcut (separation 1.0 - 1.0 = 0.0
    fails). Held out, blind at 0.3 is the best of the rest (separation 1.0 - 0.3 = 0.7). Regression caught: gating a
    family by the very stream that is its reference on an unseen layout, or excusing it on a layout the agent can know
    (its priors and streams are public)."""
    held_out = gates_of(system_set("any_name", reference="other_case", layout=HELD_OUT[0], **{OTHER: [1] * 4}))
    assert held_out["shortcuts"]["pass"] and OTHER not in held_out["shortcuts"]["scores"]
    assert held_out["shortcuts"]["value"] == pytest.approx(0.05) and held_out["shortcuts"]["worst"] == BLIND
    assert held_out["separation"]["pass"] and held_out["separation"]["best"] == BLIND
    assert held_out["separation"]["value"] == pytest.approx(0.7)
    public = gates_of(system_set("any_name", reference="other_case", **{OTHER: [1] * 4}))
    assert public["shortcuts"]["pass"] is False and public["shortcuts"]["worst"] == OTHER
    assert public["shortcuts"]["value"] == pytest.approx(0.75) and OTHER in public["shortcuts"]["scores"]
    assert public["separation"]["pass"] is False and public["separation"]["best"] == OTHER
    assert public["separation"]["value"] == pytest.approx(0.0)


def test_the_layout_constant_exemption_is_one_reference_at_a_time_and_needs_every_row():
    """Contract: on a held-out layout a group's reference excuses only the shortcut it names: `other_case` rows do not
    excuse path_log (at 1.0 it fails shortcuts, 0.75 over, and is named) and `path_log` rows do not excuse other_case
    (likewise); a group where one row of four names `other_case` is not exempt (other_case at 1.0 fails, named); the
    own-path exemption still holds there (path_log at 1.0 passes) and a family with no reference gates both. Regression
    caught: the held-out layout turning every shortcut off, the two exemptions sharing one flag, or one tagged row
    excusing a whole group."""
    held = {"layout": HELD_OUT[0]}
    g = gates_of(system_set("any_name", reference="other_case", **held, **{PATH_LOG: [1] * 4}))["shortcuts"]
    assert g["pass"] is False and g["worst"] == PATH_LOG and g["value"] == pytest.approx(0.75)
    g = gates_of(system_set("any_name", reference="path_log", **held, **{OTHER: [1] * 4}))["shortcuts"]
    assert g["pass"] is False and g["worst"] == OTHER and g["value"] == pytest.approx(0.75)
    some = system_set("any_name", **held, **{OTHER: [1] * 4})
    some[ORACLE] = [{**r, "reference": "other_case"} if r["qid"] == "any_name0" else r for r in some[ORACLE]]
    g = gates_of(some)["shortcuts"]
    assert g["pass"] is False and g["worst"] == OTHER
    own = gates_of(system_set("any_name", reference="path_log", **held, **{PATH_LOG: [1] * 4}))["shortcuts"]
    assert own["pass"] and PATH_LOG not in own["scores"] and OTHER in own["scores"]
    for shortcut in (PATH_LOG, OTHER):
        plain = gates_of(system_set("any_name", **held, **{shortcut: [1] * 4}))["shortcuts"]
        assert plain["pass"] is False and plain["worst"] == shortcut


def test_a_memory_less_system_must_stay_within_0_05_of_all_unknown_on_memory_rows():
    """Contract: current_view at 0.325 against all_unknown at 0.25 is 0.075 over the allowed 0.05: memory_needed fails
    and names it; current_view_prior likewise, and it is named when it is the worse; exactly 0.05 over passes.
    Regression caught: a family answerable without memory, which cannot separate systems that remember from those that
    do not, or the prior-keyed system left out of the gate."""
    g = gates_of(system_set(**{CURRENT: [0.1, 0.1, 0.1, 1]}))["memory_needed"]
    assert g["pass"] is False and g["point"] is False and g["worst"] == CURRENT and g["value"] == pytest.approx(0.075)
    g = gates_of(system_set(**{CURRENT_PRIOR: [0.2, 0.2, 0.2, 1]}))["memory_needed"]
    assert g["pass"] is False and g["worst"] == CURRENT_PRIOR and g["value"] == pytest.approx(0.15)
    assert gates_of(system_set(**{CURRENT: [0.05, 0.05, 0.05, 1]}))["memory_needed"]["pass"]


def test_memory_needed_reads_memory_rows_only_and_is_na_without_them():
    """Contract: `memory_needed` counts the group's `scope == "memory"` rows only. Questions 0 and 1 are `now`, 2 and 3
    `memory` (3 unknown): current_view scores 1 on the `now` ones (it is their reference) and 0 on memory question 2 and
    1 on 3, as does all_unknown on memory: 0.5 against 0.5, over by 0, which passes; counting every row it would be 0.75
    against 0.25. With no memory row the gate is n/a (None): neither a pass nor a fail, and the family and the report
    still pass. Regression caught: the reference of a `now` family failing the memory gate, or a gate with no rows
    passing vacuously and counting."""
    scopes = ["now", "now", "memory", "memory"]
    g = gates_of(system_set(scopes=scopes, **{CURRENT: [1, 1, 0, 1]}))["memory_needed"]
    assert g["pass"] is True and g["value"] == 0.0 and g["unknown_share"] == 0.5
    now_only = evaluate(system_set(scopes=["now"] * 4, **{CURRENT: [1, 1, 1, 1]}), n_boot=N_BOOT)
    g = now_only["rows"][0]["gates"]["memory_needed"]
    assert g["pass"] is None and g["point"] is None and g["note"] == "no memory rows"
    assert now_only["passed"] and now_only["passed_point"] and list(failing(now_only)) == []


def test_knowable_is_na_on_a_shared_rule_family_and_nowhere_else():
    """Contract: `knowable` is `n/a` for a family in `shared_rule` (the gold and the systems call one grid function, so
    the oracle and memory_table agree by construction), whatever memory_table scores; the other gates still apply; a
    family not named stays gated. Regression caught: a free-space family failing a gate about identity association it
    has none of, or the exemption applied to every family."""
    drifting = {"memory_table": [0.9] * 4}
    shared = evaluate(system_set("path_ahead", **drifting), shared_rule=frozenset({"path_ahead"}), n_boot=N_BOOT)
    g = shared["rows"][0]["gates"]
    assert g["knowable"]["pass"] is None and g["knowable"]["point"] is None and g["ceiling"]["pass"] is True
    assert ("apartment_a", "vacuum/1", "path_ahead", "knowable") not in set(failing(shared))
    alone = evaluate(merged(system_set("path_ahead", **drifting), system_set("object_where", **drifting)),
                     shared_rule=frozenset({"path_ahead"}), n_boot=N_BOOT)
    verdicts = {r["family"]: r["gates"]["knowable"]["pass"] for r in alone["rows"]}
    assert verdicts == {"object_where": False, "path_ahead": None}


def test_na_is_never_a_pass_or_a_fail_and_is_printed_as_na():
    """Contract: a gate that is `n/a` counts toward neither `pass` nor `point` of its family or the report, is absent
    from `failing`, and prints `n/a` in the table. A family whose only applicable gates fail still fails. Regression
    caught: an `n/a` read as a failure (blocking a valid family) or as a pass (hiding a failing one)."""
    results = system_set("path_ahead", scopes=["now"] * 4, memory_table=[0.9] * 4)
    report = evaluate(results, shared_rule=frozenset({"path_ahead"}), n_boot=N_BOOT)
    (row,) = report["rows"]
    assert row["gates"]["knowable"]["pass"] is None and row["gates"]["memory_needed"]["pass"] is None
    assert row["pass"] and row["point"] and report["passed"]
    assert format_table(report).splitlines()[1].count("n/a") == 3   # knowable, memory_needed, time_needed (no temporal)
    assert not evaluate(results, n_boot=N_BOOT)["passed"]  # the same rows without the shared rule fail knowable
    bad = evaluate(system_set("path_ahead", scopes=["now"] * 4, oracle=[0.5] * 4),
                   shared_rule=frozenset({"path_ahead"}), n_boot=N_BOOT)
    assert not bad["passed"] and list(failing(bad)) == [("apartment_a", "vacuum/1", "path_ahead", "ceiling")]


def test_a_system_with_no_rows_gives_na_with_a_note_and_never_a_key_error():
    """Contract (map X-H15): a system with no rows in a group (absent from `results`, or present for other groups)
    leaves the gates that read it unraised: with no failure among the systems that have rows the gate is `n/a` with a
    note naming what is missing (a pass proven on part of the systems is not a pass); a failure among the rest stands.
    No memory_table makes knowable and separation `n/a`; no all_unknown makes shortcuts and memory_needed `n/a`. The
    report lists the missing systems of the group. Regression caught: a `KeyError` on a system that did not run, or a
    gate that passes because the system that would fail it left no rows."""
    without_other = {n: r for n, r in system_set().items() if n != OTHER}
    report = evaluate(without_other, n_boot=N_BOOT)
    g = report["rows"][0]["gates"]
    assert g["shortcuts"]["pass"] is None and g["shortcuts"]["note"] == f"no rows for {OTHER}"
    assert g["separation"]["pass"] is None and g["ceiling"]["pass"] is True and report["rows"][0]["missing"] == [OTHER]
    assert report["passed"] and f"no rows for {OTHER}" in format_table(report)
    failing_rest = {**without_other, BLIND: rows("where_last", [0.4] * 4)}
    g = evaluate(failing_rest, n_boot=N_BOOT)["rows"][0]["gates"]["shortcuts"]
    assert g["pass"] is False and g["worst"] == BLIND and g["note"] == f"no rows for {OTHER}"
    bare = {n: r for n, r in system_set().items() if n not in (TABLE, UNKNOWN)}
    g = evaluate(bare, n_boot=N_BOOT)["rows"][0]["gates"]
    assert [g[k]["pass"] for k in ("knowable", "separation", "shortcuts", "memory_needed")] == [None] * 4
    partial = merged(system_set("a"), system_set("b"))
    partial[OTHER] = [r for r in partial[OTHER] if r["family"] == "a"]
    by_family = {r["family"]: r["gates"]["shortcuts"]["pass"] for r in evaluate(partial, n_boot=N_BOOT)["rows"]}
    assert by_family == {"a": True, "b": None}


def test_the_oracles_rows_are_required_because_they_define_the_groups():
    """Contract: `evaluate` groups by the oracle's rows, so a missing or empty oracle is a ValueError. Regression
    caught: a gate run with no groups reporting `passed`."""
    with pytest.raises(ValueError, match="oracle"):
        evaluate({n: r for n, r in system_set().items() if n != ORACLE}, n_boot=N_BOOT)
    with pytest.raises(ValueError, match="oracle"):
        evaluate({**system_set(), ORACLE: []}, n_boot=N_BOOT)


def test_separation_needs_0_40_over_the_best_shortcut():
    """Contract: table 1.0 against a best shortcut at 0.65 is 0.35 < 0.40 (that shortcut is inside the unknown share's
    0.10 margin only because the unknown share is 0.75 here): separation fails on both verdicts; at 0.60 it is 0.40,
    which passes. Regression caught: separation measured against the floor instead of the best shortcut, or an exclusive
    threshold."""
    shaped = {UNKNOWN: [0, 1, 1, 1], CURRENT: [0, 1, 1, 1], CURRENT_PRIOR: [0, 1, 1, 1], FORM: [0] * 4}
    g = gates_of(system_set(unknown=(1, 2, 3), **shaped, **{BLIND: [0.65] * 4}))
    assert g["shortcuts"]["pass"] and g["separation"]["pass"] is False and g["separation"]["point"] is False
    assert g["separation"]["value"] == pytest.approx(0.35) and g["separation"]["best"] == BLIND
    g = gates_of(system_set(unknown=(1, 2, 3), **shaped, **{BLIND: [0.6] * 4}))
    assert g["separation"]["pass"] and g["separation"]["point"]


def test_the_point_and_the_interval_verdicts_differ_when_the_lower_bound_misses():
    """Contract (recommendations §2.3 item 4): `point` judges the point estimates, `pass` the interval. The table beats
    the best shortcut by 0.25 and 0.65 in scripts alternating, mean 0.45 (>= 0.40: the point verdict holds) with a lower
    bound of 0.25 (not > 0.30: the interval verdict fails); the report then has `passed_point` and not `passed`, and
    `failing` names separation. With one script there is no interval at all: `pass` fails and `point` holds, for
    knowable and separation both. Regression caught: M1's point-estimate exit and M2's interval exit collapsing into
    one verdict, or a gate passing on a missing interval."""
    shaped = {BLIND: lambda e: [0.75] * 4 if e % 2 == 0 else [0.35] * 4, UNKNOWN: [0, 0, 1, 1],
              CURRENT: [0, 0, 1, 1], CURRENT_PRIOR: [0, 0, 1, 1], FORM: [0, 0, 1, 1], PATH_LOG: [0, 0, 0, 0]}
    report = evaluate(system_set(unknown=(2, 3), **shaped), n_boot=N_BOOT)
    g = report["rows"][0]["gates"]
    assert g["separation"]["value"] == pytest.approx(0.45) and g["separation"]["ci"][0] == pytest.approx(0.25)
    assert g["separation"]["point"] is True and g["separation"]["pass"] is False
    assert report["passed_point"] is True and report["passed"] is False
    assert list(failing(report)) == [("apartment_a", "vacuum/1", "where_last", "separation")]
    one = gates_of(system_set(scripts=1))
    for name in ("knowable", "separation"):
        assert one[name]["ci"] is None and one[name]["pass"] is False and one[name]["point"] is True


def test_groups_are_layout_job_and_family_and_each_counts_its_scripts():
    """Contract: gates are per (layout, job, family) of the oracle's rows, each row giving the scripts that carry the
    family and its questions: the same family on another layout or under another job is its own group, and a failure
    on one is reported against it alone. Here the family is on apartment_a with 3 scripts (12 questions), on
    warehouse_a (blind at 0.4, a shortcut failure) with 4 (16) and under another job with 2 (8). Regression caught: a
    shortcut that works on one layout or job hidden by a verdict pooled over both."""
    report = evaluate(merged(system_set(scripts=3), system_set(layout="warehouse_a", **{BLIND: [0.4] * 4}),
                             system_set(job="other/1", scripts=2)), n_boot=N_BOOT)
    found = {(r["layout"], r["job"]): (r["scripts"], r["n"]) for r in report["rows"]}
    assert found == {("apartment_a", "vacuum/1"): (3, 12), ("apartment_a", "other/1"): (2, 8),
                     ("warehouse_a", "vacuum/1"): (4, 16)}
    assert list(failing(report)) == [("warehouse_a", "vacuum/1", "where_last", "shortcuts")]
    assert not report["passed"]


def test_the_table_prints_every_gate_per_group_with_a_verdict():
    """Contract: `format_table` prints one line per (layout, job, family) with its scope, scripts, questions and each
    gate's verdict (`ok`, `point` for a pass on the point estimates alone, `FAIL`, `n/a`) with value and threshold, then
    the evidence verdict (`evid`) and the group's verdict, and a last line that says whether every gate passes, only the
    point estimates do, or the gates fail. Regression caught: a header and cells out of step with the rows, or a
    point-only pass printed as a full one."""
    failed = format_table(evaluate(system_set(**{BLIND: [0.4] * 4}), n_boot=N_BOOT)).splitlines()
    assert failed[0].split() == ["layout", "job", "family", "scope", "scripts", "n", *GATES, "evid", "verdict"]
    assert failed[1].split()[:6] == ["apartment_a", "vacuum/1", "where_last", "memory", "4", "16"]
    assert "FAIL  +0.15 vs 0.1" in failed[1] and failed[1].rstrip().endswith("FAIL") and failed[-1] == "GATES FAIL"
    assert failed[1].index("ok") == failed[0].index("ceiling")
    shaped = {BLIND: lambda e: [0.75] * 4 if e % 2 == 0 else [0.35] * 4, UNKNOWN: [0, 0, 1, 1], CURRENT: [0, 0, 1, 1],
              CURRENT_PRIOR: [0, 0, 1, 1], FORM: [0, 0, 1, 1], PATH_LOG: [0, 0, 0, 0]}
    point = format_table(evaluate(system_set(unknown=(2, 3), **shaped), n_boot=N_BOOT)).splitlines()
    assert "point +0.45 vs 0.4" in point[1] and point[1].rstrip().endswith("point")
    assert point[-1] == "POINT GATES PASS, INTERVAL VERDICTS FAIL"
    assert format_table(evaluate(system_set(), n_boot=N_BOOT)).splitlines()[-1] == "ALL GATES PASS"


def test_the_systems_the_registry_names_are_the_baselines_and_the_form_is_not_one():
    """Contract: `BASELINE_NAMES` are the thirteen systems E1, 0.6.0 and 0.8.0 register (the two time-blind notebooks
    and the references `kind_range` and `map_reader` among them), spelled as the registry spells them, `FORM` (the
    adversary E2 supplies) is not among them, and every shortcut is a baseline or the form. Regression caught: a system
    renamed on one side of the gates, so its rows are never read and its gate passes on no rows."""
    assert set(BASELINE_NAMES) == {"oracle", "memory_table", "current_view", "current_view_prior", "path_log", "blind",
                                   "other_case", "questionnaire_blind", "all_unknown", "latest_table", "first_table",
                                   "kind_range", "map_reader"}
    assert FORM == "form" and FORM not in BASELINE_NAMES and set(SHORTCUTS) <= {*BASELINE_NAMES, FORM}


def with_twins(family="where_last", **twins):
    """`system_set` and, for each learned shortcut named, the rows of its in-pool twin (`in_pool_name`) over the same
    questions: the scores given, one per question."""
    results = system_set(family)
    return results | {in_pool_name(name): rows(family, scores) for name, scores in twins.items()}


@pytest.mark.parametrize("learned", [BLIND, PATH_LOG, QBLIND, FORM])
def test_a_learned_shortcut_is_gated_on_the_larger_of_its_two_rules(learned):
    """Contract (VISION rule 5: every shortcut sits at the floor): a learned shortcut keeps its best response by a
    held-out rule that abstains where a fair learner would answer, so it is also scored under the pool's own winner (its
    twin, `in_pool_name`) and the `shortcuts` gate reads the larger. Hand rows, unknown share 0.25, the held-out rule at
    its base score (blind 0.30, the rest at or under 0.25: passing) and the twin at 0.40, 0.15 over the allowed 0.10:
    shortcuts fails on both verdicts and names the twin, with both scores in the gate; the twin at 0.35, exactly 0.10
    over, passes; a twin under the held-out rule changes nothing. `separation` reads the held-out rules alone (its best
    shortcut is never a twin). Regression caught: a shortcut that passes its gate on the rule that abstains while the
    same shortcut, answering, would not (a planted 65%-right yes/no shortcut scored +0.04 to +0.15 held out against
    +0.13 to +0.23 in the pool)."""
    twin = in_pool_name(learned)
    g = gates_of(with_twins(**{learned: [0.4] * 4}))
    assert g["shortcuts"]["pass"] is False and g["shortcuts"]["point"] is False and g["shortcuts"]["worst"] == twin
    assert g["shortcuts"]["value"] == pytest.approx(0.15) and {learned, twin} <= set(g["shortcuts"]["scores"])
    assert "_in_pool" not in g["separation"]["best"]
    assert gates_of(with_twins(**{learned: [0.35] * 4}))["shortcuts"]["pass"]
    base, lower = gates_of(system_set())["shortcuts"], gates_of(with_twins(**{learned: [-1] * 4}))["shortcuts"]
    assert (lower["pass"], lower["worst"], lower["value"]) == (base["pass"], base["worst"], base["value"])


def test_current_view_prior_is_gated_on_the_larger_of_its_two_rules_in_memory_needed():
    """Contract: `memory_needed` reads the in-pool twin of `current_view_prior` beside it, the larger deciding. Hand
    rows as the gate's test above: the twin at 0.325 on memory rows against all_unknown's 0.25 is 0.075 over the
    allowed 0.05, while `current_view_prior` itself stays at 0.25: the gate fails and names the twin. The boundary, by
    hand: twin rows 0.2, 0, 0, 1 have mean 1.2 / 4 = 0.30, exactly 0.05 over the 0.25, and pass; rows 0.204, 0, 0, 1
    have mean 1.204 / 4 = 0.301, 0.051 over, and fail on the twin. Regression caught: the memoryless responder's prior
    judged on the rule that abstains, or a boundary that is exclusive or off by a margin."""
    twin = in_pool_name(CURRENT_PRIOR)
    g = gates_of(with_twins(**{CURRENT_PRIOR: [0.1, 0.1, 0.1, 1]}))["memory_needed"]
    assert g["pass"] is False and g["worst"] == twin and g["value"] == pytest.approx(0.075)
    on = gates_of(with_twins(**{CURRENT_PRIOR: [0.2, 0, 0, 1]}))["memory_needed"]
    assert on["pass"] and on["worst"] == twin and on["value"] == pytest.approx(0.05)
    over = gates_of(with_twins(**{CURRENT_PRIOR: [0.204, 0, 0, 1]}))["memory_needed"]
    assert over["pass"] is False and over["worst"] == twin and over["value"] == pytest.approx(0.051)


def test_a_twin_with_no_rows_is_not_read_and_is_not_missing():
    """Contract: a system with no rows is `n/a` with a note when a gate should have read it, but the in-pool twins are
    extra readings: results without any (every report made before them) gate as they did, with no note and no `missing`
    twin. Regression caught: a twin that is required, which would make every older report and every hand-made set of
    rows `n/a`."""
    (row,) = evaluate(system_set(), n_boot=N_BOOT)["rows"]
    assert row["pass"] and row["missing"] == [] and "note" not in row["gates"]["shortcuts"]
