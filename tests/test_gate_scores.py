"""The validity checks read the gate score, never the reported one (0.10.0, FD-questions-0.10 §4.2, review B4): the
gates, the balance rule, the form adversary and the broken-instrument check all score a row by its `score_gate` (the
penalised `2s - 1`), so a catastrophe's weight, which lowers only the row's `score`, moves none of them."""

from __future__ import annotations

from dataclasses import replace

import pytest

import balance_kit
from score_kit import result_row
from stbench import form_check, gates, suite, suite_balance, suite_report
from stbench.score_stats import gate_score, paired_difference, pooled
from stream_kit import qid
from test_check import EVENT, NM, at, hand_case, hand_synth, number, run_of
from test_gates import BLIND, N_BOOT, system_set

CATASTROPHE = {"weight": 2, "rule": "over", "margin_m": 0.5}


def lowered(results: dict, names: set[str]) -> dict:
    """`results` with the reported `score` set to -3 and the row flagged a catastrophe, the gate score kept, on every
    answerable row of the systems `names` and on the first question (`...0`) of every system."""
    def hit(name: str, r: dict) -> bool:
        return (name in names and r["slice"] == "answerable") or r["qid"].endswith("0")
    return {name: [{**r, "score": -3.0, "catastrophe": True} if hit(name, r) else r for r in rows]
            for name, rows in results.items()}


def test_a_catastrophe_on_a_shortcut_hides_it_from_no_gate():
    """Contract (review B4): the gates read `score_gate`. `blind` scores 0.6 on a family whose unknown share is 0.25:
    0.35 over, past the 0.10 margin, so the shortcuts gate fails. Give every `blind` answerable row a catastrophe
    (reported -3, gate score kept) and the first question of every system one too: the whole report is unchanged, the
    shortcuts gate still fails on `blind` by 0.35, and the ceiling is still the oracle's 1.0. Regression caught: a gate
    that reads the reported score, which the B4 review measured pushing a failing shortcut under the bar (the helper's
    map reader, +0.279 to +0.049)."""
    base = system_set(blind=[0.6] * 4)
    marked = lowered(base, {BLIND})
    assert any(r["catastrophe"] for rows in marked.values() for r in rows)
    assert gates.evaluate(marked, n_boot=N_BOOT) == gates.evaluate(base, n_boot=N_BOOT)
    (row,) = gates.evaluate(marked, n_boot=N_BOOT)["rows"]
    shortcuts = row["gates"]["shortcuts"]
    assert shortcuts["pass"] is False and shortcuts["worst"] == BLIND and shortcuts["value"] == pytest.approx(0.35)
    assert row["gates"]["ceiling"]["value"] == 1.0
    assert row["scores"][BLIND]["mean"] == pytest.approx(0.6)


def test_a_row_of_an_earlier_build_is_gated_on_its_score():
    """Contract (§4.6, older results stay readable): a row with no `score_gate` was written when the score was the
    penalised one, so the gates read its `score`. The same lowered rows without the key pass the shortcuts gate (the
    mean is -3 on the rows lowered), which is exactly the hole the key closes. Regression caught: a missing key read
    as 0 or as an error."""
    base = system_set(blind=[0.6] * 4)
    old = {name: [{k: v for k, v in r.items() if k not in ("score_gate", "catastrophe")} for r in rows]
           for name, rows in lowered(base, {BLIND}).items()}
    assert gate_score(old[BLIND][0]) == -3.0
    (row,) = gates.evaluate(old, n_boot=N_BOOT)["rows"]
    assert row["gates"]["shortcuts"]["pass"] is True and row["scores"][BLIND]["mean"] < 0


def test_pooled_and_paired_differences_read_the_key_they_are_given():
    """Contract: `score_stats.pooled` and `paired_difference` read `score` unless told `key="score_gate"`, and refuse a
    key that is neither. Two scripts of one row each: A scores -2 reported and 1 on the gate, B 0 and 0. By hand,
    pooled A: -2 reported, 1 gate; A minus B: -2 and 1. Regression caught: the reports reading the gate score (so the
    headline would not move) or the gates reading the reported one."""
    a = [result_row(f"c{i}", f"q{i}", -2.0, "wrong", script=f"s{i}", score_gate=1.0, catastrophe=True) for i in (1, 2)]
    b = [result_row(f"c{i}", f"q{i}", 0.0, "abstained", script=f"s{i}") for i in (1, 2)]
    assert pooled(a, n_boot=20)["mean"] == -2.0 and pooled(a, n_boot=20, key="score_gate")["mean"] == 1.0
    assert paired_difference(a, b, n_boot=20)["mean"] == -2.0
    assert paired_difference(a, b, n_boot=20, key="score_gate")["mean"] == 1.0
    with pytest.raises(ValueError, match="key must be one of"):
        pooled(a, key="s")
    with pytest.raises(ValueError, match="key must be one of"):
        paired_difference(a, b, key="catastrophe")


def test_the_balance_rule_gains_on_the_gate_score():
    """Contract (§4.2): `suite_balance`'s gain of the best constant over `all_unknown` is of the gate score. Script s0
    asks two questions whose gold is 1.0 m, s1 to s3 ten each with gold 5.0 m, every gold carrying an `over` catastrophe
    with margin 0.5. Held out, s0 is answered 5.0 (the other pool is thirty golds of 5.0): 4 m over, a catastrophe, -2
    reported and -1 on the gate; s1 to s3 are answered 5.0 and are exact, +1. By hand the gain is (30 - 2) / 32 =
    0.875, not (30 - 4) / 32 = 0.8125 as the reported score would give. Regression caught: the balance verdict moving
    with a harm weight."""
    asked = balance_kit.numbers([1.0, 5.0, 5.0, 5.0], [2, 10, 10, 10])
    for a in asked:
        a.gold["tolerances"] = {**a.gold["tolerances"], "catastrophe": CATASTROPHE}
    rows = suite_balance.best_constant_rows(asked)
    first = [r for r in rows if r["script"] == "s0"]
    assert [(r["score"], r["score_gate"], r["catastrophe"]) for r in first] == [(-2.0, -1.0, True)] * 2
    (entry,) = suite_balance.read(asked)
    assert entry["mean"] == pytest.approx(0.875)


def test_the_form_adversary_ranks_features_by_the_gate_score(monkeypatch):
    """Contract (§4.2): the best feature of a family is the one whose rows have the highest mean `score_gate`. Feature
    `x` has rows at reported -2 and gate 1 (a hand row that was a catastrophe), feature `y` rows at 0 and 0: by the gate
    score `x` is best with mean 1.0, by the reported score it would be `y`. Regression caught: the adversary's
    verdict, a shortcut gate's input, moving with a harm weight."""
    asked = [replace(a, features={"x": 1, "y": 1}) for a in balance_kit.numbers([1.0, 1.0], 2)]
    made = {"x": [result_row(f"c{i}", f"q{i}", -2.0, "wrong", score_gate=1.0, catastrophe=True) for i in range(4)],
            "y": [result_row(f"c{i}", f"q{i}", 0.0, "abstained") for i in range(4)]}
    monkeypatch.setattr(form_check, "_feature_rows", lambda group, name, answers, memo, in_pool: made[name])
    rows, table = form_check.form_adversary(asked)
    assert table["ego_now"]["feature"] == "x" and table["ego_now"]["means"] == {"x": 1.0, "y": 0.0}
    assert rows == made["x"]


def test_current_views_now_mean_is_the_gate_score(tmp_path):
    """Contract (§4.2): the broken-instrument check that `current_view` answers its own `now` rows (a mean of 0.95 or
    more) reads the gate score. The kit's four `path_ahead` now rows give `current_view` 1, 1, -0.5 and -0.5 on the gate
    (a mean of 0.25, worked in `test_check`); with a catastrophe flagged and the reported score lowered to -2 on each
    the mean the report shows is still 0.25. Regression caught: the reference check failing, or passing, on a harm
    weight."""
    specs = [(2, "red", False, 1.4, at(3.2, 1.5)), (2, "red", False, 1.4, at(3.3, 1.6)),
             (2, "blue", True, 0.3, at(3.3, 1.55)), (3, "red", False, 3.0, at(1.2, 2.2))]
    cases = [hand_case(tmp_path, hand_synth(n), cap=cap, color=color, event=EVENT, unknown=unknown)
             for n, (cap, color, unknown, *_) in enumerate(specs, 1)]
    runs = {(NM, case.id): run_of({qid(1): number(ahead), qid(2): place})
            for case, (*_, ahead, place) in zip(cases, specs, strict=True)}
    scored = [r for case in cases for r in suite.score_rows(case, runs[NM, case.id])]
    rows = {gates.CURRENT: [{**r, "score": -2.0, "catastrophe": True} for r in scored]}
    mean = suite_report.group_numbers(cases, runs, rows)["families"]["path_ahead"]["current_view_now_mean"]
    assert mean == pytest.approx(0.25)
