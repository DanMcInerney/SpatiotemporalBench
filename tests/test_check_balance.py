"""The balance rule inside `stbench check`: the verdict it follows, the lines it prints and what its report holds."""

from __future__ import annotations

import contextlib
import io
import json

import pytest

from balance_kit import yesno
from check_kit import synthetic_dataset
from stbench import check, cli, gates, suite_balance
from stbench.gates import evaluate
from test_gates import N_BOOT
from test_gates_evidence import world

LEAN, EVEN = suite_balance.read(yesno([7] * 4, 10)), suite_balance.read(yesno([5] * 4, 10))


def test_a_failing_balance_fails_the_evidence_verdict_alone_and_is_named():
    """Contract (plan §2 call 5): `--verdict evidence` fails a family whose best constant beats `all_unknown`, with a
    `BALANCE FAIL` line naming the layout, job, family, the constant and its gain; the point and interval readings, and
    a family that is balanced, are unchanged. The gates of a healthy family pass every reading (the oracle exact, the
    shortcuts at `all_unknown`); a lean coin (70% yes, a gain of +0.4) added to the balance entries fails the evidence
    reading only. Regression caught: the balance dropped on the way to the exit code, or applied to the other readings
    whose gates it is not part of."""
    healthy = evaluate(world(scripts=4), n_boot=N_BOOT)
    assert healthy["passed_evidence"] and healthy["passed_point"]
    assert check.outcome(healthy, "evidence", balance=EVEN) == (True, [])
    passed, lines = check.outcome(healthy, "evidence", balance=[*EVEN, *LEAN])
    assert passed is False and lines == [suite_balance.failure(LEAN[0])]
    assert lines[0].startswith("BALANCE FAIL apartment_a vacuum/1 same_place: the best constant (over every script: "
                               "yes) scores +0.400 held out, over all_unknown")
    for verdict in ("point", "interval"):
        assert check.outcome(healthy, verdict, balance=LEAN) == (True, [])


def test_a_failing_balance_is_listed_after_the_gates_and_before_the_broken_instruments():
    """Contract: the failure lines run gates, balance, then `BROKEN`, and a broken instrument fails the verdict whatever
    the balance says. Regression caught: the lines out of order, or the broken instrument's line lost behind a balance
    failure."""
    healthy = evaluate(world(scripts=4), n_boot=N_BOOT)
    passed, lines = check.outcome(healthy, "evidence", ["BROKEN x"], LEAN)
    assert passed is False and lines == [suite_balance.failure(LEAN[0]), "BROKEN x"]
    assert check.outcome(healthy, "evidence", ["BROKEN x"], EVEN) == (False, ["BROKEN x"])


def test_the_evidence_verdict_line_comes_after_the_balance_lines_and_counts_them():
    """Contract (coordinator, review of A9): the check's `EVIDENCE VERDICT` line is the check's, not the gates' alone:
    `format_table` prints the balance lines before it, and it fails when the balance rule (or a broken instrument) did
    not hold even though every gate passed. A healthy family with a balanced coin passes; with the lean coin's
    `BALANCE FAIL` line and `also_holds` false it fails. Regression caught: a verdict line that says PASSES above a
    `BALANCE FAIL`, the gates' alone, as it was before the rule."""
    healthy = evaluate(world(scripts=4), n_boot=N_BOOT)
    ok = gates.format_table(healthy, [suite_balance.line(EVEN[0])], True).splitlines()
    assert ok[-3].startswith("BALANCE ok") and ok[-2] == "EVIDENCE VERDICT PASSES"
    bad = gates.format_table(healthy, [suite_balance.line(LEAN[0])], False).splitlines()
    assert bad[-3].startswith("BALANCE FAIL") and bad[-2:] == ["EVIDENCE VERDICT FAILS", "ALL GATES PASS"]
    assert gates.format_table(healthy).splitlines()[-2] == "EVIDENCE VERDICT PASSES"   # the default: the gates' alone


def test_the_class_shares_are_printed_against_the_draw():
    """Contract: `check` prints a family's asked class shares, its pools where they differ from the classes, and the
    draw's share (a cap with `at most`), in two decimals. Regression caught: shares printed against the wrong pool, or a
    cap read as an exact share."""
    shares = suite_balance.class_shares("same_place", [{"answer_class": c} for c in ("yes", "moved", "vacated", "yes")])
    assert check._shares(shares) == "moved 0.25, vacated 0.25, yes 0.50; pools yes 0.50, no 0.50 (the draw: no 0.50)"
    capped = suite_balance.class_shares("task_things", [{"answer_class": c} for c in ("empty", "items", "items")])
    assert check._shares(capped) == "empty 0.33, items 0.67 (the draw: empty at most 0.33)"


@pytest.fixture(scope="module")
def checked(tmp_path_factory):
    """`stbench check --verdict evidence` over a synthetic dataset of six compiled cases (alike, so the priors and the
    constants memorise them): (the exit code, the report, stdout, stderr)."""
    root = tmp_path_factory.mktemp("balance")
    synthetic_dataset(root, 6)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["check", "--data", str(root), "--verdict", "evidence", "--workers", "1",
                         "--out", str(root / "report.json")])
    return code, json.loads((root / "report.json").read_text(encoding="utf-8")), out.getvalue(), err.getvalue()


def test_check_prints_a_balance_line_per_family_and_the_report_holds_the_entries(checked):
    """Contract: the printout has one `BALANCE` line for each (layout, job, family) of the groups, and the report's
    `balance` holds the entries those lines say (`suite_balance.read`'s), the failing ones failing `passed_evidence`
    and, under `--verdict evidence`, the exit code and stderr. The synthetic cases are alike worlds, so the answer to a
    `path_ahead` question is the same in every script and a constant gets it right: that family fails. Regression
    caught: the balance computed and dropped before the printout or the report, or a failure that leaves the verdict
    passing."""
    code, report, out, err = checked
    entries = report["balance"]
    assert [(b["layout"], b["job"], b["family"]) for b in entries] == [
        ("apartment_a", "vacuum/1", "object_where"), ("apartment_a", "vacuum/1", "path_ahead")]
    printed = [line for line in out.splitlines() if line.startswith("BALANCE")]
    assert printed == [suite_balance.line(b) for b in entries]
    failing = [b for b in entries if b["fails"]]
    assert [b["family"] for b in failing] == ["path_ahead"] and failing[0]["mean"] > 0
    assert report["passed_evidence"] is False and code == 1
    lines = out.splitlines()
    assert lines.index("EVIDENCE VERDICT FAILS") > max(i for i, line in enumerate(lines) if line.startswith("BALANCE"))
    assert suite_balance.failure(failing[0]) in err.splitlines()
