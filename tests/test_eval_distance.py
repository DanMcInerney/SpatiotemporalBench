"""The distance breakdown by range band (`eval_distance`, `stbench eval distance`): errors by hand, the bands per job,
the score above a reference run. The dataset is a folder of gold files and the run a folder of answers, written here."""

from __future__ import annotations

import json
import math

import pytest

from ability_kit import r, results_doc
from eval_kit import cli
from recompile_kit import recompile
from stbench import eval_results
from stbench.answers import Offset
from stbench.eval_distance import distance, measure

OFFSET = "offset"


def off(ahead, left):
    return {"type": OFFSET, "ahead_m": ahead, "left_m": left}


def test_the_errors_of_an_offset_by_hand():
    """Contract: `measure` gives the signed range error (answered ground range less true, over `range_m`), the 2-D error
    over `range_m` and the bearing error in degrees (None for a zero vector). By hand: gold (0.8, 0.6) answered
    (1.2, 0.9) at range 1: ranges 1.0 and 1.5, so +0.5; the offsets differ by (0.4, 0.3), 0.5; the same direction,
    0 deg. Gold (2, 0) answered (-2, 0) at range 4: same range 0, difference 4/4 = 1, opposite directions 180 deg.
    Gold (1, 0) answered (0, 1) at 1: range 0, error sqrt(2), 90 deg. A zero answer has no bearing; its range error
    is -1 range. Regression caught: a range error without its sign (an under- and an over-estimate alike), a bearing
    in radians or folded past 180, or a division by the ground range, which is 0 straight below a drone."""
    got = measure(Offset(1.2, 0.9), Offset(0.8, 0.6), 1.0)
    assert (got["range_error"], got["relative_error"], got["bearing_deg"]) == pytest.approx((0.5, 0.5, 0.0), abs=1e-9)
    got = measure(Offset(-2, 0), Offset(2, 0), 4.0)
    assert (got["range_error"], got["relative_error"], got["bearing_deg"]) == pytest.approx((0.0, 1.0, 180.0))
    got = measure(Offset(0, 1), Offset(1, 0), 1.0)
    assert (got["range_error"], got["relative_error"], got["bearing_deg"]) == pytest.approx((0.0, math.sqrt(2), 90.0))
    got = measure(Offset(0, 0), Offset(3, 0), 3.0)
    assert (got["range_error"], got["bearing_deg"]) == (pytest.approx(-1.0), None)
    with pytest.raises(ValueError):
        measure(Offset(1, 0), Offset(1, 0), 0.0)


def gold_row(rows, case, qid, gold):
    """The gold row of a question: its answer and the keys of its result row that `score.row` reads (`eval distance`
    also re-scores the run for its credit sensitivity). The ladder is the scorer's default; no test reads a re-scored
    value."""
    row = next(x for x in rows if (x["case"], x["qid"]) == (case, qid))
    return {"qid": qid, "gold": gold, "family": row["family"], "scope": row["scope"], "reference": row["reference"],
            "critical": row["critical"], "slice": "answerable", "credit": "ladder"}


def write_run(root, rows, answers, golds):
    """`root/run` with `results.json` (the rows) and the answers by case, `root/data/private/<case>/gold.jsonl`."""
    run, data = root / "run", root / "data"
    run.mkdir()
    for case, by_qid in answers.items():
        (run / "cases" / case).mkdir(parents=True)
        (run / "cases" / case / "answers.jsonl").write_text(
            "".join(json.dumps({"qid": q, "answer": a}) + "\n" for q, a in by_qid.items()), encoding="utf-8")
    for case, by_qid in golds.items():
        (data / "private" / case).mkdir(parents=True)
        (data / "private" / case / "gold.jsonl").write_text(
            "".join(json.dumps(gold_row(rows, case, q, g)) + "\n" for q, g in by_qid.items()), encoding="utf-8")
    eval_results.write(run, results_doc(rows))
    return run, data


def test_the_report_by_band_and_above_the_reference_by_hand(tmp_path):
    """Contract (FD-0.6.0 §4): per layout and family, a line per range band (`ego_frame.band` at the job's place unit:
    the helper's 0.25 m gives near under 1.5 m and far from 4 m; the AMR's 0.5 m near under 3 m) and one for all, with
    the questions, those answered with an offset, the mean score, the median signed range error, the median 2-D error
    over the range and the median bearing error, and, given a reference run, the score above it over the questions both
    answered. By hand, `ego_now` on the helper, ranges 1 (near), 2 (mid), 6 and 5 (far):
    - q1 gold (0.8, 0.6) answered (1.2, 0.9), score 0.5: range error +0.5, 2-D error 0.5, bearing 0;
    - q2 gold (1.6, 1.2) answered (1.6, -1.2), score -0.5: range error 0, 2-D error 2.4/2 = 1.2, bearing
      acos(0.28) = 73.7 deg;
    - q3 gold (4, 0) answered `unknown`, score 0: no error;
    - q4 gold (3, 4) answered (1.5, 2), score 0.5: ranges 5 and 2.5, so -0.5; 2-D error 2.5/5 = 0.5; bearing 0.
    Far holds q3 and q4: 2 questions, 1 answered, score 1/4, range error -0.500, error/range 0.500, bearing 0.0; all
    holds 4, 3 answered, score 1/8, medians of (0.5, 0, -0.5), (0.5, 1.2, 0.5) and (0, 73.7, 0): +0.000, 0.500, 0.0.
    A reference scoring 0, -0.5, 0, 0 on the four leaves +0.5, 0, 0, +0.5, so far is above it by +0.250 over 2, and
    all by 1.0/4 = +0.250 over 4. The AMR's question at range 2 is `near` (the helper's would be `mid`) and sits in
    its own layout's section; a row with no `range_m` is not in the report. Regression caught: bands read at one unit
    for every job, layouts pooled, an unanswered question counted in the medians, or the score above the reference
    taken over questions the reference did not answer."""

    def helper(qid, score, range_m):
        return r("c1", qid, score, "ego_now", script="s", range_m=range_m)

    rows = [helper("q1", 0.5, 1.0), helper("q2", -0.5, 2.0), helper("q3", 0.0, 6.0), helper("q4", 0.5, 5.0),
            r("c1", "q5", 1.0, "object_where", script="s"),
            r("c2", "q6", 0.5, "ego_now", job="amr/1", layout="warehouse_a", range_m=2.0)]
    answers = {"c1": {"q1": off(1.2, 0.9), "q2": off(1.6, -1.2), "q3": {"type": "unknown"}, "q4": off(1.5, 2.0)},
               "c2": {"q6": off(1.0, 0.0)}}
    golds = {"c1": {"q1": off(0.8, 0.6), "q2": off(1.6, 1.2), "q3": off(4, 0), "q4": off(3, 4),
                    "q5": {"type": "position", "xy": [0, 0]}}, "c2": {"q6": off(2.0, 0.0)}}
    run, data = write_run(tmp_path, rows, answers, golds)
    reference = tmp_path / "ref"
    reference.mkdir()
    refs = [helper("q1", 0.0, 1.0), helper("q2", -0.5, 2.0), helper("q3", 0.0, 6.0), helper("q4", 0.0, 5.0)]
    eval_results.write(reference, results_doc(refs))
    text = distance(run, data, reference)
    lines = text.splitlines()
    assert "## apartment_a, ego_now" in lines and "## warehouse_a, ego_now" in lines
    assert not any("object_where" in line for line in lines)
    head = "| band | n | answered | score | range error | error / range | bearing error (deg) | above the reference |"
    assert head in lines
    for expected in ("| near | 1 | 1 | +0.500 | +0.500 | 0.500 | 0.0 | +0.500 (one script) over 1 |",
                     "| mid | 1 | 1 | -0.500 | +0.000 | 1.200 | 73.7 | +0.000 (one script) over 1 |",
                     "| far | 2 | 1 | +0.250 | -0.500 | 0.500 | 0.0 | +0.250 (one script) over 2 |",
                     "| all | 4 | 3 | +0.125 | +0.000 | 0.500 | 0.0 | +0.250 (one script) over 4 |"):
        assert expected in lines, expected
    amr = lines[lines.index("## warehouse_a, ego_now"):]
    assert any(line.startswith("| near | 1 | 1 | +0.500 |") for line in amr) and not any(
        line.startswith("| mid") for line in amr)
    assert distance(run, data).count("above the reference") == 0


def test_the_command_prints_the_report_and_refuses_a_missing_run(tmp_path, capsys):
    """Contract: `stbench eval distance RUN --data D` prints the report and exits 0; a run folder with no results, or no
    `--data`, exits 2 with a sentence. Regression caught: a traceback for a mistyped folder, or a command that writes
    nothing."""
    rows = [r("c1", "q1", 0.5, "ego_now", range_m=1.0)]
    run, data = write_run(tmp_path, rows, {"c1": {"q1": off(1.2, 0.9)}}, {"c1": {"q1": off(0.8, 0.6)}})
    assert cli("distance", run, "--data", data, "--out", tmp_path / "report.md") == 0
    assert "## apartment_a, ego_now" in capsys.readouterr().out
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# SpatiotemporalBench: distance")
    assert cli("distance", tmp_path / "nowhere", "--data", data) == 2
    assert cli("distance", run) == 2


def test_a_run_is_refused_against_golds_it_was_not_scored_on(tmp_path):
    """Contract: `eval distance` checks that the qids of each scored case's rows in `results.json` are all in the case's
    gold in `--data` (as the resume, `eval compare` and the export do), and refuses a recompile (every qid drawn anew),
    a case with no gold and a reference run scored on another compile. Regression caught: a report that pairs one
    compile's answers with another compile's golds, whose errors would be meaningless and read as the system's, or a
    refusal that never comes because the row loop of `entries` passes over a qid it does not find."""
    rows = [r("c1", "q1", 0.5, "ego_now", range_m=1.0)]
    run, data = write_run(tmp_path, rows, {"c1": {"q1": off(1.2, 0.9)}}, {"c1": {"q1": off(0.8, 0.6)}})
    stale = reference_run(tmp_path, "stale", [r("c1", "q9", 0.5, "ego_now", range_m=1.0)])   # a qid the gold lacks
    assert "## apartment_a, ego_now" in distance(run, data)
    with pytest.raises(ValueError, match="stale scored case c1 on another compile"):
        distance(run, data, stale)
    recompile(data, "c1")
    with pytest.raises(ValueError, match="scored case c1 on another compile"):
        distance(run, data)
    (data / "private" / "c1" / "gold.jsonl").unlink()
    with pytest.raises(ValueError, match="has no gold"):
        distance(run, data)
    assert cli("distance", run, "--data", data) == 2


def reference_run(tmp_path, name, rows):
    """`tmp_path/name`, a results folder of the system `name` scoring `rows`."""
    folder = tmp_path / name
    folder.mkdir()
    eval_results.write(folder, results_doc(rows, system=name))
    return folder


AMR = {"job": "amr/1", "layout": "warehouse_a"}
QIDS = {"c1": ("q1", "q2"), "c2": ("q3", "q4")}   # the helper's case (apartment_a) and the AMR's (warehouse_a)


def ego(case, qid, score):
    """An `ego_now` row at range 2 m in `case` (c1 on the helper, c2 on the AMR)."""
    return r(case, qid, score, "ego_now", script=case, range_m=2.0, **(AMR if case == "c2" else {}))


def two_cells(root, scores, unknown=False):
    """A run in `root/run` over the two cells (the golds all (1.6, 1.2)), its four scores in the order q1..q4;
    answered in the gold's direction at 0.75 of it, or `unknown` throughout."""
    answer = {"type": "unknown"} if unknown else off(1.2, 0.9)
    rows = [ego(case, q, s) for (case, qids), pair in zip(QIDS.items(), (scores[:2], scores[2:]), strict=True)
            for q, s in zip(qids, pair, strict=True)]
    answers = {case: dict.fromkeys(qids, answer) for case, qids in QIDS.items()}
    golds = {case: dict.fromkeys(qids, off(1.6, 1.2)) for case, qids in QIDS.items()}
    return write_run(root, rows, answers, golds)


def system(root, name, helper, amr):
    """The scripted system `name`, in `root/name`: `helper` on both helper questions, `amr` on both AMR questions."""
    scores = {"q1": helper, "q2": helper, "q3": amr, "q4": amr}
    return reference_run(root, name, [ego(case, q, scores[q]) for case, qids in QIDS.items() for q in qids])


def references(root):
    """`kind_range` (0.5 on the helper, -0.5 on the AMR) and `all_unknown` (0 throughout)."""
    return system(root, "kind_range", 0.5, -0.5), system(root, "all_unknown", 0, 0)


def above(text, layout):
    """The `above the reference` cell of the `all` line of `layout`'s `ego_now` table."""
    lines = text.splitlines()
    table = lines[lines.index(f"## {layout}, ego_now"):]
    return next(line for line in table if line.startswith("| all |")).split(" | ")[-1].rstrip(" |")


def test_distance_is_above_the_better_reference_of_each_layout_and_job(tmp_path):
    """Contract (DECISIONS, "Before the binding read", 3): given `kind_range` and `all_unknown` runs, the `above the
    reference` column is taken against the one whose mean score is larger over the questions the run and both references
    scored, per (layout, job), and the report names it and each mean. By hand, the run scoring 0.5, 0.25 on the
    helper and 0.5, 0.0 on the AMR, `kind_range` 0.5 on the helper and -0.5 on the AMR, `all_unknown` 0 throughout:
    on the helper the reference is `kind_range`, and the run is above it by (0 - 0.25)/2 = -0.125 over 2; on the AMR
    it is `all_unknown` (-0.5 is under 0), and the run is above it by 0.5/2 = +0.250 over 2 (against `kind_range`
    alone it would be (1.0 + 0.5)/2 = +0.750; the order the references are given in changes nothing). Regression
    caught: a report above `kind_range` alone, which reads a system that merely abstains as above it on a job where
    `kind_range` scores under `all_unknown`; or one reference chosen for the whole run and not per cell."""
    run, data = two_cells(tmp_path, (0.5, 0.25, 0.5, 0.0))
    kind_range, all_unknown = references(tmp_path)
    text = distance(run, data, [kind_range, all_unknown])
    assert above(text, "apartment_a") == "-0.125 (one script) over 2"
    assert above(text, "warehouse_a") == "+0.250 (one script) over 2"
    lines = text.splitlines()
    assert "- apartment_a helper/1: kind_range (kind_range +0.500, all_unknown +0.000 over 2 questions)" in lines
    assert "- warehouse_a amr/1: all_unknown (kind_range -0.500, all_unknown +0.000 over 2 questions)" in lines
    swapped = distance(run, data, [all_unknown, kind_range])
    assert [above(swapped, "apartment_a"), above(swapped, "warehouse_a")] == [
        above(text, "apartment_a"), above(text, "warehouse_a")]
    assert above(distance(run, data, kind_range), "warehouse_a") == "+0.750 (one script) over 2"   # one reference


def test_a_system_that_abstains_everywhere_is_never_above_the_reference(tmp_path):
    """Contract (DECISIONS, "Before the binding read", 3): a run that answers `unknown` to every question scores what
    `all_unknown` does (0), and so is never above the better of `kind_range` and `all_unknown`, wherever `kind_range`
    scores. By hand, with `kind_range` at 0.5 on the helper and -0.5 on the AMR: the helper's reference is `kind_range`,
    so the abstainer is -0.500 below it; the AMR's is `all_unknown` at 0, so the abstainer is +0.000 above it, not
    +0.500 (above `kind_range`). Regression caught: a distance report that credits abstaining with the amount
    `kind_range` falls below `all_unknown` (the AMR's -0.009 and the vacuum's -0.129)."""
    run, data = two_cells(tmp_path, (0, 0, 0, 0), unknown=True)
    text = distance(run, data, list(references(tmp_path)))
    assert above(text, "apartment_a") == "-0.500 (one script) over 2"
    assert above(text, "warehouse_a") == "+0.000 (one script) over 2"


def test_the_command_takes_several_references(tmp_path, capsys):
    """Contract: `stbench eval distance RUN REF [REF ...] --data D` reports above the better reference and exits 0.
    Regression caught: the command still refusing a third argument, so the two references cannot be given."""
    run, data = two_cells(tmp_path, (0.5, 0.25, 0.5, 0.0))
    kind_range, all_unknown = references(tmp_path)
    assert cli("distance", run, kind_range, all_unknown, "--data", data) == 0
    assert "- warehouse_a amr/1: all_unknown" in capsys.readouterr().out
