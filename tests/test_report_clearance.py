"""The reports for credit by the job's clearance (0.7.0, FD-0.7.0 §2.4 and its review, lane D4): the absolute error
against C in `eval distance`, the credit bands, the unpenalised credit and the abstention bound in `results.md`, and the
credit sensitivity. Rows, golds and answers are written here and every expected number is worked by hand, never by the
code under test."""

from __future__ import annotations

import json

import pytest

from ability_kit import r, results_doc
from eval_kit import cli, hand_dataset, results
from score_kit import result_row
from stbench import eval_results, job_registry
from stbench.eval_distance import distance
from stbench.eval_report import bound_score, credit_band, render
from stbench.eval_sensitivity import (
    credit_sensitivity,
    place_widening,
    scaled_offset,
    scaled_place,
    sensitivity,
    sensitivity_lines,
)


def off(ahead, left):
    return {"type": "offset", "ahead_m": ahead, "left_m": left}


def ladder_gold(qid, gold, rungs):
    """A gold row of the ladder credit with `rungs` (a clearance ladder is `[C/2, C, 2C]`)."""
    return {"qid": qid, "gold": gold, "tolerances": {"ladder": rungs}}


def write_run(root, rows, answers, golds):
    """`root/run` holding `results.json` (the `rows`) and the answers by case,
    and `root/data/private/<case>/gold.jsonl` with the `golds` (a list of gold rows a case)."""
    run, data = root / "run", root / "data"
    for case, by_qid in answers.items():
        (run / "cases" / case).mkdir(parents=True)
        (run / "cases" / case / "answers.jsonl").write_text(
            "".join(json.dumps({"qid": q, "answer": a}) + "\n" for q, a in by_qid.items()), encoding="utf-8")
    for case, records in golds.items():
        (data / "private" / case).mkdir(parents=True)
        (data / "private" / case / "gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in records),
                                                            encoding="utf-8")
    eval_results.write(run, results_doc(rows))
    return run, data


def test_the_clearance_sections_read_the_absolute_error_against_the_ladders_own_rungs(tmp_path):
    """Contract (FD-0.7.0 §2.4, §10 D4): `eval distance` opens with a section per layout giving, for each job and family
    and for the job's questions together, the share of the answered offsets within the ladder's three rungs (C/2, C and
    2C) and the median error in metres, where C is the middle rung of the gold row's own ladder (the number the scorer
    uses), whatever the range; an `unknown` is not answered, and a ladder that is not [C/2, C, 2C] to 1e-6 is not in the
    section. By hand, the helper, ladder [0.075, 0.15, 0.30] (C = 0.15), golds at (1, 0) or (0, 2):
    - `ego_now` errors 0.05, 0.15 (on the rung, met), 0.25, 2.0 and an `unknown`: 5 questions, 4 answered, within C/2
      1/4, within C 2/4, within 2C 3/4, the median (0.15 + 0.25)/2 = 0.200; the mean of the scores 1, .5, .5, -1, 0 is
      +0.200. Two more questions are left out, one on a 4-rung ladder and one on the 3-rung [0.1, 0.25, 0.5] (its first
      rung is not half its second);
    - `ego_mem` errors 0 and 0.6: shares 1/2 each, the median 0.300, the mean +0.000;
    - all 7 questions, 6 answered, errors 0, 0.05, 0.15, 0.25, 0.6, 2.0: 2/6, 3/6, 4/6 and (0.15 + 0.25)/2 = 0.200.
    A vacuum's two questions with ladders [0.05, 0.1, 0.2] and [0.125, 0.25, 0.5] (errors 0.06, 0.2) are each judged on
    their own rungs: the C cell is 0.100-0.250, the shares 0, 1, 1, the median 0.130. The AMR's (C = 0.5, error 0.4) is
    in its own layout's section: 0, 1, 1. None of the rows has a `range_m`, so no band section is written. Regression
    caught: an error read against the range and not C, a ladder of another form taken for a clearance one, a boundary
    error counted as outside, an `unknown` counted in the shares, or a clearance report that needs `range_m`."""
    def ego(case, qid, score, family="ego_now", **kw):
        return r(case, qid, score, family, script=case, **kw)

    rows = [ego("c1", "q1", 1.0), ego("c1", "q2", 0.5), ego("c1", "q3", 0.5), ego("c1", "q4", -1.0),
            ego("c1", "q5", 0.0), ego("c1", "q8", 1.0), ego("c1", "q12", 1.0), ego("c1", "q6", 1.0, "ego_mem"),
            ego("c1", "q7", -1.0, "ego_mem"), ego("c2", "q10", 0.5, job="vacuum/2"),
            ego("c2", "q11", 0.5, job="vacuum/2"), ego("c3", "q9", 0.5, job="amr/1", layout="warehouse_a")]
    answers = {"c1": {"q1": off(1.05, 0), "q2": off(1.15, 0), "q3": off(1.0, 0.25), "q4": off(3.0, 0),
                      "q5": {"type": "unknown"}, "q8": off(1, 0), "q12": off(1, 0), "q6": off(0, 2),
                      "q7": off(0, 2.6)},
               "c2": {"q10": off(1.06, 0), "q11": off(1.2, 0)}, "c3": {"q9": off(2.4, 0)}}
    helper = [0.075, 0.15, 0.3]
    golds = {"c1": [ladder_gold(q, off(1, 0), helper) for q in ("q1", "q2", "q3", "q4", "q5")]
             + [ladder_gold("q8", off(1, 0), [0.25, 0.5, 1.0, 2.0]), ladder_gold("q12", off(1, 0), [0.1, 0.25, 0.5])]
             + [ladder_gold(q, off(0, 2), helper) for q in ("q6", "q7")],
             "c2": [ladder_gold("q10", off(1, 0), [0.05, 0.1, 0.2]), ladder_gold("q11", off(1, 0), [0.125, 0.25, 0.5])],
             "c3": [ladder_gold("q9", off(2, 0), [0.25, 0.5, 1.0])]}
    run, data = write_run(tmp_path, rows, answers, golds)
    lines = distance(run, data).splitlines()
    head = "| job, family | n | answered | C (m) | score | within C/2 | within C | within 2C | median error (m) |"
    assert lines.count(head) == 2   # a table a layout
    for expected in ("## Clearance, apartment_a", "## Clearance, warehouse_a",
                     "| helper/1, ego_mem | 2 | 2 | 0.150 | +0.000 | 0.500 | 0.500 | 0.500 | 0.300 |",
                     "| helper/1, ego_now | 5 | 4 | 0.150 | +0.200 | 0.250 | 0.500 | 0.750 | 0.200 |",
                     "| helper/1, all | 7 | 6 | 0.150 | +0.143 | 0.333 | 0.500 | 0.667 | 0.200 |",
                     "| vacuum/2, ego_now | 2 | 2 | 0.100-0.250 | +0.500 | 0.000 | 1.000 | 1.000 | 0.130 |",
                     "| amr/1, ego_now | 1 | 1 | 0.500 | +0.500 | 0.000 | 1.000 | 1.000 | 0.400 |"):
        assert expected in lines, expected
    assert not any(line.startswith("| vacuum/2, all") for line in lines)   # one family: no line for all
    assert "## apartment_a, ego_now" not in lines   # no range_m, so no band section


def shares_rows():
    """Eight rows in one script, four of the `distance` ability (`ego_now`, `now`) and four of `memory`
    (`object_where`), with the credit `s` and the penalised score at k = 1 (2s - 1 on an answer)."""
    def row(i, family, s, score, outcome, scope, slice_="answerable"):
        return {**result_row("c1", f"q{i}", score, outcome, script="s", family=family, scope=scope, slice_=slice_,
                             job="helper/1"), "s": s}
    return [row(1, "ego_now", 1.0, 1.0, "correct", "now"), row(2, "ego_now", 2 / 3, 1 / 3, "partial", "now"),
            row(3, "ego_now", 1 / 3, -1 / 3, "partial", "now"), row(4, "ego_now", 0.0, 0.0, "abstained", "now"),
            row(5, "object_where", 1.0, 1.0, "correct", "memory"),
            row(6, "object_where", 0.0, -1.0, "wrong", "memory"),
            row(7, "object_where", 1.0, 1.0, "correct", "memory", "unknown"),
            row(8, "object_where", 0.0, -1.0, "wrong", "memory", "unknown")]


def test_the_report_gives_the_credit_bands_the_unpenalised_credit_and_the_abstention_bound():
    """Contract (FD-0.7.0 review, change 11; the tiers of the scorer): `results.md` gives per ability the shares of
    full, within need, right thing, direction, miss and `unknown` answers, the mean penalised score, the mean
    unpenalised credit and the mean had every answerable answer scoring below 0 been `unknown`, and the same two for the
    headline, beside the unchanged headline. By hand, `distance` (4 questions): credits 1, 2/3, 1/3 and an `unknown` on
    an answerable question are full, right thing, direction, unknown (1/4 each); scores 1, 1/3, -1/3, 0 average +0.250;
    the credits average (1 + 2/3 + 1/3 + 0)/4 = 0.500;
    under the bound -1/3 becomes 0 and the mean is (1 + 1/3)/4 = +0.333. `memory` (4): an answerable 1 (full) and a miss
    (-1), a correct `unknown` on an unknown question (credit 1, band `unknown`) and an answer there (a miss, -1):
    full 1/4, miss 2/4, unknown 1/4; the mean 0; the credit (1 + 0 + 1 + 0)/4 = 0.500; under the bound only the
    answerable -1 becomes 0 and the unknown question's stays: (1 + 0 + 1 - 1)/4 = +0.250. The headline is the mean of
    the two columns, +0.125; the credit's (0.5 + 0.5)/2 = +0.500 and the bound's (1/3 + 1/4)/2 = +0.292. Regression
    caught: a wrong answer on an unknown question lifted by the bound (it would read +0.750, crediting the right answer
    itself), a negative partial score kept, a bound or a credit that moves the headline, a correct `unknown` counted as
    full, or shares over the wrong denominator."""
    lines = render(results_doc(shares_rows(), n_boot=50)).splitlines()
    for expected in ("| headline | 8 | +0.125 | n/a |",
                     "| unpenalised credit (the headline's statistic over the credit of each answer) | 8 | +0.500 "
                     "| n/a |",
                     "| abstention bound (the headline had every answerable answer scoring below 0 been `unknown`) "
                     "| 8 | +0.292 | n/a |",
                     "| ability | n | full | within need | right thing | direction | miss | unknown | mean | credit | "
                     "abstention bound |",
                     "| distance | 4 | 0.250 | 0.000 | 0.250 | 0.250 | 0.000 | 0.250 | +0.250 | +0.500 | +0.333 |",
                     "| memory | 4 | 0.250 | 0.000 | 0.000 | 0.000 | 0.500 | 0.250 | +0.000 | +0.500 | +0.250 |"):
        assert expected in lines, expected
    assert any(line.startswith("The abstention bound is the score the same answers would get") for line in lines)


def test_the_bound_and_the_credit_count_the_questions_the_headline_counts():
    """Contract: the abstention bound and the unpenalised credit are the headline's own statistic over replaced values,
    so they count what the headline counts: with the (helper/1, memory) pair thin (under 20 questions in the dataset,
    counted in no mean) the headline is the `now` column alone, +0.250, its credit 0.500 and its bound +0.333; the thin
    ability still shows its bands and its bound (+0.250). `bound_score` itself keeps a score of 0 or more (0.5, 0),
    makes a negative answerable one 0 and keeps an unknown question's (-1). `credit_band` puts credit 0.6667 at right
    thing (a tier met up to rounding) and an answer on an unknown question at miss. A document whose rows are not the
    ones its aggregate counted is refused. Regression caught: a bound aggregated without the layout's thin pairs,
    which would put a diagnostic ability into the number beside a headline that leaves it out; a report that renders a
    document whose aggregate belongs to other rows."""
    doc = results_doc(shares_rows(), n_boot=50, thin=frozenset({("helper/1", "memory")}))
    lines = render(doc).splitlines()
    assert "| headline | 4 | +0.250 | n/a |" in lines
    assert ("| unpenalised credit (the headline's statistic over the credit of each answer) | 4 | +0.500 | n/a |"
            in lines)
    assert ("| abstention bound (the headline had every answerable answer scoring below 0 been `unknown`) | 4 | +0.333 "
            "| n/a |" in lines)
    assert ("| memory (diagnostic) | 4 | 0.250 | 0.000 | 0.000 | 0.000 | 0.500 | 0.250 | +0.000 | +0.500 | +0.250 |"
            in lines)
    kept = [result_row("c", "q", score, "wrong", slice_=slice_)
            for score, slice_ in ((0.5, "answerable"), (0.0, "answerable"), (-0.5, "answerable"), (-1.0, "unknown"))]
    assert [bound_score(row) for row in kept] == [0.5, 0.0, 0.0, -1.0]
    assert credit_band({**kept[0], "s": 0.6666666666667, "outcome": "partial"}) == "right thing"
    assert credit_band({**kept[3], "s": 0.0}) == "miss"
    with pytest.raises(ValueError, match="aggregate counts"):
        render({**doc, "rows": doc["rows"][:5]})


def sensitivity_inputs():
    """Six rows of one case and script, their golds and answers, on the `helper/1` job whose re-find radius is R (read
    from the registry as an input, never an expected value). The gold ladders: the offsets [0.1, 0.2, 0.4] (answers 0.3
    and 0.5 off), `path_ahead` [0.25, 0.5, 1, 2] (0.3 off), the places [R/2 + w, R + w, 2R + w] with w = 0.2 (answers
    0.8 R + w and 1.5 R + w off) and the time ladder [3, 10, 30] seconds (5 s off)."""
    reach = job_registry.get("helper/1").reacquire_m
    w = 0.2
    place = [reach / 2 + w, reach + w, 2 * reach + w]
    info = {"case": "c1", "script": "s", "layout": "apartment_a", "robot": "helper", "job": "helper/1",
            "length": "short", "when": "end"}

    def gold(qid, family, scope, value, rungs):
        return {"qid": qid, "family": family, "gold": value, "slice": "answerable", "answer_class": None,
                "range_m": None, "credit": "ladder", "region": None, "tolerances": {"ladder": rungs}, "metric": None,
                "scope": scope, "reference": None, "critical": False}

    origin, at = {"type": "position", "xy": [0, 0]}, lambda x: {"type": "position", "xy": [x, 0]}
    spec = [("e1", "ego_now", "now", off(1, 0), [0.1, 0.2, 0.4], off(1.3, 0)),
            ("e2", "ego_now", "now", off(1, 0), [0.1, 0.2, 0.4], off(1.5, 0)),
            ("a1", "path_ahead", "now", {"type": "number", "value": 3.0}, [0.25, 0.5, 1.0, 2.0],
             {"type": "number", "value": 3.3}),
            ("w1", "object_where", "memory", origin, place, at(0.8 * reach + w)),
            ("w2", "object_where", "memory", origin, place, at(1.5 * reach + w)),
            ("t1", "when_seen", "memory", {"type": "time", "clock": "2026-01-01T10:00:00"}, [3, 10, 30],
             {"type": "time", "clock": "2026-01-01T10:00:05"})]
    rows = [{**info, "qid": q} for q, *_ in spec]
    golds = {("c1", q): gold(q, family, scope, value, rungs) for q, family, scope, value, rungs, _ in spec}
    return rows, golds, {("c1", q): answer for q, *_, answer in spec}


def test_the_credit_sensitivity_scales_the_offsets_and_the_places_apart_and_nothing_else():
    """Contract (FD-0.7.0 §2.3, review change 2): `credit_sensitivity` re-scores the same answers through `score.row`
    twice. The C line multiplies the rungs of the offset golds; the R line multiplies the R part of an `object_where`
    gold whose ladder is [R/2 + w, R + w, 2R + w] for its job and keeps w (the new rungs are (rung - w) x factor + w).
    Every other ladder stays: `path_ahead`'s, the time ladder, a relative `object_where` ladder. By hand, the
    penalised score is 2s - 1, s the share of rungs met:
    - offsets 0.3 and 0.5 off (0.1, 0.2, 0.4): x1 s = 1/3 and 0 (-1/3, -1; mean -2/3); x0.5 (.05, .1, .2) s = 0 and 0
      (-1); x2 (.2, .4, .8) s = 2/3 and 1/3 (1/3, -1/3; mean 0);
    - places 0.8 R + w and 1.5 R + w off: x1 s = 2/3 and 1/3 (mean 0); x0.5 (R/4 + w, R/2 + w, R + w) s = 1/3 and 0
      (mean -2/3); x2 (R + w, 2R + w, 4R + w) s = 1 and 2/3 (mean 2/3);
    - `path_ahead` 0.3 off (.25, .5, 1, 2): s = 3/4, score 1/2 at every factor; the time answer 5 s off: s = 2/3, 1/3.
    The `now` column is the mean of `distance` and `path_ahead`, the `memory` column the mean of `memory` (the places)
    and `when_seen`. C line: now -1/4, -1/12, 1/4 at x0.5, 1, 2 with memory (0 + 1/3)/2 = 1/6 fixed: headline -1/24,
    1/24, 5/24. R line: now -1/12 fixed with memory -1/6, 1/6, 1/2: headline -1/8, 1/24, 5/24. Two offsets and two
    places are scaled of the six questions. Regression caught: a re-implemented credit, the widening scaled with R
    (the x2 places would score 2/3 and 1/3), a number or time ladder scaled with the offsets, the two lines swapped or
    merged, or a headline pooling the questions and not the columns."""
    rows, golds, answers = sensitivity_inputs()
    (entry,) = credit_sensitivity(rows, golds, answers).values()
    assert (entry["questions"], entry["scaled"]) == (6, {"C": 2, "R": 2})
    got = {line: [stat["mean"] for stat in entry["headline"][line].values()] for line in ("C", "R")}
    assert list(entry["headline"]["C"]) == [0.5, 1.0, 2.0]
    assert got["C"] == pytest.approx([-1 / 24, 1 / 24, 5 / 24])
    assert got["R"] == pytest.approx([-1 / 8, 1 / 24, 5 / 24])
    reach = job_registry.get("helper/1").reacquire_m
    place = golds["c1", "w1"]
    assert place_widening(place, reach) == pytest.approx(0.2)
    assert scaled_place(place, 2.0, reach)["tolerances"]["ladder"] == pytest.approx(
        [reach + 0.2, 2 * reach + 0.2, 4 * reach + 0.2])
    assert place["tolerances"]["ladder"][1] == pytest.approx(reach + 0.2)   # the input is not changed
    relative = {**place, "tolerances": {"ladder": [0.1, 0.25, 0.5]}}
    assert place_widening(relative, reach) is None and scaled_place(relative, 2.0, reach) is relative
    for kept in ("a1", "t1", "w1"):   # not an offset: the C line leaves it
        assert scaled_offset(golds["c1", kept], 2.0) is golds["c1", kept]
    assert scaled_offset(golds["c1", "e1"], 2.0)["tolerances"]["ladder"] == [0.2, 0.4, 0.8]
    with pytest.raises(ValueError, match="factor"):
        scaled_offset(golds["c1", "e1"], 0)

    text = sensitivity_lines({"apartment_a": entry}, {"apartment_a": {"mean": 1 / 24, "ci": None}})
    assert "- the x1 headline is the published headline" in text
    for expected in ("### C: 2 of 6 questions: offset, number and heading answers, the clearance C scaled",
                     "### R: 2 of 6 questions: `object_where` places, the re-find radius R scaled",
                     "| x0.5 | -0.042 | n/a |", "| x2 | +0.208 | n/a |", "| x0.5 | -0.125 | n/a |"):
        assert expected in text, expected
    other = sensitivity_lines({"apartment_a": entry}, {"apartment_a": {"mean": 0.5, "ci": None}})
    assert any(line.startswith("- the x1 headline is NOT the published one (+0.500)") for line in other)


def test_the_sensitivity_of_a_run_reads_its_answers_and_flags_answers_it_was_not_scored_on(tmp_path):
    """Contract: `eval_sensitivity.sensitivity` re-scores a run from its case folders' answers and the dataset's golds,
    counting the pairs `ability_counts.thin_rule` does, and its factor 1 equals the published headline. By hand, the
    hand dataset under `all_unknown` (golds [answerable, answerable, unknown] and [answerable, unknown, unknown], each
    an unknown scoring +1 and an answerable 0): `path_ahead` 0 on the `now` side, `object_where` (0 + 1 + 1 + 1)/4 on
    the `memory` side, headline 0.375 at every factor of both lines (abstaining ignores the rungs; no offset gold, and
    the two-rung `object_where` ladders are not on R's rungs, so 0 of 6 questions are scaled). Then one answerable
    `object_where` answer is replaced by the gold position: the memory side is 1 and the headline 0.5 at every factor,
    but the published headline does not move, and the report says so. Regression caught: answers read from the wrong
    folder, a thin rule not applied as the run applied it, or a mismatch that prints as agreement."""
    hand_dataset(tmp_path)
    out = tmp_path / "run"
    assert cli("all_unknown", "--data", tmp_path, "--out", out) == 0
    lines = sensitivity(out, tmp_path).splitlines()
    assert "### C: 0 of 6 questions: offset, number and heading answers, the clearance C scaled" in lines
    assert "### R: 0 of 6 questions: `object_where` places, the re-find radius R scaled" in lines
    for expected in ("| x0.5 | +0.375 |", "| x1 | +0.375 |", "| x2 | +0.375 |"):
        assert sum(line.startswith(expected) for line in lines) == 2, expected   # a line each
    assert "- the x1 headline is the published headline" in lines

    doc = results(out)
    row = next(x for x in doc["rows"] if x["family"] == "object_where" and x["slice"] == "answerable")
    path = out / "cases" / row["case"] / "answers.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["qid"] == row["qid"]:
            record["answer"] = {"type": "position", "xy": [3.2, 1.5]}   # the gold of `stream_kit.gold_row`
    path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
    lines = sensitivity(out, tmp_path).splitlines()
    assert any(line.startswith("- the x1 headline is NOT the published one (+0.375)") for line in lines)
    assert sum(line.startswith("| x1 | +0.500 |") for line in lines) == 2


def test_the_distance_command_prints_the_clearance_report_and_the_sensitivity(tmp_path, capsys):
    """Contract (review change 4): `stbench eval distance RUN --data D` prints the distance report and then the credit
    sensitivity, and `--out` writes both; a run folder with no results still exits 2 with one sentence. By hand, the
    hand dataset under `all_unknown` has no offset question, so the distance part is its one sentence and the
    sensitivity reads the headline 0.375 at the three factors. Regression caught: a command that prints the bands and
    not the sensitivity, or one that writes only the first to `--out`."""
    hand_dataset(tmp_path)
    out = tmp_path / "run"
    assert cli("all_unknown", "--data", tmp_path, "--out", out) == 0
    capsys.readouterr()
    assert cli("distance", out, "--data", tmp_path, "--out", tmp_path / "report.md") == 0
    printed = capsys.readouterr().out
    assert printed.startswith("No offset question was scored in this run.")
    assert "# SpatiotemporalBench: credit sensitivity" in printed and "| x2 | +0.375 |" in printed
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == printed
    assert cli("distance", tmp_path / "nowhere", "--data", tmp_path) == 2
