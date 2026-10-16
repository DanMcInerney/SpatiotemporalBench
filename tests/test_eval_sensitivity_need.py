"""The credit sensitivity on the need (`eval_sensitivity`): the C line scales `need_m` of the offsets,
numbers and headings, the R line `need_m` of the places, and nothing else moves. Rows, golds and answers are written
here and every expected number is worked by hand from the scorer's four quarters, never by the code under test."""

from __future__ import annotations

import pytest

from stbench.eval_sensitivity import credit_sensitivity, scaled_offset, scaled_place, sensitivity_lines

INFO = {"case": "c1", "script": "s", "layout": "apartment_a", "robot": "helper", "job": "helper/1", "length": "short",
        "when": "end"}


def graded(qid, family, scope, value, tolerances, credit="graded"):
    return {"qid": qid, "family": family, "gold": value, "slice": "answerable", "answer_class": None, "range_m": None,
            "credit": credit, "region": None, "tolerances": tolerances, "metric": None, "scope": scope,
            "reference": None, "critical": False}


def inputs():
    """Three rows of one script: an `ego_now` offset (gold 1 m ahead, C = 0.10, answered 0.3 m to the side), a
    `clear_path` number (gold 3.0 m, C = 0.15, T = 0.49, answered 2.7) and an `object_where` place (R = 1.0, answered
    1.5 m off)."""
    golds = {("c1", "e1"): graded("e1", "ego_now", "now", {"type": "offset", "ahead_m": 1.0, "left_m": 0.0},
                                  {"resolution_m": 0.02, "need_m": 0.10, "harm": "radial"}),
             ("c1", "n1"): graded("n1", "clear_path", "now", {"type": "number", "value": 3.0},
                                  {"resolution_m": 0.02, "need_m": 0.15, "harm": "over", "threshold_m": 0.49}),
             ("c1", "p1"): graded("p1", "object_where", "memory", {"type": "position", "xy": [0, 0]},
                                  {"resolution_m": 0.02, "need_m": 1.0, "harm": None})}
    answers = {("c1", "e1"): {"type": "offset", "ahead_m": 1.0, "left_m": 0.3},
               ("c1", "n1"): {"type": "number", "value": 2.7}, ("c1", "p1"): {"type": "position", "xy": [1.5, 0]}}
    return [{**INFO, "qid": q} for _, q in golds], golds, answers


def test_the_c_line_scales_the_need_of_offsets_and_numbers_and_the_r_line_that_of_places():
    """Contract (brief §3, plan S1): `credit_sensitivity` re-scores the same answers, the C line with `need_m` of every
    offset and number gold multiplied by the factor, the R line with that of the places, nothing else moving. The need
    is max(need_m, a tenth of the gold's distance) (FD-0.11 §3.1), so a scaled C below that tenth is the tenth. By hand,
    the penalised score is 2s - 1:
    - the offset (1 m ahead, so a tenth is 0.10), error 0.3 sideways (direction right by bearing): x0.5 (N = 0.05, so
      0.10) t = (0.4 - 0.3) / 0.2 = 0.5, s = 0.375; x1 (N = 0.10) the same; x2 (N = 0.20) t = 1, a = 0.5, s = 0.625:
      scores -0.25, -0.25, 0.25;
    - the number, 2.7 against 3.0 (cautious; a tenth is 0.30, over C at every factor): N = 0.30 at x0.5, x1 and x2,
      t = a = 1, f = 0, s = 0.75: scores 0.5 at each;
    - the place, 1.5 m off: x1 (R = 1) a = 0.5, s = 0.375; x0.5 (R = 0.5) t = 0.5, s = 0.125; x2 (R = 2) the
      error is within R so the direction counts, t = a = 1, f = 1 - ln 75 / ln 100 = 0.0625, s = 0.7656: scores -0.25,
      -0.75, +0.5312.
    The `now` column is the mean of the offset's and the number's cells, the `memory` column the place's. C line:
    now +0.125, +0.125, +0.375 with memory -0.25 fixed: headline -0.0625, -0.0625, +0.0625. R line: now +0.125 fixed
    with memory -0.75, -0.25, +0.5312: headline -0.3125, -0.0625, +0.3281. Two of the three are scaled on C and one on
    R.
    Regression caught: a scorer re-implemented here that disagrees with `score.row`, the lines swapped, the place
    scaled by the C line, or a headline that pools the questions and not the columns."""
    rows, golds, answers = inputs()
    (entry,) = credit_sensitivity(rows, golds, answers).values()
    assert (entry["questions"], entry["scaled"]) == (3, {"C": 2, "R": 1})
    got = {line: [stat["mean"] for stat in entry["headline"][line].values()] for line in ("C", "R")}
    assert list(entry["headline"]["C"]) == [0.5, 1.0, 2.0]
    assert got["C"] == pytest.approx([-0.0625, -0.0625, 0.0625])
    assert got["R"] == pytest.approx([-0.3125, -0.0625, 0.3281], abs=1e-4)
    text = sensitivity_lines({"apartment_a": entry}, {"apartment_a": {"mean": -0.0625, "ci": None}})
    assert "### C: 2 of 3 questions: offset, number and heading answers, the clearance C scaled" in text
    assert "### R: 1 of 3 questions: `object_where` places, the re-find radius R scaled" in text
    assert "- the x1 headline is the published headline" in text


def test_a_threshold_moves_with_the_need_it_is_made_of_and_the_resolution_stays():
    """Contract (brief §3.2: T = L + N): scaling C by a factor scales `need_m` and moves `threshold_m` by the change
    in C, keeping the resolution; a heading's profile stays. By hand, a number gold with C = 0.15 and T = 0.49 (a body
    0.34 m long) at x2 has C = 0.30 and T = 0.64, at x0.5 C = 0.075 and T = 0.415. A place has no threshold: its R is
    multiplied alone. The input rows are not changed, a list gold (match radius M) is left to both
    lines, and a factor of 0 or less is refused. Regression caught: a threshold left at the old body-plus-C while C
    changes (the direction quarter of a short corridor would not follow the sensitivity), a resolution scaled with the
    need, or a list scaled as if its match radius were C."""
    _, golds, _ = inputs()
    number = golds["c1", "n1"]
    doubled = scaled_offset(number, 2.0)["tolerances"]
    assert (doubled["need_m"], doubled["threshold_m"]) == pytest.approx((0.30, 0.64))
    assert doubled["resolution_m"] == 0.02 and doubled["harm"] == "over"
    halved = scaled_offset(number, 0.5)["tolerances"]
    assert (halved["need_m"], halved["threshold_m"]) == pytest.approx((0.075, 0.415))
    assert number["tolerances"]["need_m"] == 0.15 and number["tolerances"]["threshold_m"] == 0.49
    place = scaled_place(golds["c1", "p1"], 2.0, 1.0)["tolerances"]
    assert place == {"resolution_m": 0.02, "need_m": 2.0, "harm": None}
    assert scaled_offset(golds["c1", "p1"], 2.0) is golds["c1", "p1"]   # a place is not on the C line
    assert scaled_place(number, 2.0, 1.0) is number                      # nor a number on the R line
    heading = graded("h1", "open_heading", "memory", {"type": "heading", "deg_left": 0},
                     {"resolution_m": 0.02, "need_m": 0.10, "threshold_m": 0.44, "step_deg": 90,
                      "profile_m": [2.0, 1.0, 0.5, None]}, credit="regret")
    scaled = scaled_offset(heading, 2.0)["tolerances"]
    assert (scaled["need_m"], scaled["threshold_m"]) == pytest.approx((0.20, 0.54))
    assert scaled["profile_m"] == [2.0, 1.0, 0.5, None]
    listed = graded("t1", "task_things", "memory", {"type": "positions", "items": []},
                    {"match_m": 0.5, "resolutions_m": [], "fp_weight": 0.5}, credit="graded_set")
    assert scaled_offset(listed, 2.0) is listed and scaled_place(listed, 2.0, 1.0) is listed
    with pytest.raises(ValueError, match="factor"):
        scaled_offset(number, 0)
