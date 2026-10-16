"""The clearance sections of `eval distance` on graded rows: the shares within the question's resolution, its
need and twice its need, and the share of the answered errors on the harmful side. Rows, golds and answers are written
here and every number is worked by hand."""

from __future__ import annotations

import json

from ability_kit import r, results_doc
from stbench import eval_results
from stbench.eval_distance import distance, graded_rungs_of


def off(ahead, left):
    return {"type": "offset", "ahead_m": ahead, "left_m": left}


def graded_gold(qid, gold, **tolerances):
    return {"qid": qid, "gold": gold, "credit": "graded",
            "tolerances": {"resolution_m": 0.02, "need_m": 0.15, "harm": "radial", **tolerances}}


def write_run(root, rows, answers, golds):
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


def test_the_clearance_section_reads_the_graded_rows_resolution_and_need_and_the_harmful_share(tmp_path):
    """Contract (plan S1, `eval_distance`): the clearance section of a graded row has the columns `within rho`,
    `within N`, `within 2N` and `harmful share`, read from the gold row's `resolution_m` and `need_m` (the scorer's own
    numbers), the harmful share being the answered errors that point away from the camera along the gold's bearing; a
    layout holding a ladder row as well gets its table too. By hand, the helper's `ego_now` golds at (1, 0),
    rho 0.02, N = 0.15, answers:
    - (1.0, 0): error 0, inside rho, N and 2N, not harmful;
    - (1.05, 0): 0.05, inside N and 2N, farther than the gold: harmful;
    - (1.0, 0.25): 0.25, inside 2N = 0.30 only, sideways: not harmful;
    - (0.5, 0): 0.5, outside all, nearer: not harmful;
    - `unknown`: not answered.
    5 questions, 4 answered, the need 0.150, the mean of the scores 1, .5, 0, -1, 0 +0.100, within rho 1/4, within N
    2/4, within 2N 3/4, harmful 1/4, the median (0.05 + 0.25) / 2 = 0.150. A ladder offset [0.075, 0.15, 0.3] answered
    0.1 off in the same layout is in a table of its own: shares 0, 1, 1 and the median 0.100. Regression caught: a
    graded row read as a ladder of its resolution and need (the shares would be of C/2), a harmful share read on the
    wrong sign, or a ladder row lost from the layout."""
    ego = lambda qid, score: r("c1", qid, score, "ego_now", script="c1")   # noqa: E731
    rows = [ego("q1", 1.0), ego("q2", 0.5), ego("q3", 0.0), ego("q4", -1.0), ego("q5", 0.0), ego("q6", 0.5)]
    answers = {"c1": {"q1": off(1.0, 0), "q2": off(1.05, 0), "q3": off(1.0, 0.25), "q4": off(0.5, 0),
                      "q5": {"type": "unknown"}, "q6": off(1.1, 0)}}
    golds = {"c1": [graded_gold(q, off(1, 0)) for q in ("q1", "q2", "q3", "q4", "q5")]
             + [{"qid": "q6", "gold": off(1, 0), "tolerances": {"ladder": [0.075, 0.15, 0.3]}}]}
    run, data = write_run(tmp_path, rows, answers, golds)
    lines = distance(run, data).splitlines()
    graded_head = ("| job, family | n | answered | need (m) | score | within rho | within N | within 2N "
                   "| harmful share | median error (m) |")
    ladder_head = ("| job, family | n | answered | C (m) | score | within C/2 | within C | within 2C "
                   "| median error (m) |")
    assert graded_head in lines and ladder_head in lines
    assert "| helper/1, ego_now | 5 | 4 | 0.150 | +0.100 | 0.250 | 0.500 | 0.750 | 0.250 | 0.150 |" in lines
    assert "| helper/1, ego_now | 1 | 1 | 0.150 | +0.500 | 0.000 | 1.000 | 1.000 | 0.100 |" in lines


def test_a_graded_row_is_a_clearance_row_only_with_both_its_keys():
    """Contract (`graded_rungs_of`): the rungs of a graded row are (rho, N, 2N) from its own tolerances; a row that is
    not of the graded credit, or lacks either key, is not one. By hand: rho 0.02 and N = 0.15 give (0.02, 0.15, 0.30).
    Regression caught: a row with a missing resolution counted with a default."""
    assert graded_rungs_of(graded_gold("q", off(1, 0))) == (0.02, 0.15, 0.30)
    assert graded_rungs_of({"credit": "ladder", "tolerances": {"resolution_m": 0.02, "need_m": 0.15}}) is None
    assert graded_rungs_of({"credit": "graded", "tolerances": {"need_m": 0.15}}) is None
