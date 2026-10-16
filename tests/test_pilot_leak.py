"""`stbench pilot report --leak` in its two readings (`pilot_leak`, FD-0.11 §3.6): the verdict reads the pairs whose
gold is answerable, the pairs whose gold is `unknown` are an abstention shift with no verdict, and each (job, family)
has its rows. The results documents are built by hand (`test_cmd_pilot`'s helpers), with no run behind them."""

from __future__ import annotations

from test_cmd_pilot import SOL, case, doc, report, row, run, save

CASES = {"c1": "s1", "c2": "s2"}


def leak_docs(tmp_path, ends: dict[str, list[tuple]]):
    """(ON folder, OFF folder) of a vacuum job with one case a script; `ends` is {case: [(qid, family, gold slice, on
    score, off score)]}, the end questions of each, every ON case also asking a during question `d1` (never paired)."""
    on, off = [], []
    for c, questions in ends.items():
        s = CASES[c]
        on.append(row(c, s, "d1", 1, job="vacuum", when="during"))
        for qid, family, slice_, a, b in questions:
            on.append({**row(c, s, qid, a, job="vacuum", family=family), "slice": slice_})
            off.append({**row(c, s, qid, b, job="vacuum", family=family), "slice": slice_})
    records = [case(c, s, "vacuum", "short") for c, s in CASES.items()]
    return save(tmp_path, "on", doc(SOL, records, on)), save(tmp_path, "off", doc(SOL, records, off, during=False))


def leak_report(capsys, tmp_path, ends):
    return report(capsys, "--leak", *leak_docs(tmp_path, ends)).splitlines()


def test_the_verdict_reads_answerable_pairs_and_the_unknown_ones_are_an_abstention_shift(tmp_path, capsys):
    """Contract (FD-0.11 §3.6): asking during the job changes whether the agent says `unknown`, not what it recalls, so
    the pooled verdict and the per-job rows read only the pairs with an answerable gold, and the pairs with an
    `unknown` gold are a row of their own with a d and an interval and no verdict. By hand, two scripts, each with an
    `object_where` and a `task_things` answerable pair that both arms answer alike (d 0 each) and an `object_where`
    `unknown` pair the on arm answers right (1) and the off arm wrong (-1), d +2: the four answerable pairs give d
    +0.000 [+0.000, +0.000], a pass; the two unknown pairs +2.000 [+2.000, +2.000]; all six pairs +0.667 (script totals
    +2 and +2 over three pairs each, so every resample has mean 2/3), which would fail the 0.05 margin were the verdict
    to read them. Regression caught: the verdict pooling the abstention shift, which calls a calibration effect a
    leak (the 0.9.0 report: vacuum fail on two pairs)."""
    ends = {c: [("e1", "object_where", "answerable", 1, 1), ("e2", "task_things", "answerable", 0.5, 0.5),
                ("e3", "object_where", "unknown", 1, -1)] for c in CASES}
    lines = leak_report(capsys, tmp_path, ends)
    assert "| vacuum (apartment_a) | 4 | 2 | +0.000 | [+0.000, +0.000] | pass |" in lines
    assert "| pooled | 4 | 2 | +0.000 | [+0.000, +0.000] | pass |" in lines
    assert "Pooled verdict: **pass**: the during questions ship." in lines
    assert "| vacuum (apartment_a) | 2 | 2 | +2.000 | [+2.000, +2.000] |" in lines   # the abstention shift: no verdict
    assert "| pooled | 2 | 2 | +2.000 | [+2.000, +2.000] |" in lines
    assert "| all pairs (the reading 0.9.0 published) | 6 | 2 | +0.667 | [+0.667, +0.667] |" in lines
    assert "### Abstention shift" in lines and "### By family" in lines


def test_each_job_and_family_has_its_own_rows_for_both_golds(tmp_path, capsys):
    """Contract (FD-0.11 §3.6): the by-family table gives each (job, family) a row per gold it has pairs for, with the
    pairs, scripts, d and interval, so a family whose own d rises is seen where it is. By hand, over the documents of
    the test above and a second `task_things` `unknown` pair for the first script only (on 0, off 1: d -1): the
    `object_where` answerable row is 2 pairs, 2 scripts, +0.000; its `unknown` row 2, 2, +2.000; `task_things`
    answerable 2, 2, +0.000, and its `unknown` row 1 pair, 1 script, -1.000 with no interval
    (one script has none). A family with no pair of a gold has no row for it. Regression caught: families pooled, or a
    row for a gold with no pair."""
    ends = {c: [("e1", "object_where", "answerable", 1, 1), ("e2", "task_things", "answerable", 0.5, 0.5),
                ("e3", "object_where", "unknown", 1, -1)] for c in CASES}
    ends["c1"].append(("e4", "task_things", "unknown", 0, 1))
    lines = leak_report(capsys, tmp_path, ends)
    assert "| vacuum | object_where | answerable | 2 | 2 | +0.000 | [+0.000, +0.000] |" in lines
    assert "| vacuum | object_where | unknown | 2 | 2 | +2.000 | [+2.000, +2.000] |" in lines
    assert "| vacuum | task_things | answerable | 2 | 2 | +0.000 | [+0.000, +0.000] |" in lines
    assert "| vacuum | task_things | unknown | 1 | 1 | -1.000 | n/a |" in lines
    assert len([x for x in lines if x.startswith("| vacuum | ")]) == 4


def test_a_leak_pair_set_with_no_answerable_end_question_is_refused(tmp_path, capsys):
    """Contract (FD-0.11 §3.6): the verdict reads answerable pairs, so pairs that are all `unknown`-gold cannot give
    one, and the report is refused (exit 2, naming why) rather than passed on nothing. By hand: the documents hold only
    `unknown`-gold end questions. Regression caught: a verdict of `pass` from an empty set of pairs, or a traceback."""
    ends = {c: [("e1", "object_where", "unknown", 1, -1)] for c in CASES}
    assert run("--leak", *leak_docs(tmp_path, ends)) == 2
    assert "hold no end question with an answerable gold" in capsys.readouterr().err


def test_a_job_with_pairs_of_only_one_gold_shows_a_dash_for_the_other(tmp_path, capsys):
    """Contract (FD-0.11 §3.6): a job with no pair of a reading has a dash row in that reading's table, not a missing
    job or a traceback (the AMR asks no `unknown`-gold end question in 0.9.0). By hand: a vacuum job with answerable
    pairs only has the dash row `| vacuum (apartment_a) | - | - | n/a | n/a |` under the abstention shift. Regression
    caught: a job that vanishes from the shift table, or a crash on an empty split."""
    ends = {c: [("e1", "object_where", "answerable", 1, 0), ("e2", "task_things", "answerable", 0, 0)] for c in CASES}
    lines = leak_report(capsys, tmp_path, ends)
    assert "| vacuum (apartment_a) | 4 | 2 | +0.500 | [+0.500, +0.500] | fail |" in lines
    assert "| vacuum (apartment_a) | - | - | n/a | n/a |" in lines
    assert "| pooled | - | - | n/a | n/a |" in lines
