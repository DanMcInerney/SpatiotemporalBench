"""`stbench eval compare` (`eval_compare.py`, plan §5.1's B3): results side by side, paired on (case, qid), and the
refusals. The dataset is `eval_kit.hand_dataset`: golds [answerable, answerable, unknown] and [answerable, unknown,
unknown] over a during question (`path_ahead`) and two end ones (`object_where`). `all_unknown` scores 0 on an
answerable gold and 1 on an unknown one: `path_ahead` (the `now` column) 0, `object_where` (the `memory` column) 3/4, so
the headline is 3/8. `fixed_guess` answers every question with the gold of the answerable ones (a number 1.4, a
position (3.2, 1.5)): 1 on an answerable gold and -1 on an unknown one (a wrong answer at t = 0.5), so `path_ahead` 1,
`object_where` -1/2 and the headline (1 - 1/2)/2 = 1/4."""

from __future__ import annotations

import json
import re
import shutil

import pytest

from ability_kit import r, results_doc
from eval_kit import cli, hand_dataset, register_agent, results
from recompile_kit import recompile
from stbench.baselines import BASELINES, Baseline
from stbench.episode import write_jsonl
from stbench.eval_compare import Refused, compare, same_compile
from stbench.eval_report import HEADLINE_RULE
from stbench.stream_protocol import System
from stbench.suite import read_jsonl


class FixedGuess(System):
    """Answers every question as the answerable golds of `eval_kit` are: a number 1.4, a position (3.2, 1.5)."""

    def observe(self, steps):
        pass

    def answer(self, question):
        return ({"type": "number", "value": 1.4} if question["answer_type"] == "number"
                else {"type": "position", "xy": [3.2, 1.5]})


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("compare")
    hand_dataset(root)
    return root


@pytest.fixture
def two(data, tmp_path, monkeypatch):
    """Two results of the same dataset: (all_unknown's folder, fixed_guess's folder)."""
    monkeypatch.setitem(BASELINES, "fixed_guess", Baseline(FixedGuess))
    assert cli("all_unknown", "--data", data, "--out", tmp_path / "au") == 0
    assert cli("fixed_guess", "--data", data, "--out", tmp_path / "fg") == 0
    return tmp_path / "au", tmp_path / "fg"


def cells(md: str, start: str) -> list[str]:
    """The cells of the first table row of `md` that starts with `start`."""
    line = next(x for x in md.splitlines() if x.startswith(start))
    return [c.strip() for c in line.strip("|").split("|")]


def test_compare_by_hand(two, tmp_path, capsys):
    """Contract (plan §5.1): `eval compare A B` prints each result's headline and family means per layout side by side
    and, for each later result, the difference from the first over the questions both answered, paired on (case, qid),
    with its CI and the number of pairs; a run folder and a `results.json` are both accepted. By hand (module
    docstring): the headlines are 0.375 and 0.25 (the mean of the `now` and `memory` lines: (0 + 0.75)/2 and
    (1 - 0.5)/2); paired per question fixed_guess minus all_unknown is 1, 1, -2 in the first case and 1, -2, -2 in the
    second. The `now` difference is (1 + 1)/2 = +1.0 over 2 and the `memory` one (1 - 2 - 2 - 2)/4 = -1.25 over 4
    (`object_where` is all the `memory` rows), so the headline's difference is (1 - 1.25)/2 = -0.125 over 6 questions
    (pooled over the six it would be -0.5). Regression caught: a difference with the wrong sign or paired by position,
    a mean that is not the result's own, a headline difference that pools the questions the columns weigh equally, or
    a missing statement of the rule."""
    capsys.readouterr()
    assert cli("compare", two[0], two[1] / "results.json", "--out", tmp_path / "cmp.md") == 0
    md = capsys.readouterr().out
    assert md == (tmp_path / "cmp.md").read_text(encoding="utf-8")
    assert "## apartment_a\n" in md and "end questions only" not in md
    head, where, path, now, memory = (cells(md, f"| {k}") for k in (
        "**headline**", "object_where", "path_ahead", "`now`", "`memory`"))
    assert [c.split(" ")[0] for c in head[1:]] == ["+0.375", "+0.250", "-0.125"]
    assert [c.split(" ")[0] for c in where[1:]] == ["+0.750", "-0.500", "-1.250"]
    assert [c.split(" ")[0] for c in path[1:]] == ["+0.000", "+1.000", "+1.000"]
    assert [c.split(" ")[0] for c in now[1:]] == ["+0.000", "+1.000", "+1.000"] and now[3].endswith("over 2")
    assert [c.split(" ")[0] for c in memory[1:]] == ["+0.750", "-0.500", "-1.250"] and memory[3].endswith("over 4")
    assert HEADLINE_RULE in md
    assert re.fullmatch(r"-0\.125 \[[-+]\d\.\d{3}, [-+]\d\.\d{3}\] over 6", head[3])
    assert "A = all_unknown" in md and "B = fixed_guess" in md and "Differences are each column minus A" in md


@pytest.fixture
def critical_two(data, tmp_path, monkeypatch):
    """`two`'s results on a copy of the dataset whose golds are tagged: the last answerable question of each case (the
    second of the first case, the first of the second) is critical."""
    monkeypatch.setitem(BASELINES, "fixed_guess", Baseline(FixedGuess))
    tagged = tmp_path / "tagged"
    shutil.copytree(data, tagged)
    for path in (tagged / "private").glob("*/gold.jsonl"):
        rows = read_jsonl(path)
        at = max(i for i, row in enumerate(rows) if row["slice"] == "answerable")
        rows[at] = {**rows[at], "critical": True}
        write_jsonl(path, rows)
    assert cli("all_unknown", "--data", tagged, "--out", tmp_path / "au") == 0
    assert cli("fixed_guess", "--data", tagged, "--out", tmp_path / "fg") == 0
    return tmp_path / "au", tmp_path / "fg"


def test_compare_gives_the_paired_difference_on_the_critical_rows(critical_two, tmp_path):
    """Contract (FD-0.5.0-impl §2, "Reporting"; DECISIONS "0.5.0 declared": conditioning is shown only if the lower
    bound of the paired critical difference is above 0): compare gives, per layout, the means and the paired difference
    of the critical answerable rows, the situational answerable rows and the unknown rows apart, and says whether the
    later result's critical difference is shown. By hand (`fixed_guess` minus `all_unknown`, the two tagged golds being
    answerable): critical +1 and +1 = +1.0 over 2 pairs, the other answerable one +1.0 over 1, the three unknown golds
    -2 each = -2.0 over 3, while the headline difference is -0.125 over 6 (the `now` +1.0 and the `memory` -1.25,
    weighed equally); each script's critical difference is +1, so the interval is [+1, +1] and the lower bound is above
    0: shown. The other way round (`all_unknown` minus `fixed_guess`) the lower bound is -1: not shown. Regression
    caught: the unknown slice hiding a difference on the critical rows in the headline (-0.125 here, with the sign of
    the unknown rows), or a verdict read off the mean."""
    au, fg = critical_two
    md = compare([str(au), str(fg)])
    head, critical, situational, unknown = (cells(md, f"| {k}") for k in (
        "**headline**", "answerable, critical", "answerable, situational", "unknown"))
    assert [c.split(" ")[0] for c in head[1:]] == ["+0.375", "+0.250", "-0.125"]
    assert [c.split(" ")[0] for c in critical[1:]] == ["+0.000", "+1.000", "+1.000"] and critical[3].endswith("over 2")
    assert [c.split(" ")[0] for c in situational[1:]] == ["+0.000", "+1.000", "+1.000"]
    assert situational[3].endswith("over 1")
    assert [c.split(" ")[0] for c in unknown[1:]] == ["+1.000", "-1.000", "-2.000"] and unknown[3].endswith("over 3")
    assert "- B - A on the critical answerable rows: shown (lower bound +1.000, over 2 pairs)" in md
    flipped = compare([str(fg), str(au)])
    assert "- B - A on the critical answerable rows: not shown (lower bound -1.000, over 2 pairs)" in flipped


def test_compare_of_the_brief_lever_reads_the_critical_rows(critical_two, tmp_path):
    """Contract (FD-0.5.0-impl §5, "Conditioning"): two runs of one system that differ in the brief (`correct` against
    `neutral`) are compared on the critical answerable rows, paired; `all_unknown` ignores its brief, so the difference
    is exactly 0 over the 2 critical pairs and, its lower bound not above 0, is not shown; a result with no critical
    row pairs nothing and is not shown either. Regression caught: a conditioning claim made from a zero or an unpaired
    difference."""
    au, _ = critical_two
    data = au.parent / "tagged"
    assert cli("all_unknown", "--data", data, "--out", au.parent / "neutral", "--brief", "neutral") == 0
    md = compare([str(au), str(au.parent / "neutral")])
    assert cells(md, "| answerable, critical")[3].startswith("+0.000 [+0.000, +0.000] over 2")
    assert "- B - A on the critical answerable rows: not shown (lower bound +0.000, over 2 pairs)" in md
    plain = au.parent / "plain"
    hand_dataset(plain)
    assert cli("all_unknown", "--data", plain, "--out", plain / "a") == 0
    assert cli("all_unknown", "--data", plain, "--out", plain / "b", "--brief", "neutral") == 0
    bare = compare([str(plain / "a"), str(plain / "b")])
    assert "- B - A on the critical answerable rows: not shown (no pairs)" in bare


def test_a_during_pair_compares_the_end_questions_alone(tmp_path):
    """Contract (plan §5.1; design D2): results that differ in `during` have no during rows in the off arm, so every
    mean and every pair is over the end questions alone and the table says so. `all_unknown` with the during questions
    off scores the four end questions 3/4, and so it does with them on once the two during ones are left out (the
    difference over those 4 pairs is 0); the during arm's own headline (3/8) is not shown against it. The off arm's
    qids are a subset of the on arm's, which `same_compile` accepts (the sets meet; they need not be equal).
    Regression caught: a difference over unpaired questions, or means over different question sets shown side by
    side, or a during-off arm refused as another compile."""
    hand_dataset(tmp_path)
    assert cli("all_unknown", "--data", tmp_path, "--out", tmp_path / "arms", "--arms", "during=on,off") == 0
    out = tmp_path / "cmp.md"
    assert cli("compare", tmp_path / "arms" / "during-on", tmp_path / "arms" / "during-off", "--out", out) == 0
    md = out.read_text(encoding="utf-8")
    assert "## apartment_a: end questions only" in md and "over the end questions only" in md
    head = cells(md, "| **headline**")
    assert [c.split(" ")[0] for c in head[1:]] == ["+0.750", "+0.750", "+0.000"] and head[3].endswith("over 4")
    assert "path_ahead" not in md


def test_compare_refuses(data, two, tmp_path, monkeypatch, capsys):
    """Contract (plan §5.1; VISION rule 13): compare refuses (exit 2, saying why, nothing printed) one result, a file
    that is not a results document, results of two levers apart (the system and the brief), results scored on different
    compiles (a case both hold rows of whose qids do not meet: a recompile draws every qid anew), and results that
    share no case. Regression caught: a comparison of two axes at once, an empty table for results that pair nothing,
    or pairs across two compiles (qids are random per compile), which would print `-` in every cell and say nothing."""
    register_agent(monkeypatch)
    assert cli("fake:ok", "--data", data, "--out", tmp_path / "ok") == 0
    assert cli("all_unknown", "--data", data, "--out", tmp_path / "neutral", "--brief", "neutral") == 0
    recompiled = tmp_path / "recompiled"
    shutil.copytree(data, recompiled)
    recompile(recompiled)
    assert cli("all_unknown", "--data", recompiled, "--out", tmp_path / "again") == 0
    other = tmp_path / "other"
    hand_dataset(other)
    assert cli("all_unknown", "--data", other, "--out", tmp_path / "apart") == 0
    doc = results(two[0])
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "results.json").write_text(json.dumps({"system": doc["system"]}), encoding="utf-8")
    capsys.readouterr()
    for argv, message in (
            ([two[0]], "between 2 and 26 results"),
            ([two[0], tmp_path / "broken"], "is not a results document"),
            ([two[0], tmp_path / "missing"], "cannot be read"),
            ([two[0], tmp_path / "ok"], None),   # the system alone differs: allowed
            ([two[0], tmp_path / "neutral"], None),   # the brief alone differs: allowed
            ([tmp_path / "ok", tmp_path / "neutral"], "differ in more than one lever (agent, brief)"),
            ([two[0], tmp_path / "again"], "were scored on different compiles"),
            ([two[0], tmp_path / "apart"], "shares no case")):
        code = cli("compare", *argv)
        shown = capsys.readouterr()
        if message is None:
            assert code == 0 and shown.out.startswith("# SpatiotemporalBench: comparison"), argv
        else:
            assert code == 2 and shown.out == "" and message in shown.err, (argv, shown.err)


def test_same_compile_asks_one_shared_qid_a_case_and_skips_a_case_without_rows():
    """Contract: `same_compile` refuses two results only when a case both hold rows of has two qid sets that do not
    meet; equal sets are not asked (a `during` off arm holds a subset of the on arm's), and a case that one result
    holds no rows of (the drone's during-off arm asks nothing there) is skipped, the other result's rows being all
    there is. Regression caught: equality demanded of the sets (a during-off arm refused), a case with no rows read as
    disjoint from the other result's, or a recompile of one case of several going unnoticed."""
    both = results_doc([r("c1", "q1", 1, "ego_now"), r("c1", "q2", 1, "ego_now"), r("c2", "q9", 1, "ego_now")])
    off = results_doc([r("c1", "q2", 1, "ego_now")])   # a subset of c1's qids, and no rows of c2
    same_compile(["both", "off"], [both, off])
    same_compile(["off", "both"], [off, both])
    other = results_doc([r("c1", "q7", 1, "ego_now")])   # c1 drawn anew
    with pytest.raises(Refused, match=r"other and both were scored on different compiles: case c1 shares no question"):
        same_compile(["both", "other"], [both, other])
    third = results_doc([r("c3", "q1", 1, "ego_now")])   # a case none of the others holds
    same_compile(["both", "third"], [both, third])
    one_of_two = results_doc([r("c1", "q1", 1, "ego_now"), r("c2", "zz", 1, "ego_now")])   # c2 drawn anew, c1 not
    with pytest.raises(Refused, match="case c2 shares no question"):
        same_compile(["both", "mixed"], [both, one_of_two])
