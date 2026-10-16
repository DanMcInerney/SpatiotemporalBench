"""`tools/fill_report.py`: the tables of the first-draft report and dataset card, filled from the release's own
outputs (FD-M6-M7 §4, items 7 and 12: pasted, never typed). The inputs are real outputs of the package on its hand-made
datasets (`eval`, `check`, `pilot report`) and hand-written ones where a number is worked out here; the expected cells
are hand values or the text the package itself printed (`results.md`, `eval compare`, the pilot report)."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from check_kit import synthetic_dataset
from eval_kit import COST, PRICES, cli, hand_dataset, register_agent
from recompile_kit import recompile
from stbench import cmd_pilot, prices
from stbench.baselines import BASELINES, Baseline
from stbench.cli import main as stbench_main
from test_cmd_pilot import AMR, SOL, SOL_SCORES, VACUUM, brief_pair, leak_pair, pilot_doc, plan_file, save
from test_eval_compare import FixedGuess

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fill_report", ROOT / "tools" / "fill_report.py")
fr = importlib.util.module_from_spec(SPEC)
sys.modules["fill_report"] = fr
SPEC.loader.exec_module(fr)

GATE = {"pass": True, "point": True, "value": 0.0, "threshold": 0.05}
REPORT = {   # a check report by hand: two groups, one family that passes on its point estimate only, one that fails
    "verdict": "point", "passed": False, "passed_point": False, "passed_interval": False,
    "broken": ["BROKEN warehouse_a amr/1 zone_clear: blind scores 0.100, 0.200 under all_unknown (0.300)."],
    "asked_nothing": [{"layout": "warehouse_a", "job": "amr/1", "compiled": 7, "asked_nothing": 1},
                      {"layout": "apartment_a", "job": "helper/1", "compiled": 6, "asked_nothing": 0}],
    "groups": [{"layout": "apartment_a", "job": "helper/1", "families": {"object_where": {"unknown_share": 0.25}}},
               {"layout": "warehouse_a", "job": "amr/1", "families": {"zone_clear": {"unknown_share": 0.0}}}],
    "gates": [
        {"layout": "apartment_a", "job": "helper/1", "family": "object_where", "scripts": 6, "n": 77, "pass": False,
         "point": True, "gates": {
             "ceiling": {**GATE, "value": 1.0, "threshold": 0.95},
             "knowable": {**GATE, "value": 0.0126, "threshold": 0.02, "ci": [0.0, 0.0187]},
             "memory_needed": {**GATE, "value": 0.04},
             "shortcuts": {**GATE, "value": 0.1, "threshold": 0.1},
             "separation": {"pass": False, "point": True, "value": 0.45, "threshold": 0.4, "ci": [0.28, 0.6]},
             "time_needed": {"pass": None, "point": None, "value": None, "threshold": 0.1}}},
        {"layout": "warehouse_a", "job": "amr/1", "family": "zone_clear", "scripts": 6, "n": 12, "pass": False,
         "point": False, "gates": {
             "ceiling": {**GATE, "value": 1.0, "threshold": 0.95},
             "knowable": {"pass": None, "point": None, "value": None, "threshold": 0.02, "note": "one rule"},
             "memory_needed": GATE,
             "shortcuts": {"pass": False, "point": False, "value": 0.6667, "threshold": 0.1},
             "separation": {"pass": False, "point": False, "value": 0.3333, "threshold": 0.4, "ci": [0.0, 0.5714]},
             "time_needed": {"pass": False, "point": False, "value": 0.25, "threshold": 0.1}}}]}
GATE_ROWS = [
    ["`apartment_a`", "`helper/1`", "`object_where`", "6", "77", "0.250", "ok +1.000 vs 0.95",
     "ok +0.013 vs 0.02 [+0.000, +0.019]", "ok +0.040 vs 0.05", "ok +0.100 vs 0.1",
     "point +0.450 vs 0.4 [+0.280, +0.600]", "n/a", "point"],
    ["`warehouse_a`", "`amr/1`", "`zone_clear`", "6", "12", "0.000", "ok +1.000 vs 0.95", "n/a", "ok +0.000 vs 0.05",
     "FAIL +0.667 vs 0.1", "FAIL +0.333 vs 0.4 [+0.000, +0.571]", "FAIL +0.250 vs 0.1", "FAIL"]]


def rows_of(fragment: str) -> list[list[str]]:
    """The data rows of the first markdown table of a fragment."""
    return fr.first_table(fragment.splitlines())[1]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The folders of `eval` over `hand_dataset` (one layout, one job, two short cases of two scripts): the scripted
    `all_unknown` and `fixed_guess`, the fake agent that reports usage, and the same agent stopped by a spend cap
    after one case."""
    root = tmp_path_factory.mktemp("runs")
    hand_dataset(root)
    with pytest.MonkeyPatch.context() as patch:
        register_agent(patch)
        patch.setattr(prices, "table", lambda: PRICES)
        patch.setitem(BASELINES, "fixed_guess", Baseline(FixedGuess))
        for name, argv, code in (("floor", ["all_unknown"], 0), ("guess", ["fixed_guess"], 0),
                                 ("det", ["all_unknown", "--channel", "detections"], 0), ("agent", ["fake:usage"], 0),
                                 ("part", ["fake:usage", "--max-usd", "0.01", "--case-usd", "0.008"], 1)):
            assert cli(*argv, "--data", root, "--out", root / name) == code
    return root


@pytest.fixture(scope="module")
def checked(tmp_path_factory):
    """The report `stbench check --verdict point` wrote over six synthetic scripts."""
    root = tmp_path_factory.mktemp("checked")
    synthetic_dataset(root, 6)
    assert stbench_main(["check", "--data", str(root), "--verdict", "point", "--workers", "1",
                         "--out", str(root / "check.json")]) == 1
    return root / "check.json"


def printed(folder: Path, name: str) -> str:
    """The `mean [lo, hi]` of a line of `results.md` (`| memory | 4 | +0.750 | [+0.500, +1.000] |`)."""
    md = (folder / "results.md").read_text(encoding="utf-8")
    found = re.search(rf"^\| {name} \| \d+ \| (\S+) \| (\[[^\]]+\]) \|$", md, re.M)
    return f"{found[1]} {found[2]}"


def test_the_gate_table_is_the_check_reports_numbers():
    """Contract (FD-M6-M7 §4 item 7): a row of the gate table is a (layout, job, family) of the check report with its
    scripts, questions, unknown share (the group's, 0.25 of 77), each gate's verdict, value and bound, with the interval
    where the gate has one, and the group's verdict; a gate that does not apply reads `n/a`; the BROKEN lines, the
    readings and the cases that ask nothing follow. By hand: 0.0126 is +0.013, 0.0187 +0.019, 0.6667 +0.667. Regression
    caught: a value from another gate or row, a rounding other than `results.md`'s three places, a pass on the point
    estimate shown as a pass, or a broken instrument left out."""
    fragment = fr.gate_fragment(REPORT)
    assert rows_of(fragment) == GATE_ROWS
    assert fragment.splitlines()[0].split(" | ")[6:12] == ["ceiling", "knowable", "memory_needed", "shortcuts",
                                                            "separation", "time_needed"]
    assert "Point reading: FAIL. Interval reading: FAIL." in fragment
    assert f"- {REPORT['broken'][0]}" in fragment.splitlines()
    assert "- warehouse_a amr/1: 1 of 7 compiled cases ask nothing and are not checked" in fragment.splitlines()
    assert "helper/1: 0 of 6" not in fragment
    with pytest.raises(fr.Refused, match="not the JSON of `stbench check`"):
        fr.gate_fragment({"gates": []})


def test_the_gate_table_covers_every_row_of_a_real_check_report(checked):
    """Contract: over the report that `stbench check` itself wrote, the gate table has one row for each (layout, job,
    family) in the report's order and, in each gate's cell, the verdict word, the value to three places, the bound and
    the interval of the report's own numbers. Regression caught: a reader that drifts from the check report's shape (a
    renamed key), a family dropped, or two rows' cells swapped."""
    report = json.loads(checked.read_text(encoding="utf-8"))
    rows = rows_of(fr.gate_fragment(report))
    assert [r[:3] for r in rows] == [[f"`{g['layout']}`", f"`{g['job']}`", f"`{g['family']}`"]
                                     for g in report["gates"]] and len(rows) == 2
    for row, g in zip(rows, report["gates"], strict=True):
        assert row[3:5] == [str(g["scripts"]), str(g["n"])]
        for shown, name in zip(row[6:12], ("ceiling", "knowable", "memory_needed", "shortcuts", "separation",
                                           "time_needed"), strict=True):
            gate = g["gates"][name]
            if gate["pass"] is None:
                assert shown == "n/a"
                continue
            word = "ok" if gate["pass"] else "point" if gate["point"] else "FAIL"
            ci = f" [{gate['ci'][0]:+.3f}, {gate['ci'][1]:+.3f}]" if gate.get("ci") else ""
            assert shown == f"{word} {gate['value']:+.3f} vs {gate['threshold']}{ci}"


def test_a_results_headline_and_splits_are_as_results_md_prints_them(runs):
    """Contract (VISION rule 14): a row of the headline table is a layout and a system with the headline and its 95%
    interval, `now`, `memory` and own path exactly as the document's `results.md` prints them, the `now` and `memory`
    cells with the number of questions each rests on, and the cases scored. By hand (`all_unknown` over
    `eval_kit.hand_dataset`): the two during questions are `now` and score 0, the four end ones `memory` and score 3/4,
    so the headline, the mean of the two columns, is +0.375 (not the pooled +0.500) and the cells read "over 2" and
    "over 4"; no row is own-path, so that cell is `-`. Regression caught: a mean pooled over layouts or jobs, an
    interval from another split, a column shown without its count, or rounding that differs from `eval_report`."""
    docs = fr.load_results([str(runs / "floor")])
    (row,) = rows_of(fr.headline_fragment(docs))
    assert row == ["`apartment_a`", "`all_unknown`", printed(runs / "floor", "headline"),
                   printed(runs / "floor", "now") + " over 2", printed(runs / "floor", "memory") + " over 4", "-", "2",
                   "rgb_pose", "correct brief, during on"]
    assert [x.split(" ")[0] for x in row[2:5]] == ["+0.375", "+0.000", "+0.750"]


def test_reference_rows_carry_the_resources_as_results_md_prints_them(runs):
    """Contract (VISION rule 14): a reference row names the system, reasoning, channel and arm, the cases scored (and
    whether it is partial) and what they cost: tokens in, cached and out, notional USD and wall time, summed over the
    finished cases and `n/a` where a case reported nothing. By hand (`eval_kit`): each case of the fake agent reports
    1,500 tokens in, 200 cached, 150 out and costs $0.00339; two cases are 3,000, 400, 300 and $0.0068, the spend cap
    leaves one case at 1,500, 200, 150 and $0.0034; the scripted floor reports none. Regression caught: a sum over
    cases that never ran, an unknown counted as zero, or a partial row shown as complete."""
    docs = fr.load_results([str(runs / "floor"), str(runs / "agent"), str(runs / "part")])
    floor, agent, part = rows_of(fr.reference_fragments(docs)["reference-rows.md"])
    assert floor[2:8] == ["`all_unknown`", "-", "rgb_pose", "correct brief, during on", "2 of 2", "no"]
    assert floor[12:16] == ["n/a", "n/a", "n/a", "n/a"]
    assert agent[2:8] == ["`fake:usage`", "-", "rgb_pose", "correct brief, during on", "2 of 2", "no"]
    assert agent[12:16] == ["3,000", "400", "300", f"${2 * COST:.4f}"]
    assert part[6:8] == ["1 of 2", "yes"] and part[12:16] == ["1,500", "200", "150", f"${COST:.4f}"]
    assert re.fullmatch(r"\d+\.\d", agent[16])
    md = (runs / "agent" / "results.md").read_text(encoding="utf-8")
    assert "- tokens: in 3,000, of them read from a cache 400, written to one 0, out 300" in md.splitlines()
    assert f"- cost: ${2 * COST:.4f} (${COST:.4f} a case)" in md.splitlines()
    (script,) = [c["script"] for c in docs[2]["cases"] if c["status"] != "complete"]
    phrases = fr.phrase_fragments(docs)
    assert phrases["coverage.md"].endswith(f"`fake:usage`: 1 of 2 cases, not finished in the scripts {script}")
    assert "`fake:usage`: 2 of 2 cases;" in phrases["coverage.md"]


def test_the_splits_list_each_family_and_length_of_each_system(runs):
    """Contract (VISION rule 14): the splits table has a row for each family and each case length of each system and
    job, with the questions and the mean and interval `results.md` prints. By hand (`all_unknown` over the hand
    dataset): `object_where` 4 questions at +0.750, `path_ahead` 2 at +0.000, and the one length `short` 6 at +0.500.
    Regression caught: a family missing, the lengths in alphabetical order, or a split's interval from another split."""
    docs = fr.load_results([str(runs / "floor")])
    rows = rows_of(fr.reference_fragments(docs)["reference-splits.md"])
    assert [r[3:] for r in rows] == [
        ["family", "object_where", "4", printed(runs / "floor", "object_where")],
        ["family", "path_ahead", "2", printed(runs / "floor", "path_ahead")],
        ["length", "short", "6", printed(runs / "floor", "short")]]
    assert [r[6].split(" ")[0] for r in rows] == ["+0.750", "+0.000", "+0.500"]


def test_the_comparison_is_eval_compares_output_and_each_job_alone(runs, capsys):
    """Contract (VISION rule 13, FD-M6-M7 §4 item 12): the comparison holds the output of `stbench eval compare` for
    results one lever apart, line for line (its headings only moved down under the report's own section), then the
    same table for each job and the case lengths side by side; results two levers apart are never compared, each set
    is its own. By hand (`test_eval_compare`): the headlines are +0.375 for `all_unknown` and +0.250 for `fixed_guess`
    and the difference -0.125; with one job the job's table is the layout's. Regression caught: a comparison
    re-implemented and drifting from `eval compare`, a difference taken the wrong way round, or results two levers apart
    paired."""
    capsys.readouterr()
    assert cli("compare", runs / "floor", runs / "guess") == 0
    out = capsys.readouterr().out.splitlines()
    paths = [str(runs / "floor"), str(runs / "guess")]
    fragment = fr.comparison_fragment(paths, fr.load_results(paths))
    lines = fragment.splitlines()
    demoted = {re.sub(r"^(#+) ", r"##\1 ", line) for line in out[1:] if line}
    assert len(demoted) > 8 and demoted <= set(lines)
    headline = next(x for x in lines if x.startswith("| **headline**"))
    assert [c.split(" ")[0] for c in headline.split(" | ")[1:]] == ["+0.375", "+0.250", "-0.125"]
    assert "##### apartment_a, vacuum/1" in lines and "#### Per job" in lines
    assert lines.count(headline) == 2   # the layout's table and the job's
    assert "| length | A | B |" in lines and any(x.startswith("| short | +0.500") for x in lines)
    more = [*paths, str(runs / "det")]   # `det` differs from the others in the channel and the agent: its own set
    sets = fr.comparison_fragment(more, fr.load_results(more)).splitlines()
    assert [x for x in sets if x.startswith("### ")] == ["### detections channel, correct brief, during on",
                                                         "### rgb_pose channel, correct brief, during on"]
    assert "- A = all_unknown, detections channel, correct brief, during on, sandbox in-process (the benchmark's own " \
           "code), 2 of 2 cases finished" in sets


def test_free_rows_and_the_reference_sentence_name_each_system_and_layout(runs):
    """Contract (VISION rule 14): the scripted baselines' headlines and the reference agents' are one line each, per
    layout, as the headline table gives them; a scripted baseline is told by the package's registry, not by the folder.
    By hand: the floor scores +0.500 on the hand dataset's one layout. Regression caught: a baseline listed as a
    reference agent, or a headline missing its interval."""
    docs = fr.load_results([str(runs / "floor"), str(runs / "agent")])
    phrases = fr.phrase_fragments(docs)
    floor = printed(runs / "floor", "headline")
    assert phrases["free-baselines.md"] == f"`apartment_a`: `all_unknown` (rgb_pose) {floor}"
    assert phrases["reference-sentence.md"] == (
        f"`fake:usage` scores {printed(runs / 'agent', 'headline')} on `apartment_a`")


def test_cost_is_cmd_pilots_per_length_table_of_each_document(runs):
    """Contract: the cost fragment is the pilot report's own per case-run table (tokens, wall time and USD by length),
    one for each results document, and a line for a sentence. By hand: the fake agent's two short cases report 1,500 in,
    200 cached and 150 out each, so the mean of the short ones is 1,500, 200, 150. Regression caught: a cost table that
    recomputes means differently from `stbench pilot report`."""
    docs = fr.load_results([str(runs / "agent")])
    fragments = fr.cost_fragments(docs)
    assert "| short | 2 | 1,500 | 200 | 150 |" in fragments["cost-table.md"]
    assert fragments["cost.md"].startswith("`fake:usage`, a case: short 1,500 tokens in (200 cached), 150 out, ")


def test_tables_refuses_results_from_two_compiles_and_a_foreign_folder(runs, tmp_path, capsys):
    """Contract: results scored on different compiles (a case both hold rows of, with no qid in common: a compile draws
    every qid anew; `eval_compare.same_compile`, which `fill_report` calls and does not copy) are never put in one
    table, nor are results that share no case, and a folder that is not one `tables` made is never written into.
    Regression caught: a comparison across compiles that silently pairs nothing, or fragments dropped into (and stale
    files deleted from) someone's `docs` folder."""
    again = tmp_path / "again"
    shutil.copytree(runs, again, ignore=shutil.ignore_patterns("floor", "guess", "det", "agent", "part"))
    recompile(again)
    assert cli("all_unknown", "--data", again, "--out", again / "floor") == 0
    out = tmp_path / "out"
    code = fr.main(["tables", "--results", str(runs / "floor"), str(again / "floor"), "--out", str(out)])
    assert code == 2 and "were scored on different compiles" in capsys.readouterr().err and not out.exists()
    other = tmp_path / "other"
    hand_dataset(other)
    assert cli("all_unknown", "--data", other, "--out", other / "floor") == 0
    code = fr.main(["tables", "--results", str(runs / "floor"), str(other / "floor"), "--out", str(out)])
    assert code == 2 and "shares no case" in capsys.readouterr().err and not out.exists()
    foreign = tmp_path / "docs"
    foreign.mkdir()
    (foreign / "notes.md").write_text("mine", encoding="utf-8")
    assert fr.main(["tables", "--results", str(runs / "floor"), "--out", str(foreign)]) == 2
    assert sorted(p.name for p in foreign.iterdir()) == ["notes.md"]
    assert fr.main(["tables", "--out", str(tmp_path / "none")]) == 2


RELEASE = {
    "package": "0.4.0",
    "groups": [
        {"layout": "apartment_a", "job": "vacuum/2", "scripts": 6, "cases": {"short": 6, "medium": 6, "long": 6},
         "asked_nothing": 1, "questions": 12345, "frames": 4821},
        {"layout": "warehouse_a", "job": "amr/1", "scripts": 4, "cases": {"short": 4, "medium": 3, "long": 4},
         "asked_nothing": 0, "questions": 2000, "frames": 9000}]}


def test_the_release_counts_are_release_jsons_numbers():
    """Contract (A6): the jobs table gives each job's scripts, cases by length, questions and frames and the abstract's
    totals are their sums, all from `release.json`. By hand: 6 + 4 scripts, 18 + 11 cases, 12,345 + 2,000 questions,
    4,821 + 9,000 frames. Regression caught: a total that double-counts, or a count from another job's row."""
    fragments = fr.release_fragments(RELEASE)
    assert rows_of(fragments["jobs.md"]) == [
        ["`vacuum/2`", "`apartment_a`", "6", "6 / 6 / 6", "12,345", "4,821"],
        ["`amr/1`", "`warehouse_a`", "4", "4 / 3 / 4", "2,000", "9,000"]]
    assert [fragments[f"total-{k}.md"] for k in ("scripts", "cases", "questions", "frames")] == [
        "10", "29", "14,345", "13,821"]
    assert fragments["scripts-a-job.md"] == "6 for `vacuum/2`, 4 for `amr/1`"


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    """`stbench pilot report` over `test_cmd_pilot`'s documents: one agent's ten vacuum scripts, the leak pairs (vacuum
    fails, amr passes), the brief swap and a release plan for the vacuum; its text and the plan."""
    tmp = tmp_path_factory.mktemp("pilot")
    leaks = [list(leak_pair(tmp, "v", "vacuum", "apartment_a", VACUUM, stray=True)),
             list(leak_pair(tmp, "a", "amr", "warehouse_a", AMR))]
    plan = plan_file(tmp, render_hours=1e6, points=1e6, jobs={
        "vacuum": {"frames_per_script": 100, "seconds_per_frame": 1, "points_per_script": 1}})
    return cmd_pilot.build([save(tmp, "sol", pilot_doc(SOL, SOL_SCORES))], leaks, [list(brief_pair(tmp))], plan)


def test_the_pilot_fragments_are_the_pilot_reports_numbers(pilot):
    """Contract (FD-M6-M7 §2.2 and §2.4): the spread table, D2's scalars, the brief swap and the release count are cut
    from the pilot report's text and carry its numbers unchanged. By hand (`test_cmd_pilot`): the ten vacuum scripts
    have sigma 0.506, design effect 1.083 and rho 0.028 over short cases of 4 questions; the leak pairs pool to 8 pairs
    in 4 scripts, d +0.000 [-0.500, +0.500], inconclusive, with vacuum failing at +0.500 and amr passing at -0.500; the
    swap's critical rows differ by +0.375 over three cases (two short, one medium) and the others by +0.000; the count
    is 13 scripts needed and released at the half-width 0.099 (0.4 GPU hours, 13.0 points). Regression caught: a
    number re-derived and different from the report's, an agent's row taken for another's, or a job's rows swapped."""
    fragments = fr.pilot_fragments(pilot)
    assert rows_of(fragments["pilot-spread.md"]) == [["`vacuum`", "10", "4.0 / - / -", "0.506", "1.083", "0.028"]]
    assert [fragments[f"leak-{k}.md"] for k in ("pairs", "scripts", "d", "interval", "verdict")] == [
        "8", "4", "+0.000", "[-0.500, +0.500]", "inconclusive"]
    assert fragments["leak-per-job.md"].startswith(
        "`amr` d -0.500 [-0.500, -0.500] (pass); `vacuum` d +0.500 [+0.500, +0.500] (fail). The during questions ship;")
    assert fragments["leak-per-job.md"].endswith("a leak of the upper bound (+0.500)")
    (brief,) = rows_of(fragments["pilot-brief.md"])
    interval = r"\[[-+]\d\.\d{3}, [-+]\d\.\d{3}\]"
    assert brief[0] == "`amr`" and brief[2] == brief[4] == "-" and brief[5] == "2 / 1 / 0"
    assert re.fullmatch(rf"\+0\.375 {interval}", brief[1]) and re.fullmatch(rf"\+0\.000 {interval}", brief[3])
    assert rows_of(fragments["pilot-count.md"]) == [["`vacuum`", "13", "13", "0.099", "0.4", "13.0"]]
    assert "| vacuum | 0.506 (pilot) | 12.0 | 13 | 13 | 0.099 | 1,300 | 0.4 | 13.0 |" in pilot.splitlines()


def test_two_agents_leave_the_per_job_pilot_cells_for_a_person(tmp_path):
    """Contract: the pilot's rows are per agent, a skeleton's per job; with two agents on a job the spread table keeps
    an Agent column, so `fill` finds two rows for the job and leaves the cell, rather than choosing an agent. By hand:
    two agents on the vacuum job make two rows. Regression caught: one agent's sigma filled in silently for both."""
    other = pilot_doc(("codex:gpt-6-luna", "xhigh"), SOL_SCORES)
    text = cmd_pilot.build([save(tmp_path, "sol", pilot_doc(SOL, SOL_SCORES)), save(tmp_path, "luna", other)], [], [],
                           None)
    spread = fr.pilot_fragments(text)["pilot-spread.md"]
    assert [r[:2] for r in rows_of(spread)] == [["codex:gpt-6-luna xhigh", "`vacuum`"],
                                                ["codex:gpt-6.1-sol medium", "`vacuum`"]]
    (tmp_path / "frags").mkdir()
    (tmp_path / "frags" / "pilot-spread.md").write_text(spread, encoding="utf-8")
    skeleton = ("| Job | Scripts | Score spread |\n| --- | --- | --- |\n"
                "| `vacuum` | <!-- FILL: pilot scripts --> | <!-- FILL: sigma --> |\n")
    new, left = fr.fill_text(skeleton, tmp_path / "frags")
    assert new == skeleton and [(n, w[:20]) for _, n, w in left] == [("pilot scripts", "2 rows of pilot-spre"),
                                                                    ("sigma", "2 rows of pilot-spre")]


DOC = """\
# Title é
Dated <!-- FILL: release date --> with <!-- FILL: total scripts --> scripts.
Example `<!-- FILL: what -->` and <!-- counts --> stay.
Over <!-- FILL: the cases each row covers; a partial row
names its missing scripts --> done.
<!-- FILL: the gate table per layout, job and family, pasted from `stbench check` output -->
In a sentence (<!-- FILL: the gate table per layout, job and family, pasted from `stbench check` output -->) here.
New <!-- FILL: some new thing --> and <!-- FILL: leak d -->.

| Layout | System | Headline (95% interval) | `now` | Robot |
| --- | --- | --- | --- | --- |
| `apartment_a` | `all_unknown` | <!-- FILL: headline --> | <!-- FILL: now --> | a robot |
| `apartment_a` | `x` (high) | <!-- FILL: headline --> | <!-- FILL: now --> | a robot |
| `warehouse_a` | `all_unknown` | <!-- FILL: headline --> | <!-- FILL: now --> | a robot |
"""
FILLED = """\
# Title é
Dated <!-- FILL: release date --> with 7 scripts.
Example `<!-- FILL: what -->` and <!-- counts --> stay.
Over `x`: 2 of 2 cases done.
| A | B |
| --- | --- |
| 1 | 2 |
In a sentence (<!-- FILL: the gate table per layout, job and family, pasted from `stbench check` output -->) here.
New <!-- FILL: some new thing --> and <!-- FILL: leak d -->.

| Layout | System | Headline (95% interval) | `now` | Robot |
| --- | --- | --- | --- | --- |
| `apartment_a` | `all_unknown` | +0.500 [+0.333, +0.667] | - | a robot |
| `apartment_a` | `x` (high) | <!-- FILL: headline --> | <!-- FILL: now --> | a robot |
| `warehouse_a` | `all_unknown` | <!-- FILL: headline --> | <!-- FILL: now --> | a robot |
"""
FRAGMENTS = {
    "total-scripts.md": "7\n", "coverage.md": "`x`: 2 of 2 cases\n",
    "gates.md": "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
    "headline.md": ("| Layout | System | Headline (95% interval) | `now` | Cases scored |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| `apartment_a` | `all_unknown` | +0.500 [+0.333, +0.667] | - | 2 |\n"
                    "| `apartment_a` | `x` (high) | +0.1 [0, 1] | - | 2 |\n"
                    "| `apartment_a` | `x` (high) | +0.2 [0, 1] | - | 2 |\n")}


def test_fill_replaces_only_what_a_fragment_fills_and_lists_the_rest(tmp_path):
    """Contract: `fill` replaces a placeholder only where the mapped fragment is there (a phrase for a phrase, a block
    for a placeholder alone on its line, a table cell for the one row that agrees with the row's other cells, a column
    the fragment lacks ignored) and lists every other with its line and why: a person's, an unmapped one, a fragment
    never written, a block inside a sentence, an ambiguous row and a row the fragment lacks. A placeholder split over
    two lines is one placeholder, and a code span's example and other comments are not placeholders. Regression caught:
    a placeholder filled with the wrong fragment, a half-filled sentence, a silent skip, or a table cell filled from
    a row that only partly agrees."""
    frags = tmp_path / "frags"
    frags.mkdir()
    for name, text in FRAGMENTS.items():
        (frags / name).write_text(text, encoding="utf-8")
    new, left = fr.fill_text(DOC, frags)
    assert new == FILLED
    gate = "the gate table per layout, job and family, pasted from `stbench check` output"
    assert [(line, name) for line, name, _ in left] == [
        (2, "release date"), (7, gate), (8, "some new thing"), (8, "leak d"), (13, "headline"), (13, "now"),
        (14, "headline"), (14, "now")]
    why = {(line, name): text for line, name, text in left}
    assert why[2, "release date"] == "the date is the maintainers'"
    assert "gates.md is a block and the placeholder is inside a sentence" in why[7, gate]
    assert why[8, "some new thing"] == "no fragment is mapped to this placeholder"
    assert why[8, "leak d"].startswith("leak-d.md was not written")
    assert why[13, "now"].startswith("2 rows of headline.md agree with {'Layout': '`apartment_a`'")
    assert why[14, "now"].startswith("0 rows of headline.md agree with {'Layout': '`warehouse_a`'")


def test_fill_prints_a_diff_and_writes_only_when_asked(tmp_path, capsys):
    """Contract: without `--write` the document is not touched and the diff is printed with the unfilled count; with
    `--write` the document becomes the filled text (UTF-8, the lines it did not fill as they were) and a second fill
    changes nothing, the placeholders being gone. Regression caught: a dry run that writes, or a write that changes
    bytes of lines it did not fill."""
    frags = tmp_path / "frags"
    frags.mkdir()
    for name, text in FRAGMENTS.items():
        (frags / name).write_text(text, encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_bytes(DOC.encode("utf-8"))
    assert fr.main(["fill", "--doc", str(doc), "--fragments", str(frags)]) == 0
    out = capsys.readouterr().out
    assert doc.read_bytes() == DOC.encode("utf-8")
    assert "-Over <!-- FILL: the cases" in out and "+Over `x`: 2 of 2" in out and "8 placeholders left unfilled" in out
    assert fr.main(["fill", "--doc", str(doc), "--fragments", str(frags), "--write"]) == 0
    assert doc.read_bytes() == FILLED.encode("utf-8")
    assert fr.main(["fill", "--doc", str(doc), "--fragments", str(frags), "--write"]) == 0
    assert "unchanged" in capsys.readouterr().out and doc.read_bytes() == FILLED.encode("utf-8")


def test_the_report_and_the_card_placeholders_are_all_accounted_for(runs, checked, pilot, tmp_path):
    """Contract (FD-M6-M7 §4 items 7 and 12): every placeholder of `docs/report.md`, the README, the CHANGELOG and the
    dataset card is filled by a table cell the fragments cover, mapped to a fragment the tables produce, or listed as a
    person's with its reason, and the fragments the mapping names are all produced by the inputs it documents.
    Regression caught: a placeholder renamed in the document so that no fragment fills it, or a mapped fragment
    `tables` no longer writes."""
    (tmp_path / "pilot.md").write_text(pilot, encoding="utf-8")
    (tmp_path / "release.json").write_text(json.dumps(RELEASE), encoding="utf-8")
    argv = ["tables", "--check", str(checked), "--results", str(runs / "floor"), str(runs / "agent"),
            "--pilot", str(tmp_path / "pilot.md"), "--release", str(tmp_path / "release.json")]
    assert fr.main([*argv, "--out", str(tmp_path / "frags")]) == 0
    made = {p.name for p in (tmp_path / "frags").iterdir()}
    assert {*fr.BLOCKS.values(), *fr.CELL_FRAGMENTS} <= made
    for doc in (ROOT / "docs" / "report.md", ROOT / "README.md", ROOT / "CHANGELOG.md",
                ROOT / "src" / "stbench" / "release" / "dataset-card.md"):
        _, left = fr.fill_text(doc.read_text(encoding="utf-8"), tmp_path / "frags")
        reasons = {why for _, _, why in left}
        assert not [w for w in reasons if "no fragment is mapped" in w or "no fragment table has" in w
                    or "was not written" in w or "inside a sentence" in w], (doc.name, reasons)


def test_tables_twice_give_identical_bytes_and_a_rerun_replaces_its_own_files(runs, checked, pilot, tmp_path):
    """Contract (the bootstrap seeds are fixed): the same inputs give the same bytes, in a fresh folder or over a
    previous run's, and a rerun with fewer inputs removes the fragments of the earlier one (so `fill` cannot paste a
    stale table). Regression caught: a fragment that depends on the order of a dict, the time or the run, or a stale
    fragment left behind."""
    (tmp_path / "pilot.md").write_text(pilot, encoding="utf-8")
    argv = ["tables", "--check", str(checked), "--results", str(runs / "floor"), str(runs / "guess"),
            "--pilot", str(tmp_path / "pilot.md")]
    first, second = tmp_path / "one", tmp_path / "two"
    assert fr.main([*argv, "--out", str(first)]) == 0 and fr.main([*argv, "--out", str(second)]) == 0
    before = {p.name: p.read_bytes() for p in sorted(first.iterdir())}
    assert before == {p.name: p.read_bytes() for p in sorted(second.iterdir())} and len(before) > 10
    assert fr.main([*argv, "--out", str(first)]) == 0
    assert before == {p.name: p.read_bytes() for p in sorted(first.iterdir())}
    assert fr.main(["tables", "--check", str(checked), "--out", str(first)]) == 0
    assert sorted(p.name for p in first.iterdir()) == ["fragments.txt", "gates.md"]
