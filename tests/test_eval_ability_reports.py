"""`results.md` and `eval compare` for scoring by ability (0.6.0): a line per ability with its count, the columns, the
secondary means, the diagnostics named, and a compare whose headline difference is the difference of the headlines."""

from __future__ import annotations

import json

import pytest

from ability_kit import results_doc, scored
from eval_kit import cli, fake_agent, hand_dataset
from stbench import cmd_run, eval_results
from stbench.cli import main
from stbench.eval_compare import Refused, compare
from stbench.eval_report import render

THIN = frozenset({("amr/1", "change", "memory")})
TEXT_ONLY: list = []   # `args.text_only` as each run of `flag_agent` was handed it


def flag_agent(model, case, args):
    """A runner as `cmd_run.AGENTS` names one: notes the `text_only` it is handed, then runs `eval_kit.fake_agent`."""
    TEXT_ONLY.append(args.text_only)
    return fake_agent(model, case, args)


def mixed() -> list[dict]:
    """`ego_now` [1, 1] (distance); `mover_next` [-1, -1] (prediction, a diagnostic); `object_where` [1, 0, 0]
    (memory); `object_was` [1, 1] on the helper and [-1] on the AMR (change, thin on the AMR)."""
    return [*scored("ego_now", [1, 1]), *scored("mover_next", [-1, -1]), *scored("object_where", [1, 0, 0]),
            *scored("object_was", [1, 1]), *scored("object_was", [-1], job="amr/1")]


def worse() -> list[dict]:
    """The same ten questions scored [0, 1], [-1, -1], [1, 1, 0], [1, 0] and [-1]."""
    return [*scored("ego_now", [0, 1]), *scored("mover_next", [-1, -1]), *scored("object_where", [1, 1, 0]),
            *scored("object_was", [1, 0]), *scored("object_was", [-1], job="amr/1")]


def test_the_report_names_the_columns_the_means_each_ability_and_the_diagnostics():
    """Contract (docs/report.md): a layout block shows the headline over the counted questions, the `now` and `memory`
    columns, the cells' equal mean and the pooled mean, a table with a line a cell (an ability in a column): its column,
    questions, counted questions, weight in the headline, mean over all of them, mean over the counted ones; the cells
    the means count; and the cells that are diagnostics. By hand (`mixed`, the AMR's change thin): 7 counted questions;
    distance 1 over 2, memory 1/3 over 3, change 1/3 over 3 asked and 1 over the 2 counted (the helper's), prediction
    -1 over 2 and counted in none; the columns are now 1 and memory (1/3 + 1)/2 = 2/3, the headline 5/6 = 0.833, the
    equal mean of the three cells 7/9 = 0.778 and the pooled 5/7 = 0.714. The weights: `now` holds one cell, which
    carries its column's half, 0.500; `memory` holds two, a quarter each. Regression caught: a diagnostic ability left
    out of the table or counted in a mean, the counted questions shown as all, the thin cell unnamed, a weight that is
    not the cell's share of the headline."""
    lines = render(results_doc(mixed(), thin=THIN)).splitlines()
    for expected in ("| headline | 7 | +0.833 | n/a |", "| now | 2 | +1.000 | n/a |", "| memory | 5 | +0.667 | n/a |",
                     "| cells mean (the counted cells by their weights) | 7 | +0.778 | n/a |",
                     "| pooled (the counted questions pooled) | 7 | +0.714 | n/a |",
                     "| ability | column | n | counted | weight | mean | counted mean | 95% CI |",
                     "| distance | now | 2 | 2 | 0.500 | +1.000 | +1.000 | n/a |",
                     "| memory | memory | 3 | 3 | 0.250 | +0.333 | +0.333 | n/a |",
                     "| change | memory | 3 | 2 | 0.250 | +0.333 | +1.000 | n/a |",
                     "| prediction (diagnostic: counted in no mean) | now | 2 | 0 | - | -1.000 | - | n/a |",
                     "The means count these cells, an ability in a column: now: distance; memory: change, memory.",
                     "Under 20 questions in the dataset, so counted at part weight in the means (0.00 counts in "
                     "none): amr/1 change (memory) 0.00."):
        assert expected in lines, expected


def test_a_dataset_too_small_for_the_rule_says_so_and_a_run_of_diagnostics_still_renders():
    """Contract: when the dataset is too small for the under-20 rule (`thin_ignored`), `results.md` says the rule
    was not
    applied and the headline is not the release statistic, rather than naming every pair a diagnostic. A layout whose
    every row is a diagnostic has no headline and no unknown share to subtract: the report prints `n/a` for them and
    does not fail. Regression caught: a development run whose report is silent about a headline the release would not
    give, a false sentence about "no ability" when the dataset decided, or a `None - None` crash."""
    rows = [*scored("ego_now", [1]), *scored("object_where", [0])]
    text = render(results_doc(rows, thin_ignored=True))
    assert ("Every ability is under 20 questions on its job in this dataset, so the under-20 rule was not applied and "
            "this headline is not the release statistic.") in text
    assert "Under 20 questions in the dataset" not in text
    thin = frozenset({("helper/1", "distance", "now"), ("helper/1", "memory", "memory")})
    lines = render(results_doc(rows, thin=thin)).splitlines()
    assert "| headline | 0 | n/a | n/a |" in lines
    assert "| above the floor (the headline less the unknown share, n/a) | 0 | n/a | n/a |" in lines
    assert "| distance (diagnostic: counted in no mean) | now | 1 | 0 | - | +1.000 | - | n/a |" in lines


def test_compare_gives_the_headline_difference_a_line_per_ability_and_flags_the_diagnostics(tmp_path):
    """Contract (docs/eval.md): `eval compare` shows the headline, the two columns, a line per ability with its
    questions in the base (those counted if fewer, a diagnostic flagged) and a line per family, each with the later
    result's paired difference.
    The headline's difference is the difference of the two headlines over the shared questions, with the same
    diagnostics left out. By hand (`mixed` is A, `worse` is B; the AMR's change thin): A's headline is 5/6, B's `now`
    is 1/2 and `memory` (2/3 + 1/2)/2 = 7/12, so 13/24, and B - A is -7/24 = -0.292 over 10 pairs; the `now` column
    differs by 1/2 - 1 = -0.5; the ability `change`, over its 2 counted questions, by (0 - 1)/2 = -0.5. Regression
    caught: a
    difference of pooled columns, a diagnostic or thin ability inside the headline's difference, or an ability line
    without its count."""
    paths = []
    for name, rows in (("a", mixed()), ("b", worse())):
        eval_results.write(tmp_path / name, results_doc(rows, thin=THIN, system=name))
        paths.append(str(tmp_path / name))
    lines = compare(paths).splitlines()

    def line(prefix: str) -> str:
        (found,) = [x for x in lines if x.startswith(prefix)]
        return found
    head = line("| **headline**")
    assert "+0.833" in head and "+0.542" in head and "-0.292 n/a over 10" in head
    assert "-0.500 n/a over 4" in line("| `now`")
    assert line("| change (n 3, counted 2) |").endswith("-0.500 n/a over 2 |")
    assert "+1.000" in line("| change (n 3, counted 2) |") and "+0.500" in line("| change (n 3, counted 2) |")
    line("| prediction (n 2, diagnostic) |")
    line("| distance (n 2) |")
    line("| object_was |")


def test_the_text_only_flag_reaches_a_codex_agent_and_is_refused_for_anything_else(tmp_path, monkeypatch, capsys):
    """Contract (`--text-only`): the option is the run's, so it reaches the Codex runner in its namespace, is stamped in
    `results.json` (and so in `RESUME_KEYS`: a rerun without it is refused) and is refused for a baseline and for an
    agent that is neither Codex nor Claude Code, which have no image to withhold. Regression caught: a flag parsed and
    dropped before the agent, a text-only arm resumed as an image one, or a flag that a system without images silently
    accepts."""
    monkeypatch.setitem(cmd_run.AGENTS, "codex", "test_eval_ability_reports.flag_agent")
    monkeypatch.setitem(cmd_run.AGENTS, "other", "test_eval_ability_reports.flag_agent")
    TEXT_ONLY.clear()
    (case, _) = hand_dataset(tmp_path)
    out = tmp_path / "text"
    assert cli("codex:ok", "--data", tmp_path, "--out", out, "--text-only") == 0
    assert eval_results.load(out)["text_only"] is True and "text_only" in eval_results.RESUME_KEYS
    assert TEXT_ONLY and all(seen is True for seen in TEXT_ONLY)
    assert cli("codex:ok", "--data", tmp_path, "--out", out) == 2   # the folder holds a text-only run
    assert "text_only" in capsys.readouterr().err
    for system in ("all_unknown", "other:ok"):
        assert cli(system, "--data", tmp_path, "--out", tmp_path / "b", "--text-only") == 2
        assert "--text-only applies to claude or codex agents" in capsys.readouterr().err
    assert main(["run", "all_unknown", str(case.private), "--text-only"]) == 2
    assert "--text-only applies to claude or codex agents" in capsys.readouterr().err
    assert {"claude", "codex"} == cmd_run.TEXT_ONLY_AGENTS and "other" not in cmd_run.TEXT_ONLY_AGENTS
    assert cli("codex:ok", "--data", tmp_path, "--out", tmp_path / "d", "--channel", "detections", "--text-only") == 2
    assert "withholds the image" in capsys.readouterr().err and not (tmp_path / "d").exists()


def test_a_text_only_run_has_a_folder_a_summary_and_rows_of_its_own(tmp_path, monkeypatch):
    """Contract: a text-only run is never in a camera run's place or unmarked. By default `eval` writes
    `runs/<system>-<channel>-text-only` and `run` `runs/<system>-<case>-text-only`, beside the camera folders of the
    same names without the suffix; the run's `summary.json` says `text_only`, its rows carry `"text_only": true` and a
    camera run's rows gain no key; the results header and `results.md` name the option. Regression caught: a text-only
    result written over, or read as, the camera one."""
    monkeypatch.setitem(cmd_run.AGENTS, "codex", "test_eval_ability_reports.flag_agent")
    cases = hand_dataset(tmp_path)
    assert cli("codex:ok", "--data", tmp_path, "--text-only") == 0
    assert cli("codex:ok", "--data", tmp_path) == 0
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["codex_ok-rgb_pose", "codex_ok-rgb_pose-text-only"]
    text, camera = tmp_path / "runs" / "codex_ok-rgb_pose-text-only", tmp_path / "runs" / "codex_ok-rgb_pose"
    assert "text only (no image)" in (text / "results.md").read_text(encoding="utf-8")
    assert "text only" not in (camera / "results.md").read_text(encoding="utf-8")
    assert cli("codex:ok", "--data", tmp_path, "--out", tmp_path / "x", "--text-only") == 0
    case = cases[0]
    assert main(["run", "codex:ok", str(case.private), "--text-only"]) == 0
    assert main(["run", "codex:ok", str(case.private)]) == 0
    for folder, flag in ((tmp_path / "runs" / f"codex_ok-{case.id}-text-only", True), (
            tmp_path / "runs" / f"codex_ok-{case.id}", False)):
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(x) for x in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        assert summary["text_only"] is flag and rows and all(("text_only" in row) is flag for row in rows)


def test_compare_takes_text_only_as_a_lever_of_its_own(tmp_path):
    """Contract (VISION rule 13): two results that differ in `text_only` alone compare (one axis); one that differs in
    that and the channel is refused. Regression caught: a text-only arm compared with a camera arm at another channel,
    a difference that is the effect of two things."""
    rows = mixed()
    for name, over in (("a", {}), ("b", {"text_only": True}), ("c", {"text_only": True, "channel": "detections"})):
        eval_results.write(tmp_path / name, results_doc(rows, thin=THIN, system="s", **over))
    text = compare([str(tmp_path / "a"), str(tmp_path / "b")])
    assert "B = s, rgb_pose channel, correct brief, during on, text only (no image)" in text
    with pytest.raises(Refused, match="more than one lever"):
        compare([str(tmp_path / "a"), str(tmp_path / "c")])


def clearance_rows(clear: list[float], person: list[float]) -> list[dict]:
    """`ego_now` [1, 0] (distance, `now`); `clear_path` `clear` (clearance, `now`); `closest_person` `person`
    (clearance, `memory`); `object_where` [1, 1] (memory)."""
    return [*scored("ego_now", [1, 0]), *scored("clear_path", clear, scope="now"),
            *scored("closest_person", person, scope="memory"), *scored("object_where", [1, 1])]


def test_clearance_prints_as_two_rows_one_for_each_column_with_its_weight():
    """Contract (FD-task-questions review §5): the per-ability table has a row for each (ability, column) cell with its
    questions, counted questions and weight in the headline, and `results.md` names the cells the means count. By hand
    (`clearance_rows([1, 1], [1, 0, 0])`): `now` holds distance 1/2 and clearance 1, `memory` holds memory 1 and
    clearance 1/3, four cells of weight 1/4; the columns are 3/4 and 2/3, the headline 17/24 = 0.708. Regression
    caught: clearance one row under a single column, so a reader sees the corridor questions moved into `memory`, or a
    weight that is not the cell's share."""
    lines = render(results_doc(clearance_rows([1, 1], [1, 0, 0]))).splitlines()
    for expected in ("| headline | 9 | +0.708 | n/a |", "| now | 4 | +0.750 | n/a |", "| memory | 5 | +0.667 | n/a |",
                     "| clearance | now | 2 | 2 | 0.250 | +1.000 | +1.000 | n/a |",
                     "| clearance | memory | 3 | 3 | 0.250 | +0.333 | +0.333 | n/a |",
                     "The means count these cells, an ability in a column: now: clearance, distance; "
                     "memory: clearance, memory."):
        assert expected in lines, expected


def test_compare_gives_a_line_to_each_cell_of_an_ability_asked_in_both_columns(tmp_path):
    """Contract (docs/eval.md): `eval compare` names an ability's column where it has two cells and pairs each cell's
    questions apart. By hand (A scores clearance [1, 1] and [1, 0, 0], B [0, 1] and [1, 1, 0]): the `now` cell differs
    by (0 + 1)/2 - 1 = -0.5 over 2 pairs, the `memory` cell by 2/3 - 1/3 = +0.333 over 3. Regression caught: the two
    cells pooled into one line, where a difference on the corridor questions is lost among the person ones."""
    paths = []
    for name, rows in (("a", clearance_rows([1, 1], [1, 0, 0])), ("b", clearance_rows([0, 1], [1, 1, 0]))):
        eval_results.write(tmp_path / name, results_doc(rows, system=name))
        paths.append(str(tmp_path / name))
    lines = compare(paths).splitlines()

    def line(prefix: str) -> str:
        (found,) = [x for x in lines if x.startswith(prefix)]
        return found
    assert line("| clearance, now (n 2) |").endswith("-0.500 n/a over 2 |")
    assert line("| clearance, memory (n 3) |").endswith("+0.333 n/a over 3 |")
    line("| distance (n 2) |")
