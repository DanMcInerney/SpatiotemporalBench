"""`--compact-at TOKENS` for the Claude Code agent (`claude_exec.compaction_env`, `claude_exec.Claude`, `cmd_run`,
`cmd_claude`, `eval_results`): Claude Code has no flag for it, so a run with a limit sets the two variables that move
Claude Code's own auto-compaction (`CLAUDE_CODE_AUTO_COMPACT_WINDOW` and `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`, measured on
`claude -p`) in the one process's environment, and says so in every record it leaves. The Codex agent's is
`test_compact_at.py`. The agent is the fake command (`claude_cli_fake`) behind the real driver, which records the
environment it was started with. No model, no `claude`, no network. Expected values are written by hand.
"""

from __future__ import annotations

import json

import pytest

from claude_kit import Fake, claude_run
from codex_kit import job_also_asks
from eval_kit import cli as eval_cli
from eval_kit import compiled, results, short_script
from stbench import cmd_claude
from stbench.claude_exec import Claude, compaction_env, compaction_problem
from stream_kit import end_q
from test_cmd_codex import jsonl, small_case

WINDOW, PCT = "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"


@pytest.fixture(autouse=True)
def the_job_asks_what_the_case_asks(monkeypatch):
    """`small_case` holds a `number` during question, which `vacuum/1` asks none of: see `test_cmd_claude.py`."""
    job_also_asks(monkeypatch, "distance_between")


@pytest.mark.parametrize(("tokens", "percent"), [(1, "1"), (800, "1"), (801, "2"), (24_000, "30"), (25_000, "32"),
                                                  (50_000, "63"), (66_400, "83")])
def test_a_limit_is_a_percentage_of_a_window_rounded_up(tokens, percent):
    """Contract (`claude_exec.compaction_env`): Claude Code compacts at (window - 20,000) x percent / 100 tokens, as
    measured on `claude -p` (a window of 100,000 at 30 percent compacted at about 25,000), so a limit of N tokens is a
    window of 100,000 and the smallest percentage that reaches N (a multiple of 800 tokens, rounded up, never below 1
    percent), both as text. Regression caught: a limit rounded down (compaction earlier than asked, with the run named
    for the asked one), or a percentage of the wrong window."""
    assert compaction_env(tokens) == {WINDOW: "100000", PCT: percent}
    assert compaction_problem(tokens) is None


@pytest.mark.parametrize("bad", [0, -5, True, 1.5, "50000", 66_401, 200_000])
def test_a_limit_that_cannot_be_given_is_refused_before_a_process_is_made(bad):
    """Contract: a count that is no whole number of tokens (0, a negative, a bool, a float, text) and one past what a
    percentage of the window reaches (84 percent and over lies beyond the window less Claude Code's 33,000 buffer:
    66,400 tokens is the last) is a `ValueError` of the variables and of the transport; the problem is also found
    without making a transport (`compaction_problem`), which is how `classify` refuses it for the command line.
    Regression caught: a limit passed to Claude Code that it ignores (a run labelled with a compaction that never
    happened)."""
    with pytest.raises(ValueError, match=r"compact_at must be a whole number of tokens|beyond what Claude Code"):
        Claude("sonnet", command=["claude"], compact_at=bad)
    with pytest.raises(ValueError, match=r"compact_at must be a whole number of tokens|beyond what Claude Code"):
        compaction_env(bad)
    if isinstance(bad, int) and not isinstance(bad, bool) and bad > 0:
        assert "66400 tokens" in compaction_problem(bad)


def test_a_run_with_a_limit_sets_the_two_variables_and_nothing_else_changes(tmp_path, monkeypatch):
    """Contract (`claude_exec.Claude.popen`, `cmd_claude`): `stbench run claude:<model> CASE --compact-at N` starts the
    one process with the window and the percentage in its environment and with the user's own `DISABLE_AUTO_COMPACT`
    taken out (a switch that would win over the limit); its argv, messages and every other variable are the reference
    run's; the journal's `run` event and `claude.summary.json` record `compact_at`, a run without the option records
    null there and leaves the user's environment as it is (the switch reaches the process, no variable is added).
    Regression caught: a limit parsed and never reaching the process (a run labelled with a limit it did not have), the
    user's switch silently winning, or a reference run that has the variables set."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, env={"DISABLE_AUTO_COMPACT": "1"})
    capped, plain = tmp_path / "capped", tmp_path / "plain"
    assert claude_run(monkeypatch, fake, "claude:sonnet", case.private, "--compact-at", 50000, "--out", capped) == 0
    assert claude_run(monkeypatch, fake, "claude:sonnet", case.private, "--out", plain) == 0
    ours, theirs = fake.processes()
    assert ours["compaction"] == {WINDOW: "100000", PCT: "63", "DISABLE_AUTO_COMPACT": None}
    assert theirs["compaction"] == {WINDOW: None, PCT: None, "DISABLE_AUTO_COMPACT": "1"}
    assert (ours["env"], ours["messages"], ours["system"]) == (theirs["env"], theirs["messages"], theirs["system"])

    def argv(record):   # the temp file of the mcp config differs between two processes
        found = record["argv"]
        return [("<file>" if i and found[i - 1] == "--mcp-config" else a) for i, a in enumerate(found)]

    assert argv(ours) == argv(theirs)
    for out, wanted in ((capped, 50000), (plain, None)):
        summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
        assert "compact_at" in summary and summary["compact_at"] == wanted
        assert jsonl(out / "claude.calls.jsonl")[0]["compact_at"] == wanted


def test_a_compaction_the_cli_reports_is_a_warning_of_its_turn(tmp_path, monkeypatch):
    """Contract (`claude_exec.digest`): a `compact_boundary` system event, which Claude Code prints when it compacts
    (its `compact_metadata` holds the trigger and the tokens before), is a warning of the turn, `compaction (auto, N
    tokens before)`, counted in the summary, so a run with a limit shows whether the limit took. It does not fail the
    turn. Regression caught: a compaction that leaves no trace in the record (a memory arm that cannot be told from one
    whose context never filled)."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"compact": 25111}})
    out = tmp_path / "o"
    assert claude_run(monkeypatch, fake, "claude:sonnet", case.private, "--compact-at", 25000, "--out", out) == 0
    (record,) = [r for r in jsonl(out / "claude.calls.jsonl") if "event" not in r]
    assert record["warnings"] == ["compaction (auto, 25111 tokens before)"]
    summary = json.loads((out / "claude.summary.json").read_text(encoding="utf-8"))
    assert (summary["status"], summary["warnings"], summary["compact_at"]) == ("complete", 1, 25000)


def test_a_limit_beyond_the_window_is_refused_with_exit_2_and_nothing_is_made(tmp_path, monkeypatch, capsys):
    """Contract (`cmd_run.classify`): `--compact-at 70000` for `claude:` is refused with exit 2 in `run` and in `eval`
    and says what Claude Code can do, before anything is sent: no output folder, no process; a small limit passes.
    Regression caught: a limit that fails in the middle of a paid run, or one that Claude Code never honours."""
    case = small_case(tmp_path / "data")
    fake, out = Fake(tmp_path), tmp_path / "out"
    assert claude_run(monkeypatch, fake, "claude:sonnet", case.private, "--compact-at", 70000, "--out", out) == 2
    assert "beyond what Claude Code can be made to compact at: 1 to 66400 tokens" in capsys.readouterr().err
    assert not out.exists() and fake.processes() == []
    monkeypatch.setattr(cmd_claude, "make_claude", lambda *a, **k: pytest.fail("a transport was made"))
    assert eval_cli("claude:sonnet", "--data", tmp_path / "data", "--out", out, "--compact-at", 70000) == 2
    assert "beyond what Claude Code can be made to compact at" in capsys.readouterr().err and not out.exists()


def test_eval_names_the_limit_for_claude_and_a_rerun_with_another_is_refused(tmp_path, monkeypatch, capsys):
    """Contract (`eval_results`, `eval_report`): `results.json` of `eval claude:sonnet --compact-at N` holds
    `"compact_at": N` and `results.md` names the system `claude:sonnet, compacts at 50k tokens`; a reference row has no
    `compact_at` key; a rerun into the folder with another limit is refused (exit 2) saying which. Regression caught:
    a Claude row with a limit whose results read as the reference's, or a rerun that mixes two limits' cases."""
    data = tmp_path / "data"
    compiled(data, short_script(1), [end_q(1)])
    fake = Fake(tmp_path)
    monkeypatch.setattr(cmd_claude, "make_claude", lambda model, reasoning, **given: fake.new(
        model=model, reasoning=reasoning, **given))
    capped, plain = tmp_path / "capped", tmp_path / "plain"
    assert eval_cli("claude:sonnet", "--data", data, "--out", capped, "--compact-at", 50000) == 0
    assert results(capped)["compact_at"] == 50000
    assert "# SpatiotemporalBench results: claude:sonnet, compacts at 50k tokens" in (capped / "results.md").read_text(
        encoding="utf-8")
    assert eval_cli("claude:sonnet", "--data", data, "--out", plain) == 0
    assert "compact_at" not in results(plain)
    capsys.readouterr()
    assert eval_cli("claude:sonnet", "--data", data, "--out", capped, "--compact-at", 60000) == 2
    assert "compact_at: there 50000, now 60000" in capsys.readouterr().err
