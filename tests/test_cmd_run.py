"""`stbench run` (`cmd_run.py`, plan §4.4): one system over one case through the stream protocol. A baseline and a
local system over the shim run a synthetic dataset end to end, offline; the registry of agents, the refusals and the
exit codes are tested with fake agents and baselines. A Docker run is in test_command_system.py."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from run_kit import scored_case, scored_scripted
from stbench import baselines, cli, cmd_run
from stbench.stream_protocol import System
from stream_kit import (
    AGENT_CALLS,
    QUESTIONS,
    end_q,
    hand20,
    qid,
)

HEX16 = re.compile(r"[0-9a-f]{16}")


def run(*argv) -> int:
    return cli.main(["run", *map(str, argv)])


def three_cases(root: Path):
    """Three scripts of the 20-step hand case (seeds 1 to 3), a case each, with its three questions."""
    cases = []
    for seed in (1, 2, 3):
        synth = hand20()
        synth.seed = seed
        cases.append(scored_case(root, synth, QUESTIONS))
    return cases


def test_a_baseline_runs_every_case_of_a_dataset_and_writes_its_answers_and_summary(tmp_path, capsys):
    """Contract (plan §4.4, B.1's exit check): `stbench run all_unknown CASE_DIR` on each of a synthetic dataset's three
    cases exits 0 and writes, under `runs/all_unknown-<case id>/` of the current directory, `answers.jsonl` (one
    `{"qid", "answer"}` row a question, in the order of the case's file) and `summary.json` (the case, its script, the
    system, the channel, the brief, the during flag, the status, no failures, per-message wall times, the state's sizes,
    usage stamped self-reported, whether it names a remote store, what the tokens cost and why not (a scripted baseline
    reports no tokens, so no price), a 16-hex session and where it ran: no version, schema or price stamp).
    Hand-written expectations. Regression caught: a smoke run that cannot be run or read back, or a summary
    that drops what a result is stamped with (the four resource keys `stbench eval` reads M5's way)."""
    cases = three_cases(tmp_path)
    assert len({c.script.id for c in cases}) == 3
    for case in cases:
        assert run("all_unknown", case.private) == 0
        out = tmp_path / "runs" / f"all_unknown-{case.id}"
        rows = [json.loads(line) for line in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows == [{"qid": qid(n), "answer": {"type": "unknown"}} for n in (1, 2, 3)]
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert HEX16.fullmatch(summary.pop("session")) and summary["usage"].pop("self_reported") is True
        assert set(summary.pop("wall_s")) == {"start", "task", "observe", "ask_during", "ask_end", "finish"}
        summary.pop("scores")  # test_cmd_run_score.py holds what a run is scored as
        assert summary == {
            "case": case.id, "script": case.script.id, "system": "all_unknown", "channel": "rgb_pose",
            "brief": "correct", "during": True, "status": "complete", "failures": {}, "sandbox": "in-process",
            "state_bytes": {"0": 0, "finish": 0},
            "usage": {"models": None, "peak_context": None, "remote_store_bytes": None}, "remote_memory": False,
            "cost_usd": None, "cost_note": "no token counts reported", "text_only": False}
    assert "complete: 3 answers" in capsys.readouterr().out


def test_options_reach_the_run_and_are_stamped(tmp_path):
    """Contract (plan §4.4): `--brief`, `--during` and `--channel` are options of the run, stamped in the summary and
    never part of the case; with `--during off` the during question is not asked and so has no answer row, and `--out`
    names the folder. Regression caught: an option that is ignored, or stamped as another value."""
    case = scored_case(tmp_path, hand20(), QUESTIONS)
    out = tmp_path / "mine"
    assert run("all_unknown", case.private, "--during", "off", "--brief", "neutral", "--channel", "detections",
               "--out", out) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["during"], summary["brief"], summary["channel"]) == (False, "neutral", "detections")
    rows = [json.loads(line) for line in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["qid"] for r in rows] == [qid(2), qid(3)]
    assert not (tmp_path / "runs").exists()


def test_a_local_system_runs_a_case_over_the_shim(tmp_path, capsys):
    """Contract (plan §4.4, B.1's exit check): `stbench run FILE.py:Class CASE_DIR --local` runs a scripted
    standard-library system through the shim, stamps the run `sandbox: none` and says it is not a verified score; its
    answers and usage are in the files. A missing file, a system folder inside the dataset and `--local` for a
    baseline are refused (exit 2) before anything is sent. Regression caught: a local run stamped as a verified one, or
    a refusal that comes only after the case was started."""
    case, file = scored_scripted(tmp_path)
    out = tmp_path / "out"
    assert run(f"{file}:Scripted", case.private, "--local", "--timeout", "60", "--out", out) == 0
    assert "not a verified score" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["sandbox"] == "none" and summary["usage"]["models"]["scripted"]["tokens_in"] == 5
    rows = {r["qid"]: r["answer"] for r in map(json.loads, (out / "answers.jsonl").read_text().splitlines())}
    assert rows[qid(1)]["when"] == "during" and rows[qid(2)]["seen"] == 20
    inside = case.dataset / "agent"
    inside.mkdir()
    (inside / "scripted.py").write_text(file.read_text(encoding="utf-8"), encoding="utf-8")
    for system, why in [(f"{tmp_path / 'missing.py'}:Scripted", "not an existing Python file"),
                        (f"{inside / 'scripted.py'}:Scripted", "must not contain or sit inside the dataset")]:
        assert run(system, case.private, "--local", "--out", tmp_path / "no") == 2
        assert why in capsys.readouterr().err
    assert run("all_unknown", case.private, "--local", "--out", tmp_path / "no") == 2
    assert "--local applies to a FILE.py:Class" in capsys.readouterr().err
    assert not (tmp_path / "no").exists()


def test_agents_resolve_through_one_registry(tmp_path, monkeypatch, capsys):
    """Contract (plan §4.4): `<agent>:<model>` is reached through `AGENTS`, a dotted path imported when it is named, and
    called as `run(model, case, args)` with the options of the command, `--reasoning` included and `args.out` a folder
    that exists; the runner's `CaseRun` is written like any other. `--reasoning` with a baseline is refused (exit 2); an
    agent that raises `Refused` exits 2 with its message and any other exception exits 1; an agent that is not
    registered, or whose module is not there, is refused. Regression caught: two lanes defining `stbench run`, an
    agent option lost on the way, or an agent's refusal that exits like a crash. The Codex driver's entry is by hand."""
    assert cmd_run.AGENTS == {"codex": "stbench.cmd_codex.run", "claude": "stbench.cmd_claude.run"}
    monkeypatch.setitem(cmd_run.AGENTS, "fake", "stream_kit.fake_agent")
    monkeypatch.setitem(cmd_run.AGENTS, "ghost", "stbench.no_such_agent_module.run")
    case = scored_case(tmp_path, hand20(), QUESTIONS)
    AGENT_CALLS.clear()
    out = tmp_path / "agent-out"
    assert run("fake:model-x", case.private, "--reasoning", "high", "--channel", "detections", "--brief", "neutral",
               "--during", "off", "--timeout", "90", "--out", out) == 0
    called = {"model": "model-x", "case": case.id, "reasoning": "high", "channel": "detections", "brief": "neutral",
              "during": "off", "timeout": 90.0, "out": out.resolve(), "out_exists": True}
    assert [called] == AGENT_CALLS
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["system"] == "fake:model-x" and summary["sandbox"] == "in-process"
    assert len((out / "answers.jsonl").read_text(encoding="utf-8").splitlines()) == 2  # during off: two end questions
    capsys.readouterr()

    assert run("all_unknown", case.private, "--reasoning", "high", "--out", tmp_path / "no") == 2
    assert "--reasoning applies to an agent" in capsys.readouterr().err
    assert run("fake:refuse", case.private, "--out", tmp_path / "no") == 2
    assert "no such login" in capsys.readouterr().err
    assert run("fake:crash", case.private, "--out", tmp_path / "crash") == 1
    assert "RuntimeError: the agent fell over" in capsys.readouterr().err
    assert not (tmp_path / "crash" / "answers.jsonl").exists()  # a crashed runner returned no case
    for system, why in [("ghost:m", "stbench.no_such_agent_module does not exist"), ("nonesuch:m", "SYSTEM must be"),
                        ("not-a-system", "SYSTEM must be"), ("C:\\x\\agent", "SYSTEM must be")]:
        assert run(system, case.private, "--out", tmp_path / "no") == 2
        assert why in capsys.readouterr().err
    assert not (tmp_path / "no").exists()
    with pytest.raises(SystemExit) as stale:  # a stale channel name is refused by the parser
        run("all_unknown", case.private, "--channel", "rgb")
    assert stale.value.code == 2


def test_a_case_that_ends_exits_1_and_one_refused_before_start_exits_2(tmp_path, monkeypatch, capsys):
    """Contract (plan §4.4): a case whose system fails ends, writes its answers (the failed ones carry their reason) and
    its summary, and exits 1; a case that cannot be run (no question the channel can answer) exits 2 with the reason and
    writes nothing. Regression caught: a failed run reported as a good one, an ended case with no summary to say why, or
    a refusal that leaves files behind."""

    class Crashing(System):
        def observe(self, steps):
            raise RuntimeError("out of memory")

    monkeypatch.setitem(baselines.BASELINES, "crashing",
                        baselines.Baseline(Crashing, needs_prior=False, prior_nm=False, needs_map=False,
                                           needs_events=False, other_stream=False))
    case = scored_case(tmp_path, hand20(), QUESTIONS)
    out = tmp_path / "ended"
    assert run("crashing", case.private, "--out", out) == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ended: observe: RuntimeError: out of memory"
    assert summary["failures"] == {"case ended: observe": 3} and summary["usage"] is None
    rows = [json.loads(line) for line in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["answer"] for r in rows] == [{"type": "failed", "reason": "case ended: observe"}] * 3
    capsys.readouterr()

    bare = scored_case(tmp_path / "bare", hand20(), [end_q(1, needs=("detections",))])
    assert run("all_unknown", bare.private, "--out", tmp_path / "refused") == 2
    assert "no question to ask on the rgb_pose channel" in capsys.readouterr().err
    assert not (tmp_path / "refused").exists()


def test_the_output_folder_is_never_under_the_datasets_scripts_or_private(tmp_path, capsys):
    """Contract (plan §4.4): `--out` under `DATASET/scripts/` or `DATASET/private/` (the agent-facing files and the
    truth) is refused with exit 2, and the default folder is `runs/<system>-<case id>` in the current directory, never
    in the dataset, with every character of the system outside [A-Za-z0-9._-] turned into `_`. Regression caught: a
    run that writes its answers among the truth (or among what a system is shown) or anywhere in the dataset it
    ran on, or a system name that makes a path of its own."""
    case = scored_case(tmp_path, hand20(), QUESTIONS)
    for inside in (case.script.root / "runs", case.private / "runs", case.dataset / "scripts",
                   case.dataset / "private"):
        assert run("all_unknown", case.private, "--out", inside) == 2
        assert "which a run never writes to" in capsys.readouterr().err
        assert not inside.exists() or inside in (case.dataset / "scripts", case.dataset / "private")
    scripted, file = scored_scripted(tmp_path / "s")
    assert run(f"{file}:Scripted", scripted.private, "--local") == 0
    assert not (scripted.dataset / "runs").exists()
    folders = [p.name for p in (tmp_path / "runs").iterdir()]
    assert len(folders) == 1 and re.fullmatch(r"[A-Za-z0-9._-]+-" + scripted.id, folders[0])
    assert folders[0].endswith("scripted.py_Scripted-" + scripted.id)
