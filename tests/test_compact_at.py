"""`--compact-at TOKENS` through `codex_exec`, `stbench run` and `stbench eval` (`cmd_run`, `cmd_codex`, `eval_cmd`,
`eval_results`, `eval_compare`, `eval_report`, `cmd_pilot`, `export_files`, `view`): Codex is told to compact the thread
past a number of tokens, the run says so in every record it leaves, and a row with a limit is never taken for one
without. The agent is the fake command (`codex_fake`) behind the real driver, or `eval_kit`'s fake agent behind the real
`eval`. No model, no `codex`, no network. The default (no limit) is `test_codex_exec.py`'s.
"""

from __future__ import annotations

import json
import re

import pytest

import test_cmd_pilot as pilot
from ability_kit import results_doc, scored
from codex_kit import Fake, job_also_asks, run_exec
from eval_kit import RUNS, cli, hand_dataset, register_agent, results
from stbench import cli as stbench
from stbench import cmd_codex, cmd_run, eval_results
from stbench.codex_exec import Codex
from stbench.eval_compare import Refused, compare
from stbench.export_files import result_name
from stbench.view import read_run
from test_cmd_codex import jsonl, small_case

SETTING = "model_auto_compact_token_limit"


@pytest.fixture(autouse=True)
def the_job_asks_what_the_case_asks(monkeypatch):
    """`small_case` holds a `number` during question, which `vacuum/1` asks none of: see `test_cmd_codex.py`."""
    job_also_asks(monkeypatch, "distance_between")


def settings(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]


def shape(argv: list[str]) -> list[str]:
    """`argv` with the temp-file paths of one transport blanked: what is left is the same for any transport."""
    return [re.sub(r"^(model_catalog_json|model_instructions_file)=.*", r"\1=<file>", a) for a in argv]


def codex_run(monkeypatch, fake: Fake, *argv) -> int:
    """`stbench run ARGV` with the Codex transport replaced by one over `fake`, built with what the driver hands it."""
    monkeypatch.setattr(cmd_codex, "make_codex", lambda model, reasoning, guide=None, compact_at=None: fake.new(
        model=model, reasoning=reasoning, guide=guide, compact_at=compact_at))
    return stbench.main(["run", *map(str, argv)])


def test_the_limit_is_one_setting_of_every_exec_and_nothing_else_of_the_argv_changes(tmp_path):
    """Contract (`codex_exec.Codex`): a transport made with `compact_at=N` puts `model_auto_compact_token_limit=N` (a
    plain integer, once) in the argv of a start and of a resume alike, with a memory's server or without, and the argv
    is otherwise the argv of a transport without it (every other setting in its order, every flag, the thread and the
    stdin marker); the execs the fake command receives hold it, the resume's too, and a transport without it launches
    none with the key. A value that is no count of tokens (0, a negative, a bool, a float, text) is a `ValueError`.
    Regression caught: a limit on the start only (the thread would be compacted at Codex's own limit after its first
    resume), a flag or setting changed beside it, the number written in another form, or a 0 or `True` taken for a
    limit."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    plain, limited = Fake(tmp_path / "a"), Fake(tmp_path / "b", compact_at=50_000)
    plain.codex.prepare()
    limited.codex.prepare()
    store = tmp_path / "store"
    for kind, thread in (("start", None), ("resume", "thread-0000")):
        for extra in ({}, {"memory": ("mem", ["python", "-m", "memory_server", str(store)])}):
            without = plain.codex.argv(kind, thread, tmp_path / "server.json", tmp_path / "empty", **extra)
            with_it = limited.codex.argv(kind, thread, tmp_path / "server.json", tmp_path / "empty", **extra)
            assert settings(with_it).count(f"{SETTING}=50000") == 1, (kind, extra)
            at = with_it.index(f"{SETTING}=50000")
            rest = with_it[:at - 1] + with_it[at + 1:]   # the argv without the `-c` and its setting
            assert with_it[at - 1] == "-c" and shape(rest) == shape(without), (kind, extra)
            assert not [s for s in settings(without) if s.startswith(SETTING)]
    first = run_exec(limited)
    run_exec(limited, kind="resume", thread=first.thread)
    run_exec(plain)
    assert [e["argv"].count(f"{SETTING}=50000") for e in limited.execs()] == [1, 1]
    assert [e["argv"].count(f"{SETTING}=50000") for e in plain.execs()] == [0]
    for bad in (0, -5, True, 1.5, "50000"):
        with pytest.raises(ValueError, match="compact_at must be a whole number of tokens"):
            Codex("some-model", compact_at=bad)


def test_a_run_with_a_limit_passes_it_to_every_exec_and_records_it_and_one_without_records_none(tmp_path, monkeypatch):
    """Contract (`cmd_codex`): `stbench run codex:<model> CASE --compact-at N` runs the case's every exec, the resume
    included, with the limit, and records it as `compact_at` in the journal's `run` event and in `codex.summary.json`;
    a run without the option records `compact_at: null` there and launches no exec with the key. Regression caught: a
    limit that is parsed and never reaches the transport (a run labelled with a limit it did not have), a record that
    states it only when it is set (a reader cannot tell a reference run from one of an earlier build), or a resume that
    forgets it."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path, {0: {"pulls": 3}})  # the first exec ends its turn early, so the thread is resumed once
    capped = tmp_path / "capped"
    assert codex_run(monkeypatch, fake, "codex:fake-model", case.private, "--compact-at", 50000, "--out", capped) == 0
    summary = json.loads((capped / "codex.summary.json").read_text(encoding="utf-8"))
    assert summary["compact_at"] == 50000 and summary["invalid"] is False
    assert summary["execs"] == {"total": 2, "by_kind": {"start": 1, "resume": 1}}
    first = jsonl(capped / "codex.calls.jsonl")[0]
    assert first["event"] == "run" and first["compact_at"] == 50000
    assert [e["argv"].count(f"{SETTING}=50000") for e in fake.execs()] == [1, 1]

    plain = tmp_path / "plain"
    assert codex_run(monkeypatch, fake, "codex:fake-model", case.private, "--out", plain) == 0
    summary = json.loads((plain / "codex.summary.json").read_text(encoding="utf-8"))
    assert "compact_at" in summary and summary["compact_at"] is None
    assert jsonl(plain / "codex.calls.jsonl")[0]["compact_at"] is None
    assert not [a for e in fake.execs()[2:] for a in e["argv"] if a.startswith(SETTING)]


def test_a_limit_is_refused_where_it_means_nothing_and_for_a_value_that_is_no_count(tmp_path, monkeypatch, capsys):
    """Contract (`cmd_run.classify`, `cmd_run.token_limit`): `--compact-at` is refused with exit 2 for a system that is
    neither a Codex nor a Claude agent, in `run` and in `eval`, and a value that is not a whole number of tokens of at
    least 1 (0, a negative, a fraction, text) is a usage error (exit 2) of both commands; nothing is sent or made
    first: no output folder, no exec. Regression caught: a limit accepted by a baseline (a row labelled with a setting
    that did nothing), or a 0 or a typo passed on to Codex, which would fail in the middle of a paid run."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    out = tmp_path / "out"
    assert codex_run(monkeypatch, fake, "all_unknown", case.private, "--compact-at", 5, "--out", out) == 2
    assert "--compact-at applies to claude or codex agents" in capsys.readouterr().err and not out.exists()
    hand_dataset(tmp_path / "hand")
    register_agent(monkeypatch)
    for system in ("all_unknown", "fake:ok"):  # the fake agent is no Codex agent
        assert cli(system, "--data", tmp_path / "hand", "--out", out, "--compact-at", 5) == 2
        assert "--compact-at applies to claude or codex agents" in capsys.readouterr().err and not out.exists()
    for bad in ("0", "-5", "1.5", "50k", "lots"):
        for command in (["run", "codex:fake-model", str(case.private)],
                        ["eval", "codex:fake-model", "--data", str(tmp_path)]):
            with pytest.raises(SystemExit) as stopped:
                stbench.main([*command, "--compact-at", bad, "--out", str(out)])
            assert stopped.value.code == 2 and "must be a whole number of tokens, at least 1" in capsys.readouterr().err
    assert fake.execs() == [] and not out.exists() and RUNS == []


def test_eval_names_the_limit_and_a_rerun_with_another_is_refused(tmp_path, monkeypatch, capsys):
    """Contract (`eval_results`, `eval_report`, `view`): `results.json` of an agent run with `--compact-at` holds
    `"compact_at": N` and `results.md` names the system `compacts at 50k tokens`; every case's runner is handed the
    limit; a reference row has no `compact_at` key (and its runners get none); the viewer's header carries it; a rerun
    into the folder with another limit, with none or (into a reference folder) with one is refused (exit 2) saying
    which, and with the same limit it goes on. Regression caught: a rerun that mixes cases made at two limits in one
    results file, or a row with a limit whose results read as the reference's."""
    data = tmp_path / "data"
    hand_dataset(data)
    register_agent(monkeypatch)
    monkeypatch.setattr(cmd_run, "COMPACT_AGENTS", frozenset({"codex", "fake"}))  # eval_kit's fake agent takes one
    capped, plain = tmp_path / "capped", tmp_path / "plain"
    assert cli("fake:ok", "--data", data, "--out", capped, "--compact-at", 50000) == 0
    assert results(capped)["compact_at"] == 50000
    report = (capped / "results.md").read_text(encoding="utf-8")
    assert "# SpatiotemporalBench results: fake:ok, compacts at 50k tokens" in report
    assert {run["compact_at"] for run in RUNS} == {50000} and len(RUNS) == 2
    assert cli("fake:ok", "--data", data, "--out", plain) == 0
    assert "compact_at" not in results(plain) and {run["compact_at"] for run in RUNS[2:]} == {None}
    assert read_run(capped, data)["arms"][0]["header"]["compact_at"] == 50000
    assert read_run(plain, data)["arms"][0]["header"]["compact_at"] is None
    capsys.readouterr()
    for folder, extra, there, now in ((capped, ["--compact-at", 60000], "50000", "60000"),
                                      (capped, [], "50000", "None"), (plain, ["--compact-at", 50000], "None", "50000")):
        assert cli("fake:ok", "--data", data, "--out", folder, *extra) == 2
        assert f"compact_at: there {there}, now {now}" in capsys.readouterr().err
    assert cli("fake:ok", "--data", data, "--out", capped, "--compact-at", 50000) == 0
    assert results(capped)["compact_at"] == 50000


def test_compare_takes_the_limit_as_part_of_the_agent_lever(tmp_path):
    """Contract (`eval_compare.LEVER_GROUPS`; VISION rule 13): a row with a limit compares with the reference row of its
    model and effort and with a row of another limit (the agent alone differs), and two rows that differ only in the
    memory and share their limit compare as one lever too; the report names each `compacts at 50k tokens` (a number that
    is not a round thousand as it is), so a row with a limit never stands in for one without; a row that also differs in
    the channel is two levers and is refused. Regression caught: a limited arm that cannot be compared with its
    baseline (the point of the option), a limited row labelled as the reference row, or a comparison across a second
    lever."""
    rows = [*scored("ego_now", [1, 1]), *scored("object_where", [1, 0, 0])]
    docs = {"ref": {}, "capped": {"compact_at": 50000}, "other": {"compact_at": 100000}, "odd": {"compact_at": 50500},
            "held": {"memory": "mem"}, "heldcap": {"memory": "mem", "compact_at": 50000},
            "moved": {"compact_at": 50000, "channel": "detections"}}
    for name, over in docs.items():
        eval_results.write(tmp_path / name, results_doc(rows, system="codex:m", reasoning="low", **over))
    assert eval_results.problems(results_doc(rows, compact_at=50000)) == []
    assert eval_results.problems(results_doc(rows, compact_at="lots")) != []
    text = compare([str(tmp_path / n) for n in ("ref", "capped", "other", "odd")])
    assert "- A = codex:m, rgb_pose channel" in text
    assert "- B = codex:m, compacts at 50k tokens, rgb_pose channel" in text
    assert "- C = codex:m, compacts at 100k tokens, rgb_pose channel" in text
    assert "- D = codex:m, compacts at 50500 tokens, rgb_pose channel" in text
    text = compare([str(tmp_path / "capped"), str(tmp_path / "heldcap")])
    assert "- B = codex:m with memory mem, compacts at 50k tokens, rgb_pose channel" in text
    compare([str(tmp_path / "held"), str(tmp_path / "heldcap")])
    with pytest.raises(Refused, match=r"differ in more than one lever \(agent, channel\)"):
        compare([str(tmp_path / "ref"), str(tmp_path / "moved")])


def test_a_row_with_a_limit_has_its_own_export_name_and_its_own_pilot_agent(tmp_path, capsys):
    """Contract: a results document with a limit ships under a name with the limit in it, so that it and the reference
    row of its model and effort can be exported together, and the pilot report names an agent with its limit and keeps
    the limited row and the reference row apart. Regression caught: two rows exported under one name (the second
    refused, or the first overwritten), or a pilot that pools a limited arm with its baseline."""
    base = {"system": "codex:m", "reasoning": "low", "channel": "rgb_pose", "brief": "correct", "during": True}
    assert result_name(base) == "codex_m-low-rgb_pose"
    assert result_name({**base, "compact_at": 50000}) == "codex_m-low-compact-50000-rgb_pose"
    held = {**base, "memory": "mem", "compact_at": 50000}
    assert result_name(held) == "codex_m-low-memory-mem-compact-50000-rgb_pose"
    ref = pilot.save(tmp_path, "ref", pilot.pilot_doc(pilot.SOL, {"r0": [1, 0], "r1": [1, 0]}))
    capped = {**pilot.pilot_doc(pilot.SOL, {"m0": [1, 1], "m1": [1, 1]}), "compact_at": 50000}
    text = pilot.report(capsys, "--runs", ref, pilot.save(tmp_path, "capped", capped))
    assert "codex:gpt-6.1-sol medium" in text and "codex:gpt-6.1-sol, compacts at 50k tokens medium" in text
