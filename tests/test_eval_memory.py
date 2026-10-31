"""`--memory SPEC` through `stbench run` and `stbench eval` (`cmd_run`, `cmd_codex`, `eval_cmd`, `eval_results`,
`eval_compare`, `eval_report`, `cmd_pilot`, `export_files`): what a run with a memory leaves and is refused, how
`results.json` names the memory, that a resume cannot cross memories, and that a memory row compares with the reference
row of its model and effort as one lever. The agent itself is `test_codex_memory.py`'s; here it is the fake command
(`codex_fake`) behind the real driver, or `eval_kit`'s fake agent behind the real `eval`. No model, no `codex`, no
network."""

from __future__ import annotations

import json
import sys

import pytest

import test_cmd_pilot as pilot
from ability_kit import results_doc, scored
from codex_kit import GUIDE, MEMORY_SENSE, MEMORY_SERVER, Fake, job_also_asks, memory
from eval_kit import RUNS, cli, hand_dataset, register_agent, results
from stbench import cli as stbench
from stbench import cmd_codex, cmd_run, eval_results
from stbench.eval_compare import Refused, compare
from stbench.export_files import result_name
from test_cmd_codex import jsonl, small_case

CLAIM = {"server": "mem", "tool": "claim", "arguments": {"text": "frame {k}"}}


@pytest.fixture(autouse=True)
def the_job_asks_what_the_case_asks(monkeypatch):
    """`small_case` holds a `number` during question, which `vacuum/1` asks none of: see `test_cmd_codex.py`."""
    job_also_asks(monkeypatch, "distance_between")


def codex_run(monkeypatch, fake: Fake, *argv) -> int:
    """`stbench run ARGV` with the Codex transport replaced by one over `fake`, which is given the guide a run with a
    memory hands `make_codex`."""
    monkeypatch.setattr(cmd_codex, "make_codex", lambda model, reasoning, guide=None: fake.new(
        model=model, reasoning=reasoning, guide=guide))
    return stbench.main(["run", *map(str, argv)])


def test_a_run_with_a_memory_names_it_in_every_record_and_keeps_its_store(tmp_path, monkeypatch):
    """Contract (`cmd_codex`): `stbench run codex:<model> CASE --memory SPEC` completes the case with the memory's
    server in the exec and its guide in the instructions; the journal's `run` event records the memory as loaded (its
    name, both argvs and the guide's text) and no digest of it, `codex.summary.json` names it and counts its tool calls
    by name (the claims, once for each of the six frames), the store is kept beside the case as `<name>-store` and the
    harness's `state_bytes` counts it; a default folder gains `-memory-<SPEC's stem>`, and a run without `--memory`
    leaves none of these marks (no `memory` key anywhere). Regression caught: a memory row that cannot be told from a
    reference row, a stamp that is a hash (no way to read what ran), a memory run landing in the reference agent's
    folder, or a store lost with the case."""
    case = small_case(tmp_path / "data")
    held = memory(tmp_path / "spec")
    fake = Fake(tmp_path, {0: {"memory": [CLAIM]}})
    out = tmp_path / "out"
    assert codex_run(monkeypatch, fake, "codex:fake-model", case.private, "--memory", held.path, "--out", out) == 0
    summary = json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))
    assert (summary["memory"], summary["invalid"], summary["scan"]) == ("mem", False, [])
    assert summary["tool_calls"]["claim"] == 6 and summary["tool_calls"]["next_frame"] == 7
    first = jsonl(out / "codex.calls.jsonl")[0]
    assert first["memory"] == {"name": "mem", "serve": [sys.executable, str(MEMORY_SERVER), "{store}"],
                               "sense": [sys.executable, str(MEMORY_SENSE), "{store}"], "guide": GUIDE}
    claims = jsonl(out / "mem-store" / "claims.jsonl")
    assert [c["text"] for c in claims] == [f"frame {k}" for k in range(6)]
    assert not (out / "mem-store" / "cache").exists()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["state_bytes"]["finish"] > 0
    (only,) = fake.execs()
    assert only["servers"] == ["mem", "stbench"] and only["instructions"].endswith(f"\n\n{GUIDE}")

    assert codex_run(monkeypatch, fake, "codex:fake-model", case.private, "--memory", held.path) == 0
    named = tmp_path / "runs" / f"codex_fake-model-{case.id}-memory-memory"
    assert json.loads((named / "codex.summary.json").read_text(encoding="utf-8"))["memory"] == "mem"
    assert codex_run(monkeypatch, fake, "codex:fake-model", case.private) == 0
    plain = tmp_path / "runs" / f"codex_fake-model-{case.id}"
    assert "memory" not in json.loads((plain / "codex.summary.json").read_text(encoding="utf-8"))
    assert "memory" not in jsonl(plain / "codex.calls.jsonl")[0] and not list(plain.glob("*-store"))
    assert [e["instructions"] for e in fake.execs()][-1].startswith("You are a helpful assistant.")
    assert fake.execs()[-1]["servers"] == ["stbench"]


def test_a_memory_is_refused_where_it_cannot_be_fed_or_loaded_and_nothing_is_made(tmp_path, monkeypatch, capsys):
    """Contract (`cmd_run.classify`, `cmd_run.load_memory`): `--memory` is refused with exit 2 on the detections
    channel, with `--text-only` (the memory is fed the camera images), for a system that is no Codex or Claude agent,
    and for a SPEC that cannot be read, before anything is sent or made: no output folder, no exec. Regression
    caught: a memory run on a channel that sends it no image (a store that sees nothing, measured as if it saw), or a
    SPEC mistake found after the first exec."""
    case = small_case(tmp_path / "data")
    held = memory(tmp_path / "spec")
    fake = Fake(tmp_path)
    images = "--memory feeds the memory the camera images"
    cases = [(["codex:fake-model", "--channel", "detections"], held.path, images),
             (["codex:fake-model", "--text-only"], held.path, images),
             (["all_unknown"], held.path, "--memory applies to claude or codex agents"),
             (["codex:fake-model"], tmp_path / "no-such.json", "cannot read the memory SPEC")]
    for options, spec, message in cases:
        out = tmp_path / "out"
        system, *rest = options
        assert codex_run(monkeypatch, fake, system, case.private, *rest, "--memory", spec, "--out", out) == 2, options
        assert message in capsys.readouterr().err and not out.exists(), options
    assert fake.execs() == []


def test_eval_names_the_memory_and_a_rerun_with_another_is_refused(tmp_path, monkeypatch, capsys):
    """Contract (`eval_results`): `results.json` of an agent run with `--memory` holds `"memory": "<name>"` (the SPEC's
    name, not its path) and `results.md` names the system `with memory <name>`; every case's runner is handed the SPEC;
    a reference row has no `memory` key; a rerun into the folder with another memory, with none, or (into a reference
    folder) with one is refused (exit 2) saying which, and with the same memory it goes on; a default folder gains
    `-memory-<SPEC's stem>`; `--memory` is refused for a system that is not a Codex agent. Regression caught: a rerun
    that mixes two memories' cases in one results file, or a memory row whose results read as the reference's."""
    data = tmp_path / "data"
    hand_dataset(data)
    register_agent(monkeypatch)
    monkeypatch.setattr(cmd_run, "MEMORY_AGENTS", frozenset({"codex", "fake"}))  # eval_kit's fake agent takes one
    alpha, beta = memory(tmp_path / "a", name="alpha"), memory(tmp_path / "b", name="beta")
    assert cli("fake:ok", "--data", data, "--memory", alpha.path) == 0
    out = tmp_path / "runs" / "fake_ok-rgb_pose-memory-memory"
    assert results(out)["memory"] == "alpha" and "fake:ok with memory alpha" in (out / "results.md").read_text("utf-8")
    assert {run["memory"] for run in RUNS} == {str(alpha.path.resolve())} and len(RUNS) == 2
    assert cli("fake:ok", "--data", data, "--out", tmp_path / "plain") == 0
    assert "memory" not in results(tmp_path / "plain") and {run["memory"] for run in RUNS[2:]} == {None}
    capsys.readouterr()
    for folder, extra, there, now in ((out, ["--memory", beta.path], "'alpha'", "'beta'"), (out, [], "'alpha'", "None"),
                                      (tmp_path / "plain", ["--memory", alpha.path], "None", "'alpha'")):
        assert cli("fake:ok", "--data", data, "--out", folder, *extra) == 2
        assert f"memory: there {there}, now {now}" in capsys.readouterr().err
    assert cli("fake:ok", "--data", data, "--out", out, "--memory", alpha.path) == 0
    monkeypatch.setattr(cmd_run, "MEMORY_AGENTS", frozenset({"codex"}))
    assert cli("fake:ok", "--data", data, "--out", tmp_path / "no", "--memory", alpha.path) == 2
    assert cli("all_unknown", "--data", data, "--out", tmp_path / "no", "--memory", alpha.path) == 2
    assert "--memory applies to codex agents" in capsys.readouterr().err and not (tmp_path / "no").exists()


def test_compare_takes_the_memory_as_part_of_the_agent_lever(tmp_path):
    """Contract (`eval_compare.LEVER_GROUPS`; VISION rule 13): a memory row compares with the reference row of the same
    model and effort (the agent alone differs) and with a row of another memory, the report naming each `with memory
    <name>`; a memory row that also differs in the channel is two levers and is refused; a results document of an
    earlier build, which has no `memory` key, is a reference row. Regression caught: a memory arm that cannot be
    compared with its baseline (the point of the hook), or one compared across a second lever."""
    rows = [*scored("ego_now", [1, 1]), *scored("object_where", [1, 0, 0])]
    docs = {"ref": {}, "mem": {"memory": "mem"}, "other": {"memory": "other"},
            "moved": {"memory": "mem", "channel": "detections"}}
    for name, over in docs.items():
        eval_results.write(tmp_path / name, results_doc(rows, system="codex:m", reasoning="low", **over))
    older = results_doc(rows, system="codex:m", reasoning="low")
    assert "memory" not in older
    eval_results.write(tmp_path / "older", older)
    text = compare([str(tmp_path / "ref"), str(tmp_path / "mem"), str(tmp_path / "other"), str(tmp_path / "older")])
    assert "- A = codex:m, rgb_pose channel" in text and "- B = codex:m with memory mem, rgb_pose channel" in text
    assert "- C = codex:m with memory other, rgb_pose channel" in text and "- D = codex:m, rgb_pose channel" in text
    with pytest.raises(Refused, match=r"differ in more than one lever \(agent, channel\)"):
        compare([str(tmp_path / "ref"), str(tmp_path / "moved")])


def test_a_memory_row_has_its_own_export_name_and_its_own_pilot_agent(tmp_path, capsys):
    """Contract: a results document with a memory ships under a name with the memory in it, so that it and the
    reference row of its model and effort can be exported together; the pilot report names an agent with its memory,
    keeps the memory row and the reference row apart, and reads a leak pair of a memory row and a reference row as two
    levers apart (the agent and `during`). Regression caught: two rows exported under one name (the second refused, or
    the first overwritten), a pilot that pools a memory arm with its baseline, or a KeyError on a document that lacks
    the key."""
    base = {"system": "codex:m", "reasoning": "low", "channel": "rgb_pose", "brief": "correct", "during": True}
    assert result_name(base) == "codex_m-low-rgb_pose"
    assert result_name({**base, "memory": "mem"}) == "codex_m-low-memory-mem-rgb_pose"
    ref = pilot.save(tmp_path, "ref", pilot.pilot_doc(pilot.SOL, {"r0": [1, 0], "r1": [1, 0]}))
    held = pilot.save(tmp_path, "held", {**pilot.pilot_doc(pilot.SOL, {"m0": [1, 1], "m1": [1, 1]}), "memory": "mem"})
    text = pilot.report(capsys, "--runs", ref, held)
    assert "codex:gpt-6.1-sol medium" in text and "codex:gpt-6.1-sol with memory mem medium" in text
    records, rows = [pilot.case("c1", "s1", "vacuum", "short")], [pilot.row("c1", "s1", "e1", 1, job="vacuum")]
    on = pilot.save(tmp_path, "on", {**pilot.doc(pilot.SOL, records, rows), "memory": "mem"})
    off = pilot.save(tmp_path, "off", pilot.doc(pilot.SOL, records, rows, during=False))
    assert pilot.run("--leak", on, off) == 2
    assert "differ in more than one lever (agent, during)" in capsys.readouterr().err
