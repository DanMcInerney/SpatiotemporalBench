"""`stbench eval` over an external system and over the Codex agent (`eval_systems.py`, plan §5.1's B3): a local agent
stamped unsandboxed, a system in the Docker sandbox (marker `docker`), and a Codex case that cannot be replayed, over
the fake command (`codex_fake`) with no model, no network and no Docker daemon but for the one Docker test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_kit import Fake
from eval_kit import cli, compiled, hand_dataset, results, short_script
from stbench import cmd_codex
from stbench.eval_results import RELAXED_NOTE
from stream_kit import SCRIPTED, end_q

MINIMAL = Path(__file__).resolve().parents[1] / "examples" / "minimal_agent" / "agent.py"


def test_a_local_agent_is_stamped_unsandboxed(tmp_path, capsys):
    """Contract (plan §5.1; docs/agent.md): `eval FILE.py:Class --local` runs the agent as a plain subprocess through
    the shim, two cases at a time, scores it (the minimal example answers `unknown` everywhere, so by hand the headline
    is the unknown share, weighted as the headline is: (0 + 3/4)/2 = 3/8) and stamps the result `sandbox: none`, not
    verified, with the note that says so, in the document, the report and the console. Regression caught: an unsandboxed
    score passed off as a verified one."""
    data = tmp_path / "data"
    hand_dataset(data)
    out = tmp_path / "out"
    assert cli(f"{MINIMAL}:MinimalAgent", "--data", data, "--out", out, "--local", "--timeout", "60",
               "--workers", "2") == 0
    shown = capsys.readouterr().out
    doc = results(out)
    assert (doc["sandbox"], doc["verified"], doc["partial"], doc["reasoning"]) == ("none", False, False, None)
    assert doc["layouts"]["apartment_a"]["headline"]["mean"] == pytest.approx(0.375) and len(doc["rows"]) == 6
    assert doc["notes"] == ["sandbox none: run without the Docker sandbox: not a verified score", RELAXED_NOTE]
    assert "not a verified score" in shown and "- sandbox: none (UNSANDBOXED" in shown
    assert "*** sandbox none: run without the Docker sandbox: not a verified score ***" in (
        out / "results.md").read_text(encoding="utf-8")
    assert all(c["state_bytes"]["0"] > 0 for c in doc["cases"])   # the example logs what it is sent
    assert cli(f"{tmp_path / 'missing.py'}:Agent", "--data", data, "--out", tmp_path / "no", "--local") == 2
    assert "not an existing Python file" in capsys.readouterr().err and not (tmp_path / "no").exists()


@pytest.mark.docker
def test_a_docker_system(tmp_path):
    """Contract (plan §5.1; VISION rule 6): a scripted standard-library system (`stream_kit.SCRIPTED`) in the Docker
    sandbox, two cases at a time, completes both: stamped `sandbox: docker` and verified, its answers by hand (each
    question answered when it was asked), its state measured (1,000 bytes at `finish`) and its usage reported as a
    model the price table lacks (no cost, and the note says why). Regression caught: the Docker path of `eval`
    unexercised, or a case that cannot share the sandbox's image with another."""
    data, agent = tmp_path / "data", tmp_path / "agent"
    hand_dataset(data)
    agent.mkdir()
    (agent / "scripted.py").write_text(SCRIPTED, encoding="utf-8")
    out = tmp_path / "out"
    assert cli(f"{agent / 'scripted.py'}:Scripted", "--data", data, "--out", out, "--workers", "2",
               "--timeout", "120") == 0
    doc = results(out)
    assert (doc["sandbox"], doc["verified"], doc["partial"]) == ("docker", True, False)
    assert [c["status"] for c in doc["cases"]] == ["complete"] * 2 and len(doc["rows"]) == 6
    assert {c["state_bytes"]["finish"] for c in doc["cases"]} == {1000}
    assert {c["cost_note"] for c in doc["cases"]} == {"no price for scripted"}
    answers = [json.loads(line) for c in doc["cases"]
               for line in (out / "cases" / c["id"] / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted((a["answer"]["when"], a["answer"]["seen"]) for a in answers) == [("during", 13)] * 2 + [
        ("end", 20)] * 4


def test_eval_runs_the_codex_agent_on_the_detections_channel(tmp_path, monkeypatch):
    """Contract (docs/codex.md, "The detections arm"): `eval codex:<model> --channel detections` is the documented way
    to run the diagnostic over a dataset: the channel reaches the agent (it is not refused, as the picture-only agent
    was), the case completes with no image block sent to the model, and the result says `detections`, so `eval compare`
    can set it beside a `rgb_pose` result. Regression caught: `eval` refusing or ignoring the channel for an agent, or
    a result that does not say which channel the agent read."""
    data = tmp_path / "data"
    case = compiled(data, short_script(1), [end_q(1)])
    fake = Fake(tmp_path)
    monkeypatch.setattr(cmd_codex, "make_codex", lambda model, reasoning: fake.new(model=model,
                                                                                         reasoning=reasoning))
    assert cli("codex:fake-model", "--data", data, "--channel", "detections", "--reasoning", "medium") == 0
    doc = results(tmp_path / "runs" / "codex_fake-model-detections")
    (done,) = doc["cases"]
    assert (doc["channel"], done["id"], done["status"]) == ("detections", case.id, "complete")
    (only,) = fake.execs()
    assert all(c["images"] == [] and set(c["blocks"]) == {"text"}   # the last frame comes with the end question
               for c in only["calls"] if c["tool"] == "next_frame")


def test_a_codex_case_reruns_from_scratch(tmp_path, monkeypatch):
    """Contract (plan §5.1; docs/codex.md): `eval codex:<model>` runs the agent through `cmd_run.AGENTS` with the
    options of the command (the effort goes to the transport). A case stopped part-way (its exec hit a usage limit
    after the first frame: `ended: observe: TransportError: limit`) leaves its journal, and the rerun removes the folder
    and runs the case from its start (`cmd_codex` refuses a folder holding a journal, and a Codex call cannot be
    replayed): the second fake is run once (the whole case is one exec), the journal holds that exec and no stale one,
    and the result is complete, stamped with the effort. Regression caught: `cmd_codex`'s refusal blocking a
    resume, a stale journal mixed into a rerun, or an effort that never reaches the results."""
    data = tmp_path / "data"
    case = compiled(data, short_script(1), [end_q(1)])
    folder = tmp_path / "runs" / "codex_fake-model-rgb_pose" / "cases" / case.id
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    first, second = Fake(tmp_path / "one", {0: {"pulls": 1, "fail_late": "limit"}}), Fake(tmp_path / "two")

    def run(fake: Fake) -> int:
        monkeypatch.setattr(cmd_codex, "make_codex", lambda model, reasoning: fake.new(model=model,
                                                                                         reasoning=reasoning))
        return cli("codex:fake-model", "--data", data, "--reasoning", "medium")

    assert run(first) == 1
    assert (folder / "codex.calls.jsonl").exists() and len(first.execs()) == 1
    doc = results(folder.parents[1])
    assert [c["status"] for c in doc["cases"]] == ["ended: observe: TransportError: limit"] and doc["partial"]
    assert (doc["reasoning"], doc["rows"]) == ("medium", [])
    assert run(second) == 0
    assert len(second.execs()) == 1
    journal = [json.loads(line) for line in (folder / "codex.calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([r for r in journal if "event" not in r]) == 1 and journal[0]["event"] == "run"
    doc = results(folder.parents[1])
    (done,) = doc["cases"]
    assert (done["status"], doc["partial"], doc["reasoning"]) == ("complete", False, "medium")
    assert done["tokens"] == {"tokens_in": 100, "tokens_cached_in": 40, "tokens_cache_write_in": 0, "tokens_out": 10}
    assert done["cost_note"] == "no price for fake-model" and doc["cost_usd"] is None and len(doc["rows"]) == 1
