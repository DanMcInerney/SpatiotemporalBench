"""The two example agents (`examples/`): they complete a case on the stream protocol through `stbench run --local`, and
they stay what docs/agent.md says they are: standard library only (and the shim's `system_shim`), so they run unchanged
in the Docker sandbox.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from run_kit import scored_case
from stbench import cli
from stream_kit import QUESTIONS, hand20, qid

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
AGENTS = [pytest.param("minimal_agent", "MinimalAgent", id="minimal"),
          pytest.param("template_agent", "TemplateAgent", id="template")]


def load(folder: str):
    spec = importlib.util.spec_from_file_location(f"{folder}_under_test", EXAMPLES / folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("folder", "cls"), AGENTS)
def test_the_examples_complete_a_case(tmp_path, capsys, folder, cls):
    """Contract (plan 4.9; docs/agent.md): each example, as `stbench run FILE.py:Class CASE_DIR --local` runs it through
    the shim on a synthetic case, completes the case: exit 0, status complete, every question answered `unknown` (the
    during one and the two end ones), no failures, and a state folder that is no longer empty (the agent logged what
    it was sent: the harness measured it at the end of the task and at `finish`), and the run is scored: each `unknown`
    on an answerable gold is `abstained` and scores 0. Regression caught: the documented
    plug-in contract broken, an example still on the retired wire (`ask_after_k`, the structured question), or one
    that cannot complete a case."""
    case = scored_case(tmp_path / "data", hand20(), QUESTIONS)
    out = tmp_path / "out"
    assert cli.main(["run", f"{EXAMPLES / folder / 'agent.py'}:{cls}", str(case.private), "--local", "--timeout", "60",
                     "--out", str(out)]) == 0
    assert "not a verified score" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["status"], summary["failures"], summary["sandbox"]) == ("complete", {}, "none")
    assert summary["state_bytes"]["0"] > 0 and summary["state_bytes"]["finish"] >= summary["state_bytes"]["0"]
    rows = [json.loads(line) for line in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"qid": qid(n), "answer": {"type": "unknown"}} for n in (1, 2, 3)]
    scored = [json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["qid"], r["outcome"], r["score"]) for r in scored] == [(qid(n), "abstained", 0.0) for n in (1, 2, 3)]
    assert summary["scores"]["headline"]["mean"] == 0.0  # `unknown` on an answerable gold scores 0


def test_the_minimal_agent_logs_each_steps_clock_base_and_status(tmp_path):
    """Contract (the example's docstring): `observe` appends one line a step to `state_dir/steps.jsonl` with the step's
    clock, the base's position and the status, in order and across observes; a question gets `unknown`. Regression
    caught: an example that logs the camera's pose for the base's (the keys of the stream protocol's step), or reads a
    step key the wire no longer carries."""
    (tmp_path / "state").mkdir()
    agent = load("minimal_agent").MinimalAgent(tmp_path / "state", tmp_path / "work")
    status = {"phase": "cleaning", "task": 0, "battery_pct": 97, "brush": True}
    steps = [{"k": k, "clock": f"2026-03-02T08:00:0{k}", "pose": {"pos": [1.0 + k, 2.0, 0.0], "rot": [1, 0, 0, 0],
                                                                   "yaw_deg": 0.0},
              "camera": {"pos": [9.0, 9.0, 9.0]}, "status": status} for k in (0, 1, 2)]
    agent.observe(steps[:2])
    agent.observe(steps[2:])
    lines = [json.loads(line) for line in (tmp_path / "state" / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines == [{"clock": f"2026-03-02T08:00:0{k}", "base": [1.0 + k, 2.0, 0.0], "status": status}
                     for k in (0, 1, 2)]
    assert agent.answer({"qid": "q", "text": "?", "answer_type": "count"}) == {"type": "unknown"}


@pytest.mark.parametrize("folder", ["minimal_agent", "template_agent"])
def test_the_examples_import_only_the_standard_library_and_the_shim(folder):
    """Contract (docs/agent.md): an example runs unchanged in the Docker sandbox, which has the standard library and
    what the agent's folder vendors, so it imports nothing else but the shim's own `system_shim` (`from __future__`
    aside). Regression caught: a numpy or Pillow import added to an example to show off decoding, which the sandbox
    lacks."""
    tree = ast.parse((EXAMPLES / folder / "agent.py").read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= set(sys.stdlib_module_names) | {"system_shim"}
