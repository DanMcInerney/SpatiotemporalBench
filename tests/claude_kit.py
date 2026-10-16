"""Helpers for the tests of the Claude Code route (`test_claude_exec`, `test_cmd_claude`): a `Claude` over the fake
command (`claude_cli_fake`) and the fake's record of what it was run with and did.

`Fake` holds the transport and the folder the fake keeps its record in. `Fake.plan` is the JSON of the fake's per-turn
changes. `Fake.processes` gives one dict a process: what it was started with (`argv`, `cwd`, `cwd_files`, `env`,
`system`, `mcp`) and what it did (`messages`, the user messages it was sent on stdin in order, and `calls`, the tool
calls it made, each with the reply's `text`, `error`, `images` as the hash of each image block and `blocks`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stbench.claude_exec import Claude

FAKE = Path(__file__).with_name("claude_cli_fake.py")


class Fake:
    """A `Claude` that runs `claude_cli_fake.py`, with the folder it keeps its record in."""

    def __init__(self, tmp_path: Path, plan: dict | None = None, *, model: str = "sonnet", reasoning: str = "low"):
        self.home = tmp_path / "fake-claude"
        self.home.mkdir()
        self.plan_file = self.home / "plan.json"
        self.plan(plan or {})
        self.settings = {"model": model, "reasoning": reasoning}

    def new(self, **overrides) -> Claude:
        """Another transport on the same fake (a run makes its own, as it would the real command)."""
        found = {**self.settings, **overrides}
        return Claude(found["model"], reasoning=found["reasoning"], command=[sys.executable, str(FAKE)],
                      env={"FAKE_CLAUDE_DIR": str(self.home), "FAKE_CLAUDE_PLAN": str(self.plan_file)})

    def plan(self, plan: dict) -> None:
        self.plan_file.write_text(json.dumps({str(k): v for k, v in plan.items()}), encoding="utf-8")

    def processes(self) -> list[dict]:
        """What the fake was run with and did, one record per process, in order."""
        path = self.home / "processes.jsonl"
        found = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        for record in found:
            log, said = self.home / f"calls-{record['n']}.jsonl", self.home / f"messages-{record['n']}.jsonl"
            record["calls"] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] \
                if log.exists() else []
            record["messages"] = [json.loads(line) for line in said.read_text(encoding="utf-8").splitlines()] \
                if said.exists() else []
        return found
