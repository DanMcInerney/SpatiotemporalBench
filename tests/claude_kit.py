"""Helpers for the tests of the Claude Code route (`test_claude_exec`, `test_cmd_claude`): a `Claude` over the fake
command (`claude_cli_fake`) and the fake's record of what it was run with and did.

`Fake` holds the transport and the folder the fake keeps its record in. `Fake.plan` is the JSON of the fake's per-turn
changes. `Fake.processes` gives one dict a process: what it was started with (`argv`, `cwd`, `cwd_files`, `env`,
`system`, `mcp`, the config's servers by name, `servers`, their names and `server_instructions`, what each
`initialize` reply held) and what it did (`messages`, the user messages it was sent on stdin in order, and `calls`,
the tool calls it made, each with its `server`, the reply's `text`, `error`, `images` as the hash of each image block
and `blocks`). `pulled` is the replies to its `next_frame` calls. `guide` and `compact_at` are the transport's; `env`
is added to the fake's environment (the way a test gives the fake `DISABLE_AUTO_COMPACT`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stbench import cli, cmd_claude
from stbench.claude_exec import Claude

FAKE = Path(__file__).with_name("claude_cli_fake.py")


class Fake:
    """A `Claude` that runs `claude_cli_fake.py`, with the folder it keeps its record in."""

    def __init__(self, tmp_path: Path, plan: dict | None = None, *, model: str = "sonnet", reasoning: str = "low",
                 guide: str | None = None, compact_at: int | None = None, env: dict[str, str] | None = None):
        self.home = tmp_path / "fake-claude"
        self.home.mkdir()
        self.plan_file = self.home / "plan.json"
        self.plan(plan or {})
        self.settings = {"model": model, "reasoning": reasoning, "guide": guide, "compact_at": compact_at}
        self.env = env or {}

    def new(self, **overrides) -> Claude:
        """Another transport on the same fake (a run makes its own, as it would the real command)."""
        found = {**self.settings, **overrides}
        return Claude(found["model"], reasoning=found["reasoning"], guide=found["guide"],
                      compact_at=found["compact_at"], command=[sys.executable, str(FAKE)],
                      env={"FAKE_CLAUDE_DIR": str(self.home), "FAKE_CLAUDE_PLAN": str(self.plan_file), **self.env})

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


def pulled(record: dict) -> list[dict]:
    """The calls of `next_frame` a process made, in order."""
    return [call for call in record["calls"] if call["tool"] == "next_frame"]


def claude_run(monkeypatch, fake: Fake, *argv) -> int:
    """`stbench run ARGV` with the Claude transport replaced by one over `fake`, built with what the driver hands it
    (the model, the effort, a memory's guide and a compaction limit, as `cmd_claude.make_claude` is)."""
    monkeypatch.setattr(cmd_claude, "make_claude", lambda model, reasoning, **given: fake.new(
        model=model, reasoning=reasoning, **given))
    return cli.main(["run", *map(str, argv)])
