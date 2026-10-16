"""The run driver of the Claude Code reference agent: what `stbench run claude:<model>` calls (`cmd_run.AGENTS`).

`run(model, case, args) -> CaseRun` is the Codex agent's driver (`cmd_codex.drive`) over `claude_system.ClaudeSystem`,
with the transport `make_claude` builds (`claude` on PATH, the user's own Claude Code login: the subscription's quota or
the account's key, whichever that login uses). `<model>` is `haiku`, `sonnet`, `opus` or a full `claude-...` id
(`claude_exec.model_id`); the id is what the journal, the summary and the price table carry. `--reasoning` is Claude
Code's `--effort` (low, medium, high, xhigh or max; `low` by default, as for Codex). The case is one `claude -p`
process whose argv and isolation are `claude_exec`'s.

Everything `cmd_codex` says of a run holds here, with the files under the agent's own name: a refusal before anything is
sent (a channel the agent cannot read, `text_only` on `detections`, an `--out` that holds a journal, no `claude`, a
model or effort that is none, a case with a question of an answer type its job does not ask), `claude.calls.jsonl` (the
journal: a `run` event, the events of the case, one record per turn) and `claude.summary.json` (written even when the
run fails: `llm_summary`'s counts beside the model id, the effort, `provider` `claude`, the CLI's version under
`claude`, the case's status and the leak scan's hits), the invalid run that keeps its journal and the run that spent
nothing that does not. The summary adds `cost_usd` and `cost_note`: the sum of the CLI's own `total_cost_usd` of each
turn, which is what the tokens would cost on the API; a subscription pays by quota and not by these. The harness's
`summary.json` keeps pricing the tokens itself (`prices.cost_usd`, None for a model the table lacks).
"""

from __future__ import annotations

import argparse

from . import cmd_codex
from .case import CasePaths
from .claude_exec import Claude
from .claude_system import ClaudeSystem
from .cmd_run import Refused
from .codex_exec import CodexError
from .stream import CaseRun

COST_NOTE = ("API-equivalent dollars, the CLI's total_cost_usd summed over the turns; a subscription pays by quota, "
             "not these")
CLAUDE = cmd_codex.Agent("claude", "Claude Code", ClaudeSystem, COST_NOTE)


def make_claude(model: str, reasoning: str) -> Claude:
    """The transport of a run: `claude` on PATH. The one seam to replace (tests give a `Claude` over a fake one)."""
    return Claude(model, reasoning=reasoning)


def _transport(model: str, args: argparse.Namespace) -> tuple[Claude, str]:
    """(the prepared transport, the effort). Refused as `cmd_codex.refuse` says, for a model or effort that is none and
    for a transport that cannot run."""
    cmd_codex.refuse(CLAUDE, args)
    reasoning = args.reasoning or cmd_codex.DEFAULT_REASONING
    try:
        claude = make_claude(model, reasoning)
        claude.prepare()
    except (CodexError, ValueError) as error:   # CodexMissing: no claude, a name that is no model, an effort it lacks
        raise Refused(str(error)) from None
    return claude, reasoning


def run(model: str, case: CasePaths, args: argparse.Namespace) -> CaseRun:
    """Run the reference agent on `claude:<model>` over `case` and return the harness's `CaseRun`. `args` is
    `stbench run`'s namespace, as for `cmd_codex.run`. Raises `Refused` before anything is sent or written,
    `RunInvalid` for a run that cannot be scored, and what `run_case` raises for a case that does not fit its
    records."""
    claude, reasoning = _transport(model, args)
    try:
        return cmd_codex.drive(CLAUDE, claude.model, case, args, claude, reasoning)
    finally:
        claude.close()
