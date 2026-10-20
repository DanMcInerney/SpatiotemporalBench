"""The run driver of the Codex reference agent: what `stbench run codex:<model>` calls (`cmd_run.AGENTS`).

`run(model, case, args) -> CaseRun` runs `codex_system.CodexSystem` over one case through `stream.run_case`, with the
transport `make_codex` builds (`codex` on PATH, a ChatGPT login: the plan's quota, no dollars). It sits on the harness
side because it calls `run_case`; the agent side (`codex_*`, `llm_*`) reaches none of that. The model id arrives from
the command line and goes nowhere else.

`args.text_only` (absent or false for a run that sends the camera images, the default) is the zero-shot check of what
the questions give away without the pictures: the run sends no image at all, the prompt says so (`llm_render`), and the
journal's `run` event and `codex.summary.json` carry `"text_only": true`, so such a run is never read as a camera one.
It is an option and not a channel: the stream is `rgb_pose`'s.

`drive` is the driver both CLI agents share: an `Agent` names the provider (`codex` here, `claude` in `cmd_claude`), its
two files (`<name>.calls.jsonl`, `<name>.summary.json`) and its `CodexSystem` class, and `refuse` is the check of what
cannot start. Everything below is said of the Codex agent; the Claude Code agent's journal and summary are the same
files under its own name, with `provider` and the CLI's version under `claude` (docs/claude.md).

`args.memory` (absent or None for the reference agent) is the path of a memory SPEC (`codex_memory`): the agent then
keeps that memory beside its context, on `rgb_pose` with the images only, and every message it is sent stays the
reference agent's (its guide is in the base instructions). The journal's `run` event records the memory (its name,
SPEC's argvs as loaded and the guide's text) and `codex.summary.json` names it, as `results.json` does (`memory`), so a
row with a memory is never read as the reference agent's; the case's store is kept beside the other files as
`<name>-store`. A SPEC that cannot be loaded is refused before anything is sent. Both CLI agents take a memory
(`cmd_run.MEMORY_AGENTS`); the guide goes where the CLI keeps its instructions (Codex's base instructions, Claude Code's
system prompt), and no message the model is sent changes.

`args.compact_at` (absent or None for the reference agent) is a token count (`--compact-at TOKENS`): every exec of the
case is run with it as Codex's `model_auto_compact_token_limit`, so the thread is compacted earlier than Codex's own
limit (`codex_exec.Codex`; for Claude Code the same lever is its two auto-compaction variables, `claude_exec.Claude`).
The journal's `run` event and `codex.summary.json` carry `compact_at` (null when not set), as `results.json` does when
it is set, so a row with a limit is never read as the reference agent's. Both CLI agents take one
(`cmd_run.COMPACT_AGENTS`).

Before anything is sent, `run` refuses (`cmd_run.Refused`, exit 2) a channel the agent cannot read (it reads the
camera images of `rgb_pose`, or the detections of `detections`), `text_only` on `detections` (which has no image to
withhold), a `--out` that already holds a Codex journal (a Codex call cannot be replayed, so a run is never repeated
into the same folder), a transport that cannot run: no `codex`, an unlisted model, a reasoning effort the model's
catalog entry does not list (`Codex.prepare`), and a case with a question the run asks of an answer type its job does
not ask (the system prompt shows the shapes of the job's types only: `brief.answer_types`, and its credit section the
rules of those types and of the job's families only: `brief.asked_families`; such a case is from another version of the
job). It creates nothing before those checks pass.

Beside the `answers.jsonl`, `rows.jsonl` and `summary.json` that `stbench run` writes, a run leaves in `--out`:
- `codex.calls.jsonl`, the journal (`codex_log`): a `run` event (model, effort, compaction limit, Codex's version, the
  case, channel, brief, the call bound and resume limit, the timeouts, the tools), the events of the case
  (`exec_started`, an item `served` to the model, each `call` it made, `answer_outcome` and `exec_failed`) and one
  record per `codex exec`. It holds the questions' text, so it never goes under the dataset;
- `codex.summary.json`, written even when the run fails: `llm_summary`'s counts of the journal (execs by kind, frames
  served, pulls, resumes, the per-frame latency, tokens with their cached share and whether the count is a lower bound,
  tool calls by name, retries, warnings, answers by outcome, failed execs by reason) beside the model, the effort,
  Codex's version, the case's status, failures by reason and wall time per message (`CaseRun`), the leak scan's hits
  and `invalid`.

After the run the leak scan (`llm_scan`) reads the conversations rebuilt from the journal. A hit, or an exec that
used a tool outside the server's (`escaped`), makes the run invalid: the summary says `"invalid": true` and `run`
raises `RunInvalid`, so `stbench run` exits 1 and writes no answers; the journal stays, with the summary, since its
`exec_failed` record names the tool that escaped. A Codex failure (a usage limit, a timeout, an auth failure, a full
context window, a pull that did not come in time, a model that stopped pulling) ends the case as the transport failure
it is (`CodexSystem`), and the summary counts it. A run in which the model made no call and no exec finished, and that
is not invalid, leaves no journal behind, so that it can be run again in the same folder.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import time
from dataclasses import dataclass

from . import job_registry, llm_render, suite
from .agent_tools import TOOLS
from .brief import (
    answer_types,
    asked_families,
    asks_during,
    asks_end,
    asks_headings,
    asks_lists,
    asks_turns,
    list_hazards,
)
from .case import CasePaths
from .cmd_run import Refused, RunInvalid, load_memory
from .codex_exec import Codex, CodexError
from .codex_log import Journal, read_log
from .codex_memory import Memory
from .codex_system import DEFAULTS, CodexSystem
from .job_registry import JobSpec
from .llm_scan import scan_log
from .llm_summary import format_summary, summarize
from .stream import CaseRun, run_case
from .stream_protocol import TIMEOUTS
from .system import timeouts_of
from .timeline import load

DEFAULT_REASONING = "low"  # the effort the reference rows run at
INVALID_FAILURES = ("escaped",)  # an exec failure that invalidates the run: a tool outside the server's was used


@dataclass(frozen=True)
class Agent:
    """A CLI agent this driver runs (`cmd_claude` has the other): `name` is its provider and the stem of its two files,
    `label` how a refusal names it, `system` its `CodexSystem` class and `cost_note`, for an agent whose CLI prints
    API-equivalent dollars, what its summary says about them (None for a plan whose quota is not dollars)."""

    name: str
    label: str
    system: type[CodexSystem]
    cost_note: str | None = None

    @property
    def journal(self) -> str:
        return f"{self.name}.calls.jsonl"

    @property
    def summary(self) -> str:
        return f"{self.name}.summary.json"


CODEX = Agent("codex", "Codex", CodexSystem)
JOURNAL, SUMMARY = CODEX.journal, CODEX.summary


def make_codex(model: str, reasoning: str, guide: str | None = None, compact_at: int | None = None) -> Codex:
    """The transport of a run: `codex` on PATH, with a memory's `guide` in its base instructions and a compaction limit
    `compact_at` when the run has them. The one seam to replace (tests give a `Codex` over the fake command)."""
    return Codex(model, reasoning=reasoning, guide=guide, compact_at=compact_at)


def _text_only(args: argparse.Namespace) -> bool:
    """Whether the run withholds every image (`args.text_only`; a namespace without the key is a camera run)."""
    return bool(getattr(args, "text_only", False))


def refuse(agent: Agent, args: argparse.Namespace) -> None:
    """Refused for a channel the agent cannot read, the text-only option on a channel with no image to withhold, and an
    `--out` that holds the agent's journal (a CLI call cannot be replayed). Both CLI agents are asked this first."""
    if args.channel not in agent.system.channels:
        raise Refused(f"the {agent.label} agent reads camera images or detections and runs on the "
                      f"{' and '.join(agent.system.channels)} channels only, not {args.channel!r}")
    if _text_only(args) and args.channel != "rgb_pose":
        raise Refused(f"text_only withholds the camera images of rgb_pose, and {args.channel!r} sends none")
    if (args.out / agent.journal).exists():
        raise Refused(f"{args.out / agent.journal} exists, the record of an earlier run: a {agent.label} run cannot "
                      "be replayed, so choose another --out (or remove the folder)")


def transport(agent: Agent, make, model: str, args: argparse.Namespace, memory: Memory | None = None):
    """(the prepared transport `make` builds for `agent`, the effort), for either CLI agent. Refused as `refuse` says,
    and for a transport that cannot run (`CodexMissing`: no CLI, a model or an effort it does not list, a limit it
    cannot be given). A memory's `guide` and `args.compact_at` (absent or None: the CLI's own limit) go to `make`."""
    refuse(agent, args)
    reasoning = args.reasoning or DEFAULT_REASONING
    given = {**({"guide": memory.guide} if memory else {}),
             **({"compact_at": args.compact_at} if getattr(args, "compact_at", None) is not None else {})}
    try:
        cli = make(model, reasoning, **given)
        cli.prepare()
    except (CodexError, ValueError) as error:
        raise Refused(str(error)) from None
    return cli, reasoning


def _transport(model: str, args: argparse.Namespace, memory: Memory | None = None) -> tuple[Codex, str]:
    """`transport` for Codex: a memory's guide goes into the base instructions, the limit into every exec's argv."""
    return transport(CODEX, make_codex, model, args, memory)


def run(model: str, case: CasePaths, args: argparse.Namespace) -> CaseRun:
    """Run the reference agent on `codex:<model>` over `case` and return the harness's `CaseRun`. `args` is
    `stbench run`'s namespace: `out` (an existing folder), `channel`, `brief`, `during` (`on` or `off`), `reasoning`
    (or None, for the default) and `timeout` (or None, for each message's own, `stream_protocol.TIMEOUTS`). Raises
    `Refused` before anything is sent or written, `RunInvalid` for a run that cannot be scored (its files are written to
    inspect), and what `run_case` raises for a case that does not fit its records."""
    memory = load_memory(args)
    codex, reasoning = _transport(model, args, memory)
    try:
        return drive(CODEX, model, case, args, codex, reasoning, memory)
    finally:
        codex.close()


def _kinds(job: JobSpec, case: CasePaths, during: bool) -> tuple[str, ...]:
    """The answer types the system prompt shows (`brief.answer_types`, the job's and not the arm's, so both arms of the
    leak check get the same bytes). `Refused` when a question the run asks (the during ones only on the arm that asks
    them, `during`) has another type: the prompt would never have shown the agent that type's form, and a case so built
    is from another version of the job."""
    kinds = answer_types(job)
    shown = set(llm_render.listed(kinds))
    asked = [q for q in suite.read_jsonl(case.questions) if during or q["when"] == "end"]
    if unshown := sorted({q["answer_type"] for q in asked} - shown):
        raise Refused(f"the case {case.id} holds questions of answer type {', '.join(unshown)}, which the job {job.id} "
                      f"does not ask (it asks {', '.join(kinds)}): the system prompt would not show their form. The "
                      "dataset was compiled from another version of the job, so compile it again")
    return kinds


def drive(agent: Agent, model: str, case: CasePaths, args: argparse.Namespace, transport, reasoning: str,
          memory: Memory | None = None) -> CaseRun:
    """Run `agent` (its `system` over `transport`, the prepared `Codex` or `Claude`) on `case` with the model's id
    `model` and `reasoning`, and write its journal and summary. `memory` (the Codex agent's, from `--memory`) is kept
    beside the agent's context. `run` and `cmd_claude.run` call it; what they raise, this does."""
    job = job_registry.get(load(case.script.timeline).episode.job)
    kinds = _kinds(job, case, args.during == "on")
    job_asks = asks_during(job)  # the job's (its decisions and looks), not the arm's
    text_only = _text_only(args)
    journal = Journal(args.out / agent.journal, provider=agent.name)
    timeouts = timeouts_of(args.timeout) if args.timeout else dict(TIMEOUTS)
    journal.note({"event": "run", "model": model, "reasoning": reasoning, "provider": agent.name,
                  **({"compact_at": transport.compact_at} if hasattr(transport, "compact_at") else {}),
                  agent.name: transport.version, "case": case.id, "channel": args.channel, "text_only": text_only,
                  "brief": args.brief, "during": args.during == "on",
                  "settings": dataclasses.asdict(DEFAULTS), "timeouts": timeouts,
                  "tools": [t["function"]["name"] for t in TOOLS],
                  **({"memory": memory.record()} if memory else {})})
    began = time.monotonic()
    try:
        kept = {"memory": memory, "keep": args.out / f"{memory.name}-store"} if memory else {}
        run = run_case(lambda state, work: agent.system(state, work, codex=transport, journal=journal,
                                                        timeouts=timeouts, asks_during=job_asks, asks_end=asks_end(job),
                                                        match_m=job.match_m, clearance_m=job.clearance_m,
                                                        reacquire_m=job.reacquire_m, lists=asks_lists(job),
                                                        headings=asks_headings(job), hazards=list_hazards(job),
                                                        kinds=kinds, families=asked_families(job),
                                                        turns=asks_turns(job), text_only=text_only, **kept),
                       case, channel=args.channel, brief=args.brief, during=args.during == "on")
    except BaseException as error:
        _unran(journal)
        if journal.path.exists():  # something ran: the summary says what stopped the run
            with contextlib.suppress(Exception):
                _summarise(agent, model, case, args, transport, reasoning, journal, None, time.monotonic() - began,
                           f"error: {type(error).__name__}: {error}", memory)
        raise
    summary = _summarise(agent, model, case, args, transport, reasoning, journal, run, time.monotonic() - began,
                         run.status, memory)
    if not summary["invalid"]:  # an invalid run keeps its journal: `exec_failed` names the tool that escaped
        _unran(journal)
    for line in format_summary(summary):
        print(line)
    if summary["text_only"]:
        print("text-only run: no image was sent to the model")
    if summary["invalid"]:
        why = (f"the leak scan found {len(summary['scan'])} hit(s) in what the harness sent to the model, e.g. "
               f"{summary['scan'][0]}" if summary["scan"] else "an exec used a tool outside the server's")
        raise RunInvalid(f"{why}: the run is invalid (its journal and summary are kept to inspect: {args.out})")
    return run


def _unran(journal: Journal) -> None:
    """A journal in which the model made no call and no exec finished is removed: nothing was spent, and it must not
    block a rerun. A run that is invalid never comes here: its journal is the evidence of why."""
    if journal.spent == 0:
        journal.path.unlink(missing_ok=True)


def _summarise(agent: Agent, model: str, case: CasePaths, args: argparse.Namespace, transport, reasoning: str,
               journal: Journal, run: CaseRun | None, wall_s: float, status: str, memory: Memory | None = None) -> dict:
    """Write the agent's summary file (`codex.summary.json`) from the journal and the run (None when `run_case` raised)
    and return it."""
    questions = suite.read_jsonl(case.questions)
    hits = scan_log(journal.path, [load(case.script.timeline).episode.map], questions=questions, episode=case.id,
                    dataset=case.dataset)
    summary = summarize(
        journal.path, model=model, reasoning=reasoning, provider=agent.name, case=case.id, script=case.script.id,
        **{agent.name: transport.version}, channel=args.channel, brief=args.brief, during=args.during == "on",
        status=status,
        failures=run.failures if run else {}, wall_s=run.wall_s if run else {}, run_s=round(wall_s, 1), scan=hits)
    summary["text_only"] = _text_only(args)  # the stamp: a text-only run is never read as a camera one
    if hasattr(transport, "compact_at"):   # likewise a run with a compaction limit (null: Codex's own)
        summary["compact_at"] = transport.compact_at
    if memory is not None:
        summary["memory"] = memory.name  # likewise a run with a memory is never read as the reference agent's
    if agent.cost_note:   # the CLI's own dollars, as the turns carry them (the harness prices the tokens itself)
        turns = read_log(journal.path) if journal.path.exists() else []
        summary |= {"cost_usd": round(sum(r["cost"] for r in turns if "event" not in r), 6),
                    "cost_note": agent.cost_note}
    summary["invalid"] = bool(hits) or any(reason in summary["exec_failures"] for reason in INVALID_FAILURES)
    (args.out / agent.summary).write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    return summary
