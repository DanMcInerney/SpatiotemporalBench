"""`stbench run`: one system over one case, through the stream protocol.

    stbench run SYSTEM CASE_DIR [--channel C] [--brief V] [--during on|off] [--local] [--timeout S] [--reasoning E]
                [--out DIR]

CASE_DIR is `DATASET/private/<cid>`. SYSTEM is one of:
- a baseline name (`baselines.BASELINES`), built with the options `suite_priors.options_for` gives it over the dataset's
  cases of the case's layout and job: one whose record sets `prior_nm` (its priors carry another system's answers, which
  only the first wave of `stbench check` or `stbench eval` makes) is refused;
- `FILE.py:Class`, an external system: in the Docker sandbox (the verified mode), or with `--local` as a plain process
  through the same shim, stamped `sandbox: none` and not a verified score (docs/agent.md);
- `<agent>:<model>`, a reference agent from `AGENTS` (one registry: each agent is a dotted path to a runner,
  `run(model, case, args) -> CaseRun`, imported when it is named). `--reasoning` and `--text-only` go to an agent and
  are refused for the others; `--local` is refused for anything but `FILE.py:Class`. An agent's runner raises
  `Refused` for a run that must not start, which exits 2 with its message; `RunInvalid` for a run that finished but
  cannot be scored (the files it wrote to be inspected stay), which exits 1 and writes no answers; any other
  exception it raises exits 1. A runner finds `args.out` already set (the folder it may write its own files to, made
  before it is called) and writes nothing else of `stbench run`'s.

It writes under `--out` (default `DATASET/runs/<system>-<case id>/`, with `-text-only` added when set, never under the
dataset's `scripts/` or `private/`)
`answers.jsonl` (`{"qid", "answer"}` in the order of the case's questions: a question the harness could not get an
answer for holds `stream_protocol.failed(reason)`), `rows.jsonl` (one scored row a gold: `suite.score_rows`) and
`summary.json` (what was run, how it ended, wall time and state size, usage as the system reported it, whether that
names a remote store, what the tokens cost by the price table (never a figure of the system's own) and why not when
they cannot be priced, where it ran, and `scores`: the headline, the splits and the outcome counts of the rows). The
brief variant and the during flag are options of the run, never part of the case; with `--during off` the during
questions were not asked and have no row. A run scores what it ran, so a case with no
`gold.jsonl` is refused. Exit 0 when the case completed, 1 when it ended early, 2 when the run was refused before
anything was sent.

`stbench eval` runs many cases through the same pieces: `classify` and `resolve` say what SYSTEM is and build its
runner, `write_run` writes each case's three files, and `refuse_truth_folder` keeps an output folder out of the truth.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import ability_counts, local_system, prices, sandbox, score_stats, suite, suite_priors
from . import episode as ep
from .baselines import BASELINES
from .brief import VARIANTS
from .case import PRIVATE, SCRIPTS, CasePaths
from .local_system import LocalCommandSystem
from .safecopy import UnsafeTree
from .stream import CaseRun, run_case
from .stream_protocol import CHANNELS, DEFAULT_CHANNEL, System, remote_memory
from .system import MIN_TIMEOUT, CommandSystem, timeouts_of

# agent -> the dotted path of its runner (see the docstring)
AGENTS = {"codex": "stbench.cmd_codex.run", "claude": "stbench.cmd_claude.run"}
TEXT_ONLY_AGENTS = frozenset({"codex", "claude"})   # the agents that can withhold every image (`--text-only`)
FILE_SYSTEM = re.compile(r"(?P<file>.+\.py):(?P<cls>[A-Za-z_]\w*)")   # the last `:` splits, so C:\x\sys.py:Echo works
AGENT_SYSTEM = re.compile(r"(?P<agent>[A-Za-z][\w-]*):(?P<model>.+)")
UNSANDBOXED = "run without the Docker sandbox: not a verified score"
BASELINE, EXTERNAL, AGENT = "baseline", "external", "agent"   # the three kinds of SYSTEM

Runner = Callable[[CasePaths, argparse.Namespace], CaseRun]


class Refused(Exception):
    """A run that must not start (no `codex` or `claude`, not logged in, an output folder in use): `stbench run` prints
    the message and exits 2. An agent's runner raises it; any other exception a runner raises exits 1."""


class RunInvalid(RuntimeError):
    """A run that finished but cannot be scored (a leak-scan hit, a tool outside the allowed set). An agent's runner
    raises it after writing the files that say why; `stbench run` exits 1 and writes no answers, `stbench eval` marks
    the case invalid."""


def seconds(text: str) -> float:
    """A `--timeout` value (`stbench eval` takes it too)."""
    value = float(text)
    if not (math.isfinite(value) and value >= MIN_TIMEOUT):  # a container takes about a second to start
        raise argparse.ArgumentTypeError(f"{text!r} must be a finite number of seconds, at least {MIN_TIMEOUT:g}")
    return value


def safe_name(text: str) -> str:
    """`text` with every character outside [A-Za-z0-9._-] turned into `_`: a system's name as a folder name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def refuse_truth_folder(out: Path, dataset: Path) -> Path:
    """`out` resolved; Refused when it is under the dataset's `scripts/` or `private/`."""
    out = Path(out).resolve()
    for folder in (SCRIPTS, PRIVATE):
        if out.is_relative_to((Path(dataset) / folder).resolve()):
            raise Refused(f"the output folder {out} is under the dataset's {folder}/, which a run never writes to")
    return out


def _out_dir(args: argparse.Namespace, case: CasePaths) -> Path:
    """`--out`, or `DATASET/runs/<system>-<case id>` with the system as `safe_name`; Refused under the dataset's
    `scripts/` or `private/`. A `--text-only` run adds `-text-only`, so it cannot land in a camera run's folder."""
    suffix = "-text-only" if getattr(args, "text_only", False) else ""
    return refuse_truth_folder(args.out or case.dataset / "runs" / f"{safe_name(args.system)}-{case.id}{suffix}",
                               case.dataset)


def _streaming(make: Callable[[Path, Path], System]) -> Runner:
    """The runner of a system that `stream.run_case` builds and streams to."""
    return lambda case, args: run_case(make, case, channel=args.channel, brief=args.brief, during=args.during == "on")


def _baseline(args: argparse.Namespace, case: CasePaths) -> Runner:
    if BASELINES[args.system].prior_nm:
        raise Refused(f"{args.system}'s priors carry another system's answers (prior_nm), which only the first wave of "
                      "`stbench check` or `stbench eval` makes: run it with one of them")
    ro = suite_priors.options_for(args.system, case, suite.cases_like(case), {})
    return lambda c, a: suite.run_baseline(a.system, c, a.channel, ro, brief=a.brief, during=a.during == "on")


def _external(args: argparse.Namespace, case: CasePaths, found: re.Match, stack: contextlib.ExitStack) -> Runner:
    """FILE.py:Class as a system: refused unless it can run here, then built in Docker, or locally with `--local`."""
    file, cls = Path(found["file"]), found["cls"]
    kw = {"timeouts": timeouts_of(args.timeout)} if args.timeout else {}
    try:
        if args.local:
            local_system.check(file, cls, case.dataset)
            print(f"local: {file.name}:{cls} as a subprocess; {UNSANDBOXED}")
            return _streaming(lambda state, work: LocalCommandSystem(state, work, file, cls,
                                                                     dataset=case.dataset, **kw))
        sandbox.require()  # fail fast, before anything is copied
        program = stack.enter_context(sandbox.program(file, cls, deny=[case.dataset]))
        # the run folders are made under the temp root: a folder in it stands for them
        sandbox.check_sources(program, Path(tempfile.gettempdir()) / "stbench-case")
    except (sandbox.SandboxUnavailable, UnsafeTree, ValueError) as e:
        raise Refused(str(e)) from None
    print(f"sandbox: {sandbox.IMAGE}")
    return _streaming(lambda state, work: CommandSystem(state, work, program, **kw))


def _agent(name: str, model: str) -> Runner:
    """The agent's runner with `model` bound, imported now (a module that does not exist yet is a refusal); the folder
    it writes to is made before it is called."""
    module, _, function = AGENTS[name].rpartition(".")
    try:
        runner = getattr(importlib.import_module(module), function)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        raise Refused(f"the agent {name!r} is not available: {module} does not exist") from None

    def run(case: CasePaths, args: argparse.Namespace) -> CaseRun:
        made = not args.out.exists()
        args.out.mkdir(parents=True, exist_ok=True)
        try:
            return runner(model, case, args)
        except Refused:
            if made:
                with contextlib.suppress(OSError):
                    args.out.rmdir()  # a refusal leaves nothing behind (rmdir takes only an empty folder)
            raise

    return run


def classify(args: argparse.Namespace) -> tuple[str, re.Match | None]:
    """(the kind of `args.system`: BASELINE, EXTERNAL or AGENT, the match of the last two). Refused for a SYSTEM that is
    none of the three kinds and for an option it cannot take (`args.local`, `args.reasoning` and `args.text_only`)."""
    external = FILE_SYSTEM.fullmatch(args.system)
    agent = None if external else AGENT_SYSTEM.fullmatch(args.system)
    if agent and agent["agent"] not in AGENTS:
        agent = None
    if not (external or agent or args.system in BASELINES):
        raise Refused(f"SYSTEM must be a baseline ({', '.join(sorted(BASELINES))}), FILE.py:Class or <agent>:<model> "
                      f"(the agents: {', '.join(sorted(AGENTS))}), not {args.system!r}")
    kind = EXTERNAL if external else AGENT if agent else BASELINE
    if args.local and kind != EXTERNAL:
        raise Refused(f"--local applies to a FILE.py:Class SYSTEM, not {args.system!r}")
    if args.reasoning and kind != AGENT:
        raise Refused(f"--reasoning applies to an agent (<agent>:<model>), not {args.system!r}")
    if getattr(args, "text_only", False):
        if not (agent and agent["agent"] in TEXT_ONLY_AGENTS):
            raise Refused(f"--text-only applies to {' or '.join(sorted(TEXT_ONLY_AGENTS))} agents "
                          f"(<agent>:<model>), not {args.system!r}")
        if "rgb" not in CHANNELS[getattr(args, "channel", DEFAULT_CHANNEL)]:
            raise Refused(f"--text-only withholds the image, and the {args.channel} channel sends none")
    return kind, external or agent


def sandbox_of(kind: str, local: bool) -> str:
    """Where a system of this kind runs, for a summary's `sandbox`: `docker`, `none` (an external system with `--local`)
    or `in-process`."""
    return ("none" if local else "docker") if kind == EXTERNAL else "in-process"


def resolve(args: argparse.Namespace, case: CasePaths, stack: contextlib.ExitStack) -> tuple[str, Runner]:
    """(where the system runs, for the summary's `sandbox`; the runner of one case at a time). Refused as `classify`
    says. An external system's sandbox program lives as long as `stack`; `case` names the dataset it stays out of."""
    kind, found = classify(args)
    where = sandbox_of(kind, args.local)
    if kind == EXTERNAL:
        return where, _external(args, case, found, stack)
    if kind == AGENT:
        return where, _agent(found["agent"], found["model"])
    return where, _baseline(args, case)


def write_run(out: Path, case: CasePaths, args: argparse.Namespace, run: CaseRun, where: str) -> dict:
    """`answers.jsonl` in the case's question order, `rows.jsonl` and `summary.json` under `out`; the scores block it
    returns. `args` names `system`, `channel`, `brief` and `during` ("on" or "off"). The answers and the summary are
    kept when the scoring raises (a gold the scorer rejects)."""
    out.mkdir(parents=True, exist_ok=True)
    order = [q["qid"] for q in suite.read_jsonl(case.questions)]
    ep.write_jsonl(out / "answers.jsonl", [{"qid": qid, "answer": run.answers[qid]} for qid in order
                                           if qid in run.answers])
    summary = {"case": case.id, "script": case.script.id, "system": args.system, "channel": args.channel,
               "brief": args.brief, "during": args.during == "on", "status": run.status, "failures": run.failures,
               "wall_s": run.wall_s, "state_bytes": run.state_bytes, "usage": run.usage,
               "remote_memory": remote_memory(run.usage), "cost_usd": prices.cost_usd(run.usage),
               "cost_note": prices.cost_note(run.usage), "session": run.session, "sandbox": where,
               "text_only": bool(getattr(args, "text_only", False))}
    try:
        rows = suite.score_rows(case, run, during=summary["during"])
        if summary["text_only"]:   # a row of a text-only run says so; the rows of a camera run gain no key
            rows = [{**row, "text_only": True} for row in rows]
        ep.write_jsonl(out / "rows.jsonl", rows)
        thin, relaxed = ability_counts.thin_rule(case.dataset)   # the under-20 rule is the whole dataset's
        report = score_stats.aggregate(rows, thin=thin, thin_ignored=relaxed)
        summary["scores"] = {key: report[key] for key in ("headline", "columns", "by", "counts")}
    finally:
        (out / "summary.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    return summary["scores"]


def _run(args: argparse.Namespace) -> int:
    try:
        case = CasePaths.from_dir(args.case_dir)
        args.out = _out_dir(args, case)
        if not case.gold.is_file():  # a run scores what it ran: no gold, no run (an agent's quota is not spent for it)
            raise Refused(f"case {case.id} has no {case.gold.name}: `stbench run` scores what it runs, so compile the "
                          "case first")
        with contextlib.ExitStack() as stack:
            where, runner = resolve(args, case, stack)
            run = runner(case, args)
    except (Refused, ValueError, OSError) as e:  # refused before anything was sent (`run_case` raises ValueError)
        print(f"stbench run: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001  (a runner that failed: its message, and exit 1)
        print(f"stbench run: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    try:
        scores = write_run(args.out, case, args, run, where)
    except ValueError as e:  # the answers and the summary are written; the gold is what the scorer cannot read
        print(f"stbench run: the run could not be scored ({e}); its answers are in {args.out}", file=sys.stderr)
        return 1
    print(f"{run.status}: {len(run.answers)} answers -> {args.out}")
    mean, n = scores["headline"]["mean"], sum(c["n"] for c in scores["columns"].values())
    asked = sum(scores["counts"].values())
    print("scores: no question scored" if not asked else
          f"scores: every one of {asked} questions is a diagnostic here, so there is no headline" if mean is None
          else f"scores: headline {mean:.3f} over {n} questions (now and memory equal, abilities weighted within each)")
    return 0 if run.status == "complete" else 1


def register(sub) -> None:
    r = sub.add_parser("run", help="run a system (a baseline, FILE.py:Class in the Docker sandbox or with --local, or "
                                   "<agent>:<model>) over one case through the stream protocol")
    r.add_argument("system", help="a baseline name; FILE.py:Class, an external system run in the Docker sandbox (or "
                                  "unsandboxed with --local); or <agent>:<model>, such as codex:<model id> or "
                                  "claude:sonnet")
    r.add_argument("case_dir", type=Path, help="DATASET/private/<case id>")
    r.add_argument("--channel", choices=sorted(CHANNELS), default=DEFAULT_CHANNEL,
                   help="rgb_pose (the camera, the pose and the status: the agents' channel, no depth) or "
                        "detections (the gate channel: depth, pose and perfect detections, never an agent's); "
                        "default %(default)s")
    r.add_argument("--brief", choices=VARIANTS, default="correct", help="the brief variant (default %(default)s)")
    r.add_argument("--during", choices=("on", "off"), default="on",
                   help="whether the during-job questions are asked (default %(default)s)")
    r.add_argument("--local", action="store_true",
                   help="FILE.py:Class only: run the system as a plain local subprocess, WITHOUT the Docker sandbox "
                        "(docs/agent.md); the summary says `sandbox: none` and the result is not a verified score")
    r.add_argument("--timeout", type=seconds, metavar="SECONDS",
                   help="seconds an external system may take over any one message, and an agent from one item it "
                        "is served to its next pull (default: each message's own)")
    r.add_argument("--reasoning", metavar="EFFORT", help="an agent's reasoning effort (refused for other systems)")
    r.add_argument("--text-only", action="store_true",
                   help="codex or claude agent only: send no image (zero-shot check)")
    r.add_argument("--out", type=Path, help="the folder for answers.jsonl, rows.jsonl and summary.json (default "
                                            "DATASET/runs/<system>-<case id>)")
    r.set_defaults(func=_run)
