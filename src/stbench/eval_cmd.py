"""`stbench eval`: one command to evaluate a system over a dataset's cases, and `eval compare` (docs/eval.md).

    stbench eval SYSTEM --data D [--out RUNDIR] [--cases ID,...] [--select KEY=VALUE ...] [--channel C] [--brief V]
                 [--during on|off] [--arms LEVER=V1,V2[,...]] [--workers N] [--local] [--timeout S] [--reasoning E]
                 [--text-only] [--cells-from DEVROOT]
                 [--max-usd TOTAL --case-usd X]
    stbench eval compare RESULTS_A RESULTS_B [...] [--out FILE.md] [--cells-from DEVROOT]
    stbench eval distance RUN [REFERENCE_RUN ...] --data D [--out FILE.md] [--cells-from DEVROOT]

SYSTEM is a baseline, `FILE.py:Class` (an external system: the Docker sandbox, or with `--local` a plain subprocess,
unsandboxed and not a verified score) or `<agent>:<model>` through `cmd_run.AGENTS` (`--reasoning` goes to an agent),
as for `stbench run`. The cases are the dataset's compiled cases that ask a question, narrowed by `--cases` and by
`--select` on `layout`, `robot`, `job` and `length`. Each is run through `stream.run_case` (`--workers` at a time),
scored, and written under RUNDIR (by default `D/runs/<system>-<channel>`, with `-<brief>` and `-during-off` added when
they are not the defaults) as `cases/<cid>/` (`answers.jsonl`, `rows.jsonl`, `summary.json`, an agent's own files), and
RUNDIR gets `results.json` (`eval_results`) and `results.md` (`eval_report`), one aggregate per
layout, never pooled. Running the same command again skips the finished cases and runs the rest from their start (an
agent's call cannot be replayed); `results.json` is first written, with no case finished, before the first case runs,
so a run killed at any point still says what it was, and a rerun with another system, channel, brief, during flag,
sandbox, reasoning effort or text-only flag, or onto another compile, is refused, never relabelled.

`--arms LEVER=V1,V2` runs one lever's arms on every case: LEVER is `during` (`on`, `off`) or `brief` (`correct`,
`neutral`; `swapped` is retired and refused, as `--brief swapped` is), each value runs into `RUNDIR/<lever>-<value>` (a
run folder of its own, stamped with the value), a case's arms one after another in an order seeded by the case, and the
lever is refused as an option of its own.
`--max-usd TOTAL --case-usd X` caps the spend of an agent or an external system: a case starts only while what the
folders spent (the tokens each case reports times the price table, or X for a case that cannot be priced) plus what the
cases in flight reserve (X each) plus X stays within TOTAL.
`compare` lays results side by side, per layout, with paired differences from the first, and refuses results that
differ in more than one lever group (`eval_compare`).
`distance` reads a finished run folder and the dataset's gold and reports the offset questions' errors against the job's
clearance and by range band: how far off, in which direction, and (given reference runs, the scripted `kind_range` and
`all_unknown`) above the better of them per layout and job (`eval_distance`). It then prints the run's credit
sensitivity, the headline re-scored with the clearance C and the re-find radius R each scaled by 0.5 and 2
(`eval_sensitivity`); `--out` writes both.

`--cells-from DEVROOT` takes the cells of the headline, which a cell weighs (`ability_counts.weights`: a cell of
fewer than 20 questions on a job weighs n / 20), from DEVROOT's golds and not from DATA's own, so a subset (the fast
tier) and a fresh set (the held-out one) are scored on the cells of the release's dev root. Default: the dataset
itself.

Exit 0: every selected case finished (in every arm). Exit 1: some did not (ended, invalid, refused, the cap); the
results so far are written and marked partial. Exit 2: refused before anything ran (a bad option, no
Docker, results of another run or compile in RUNDIR, an agent that cannot run).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

from . import ability_counts, cmd_run
from . import eval_results as results
from . import eval_run as running
from . import eval_systems as systems
from .brief import VARIANTS
from .cmd_run import Refused
from .eval_compare import Refused as CompareRefused
from .eval_compare import compare
from .eval_distance import distance
from .eval_report import count, render
from .eval_sensitivity import sensitivity
from .eval_systems import Arm, Config
from .safecopy import UnsafeTree
from .sandbox import SandboxUnavailable
from .stream_protocol import CHANNELS, DEFAULT_CHANNEL


def _workers(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be a whole number, at least 1")
    return int(text)


def _usd(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"{text!r} must be a finite number of dollars above 0")
    return value


def _out(args: argparse.Namespace, lever: str | None, brief: str, during: bool) -> Path:
    """RUNDIR: `--out`, or `D/runs/<system>-<channel>` with `-<brief>` and `-during-off` when an option that is not
    the arms' lever is not its default. Refused under the dataset's `scripts/` or `private/`."""
    suffix = ("" if lever == "brief" or brief == "correct" else f"-{brief}") + (
        "" if lever == "during" or during else "-during-off")
    name = f"{cmd_run.safe_name(args.system)}-{args.channel}{suffix}" + ("-text-only" if args.text_only else "")
    return cmd_run.refuse_truth_folder(args.out or args.data / "runs" / name, args.data)


def _config(args: argparse.Namespace) -> Config:
    """The run the arguments ask for. Refused (`cmd_run.Refused`) for what cannot run, before anything is written."""
    kind, _ = cmd_run.classify(args)
    if args.data is None:
        raise Refused("--data DIR is required")
    if (args.max_usd is None) != (args.case_usd is None):
        raise Refused("--max-usd TOTAL and --case-usd X (what a case in flight reserves) go together")
    if args.max_usd is not None and kind == cmd_run.BASELINE:
        raise Refused(f"--max-usd caps what a system spends in tokens, and {args.system!r} spends none")
    lever, values = running.parse_arms(args.arms) if args.arms else (None, [])
    for name in ("brief", "during"):
        if lever == name and getattr(args, name) is not None:
            raise Refused(f"--arms {name}=... runs each value as an arm: do not give --{name} as well")
    brief, during = args.brief or "correct", args.during != "off"
    if args.cells_from is not None:
        ability_counts.require_cases(args.cells_from)   # ValueError: refused with the other bad options
    if problem := running.brief_problem(brief):
        raise Refused(problem)
    if problem := systems.channel_problem(args.system, kind, args.channel):
        raise Refused(problem)
    selected = running.select(args.data, args.cases.split(",") if args.cases else None,
                              running.parse_select(args.select))
    out = _out(args, lever, brief, during)
    return Config(args.system, kind, args.data, out, args.channel, selected,
                  running.make_arms(out, lever, values, brief, during), lever, args.workers, args.local, args.timeout,
                  args.reasoning, args.max_usd, args.case_usd, args.text_only, args.cells_from)


def _clear(cfg: Config) -> None:
    """Take back what a run that never started wrote: each arm's `results.json`, then its empty folders."""
    for arm in cfg.arms:
        (arm.path / results.RESULTS).unlink(missing_ok=True)
        for folder in (arm.path / systems.CASES, arm.path):
            with contextlib.suppress(OSError):
                folder.rmdir()
    with contextlib.suppress(OSError):
        cfg.out.rmdir()


def _write(cfg: Config, arm: Arm, run: results.Run) -> dict:
    """`results.json` and `results.md` of an arm's folder as it stands, shown; the document."""
    doc = results.build(cfg, arm, run)
    path = results.write(arm.path, doc)
    md = render(doc)
    (arm.path / "results.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    print(f"results -> {path} and {arm.path / 'results.md'}")
    return doc


def _run(cfg: Config) -> int:
    try:
        previous = {arm.path: results.load(arm.path) if (arm.path / results.RESULTS).exists() else None
                    for arm in cfg.arms}
        for arm in cfg.arms:
            results.check_resume(cfg, arm, previous[arm.path])
    except ValueError as error:   # results of another run, of another compile, or not results at all
        print(f"stbench eval: {error}", file=sys.stderr)
        return 2
    print(f"{cfg.system} over {count(len(cfg.cases), 'case')}, {len(cfg.arms)} run folder(s) under {cfg.out}")
    with contextlib.ExitStack() as stack:
        try:
            work = systems.prepare(cfg, stack)
        except (Refused, ValueError, SandboxUnavailable, UnsafeTree) as error:   # refused before any case
            print(f"stbench eval: {error}", file=sys.stderr)
            return 2
        run = results.Run()
        for arm in cfg.arms:
            if previous[arm.path] is None:   # from here a kill leaves the stamp for a rerun to check
                results.write(arm.path, results.build(cfg, arm, run))
        docs = []
        try:
            running.execute(cfg, work, run)
        finally:   # an interrupt or a crash still leaves the results of what was finished
            if run.fatal and not any(results.summary_of(arm, c.id) for arm in cfg.arms for c in cfg.cases) and all(
                    p is None for p in previous.values()):   # the system refused to run: leave nothing behind
                _clear(cfg)
            else:
                docs = [_write(cfg, arm, run) for arm in cfg.arms]
    if run.fatal:
        print(f"stbench eval: {run.fatal}", file=sys.stderr)
    return 2 if run.fatal else 1 if any(d["partial"] for d in docs) else 0


def _compare(args: argparse.Namespace) -> int:
    try:
        text = compare(args.results, args.out, args.cells_from)
    except CompareRefused as error:
        print(f"stbench eval compare: {error}", file=sys.stderr)
        return 2
    print(text, end="")
    return 0


def _distance(args: argparse.Namespace) -> int:
    if args.data is None or not args.results:
        print("stbench eval distance: takes RUN [REFERENCE_RUN ...] and --data D", file=sys.stderr)
        return 2
    run, *reference = map(Path, args.results)
    try:
        text = distance(run, args.data, reference or None) + "\n" + sensitivity(run, args.data,
                                                                               cells_from=args.cells_from)
    except (ValueError, OSError) as error:   # results that cannot be read, a gold that is missing
        print(f"stbench eval distance: {error}", file=sys.stderr)
        return 2
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0


def _eval(args: argparse.Namespace) -> int:
    if args.system == "compare":
        return _compare(args)
    if args.system == "distance":
        return _distance(args)
    try:
        if args.results:
            raise Refused(f"unexpected arguments {args.results}: only `eval compare` takes result files")
        cfg = _config(args)
    except (Refused, ValueError) as error:   # ValueError: a dataset whose index lacks an entry a case names
        print(f"stbench eval: {error}", file=sys.stderr)
        return 2
    return _run(cfg)


def register(sub) -> None:
    p = sub.add_parser("eval", help="evaluate a system (a baseline, FILE.py:Class or <agent>:<model>) over a dataset's "
                                    "cases, `eval compare` results, or `eval distance` of a run")
    p.add_argument("system", metavar="SYSTEM", help="a baseline name; FILE.py:Class (Docker, or --local); "
                                                    "<agent>:<model>, such as codex:<model id> or claude:sonnet; "
                                                    "`compare`; or `distance`")
    p.add_argument("results", nargs="*", metavar="RESULTS", help="compare: two or more results.json files or run "
                                                                 "folders (the first is the one differences are from); "
                                                                 "distance: a run folder, then optionally the "
                                                                 "references' (the scripted kind_range and "
                                                                 "all_unknown: the better one is used)")
    p.add_argument("--data", type=Path, help="the dataset folder (distance: where the golds are)")
    p.add_argument("--out", type=Path, help="run: the run folder (default DATA/runs/<system>-<channel>, with -<brief> "
                                            "and -during-off added when they are not the defaults); "
                                            "compare, distance: also write the report to this file")
    p.add_argument("--cases", metavar="ID,...", help="only these case ids (compiled cases that ask a question)")
    p.add_argument("--select", action="append", default=[], metavar="KEY=VALUE",
                   help="only cases whose layout, robot, job or length is VALUE (repeat a key for either value; "
                        "different keys must all hold)")
    p.add_argument("--channel", choices=sorted(CHANNELS), default=DEFAULT_CHANNEL,
                   help="rgb_pose (the camera, the pose and the status: the agents' channel, no depth) or "
                        "detections (the gate channel: depth, pose and perfect detections, never an agent's); "
                        "default %(default)s")
    p.add_argument("--brief", choices=VARIANTS, help="the brief variant: correct or neutral (default correct); swapped "
                                                     "is retired and refused")
    p.add_argument("--during", choices=("on", "off"), help="whether the during-job questions are asked (default on)")
    p.add_argument("--arms", metavar="LEVER=V1,V2", help="run each value of one lever (during=on,off or "
                                                         "brief=correct,neutral) on every case, each into "
                                                         "RUNDIR/<lever>-<value>")
    p.add_argument("--workers", type=_workers, default=1,
                   help="baselines: worker processes; agents and external systems: cases at a time, a thread each "
                        "(a codex: or claude: case is a CLI process: keep it at about 20 or fewer); "
                        "default %(default)d")
    p.add_argument("--local", action="store_true",
                   help="FILE.py:Class as a plain subprocess, without Docker: stamped sandbox none, not a verified "
                        "score")
    p.add_argument("--timeout", type=cmd_run.seconds, metavar="SECONDS",
                   help="an external system: seconds it may take over any one message; an agent: from one item it "
                        "is served to its next pull (default: each message's own)")
    p.add_argument("--reasoning", metavar="EFFORT", help="an agent's reasoning effort (refused for other systems)")
    p.add_argument("--text-only", action="store_true",
                   help="codex or claude agent only: send no image (zero-shot check)")
    p.add_argument("--cells-from", type=Path, metavar="DEVROOT",
                   help="the dataset whose golds count the cells, and weigh each (a cell of n < 20 questions on a job "
                        "weighs n / 20), instead of the dataset scored: the dev root, for a subset or a held-out set")
    p.add_argument("--max-usd", type=_usd, metavar="TOTAL",
                   help="the total spend cap over the run folders, finished cases included (needs --case-usd)")
    p.add_argument("--case-usd", type=_usd, metavar="USD",
                   help="what a case in flight reserves, and what a case that cannot be priced counts as spending")
    p.set_defaults(func=_eval)
