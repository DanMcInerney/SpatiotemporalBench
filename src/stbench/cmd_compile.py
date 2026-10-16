"""`stbench compile DATASET [--cases ID,...] [--workers N]`: write each case's private questions and gold.

A case is compiled from its script's files (`questions.compile_case`), but only a script whose renders are whole: the
check is `render_cli.render_binding_problems` (a render log, and a render that covers every keyframe), and a script
that fails it is refused, naming the problems, because a gold over missing or partial renders would be wrong without
saying so. So is a script whose `scripts/` folder still holds depth images (the layout before 0.6.0,
`case.depth_move_problem`): the refusal names the move to `private/<sid>/depth/`.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import questions
from .case import CasePaths, depth_move_problem, read_index

DEFAULT_WORKERS = 4


def compile_case(job: tuple[str, str]) -> tuple[str, dict[str, int] | None, str | None]:
    """Compile one case of a dataset: (its id, the questions asked per family, None) or (its id, None, why it was
    refused). Never raises for a refusal, so a worker process reports it instead of dying."""
    from . import render_cli  # another lane's module, and only a compile needs it

    dataset, cid = job
    case = CasePaths.from_id(dataset, cid)
    if moved := depth_move_problem([case.script]):
        return cid, None, moved
    if problems := render_cli.render_binding_problems(case.script):
        return cid, None, f"the renders of script {case.script.id} are missing or partial: " + "; ".join(problems)
    return cid, questions.write(case), None


def _compile(args: argparse.Namespace) -> int:
    cases = sorted(i for i, e in read_index(args.dataset).items() if e.get("kind") == "case")
    if args.cases is not None:
        if missing := [c for c in args.cases if c not in cases]:
            print(f"stbench compile: no such case in {args.dataset}: {', '.join(missing)}", file=sys.stderr)
            return 2
        cases = args.cases
    jobs = [(str(args.dataset), cid) for cid in cases]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            results = list(pool.map(compile_case, jobs))
    else:
        results = [compile_case(job) for job in jobs]
    total: Counter[str] = Counter()
    refused = 0
    for cid, found, why in results:
        if found is None:
            refused += 1
            print(f"stbench compile: {cid}: {why}", file=sys.stderr)
            continue
        total.update(found)
        print(f"{cid}: {sum(found.values())} questions " + " ".join(f"{k}={v}" for k, v in sorted(found.items())))
    print(f"all: {len(results) - refused} cases, {sum(total.values())} questions "
          + " ".join(f"{k}={v}" for k, v in sorted(total.items())))
    return 2 if refused else 0


def _workers(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be a whole number of workers, at least 1")
    return value


def register(sub) -> None:
    p = sub.add_parser("compile", help="compile each case's private questions and gold")
    p.add_argument("dataset", type=Path, help="the dataset folder (holding scripts/ and private/)")
    p.add_argument("--cases", type=lambda text: text.split(","), default=None,
                   help="only these case ids, comma-separated (default: every case)")
    p.add_argument("--workers", type=_workers, default=DEFAULT_WORKERS, help="processes (default %(default)s)")
    p.set_defaults(func=_compile)
