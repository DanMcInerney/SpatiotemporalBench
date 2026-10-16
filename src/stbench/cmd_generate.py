"""The commands that make a dataset's scripts: `stbench generate`, `stbench cases` and `stbench build`.

- `generate --map M --seed S --robot R --job J --tasks N --out DATASET` writes one script (`generate.generate_script`,
  checked by `validate` before it is written, `episode.write_script`) and prints its id and its keyframe count; `N` is a
  count or a range (`4-6`) from which the seed draws its own count (`build.tasks_for`);
- `cases DATASET` writes the cases of every script in the index (`build.write_cases`);
- `build` runs generate, render, perceive, cases and compile over a range of seeds (`build.py`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import build
from .episode import write_script


def _generate(args: argparse.Namespace) -> int:
    tl = build.generate_checked(args.map, args.seed, args.robot, args.job,
                                build.tasks_for(args.tasks, args.map, args.seed, args.robot, args.job))
    paths = write_script(args.out, tl)
    print(f"{paths.id}: {tl.n_keyframes} keyframes, {len(tl.ego.tasks)} tasks")
    return 0


def _cases(args: argparse.Namespace) -> int:
    ids = build.write_cases(args.dataset)
    print(f"{len(ids)} cases in {args.dataset}")
    return 0


def _add_script_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--map", default="apartment_a")
    parser.add_argument("--robot", default="vacuum", help="a key of `robots.ROBOTS`")
    parser.add_argument("--job", default="vacuum/1", help="the robot's job, by id")
    parser.add_argument("--tasks", type=build.parse_tasks, default=(1, 1),
                        help="tasks in a script: a count (5) or a range (4-6), each seed drawing its own count")
    parser.add_argument("--out", required=True, type=Path, help="dataset directory")


def register(sub) -> None:
    from .render_cli import add_appearance_flags

    g = sub.add_parser("generate", help="write one script: a robot's job over a generated world")
    _add_script_flags(g)
    g.add_argument("--seed", required=True, type=int)
    g.set_defaults(func=_generate)

    c = sub.add_parser("cases", help="write the cases (windows of tasks) of every script of a dataset")
    c.add_argument("dataset", type=Path)
    c.set_defaults(func=_cases)

    b = sub.add_parser("build", help="generate, render, perceive, make the cases and compile scripts")
    _add_script_flags(b)
    b.add_argument("--seeds", required=True, help="a seed, a range (1-3) or a list (1,4,7)")
    b.add_argument("--fast", action="store_true", help="fewer RGB samples (depth and ids are unchanged)")
    add_appearance_flags(b)
    b.add_argument("--keep-linear", type=Path, metavar="DIR",
                   help="keep each script's linear passes in DIR/<script id> (look-dev: a camera change re-develops)")
    b.add_argument("--workers", type=int, default=4, help="parallel processes for everything except rendering")
    b.set_defaults(func=build.build)
