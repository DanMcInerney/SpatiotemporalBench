"""The `stbench` command line.

Subcommands live in modules that define `register(subparsers)`; this file imports each module named in
COMMAND_MODULES (skipping the ones that do not exist yet) and lets it add its parsers. To add a command, add a module
with `register` and list its name here. A parser sets its handler with `set_defaults(func=...)`; the handler takes the
parsed arguments and returns an exit code (or None for 0).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path

from .case import ScriptPaths
from .leak import find_leaks
from .mapspec import load_map
from .timeline import load
from .validate import validate

# Modules tried at start-up; each may define register(subparsers). A module that does not exist yet is skipped, so a
# lane adds its module's code and never edits this list (plan §2.8): the table is the lanes' commands as they land.
COMMAND_MODULES = (
    "stbench.visibility",
    "stbench.perception",
    "stbench.cmd_generate",
    "stbench.render_cli",
    "stbench.perceive_cmd",
    "stbench.cmd_compile",
    "stbench.cmd_run",
    "stbench.check",
    "stbench.realism_cmd",
    "stbench.asset_cmd",
    "stbench.asset_render",
    "stbench.eval_cmd",
    "stbench.cmd_pilot",
    "stbench.export_cmd",
    "stbench.view_cmd",
)


def parse_seeds(text: str) -> list[int]:
    """"3", "1-3" or "1,4,7"."""
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out += range(int(lo), int(hi) + 1)
        else:
            out.append(int(part))
    return out


def _validate(args: argparse.Namespace) -> int:
    paths = ScriptPaths.from_dir(args.script_dir)
    tl = load(paths.timeline)
    m = load_map(tl.episode.map)
    problems = [f"{p.code}: {p.message}" for p in validate(tl, m)]
    problems += [f"leak: {x.file}: {x.rule} {x.match!r}" for x in find_leaks(paths.root, [tl.episode.map])]
    for line in problems:
        print(line)
    if not problems:
        print(f"{paths.id}: ok ({tl.n_keyframes} keyframes, {len(tl.ego.tasks)} tasks)")
    return 1 if problems else 0


def _version() -> str:
    """The installed package's version (`stbench --version`), or `unknown` from an uninstalled copy."""
    try:
        return importlib.metadata.version("spatiotemporalbench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stbench", description="SpatiotemporalBench")
    parser.add_argument("--version", action="version", version=f"stbench {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="check a script's timeline and scan its agent-facing files for leaks")
    v.add_argument("script_dir", type=Path, help="DATASET/scripts/<sid> or DATASET/private/<sid>")
    v.set_defaults(func=_validate)

    for name in COMMAND_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
            continue
        register = getattr(module, "register", None)
        if register is not None:
            register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
