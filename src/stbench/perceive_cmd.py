"""`stbench perceive SCRIPT_DIR`: turn a script's renders into the private visibility log and the oracle channel.

Only renders that are whole are perceived: `render_cli.render_binding_problems` (a render log, and a render that covers
every keyframe) must be clean, or the command refuses, naming the problems, since a visibility log over missing or
partial renders would put wrong evidence under every gold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import perception, visibility
from .case import ScriptPaths, depth_move_problem


def _perceive(args: argparse.Namespace) -> int:
    from . import render_cli  # another lane's module, and only this command needs it

    paths = ScriptPaths.from_dir(args.script_dir)
    if moved := depth_move_problem([paths]):
        print(f"stbench perceive: {moved}", file=sys.stderr)
        return 2
    if problems := render_cli.render_binding_problems(paths):
        print(f"stbench perceive: the renders of script {paths.id} are missing or partial: " + "; ".join(problems),
              file=sys.stderr)
        return 2
    n = visibility.write(paths)
    seen, detections = perception.write(paths)
    print(f"{paths.id}: visibility for {n} keyframes; oracle channel with {detections} detections "
          f"({seen} distinct entities seen)")
    return 0


def register(sub) -> None:
    p = sub.add_parser("perceive", help="compute the private visibility log and the agent-facing oracle channel")
    p.add_argument("script_dir", type=Path, help="DATASET/scripts/<sid> or DATASET/private/<sid>")
    p.set_defaults(func=_perceive)
