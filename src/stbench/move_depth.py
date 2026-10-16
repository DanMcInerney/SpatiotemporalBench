"""`stbench move-depth --data DATASET`: move a dataset made before 0.6.0 to the layout that keeps depth private.

Before 0.6.0 a script's depth images lived in `scripts/<sid>/sensors/depth/`, which an agent-facing folder must not
hold. For each script of the dataset this moves that folder to `private/<sid>/depth/` (`os.replace`, so a folder moves
whole and its files are not copied) when the private place is absent or holds no file. A script with nothing to move is
left alone, so a second run moves nothing and exits 0. A script that holds frames in both places is refused and left
as it is, since the command cannot say which are current (a render made after the move leaves the old frames behind):
delete the old folder by hand, and the command's exit code is 2 until none is left. It prints what it moved.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .case import ScriptPaths, depth_before_0_6_0, holds_files, read_index


def move_depth(dataset: str | Path) -> tuple[list[str], list[str]]:
    """Move each script's old depth folder to its private place; (the ids moved, the ids refused for holding frames in
    both places), each sorted."""
    moved, refused = [], []
    for sid in sorted(i for i, e in read_index(dataset).items() if e.get("kind") == "script"):
        script = ScriptPaths(dataset, sid)
        old = depth_before_0_6_0(script)
        if not holds_files(old):
            continue
        if holds_files(script.depth_dir):
            refused.append(sid)
            continue
        shutil.rmtree(script.depth_dir, ignore_errors=True)   # a private place with no file in it: empty folders only
        script.private.mkdir(parents=True, exist_ok=True)
        os.replace(old, script.depth_dir)
        moved.append(sid)
    return moved, refused


def _move(args: argparse.Namespace) -> int:
    if not (args.data / "private" / "index").is_dir():
        print(f"stbench move-depth: {args.data} has no private/index folder: it is not a dataset", file=sys.stderr)
        return 2
    moved, refused = move_depth(args.data)
    names = f": {', '.join(moved)}" if moved else ""
    print(f"moved the depth of {len(moved)} script(s) to private/<id>/depth/{names}")
    if refused:
        print(f"stbench move-depth: {len(refused)} script(s) hold frames in both scripts/<id>/sensors/depth/ and "
              f"private/<id>/depth/ and were left as they are, e.g. {refused[0]}: delete the old "
              "scripts/<id>/sensors/depth/ folder of each if the private one is current", file=sys.stderr)
        return 2
    return 0


def register(sub) -> None:
    p = sub.add_parser("move-depth", help="move a pre-0.6.0 dataset's depth from scripts/ to private/ (idempotent)")
    p.add_argument("--data", required=True, type=Path, help="the dataset folder (holding scripts/ and private/)")
    p.set_defaults(func=_move)
