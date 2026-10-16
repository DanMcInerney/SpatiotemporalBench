"""`stbench view`: one HTML page of a run's cases, answers, gold, stream and transport log, to troubleshoot a system.

    stbench view RUN_DIR [--data DATASET] [--out FILE.html]

RUN_DIR is a run folder of `stbench eval` (or a folder of them, the arms of `--arms`, or the folder of one `stbench
run`); DATASET is the dataset it ran on, found by itself when RUN_DIR is under `DATASET/runs/` and given with `--data`
otherwise. The page is one self-contained file (no server, no network, nothing to install) at `RUN_DIR/view.html`, or
`--out`; its path is printed. It shows the cases the run lists with their status and outcome counts, and opens each
into its questions (the solver's answer beside the gold and the score), the stream it was sent frame by frame and its
transport log. It holds the gold: a post-run view for the person troubleshooting a system, never something a system is
handed, so it is refused under the dataset's `scripts/` or `private/`. The command reads the run and writes the one file
and, beside it, the depth pictures drawn for viewing in a folder named by the page's whole name plus `.frames`
(`view.html.frames/`). Exit 2, nothing written, when RUN_DIR holds no run, or the dataset is not given and not found,
or `--out` is under a truth folder (docs/eval.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import case as case_mod
from .cmd_run import Refused, refuse_truth_folder
from .view import find_dataset, read_run, run_folders
from .view_page import write

FRAMES_SUFFIX = ".frames"   # the page's whole name plus this is the folder of the depth pictures drawn for it


def _view(args: argparse.Namespace) -> int:
    try:
        run_dir = args.run_dir
        if not run_dir.is_dir():
            raise Refused(f"{run_dir} is not a folder")
        if not run_folders(run_dir):
            raise Refused(f"{run_dir} holds no run: no cases/, results.json or summary.json, here or one folder down")
        data = args.data or find_dataset(run_dir)
        if data is None:
            raise Refused("the dataset is not given and RUN_DIR is not under one (DATASET/runs/...): "
                          "pass --data DATASET")
        if not (data / case_mod.PRIVATE).is_dir():
            raise Refused(f"{data} is not a dataset (it has no {case_mod.PRIVATE}/)")
        out = refuse_truth_folder(args.out or run_dir / "view.html", data)
    except Refused as error:
        print(f"stbench view: {error}", file=sys.stderr)
        return 2
    doc = read_run(run_dir, data, out.with_name(out.name + FRAMES_SUFFIX))
    path = write(doc, out)
    cases = sum(len(arm["cases"]) for arm in doc["arms"])
    print(f"{cases} case(s) in {len(doc['arms'])} run folder(s) -> {path} ({path.stat().st_size / 1e6:.1f} MB); "
          f"{len(doc['streams'])} stream(s), their depth pictures in {doc['depth_dir']}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("view", help="write one HTML page of a run's cases, answers, gold, stream and transport log")
    p.add_argument("run_dir", type=Path, metavar="RUN_DIR",
                   help="an eval run folder (or a folder of arm run folders, or one `run` folder)")
    p.add_argument("--data", type=Path, help="the dataset the run ran on (default: the one RUN_DIR is under)")
    p.add_argument("--out", type=Path, help="the page to write (default RUN_DIR/view.html)")
    p.set_defaults(func=_view)
