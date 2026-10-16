"""`stbench export`: write the folder that is uploaded as the dataset.

    stbench export --data DATASET --out DIR [--results RESULTS ...]

A built dataset holds more than a user needs. The export is the scripts' camera images and sensor files, the private
files that `run`, `eval`, `score` and `check` read (and that let anyone recompute every gold from code: the timeline,
the visibility log and the depth images, which are the scorer's and ship under `private/<sid>/depth/`, never under
`scripts/`), the gate report, the reference rows given with `--results`, and the top level that vouches for them:
`README.md` (the dataset card), `LICENSE` (CC BY 4.0), `CREDITS.md`, `check_report.json`, `release.json` (the
package version, the counts per layout and job and by ability, each case's length and question count, the names of
the shipped results; no timestamp) and `results/<name>.json`. `export_files` names the released set exactly;
nothing else is released, and two exports of one dataset are byte-identical. The export works from its own files:
`stbench eval all_unknown --data DIR` scores it.

Exit 2, with every check done before the first write and so nothing written, when: `--out` is not empty or overlaps
`--data`; a script is held out (its split, or its map is in `map_files.HELD_OUT`); a case is not compiled;
a script's render is missing or partial, or a keyframe has no rgb or depth image; the leak scan finds
anything under a script's agent-facing folder; a released json, jsonl or md file holds an absolute path or the name of
this machine's user; `CREDITS.md` is missing; `check_report.json` is missing, was made over
other cases than the dataset holds, was not made with `--verdict evidence`, fails the evidence verdict or names a
broken instrument; or a `--results` file is not a results document, is partial, names a case the export does
not hold or holds rows of another compile (a qid its case's gold lacks).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from importlib import metadata, resources
from pathlib import Path

from . import export_card as card
from . import export_files as ex
from . import move_depth
from .case import read_index
from .export_files import Refusal, Source

DATASET_LICENCE = "CC-BY-4.0.txt"   # package data under stbench/release/, written to the export as LICENSE
RESULTS = "results"


def package_version() -> str | None:
    try:
        return metadata.version("spatiotemporalbench")
    except metadata.PackageNotFoundError:
        return None


def export(data: Path, out: Path, results: list[Path] | tuple[Path, ...] = (),
           log: Callable[[str], None] = lambda msg: None) -> dict:
    """Write the release of `data` to the fresh folder `out` and return `release.json`'s content. Raises Refusal before
    writing anything: every check below runs first."""
    data, out = Path(data), Path(out)
    ex.check_paths(data, out)
    index = read_index(data)
    ex.refuse("index problems", ex.index_problems(index))
    ex.refuse("cases not compiled", ex.case_problems(data, index))
    log(f"checking {sum(e.get('kind') == 'script' for e in index.values())} scripts")
    scripts, problems = [], []
    for sid in sorted(i for i, e in index.items() if e.get("kind") == "script"):
        script, found = ex.load_script(data, sid, index[sid])
        problems += found
        if script is not None:
            scripts.append(script)
    ex.refuse("script problems", problems)
    if not (data / ex.CREDITS).is_file():
        raise Refusal(f"{data} has no {ex.CREDITS}: the CC BY assets need their attribution (`stbench build "
                      "--appearance assets` writes it, or `stbench assets credits`)")
    cases = ex.case_facts(data, index)
    ex.refuse("check report problems", ex.report_problems(data))
    shipped, problems = ex.read_results(list(results), cases)
    ex.refuse("results problems", problems)
    log(f"scanning {len(scripts)} scripts for leaks")
    ex.refuse("leak scan hits", ex.leak_problems(scripts))
    release = card.release_doc(package=package_version(), scripts=scripts, cases=cases, results=list(shipped))
    files: list[tuple[str, Source]] = [
        *ex.dataset_files(data, scripts, cases),
        *((f"{RESULTS}/{name}.json", raw) for name, raw in shipped.items()),
        (ex.LICENCE, (resources.files("stbench") / "release" / DATASET_LICENCE).read_bytes()),
        (ex.RELEASE, (json.dumps(release, indent=2) + "\n").encode("utf-8")),
        (ex.CARD, card.card_bytes(release))]
    files.sort(key=lambda f: f[0])
    log(f"scanning {len(files)} files for absolute paths and user names")
    ex.refuse("released files hold an absolute path or a user name", ex.machine_traces(files, ex.machine_names()))
    log(f"writing {len(files)} files")
    out.mkdir(parents=True, exist_ok=True)
    ex.write_files(out, files)
    return release


def _export(args: argparse.Namespace) -> int:
    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    try:
        release = export(args.data, args.out, args.results, log)
    except Refusal as error:
        print(f"stbench export: {error}", file=sys.stderr)
        return 2
    groups = "; ".join(f"{g['layout']} {g['job']}: {g['scripts']} scripts, {sum(g['cases'].values())} cases"
                       for g in release["groups"])
    questions = sum(g["questions"] for g in release["groups"])
    print(f"exported {len(release['cases'])} cases ({groups}), {questions:,} questions, "
          f"{len(release['results'])} results -> {args.out}")
    return 0


def register(sub) -> None:
    move_depth.register(sub)   # the dataset's layout commands sit beside the export; `cli.COMMAND_MODULES` is fixed
    p = sub.add_parser(
        "export",
        help="write the folder to upload as the dataset (scripts, the private files that score and check need, the "
             "gate report, licence, card, release.json); refuses anything not releasable",
        description="Write the folder to upload as the dataset: scripts/<sid>/ (every file), private/index/<id>.json, "
                    "private/<sid>/ (timeline.json, visibility.jsonl, perception_ids.jsonl, render_log.json, "
                    "depth/), "
                    "private/<cid>/ (case.json, questions.jsonl, gold.jsonl), README.md (the card), LICENSE, "
                    "CREDITS.md, check_report.json, release.json and results/<name>.json (one per "
                    "--results). Never instances/, instance_map.json, render_tmp/ or runs/.",
        epilog="Refused (exit 2, nothing written): --out not empty or overlapping --data; a held-out script; a case "
               "not compiled; a script whose render is missing or partial or that lacks an rgb or depth image; a "
               "leak-scan hit; an absolute path or the user's name in a released json, jsonl or md "
               "file; no CREDITS.md; a check_report.json that is missing, made over other "
               "cases, was not made with --verdict evidence, fails it or names a broken instrument; a --results "
               "file that is not a results document, is partial, names a case not exported or was scored on another "
               "compile.")
    p.add_argument("--data", required=True, type=Path, help="the built and checked dataset folder")
    p.add_argument("--out", required=True, type=Path, help="the folder to write: it must be empty or absent")
    p.add_argument("--results", nargs="+", action="extend", default=[], type=Path, metavar="RESULTS",
                   help="reference rows to ship: results.json files or run folders (not partial), each "
                        "written as results/<name>.json")
    p.set_defaults(func=_export)
