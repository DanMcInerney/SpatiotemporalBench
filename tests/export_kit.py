"""Helpers for the tests of `stbench export` (`test_export*`): a small compiled dataset that can be released, with the
files an export must leave out lying beside the ones it takes.

`build` lays three scripts out under a root, each with its pictures, a render log that binds to its timeline (written
here from the timeline's hashes and the camera file's bytes), `CREDITS.md` and a check report whose counts are worked
out by hand:
- `short`: an apartment script of one task (`vacuum/1`), a short case asking three questions, golds [answerable,
  answerable, unknown];
- `medium`: an apartment script of two tasks, a medium case asking three questions, golds [answerable, unknown,
  unknown];
- `silent`: a warehouse script (`amr/1`) of 6 steps, a case that asks nothing (its questions and gold files are empty).

So the dataset has two (layout, job) groups: `apartment_a vacuum/1` (2 scripts of 6 and 12 keyframes, 2 cases, 6
questions) and `warehouse_a amr/1` (1 script of 6 keyframes, 1 case that asks nothing). `JUNK` and `SCRIPT_JUNK` name
the files of a built dataset that an export must not release. Every count a test expects is written out in the test,
never read back from the export.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from check_kit import empty_case
from eval_kit import SHORT_QUESTIONS, compiled
from stbench.case import CasePaths, ScriptPaths
from stbench.timeline import load
from stream_kit import K, write_pictures
from synth import WAREHOUSE, Synth

FRAMES = {"short": 6, "medium": 12, "silent": 6}   # keyframes of each script
JUNK = ("runs/other/results.json", "check.out", "notes.txt")   # at the dataset's top level or under runs/
SCRIPT_JUNK = ("instances/000000.png", "instance_map.json", "render_tmp/000000.depth.npy")   # under private/<sid>/
# Text with a URL and relative paths in it, none of them a machine path.
CREDITS_TEXT = ("# Credits\n\nThe assets are CC0 or CC BY; this fixture uses none. See https://polyhaven.com/license "
                "and http://example.org/home/ or scripts/<sid>/robot.json, private/index/<id>.json.\n")
# What `docs/format.md`'s release section must name: the files of an export, then what an export leaves out.
PATTERNS = ("README.md", "LICENSE", "CREDITS.md", "check_report.json", "release.json",
            "results/<name>.json", "scripts/<sid>/", "private/index/<id>.json", "timeline.json", "visibility.jsonl",
            "perception_ids.jsonl", "render_log.json", "case.json", "questions.jsonl", "gold.jsonl")
LEFT_OUT = ("instances/", "instance_map.json", "render_tmp/", "runs/")
CHECK_COUNTS = [{"layout": "apartment_a", "job": "vacuum/1", "compiled": 2, "asked_nothing": 0},
                {"layout": "warehouse_a", "job": "amr/1", "compiled": 1, "asked_nothing": 1}]
# the gate rows of that dataset, by hand: its two cases of apartment_a ask `SHORT_QUESTIONS` (2 `path_ahead` and 4
# `object_where` between them); the warehouse case asks nothing and has no row
GATE_ROWS = [{"layout": "apartment_a", "job": "vacuum/1", "family": "path_ahead", "n": 2},
             {"layout": "apartment_a", "job": "vacuum/1", "family": "object_where", "n": 4}]


@dataclass(frozen=True)
class Fixture:
    data: Path
    scripts: dict[str, str]   # "short", "medium", "silent" -> script id
    cases: dict[str, str]     # the same names -> case id

    def script(self, name: str) -> ScriptPaths:
        return ScriptPaths(self.data, self.scripts[name])

    def case(self, name: str) -> CasePaths:
        return CasePaths.from_id(self.data, self.cases[name])


def short_script(seed: int) -> Synth:
    """Six steps of one task in the kitchen, a decision point at step 3 and a mug in view all along."""
    synth = Synth(tasks=[[(K, FRAMES["short"])]], decisions=[(3, "before_doorway")], seed=seed)
    synth.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    synth.see(range(FRAMES["short"]), "E01")
    return synth


def medium_script(seed: int) -> Synth:
    """Two tasks of six steps each, a decision point at step 3 and the mug in view all along."""
    synth = Synth(tasks=[[(K, 6)], [(K, 6)]], decisions=[(3, "before_doorway")], seed=seed)
    synth.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    synth.see(range(FRAMES["medium"]), "E01")
    return synth


def write_render_log(paths: ScriptPaths, **over) -> None:
    """A render log of a whole render (`render_cli.render_binding_problems` finds nothing): the whole span of keyframes,
    with the keys a real log's provenance holds that a released file must not carry a path in."""
    tl = load(paths.timeline)
    log = {"appearance": "assets", "fast": False, "samples": 32, "seed": 1, "frames": [0, tl.n_keyframes],
           "n_frames": tl.n_keyframes, "isp_override": False, "linear_kept": None}
    paths.render_log.write_text(json.dumps({**log, **over}, indent=1) + "\n", encoding="utf-8")


def check_report(**over) -> dict:
    """A check report that passes the declared verdict (`--verdict evidence`), over the counts of `build`'s dataset
    (`CHECK_COUNTS`)."""
    report = {"verdict": "evidence", "passed": True, "passed_point": True, "passed_interval": True,
              "passed_evidence": True, "broken": [], "balance": [], "asked_nothing": CHECK_COUNTS, "groups": [],
              "gates": GATE_ROWS, "wall_seconds": 1.0}
    return {**report, **over}


def write_report(data: Path, **over) -> None:
    (data / "check_report.json").write_text(json.dumps(check_report(**over), indent=2) + "\n", encoding="utf-8")


def build(root: Path) -> Fixture:
    """The dataset described above under `root`, ready to export, with `JUNK` and `SCRIPT_JUNK` lying in it."""
    short = compiled(root, short_script(1), SHORT_QUESTIONS, unknown=(2,))
    medium = compiled(root, medium_script(2), SHORT_QUESTIONS, unknown=(1, 2), a=0, b=1, length="medium")
    silent = empty_case(root, Synth(tasks=[[(WAREHOUSE.rooms[0], FRAMES["silent"])]], seed=5, stage=WAREHOUSE))
    write_pictures(silent.script, FRAMES["silent"])   # `empty_case` writes no pictures
    fx = Fixture(root, {"short": short.script.id, "medium": medium.script.id, "silent": silent.script.id},
                 {"short": short.id, "medium": medium.id, "silent": silent.id})
    for name in fx.scripts:
        write_render_log(fx.script(name))
        for junk in SCRIPT_JUNK:
            (fx.script(name).private / junk).parent.mkdir(parents=True, exist_ok=True)
            (fx.script(name).private / junk).write_bytes(b"render intermediate")
    (root / "CREDITS.md").write_text(CREDITS_TEXT, encoding="utf-8")
    for junk in JUNK:
        (root / junk).parent.mkdir(parents=True, exist_ok=True)
        (root / junk).write_text("working output\n", encoding="utf-8")
    write_report(root)
    return fx


def clone(fx: Fixture, dest: Path) -> Fixture:
    """A copy of the dataset at `dest` (a test mutates its own)."""
    shutil.copytree(fx.data, dest)
    return Fixture(dest, fx.scripts, fx.cases)


def tree(folder: Path) -> dict[str, bytes]:
    """Every file under `folder` by posix path, with its bytes."""
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def files(folder: Path) -> set[str]:
    """The posix paths of the files under `folder`."""
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def released(fx: Fixture) -> set[str]:
    """The files an export of `build`'s dataset holds, written out by hand from the release's definition."""
    found = {"README.md", "LICENSE", "CREDITS.md", "check_report.json", "release.json"}
    for name, sid in fx.scripts.items():
        for k in range(FRAMES[name]):
            found |= {f"scripts/{sid}/sensors/rgb/{k:06d}.jpg", f"private/{sid}/depth/{k:06d}.png"}
        found |= {f"scripts/{sid}/{f}" for f in ("robot.json", "sensors/camera.json", "sensors/pose.jsonl",
                                                 "perception/oracle.jsonl")}
        found |= {f"private/{sid}/{f}" for f in ("timeline.json", "visibility.jsonl", "perception_ids.jsonl",
                                                 "render_log.json")}
    for cid in fx.cases.values():
        found |= {f"private/{cid}/{f}" for f in ("case.json", "questions.jsonl", "gold.jsonl")}
    return found | {f"private/index/{i}.json" for i in (*fx.scripts.values(), *fx.cases.values())}


def edit(path: Path, change: Callable[[dict], object]) -> None:
    """Load the JSON file, let `change` edit the document in place and write it back."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
