"""The released set of a dataset, and the checks that decide whether it can be released (`stbench export`, A1 to A3).

A built dataset holds more than a user needs: the render intermediates, the working output, scratch files. The release
is exactly the files named here, and nothing else is released, so a file the generator starts to write stays out until
it is added to this list (`docs/format.md`, "A release"):

```text
README.md  LICENSE  CREDITS.md  check_report.json  release.json
results/<name>.json                 one per --results
scripts/<sid>/**                    every file under each script's agent-facing folder (never a depth image)
private/index/<id>.json             the index entry of each script and case
private/<sid>/                      timeline.json  visibility.jsonl  perception_ids.jsonl  render_log.json
                                    depth/<k>.png for each keyframe k, and no other file
private/<cid>/                      case.json  questions.jsonl  gold.jsonl
```

The depth is the scorer's: it ships under `private/<sid>/depth/`, which the `detections` gate
channel is sent and `stbench check` needs, and never under `scripts/`, where the leak scan (`depth`) refuses it.

Left out: `private/<sid>/instances/`, `instance_map.json` and `render_tmp/` (the render's intermediates, read only by
`stbench perceive`, which rebuilds the released visibility log), `runs/` and anything else under the dataset.

This module reads the dataset and says what is wrong with it; it writes only inside the folder it is given. Every check
returns its problems as sentences, and `export_cmd` refuses on the first stage that has any, before the first write.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import re
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import abilities, check, cmd_run, map_files, suite
from . import eval_results as results_mod
from .case import PRIVATE, CasePaths, ScriptPaths, index_dir
from .leak import find_leaks
from .render_cli import render_binding_problems
from .timeline import load

RELEASE, REPORT, CREDITS, CARD, LICENCE = "release.json", "check_report.json", "CREDITS.md", "README.md", "LICENSE"
AGENT_FILES = ("robot.json", "sensors/camera.json", "sensors/pose.jsonl", "perception/oracle.jsonl")  # + the frames
SCRIPT_FILES = ("timeline.json", "visibility.jsonl", "perception_ids.jsonl", "render_log.json")
CASE_FILES = ("case.json", "questions.jsonl", "gold.jsonl")
TEXT_SUFFIXES = (".json", ".jsonl", ".md")
WORKERS = 8
SHOWN = 5   # problems a refusal names
# An absolute path of a machine: a drive (C:\ or C:/), a system folder at the start of a path, a ~/ home. A URL is none
# (the letter before its colon follows another letter) and neither is a relative path such as private/index.
ABSOLUTE_PATH = (r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
                 r"|(?<![\w.:/~-])/(?:home|Users|root|tmp|var|mnt|opt|usr|etc|private|Volumes)/"
                 r"|(?<![\w.])~[\\/]")

Source = Path | bytes   # a released file is a copy of a dataset file, or bytes made here


class Refusal(Exception):
    """The export is refused (exit 2) and nothing is written."""


@dataclass(frozen=True)
class Script:
    """A script that can be released: its index entry's layout and job (what `check` groups by) and its keyframes."""

    id: str
    paths: ScriptPaths
    layout: str
    job: str
    frames: int


@dataclass(frozen=True)
class Case:
    """A compiled case that can be released, with what `release.json` says of it."""

    id: str
    script: str
    length: str
    questions: int
    qids: frozenset[str]   # the qids of its gold: what a result's rows must be found in
    abilities: dict[str, int] = field(default_factory=dict)   # its questions by the ability their family measures


def refuse(what: str, problems: list[str]) -> None:
    """Raise Refusal naming the first few of `problems`, if there are any."""
    if problems:
        more = f" (and {len(problems) - SHOWN} more)" if len(problems) > SHOWN else ""
        raise Refusal(f"{len(problems)} {what}, e.g.\n  " + "\n  ".join(problems[:SHOWN]) + more)


def rel(path: Path, data: Path) -> str:
    return path.relative_to(data).as_posix()


# -- the folders and the index -----------------------------------------------------------------------------------
def check_paths(data: Path, out: Path) -> None:
    """Refusal unless `out` is a fresh place (absent or an empty folder) that does not overlap `data`."""
    if not index_dir(data).is_dir():
        raise Refusal(f"{data} has no {PRIVATE}/index folder: it is not a dataset")
    d, o = data.resolve(), out.resolve()
    if o == d or d in o.parents or o in d.parents:
        raise Refusal(f"--out {out} and --data {data} overlap")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refusal(f"{out} is not empty: export writes a fresh folder")


def held_out(map_name: str | None, split: str | None) -> bool:
    """Whether a script of this map and split is held out: its split says so, or its map is a held-out layout (read
    now, so a test can patch it)."""
    return split == "heldout" or map_name in map_files.HELD_OUT


def index_problems(index: dict[str, dict]) -> list[str]:
    """What stops the index from being released: nothing to release, a held-out script, an entry that is neither a
    script nor a case, a case whose script has no entry."""
    scripts = {i for i, e in index.items() if e.get("kind") == "script"}
    cases = [i for i, e in index.items() if e.get("kind") == "case"]
    out = [] if scripts and cases else [f"the index holds {len(scripts)} scripts and {len(cases)} cases: there is "
                                        "nothing to release"]
    for iid, e in index.items():
        if e.get("kind") == "script":
            if held_out(e.get("map"), e.get("split")):
                out.append(f"script {iid} ({e.get('map')}, split {e.get('split')}) is held out: a held-out script is "
                           "never released")
        elif e.get("kind") == "case":
            if e.get("script") not in scripts:
                out.append(f"case {iid}: its script {e.get('script')} has no index entry")
        else:
            out.append(f"index entry {iid}: kind {e.get('kind')!r} is neither a script nor a case")
    return out


# -- cases and scripts -------------------------------------------------------------------------------------------
def case_problems(data: Path, index: dict[str, dict]) -> list[str]:
    """Each case that is not compiled (a file of `CASE_FILES` is missing)."""
    out = []
    for cid in sorted(i for i, e in index.items() if e.get("kind") == "case"):
        case = CasePaths.from_id(data, cid)
        if missing := [name for name in CASE_FILES if not (case.private / name).is_file()]:
            out.append(f"case {cid} is not compiled: {PRIVATE}/{cid}/ lacks {', '.join(missing)}; compile it with "
                       f"`stbench compile DATASET --cases {cid}`")
    return out


def _lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _abilities(case: CasePaths) -> dict[str, int]:
    """The case's questions by ability: the ability of each gold row's family (`ability_of`), sorted by name."""
    rows = (json.loads(line) for line in case.gold.read_text(encoding="utf-8").splitlines() if line.strip())
    return dict(sorted(Counter(abilities.ability_of(row["family"]) for row in rows).items()))


def case_facts(data: Path, index: dict[str, dict]) -> list[Case]:
    """The compiled cases, by id, with their question counts, counts by ability and gold qids (the cases passed
    `case_problems`)."""
    out = []
    for cid in sorted(i for i, e in index.items() if e.get("kind") == "case"):
        case = CasePaths.from_id(data, cid)
        out.append(Case(cid, index[cid]["script"], index[cid]["length"], _lines(case.questions),
                        frozenset(results_mod.gold_qids(case.gold)), _abilities(case)))
    return out


def _names(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir()} if folder.is_dir() else set()


def load_script(data: Path, sid: str, entry: dict) -> tuple[Script | None, list[str]]:
    """(the script, []) when it can be released, else (None, what is wrong): a file missing, a render that is
    missing or partial (`render_cli.render_binding_problems`), a keyframe with no rgb or depth image, a map that
    the index and the timeline do not agree on or that is held out."""
    paths = ScriptPaths(data, sid)
    missing = [f"scripts/{sid}/{name}" for name in AGENT_FILES if not (paths.root / name).is_file()]
    missing += [f"{PRIVATE}/{sid}/{name}" for name in SCRIPT_FILES if not (paths.private / name).is_file()]
    if missing:
        return None, [f"script {sid}: missing {', '.join(missing)}"]
    try:
        tl = load(paths.timeline)
        problems = [f"script {sid}: {p}" for p in render_binding_problems(paths)]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, [f"script {sid}: its timeline or render log cannot be read ({type(error).__name__}: {error})"]
    if tl.episode.map != entry.get("map"):
        problems.append(f"script {sid}: its index entry says map {entry.get('map')!r} and its timeline "
                        f"{tl.episode.map!r}")
    if held_out(tl.episode.map, entry.get("split")):
        problems.append(f"script {sid}: its timeline's map {tl.episode.map!r} is held out")
    have_rgb, have_depth = _names(paths.rgb_dir), _names(paths.depth_dir)
    gaps = [k for k in range(tl.n_keyframes) if paths.rgb(k).name not in have_rgb
            or paths.depth(k).name not in have_depth]
    if gaps:
        problems.append(f"script {sid}: {len(gaps)} of {tl.n_keyframes} keyframes lack an rgb or a depth image, the "
                        f"first is k={gaps[0]}")
    script = Script(sid, paths, entry["map"], entry["job"], tl.n_keyframes)
    return (None, problems) if problems else (script, [])


def leak_problems(scripts: list[Script]) -> list[str]:
    """Every `leak.find_leaks` hit under the scripts' agent-facing folders; the script's map (its timeline's, which
    `load_script` found equal to its index entry's) is the name that must not appear."""
    def scan(script: Script) -> list[str]:
        return [f"script {script.id}: {x.file}: {x.rule} {x.match!r}"
                for x in find_leaks(script.paths.root, [script.layout])]

    with ThreadPoolExecutor(WORKERS) as pool:
        return [line for lines in pool.map(scan, scripts) for line in lines]


# -- the files ---------------------------------------------------------------------------------------------------
def dataset_files(data: Path, scripts: list[Script], cases: list[Case]) -> list[tuple[str, Path]]:
    """The released files that are copies of the dataset's, as (path in the export, source), sorted by path."""
    found: list[tuple[str, Path]] = []
    for s in scripts:
        found += [(rel(f, data), f) for f in sorted(p for p in s.paths.root.rglob("*") if p.is_file())]
        found += [(rel(s.paths.depth(k), data), s.paths.depth(k)) for k in range(s.frames)]   # these, no stray file
        found += [(f"{PRIVATE}/{s.id}/{name}", s.paths.private / name) for name in SCRIPT_FILES]
    found += [(f"{PRIVATE}/{c.id}/{name}", data / PRIVATE / c.id / name) for c in cases for name in CASE_FILES]
    found += [(f"{PRIVATE}/index/{i}.json", index_dir(data) / f"{i}.json")
              for i in [*(s.id for s in scripts), *(c.id for c in cases)]]
    found += [(CREDITS, data / CREDITS), (REPORT, data / REPORT)]
    return sorted(found)


def machine_names() -> list[str]:
    """This machine's user and home folder names (at least 3 characters, so a short one cannot match everything)."""
    names = set()
    with contextlib.suppress(Exception):   # no home folder, or no user: nothing to look for
        names.add(Path.home().name)
    with contextlib.suppress(Exception):
        names.add(getpass.getuser())
    return sorted(n for n in names if len(n) >= 3)


def machine_traces(files: Iterable[tuple[str, Source]], names: list[str]) -> list[str]:
    """The json, jsonl and md files of the released set that hold an absolute path or a user name, one entry a file
    (the refusal counts files): `path: first trace`, with `(+N more)` when the file holds others."""
    rx = re.compile("|".join([ABSOLUTE_PATH, *(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])" for n in names)]),
                    re.IGNORECASE)
    found = []
    for name, src in files:
        if name.lower().endswith(TEXT_SUFFIXES):
            text = src.decode("utf-8", errors="replace") if isinstance(src, bytes) \
                else src.read_text(encoding="utf-8", errors="replace")
            if hits := list(rx.finditer(text)):
                more = f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""
                found.append(f"{name}: {text[hits[0].start():hits[0].start() + 40]!r}{more}")
    return found


def write_files(out: Path, files: list[tuple[str, Source]]) -> None:
    """Write each file under `out` (a copy streams its source)."""
    def put(item: tuple[str, Source]) -> None:
        name, src = item
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, bytes):
            dst.write_bytes(src)
        else:
            with src.open("rb") as f, dst.open("wb") as g:
                while chunk := f.read(1 << 20):
                    g.write(chunk)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(put, files))


# -- the check report and the results ----------------------------------------------------------------------------
def gold_counts(data: Path) -> dict[tuple[str, str, str], int]:
    """The questions of the compiled cases that ask, by (layout, job, family): what the oracle answers in each gate row
    of a check over the whole dataset (`gates.evaluate`'s `n`)."""
    found: Counter[tuple[str, str, str]] = Counter()
    for case in check.compiled_cases(data)[0]:
        info = suite.case_info(case)
        found.update((info["layout"], info["job"], gold["family"]) for gold in suite.read_jsonl(case.gold))
    return dict(found)


def gate_row_problem(report: dict, data: Path) -> str | None:
    """Why the report's gate rows are not those of the dataset's gold, None when they are: each (layout, job, family) of
    the gold must have a row of its question count `n`, and no row may name another (final review B1: a report of an
    earlier compile, or a scoped check, can agree with a dataset on the cases that ask nothing and gate other
    questions)."""
    rows = report.get("gates")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return f"{REPORT} has no gate rows"
    gated = {(r.get("layout"), r.get("job"), r.get("family")): r.get("n") for r in rows}
    held = gold_counts(data)
    if gated == held and len(rows) == len(held):
        return None
    missing = sorted(set(held) - set(gated))
    extra = sorted(set(map(str, gated)) - set(map(str, held)))
    changed = sorted(key for key in set(held) & set(gated) if held[key] != gated[key])
    parts = [f"{len(missing)} group(s) of the gold it does not gate, e.g. {'/'.join(missing[0])}" if missing else "",
             f"{len(extra)} it gates that the gold lacks, e.g. {extra[0]}" if extra else "",
             f"{len(changed)} with another question count, e.g. {'/'.join(changed[0])}: the report says "
             f"{gated[changed[0]]}, the gold holds {held[changed[0]]}" if changed else ""]
    return (f"{REPORT}'s gate rows are not the dataset's: {'; '.join(p for p in parts if p) or 'a group twice'}: "
            "check this dataset again")


def report_problems(data: Path) -> list[str]:
    """Why `check_report.json` cannot ship: it is missing, was made over other cases than the dataset holds (compiled
    and asking nothing, per layout and job: `check.yields`; its gate rows by (layout, job, family) and question count
    against the gold: `gate_row_problem`), was not run under the evidence verdict (the declared one:
    DECISIONS, "Before the binding read", 2), fails it (`passed_evidence`, which also needs the balance rule) or names
    a broken instrument. The point and interval readings are not read: at about 20 questions a family the point
    reading is noise."""
    path = data / REPORT
    if not path.is_file():
        return [f"{REPORT} is missing: run `stbench check --data DATASET --verdict evidence` first"]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return [f"{REPORT} is not valid JSON"]
    if not isinstance(report, dict):
        return [f"{REPORT} is not a report"]
    out = []
    if report.get("asked_nothing") != (held := check.yields(*check.compiled_cases(data))):
        out.append(f"{REPORT} counts {report.get('asked_nothing')} compiled and silent cases per layout and job, but "
                   f"the export holds {held}: check this dataset again")
    if problem := gate_row_problem(report, data):
        out.append(problem)
    if report.get("verdict") != "evidence":
        out.append(f"{REPORT} was made with --verdict {report.get('verdict')}, not evidence: check this dataset again "
                   "with `--verdict evidence`")
    if report.get("passed_evidence") is not True:
        out.append(f"{REPORT}: passed_evidence is not true (a gate or the balance rule fails at the evidence verdict)")
    if report.get("broken"):
        out.append(f"{REPORT}: an instrument is broken, e.g. {report['broken'][0]}")
    return out


def result_name(doc: dict) -> str:
    """The name a results document ships under, from what it is (system, effort, channel, brief, during), so the same
    row has the same name wherever its run folder was."""
    parts = [cmd_run.safe_name(doc["system"])]
    if doc.get("reasoning"):
        parts.append(cmd_run.safe_name(doc["reasoning"]))
    parts.append(doc["channel"])
    if doc["brief"] != "correct":
        parts.append(doc["brief"])
    if not doc["during"]:
        parts.append("during-off")
    return "-".join(parts)


def read_results(given: list[Path], cases: list[Case]) -> tuple[dict[str, bytes], list[str]]:
    """({name: the file's bytes}, problems) for the `--results` files or run folders. Each must be a results document
    that is not partial, name only exported cases and hold rows whose qids are in the gold of their case (qids are
    random per compile, so a row of another compile is in none)."""
    gold = {c.id: c.qids for c in cases}
    found: dict[str, bytes] = {}
    problems = []
    for path in given:
        file = path / results_mod.RESULTS if path.is_dir() else path
        try:
            doc = results_mod.load(file)
            raw = file.read_bytes()
        except (OSError, ValueError) as error:
            problems.append(str(error))
            continue
        name = result_name(doc)
        unknown = [c["id"] for c in doc["cases"] if c["id"] not in gold]
        changed = sorted({r["case"] for r in doc["rows"] if r["case"] in gold and r["qid"] not in gold[r["case"]]})
        if doc["partial"]:
            problems.append(f"{file}: is partial (not a benchmark score): finish the run, or run only the cases it "
                            "finished")
        if unknown:
            problems.append(f"{file}: names {len(unknown)} case(s) the export does not hold, e.g. {unknown[0]}")
        if changed:
            problems.append(f"{file}: was scored on another compile for {len(changed)} case(s), e.g. {changed[0]}: "
                            "the dataset was recompiled since")
        if name in found:
            problems.append(f"{file}: ships under the name {name!r}, which another --results file has")
        found[name] = raw
    return found, problems
