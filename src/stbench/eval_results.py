"""`results.json`, the record of an `eval` (docs/eval.md).

One document a run folder (an arm of `--arms` is a run folder of its own): what was run (the system, the channel, the
brief variant, the during flag, where it ran, the reasoning effort, the text-only flag and, for an agent that kept
one, the name of its memory and, for one whose thread Codex compacted early, the limit), whether the result is
`verified` (the sandbox is Docker or the harness's own process, and no case used a remote store: D4) or `partial`,
the cases with how each ended and what it spent (`wall_s` per message, `state_bytes`, `usage` as the system reported
it, the problems of that report, its token totals, whether it names a remote store, and the cost in dollars by the
price table with the reason when there is none), the score rows of the finished cases, one aggregate per layout and the
notes that say why a result is not a benchmark score. Everything is read from the case folders (`cmd_run.write_run`
wrote them), never from memory. The resource fields come from M5's functions (`stream_protocol.usage_problems`,
`usage_totals`, `remote_memory`; `prices.cost_usd`, `cost_note`), never recomputed here. The qids of a case's rows are
how a resume, `eval compare`, `eval distance` and the export know a result was scored on the dataset's gold: qids are
random per compile, so a row whose qid the gold does not hold was scored on another compile.

Each layout is aggregated on its own and nothing pools layouts (VISION rule 13): the apartment and the warehouse are
different worlds, so a mean over both would be a number about neither. `RESUME_KEYS` are the stamp's keys a rerun into
the same folder must find unchanged. A layout's aggregate (`score_stats.aggregate`) holds, beside the headline, the
score above the floor (`above_floor`: the headline less the unknown share, weighted the same way and resampled by
script) and the splits `by` (per family, ability, length, `slice` and `relevance` among them: all derived from the
rows, so a row gained no field). The headline is the mean of the `now` and `memory` columns (`columns`), each the mean
of its abilities by their weights; `abilities_mean` and `pooled` stand beside it, and `thin` lists the (job, ability,
column) cells the dataset asks fewer than `MIN_ABILITY_N` questions of, each with its weight below 1 (0.11.0), which
`ability_counts` reads from the golds of the whole dataset (not the cases of this run; `--cells-from`: another
dataset's) and the aggregate weighs accordingly. A row holds the reported `score`, the gate's
`score_gate` and whether the answer was a `catastrophe`; an aggregate holds, beside the rest, `by.age_band`,
`job_age_band` and `catastrophe_rate` (0.10.0, `score_stats`). `SCHEMA` is the document's shape and `problems` checks
a document against it (the package has no JSON-schema library); a key it does not list is allowed, so a document of
an earlier build, with its stamps, loads, and a `May` entry is one that a document of an earlier build lacks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import ability_counts, cmd_run, prices, suite
from .brief import VARIANTS
from .case import CasePaths
from .eval_systems import Arm, Config, Outcome
from .score_stats import DIMS, NEW_DIMS, OUTCOMES, SLICES, aggregate
from .stream_protocol import CHANNELS, MODEL_TOKEN_KEYS, remote_memory, usage_problems, usage_totals

RELAXED_NOTE = ("every ability is under 20 questions on its job in this dataset; the rule is not applied, so this "
                "headline is not the release statistic")
RESUME_KEYS = ("system", "channel", "brief", "during", "sandbox", "reasoning", "text_only", "memory", "compact_at")
SANDBOXES = ("in-process", "docker", "none")
RESULTS, SUMMARY, ROWS = "results.json", "summary.json", "rows.jsonl"
MAX_FOUND = 10   # problems `problems` reports


class Mismatch(ValueError):
    """RUNDIR holds the results of another run (or of other gold): resuming into it would mix two runs."""


@dataclass
class Run:
    """What one call of `eval` did, for the notes: how each case of each arm it ran ended (by (arm folder name, case
    id)), whether the budget stopped it and why a refusing agent did."""

    outcomes: dict[tuple[str, str], Outcome] = field(default_factory=dict)
    capped: bool = False
    fatal: str | None = None


class Opt:
    """A schema entry that may also be null."""

    def __init__(self, spec):
        self.spec = spec


class May:
    """A schema entry that a document of an earlier build lacks (0.10.0): it may be missing, and when present it is of
    `spec`."""

    def __init__(self, spec):
        self.spec = spec


class Each:
    """A schema entry for an object whose every value is of `spec` and whose keys are the data's own."""

    def __init__(self, spec):
        self.spec = spec


# A type is any value of it (`float` any number, `int` any whole number, neither a bool); a string is that string; a
# tuple the values it lists; a dict the keys it lists (others are allowed); a one-item list a list of such items.
ROW = {"case": str, "script": str, "qid": str, "family": str, "when": ("during", "end"), "scope": ("now", "memory"),
       "reference": Opt(str), "critical": bool, "slice": SLICES, "answer_class": Opt(str),
       "range_m": Opt(float), "age_band": May(Opt(str)), "s": float, "score": float, "score_gate": May(float),
       "catastrophe": May(bool), "outcome": OUTCOMES, "layout": str, "robot": str, "job": str, "length": str}
CASE = {"id": str, "script": str, "layout": str, "robot": str, "job": str, "length": str, "status": str,
        "failures": Each(int), "wall_s": Each(float), "state_bytes": Each(int), "usage": Opt(dict),
        "usage_problems": [str], "tokens": Each(Opt(int)), "remote_memory": bool, "cost_usd": Opt(float),
        "cost_note": Opt(str), "error": Opt(str)}
AGGREGATE = {"n": int, "scripts": int, "cases": int, "headline": dict, "columns": dict, "abilities_mean": dict,
             "pooled": dict, "thin": [list], "thin_ignored": bool, "above_floor": dict,
             "by": {dim: May(dict) if dim in NEW_DIMS else dict for dim in DIMS},
             "counts": {name: int for name in OUTCOMES}, "confidently_wrong_rate": dict, "abstention": dict,
             "job_age_band": May(dict), "catastrophe_rate": May(dict)}
SCHEMA = {"system": str, "channel": tuple(CHANNELS), "brief": VARIANTS, "during": bool, "sandbox": SANDBOXES,
          "reasoning": Opt(str), "text_only": bool, "memory": May(Opt(str)), "compact_at": May(Opt(int)),
          "verified": bool, "partial": bool, "max_usd": Opt(float), "cost_usd": Opt(float), "cases": [CASE],
          "rows": [ROW], "layouts": Each(Opt(AGGREGATE)), "notes": [str]}


def _check(value, spec, where: str, found: list[str]) -> None:
    while isinstance(spec, May):
        spec = spec.spec
    if isinstance(spec, Opt):
        spec = None if value is None else spec.spec
        if spec is None:
            return
    if len(found) >= MAX_FOUND:
        return
    if spec is float or spec is int:
        ok = isinstance(value, spec if spec is int else int | float) and not isinstance(value, bool)
        kind = "a whole number" if spec is int else "a number"
    elif isinstance(spec, type):
        ok, kind = isinstance(value, spec), f"a {spec.__name__}"
    elif isinstance(spec, str):
        ok, kind = value == spec, f"{spec!r}"
    elif isinstance(spec, tuple):
        ok, kind = value in spec, f"one of {', '.join(map(repr, spec))}"
    elif isinstance(spec, Each):
        ok, kind = isinstance(value, dict), "an object"
        if ok:
            for key, sub in value.items():
                _check(sub, spec.spec, f"{where}.{key}", found)
    elif isinstance(spec, list):
        ok, kind = isinstance(value, list), "a list"
        if ok:
            for i, item in enumerate(value):
                _check(item, spec[0], f"{where}[{i}]", found)
    else:
        ok, kind = isinstance(value, dict), "an object"
        if ok:
            for key, sub in spec.items():
                if key in value:
                    _check(value[key], sub, f"{where}.{key}", found)
                elif not isinstance(sub, May):
                    found.append(f"{where}.{key} is missing")
    if not ok:
        found.append(f"{where} is {value!r:.60}, not {kind}")


def problems(doc) -> list[str]:
    """What is wrong with `doc` as a results document (at most ten findings), [] when nothing is."""
    found: list[str] = []
    _check(doc, SCHEMA, "results", found)
    return found


def load(path: str | Path) -> dict:
    """The validated results at PATH (a `results.json`, or the folder holding one). ValueError names the file."""
    path = Path(path)
    path = path / RESULTS if path.is_dir() else path
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"{path}: cannot be read: {error}") from None
    except ValueError as error:
        raise ValueError(f"{path}: is not JSON: {error}") from None
    if found := problems(doc):
        raise ValueError(f"{path}: is not a results document: {found[0]}")
    return doc


def write(out: Path, doc: dict) -> Path:
    """`out/results.json`, replaced atomically."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / RESULTS
    tmp = path.with_name(RESULTS + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def gold_qids(gold: Path) -> set[str]:
    """The qids of the `gold.jsonl` at `gold` (a compile draws them anew, so they say which compile a row was scored
    on). OSError for a gold that cannot be read."""
    return {row["qid"] for row in suite.read_jsonl(gold)}


# -- what a case folder holds ------------------------------------------------------------------------------------
def summary_of(arm: Arm, cid: str) -> dict | None:
    """The case's `summary.json` as `cmd_run.write_run` wrote it, None when there is none or it is not an object."""
    try:
        found = json.loads((arm.case_dir(cid) / SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def finished(arm: Arm, cid: str) -> bool:
    """A case is finished when its summary says it completed and its rows were written."""
    summary = summary_of(arm, cid)
    return bool(summary) and summary.get("status") == "complete" and (arm.case_dir(cid) / ROWS).is_file()


def resources(usage: dict | None) -> dict:
    """The resource fields of a case from the usage it reported: `usage_problems`, `tokens` (`usage_totals`, all null
    for a report with problems), `remote_memory`, `cost_usd` and `cost_note` (`prices`)."""
    found = usage_problems(usage)
    return {"usage_problems": found,
            "tokens": dict.fromkeys(MODEL_TOKEN_KEYS) if found else usage_totals(usage),
            "remote_memory": remote_memory(usage), "cost_usd": prices.cost_usd(usage),
            "cost_note": prices.cost_note(usage)}


def stamp(cfg: Config, arm: Arm) -> dict:
    """What identifies a run folder, which a resume must find unchanged (`RESUME_KEYS`)."""
    return {"system": cfg.system, "channel": cfg.channel, "brief": arm.brief, "during": arm.during,
            "sandbox": cfg.sandbox, "reasoning": cfg.reasoning, "text_only": cfg.text_only,
            **({"memory": cfg.memory.name} if cfg.memory else {}),
            **({"compact_at": cfg.compact_at} if cfg.compact_at is not None else {})}


def check_resume(cfg: Config, arm: Arm, previous: dict | None) -> None:
    """Mismatch unless the results already in the folder (None: none) are of this run and of the gold now on disk: the
    qids of every finished case's rows must all be in that case's gold."""
    if previous is None:
        return
    now = stamp(cfg, arm)
    # a reference row has neither `memory` nor `compact_at`
    if differing := [k for k in RESUME_KEYS if previous.get(k) != now.get(k)]:
        shown = ", ".join(f"{k}: there {previous.get(k)!r}, now {now.get(k)!r}" for k in differing)
        raise Mismatch(f"{arm.path} holds the results of a different run ({shown}); choose another --out")
    if changed := [c.id for c in cfg.cases if finished(arm, c.id)
                   and not {r["qid"] for r in _rows(arm, c.id)} <= gold_qids(c.gold)]:
        raise Mismatch(f"{arm.path} was scored against another compile of case(s) {', '.join(changed[:3])}: the "
                       "dataset was recompiled since; choose another --out")


# -- the document ------------------------------------------------------------------------------------------------
def _entry(arm: Arm, case: CasePaths, run: Run) -> tuple[dict, bool]:
    """(the case's record, whether it has a summary: it ran)."""
    info = suite.case_info(case)
    summary = summary_of(arm, case.id)
    outcome = run.outcomes.get((arm.path.name, case.id))
    usage = summary.get("usage") if summary else None
    status = summary.get("status", "invalid") if summary else outcome.status if outcome else "pending"
    entry = {**{key: info[key] for key in ("script", "layout", "robot", "job", "length")}, "id": case.id,
             "status": status, "failures": dict(summary.get("failures") or {}) if summary else {},
             "wall_s": {key: round(sum(times), 3) for key, times in (summary.get("wall_s") or {}).items()}
             if summary else {}, "state_bytes": dict(summary.get("state_bytes") or {}) if summary else {},
             "usage": usage, **resources(usage),
             "error": None if summary or not outcome else outcome.error}
    if not summary:
        entry["cost_note"] = "not run"
    return entry, summary is not None


def _rows(arm: Arm, cid: str) -> list[dict]:
    return suite.read_jsonl(arm.case_dir(cid) / ROWS)


def _notes(cfg: Config, stamped: dict, cases: list[dict], run: Run, done: int, relaxed: bool) -> list[str]:
    notes = []
    if done < len(cases):
        notes.append(f"partial: {done} of {len(cases)} cases finished; not a benchmark score")
    if stamped["sandbox"] == "none":
        notes.append(f"sandbox none: {cmd_run.UNSANDBOXED}")
    if remote := [c["id"] for c in cases if c["remote_memory"]]:
        notes.append(f"{len(remote)} case(s) used a remote store (remote memory, e.g. {remote[0]}): unverified")
    if bad := [c for c in cases if c["usage_problems"]]:
        notes.append(f"{len(bad)} case(s) reported a malformed usage, so their cost is unknown (e.g. {bad[0]['id']}: "
                     f"{bad[0]['usage_problems'][0]})")
    if run.capped:
        notes.append(f"the --max-usd cap of ${cfg.max_usd:.2f} stopped new cases")
    if run.fatal:
        notes.append(f"stopped: {run.fatal}")
    if relaxed:
        notes.append(RELAXED_NOTE)
    return notes


def build(cfg: Config, arm: Arm, run: Run) -> dict:
    """The results document of an arm's folder as it stands: the finished cases scored from the rows their folders
    hold and aggregated for each layout apart. `run` says how this call's cases ended (a case that left no folder, as
    a refusal does, is otherwise `pending`)."""
    entries, ran, rows, done = [], [], [], 0
    for case in cfg.cases:
        entry, had_summary = _entry(arm, case, run)
        entries.append(entry)
        if had_summary:
            ran.append(entry)
        if finished(arm, case.id):
            rows += _rows(arm, case.id)
            done += 1
    stamped = stamp(cfg, arm)
    costs = [e["cost_usd"] for e in ran]
    thin, relaxed = ability_counts.thin_rule(cfg.cells_from or cfg.data)
    layouts = {layout: aggregate(here, thin=thin, thin_ignored=relaxed)
               if (here := [r for r in rows if r["layout"] == layout]) else None
               for layout in sorted({e["layout"] for e in entries})}
    return {
        **stamped, "verified": stamped["sandbox"] != "none" and not any(e["remote_memory"] for e in entries),
        "partial": done < len(entries), "max_usd": cfg.max_usd,
        "cost_usd": None if None in costs else round(sum(costs), 6), "cases": entries, "rows": rows,
        "layouts": layouts, "notes": _notes(cfg, stamped, entries, run, done, relaxed),
    }
