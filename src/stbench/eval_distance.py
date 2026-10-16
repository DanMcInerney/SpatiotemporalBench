"""`stbench eval distance RUN [REFERENCE] --data D`: the distance questions' errors against the clearance and by range
band (docs/eval.md).

The headline scores the offset questions (`ego_now`, `ego_mem`, `person_mem`) on what their gold row carries: from
0.9.0 the resolution `tolerances.resolution_m` (what one pixel of the camera resolves at the gold) and the need
`tolerances.need_m` (the job's clearance C); before it the rungs of `tolerances.ladder` (from 0.7.0
`rules.clearance_ladder`, `[C/2, C, 2C]`). This report keeps what the credit hides: how far off the answer was, in
which direction, and whether the score is more than a typical range guess earns. It reads a run folder of `stbench
eval` (`results.json` for the scores and each row's `range_m`, `cases/<cid>/answers.jsonl` for the answers) and the
dataset's gold (the offset of each offset-type gold, and its tolerances), and adds nothing to the results.

The first sections are the clearance ones, per layout, for each job and family and for the job's questions together:
- `n`, and `answered`: those answered with an offset (an `unknown` or a malformed answer has no error);
- `need` (`C` for a ladder): the need in metres, the gold row's `need_m` (the middle rung of a ladder), which is the
  number the scorer uses, so it cannot drift from it (the cell gives the lowest and highest where questions differ);
- `score`: the mean reported score of all `n`;
- `within rho`, `within N`, `within 2N`: the share of the answered ones whose 2-D error (the distance between the
  answered and the true offset in metres, what the scorer tests) is at most the question's resolution, its need and
  twice its need (met up to `score.EPS`, as the scorer does); a ladder's `within C/2`, `within C`, `within 2C` read its
  three rungs the same way;
- `harmful share` (graded rows only): the share of the answered ones on the harmful side, the error pointing away from
  the camera along the gold's bearing (a thing placed farther than it is), where the gold row's `harm` is `radial`;
- `median error (m)`: the median of that absolute error over the answered ones.
A graded row counts when it holds both keys; a ladder counts only when it is exactly `[C/2, C, 2C]`
(`clearance_ladder_of`, to `LADDER_TOL_M`). Every other question is left out of these sections: a four-rung or relative
ladder (a compile before 0.7.0), a ladder the resolvability fallback widened. A layout with both kinds gets a table of
each.

The band sections keep the relative reading as a diagnostic (a negative range error under-estimates). A row without a
`range_m` (a run before 0.6.0, or a question that is no offset one) is not in them.

A question's band is `ego_frame.band(range_m, unit_m)` for its job's place unit: `near` under 6 units, `far` from 16,
`mid` between. Each (layout, family, band) line holds, over the questions in it:
- `n`, and `answered`: those answered with an offset (an `unknown` or a malformed answer has no error);
- `score`: the mean reported score of all `n`;
- `range error`: the median of (answered ground range - true ground range) / `range_m`, signed, so a negative number is
  a systematic under-estimate and a positive one an over-estimate (the ground range is the length of the offset: the
  height is not in it);
- `error / range`: the median of the 2-D distance between the answered and the true offset over `range_m` (the quantity
  the credit rungs are fractions of);
- `bearing error`: the median of the angle in degrees between the answered and the true direction (0 to 180; a question
  whose true or answered offset is the zero vector has none);
- `above the reference`: given reference runs (the scripted `kind_range`, the typical range of the kind, and
  `all_unknown`), the pooled mean of the score differences from the better reference over the questions both answered,
  with a 95% CI from a bootstrap over scripts (`score_stats.paired_difference`). The better one is chosen per (layout,
  job): the reference with the larger mean score over the questions the run and every reference answered there
  (DECISIONS, "Before the binding read", 3), and the report names it. So a system that abstains everywhere, which
  scores what `all_unknown` does, is never above the reference, however `kind_range` scores.

Each layout is reported on its own (nothing pools layouts), and a layout without offset questions is left out.

The bands are as they fall: the range is the 3-D slant, so a drone 15-20 m up, with u = 1.0 m, can never be in `near`
(under 6 m), and its split is lopsided toward `mid` and `far`. That is reported as it is, not corrected.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from . import eval_results, job_registry
from .answers import Offset, parse_answer
from .case import PRIVATE
from .ego_frame import BANDS, band
from .score import EPS
from .score_graded import radial_error
from .score_stats import paired_difference
from .suite import read_jsonl

COLUMNS = ("band", "n", "answered", "score", "range error", "error / range", "bearing error (deg)")
LADDER_TOL_M = 1e-6   # how far a ladder may be from `[C/2, C, 2C]` and still be a clearance ladder
CLEARANCE_COLUMNS = ("job, family", "n", "answered", "C (m)", "score", "within C/2", "within C", "within 2C",
                     "median error (m)")
GRADED_COLUMNS = ("job, family", "n", "answered", "need (m)", "score", "within rho", "within N", "within 2N",
                  "harmful share", "median error (m)")


def measure(answer: Offset, gold: Offset, range_m: float) -> dict:
    """The errors of an answered offset against the gold's, for a question built from the slant range `range_m` (> 0):
    `range_error` the signed difference of the ground ranges over `range_m`, `relative_error` the 2-D distance between
    the offsets over `range_m`, `bearing_deg` the angle between the two directions in degrees (None when either
    offset is the zero vector)."""
    if not range_m > 0:
        raise ValueError(f"range_m must be above 0, got {range_m!r}")
    asked, truth = math.hypot(answer.ahead_m, answer.left_m), math.hypot(gold.ahead_m, gold.left_m)
    bearing = None
    if asked > 0 and truth > 0:
        cosine = (answer.ahead_m * gold.ahead_m + answer.left_m * gold.left_m) / (asked * truth)
        bearing = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return {"range_error": (asked - truth) / range_m,
            "relative_error": math.hypot(answer.ahead_m - gold.ahead_m, answer.left_m - gold.left_m) / range_m,
            "bearing_deg": bearing}


def absolute_error(answer: Offset, gold: Offset) -> float:
    """The 2-D distance in metres between the answered and the true offset: the error the scorer tests its rungs against
    (`score._ladder_credit`)."""
    return math.hypot(answer.ahead_m - gold.ahead_m, answer.left_m - gold.left_m)


def clearance_ladder_of(record: dict) -> tuple[float, float, float] | None:
    """The three rungs of a gold row's `tolerances.ladder` when they are `[C/2, C, 2C]` for some C, to `LADDER_TOL_M`
    (what `rules.clearance_ladder` writes from 0.7.0), else None. The rungs are the scorer's own, so a share read from
    them cannot differ from the credit."""
    ladder = (record.get("tolerances") or {}).get("ladder")
    if not isinstance(ladder, list) or len(ladder) != 3 or not all(isinstance(x, int | float) for x in ladder):
        return None
    low, c, high = ladder
    return (low, c, high) if abs(low - c / 2) <= LADDER_TOL_M and abs(high - 2 * c) <= LADDER_TOL_M else None


def graded_rungs_of(record: dict) -> tuple[float, float, float] | None:
    """(rho, N, 2N) of a gold row of the `graded` credit, from its `tolerances.resolution_m` and `need_m`, else None."""
    tolerances = record.get("tolerances") or {}
    rho, need = tolerances.get("resolution_m"), tolerances.get("need_m")
    if record.get("credit") != "graded" or not all(
            isinstance(x, int | float) and not isinstance(x, bool) and x > 0 for x in (rho, need)):
        return None
    return (rho, need, 2 * need)


def run_folder(path: Path) -> Path:
    return path.parent if path.is_file() else path


def raw_answers(folder: Path, case: str) -> dict:
    """The case's answers by qid, as written (`{}` when the folder holds none)."""
    try:
        records = read_jsonl(folder / "cases" / case / "answers.jsonl")
    except (OSError, ValueError):
        return {}
    return {r["qid"]: r.get("answer") for r in records}


def _answers(folder: Path, case: str) -> dict:
    """The case's answers by qid, parsed as an offset (`{}` when the folder holds none)."""
    return {qid: parse_answer(answer, "offset") for qid, answer in raw_answers(folder, case).items()}


def _golds(data: Path, case: str) -> dict:
    """The case's offset golds by qid: `{qid: (Offset, rungs, graded, radial)}`, the rungs `graded_rungs_of`'s (rho, N,
    2N) for a graded row (`graded` true; `radial` whether its harmful side is the radial one) or `clearance_ladder_of`'s
    (None for a ladder that is not `[C/2, C, 2C]`)."""
    out = {}
    for gold in read_jsonl(data / PRIVATE / case / "gold.jsonl"):
        if gold["gold"].get("type") != "offset":
            continue
        graded = graded_rungs_of(gold)
        offset = Offset(gold["gold"]["ahead_m"], gold["gold"]["left_m"])
        out[gold["qid"]] = (offset, graded or clearance_ladder_of(gold), graded is not None,
                            (gold.get("tolerances") or {}).get("harm") == "radial")
    return out


def check_gold(doc: dict, data: Path, name: str) -> None:
    """ValueError unless the golds in `data` are the ones the results `doc` scored against: the qids of each case's
    rows are all in the case's gold (qids are random per compile, so another compile's gold holds none of them; the row
    loop of `entries` skips a qid it does not find, so it would read a stale folder as truth)."""
    scored: dict[str, set[str]] = {}
    for row in doc["rows"]:
        scored.setdefault(row["case"], set()).add(row["qid"])
    for case, qids in scored.items():
        try:
            held = eval_results.gold_qids(data / PRIVATE / case / "gold.jsonl")
        except OSError:
            raise ValueError(f"{name}: case {case} has no gold in {data}") from None
        if not qids <= held:
            raise ValueError(f"{name} scored case {case} on another compile than {data} holds: the dataset was "
                             "recompiled since, or is another one")


def entries(run: Path, data: Path) -> list[dict]:
    """One entry per scored offset question of the run: its result `row`, its clearance `rungs` ((rho, N, 2N) of a
    graded row, `graded` true; else `[C/2, C, 2C]` of a ladder, None for any other), its `band` (None without a
    `range_m`), the absolute `error_m` of its answer (None where the run gave no offset), whether that answer was on the
    `harmful` side (None where it gave no offset, false where the gold has no radial harm) and the relative `error`
    (`measure`; None without an offset answer or a `range_m`). ValueError for golds that are not the run's
    (`check_gold`)."""
    doc = eval_results.load(run)
    check_gold(doc, data, str(run))
    folder, golds, answers, found = run_folder(run), {}, {}, []
    for row in doc["rows"]:
        case = row["case"]
        if case not in golds:
            golds[case], answers[case] = _golds(data, case), _answers(folder, case)
        if row["qid"] not in golds[case]:
            continue
        (gold, rungs, graded, radial), range_m = golds[case][row["qid"]], row["range_m"]
        try:
            unit = job_registry.get(row["job"]).unit_m
        except KeyError as error:
            raise ValueError(f"{error.args[0]}: no place unit to put its distances in bands") from None
        given = answers[case].get(row["qid"])
        offset = isinstance(given, Offset)
        found.append({"row": row, "rungs": rungs, "graded": graded, "band": band(range_m, unit) if range_m else None,
                      "error_m": absolute_error(given, gold) if offset else None,
                      "harmful": (radial and radial_error(given, gold) > 0) if offset else None,
                      "error": measure(given, gold, range_m) if offset and range_m else None})
    return found


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):+.3f}" if values else "n/a"


def _line(name: str, found: list[dict], reference: dict[tuple[str, str], dict] | None) -> list[str]:
    """One table line: the questions `found` (all in one band, or all of the family)."""
    errors = [e["error"] for e in found if e["error"]]
    bearings = [e["bearing_deg"] for e in errors if e["bearing_deg"] is not None]
    line = [name, str(len(found)), str(len(errors)), f"{sum(e['row']['score'] for e in found) / len(found):+.3f}",
            _median([e["range_error"] for e in errors]),
            f"{statistics.median(e['relative_error'] for e in errors):.3f}" if errors else "n/a",
            f"{statistics.median(bearings):.1f}" if bearings else "n/a"]
    if reference is not None:
        try:
            d = paired_difference([e["row"] for e in found], reference.values(), n_boot=1000)
            line.append(f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] over {d['n']}" if d["ci"]
                        else f"{d['mean']:+.3f} (one script) over {d['n']}")
        except ValueError:   # no question in common
            line.append("-")
    return line


def _clearance_cell(cs: list[float]) -> str:
    """The clearances of a line: one number when they agree, else `lowest-highest`."""
    low, high = min(cs), max(cs)
    return f"{low:.3f}" if high - low <= EPS else f"{low:.3f}-{high:.3f}"


def _share(hits: list[bool]) -> str:
    return f"{sum(hits) / len(hits):.3f}" if hits else "n/a"


def _clearance_line(name: str, found: list[dict]) -> list[str]:
    """One clearance line over the questions `found` (all with `rungs`, all graded or all ladder): the shares of the
    answered ones within the three rungs (rho, N and 2N of a graded row; C/2, C and 2C of a ladder; each question
    against its own, up to `score.EPS` as the scorer's are), a graded line's share on the harmful side, and the median
    absolute error."""
    answered = [e for e in found if e["error_m"] is not None]
    within = [[e["error_m"] <= e["rungs"][i] + EPS for e in answered] for i in range(3)]
    mean = sum(e["row"]["score"] for e in found) / len(found)
    harmful = [_share([bool(e["harmful"]) for e in answered])] if found[0]["graded"] else []
    return [name, str(len(found)), str(len(answered)), _clearance_cell([e["rungs"][1] for e in found]), f"{mean:+.3f}",
            *map(_share, within), *harmful,
            f"{statistics.median(e['error_m'] for e in answered):.3f}" if answered else "n/a"]


def clearance_tables(found: list[dict]) -> list[str]:
    """The markdown of the clearance sections: per layout (the graded rows, then the ladder ones, a table each where a
    layout has both) a table with a line for each (job, family) and one for all the job's offset questions, from the
    entries `found` (`entries`); the entries without `rungs` are left out."""
    out, mine = [], [e for e in found if e["rungs"] is not None]
    for layout in sorted({e["row"]["layout"] for e in mine}):
        for graded, columns in ((True, GRADED_COLUMNS), (False, CLEARANCE_COLUMNS)):
            here_kind = [e for e in mine if e["row"]["layout"] == layout and e["graded"] is graded]
            rows = []
            for job in sorted({e["row"]["job"] for e in here_kind}):
                here = [e for e in here_kind if e["row"]["job"] == job]
                families = sorted({e["row"]["family"] for e in here})
                rows += [_clearance_line(f"{job}, {family}", [e for e in here if e["row"]["family"] == family])
                         for family in families]
                if len(families) > 1:
                    rows.append(_clearance_line(f"{job}, all", here))
            if rows:
                out += [f"## Clearance, {layout}", "", "| " + " | ".join(columns) + " |",
                        "| " + " | ".join("---" for _ in columns) + " |", *("| " + " | ".join(r) + " |" for r in rows),
                        ""]
    return out


def tables(found: list[dict], reference_rows: list[dict] | None = None) -> list[str]:
    """The markdown of the report: per layout and family a table of bands (`near`, `mid`, `far`) and all, from the
    entries `found` (`entries`); the second run's rows give the `above` column."""
    reference = None if reference_rows is None else {(r["case"], r["qid"]): r for r in reference_rows}
    out, found = [], [e for e in found if e["band"] is not None]
    for layout in sorted({e["row"]["layout"] for e in found}):
        for family in sorted({e["row"]["family"] for e in found if e["row"]["layout"] == layout}):
            mine = [e for e in found if e["row"]["layout"] == layout and e["row"]["family"] == family]
            head = [*COLUMNS, *(["above the reference"] if reference is not None else [])]
            rows = [_line(name, here, reference) for name in (*BANDS, "all")
                    if (here := mine if name == "all" else [e for e in mine if e["band"] == name])]
            out += [f"## {layout}, {family}", "", "| " + " | ".join(head) + " |",
                    "| " + " | ".join("---" for _ in head) + " |", *("| " + " | ".join(r) + " |" for r in rows), ""]
    return out


def better_reference(found: list[dict], docs: list[dict]) -> tuple[list[dict], list[str]]:
    """(the rows of the reference used, a line saying which) for the entries `found` and the reference results `docs`
    (several systems over the same cases). Per (layout, job) the reference is the one whose mean score is the largest
    over the questions of that cell that the run and every reference scored (the first listed on a tie), so the cell's
    reference is the better of the systems, as a whole, on the same questions. A cell with no question in common with
    every reference has none."""
    cells = defaultdict(set)
    for e in found:
        cells[e["row"]["layout"], e["row"]["job"]].add((e["row"]["case"], e["row"]["qid"]))
    keyed = [{(r["case"], r["qid"]): r for r in doc["rows"]} for doc in docs]
    rows, lines = [], []
    for (layout, job), keys in sorted(cells.items()):
        shared = sorted(k for k in keys if all(k in by for by in keyed))
        if not shared:
            lines.append(f"- {layout} {job}: no question scored by every reference, so no `above` there")
            continue
        means = [statistics.fmean(by[k]["score"] for k in shared) for by in keyed]
        best = max(range(len(docs)), key=lambda i: (means[i], -i))
        rows += [keyed[best][k] for k in shared]
        scores = ", ".join(f"{doc['system']} {m:+.3f}" for doc, m in zip(docs, means, strict=True))
        lines.append(f"- {layout} {job}: {docs[best]['system']} ({scores} over {len(shared)} questions)")
    return rows, lines


def distance(run: Path, data: Path, reference: Path | Sequence[Path] | None = None) -> str:
    """The distance report of the run at `run` (a results folder or file) over `data`'s golds, with the scores of
    `reference` (a results folder or file, or several: the same cases, other systems) for the `above` column, taken
    against the better reference per (layout, job) (`better_reference`) and naming it. ValueError for results that
    cannot be read."""
    found = entries(run, data)
    ref, used = None, []
    if reference is not None:
        paths = [reference] if isinstance(reference, Path) else list(reference)
        docs = [eval_results.load(path) for path in paths]
        for path, doc in zip(paths, docs, strict=True):
            check_gold(doc, data, str(path))
        ref, used = better_reference(found, docs)
    body = [*clearance_tables(found), *tables(found, ref)]
    if not body:
        return "No offset question was scored in this run.\n"
    notes = ["Errors are over the questions answered with an offset. The clearance sections read the absolute error in "
             "metres against each question's own resolution and need or, for an older compile, against C, "
             "the middle rung of its ladder when it is [C/2, C, 2C] (the job's clearance from 0.7.0). The band "
             "sections are diagnostics in the range's units; a negative range error under-estimates.", ""]
    if ref is not None:
        notes += ["`above the reference` is the score less the better reference's, per layout and job, over the "
                  "questions both answered. The reference used (the larger mean over the questions the run and every "
                  "reference scored):", "", *used, ""]
    return "\n".join(["# SpatiotemporalBench: distance against the clearance and by range band", "", *notes,
                      *body]).rstrip("\n") + "\n"

