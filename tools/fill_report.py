# SpatiotemporalBench: fills the first-draft report and dataset card from the release's own outputs (a release tool).
# Copyright (c) 2026 the SpatiotemporalBench authors. Released under the MIT licence (see LICENSE).
"""M7 (FD-M6-M7 section 2.5 step 8, section 4 items 7 and 12): no number of the report or the card is typed by hand.

    uv run python tools/fill_report.py tables [--check CHECK.json] [--results R ...] [--pilot PILOT.md]
                                                    [--release RELEASE.json] --out DIR
    uv run python tools/fill_report.py fill --doc FILE --fragments DIR [--write]

`tables` reads what the release produced: the report of `stbench check`, results documents (a results.json or its
run folder), the markdown of `stbench pilot report` and `release.json`. Each input is optional, and a fragment is
written only when its input is given. Every statistic comes from the package (`eval_results`, `score_stats.aggregate`,
`eval_compare.render`, `cmd_pilot`, `export_card`) and is formatted by `eval_report`'s helpers, so it is rounded as
`results.md` rounds it; the script adds counts and sums of what the documents hold, never a statistic. The fragments
are deterministic (same inputs, same bytes) and listed in `DIR/fragments.txt`; a rerun replaces them.

    gates.md             the gate table per (layout, job, family) of the check report, its verdict and BROKEN lines
    headline.md          per (layout, system): headline with its interval, `now`, `memory`, own path, cases scored
    reference-rows.md    per (layout, job, system): those, with reasoning, channel, arm, cases scored, partial,
                         notional USD, tokens in / cached / out and wall time
    reference-splits.md  per (layout, job, system): each family and each case length
    comparison.md        `eval compare`'s output for each set of results one lever apart, then each job alone
    cost-table.md        `cmd_pilot`'s per case-run means by length, for each results document (cost.md: one line of it)
    jobs.md              per (job, place): scripts, cases by length, questions, frames (release.json)
    pilot-*.md           the pilot's spread, brief swap and release count, cut from the pilot report
    the others           one line for a sentence: see BLOCKS

`fill` replaces `<!-- FILL: text -->` placeholders (and the card's `<!-- REFERENCE-ROWS: text -->`; one inside a code
span is an example and is left alone). A placeholder in a table row is a cell: its value is the cell of the fragment
table (`CELL_FRAGMENTS`, in that order) that has the cell's column, in the one row that agrees with the row's other
cells where the fragment has their columns; the placeholder's own text is ignored, and an absent or ambiguous row stays
unfilled. Any other placeholder is a phrase or a block: BLOCKS maps its text to a fragment file, and a fragment of
several lines needs its placeholder alone on its line. What is left is printed with the reason (`HUMAN` holds the
reasons of those that are a person's). Nothing is written without `--write`: the diff is printed instead.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from stbench import baselines, cmd_pilot, eval_compare, eval_results, gates
from stbench.case import LENGTHS
from stbench.eval_report import cell, interval, lever_name, rate, signed
from stbench.score_stats import aggregate

FRAGMENT_LIST = "fragments.txt"
DONE = "complete"
MARK = re.compile(r"<!-- (FILL|REFERENCE-ROWS): (.*?) -->", re.S)
CARD_ROWS = "tables pasted from `stbench check` and `stbench eval compare` output at the release; never typed by hand."
CELL_FRAGMENTS = ("headline.md", "jobs.md", "pilot-spread.md", "pilot-brief.md", "pilot-count.md")
TOKEN_KEYS = ("tokens_in", "tokens_cached_in", "tokens_out")
BLOCKS = {   # a placeholder's text (spaces collapsed) -> the fragment that replaces it
    "the gate table per layout, job and family, pasted from `stbench check` output": "gates.md",
    "the per-job, per-family and per-length tables, pasted from `stbench eval compare` output": "comparison.md",
    CARD_ROWS: "headline.md",
    "their headlines per layout, pasted from `stbench check` output": "free-baselines.md",
    "the cost table, from `stbench pilot report` and the run folders": "cost.md",
    "the reference rows in one sentence": "reference-sentence.md",
    "the cases each row covers; a partial row names its missing scripts": "coverage.md",
    "scripts a job": "scripts-a-job.md",
    "total scripts": "total-scripts.md", "total cases": "total-cases.md", "total frames": "total-frames.md",
    "total questions": "total-questions.md",
    "leak d": "leak-d.md", "leak interval": "leak-interval.md", "leak pairs": "leak-pairs.md",
    "leak scripts": "leak-scripts.md", "pass, inconclusive or fail": "leak-verdict.md",
    "d per job and the trim, if any": "leak-per-job.md",
}
HUMAN = {   # placeholders no fragment fills, and why
    "release date": "the date is the maintainers'",
    "families dropped at the gate checks, by job": "a dropped family is absent from the final check report: it is a "
                                                   "decision, not an output",
    "any family dropped at the extended release's final gate check": "a dropped family is absent from the final check "
                                                                     "report: it is a decision, not an output",
    "the measured half-width per job": "the package defines the predicted half-width (`power`) and no measured one",
    "quota points": "results do not record the quota (docs/eval.md); it comes from the driver's log",
    "the share of short cases in the pilot": "the pilot report prints cases by length per agent and job, no share",
    "the outcome of the look check per place": "the look approval is a person's word, not an output",
}


class Refused(ValueError):
    """The fragments or the fill cannot be made."""


# -- markdown ----------------------------------------------------------------------------------------------------
def table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def tables_in(lines: list[str]) -> list[tuple[int, list[str], list[list[str]]]]:
    """`(index of the header line, header, rows)` of each markdown table in `lines`."""
    found, i = [], 0
    while i < len(lines):
        if (lines[i].lstrip().startswith("|") and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|")
                and all(re.fullmatch(r":?-{3,}:?", c) for c in cells(lines[i + 1]))):
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            found.append((i, cells(lines[i]), [cells(line) for line in lines[i + 2:j]]))
            i = j
        else:
            i += 1
    return found


def first_table(lines: list[str], column: str | None = None) -> tuple[list[str], list[list[str]]]:
    """The header and rows of the first table in `lines` (that has `column`, if given); empty if there is none."""
    return next(((head, rows) for _, head, rows in tables_in(lines) if column is None or column in head), ([], []))


def sections(lines: list[str], level: str) -> dict[str, list[str]]:
    """The lines under each heading that starts with `level` (`## `), by the heading's text."""
    found: dict[str, list[str]] = {}
    key = None
    for line in lines:
        if line.startswith(level):
            key = line[len(level):]
            found[key] = []
        elif key is not None:
            found[key].append(line)
    return found


def length_order(name: str) -> tuple[int, str]:
    return (list(LENGTHS).index(name) if name in LENGTHS else len(LENGTHS), name)


# -- the check report --------------------------------------------------------------------------------------------
def word(ok: bool) -> str:
    return "pass" if ok else "FAIL"


def gate_cell(g: dict) -> str:
    """`ok +0.789 vs 0.4 [+0.611, +0.957]`: the gate's verdict, its value, its bound and the interval it has."""
    if g["pass"] is None:
        return "n/a"
    status = "ok" if g["pass"] else "point" if g["point"] else "FAIL"
    return f"{status} {signed(g['value'])} vs {g['threshold']}" + (f" {interval(g['ci'])}" if g.get("ci") else "")


def gate_fragment(report: dict) -> str:
    if not isinstance(report.get("gates"), list) or not isinstance(report.get("groups"), list):
        raise Refused("the check report lacks `gates` and `groups`: it is not the JSON of `stbench check`")
    shares = {(g["layout"], g["job"]): g["families"] for g in report["groups"]}
    rows = []
    for r in report["gates"]:
        family = shares.get((r["layout"], r["job"]), {}).get(r["family"])
        rows.append([f"`{r['layout']}`", f"`{r['job']}`", f"`{r['family']}`", str(r["scripts"]), str(r["n"]),
                     "n/a" if family is None else rate(family["unknown_share"]),
                     *(gate_cell(r["gates"][name]) for name in gates.GATES),
                     "pass" if r["pass"] else "point" if r["point"] else "FAIL"])
    lines = [table(["Layout", "Job", "Family", "Scripts", "n", "Unknown share", *gates.GATES, "Verdict"], rows), "",
             "Each gate shows its verdict, its value against its bound and the 95% interval where it has one "
             f"(`separation` also wants the interval's lower bound above {gates.SEPARATION_CI}). `ok` passes on the "
             "interval, `point` on the point estimate alone, `FAIL` fails, `n/a` does not apply or has no rows.", "",
             f"Point reading: {word(report['passed_point'])}. Interval reading: {word(report['passed_interval'])}.", ""]
    lines += [f"- {line}" for line in report["broken"]] or ["No instrument is broken."]
    lines += [f"- {y['layout']} {y['job']}: {y['asked_nothing']} of {y['compiled']} compiled cases ask nothing and are "
              "not checked" for y in report["asked_nothing"] if y["asked_nothing"]]
    return "\n".join(lines)


# -- the results -------------------------------------------------------------------------------------------------
def label(doc: dict) -> str:
    """`codex:gpt-6-luna` (xhigh): the system as the report's tables name it."""
    return f"`{doc['system']}`" + (f" ({doc['reasoning']})" if doc["reasoning"] else "")


def arm(doc: dict) -> str:
    return f"{doc['brief']} brief, during {'on' if doc['during'] else 'off'}"


def stat_cell(stat: dict | None, missing: str = "-") -> str:
    return missing if stat is None else "n/a" if stat["mean"] is None else cell(stat)


def over(stat: dict | None) -> str:
    """A column's cell with the number of questions it rests on (`-` when it has none)."""
    return "-" if stat is None else f"{stat_cell(stat)} over {stat['n']}"


def split_cells(agg: dict | None) -> list[str]:
    """The headline, `now`, `memory` and own-path cells of an aggregate (`-` for a split with no rows). `now` and
    `memory` are the aggregate's `columns`, the counted cells the headline averages, not `by.scope`, which pools every
    question of a scope, the diagnostic ones too."""
    if agg is None:
        return ["n/a", "-", "-", "-"]
    return [stat_cell(agg["headline"], "n/a"), *(over(agg["columns"].get(k)) for k in ("now", "memory")),
            stat_cell(agg["by"]["own_path"].get("True"))]


def resources(done: list[dict]) -> list[str]:
    """Tokens in, cached and out, notional USD and wall seconds of the finished cases; `n/a` when any case's is unknown
    (as `eval_report` does)."""
    def total(values: list, fmt) -> str:
        return "n/a" if not values or None in values else fmt(sum(values))
    return [*(total([c["tokens"][k] for c in done], "{:,}".format) for k in TOKEN_KEYS),
            total([c["cost_usd"] for c in done], lambda x: f"${x:.4f}"),
            total([sum(c["wall_s"].values()) for c in done], "{:.1f}".format)]


def load_results(paths: list[str]) -> list[dict]:
    """The validated documents, which must have been scored on one compile (`eval_compare.same_compile`)."""
    docs = [eval_results.load(p) for p in paths]
    try:
        eval_compare.same_compile(paths, docs)
    except eval_compare.Refused as error:
        raise Refused(str(error)) from None
    return docs


def headline_fragment(docs: list[dict]) -> str:
    rows = sorted(((layout, i, [f"`{layout}`", label(doc), *split_cells(doc["layouts"].get(layout)),
                                str(len([c for c in doc["cases"] if c["layout"] == layout and c["status"] == DONE])),
                                doc["channel"], arm(doc)])
                   for i, doc in enumerate(docs) for layout in sorted({c["layout"] for c in doc["cases"]})),
                  key=lambda r: r[:2])
    return table(["Layout", "System", "Headline (95% interval)", "`now`", "`memory`", "Own path", "Cases scored",
                  "Channel", "Arm"], [r[2] for r in rows])


def job_groups(docs: list[dict]):
    """`(layout, job, doc, cases, rows)` for each document and each (layout, job) it holds, in table order."""
    found = sorted({(c["layout"], c["job"], i) for i, doc in enumerate(docs) for c in doc["cases"]})
    for layout, job, i in found:
        doc = docs[i]
        yield (layout, job, doc, [c for c in doc["cases"] if (c["layout"], c["job"]) == (layout, job)],
               [r for r in doc["rows"] if (r["layout"], r["job"]) == (layout, job)])


def reference_fragments(docs: list[dict]) -> dict[str, str]:
    reference, splits = [], []
    for layout, job, doc, cases, rows in job_groups(docs):
        done = [c for c in cases if c["status"] == DONE]
        agg = aggregate(rows) if rows else None
        head, now, memory, own = split_cells(agg)
        reference.append([f"`{layout}`", f"`{job}`", f"`{doc['system']}`", doc["reasoning"] or "-", doc["channel"],
                          arm(doc), f"{len(done)} of {len(cases)}", "no" if len(done) == len(cases) else "yes", head,
                          memory, now, own, *resources(done)])
        for dim in ("family", "length"):
            for value in sorted(agg["by"][dim] if agg else (), key=length_order if dim == "length" else str):
                stat = agg["by"][dim][value]
                splits.append([f"`{layout}`", f"`{job}`", label(doc), dim, value, str(stat["n"]), cell(stat)])
    return {"reference-rows.md": table(
        ["Layout", "Job", "System", "Reasoning", "Channel", "Arm", "Cases scored", "Partial", "Headline (95% interval)",
         "`memory`", "`now`", "Own path", "Tokens in", "Cached", "Tokens out", "USD", "Wall s"], reference),
        "reference-splits.md": table(["Layout", "Job", "System", "Split", "Value", "Questions", "Mean (95% interval)"],
                                     splits)}


def job_view(doc: dict, layout: str, job: str) -> dict:
    """`doc` narrowed to the cases and rows of one job, its layout's aggregate over those rows alone."""
    rows = [r for r in doc["rows"] if r["job"] == job]
    return {**doc, "cases": [c for c in doc["cases"] if c["job"] == job], "rows": rows,
            "layouts": {layout: aggregate(rows) if rows else None}}


def length_table(aggs: list[dict | None], names: list[str]) -> str:
    lengths = sorted({k for a in aggs if a for k in a["by"]["length"]}, key=length_order)
    return table(["length", *names], [[length, *(stat_cell(a and a["by"]["length"].get(length)) for a in aggs)]
                                      for length in lengths])


def comparison_fragment(paths: list[str], docs: list[dict]) -> str:
    """`eval compare`'s output for each set of results that differ in nothing but the agent (`load_all` refuses the
    rest), then its table for each job alone, with the case lengths side by side."""
    parts: dict[tuple, list[tuple[str, dict]]] = {}
    for path, doc in zip(paths, docs, strict=True):
        parts.setdefault((doc["channel"], doc["brief"], doc["during"]), []).append((path, doc))
    out = []
    for members in (parts[key] for key in sorted(parts)):
        group, names = ([d for _, d in members], ["A"]) if len(members) == 1 else eval_compare.load_all(
            [p for p, _ in members])
        text = eval_compare.render(group, names).split("\n", 1)[1]   # without its title
        out += [f"### {lever_name(group[0])}", re.sub(r"^(#+) ", r"##\1 ", text, flags=re.M).strip("\n"), "",
                "#### Per job", ""]   # under the report's own `##` section: the output's headings go down two levels
        for layout, job in sorted({(c["layout"], c["job"]) for d in group for c in d["cases"]}):
            sub = [job_view(d, layout, job) for d in group]
            if not any(d["rows"] for d in sub):
                continue
            lines = eval_compare.render(sub, names).splitlines()
            at = max(i for i, line in enumerate(lines) if line.startswith("## "))   # the layout's, after "## Results"
            out += [lines[at].replace(f"## {layout}", f"##### {layout}, {job}", 1), *lines[at + 1:], "",
                    length_table([d["layouts"][layout] for d in sub], names), ""]
    return "\n".join(out).strip("\n")


def cost_fragments(docs: list[dict]) -> dict[str, str]:
    """`cmd_pilot`'s per-length means of each document's cases as a block, and as one line for a sentence."""
    blocks, lines = [], []
    for doc in docs:
        found = cmd_pilot._resources(cmd_pilot.Group(label(doc), "all jobs", cases=doc["cases"]))
        blocks += [f"**{label(doc)}** ({doc['channel']} channel, {arm(doc)})", "", *found]
        rows = first_table(found)[1]   # length, case-runs, tokens in, of them cached, tokens out, wall s, USD
        lines.append(f"{label(doc)}, a case: " + ("; ".join(
            f"{r[0]} {r[2]} tokens in ({r[3]} cached), {r[4]} out, {r[5]} s, ${r[6]}" for r in rows)
            or "no tokens were reported"))
    return {"cost-table.md": "\n".join(blocks).strip("\n"), "cost.md": "; ".join(lines)}


def phrase_fragments(docs: list[dict]) -> dict[str, str]:
    """The sentences about the rows: the reference agents' headlines, the free rows' and what each row covers."""
    free = [d for d in docs if d["system"] in baselines.BASELINES]
    agents = [d for d in docs if d["system"] not in baselines.BASELINES]

    def headlines(doc: dict) -> dict[str, str]:
        return {layout: stat_cell((doc["layouts"].get(layout) or {}).get("headline"), "n/a")
                for layout in sorted({c["layout"] for c in doc["cases"]})}
    out = {}
    if agents:
        out["reference-sentence.md"] = "; ".join(
            f"{label(d)} scores " + ", ".join(f"{h} on `{layout}`" for layout, h in headlines(d).items())
            for d in agents)
    if free:
        out["free-baselines.md"] = "; ".join(
            f"`{layout}`: " + ", ".join(f"{label(d)} ({d['channel']}) {headlines(d)[layout]}" for d in free
                                         if layout in headlines(d))
            for layout in sorted({c["layout"] for d in free for c in d["cases"]}))
    cover = []
    for d in docs:
        text = f"{label(d)}: {len([c for c in d['cases'] if c['status'] == DONE])} of {len(d['cases'])} cases"
        if missing := sorted({c["script"] for c in d["cases"] if c["status"] != DONE}):
            text += ", not finished in the scripts " + ", ".join(missing)
        cover.append(text)
    return {**out, "coverage.md": "; ".join(cover)}


# -- release.json ------------------------------------------------------------------------------------------------
def release_fragments(release: dict) -> dict[str, str]:
    groups = release["groups"]
    return {
        "jobs.md": table(["Job", "Place", "Scripts", "Cases (short / medium / long)", "Questions", "Frames"],
                         [[f"`{g['job']}`", f"`{g['layout']}`", str(g["scripts"]),
                           " / ".join(str(g["cases"][k]) for k in LENGTHS), f"{g['questions']:,}", f"{g['frames']:,}"]
                          for g in groups]),
        "total-scripts.md": str(sum(g["scripts"] for g in groups)),
        "total-cases.md": str(sum(sum(g["cases"].values()) for g in groups)),
        "total-questions.md": f"{sum(g['questions'] for g in groups):,}",
        "total-frames.md": f"{sum(g['frames'] for g in groups):,}",
        "scripts-a-job.md": ", ".join(f"{g['scripts']} for `{g['job']}`" for g in groups)}


# -- the pilot report --------------------------------------------------------------------------------------------
def spread_table(lines: list[str]) -> str:
    """Each (agent, job) of the pilot rows: scripts, questions a case by length, sigma, design effect and rho as the
    report printed them. The Agent column is there only when several agents ran: a per-job table then finds two rows
    for a job and `fill` leaves the cell for a person to choose."""
    rows = []
    for title, body in sections(lines, "### ").items():
        if not (m := re.match(r"(?P<agent>.+): (?P<job>\S+) \(", title)):
            continue
        found = dict.fromkeys(("scripts", "sigma", "de", "rho"), "n/a")
        for line in body:
            if s := re.match(r"- (\d+) scripts?, ", line):
                found["scripts"] = s[1]
            elif s := re.match(r"- sigma: (\S+)", line):
                found["sigma"] = s[1]
            elif s := re.match(r"- design effect ([\d.]+); rho ([\d.]+)", line):
                found["de"], found["rho"] = s[1], s[2]
        per = {r[0]: r[2] for r in first_table(body, "questions a case")[1]}
        rows.append([m["agent"], f"`{m['job']}`", found["scripts"], " / ".join(per.get(k, "-") for k in LENGTHS),
                     found["sigma"], found["de"], found["rho"]])
    head = ["Agent", "Job", "Scripts", "Questions a case (short / medium / long)", "Score spread", "Design effect",
            "Correlation between windows"]
    drop = 0 if len({r[0] for r in rows}) > 1 else 1
    return table(head[drop:], [r[drop:] for r in sorted(rows, key=lambda r: (r[1], r[0]))])


def leak_fragments(lines: list[str]) -> dict[str, str]:
    """D2's pooled row as the scalars of the paragraph, and each job's d with the report's own conclusion."""
    rows = first_table(lines, "d")[1]   # job, pairs, scripts, d, 95% CI, verdict
    if not (pooled := next((r for r in rows if r[0] == "pooled"), None)):
        raise Refused("the pilot report's D2 section holds no pooled row")
    then = next((m[1] for line in lines if (m := re.match(r"Pooled verdict: \*\*\w+\*\*: (.*)$", line))), "")
    per_job = [f"`{r[0].split()[0]}` " + (f"d {r[3]} {r[4]} ({r[5]})" if r[3] != "n/a" else "n/a")
               for r in rows if r[0] != "pooled"]
    return {"leak-pairs.md": pooled[1], "leak-scripts.md": pooled[2], "leak-d.md": pooled[3],
            "leak-interval.md": pooled[4], "leak-verdict.md": pooled[5],
            "leak-per-job.md": "; ".join(per_job) + ". " + then[:1].upper() + then[1:].rstrip(".")}


def brief_table(lines: list[str]) -> str:
    found: dict[tuple[str, str], dict] = {}   # (job, agent) -> {(rows, other brief): cell, "cases": {length: n}}
    for title, body in sections(lines, "### ").items():
        if not (m := re.match(r"correct minus (swapped|neutral): (.+)$", title)):
            continue
        for r in first_table(body, "BASE - OTHER")[1]:   # job, rows, pairs, scripts, BASE - OTHER, 95% CI, cases
            entry = found.setdefault((r[0], m[2]), {})
            entry["Critical" if r[1] == "critical" else "Other", m[1]] = f"{r[4]} {r[5]}" if r[4] != "n/a" else "n/a"
            entry.setdefault("cases", dict(re.findall(r"(\w+) (\d+)", r[6])))
    head = ["Agent", "Job", "Critical rows: correct - swapped", "Critical rows: correct - neutral",
            "Other rows: correct - swapped", "Other rows: correct - neutral", "Cases (short / medium / long)"]
    drop = 0 if len({agent for _, agent in found}) > 1 else 1
    rows = [[agent, f"`{job}`", *(e.get((kind, other), "-") for kind in ("Critical", "Other")
                                  for other in ("swapped", "neutral")),
             " / ".join(e["cases"].get(k, "0") for k in LENGTHS)] for (job, agent), e in sorted(found.items())]
    return table(head[drop:], [r[drop:] for r in rows])


def count_table(lines: list[str]) -> str:
    head, rows = first_table(lines, "K target")
    pick = ("K target", "K", "half-width at K", "GPU-hours", "points")
    return table(["Job", "Scripts needed", "Scripts released", "Predicted half-width", "GPU hours", "Quota points"],
                 [[f"`{r[0]}`", *(r[head.index(name)] for name in pick)] for r in rows if r[0] != "total"])


def pilot_fragments(text: str) -> dict[str, str]:
    out = {}
    for title, lines in sections(text.splitlines(), "## ").items():
        if title == "Pilot rows":
            out["pilot-spread.md"] = spread_table(lines)
        elif title.startswith("D2:"):
            out |= leak_fragments(lines)
        elif title == "The brief swap":
            out["pilot-brief.md"] = brief_table(lines)
        elif title == "The release's script count":
            out["pilot-count.md"] = count_table(lines)
    return out


# -- the commands ------------------------------------------------------------------------------------------------
def build_fragments(check: Path | None, results: list[str], pilot: Path | None, release: Path | None
                    ) -> dict[str, str]:
    """Every fragment the inputs give, by file name. Refused (ValueError) as the functions above say."""
    if not (check or results or pilot or release):
        raise Refused("give at least one of --check, --results, --pilot or --release")
    out = {}
    if check:
        out["gates.md"] = gate_fragment(json.loads(check.read_text(encoding="utf-8")))
    if results:
        docs = load_results(results)
        out |= {"headline.md": headline_fragment(docs), "comparison.md": comparison_fragment(results, docs),
                **reference_fragments(docs), **cost_fragments(docs), **phrase_fragments(docs)}
    if pilot:
        out |= pilot_fragments(pilot.read_text(encoding="utf-8"))
    if release:
        out |= release_fragments(json.loads(release.read_text(encoding="utf-8")))
    return out


def write_fragments(out: Path, fragments: dict[str, str]) -> None:
    """Into a folder that is empty or that a previous call made (its `fragments.txt` lists the files it replaces)."""
    listing = out / FRAGMENT_LIST
    if out.is_dir() and any(out.iterdir()) and not listing.is_file():
        raise Refused(f"{out} is not empty and holds no {FRAGMENT_LIST}: tables writes only into a folder it made")
    out.mkdir(parents=True, exist_ok=True)
    if listing.is_file():
        for name in listing.read_text(encoding="utf-8").split():
            (out / name).unlink(missing_ok=True)
    for name, text in sorted(fragments.items()):
        (out / name).write_text(text.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    listing.write_text("\n".join(sorted(fragments)) + "\n", encoding="utf-8", newline="\n")


def cell_value(head: list[str], row: list[str], col: int, frags: dict[str, tuple[list[str], list[list[str]]]]
               ) -> tuple[str | None, str]:
    """`(value, why not)` for the placeholder in column `col` of a table row: the cell of the fragment table that has
    the columns of every placeholder of the row, in the one row that agrees with the row's other cells."""
    marked = {i for i, c in enumerate(row) if MARK.search(c)}
    for name in CELL_FRAGMENTS:
        if name not in frags or not all(head[i] in frags[name][0] for i in marked):
            continue
        fhead, frows = frags[name]
        key = {h: c for i, (h, c) in enumerate(zip(head, row, strict=True)) if i not in marked and h in fhead}
        hits = [r for r in frows if all(r[fhead.index(h)] == c for h, c in key.items())]
        if len(hits) != 1:
            return None, f"{len(hits)} rows of {name} agree with {key or 'the row'}"
        return hits[0][fhead.index(head[col])], ""
    return None, "no fragment table has the columns " + ", ".join(head[i] for i in sorted(marked)) + (
        " (give `tables` the input that makes it)")


def fill_text(text: str, fragments: Path) -> tuple[str, list[tuple[int, str, str]]]:
    """`text` with each placeholder a fragment fills replaced, and `(line, placeholder, why not)` for the rest."""
    lines = text.split("\n")
    heads = {start + 2 + i: head for start, head, rows in tables_in(lines) for i in range(len(rows))}
    frags = {n: first_table((fragments / n).read_text(encoding="utf-8").split("\n"))
             for n in CELL_FRAGMENTS if (fragments / n).is_file()}
    unfilled: list[tuple[int, str, str]] = []

    def replace(m: re.Match) -> str:
        no = text.count("\n", 0, m.start())
        start, end = text.rfind("\n", 0, m.start()) + 1, text.find("\n", m.end())
        before, after = text[start:m.start()], text[m.end():end if end >= 0 else len(text)]
        name = " ".join(m[2].split())
        if before.count("`") % 2:
            return m[0]   # an example inside a code span
        why = HUMAN.get(name, "no fragment is mapped to this placeholder")
        if no in heads:
            row = cells(before + m[0] + after)
            if len(row) != len(heads[no]):
                why = "the row and the table's header differ in length"
            else:
                value, why = cell_value(heads[no], row, before.count("|") - 1, frags)
                if value is not None:
                    return value
        elif name in BLOCKS:
            path = fragments / BLOCKS[name]
            content = path.read_text(encoding="utf-8").rstrip("\n") if path.is_file() else None
            if content is None:
                why = f"{BLOCKS[name]} was not written (an input of `tables` is missing)"
            elif "\n" in content and (before.strip() or after.strip()):
                why = f"{BLOCKS[name]} is a block and the placeholder is inside a sentence: give it a line of its own"
            else:
                return content
        unfilled.append((no + 1, name, why))
        return m[0]
    return MARK.sub(replace, text), unfilled


def _tables(args: argparse.Namespace) -> int:
    fragments = build_fragments(args.check, args.results, args.pilot, args.release)
    write_fragments(args.out, fragments)
    print(f"{len(fragments)} fragments in {args.out}: " + ", ".join(sorted(fragments)))
    return 0


def _fill(args: argparse.Namespace) -> int:
    old = args.doc.read_bytes().decode("utf-8")
    new, unfilled = fill_text(old, args.fragments)
    if args.write:
        if new != old:
            args.doc.write_bytes(new.encode("utf-8"))
        print(f"{args.doc}: {'written' if new != old else 'unchanged'}")
    else:
        diff = "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), str(args.doc),
                                              f"{args.doc} (filled)", lineterm=""))
        print(diff or "no change")
    print(f"{len(unfilled)} placeholders left unfilled:")
    grouped: dict[tuple[int, str], list[str]] = {}   # the cells of one row that fail alike are one line
    for line, name, why in unfilled:
        grouped.setdefault((line, why), []).append(name)
    for (line, why), names in grouped.items():
        print(f"  line {line}: {' | '.join(names)}: {why}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tables", help="write the fragments from the release's outputs")
    t.add_argument("--check", type=Path, metavar="CHECK.json", help="the report of `stbench check`")
    t.add_argument("--results", nargs="+", default=[], metavar="R", help="results.json files or run folders")
    t.add_argument("--pilot", type=Path, metavar="PILOT.md", help="the output of `stbench pilot report`")
    t.add_argument("--release", type=Path, metavar="RELEASE.json", help="the export's release.json (the counts)")
    t.add_argument("--out", type=Path, required=True, metavar="DIR")
    t.set_defaults(func=_tables)
    f = sub.add_parser("fill", help="fill a document's placeholders from the fragments")
    f.add_argument("--doc", type=Path, required=True, metavar="FILE")
    f.add_argument("--fragments", type=Path, required=True, metavar="DIR")
    f.add_argument("--write", action="store_true", help="write FILE (default: print the diff only)")
    f.set_defaults(func=_fill)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as error:   # Refused, a results or compare refusal, an unreadable input
        print(f"fill_report: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
