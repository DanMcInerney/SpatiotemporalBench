"""The release document (`release.json`) and the dataset card (`stbench export`, A4 and A6).

`release_doc` is the one record of what an export holds: the package version, the counts per (layout, job), by ability
too, and the facts of each case (its length and its question count). It carries no timestamp, so two exports of one
dataset are byte-identical. The card is the package's template (`stbench/release/dataset-card.md`) with one block
generated from that document, the lines between `<!-- counts -->` and `<!-- /counts -->`; everything else,
`<DATASET_ID>` included, is the template's own words.
"""

from __future__ import annotations

import re
from importlib import resources

from .case import LENGTHS
from .export_files import Case, Script

TEMPLATE = "dataset-card.md"
COUNTS = re.compile(r"<!-- counts -->\n.*?<!-- /counts -->", re.DOTALL)


def release_doc(*, package: str | None, scripts: list[Script], cases: list[Case], results: list[str]) -> dict:
    """The release document: the package version; per (layout, job) its scripts, cases by length, cases that ask
    nothing, questions (in all, and by the ability their family measures) and keyframes; per case its script, layout,
    job, length and question count; the names of the shipped results. Sorted, so equal inputs give equal bytes."""
    by_script = {s.id: s for s in scripts}
    groups: dict[tuple[str, str], dict] = {}
    for s in scripts:
        g = groups.setdefault((s.layout, s.job), {"layout": s.layout, "job": s.job, "scripts": 0,
                                                  "cases": dict.fromkeys(LENGTHS, 0), "asked_nothing": 0,
                                                  "questions": 0, "abilities": {}, "frames": 0})
        g["scripts"] += 1
        g["frames"] += s.frames
    for c in cases:
        g = groups[by_script[c.script].layout, by_script[c.script].job]
        g["cases"][c.length] += 1
        g["questions"] += c.questions
        for ability, n in c.abilities.items():
            g["abilities"][ability] = g["abilities"].get(ability, 0) + n
        g["asked_nothing"] += int(c.questions == 0)
    rows = sorted(({"id": c.id, "script": c.script, "layout": by_script[c.script].layout,
                    "job": by_script[c.script].job, "length": c.length, "questions": c.questions}
                   for c in cases), key=lambda r: (r["layout"], r["job"], r["id"]))
    return {"package": package,
            "groups": [{**g, "abilities": dict(sorted(g["abilities"].items()))}
                       for g in (groups[key] for key in sorted(groups))], "cases": rows, "results": sorted(results)}


def _by_ability(groups: list[dict]) -> list[str]:
    """The questions by ability: a row per (layout, job) and a total, a column per ability any group asks, in name
    order (a count of 0 where a job asks none). No lines for a release document that counts none (an older one has
    no `abilities`)."""
    names = sorted({a for g in groups for a in g.get("abilities", {})})
    if not names:
        return []
    head = ["Layout", "Job", *names]
    rows = [head, ["---"] * len(head)]
    rows += [[f"`{g['layout']}`", f"`{g['job']}`", *(f"{g.get('abilities', {}).get(a, 0):,}" for a in names)]
             for g in groups]
    rows.append(["**Total**", "", *(f"{sum(g.get('abilities', {}).get(a, 0) for g in groups):,}" for a in names)])
    return ["| " + " | ".join(row) + " |" for row in rows]


def counts_block(release: dict) -> str:
    """The card's counts: a table with a row per (layout, job) and a total, the same questions counted by ability, and
    the package version, all read from `release`. Ends with a newline."""
    head = ["Layout", "Job", "Scripts", "Cases (short / medium / long)", "Cases asking nothing", "Questions", "Frames"]
    rows = [head, ["---"] * len(head)]
    for g in release["groups"]:
        c = g["cases"]
        rows.append([f"`{g['layout']}`", f"`{g['job']}`", str(g["scripts"]),
                     " / ".join(str(c[length]) for length in LENGTHS), str(g["asked_nothing"]),
                     f"{g['questions']:,}", f"{g['frames']:,}"])
    groups = release["groups"]
    rows.append(["**Total**", "", str(sum(g["scripts"] for g in groups)),
                 " / ".join(str(sum(g["cases"][length] for g in groups)) for length in LENGTHS),
                 str(sum(g["asked_nothing"] for g in groups)), f"{sum(g['questions'] for g in groups):,}",
                 f"{sum(g['frames'] for g in groups):,}"])
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    version = f"stbench {release['package']}" if release["package"] else "stbench"
    by_ability = _by_ability(groups)
    ability_table = "\n\nQuestions by ability:\n\n" + "\n".join(by_ability) if by_ability else ""
    return "\n".join(lines) + f"{ability_table}\n\nExported by {version}.\n"


def card_bytes(release: dict, template: str | None = None) -> bytes:
    """The card: `template` (default: the package's) with its counts block generated from `release`, as UTF-8 with LF
    line ends. ValueError unless the template holds exactly one counts block."""
    text = template if template is not None else \
        (resources.files("stbench") / "release" / TEMPLATE).read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")
    if len(COUNTS.findall(text)) != 1:
        raise ValueError("the dataset card template must hold exactly one <!-- counts --> ... <!-- /counts --> block")
    block = "<!-- counts -->\n" + counts_block(release) + "<!-- /counts -->"
    return COUNTS.sub(lambda _: block, text).encode("utf-8")
