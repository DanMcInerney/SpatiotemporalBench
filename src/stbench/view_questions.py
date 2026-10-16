"""A case's questions for `stbench view`: what was asked, what the solver answered, the gold and the score, in one row.

The questions and the gold are the dataset's (`private/<cid>/questions.jsonl`, `gold.jsonl`: line i of one is line i of
the other, in ask order, `docs/format.md`); the answers and the scored rows are the run's (`answers.jsonl`,
`rows.jsonl`). The gold is public in this release and this view is read after a run, never by a system. A question the
run did not ask (`--during off`, a case that ended first) is kept with `asked` false, so the list is the case's whole
quiz in order. Without the dataset's files the rows still list what the run holds (qid, family, answer, outcome, score)
and the page shows no text or gold. `evidence` is the script keyframes the gold is read from, each with the link to its
picture when the dataset holds it (`scripts/<sid>/sensors/rgb/%06d.jpg`, script numbering, as `evidence_k` is).
"""

from __future__ import annotations

from pathlib import Path

from .case import PRIVATE, SCRIPTS
from .view_files import json_object, jsonl

OUTCOMES = ("correct", "partial", "wrong", "abstained")


def read_case_files(dataset: Path | None, cid: str, problems: list[str]) -> tuple[dict | None, list[dict], dict]:
    """(`case.json` or None, the question rows, the gold rows by qid) of the dataset's case `cid`."""
    if dataset is None:
        return None, [], {}
    folder = dataset / PRIVATE / cid
    record = json_object(folder / "case.json", problems)
    questions = jsonl(folder / "questions.jsonl", problems)
    golds = {gold.get("qid"): gold for gold in jsonl(folder / "gold.jsonl", problems)}
    if not questions:
        problems.append(f"the dataset has no questions for this case (private/{cid}/questions.jsonl)")
    return record, questions, golds


def evidence(dataset: Path | None, script: str | None, ks: list) -> list[dict]:
    """`[{"k", "url"}]` for the keyframes `ks`: the picture's `file:` link, null when the dataset has no such file."""
    pictures = None if dataset is None or script is None else dataset / SCRIPTS / script / "sensors" / "rgb"
    found = []
    for k in ks:
        path = None if pictures is None or not isinstance(k, int) else pictures / f"{k:06d}.jpg"
        found.append({"k": k, "url": path.resolve().as_uri() if path is not None and path.is_file() else None})
    return found


def merge(dataset: Path | None, script: str | None, k0: int | None, questions: list[dict], golds: dict,
          rows: list[dict], answers: list[dict]) -> list[dict]:
    """One dict a question, in ask order (the dataset's), then any qid only the run knows."""
    row_by = {row.get("qid"): row for row in rows}
    answer_by = {rec.get("qid"): rec.get("answer") for rec in answers}
    asked = [q.get("qid") for q in questions]
    order = [*asked, *[qid for qid in [*row_by, *answer_by] if qid not in asked]]
    by_qid = {q.get("qid"): q for q in questions}
    out = []
    for qid in dict.fromkeys(order):
        question, gold, row = by_qid.get(qid, {}), golds.get(qid, {}), row_by.get(qid)
        step = question.get("decision_k")
        out.append({
            "qid": qid, "when": question.get("when") or (row or {}).get("when"),
            "step": step - k0 if isinstance(step, int) and isinstance(k0, int) else None,
            "family": question.get("template") or gold.get("family") or (row or {}).get("family"),
            "text": question.get("text"), "answer_type": question.get("answer_type"),
            "asked": qid in answer_by or row is not None, "answer": answer_by.get(qid), "gold": gold.get("gold"),
            "credit": gold.get("credit"), "tolerances": gold.get("tolerances"), "metric": gold.get("metric"),
            "region": gold.get("region"), "scope": gold.get("scope") or (row or {}).get("scope"),
            "reference": gold.get("reference"), "subjects": gold.get("subjects"),
            "slice": gold.get("slice") or (row or {}).get("slice"),
            "answer_class": gold.get("answer_class", (row or {}).get("answer_class")), "range_m": gold.get("range_m"),
            "critical": gold.get("critical") if "critical" in gold else (row or {}).get("critical"),
            "outcome": (row or {}).get("outcome"), "score": (row or {}).get("score"), "s": (row or {}).get("s"),
            "evidence": evidence(dataset, script, gold.get("evidence_k") or []),
        })
    return out


def counts(rows: list[dict]) -> dict:
    """The outcomes of the scored rows (`n`, one count each of OUTCOMES) and the mean reported score (None if none)."""
    found = {name: sum(1 for row in rows if row.get("outcome") == name) for name in OUTCOMES}
    scores = [row["score"] for row in rows if isinstance(row.get("score"), int | float)]
    return {"n": len(rows), **found, "mean": sum(scores) / len(scores) if scores else None}
