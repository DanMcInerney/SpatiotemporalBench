"""The question compiler: a case's window of a script -> private `questions.jsonl` and `gold.jsonl`.

Every gold is code over the timeline, the visibility log and the depth the sensors recorded; nothing is written or
judged by a person or a model, and every answerable gold names its evidence (VISION, truth rules 1 and 2). The
families are modules (`questions_space`, `questions_where`, ...) registered in `questions_text.FAMILIES` through its
`FAMILY_MODULES`; `questionnaire` decides which questions a case asks and in what order; `questions_world` is what the
compile may read.

    questions.jsonl  {qid, when, decision_k, text, template, slots, answer_type, needs}
    gold.jsonl       {qid, family, gold, slice, answer_class, range_m, credit, region, tolerances, metric, scope,
                      reference, critical, subjects, evidence_k, entities, age_s, age_band}

The compile adds two things to a candidate's row that no family module writes (0.10.0, FD-questions-0.10 §4): the
evidence age (`questions_age`: `age_s`, `age_band`, null on an `unknown` row) and, on a harm family's row, the
catastrophe rule of `rules.CATASTROPHES` in its `tolerances` (`rules.with_catastrophe`, which the reported score
charges: `score.reported`). A candidate that already carries a `catastrophe` keeps it.

Questions are in ask order: the during rows (decision points and look points) in keyframe order, then the end rows;
`gold.jsonl` follows. Keyframes in the
gold file (`decision_k`, `evidence_k`, the `w:` subjects) are script numbering, like every private record; the stream
and the texts are case-relative.
"""

from __future__ import annotations

import json
import secrets
from collections import Counter

from . import questionnaire, rules
from .case import CasePaths, CaseRecord, read_case
from .episode import write_jsonl
from .questions_age import age
from .questions_select import Cand
from .questions_text import FAMILIES
from .questions_world import Inputs, World, load_inputs
from .records import GOLD_KEYS, GOLD_OPTIONAL, SCOPES, FamilySpec, check_gold, check_question

GOLDS = questionnaire.GOLDS  # family -> the function that gives its candidate questions with their golds


def _scope(spec: FamilySpec, c: Cand) -> str:
    """The scope of a gold row: the spec's, or for an `"evidence"` family the candidate's own, which must be `now` or
    `memory` (`questions_world.scope_of` gives it)."""
    if spec.scope != "evidence":
        return spec.scope
    if c.scope not in SCOPES:
        raise ValueError(f"the {spec.name} candidate carries scope {c.scope!r}: an evidence family's gold function "
                         f"must set 'now' or 'memory' on every candidate")
    return c.scope


def records(cands: list[Cand], w: World | None = None) -> tuple[list[dict], list[dict]]:
    """(question rows, gold rows) for the questions asked, in the order given, with fresh random qids. Every row must
    pass `records.check_question` or `check_gold`: a compile that makes a row the harness could not read stops. A
    position answer's metric is `xy`, any other's null; the region is the candidate's (a polygon, for credit
    `region`). `w` is the world the candidates were drawn from: it gives each row's evidence age (`questions_age`);
    without it (a hand-built candidate) `age_s` and `age_band` are null."""
    qs, gs = [], []
    for c in cands:
        spec, qid = FAMILIES[c.family], secrets.token_hex(8)
        q = {"qid": qid, "when": spec.when, "decision_k": c.decision_k, "text": spec.render(c.slots),
             "template": spec.name, "slots": c.slots, "answer_type": spec.answer_type, "needs": list(spec.needs)}
        scope = _scope(spec, c)
        tolerances = rules.with_catastrophe(spec.name, c.gold, c.tolerances, c.slots.get("cap"))
        age_s, age_band = (None, None) if w is None else age(w, c, spec.when, scope)
        g = {"qid": qid, "family": spec.name, "gold": c.gold, "slice": c.slice, "answer_class": c.answer_class,
             "range_m": c.range_m, "credit": spec.credit, "region": c.region, "tolerances": tolerances,
             "metric": "xy" if spec.answer_type == "position" else None,
             "scope": scope, "reference": spec.reference, "critical": c.critical,
             "subjects": list(c.subjects), "evidence_k": list(c.evidence_k), "entities": list(c.entities),
             "age_s": age_s, "age_band": age_band}
        assert list(g) == [*GOLD_KEYS, *GOLD_OPTIONAL]
        if problems := check_question(q) + check_gold(g):
            raise ValueError(f"the compile made an invalid {spec.name} row: " + "; ".join(problems))
        qs.append(q)
        gs.append(g)
    return qs, gs


def compile_inputs(inp: Inputs, rec: CaseRecord) -> tuple[list[dict], list[dict]]:
    """Questions and gold for the window `rec` of the script `inp`, in ask order."""
    w = World(inp, rec)
    return records(questionnaire.ask(w), w)


def compile_case(case: CasePaths) -> tuple[list[dict], list[dict]]:
    """Questions and gold of a case, from its files."""
    return compile_inputs(load_inputs(case), read_case(case))


def write(case: CasePaths) -> dict[str, int]:
    """Compile a case and write its `questions.jsonl` and `gold.jsonl`; returns the questions asked per family."""
    qs, gs = compile_case(case)
    write_jsonl(case.questions, qs, sort_keys=True)
    write_jsonl(case.gold, gs)
    return yields(case)


def _gold_rows(case: CasePaths) -> list[dict]:
    return [json.loads(line) for line in case.gold.read_text(encoding="utf-8").splitlines() if line.strip()]


def yields(case: CasePaths) -> dict[str, int]:
    """The questions a case asks per family, from its written gold."""
    return dict(Counter(g["family"] for g in _gold_rows(case)))
