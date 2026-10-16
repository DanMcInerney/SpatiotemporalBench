"""The family-completeness test (plan §4.8): a question family is a registry entry (`questions_text.FAMILIES`) and every
other lane's registry has to know it. This is the one place that reads them all, so a family added to one lane and not
to another fails here, by name, before a run does."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from stbench import answers, brief, job_registry, llm_render, questions, table_answers
from stbench.baselines_path import PathLog
from stbench.questions_text import FAMILIES
from stbench.records import FamilySpec, check_family

NOW = {"k": 0, "clock": "2026-03-02T08:00:00"}


def gaps(spec: FamilySpec) -> list[str]:
    """What the registries lack of one family: a spec `records.check_family` accepts; a gold function (the compiler's),
    an answerer (`table_answers`, which every table system calls) and `path_log`'s own; an answer type with a shape in
    the reference agent's prompt, which that agent's ask message shows for a question of the type; and, for a during
    family, a phrase in the brief's sentence about what it asks."""
    out = [f"records.check_family: {problem}" for problem in check_family(spec)]
    if spec.name not in questions.GOLDS:
        out.append("no gold function in questions.GOLDS")
    if spec.name not in table_answers.ANSWERERS:
        out.append("no answerer in table_answers.ANSWERERS")
    if spec.name not in PathLog(Path("."), Path("."), map=None, events={}).answerers:
        out.append("no answerer in path_log's")
    if spec.answer_type not in answers.TYPES:
        out.append(f"answer type {spec.answer_type!r} is not in answers.TYPES")
    elif spec.answer_type not in answers.PROMPT_SHAPES:
        out.append(f"answer type {spec.answer_type!r} has no shape in answers.PROMPT_SHAPES")
    else:
        question = {"text": "?", "answer_type": spec.answer_type}
        if answers.PROMPT_SHAPES[spec.answer_type] not in llm_render.ask_message(spec.when, NOW,
                                                                                  llm_render.labelled([question])):
            out.append("its answer type's shape is not in the reference agent's ask message")
    if spec.when == "during" and spec.name not in brief.DURING_PHRASES:
        out.append("no phrase in brief.DURING_PHRASES")
    return out


def asked_unknown(jobs) -> set[tuple[str, str, str]]:
    """(job, decision kind, family) for each family the jobs' decisions ask that `FAMILIES` does not hold."""
    return {(job.id, kind, family) for job in jobs for kind, families in job.decisions.items() for family in families
            if family not in FAMILIES}


def test_families_are_complete():
    """Contract (plan §4.8, VISION rule 5): every family of `questions_text.FAMILIES` passes `records.check_family` and
    has a gold function, an answerer for the table systems and one for `path_log`, an answer type in `answers.TYPES`
    with a prompt shape that the Codex agent's ask message shows, and, if it is a during family, a phrase in the brief;
    and every family a registered job's `decisions` asks is in `FAMILIES`. Regression caught: a family added to the
    registry and left out of a gold function, an answerer, the prompt or the brief (found only when a run asks it), or
    a job that asks a family that does not exist."""
    assert {"path_ahead", "object_where"} <= set(FAMILIES)   # the scan reads the registry, so it is not vacuous
    assert {name: found for name, spec in FAMILIES.items() if (found := gaps(spec))} == {}
    assert asked_unknown(job_registry.all_jobs()) == set()   # `vacuum/1` asks none in M1; the scan below can fail


def test_the_scan_names_each_gap_of_a_family_the_registries_do_not_know():
    """Contract: `gaps` finds what is missing, so a family that passed no registry is reported for every one. A copy of
    `path_ahead` under the name `ghost` (no gold function, answerer, `path_log` answerer or brief phrase) lacks those
    four, and one of the same name with an answer type no list holds lacks the type's place too, and fails
    `check_family`; and `asked_unknown` names a family a job's decision asks that the registry lacks (the registered
    job asks none in M1, so only a job that asks `ghost` shows the scan can fail). Regression caught: a scan that
    returns nothing whatever the registries hold, which would make the test above pass for good."""
    ghost = replace(FAMILIES["path_ahead"], name="ghost")
    assert gaps(ghost) == ["no gold function in questions.GOLDS", "no answerer in table_answers.ANSWERERS",
                           "no answerer in path_log's", "no phrase in brief.DURING_PHRASES"]
    odd = gaps(replace(ghost, answer_type="blob"))
    assert "answer type 'blob' is not in answers.TYPES" in odd
    assert any(line.startswith("records.check_family:") for line in odd)
    vacuum = job_registry.get("vacuum/1")
    asks = replace(vacuum, decisions={"before_doorway": ("path_ahead", "ghost")})
    assert asked_unknown([asks]) == {("vacuum/1", "before_doorway", "ghost")}
    assert asked_unknown([replace(asks, decisions={"before_doorway": ("path_ahead",)})]) == set()
