"""The credit section as the driver builds it for a registered job, and what each job asks, by hand (the files that pin
the section: `test_credit_scope.py`, `test_llm_credit_jobs.py`, `test_subgoal_prompt.py`)."""

from __future__ import annotations

from stbench import job_registry
from stbench.brief import answer_types, asked_families, asks_headings, asks_lists, list_hazards
from stbench.job_registry import JobSpec
from stbench.llm_credit import credit_sentence

# job id -> the families it asks (its decision and look families during the job, its end families after it, the
# questionnaire's default `object_where` for a job that names none), sorted, by hand from each job's declaration
# (`job_*.py`); and the answer types of those families (`FamilySpec.answer_type`) in the order of `PROMPT_SHAPES`
JOB_FAMILIES = {
    "vacuum/1": ("object_where",),
    "vacuum/2": ("clear_path", "ego_mem", "object_where", "open_heading", "tangle_ahead", "task_things"),
    "helper/1": ("approach_mem", "clear_path", "closest_approach", "ego_mem", "ego_now", "object_where", "open_heading",
                 "task_things", "way_to"),
    "amr/1": ("approach_mem", "closest_approach", "closest_person", "ego_mem", "ego_now", "task_things"),
    "drone/1": ("clear_path", "ego_mem")}
JOB_KINDS = {
    "vacuum/1": ("position",),
    "vacuum/2": ("position", "offset", "number", "positions", "heading"),
    "helper/1": ("position", "offset", "number", "positions", "heading"),
    "amr/1": ("offset", "number", "positions"),
    "drone/1": ("offset", "number")}


def section(job: JobSpec) -> str:
    """The credit section as the driver builds it (`cmd_codex`) for `job`: its three numbers and what it asks."""
    return credit_sentence(job.match_m, job.clearance_m, job.reacquire_m, lists=asks_lists(job),
                           headings=asks_headings(job), hazards=list_hazards(job), kinds=answer_types(job),
                           families=asked_families(job))


def text_of(job_id: str) -> str:
    """The credit section the driver builds for the registered job."""
    return section(job_registry.get(job_id))
