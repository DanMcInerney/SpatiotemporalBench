"""The scripted systems' answers kept between checks: `stbench check --answers DIR` and `--reuse DIR` (0.11).

A scorer change moves no oracle, table, current-view or `all_unknown` answer: those systems never read the scorer
(`Baseline.reads_scorer` false), so their answers can be reloaded and only the learned shortcuts (`blind`, `path_log`,
`questionnaire_blind`, `current_view_prior` and their in-pool twins), whose best responses are chosen by the scorer, run
again. The answers of a system over a case are one file, `DIR/<system>/<case id>.json`: `{answers, status, failures}`,
what a `stream.CaseRun` holds that a check reads (`suite_report` and the priors read only the answers).

A file is reloaded only when its answers are exactly the case's questions by qid (a recompile redraws the qids, so the
file of an earlier compile is no answer), and a system only when every case of the check has one. A system that looks at
other cases (`Baseline.needs_prior` or `other_stream`: the priors, `other_case`'s partner) is reloaded only when the
case's group (layout and job) holds the same cases as when the answers were stored, which each file records: a check
narrowed by length or case id changes that group, and those systems are run. Anything else is run.
What this cannot see is a change to a system or to the data under the same qids: the answers are kept for a scorer
change alone, and `--answers` into a fresh folder is the way to keep them after anything else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from . import suite
from .case import CasePaths
from .stream import CaseRun


def _file(directory: Path, system: str, case: CasePaths) -> Path:
    return directory / system / f"{case.id}.json"


def save(directory: Path, system: str, case: CasePaths, run: CaseRun, group: list[str]) -> None:
    """Write the answers of `system` over `case`, one of the cases `group` (their ids) of its layout and job."""
    path = _file(directory, system, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"answers": run.answers, "status": run.status, "failures": run.failures,
                                "group": sorted(group)}, sort_keys=True) + "\n", encoding="utf-8")


def load(directory: Path, system: str, case: CasePaths, group: list[str] | None = None) -> CaseRun | None:
    """The stored run of `system` over `case`, None unless its file is whole and holds an answer to exactly the case's
    questions and, when `group` is given (the ids of the case's group now), was stored with exactly that group."""
    try:
        stored = json.loads(_file(directory, system, case).read_text(encoding="utf-8"))
        answers, status, failures, held = stored["answers"], stored["status"], stored["failures"], stored["group"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if set(answers) != {q["qid"] for q in suite.read_jsonl(case.questions)}:
        return None
    if group is not None and held != sorted(group):
        return None
    return CaseRun(answers, status, failures, {}, {}, None, "")


def load_system(directory: Path, system: str, cases: Iterable[CasePaths],
                groups: dict[str, list[str]] | None = None) -> dict[str, CaseRun] | None:
    """The stored runs of `system` over every one of `cases` by case id, None when any is missing or stale. `groups`
    (case id -> the ids of its group now) is given for a system that looks at other cases, and each file must have been
    stored with that group."""
    found = {}
    for case in cases:
        if (run := load(directory, system, case, None if groups is None else groups[case.id])) is None:
            return None
        found[case.id] = run
    return found
