"""How many questions of each ability a dataset asks of each job in each column, and the weight each cell carries
(0.6.0, FD-0.6.0 §10; per column since 0.8.0, FD-task-questions review §5; a continuous weight since 0.11.0, FD-0.11
§3.7).

A (job, ability, column) cell of `n` questions (`now` or `memory` is the column, the gold's scope) weighs
`min(1, n / abilities.MIN_ABILITY_N)` in the headline (`weights`): a full cell from 20 questions, a fraction of one
below. It used to weigh nothing under 20 (a diagnostic there), so a cell flipped in or out of the headline with one
question more or less (0.10.0: the helper's clearance `now` went 26, 14, 51 across its compiles); the weight moves
continuously. The line is per (job, ability, column) because a column counts its cells by weight (`score_stats`): an
ability asked in both scopes, as clearance is, has a cell in each, and a cell of fifteen questions would weigh as much
as a cell of a hundred. The count is over the golds of every compiled case of the dataset, whatever a run selected or
asked (`--cases`, a pilot, `--during off`), so a subset keeps the cells of the release it is a part of; a score may take
the cells from another dataset (`--cells-from DEVROOT`, the dev root), so the fast tier and the held-out set have the
dev root's cells. It reads only the golds' `family` and `scope`, which the scorer already holds, and the job of each
case's script.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import check, suite
from .abilities import DIAGNOSTIC, MIN_ABILITY_N, ability_of

Cell = tuple[str, str, str]   # (job, ability, scope)


def counts(data: Path) -> Counter[Cell]:
    """The golds of every compiled case of `data` that asks a question, by (job, ability, scope)."""
    found: Counter[Cell] = Counter()
    for case in check.compiled_cases(data)[0]:
        job = suite.case_info(case)["job"]
        found.update((job, ability_of(gold["family"]), gold["scope"]) for gold in suite.read_jsonl(case.gold))
    return found


def require_cases(root: Path) -> None:
    """ValueError unless `root` holds a compiled case that asks a question: the root `--cells-from` names. A path
    mistyped, or not a dataset, has no index (`check.read_index` reads `{}`) and so no cell, which would weigh every
    cell 1 and score the run on other cells than the dev root's without a word."""
    if not check.compiled_cases(root)[0]:
        raise ValueError(f"--cells-from {root}: no compiled case that asks a question (a dev root holds "
                         f"private/index/*.json and each case's questions.jsonl and gold.jsonl)")


def weights(data: Path) -> dict[Cell, float]:
    """The weight of each (job, ability, column) cell the dataset asks: `min(1, n / MIN_ABILITY_N)` for its `n`
    questions (a cell the dataset does not ask has none)."""
    return {cell: min(1.0, n / MIN_ABILITY_N) for cell, n in counts(data).items()}


def thin_abilities(data: Path) -> frozenset[Cell]:
    """The (job, ability, column) cells the dataset asks fewer than `MIN_ABILITY_N` questions of."""
    return frozenset(cell for cell, w in weights(data).items() if w < 1)


def thin_rule(data: Path) -> tuple[dict[Cell, float], bool]:
    """(the cells a score weighs below 1, each with its weight, whether the rule is relaxed). The rule is relaxed, and
    no cell applied (every one weighs 1), when every non-diagnostic (job, ability, column) cell of the whole dataset
    is thin: a development-size dataset, whose headline then is not the release statistic. It is decided here once
    for the dataset, never from the rows of a run, so a subset run and the full run agree."""
    found = weights(data)
    thin = {cell: w for cell, w in found.items() if w < 1}
    counted = {cell for cell in found if cell[1] not in DIAGNOSTIC}
    if counted and counted <= thin.keys():
        return {}, True
    return thin, False
