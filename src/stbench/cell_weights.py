"""The `thin` cells a score weighs below 1, in the three forms they travel in (0.11.0, FD-0.11 §3.7).

A (job, ability, column) cell of `n` questions in the dataset weighs `min(1, n / abilities.MIN_ABILITY_N)`
(`ability_counts.weights`); the cells that weigh less than 1 are the `thin` ones, and every other cell weighs 1. A score
(`score_stats.aggregate`) takes them as `{cell: weight}`, or as a set of cells, each weighing 0 (the rule of 0.6.0 to
0.10.0: counted in no mean); a results document stores them as lists ending in the weight (`aggregate`'s `thin`), and a
document of an earlier build lists the cell alone, which reads as weight 0. A cell is `(job, ability, scope)`, or
`(job, ability)` for both columns. Standard library only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

Thin = Mapping | frozenset   # (job, ability, scope) -> weight below 1; or a set of cells, each of weight 0


def thin_weights(thin: Thin) -> dict[tuple, float]:
    """`thin` as `{cell: weight}`: a mapping as given, a set of cells each at weight 0."""
    return dict(thin) if isinstance(thin, Mapping) else dict.fromkeys(thin, 0.0)


def thin_from(listed: Iterable[Iterable]) -> dict[tuple, float]:
    """The `thin` of a stored aggregate (`[job, ability, scope, weight]` each; a cell alone, as an earlier build wrote
    it, weighs 0) as the mapping `aggregate` takes."""
    out = {}
    for entry in map(list, listed):
        names = tuple(x for x in entry if isinstance(x, str))
        out[names] = next((x for x in entry if not isinstance(x, str)), 0.0)
    return out


def weight_of(thin: dict[tuple, float], job: str, name: str, scope: str) -> float:
    """The weight of the (job, ability, scope) cell: its entry in `thin`, else the (job, ability) entry (thin in both
    columns), else 1."""
    return thin.get((job, name, scope), thin.get((job, name), 1.0))
