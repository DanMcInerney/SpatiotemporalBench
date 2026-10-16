"""The identical-item swap check (VISION rule 4 test): unobserved changes must not change a kept gold.

Two things with the same description exchange everything that happens after a moment: their placements from then on,
and the sightings that go with them. Nobody saw the exchange, and nothing the robot recorded differs.
`gold_changes_under_swap` recompiles every candidate question on the swapped world and returns each one that is still
asked but whose gold differs. A question that is no longer asked at all is not a changed gold and is not reported.
The candidates are the questionnaire's (`questionnaire.candidates`): every family of `questions_text.FAMILIES` that
the world's job asks.
"""

from __future__ import annotations

from dataclasses import replace

from . import questionnaire
from .case import CaseRecord
from .questions_world import Inputs, World
from .timeline import Placement, Timeline


def swap_placements(tl: Timeline, a: str, b: str, t: float) -> Timeline:
    """The timeline in which a and b exchange the placements they hold from time t on (one spanning t is cut)."""
    swapped: dict[str, list[Placement]] = {a: [], b: []}
    for own, other in ((a, b), (b, a)):
        for p in tl.placements_of(own):
            if p.t1 is not None and p.t1 <= t:
                swapped[own].append(p)
            else:
                if p.t0 < t:
                    swapped[own].append(replace(p, t1=t))
                swapped[other].append(replace(p, entity=other, t0=max(p.t0, t)))
    return tl.with_placements(swapped)


def swap_rows(rows: list[dict], a: str, b: str, from_k: int) -> list[dict]:
    """The visibility rows in which a and b exchange every sighting at keyframe `from_k` or later."""
    other = {a: b, b: a}
    return [row if row["k"] < from_k else {**row, "entities": [{**e, "id": other.get(e["id"], e["id"])}
                                                                for e in row["entities"]]} for row in rows]


def swap_inputs(inp: Inputs, a: str, b: str, t: float) -> Inputs:
    """The inputs with a and b swapped from time t on, in the timeline and in the visibility log."""
    return replace(inp, tl=swap_placements(inp.tl, a, b, t),
                   vis=swap_rows(inp.vis, a, b, inp.tl.first_keyframe_at(t)))


def _golds(inp: Inputs, rec: CaseRecord) -> dict[str, dict]:
    """Every candidate question of the case by its key (what is asked, never its qid or gold)."""
    return {c.key: {"gold": c.gold, "slice": c.slice} for c in questionnaire.candidates(World(inp, rec))}


def gold_changes_under_swap(inp: Inputs, rec: CaseRecord, a: str, b: str, t: float) -> list[dict]:
    """Questions asked both before and after swapping a and b at time t whose gold changed: should be empty."""
    before, after = _golds(inp, rec), _golds(swap_inputs(inp, a, b, t), rec)
    return [{"question": k, "before": before[k], "after": after[k]}
            for k in sorted(before.keys() & after.keys()) if before[k] != after[k]]
