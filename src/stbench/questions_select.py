"""Candidate questions, and the moments an end family names them in.

A `Cand` is a question the world could ask, with its gold, before the questionnaire draws. Choice is by stable hash,
never by drawing from a stream, so removing or adding one candidate (as the identical-item swap does) cannot reshuffle
the others (`questionnaire`). A `Moment` is what an end family names: the unit the questionnaire visits, with the
candidates it could ask there in two pools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Cand:
    """A question that the world could ask, with its gold, before selection."""

    family: str
    slots: dict
    gold: dict
    slice: str = "answerable"
    evidence_k: tuple[int, ...] = ()
    entities: tuple[str, ...] = ()
    tolerances: dict = field(default_factory=dict)
    decision_k: int | None = None  # a during question's decision point (script k); None for an end question
    subjects: tuple[str, ...] = ()  # the gold row's subjects (records.py): what the question names or depends on
    critical: bool = True  # task-critical against the job's relevance list
    answer_class: str | None = None  # the declared class a draw balanced (`records.ANSWER_CLASSES`); None: the family
    # declares none (and the gold row's `answer_class` is null)
    # The slant range r (metres) the row's relative rungs were built from, for the distance report by band (r cannot
    # be recovered from the rungs once a floor or the widening binds); None: the family has none (`same_place`,
    # `task_things`, an unknown row). `ego_now`: the slant at the ask (`World.slant`). `ego_mem`: the larger of the
    # slant at the nearest recognisable sighting at the gold place and the slant now. `object_where` and `object_was`:
    # the nearest recognisable slant at the asked place (`questions_where.nearest_slant_at`). docs/format.md's
    # `range_m` row says the same.
    range_m: float | None = None
    scope: str | None = None  # "now" or "memory": set by a family whose spec scope is "evidence", else unused
    region: list | None = None  # a polygon of [x, y] points: the gold row's `region`, for credit `region`

    @property
    def key(self) -> str:
        """Identifies the question by what is asked, never by its qid or gold: the family and slots, and for a during
        question the decision point too (its slots hold only the cap, so two decisions would share one key; map
        Q-H11)."""
        parts = [self.family, self.slots] if self.decision_k is None else [self.family, self.slots, self.decision_k]
        return json.dumps(parts, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Moment:
    """What an end family names: a time, a task and a time, a kind, a pair. `answerable` and `unknown` are the
    candidates asked there by slice; `unknown` is empty for a family that draws no slice."""

    key: str  # what is named, for the draws: "1|when you finished cleaning", "hall|0|2", a kind
    answerable: list[Cand]
    unknown: list[Cand]
