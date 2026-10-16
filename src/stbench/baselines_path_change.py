"""`path_log`'s answers to the change families (a module of `baselines_path.PATH_MODULES`; the gold is
`questions_change`).

It reads no picture, so it cannot tell where a thing stood in a task, whether it stood still, or whether it left:
- `object_was` asks where a thing was during a named task. It answers as `object_where` does, from the pool of the
  template's own records (`blind`'s levels with the rooms its briefs named as the first ones) and, as candidates, the
  places its camera faced at a logged step up to the end of the task the question names, `unknown` when it faced none.
- `person_where` asks where a person was at an event of a task, as `object_where` does in the same words: it gets
  `object_where`'s answer.
- `same_place` is the best response over the pool of the template (`answer_keys.levels("path_log", ...)`: the exact
  question, then each slot dropped in turn), `unknown` with no pool. Yes and no are candidates (`answer_candidates`),
  and the best of them is kept only when it beats abstaining on a script left out (`blind.learned_answer`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .answer_candidates import UNKNOWN
from .blind import learned_answer, pool_at

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def where_then(pl: PathLog, question: dict, rec: dict) -> dict:
    """`path_log`'s `object_where` answer to a question in the same slots."""
    return pl.answerers["object_where"](question, rec)


def was_during(pl: PathLog, question: dict, rec: dict) -> dict:
    """The learned answer over the pool of `object_was` records with, as candidates, the places its camera faced at a
    logged step up to the end of the task the question names; `unknown` with none (or no such task)."""
    slots = question["slots"]
    pool, steps = pool_at("path_log", rec, pl.prior), pl.log.task_steps(slots["task"])
    faced = pl.faced_places(slots["kind"], [r["gold"] for r in pool], steps[-1]["k"]) if pool and steps else []
    if not faced:
        return dict(UNKNOWN)
    return learned_answer(pool, [{"type": "position", "xy": [round(p[0], 4), round(p[1], 4)]} for p in faced])


def same_place(pl: PathLog, question: dict, rec: dict) -> dict:
    return learned_answer(pool_at("path_log", rec, pl.prior))


PATH_ANSWERERS: dict[str, PathAnswerer] = {"object_was": was_during, "same_place": same_place,
                                           "person_where": where_then}
