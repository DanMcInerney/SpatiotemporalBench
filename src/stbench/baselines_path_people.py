"""`path_log`'s answers to `closest_person` and `person_mem` (a module of `baselines_path.PATH_MODULES`; the gold is
`questions_people`).

It reads no picture, so it never knows who was in view or whether anyone was. What its own log and the public map offer
about people is nothing: where a person stands is no fact of the layout (a resident or a picker walks where it likes).
So it answers only what other scripts' records of the template say, at the first level of
`answer_keys.levels("path_log", ...)` that holds enough of them, by `blind.learned_answer` (like every learned answer):
a best response is kept only where its held-out gain over abstaining has its interval above 0, else `unknown`. It is
never weaker than `blind`: everything `blind` knows is in `path_log`'s levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .blind import learned_answer, pool_at

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def pooled(pl: PathLog, question: dict, rec: dict) -> dict:
    """The pool's best response alone (`blind.learned_answer`), `unknown` where it does not beat abstaining."""
    return learned_answer(pool_at("path_log", rec, pl.prior))


PATH_ANSWERERS: dict[str, PathAnswerer] = {"closest_person": pooled, "person_mem": pooled}
