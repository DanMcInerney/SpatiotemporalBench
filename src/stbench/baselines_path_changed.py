"""`path_log`'s answer to `changed` (a module of `baselines_path.PATH_MODULES`): the best response over its pool.

It reads no picture, so it cannot tell what moved in a room: it answers with whichever answer scores best over the golds
of other scripts' `changed` questions at the first level of `answer_keys.levels("path_log", ...)` that holds enough of
them (the room and the two tasks, then the room alone, each beside the rooms its briefs named; the task indices go first
among the slots to drop), `unknown` with no pool. The empty list is the first candidate (`answer_candidates`), so a pool
of mostly unchanged rooms is answered "nothing changed".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .blind import learned_answer, pool_at

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def changed(pl: PathLog, question: dict, rec: dict) -> dict:
    return learned_answer(pool_at("path_log", rec, pl.prior))


PATH_ANSWERERS: dict[str, PathAnswerer] = {"changed": changed}
