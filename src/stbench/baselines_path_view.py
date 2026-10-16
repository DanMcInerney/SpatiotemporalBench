"""`path_log`'s answer to `out_of_view` (a module of `baselines_path.PATH_MODULES`; `questions_view` is its gold).

`path_log` never saw a thing: no picture, depth or detection reaches it, so it cannot list the movable things last seen
near the robot. It answers from other scripts' records alone, the best response over the pool its key finds
(`answer_keys.levels("path_log", ...)`: the template, the phase and the pose key, then coarser) and `unknown` with none.
No extra candidate: the public map knows the fixed furniture, which the question excludes, and nothing of what moved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .blind import learned_answer, pool_at

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def out_of_view(pl: PathLog, question: dict, rec: dict) -> dict:
    """The learned answer over the pool of the ask's key (phase and pose, then coarser), `unknown` with none."""
    return learned_answer(pool_at("path_log", rec, pl.prior))


PATH_ANSWERERS: dict[str, PathAnswerer] = {"out_of_view": out_of_view}
