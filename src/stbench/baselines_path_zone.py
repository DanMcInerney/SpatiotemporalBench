"""`path_log`'s answers to the zone families (a module of `baselines_path.PATH_MODULES`; `questions_zone` is the gold).

`path_log` reads no depth, so it never saw what stands in a zone, and a zone's state varies from case to case: nothing
in its pose log or the public map tells whether a bay held a pallet. Its own pose log can tell where it stood, which the
pool's key already holds (`answer_keys.levels("path_log", ...)`: the phase and the pose key for a during question, the
template and slots with the rooms for an end one). The answer of each of `zone_clear`, `clearance_then` and
`slots_empty` is the best response over the pool of other scripts' gold rows at the first supported level, `unknown`
with none, and no extra candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .blind import learned_answer, pool_at

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def from_pool(pl: PathLog, question: dict, rec: dict) -> dict:
    """The learned answer over the pool of the ask's key (phase and pose, then coarser), `unknown` with none."""
    return learned_answer(pool_at("path_log", rec, pl.prior))


PATH_ANSWERERS: dict[str, PathAnswerer] = dict.fromkeys(("zone_clear", "clearance_then", "slots_empty"), from_pool)
