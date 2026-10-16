"""`path_log`'s answer to `where_was_i` (a module of `baselines_path.PATH_MODULES`; `questions_self` is its gold).

`path_log` keeps the pose of every step it was sent, so it holds the path the question asks about: the base of the
logged step whose clock is nearest the one asked (`table_log.Log.nearest`), 4 decimals. This is the family's reference
answer (`reference: path_log`), which the gates exempt: it reads no picture, depth or detection, and needs neither the
map nor a prior, because the answer is in what the system was sent, not in what it sees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .answer_candidates import UNKNOWN

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def where_was_i(pl: PathLog, question: dict, rec: dict) -> dict:
    """The (x, y) of the base at the logged step nearest the clock asked, `unknown` when nothing is logged."""
    step = pl.log.nearest(question["slots"]["clock"])
    if step is None:
        return dict(UNKNOWN)
    x, y = step["pose"]["pos"][:2]
    return {"type": "position", "xy": [round(x, 4), round(y, 4)]}


PATH_ANSWERERS: dict[str, PathAnswerer] = {"where_was_i": where_was_i}
