"""The table systems' answer to `where_was_i` (a module of `table_answers.ANSWER_MODULES`; `questions_self`: the gold).

The family asks where the robot's base was at a clock the stream delivered a minute or more ago, so a system that kept
its log answers from the log: the logged step whose clock is nearest the one asked (`table_log.Log.nearest`; a clock
the stream delivered names exactly one step, and `MemoryTable` has already put `other_case`'s clock onto the stream it
was fed), and that step's delivered base, 4 decimals. No picture, depth or detection is read. A store with no step
that has a clock answers `unknown`. `current_view`'s store holds the latest frame alone, so there it is the frame's
base, which is where the robot is now: the memoryless answer the memory-needed gate holds under the floor.
"""

from __future__ import annotations

from .answer_candidates import UNKNOWN
from .table import Answerer, Store


def where_was_i(store: Store, slots: dict, ctx: dict) -> dict:
    """The (x, y) of the base at the logged step nearest `slots["clock"]`, `unknown` when nothing is logged."""
    step = store.nearest(slots["clock"])
    if step is None:
        return dict(UNKNOWN)
    x, y = step["pose"]["pos"][:2]
    return {"type": "position", "xy": [round(x, 4), round(y, 4)]}


ANSWERERS: dict[str, Answerer] = {"where_was_i": where_was_i}
