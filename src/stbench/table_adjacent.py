"""The table systems' answer to `adjacent` (a module of `table_answers.ANSWER_MODULES`; `questions_adjacent`: the gold).

"Give a point in the room that connects the kitchen to the bedroom." A table system is an instrument, so it reads room
identity from the case layout's map on every layout (`ctx["map"]`: a perfect perceiver's knowledge of which room is
which), and the rest from what it was delivered, through the functions the gold calls (`adjacency.connector`,
`crossed`, `toured`, `answer_point`):
- the regions `slots["a"]` and `slots["b"]` name are the map's regions of those kinds, each unique, else `unknown`;
- their connector R is the one region joined to both (`adjacency.connector`), else `unknown`;
- the delivered base positions of every step up to the ask cross A-R and R-B (`adjacency.crossed`, either direction),
  else `unknown`: the robot's own path is the only thing that shows the connection;
- A and B were each toured in the cells observed so far (`Store.latest_states`, any state), else `unknown`.
Then the answer is the point of R the gold names (`adjacency.answer_point`).
`current_view` holds the last frame alone, and the gold asks only when R lies in none of the end look window's
positions, so its path never shows the connection and it answers `unknown`. The question's `space` slot is words only
and is never read. The gold also leaves out rooms a brief names, the job's relevance lists or the robot started in, and
a window whose first position lies in R or whose end look window has a position in it: those decide which questions are
asked, not what a system can derive, so the table does not read them.
"""

from __future__ import annotations

from . import adjacency
from .answer_candidates import UNKNOWN
from .table import Answerer, Store


def adjacent(store: Store, slots: dict, ctx: dict) -> dict:
    """The point of the room that connects the two named rooms, when the table's own path and cells show it."""
    if ctx.get("map") is None:
        raise ValueError("adjacent names rooms of the layout: the system needs the case layout's map (option 'map')")
    m = ctx["map"]
    rooms = [[r for r in m.regions if r.kind == slots[name]] for name in ("a", "b")]
    if any(len(found) != 1 for found in rooms):
        return dict(UNKNOWN)
    (a,), (b,) = rooms
    if (via := adjacency.connector(m, a.id, b.id)) is None:
        return dict(UNKNOWN)
    k = ctx["now"]["k"]
    path = [tuple(step["pose"]["pos"][:2]) for step in store.steps_until(k)]
    if not {frozenset((a.id, via)), frozenset((via, b.id))} <= adjacency.crossed(m, path):
        return dict(UNKNOWN)
    cells = store.latest_states(k)
    if not (adjacency.toured(a, cells) and adjacency.toured(b, cells)):
        return dict(UNKNOWN)
    point = adjacency.answer_point(m.region(via))
    return dict(UNKNOWN) if point is None else {"type": "position", "xy": list(point)}


ANSWERERS: dict[str, Answerer] = {"adjacent": adjacent}
