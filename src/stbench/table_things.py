"""The table systems' answers to `when_seen`, `order` and `distance_between` (a module of
`table_answers.ANSWER_MODULES`), each from the sightings of the tracks a description names (`Store.matching`).

A thing is named by a description, `{"kind", "attributes"}`: the slots themselves for `when_seen`, `slots["first"]` and
`slots["second"]` for the two that name a pair. A sighting is `[k, x, y, z]`, so the answers read only what the store
kept, never a clock or a place the stream did not deliver:
- `when_seen`: the case clock of the step of the latest sighting of the thing (`Store.steps[k]["clock"]`);
- `order`: whether the first thing's earliest sighting is before the second's; `unknown` when the two lie at one step
  (the table cannot order them);
- `distance_between`: the distance in metres, 3 places, between the places of the two things' latest sightings (each
  4 places, as `object_where` gives one): a thing moved after its last sighting keeps the place it was last seen.
Each is `unknown` when a named description matches no track. The gold asks only things the memory reads as the gold
does (`questions_things`: settled things, the later-seen of an `order` pair unseen until after the earlier's first
recognisable sighting), so the oracle reads the same sightings and scores 1.
"""

from __future__ import annotations

import math

from .answer_candidates import UNKNOWN
from .table import Answerer, Store


def _sightings(store: Store, thing: dict) -> list[list[float]]:
    """Every sighting of the tracks the description `thing` names."""
    return [s for t in store.matching(thing["kind"], thing["attributes"]) for s in t.sightings]


def _last(store: Store, thing: dict) -> list[float] | None:
    """The latest sighting (by step `k`) of the thing, or None when it has none."""
    return max(_sightings(store, thing), key=lambda s: s[0], default=None)


def when_seen(store: Store, slots: dict, ctx: dict) -> dict:
    """The clock of the step of the thing's latest sighting; `unknown` when it has none."""
    last = _last(store, slots)
    return dict(UNKNOWN) if last is None else {"type": "time", "clock": store.steps[last[0]]["clock"]}


def order(store: Store, slots: dict, ctx: dict) -> dict:
    """Yes when the first thing's earliest sighting is at an earlier step than the second's; `unknown` when either has
    none or both are first seen at one step."""
    k_first, k_second = (min((s[0] for s in _sightings(store, slots[name])), default=None)
                         for name in ("first", "second"))
    if k_first is None or k_second is None or k_first == k_second:
        return dict(UNKNOWN)
    return {"type": "yesno", "value": k_first < k_second}


def distance_between(store: Store, slots: dict, ctx: dict) -> dict:
    """The distance between the places of the two things' latest sightings; `unknown` when either has none."""
    ends = [_last(store, slots[name]) for name in ("first", "second")]
    if None in ends:
        return dict(UNKNOWN)
    (_, x0, y0, _), (_, x1, y1, _) = ends
    return {"type": "number", "value": round(math.dist((round(x0, 4), round(y0, 4)), (round(x1, 4), round(y1, 4))), 3)}


ANSWERERS: dict[str, Answerer] = {"when_seen": when_seen, "order": order, "distance_between": distance_between}
