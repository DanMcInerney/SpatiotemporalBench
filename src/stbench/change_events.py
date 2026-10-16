"""What changed in a room between two passes: the event rules `changed` is scored by, for the gold and a system alike.

A pass is a task (the world's processes act only in the gaps between tasks, so nothing changes in view). The gold
(`questions_changed`) and the table systems (`table_changed`) each read, thing by thing, where it last stood on the
floor in each of two tasks and then call `events` and `swallow` here, so the event classes, the place test's use and the
swallow loop have one owner; only how a side finds a thing's sightings differs. Neutral: it imports `grid` and `rules`
(the thresholds of "moved"), no truth.

For one thing seen at `a` in pass i and at `b` in pass j (either may be None):
- both, `d = |a - b|`: under `MOVED_NO_M` the same place, nothing; from there to `MOVED_YES_M` a shift that is neither a
  move nor a rest, a don't-care place at each end inside the room; at least `MOVED_YES_M` it moved: an event where it is
  now (`b`) when that is inside the room (and a don't-care place at `a` when that is), else where it was (`a`) when only
  that is inside, a removal;
- `a` only, inside the room: it was there and is not seen later. A removal is an event only when its place was seen
  empty in pass j (`free_j`: the shared place test, `grid.place_free`), else the place is a don't-care one (an unseen
  place says nothing; truth rule 2);
- `b` only, inside the room: an addition, an event only when its place was seen empty in pass i (`free_i`).
A thing with `unsure` places (a glimpse too faint to count, an identical twin, a person) gives no event: those places
are don't-care places. `swallow` then turns every event within `match_m` of a don't-care place into one, again and
again, since an answer there could be true or false by the same doubt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .grid import EPS
from .rules import MOVED_NO_M, MOVED_YES_M  # the thresholds of "moved": one owner, `rules`

Xy = tuple[float, float]
PLACES_M = 0.01  # don't-care places this close are one place


@dataclass(frozen=True)
class Seen:
    """One thing in a room's two passes, as a side read it."""

    a: Xy | None  # its xy at its last floor sighting in pass i (None: none)
    b: Xy | None  # the same in pass j
    radius: float  # min(size[0], size[1]) / 2: the disc `grid.place_free` tests
    unsure: tuple[Xy, ...] = ()  # places it can only be don't-care at (a glimpse, a twin, a person's sightings)


def _round(p: Xy) -> Xy:
    return (round(float(p[0]), 4), round(float(p[1]), 4))


def events(things: Sequence[Seen], inside: Callable[[Xy], bool], free_i: Callable[[Xy, float], bool],
           free_j: Callable[[Xy, float], bool]) -> tuple[list[Xy], list[Xy]]:
    """(events, don't-care places) of the things in a room, before the swallow. `inside(p)` says a place is in the room;
    `free_t(p, radius)` says the place was seen empty in pass t for a thing of that radius."""
    found: list[Xy] = []
    care: list[Xy] = []
    for t in things:
        if t.unsure:
            care += [_round(p) for p in t.unsure if inside(_round(p))]
            continue
        a = None if t.a is None else _round(t.a)
        b = None if t.b is None else _round(t.b)
        a_in, b_in = a is not None and inside(a), b is not None and inside(b)
        if a is not None and b is not None:
            d = math.dist(a, b)
            if d < MOVED_NO_M - EPS:
                continue
            if d < MOVED_YES_M - EPS:
                care += [p for p, here in ((a, a_in), (b, b_in)) if here]
            elif b_in:
                found.append(b)
                if a_in:
                    care.append(a)
            elif a_in:
                found.append(a)
        elif a is not None and a_in:
            (found if free_j(a, t.radius) else care).append(a)
        elif b is not None and b_in:
            (found if free_i(b, t.radius) else care).append(b)
    return found, care


def _places(points: Sequence[Xy]) -> list[Xy]:
    """The places sorted, to 4 decimals, those within `PLACES_M` of one already kept dropped."""
    out: list[Xy] = []
    for p in sorted(_round(p) for p in points):
        if all(math.dist(p, q) > PLACES_M for q in out):
            out.append(p)
    return out


def swallow(found: list[Xy], dont_care: list[Xy], match_m: float) -> tuple[list[Xy], list[Xy]]:
    """(events, don't-care places) once every event within `match_m` of a don't-care place has become one, until none
    moves; both sorted and to 4 decimals, don't-care places within `PLACES_M` merged."""
    kept = sorted(_round(p) for p in found)
    care = [_round(p) for p in dont_care]
    while near := [e for e in kept if any(math.dist(e, p) <= match_m + EPS for p in care)]:
        care += near
        kept = [e for e in kept if e not in near]
    return kept, _places(care)
