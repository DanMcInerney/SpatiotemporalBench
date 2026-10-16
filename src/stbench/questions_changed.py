"""`changed` (end, `memory`): where in a room a thing was added, moved or removed between two of the window's tasks.

A moment is a room and two tasks at least two apart (`j - i >= 2`: more elapsed time, so more change; a medium window's
tasks stay with `object_where`), the room named by its kind, which is unique in the layout, and toured in both tasks:
at least `adjacency.TOURED` of its inside fine cells observed, in any state, in that task's keyframes
(`adjacency.toured`, the rule `adjacent` asks too); else there is no moment, a dead band. The world's processes act
only between tasks, so the tasks are the passes the question compares.

What a side reads of a thing (`change_events.Seen`; the table systems read their tracks the same way) is where it last
stood on the floor in each of the two tasks: a sighting counts only when the thing stood on the floor then
(`World.on_floor`), so a thing moved from a table to the floor has no `a` and a `b`, and a mug moved on a table has no
sighting at all. `a` and `b` come from the last seen-bound sighting (`Sightings.seen`) in task i and in task j. A
dead-band sighting in a task with no seen one adds its place to `unsure` (a thing only glimpsed there has no place the
robot can vouch for); one in a task that has a seen sighting adds nothing, since nothing moves within a task, so it is
at the thing's own seen place. Every floor sighting of a thing whose description another entity sighted in the window
shares is `unsure` too (the table tells things apart by description, so it can tell that twin too; an entity no frame
shows changes nothing, truth rule 2), and so is every sighting of a mover (a person is somewhere else in every pass,
which is not a change). `change_events.events` and `swallow` then give the gold: events by the shared rules, "removed"
and "added" certified by `grid.place_free` over the latest cells of the other task, and don't-care places of radius
`MATCH_M`. Nothing is read of the timeline but descriptions and sizes, and the placements at keyframes where a thing
was seen. The gold's "removed" and "added" are certified from the recorded depth, but `needs` is what a system needs to
answer: the camera's picture and the pose.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` (`moments`, an end family's) and `SLICED` (none).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import combinations

from . import adjacency, change_events, grid
from .change_events import Seen, Xy
from .descriptions import Key, desc_key
from .geometry import point_in_polygon
from .mapspec import Region
from .questions_select import Cand, Moment
from .questions_words import ORDINALS
from .questions_world import World
from .records import FamilySpec
from .rules import MATCH_M
from .timeline import Entity

THINGS = ("light", "carried", "clutter")  # the mobilities that are things; a mover's sightings are don't-care places
Read = tuple[Seen, tuple[int, ...]]  # one entity as `change_events` takes it, and the keyframes of its `a` and `b`


def _render(slots: dict) -> str:
    first, second = (ORDINALS[i] for i in slots["tasks"])
    return (f"Where in the {slots['room']} did something change between your {first} task and your {second} task? "
            f"List where each thing was added or moved to, and where each was removed from.")


class _Reader:
    """What `changed` reads of one window, each piece computed once however many moments ask for it."""

    def __init__(self, w: World):
        self.w = w
        self.span = {i: (k0, k1) for i, k0, k1 in w.tasks}
        self.sighted: dict[Key, list[str]] = defaultdict(list)  # the entities some frame shows, by description
        for e in w.tl.entities:
            if w.sighted(e.id):
                self.sighted[desc_key(e.kind, e.attributes)].append(e.id)
        self._cells: dict[int, dict[tuple[int, int], tuple[str, int]]] = {}

    def cells(self, task: int) -> dict[tuple[int, int], tuple[str, int]]:
        """The latest state of every fine cell the task's keyframes observed."""
        if task not in self._cells:
            self._cells[task] = self.w.latest_cells(range(*self.span[task]))
        return self._cells[task]

    def toured(self, region: Region, task: int) -> bool:
        """The room is toured in the task (`adjacency.toured` over the task's cells)."""
        return adjacency.toured(region, self.cells(task))

    def xy(self, e: Entity, k: int) -> Xy:
        """The placement's (x, y) at keyframe `k`, where the entity was sighted."""
        place = self.w.xy_at(e.id, k)
        if place is None:
            raise ValueError(f"{e.id} is sighted at keyframe {k}, where the timeline has no placement of it")
        return place

    def sightings(self, e: Entity, task: int) -> tuple[list[int], list[int]]:
        """(seen, dead band) keyframes of the entity in the task, sorted."""
        k0, k1 = self.span[task]
        return tuple([k for k in ks if k0 <= k < k1] for ks in (self.w.sight.seen(e.id), self.w.sight.band(e.id)))

    def thing(self, e: Entity, i: int, j: int) -> Read | None:
        """A light, carried or clutter entity in tasks i and j; None when it has no floor sighting in either."""
        seen, band, last, unsure = {}, {}, {}, []
        for t in (i, j):
            seen[t], band[t] = ([k for k in ks if self.w.on_floor(e.id, k)] for ks in self.sightings(e, t))
            last[t] = seen[t][-1] if seen[t] else None
            if last[t] is None:  # only glimpsed in this task; a glimpse beside a seen sighting is at its place
                unsure += [self.xy(e, k) for k in band[t]]
        if any(other != e.id for other in self.sighted[desc_key(e.kind, e.attributes)]):
            unsure = [self.xy(e, k) for t in (i, j) for k in seen[t] + band[t]]
        a, b = (None if last[t] is None else self.xy(e, last[t]) for t in (i, j))
        if a is None and b is None and not unsure:
            return None
        radius = min(e.size[0], e.size[1]) / 2
        return Seen(a, b, radius, tuple(sorted(set(unsure)))), tuple(k for k in (last[i], last[j]) if k is not None)

    def mover(self, e: Entity, i: int, j: int) -> Read | None:
        """A mover in tasks i and j: every sighting of it is a place it can only be don't-care at."""
        places = {self.xy(e, k) for t in (i, j) for ks in self.sightings(e, t) for k in ks}
        return (Seen(None, None, 0.0, tuple(sorted(places))), ()) if places else None

    def read(self, e: Entity, i: int, j: int) -> Read | None:
        """The entity as `change_events` takes it: a thing, a mover's places, or nothing (fixed, structure)."""
        if e.mobility == "mover":
            return self.mover(e, i, j)
        return self.thing(e, i, j) if e.mobility in THINGS else None

    def cand(self, region: Region, i: int, j: int) -> Cand:
        """The question for the room and the two tasks, with its gold."""
        poly = list(region.polygon)

        def inside(p: Xy) -> bool:
            return point_in_polygon(p[0], p[1], poly)

        def free(task: int) -> Callable[[Xy, float], bool]:
            return lambda p, radius: grid.place_free(self.cells(task), p, radius)

        found: list[tuple[Xy, str, tuple[int, ...]]] = []  # (an event, its thing, the keyframes it read)
        care: list[Xy] = []
        for e in self.w.tl.entities:
            if (read := self.read(e, i, j)) is not None:
                events, dont_care = change_events.events([read[0]], inside, free(i), free(j))
                found += [(p, e.id, read[1]) for p in events]
                care += dont_care
        kept, places = change_events.swallow([p for p, _, _ in found], care, MATCH_M)
        asked = [(thing, ks) for p, thing, ks in found if p in kept]
        evidence = {self.span[i][0], self.span[j][0]} | {k for _, ks in asked for k in ks}
        gold: dict = {"type": "positions", "items": [{"xy": list(p)} for p in kept]}
        if places:
            gold["dont_care"] = [{"xy": list(p), "radius": MATCH_M} for p in places]
        times = [f"w:{self.span[t][0]}-{self.span[t][1] - 1}" for t in (i, j)]
        subjects = (f"r:{region.id}", *times, *(f"e:{thing}" for thing in sorted({thing for thing, _ in asked})))
        critical = any(region.kind in self.w.relevance_of(t).regions for t in (i, j))
        return Cand("changed", {"room": region.kind, "tasks": [i, j]}, gold, "answerable", tuple(sorted(evidence)), (),
                    {"match_m": MATCH_M}, subjects=subjects, critical=critical)


FIRST_LAST = True  # the declared fallback, taken at M2's fast gate check (62% of golds empty over pairs two apart)


def moments(w: World) -> list[Moment]:
    """One moment for each room whose kind is unique in the layout and each pair of the window's tasks at least two
    apart that toured it, keyed "<room>|<i>|<j>"; under `FIRST_LAST` (the fallback: the longest elapsed time) the
    window's first and last tasks only."""
    reader, kinds = _Reader(w), Counter(r.kind for r in w.map.regions)
    ends = [(w.tasks[0], w.tasks[-1])] if len(w.tasks) > 1 else []
    pairs = ends if FIRST_LAST else list(combinations(w.tasks, 2))
    return [Moment(f"{region.kind}|{i}|{j}", [reader.cand(region, i, j)], [])
            for region in w.map.regions if kinds[region.kind] == 1
            for (i, _, _), (j, _, _) in pairs
            if j - i >= 2 and reader.toured(region, i) and reader.toured(region, j)]


SPECS = (FamilySpec("changed", "end", "positions", "f1", ("see", "pose"), "memory", None, False,
                    ("room", "tasks"), _render),)
GOLDS = {"changed": moments}
SLICED: frozenset[str] = frozenset()
