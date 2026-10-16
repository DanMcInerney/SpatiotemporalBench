"""`task_things` (end, `memory`): "Where were the things {the job's predicate} in your {first} task, which you would
have to steer around next time?" The things that mattered to a task, defined by what the task was and never named,
answered as a set of places. The clause after the task says why they matter (0.10.0, a render change only: the gold, the
credit and the slots are unchanged).

The predicate is the job's (`JobSpec.task_things`, a `task_predicate.TaskThings`), declared over the delivered steps of
the task and the public map, so the tables evaluate it with the same code (`task_predicate`). One moment a task of the
window; a task whose steps give no anchor (no step of its phases, or none of its `until`) is no moment. The gold is a
`positions` list: an item is a thing at its last recognisable sighting in the task. The question names no thing, so it
has no subjects and no entities, and reserves nothing; its evidence is the keyframes of the items' last sightings and
always the task's last keyframe, so an empty gold has its evidence (`records.check_gold`). The credit is `graded_set`
(0.9.0): the answered places are matched one to one to the items within four times `JobSpec.match_m` plus the
largest item's widening (`tolerances.match_m`, stated in the prompt), each pair earning the four quarters on the item's
own resolution (`tolerances.resolutions_m`, one per item in the gold's order: one pixel at the item's last recognisable
sighting, `resolution_at`), and a false place costing `tolerances.fp_weight` (`credit_kernel.FP_WEIGHT`) of an item.

**An item** is a `carried` or `clutter` entity that stands inside the predicate (`task_predicate.verdict` is `IN`, for
the footprint as placed and for the footprint a detection gives: `judge`) at its last recognisable sighting in the task
(`World.recognised`), and none of the rules below makes it don't-care. A thing's place in a task is its placement at a
sighting in the task; a thing no frame of the task shows stands where it does at the task's last keyframe.

**Nothing**: a thing outside the predicate by every footprint; when the predicate says `floor`, a thing clearly off
the robot's floor wherever it stood in the task (its base more than the robot's `climb_m` plus `OFF_FLOOR_M` above or
below the delivered base: rack-held stock, a thing on a shelf); and, when it names `item_kinds` (0.11.0, the vacuum's
list of what its brush could catch), a carried thing or clutter of a kind that neither `item_kinds` nor
`dont_care_kinds` holds: the question does not ask for it and the gold holds no place for it, so a reader who lists it
is charged a false place, as `tangle_ahead` charges one for a kind it leaves out; that is fair because the words name
the kinds (`hazard_rule.catch_words`, the vacuum's phrase, final review S4). The tables drop such a thing too.

**A don't-care place** (a circle of the job's `match_m` plus the thing's widening about each place; the scorer drops an
answer point inside one that matches no item) is every place of a thing that is not outside the predicate and:
- lies in the band (`EDGE`): within `reach_m` +- `band_m` of the anchor, or on a doorway way or across a room's edge;
- is light furniture or a mover (a person), or of a kind in `dont_care_kinds`;
- is told: the robot's own log tells about it (`World.told`);
- has a twin: another entity of its description (`descriptions.desc_key`) that some frame of the window shows (a table
  tells its tracks apart by description alone, and identity between identical things is not observable);
- has a row in the end look window, at any pixel count (`current_view` would hold it: `memory_needed`);
- is not recognisably seen in the task, or moves within it (a later sighting `MOVED_NO_M` or more from the last
  recognisable place: a memory that reads the latest detection would hold it elsewhere);
- is off the floor by less than `OFF_FLOOR_M` beyond the robot's climb when the predicate says `floor` (a kerb, a
  low step);
- is no taller than `min_height_m` (a remote, a cable, a sheet): the prompt's "thing" counts it, so naming it must not
  cost a point, and a frame does not resolve it well enough to ask for it (0.7.0, the dev pilot's log review: it was
  nothing, and a model that listed a remote lost credit);
and the floor-level stock of the map's racks (`rack_stock.floor_stock`: scenery that is no entity, yet a pallet on the
floor) is don't-care wherever its pallet is not outside the predicate. An item's other places in the task (it stood
elsewhere earlier, `MOVED_NO_M` or more apart) are don't-care too: the question does not say "last".

**The swallow** closes free don't-care space. An answer point inside a circle that matches nothing is dropped, not
counted wrong, so a system that knows where a circle is without this case's sensors could put points in it, next to
likely items, at no risk. A circle is *sensor-free* when its place is knowable that way: the racks' stock (the map),
light furniture (the map), a told thing (the log), or a thing no frame of the window shows. An item within the circle's
radius plus the question's `match_m` (the largest item radius) of a sensor-free place is don't-care, so those points
reach no item. The circles of what the sensors show (twins that were seen, the band, people, things in the end look
window, a thing `RELAX` names, a swallowed item) swallow nothing: a system cannot place points in them without having
seen the thing.

`RELAX` makes the things a look question names don't-care (they are not asked twice, and the look draw reserves them):
the items of the named entities become don't-care places and the class (`empty` or `items`) is recomputed. `CAPPED`: the
draw may thin the empty golds (`TaskThings.empty_keep_percent`).

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` (an end family's, `task_things`), `SLICED` (none),
`CAPPED` and `RELAX`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from . import rack_stock
from .credit_kernel import FP_WEIGHT
from .descriptions import desc_key
from .geometry import Rect
from .questions_select import Cand, Moment
from .questions_words import ORDINALS
from .questions_world import World
from .records import FamilySpec
from .resolution_at import at_frame, base_of
from .rules import MOVED_NO_M, widening
from .task_predicate import IN, OUT, Anchor, TaskThings, anchor_of, detected_rect, judge
from .timeline import Entity

THINGS = ("light", "carried", "clutter", "mover")   # what can stand near a task: fixed pieces and structure never
ITEMS = ("carried", "clutter")   # of those, what can be an item; light furniture and people are don't-care
OFF_FLOOR_M = 0.05   # a base this far beyond the robot's climb from its own is clearly not on the floor it drives on
MERGE_M = 0.01   # places this close are one
EPS = 1e-9

Xy = tuple[float, float]


class Part(NamedTuple):
    """One place of the gold: an item (`item`: its entity, the keyframe `k` of its last recognisable sighting) or a
    don't-care place; `radius` is `match_m` plus the widening of the thing (the circle of a place, the match radius of
    an item). `entity` is None for the racks' stock. `free`: the place is sensor-free (the module docstring). `rho`: an
    item's resolution (`resolution_at`: one pixel at its last recognisable sighting, 0 for a don't-care place)."""

    xy: Xy
    radius: float
    entity: str | None
    k: int | None
    item: bool
    free: bool = False
    rho: float = 0.0


@dataclass
class TaskCand(Cand):
    """A `task_things` candidate with the places its gold was made of, so `relax` can make some of them don't-care
    without the world. `last_k` is the task's last keyframe; `match_base` the job's `match_m`."""

    parts: tuple[Part, ...] = ()
    last_k: int = 0
    match_base: float = 0.0


def _merged(places: list[tuple[Xy, float]]) -> list[tuple[Xy, float]]:
    """The places sorted, each within `MERGE_M` of the one kept before it merged into it (the larger radius kept)."""
    out: list[tuple[Xy, float]] = []
    for place, radius in sorted(places):
        if out and math.dist(place, out[-1][0]) <= MERGE_M:
            out[-1] = (out[-1][0], max(out[-1][1], radius))
        else:
            out.append((place, radius))
    return out


def _swallow(parts: Iterable[Part]) -> tuple[list[Part], list[Part]]:
    """(items, don't-care places) once every item within a sensor-free place's radius plus the question's `match_m` has
    become a don't-care place (which swallows nothing)."""
    items = [p for p in parts if p.item]
    care = [p for p in parts if not p.item]
    reach = max((p.radius for p in items), default=0.0)
    free = [c for c in care if c.free]
    near = [it for it in items if any(math.dist(it.xy, c.xy) <= c.radius + reach + EPS for c in free)]
    care += [it._replace(item=False, k=None) for it in near]
    return [it for it in items if it not in near], care


def _build(slots: dict, parts: Iterable[Part], last_k: int, match_base: float) -> TaskCand:
    """The candidate for one task from the places of its gold: the swallow applied, the gold, the evidence (the items'
    last sightings and the task's last keyframe), the matching radius and the class."""
    parts = tuple(parts)
    items, care = _swallow(parts)
    ordered = sorted(items, key=lambda p: p.xy)
    gold: dict = {"type": "positions", "items": [{"xy": list(p.xy)} for p in ordered]}
    if care:
        gold["dont_care"] = [{"xy": list(xy), "radius": r} for xy, r in _merged([(p.xy, p.radius) for p in care])]
    evidence = tuple(sorted({p.k for p in items} | {last_k}))
    match_m = max((p.radius for p in items), default=match_base)
    tolerances = {"match_m": match_m, "resolutions_m": [p.rho for p in ordered], "fp_weight": FP_WEIGHT}
    return TaskCand("task_things", slots, gold, "answerable", evidence, (), tolerances,
                    answer_class="items" if items else "empty", parts=tuple(items + care), last_k=last_k,
                    match_base=match_base)


def relax(c: Cand, named: Iterable[str]) -> Cand:
    """`c` with the items of the entities `named` (the things a look question asks about) made don't-care places, the
    gold, the evidence, the matching radius and the class made again."""
    named = set(named)
    parts = tuple(p._replace(item=False, k=None) if p.item and p.entity in named else p for p in c.parts)
    return _build(c.slots, parts, c.last_k, c.match_base)


class _Reader:
    """What one window's `task_things` reads, each piece once however many tasks ask: the sighted descriptions (twins),
    the racks' stock places and the entities the question may be about."""

    def __init__(self, w: World, pred: TaskThings):
        self.w, self.pred = w, pred
        self.base = w.job.match_m
        self.told = w.told()
        self.climb = w.robot.body.climb_m
        self.sighted = Counter(desc_key(o.kind, o.attributes) for o in w.tl.entities if w.sighted(o.id))
        self.stock = [xy for xy, _ in rack_stock.floor_stock(w.map)]
        self.things = [e for e in w.tl.entities if e.mobility in THINGS]
        self.named = pred.item_kinds | pred.dont_care_kinds

    def twin(self, e: Entity) -> bool:
        """Another entity of the description has a sighting in the window."""
        return self.sighted[desc_key(e.kind, e.attributes)] - (1 if self.w.sighted(e.id) else 0) > 0

    def verdict(self, anchor: Anchor, e: Entity, k: int) -> str:
        """The thing against the predicate at keyframe `k`: its footprint as placed and as a detection gives it."""
        rect = self.w.tl.footprint(e.id, self.w.tl.keyframe(k).t)
        if rect is None:
            return OUT
        return judge(self.pred, anchor, [rect, detected_rect((rect.cx, rect.cy), e.size)])

    def places(self, e: Entity, k0: int, k1: int) -> list[tuple[Xy, int]]:
        """Each place the thing stands at in the task, with a keyframe it stood there: its sightings' (places within
        `MERGE_M` are one), else its place at the task's last keyframe; empty when it does not exist then."""
        found: list[tuple[Xy, int]] = []
        for k in (j for j in self.w.sighted(e.id) if k0 <= j < k1):
            xy = self.w.xy_at(e.id, k)
            if xy is not None and all(math.dist(xy, p) > MERGE_M for p, _ in found):
                found.append((xy, k))
        if not found and (xy := self.w.xy_at(e.id, k1 - 1)) is not None:
            found.append((xy, k1 - 1))
        return found

    def is_item(self, e: Entity, anchor: Anchor, k0: int, k1: int) -> tuple[Xy, int] | None:
        """The place and the keyframe of the thing's last recognisable sighting in the task when nothing makes the thing
        don't-care (the module docstring), else None."""
        w, pred = self.w, self.pred
        recognised = [k for k in w.recognised(e.id) if k0 <= k < k1]
        if (e.mobility not in ITEMS or e.kind in pred.dont_care_kinds or e.id in self.told or self.twin(e)
                or w.in_end_window(e.id) or not recognised or (pred.floor and not w.on_floor(e.id, recognised[-1]))):
            return None
        xy = w.xy_at(e.id, recognised[-1])
        if xy is None or not w.steady(e, recognised[-1], xy, k1 - 1) or self.verdict(anchor, e, recognised[-1]) != IN:
            return None
        return xy, recognised[-1]

    def thin(self, e: Entity) -> bool:
        """The thing is no taller than the predicate's `min_height_m` (a remote, a cable, a sheet): the prompt's "thing"
        counts it, so naming it must cost nothing, but no frame resolves it well enough to be asked for."""
        return round(e.size[2], 4) <= self.pred.min_height_m

    def clearly_off(self, e: Entity, k: int) -> bool:
        """The thing's base at keyframe `k` is further than the robot's climb plus `OFF_FLOOR_M` from the delivered
        base."""
        p = self.w.tl.placement_of(e.id, self.w.tl.keyframe(k).t)
        return p is not None and abs(p.pos[2] - self.w.rows[k]["pose"]["pos"][2]) > self.climb + OFF_FLOOR_M

    def entity(self, e: Entity, anchor: Anchor, k0: int, k1: int) -> list[Part]:
        """The places the thing adds to the gold in the task: an item (and its earlier places, `MOVED_NO_M` or more
        away, as don't-care places), or its don't-care places, or none."""
        if self.pred.item_kinds and e.mobility in ITEMS and e.kind not in self.named:
            return []   # a kind the predicate leaves out: no item and no place
        places = self.places(e, k0, k1)
        if self.pred.floor and all(self.clearly_off(e, k) for _, k in places):
            return []
        radius = round(self.base + widening(e.size), 6)
        free = e.mobility == "light" or e.id in self.told or not self.w.sighted(e.id)
        care = [Part(xy, radius, e.id, None, False, free) for xy, k in places if self.verdict(anchor, e, k) != OUT]
        if self.thin(e) or (item := self.is_item(e, anchor, k0, k1)) is None:
            return care
        rho = at_frame(self.w, item[1], item[0], base_of(self.w, e.id, item[1]))
        return [Part(item[0], radius, e.id, item[1], True, rho=rho),
                *(c for c in care if math.dist(c.xy, item[0]) >= MOVED_NO_M)]

    def stock_parts(self, anchor: Anchor) -> list[Part]:
        """The racks' floor-level stock pallets that are not outside the predicate, as don't-care places: a pallet of
        the stock turned either way."""
        radius = round(self.base + widening(rack_stock.PALLET_M), 6)
        return [Part(xy, radius, None, None, False, True) for xy in self.stock
                if judge(self.pred, anchor, [Rect(xy[0], xy[1], *rack_stock.PALLET_M, yaw)
                                             for yaw in (0.0, math.pi / 2)]) != OUT]


def task_things(w: World) -> list[Moment]:
    """One moment for each task of the window with an anchor, keyed by its case index: the one candidate, whose gold is
    the task's items and don't-care places. Empty for a job that asks no `task_things`."""
    pred = w.job.task_things
    if pred is None:
        return []
    reader, moments = _Reader(w, pred), []
    for i, k0, k1 in w.tasks:
        steps = [w.rows[k] for k in range(k0, k1) if k in w.rows]
        if (anchor := anchor_of(pred, steps, w.map if pred.anchor == "rooms" else None)) is None:
            continue
        parts = [p for e in reader.things for p in reader.entity(e, anchor, k0, k1)]
        parts += reader.stock_parts(anchor)
        moments.append(Moment(str(i), [_build({"phrase": pred.phrase, "task": i}, parts, k1 - 1, reader.base)], []))
    return moments


# What the question counts (0.10.0): it was the `positions` shape's, which served only this family; the shape is now the
# list's form alone and each list family says what it counts in its own text.
COUNTED = ("A thing is an object that is not part of the room or its fixed furniture and fittings (a docking station, "
           "a charger, a door, a table, a shelf, a rack or the room's own floor mat), so a floor lamp, a fan or a "
           "heater is a thing; a pair or a stack is one thing; people and chairs are not counted either way.")


def _render(slots: dict) -> str:
    return (f"Where were the things {slots['phrase']} in your {ORDINALS[slots['task']]} task, which you would have to "
            f"steer around next time? Answer in map coordinates. {COUNTED}")


SPECS = (FamilySpec("task_things", "end", "positions", "graded_set", ("see", "pose", "status"), "memory", None, False,
                    ("phrase", "task"), _render),)
GOLDS = {"task_things": task_things}
SLICED: frozenset[str] = frozenset()
CAPPED = frozenset({"task_things"})
RELAX = {"task_things": relax}
