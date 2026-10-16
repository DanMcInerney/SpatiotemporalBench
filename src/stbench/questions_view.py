"""`out_of_view` (during, `memory`): the places, as last seen, of the movable things on the floor near the robot that
are not in the images it was just sent ("what, that I cannot see now, is within 3 m of me?", VISION §3).

The question asks for the things the robot last saw before the look window that ends at the decision (`b0`, the look
window's first keyframe), taller than `height` and standing on the floor within `radius` of its base, which no row of
the look window shows. The answer is a `positions` list: where each stood at its last recognisable sighting (the
placement's xy then), never where it is now (truth rule 2). What a memory cannot tell apart is a **don't-care place**,
so no score hinges on a boundary or a dead band. The scorer drops answer points inside a don't-care place's radius and
matches an answer to an item within the row's `match_m`; a large thing's surface is seen where its centre is not, so
both are widened by the thing's own `rules.widening`: a don't-care radius is `rules.MATCH_M` plus its thing's, and
`match_m` is `MATCH_M` plus the largest widening of an item (a pallet answered 0.6 m from its centre matches):
- a thing with any row in the look window is in view: not an item; its place now is don't-care (a system may detect it
  there), and so are its places before the look window as for a mover (a system may hold it at its old place, the thing
  having been only glimpsed since);
- a mover (a person) is never an item: its last recognisable place before the look window, and every sighting after
  it, are don't-care;
- a thing or mover with a twin (another entity of the same kind and describable attributes that some frame delivered by
  the look window's end shows; an entity no frame has shown is none, and one first shown later is none yet) is never an
  item, since a table tells its tracks apart by description alone and identity between identical things is not
  observable: every place either was sighted at before `b0` is don't-care;
- a movable thing (`MOVABLE`; fixed furniture is not one, and the layout is public) with a sighting before `b0`: `s` is
  its last recognisable keyframe (`World.recognised`: pixels, light, a box clear of the image's edge, a depth). With no
  recognisable one, or a later sighting `rules.MOVED_NO_M` or more from its place at `s` (`World.steady`), its places
  from `s` on (all its sightings when none is recognisable) are don't-care, since a memory that reads the latest
  detection holds it there. Else it is at `P`, its place at `s`, when it stood on the floor there; `d` is its distance
  from the base and `h` its height. `d` over `radius + EDGE_M` is nothing; `d` within `EDGE_M` of `radius`, or `h`
  within `BAND_M` of the slot's height, is don't-care at `P`; and when the keyframes between `s` and `b0` certify `P`
  empty (`grid.place_free`: the robot saw it gone) `P` is don't-care and no item. Else `h` above the band is an item at
  `P` and a lower thing is nothing;
- a thing the robot's own log tells about (`World.told`: the thing it carries, a blocker of a zone it was asked about)
  is never an item, and is a don't-care place as a thing with a twin is: it is not a fact the robot is asked to
  remember;
- the map's floor-level rack stock (`rack_stock.floor_stock`: pallets of cartons in the bays the map gives no surface,
  scenery and no entity) is a don't-care place at each pallet slot, in view or not: it is a movable thing standing on
  the floor, as the words say, that no frame lists;
- a don't-care place beyond `radius + EDGE_M` of the base can touch no item or answer, so none is written;
- **the swallow:** an item within a don't-care place's radius becomes one too, until none moves, since the scorer would
  drop the right answer there.
The evidence is the keyframes of the items' last recognisable sightings and the decision's (its delivered base fixes the
radius); the subjects are the time and each item; `entities` is empty. The gold is never `unknown`: nothing seen near is
the empty list, a fair answer to "what was seen". An empty gold stays (it is honest: a constant empty answer is the
`current_view` shortcut, and the share of empty golds is reported against the declared bound below).

**Declared before data** (VISION rule 5): a constant empty answer scores 2E - 1 at an empty share E, so over 52% empty
golds at the dev gate check the height falls to `FALLBACK_HEIGHT_M` (if not taken already) or the family is dropped;
`HEIGHT` is the slot's height as built (docs/families/out_of_view.md has the free bound that chose it).

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS`, and `SLICED` (none: a during family draws no
slice).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

from . import grid, rack_stock
from .descriptions import desc_key
from .questions_select import Cand
from .questions_words import cap_of
from .questions_world import World
from .records import FamilySpec
from .rules import MATCH_M, widening
from .timeline import Entity

HEIGHT_M, FALLBACK_HEIGHT_M, BAND_M, EDGE_M = 0.3, 0.1, 0.05, 0.5  # the text's height (and its fallback); the bands
# the slot's height as built: the fallback, taken at M2's fast gate check on look-dev seeds 141-145 (every gold of 15
# empty at 0.3 m; DECISIONS, "M2's fast gate check")
HEIGHT = FALLBACK_HEIGHT_M
MOVABLE = ("light", "carried", "clutter")  # things a question may place; movers (people) are don't-care places
MOVER = "mover"
MERGE_M = 0.01  # places this close are one

Xy = tuple[float, float]
Care = tuple[Xy, float]  # a don't-care place and its radius


class Item(NamedTuple):
    """An item: its place at the last recognisable sighting, the entity, the keyframe of that sighting and the radius
    around the place the scorer drops answers within (`MATCH_M` plus the thing's widening)."""

    xy: Xy
    entity: str
    k: int
    radius: float


def _render(slots: dict) -> str:
    return (f"Where, as you last saw them, are the movable things taller than {slots['height']:g} m standing on the "
            f"floor within {slots['radius']:g} m of you that are not in any of the images you were just sent?")


def _radius(e: Entity) -> float:
    """The radius of a don't-care place of the entity, and the match radius of an item of it."""
    return round(MATCH_M + widening(e.size), 6)


def _merged(places: list[Care]) -> list[Care]:
    """The places sorted, each within `MERGE_M` of the one kept before it merged into it (the larger radius kept)."""
    out: list[Care] = []
    for place, radius in sorted(places):
        if out and math.dist(place, out[-1][0]) <= MERGE_M:
            out[-1] = (out[-1][0], max(out[-1][1], radius))
        else:
            out.append((place, radius))
    return out


def _swallow(items: list[Item], care: list[Care]) -> tuple[list[Item], list[Care]]:
    """Items within a don't-care place's radius join it, repeated until none moves: the scorer drops answer points
    inside a don't-care region, so a kept item there would be a right answer dropped."""
    care = list(care)
    while True:
        near = [it for it in items if any(math.dist(it.xy, c) <= r + grid.EPS for c, r in care)]
        if not near:
            return items, care
        care += [(it.xy, it.radius) for it in near]
        items = [it for it in items if it not in near]


def _gone(e: Entity, s: int, before: Callable[[], dict], place: Xy) -> bool:
    """The keyframes after the sighting `s` and before the look window certify the place empty: the robot saw the
    thing was no longer there (`grid.place_free` over the thing's own half-width). `before()` is `grid.latest` over the
    window's keyframes up to the look window; keeping its cells observed after `s` is `latest` over those keyframes
    alone, whichever keyframe a cell was last seen at."""
    seen_since = {cell: v for cell, v in before().items() if v[1] > s}
    return grid.place_free(seen_since, place, min(e.size[0], e.size[1]) / 2)


def _sighted_descriptions(w: World, until: int) -> Counter:
    """How many entities of each description (kind and describable attributes) have a sighting before keyframe `until`:
    what the stream has delivered of them, so an entity no frame has shown yet is no twin (truth rule 2)."""
    return Counter(desc_key(o.kind, o.attributes) for o in w.tl.entities if (ks := w.sighted(o.id)) and ks[0] < until)


def _sighting_places(w: World, e: Entity, b0: int) -> list[Xy]:
    """Every place the entity was sighted at before `b0` (recognisable or not): where any track of it could be held."""
    return [xy for j in w.sighted(e.id) if j < b0 and (xy := w.xy_at(e.id, j)) is not None]


def _held_places(w: World, e: Entity, b0: int) -> list[Xy]:
    """Where a memory that tracks the entity would hold it from before the look window: its last recognisable sighting
    before `b0` and every sighting after it (a track follows a clipped, small or dark frame the recognisable one
    ignores), or every sighting before `b0` when none is recognisable."""
    last = max((j for j in w.recognised(e.id) if j < b0), default=None)
    return [xy for j in w.sighted(e.id) if j < b0 and (last is None or j >= last)
            and (xy := w.xy_at(e.id, j)) is not None]


def _verdict(w: World, e: Entity, b0: int, here: Xy, height: float, radius: float, before: Callable[[], dict]
             ) -> tuple[list[Xy], tuple[Xy, int] | None]:
    """What a movable thing out of view contributes: its don't-care places, and the item (its place and the keyframe
    `s` of its last recognisable sighting) or None."""
    recognised = [j for j in w.recognised(e.id) if j < b0]
    xy = w.xy_at(e.id, recognised[-1]) if recognised else None
    if xy is None or not w.steady(e, recognised[-1], xy, b0 - 1):
        # no recognisable sighting, or a later one elsewhere: a track may hold any place it was sighted at since
        return _held_places(w, e, b0), None
    s = recognised[-1]
    if not w.on_floor(e.id, s):
        return [], None
    d, h = math.dist(here, xy), e.size[2]
    if d > radius + EDGE_M + grid.EPS:
        return [], None
    if abs(d - radius) <= EDGE_M + grid.EPS or abs(h - height) <= BAND_M + grid.EPS or _gone(e, s, before, xy):
        return [xy], None
    return ([], (xy, s)) if h > height else ([], None)


def out_of_view(w: World, k: int, height: float | None = None) -> Cand:
    """The question asked at the decision point at script keyframe `k`, with its gold (`height`: the slot's, `HEIGHT`
    when not given). Never None: nothing seen near is the empty list, not a dead band."""
    height = HEIGHT if height is None else height
    radius, here, look = cap_of(w.robot), w.base(k), w.window_ks(k)
    b0 = look.start
    before = cache(lambda: w.latest_cells(range(w.rec.k0, b0)))  # computed once, when a place is first tested
    delivered, told = _sighted_descriptions(w, look.stop), w.told()
    items: list[Item] = []
    care: list[Care] = list(rack_stock.floor_stock(w.map))  # scenery: the stock pallets on the racks' floor level
    for e in w.tl.entities:
        if e.mobility not in (*MOVABLE, MOVER):
            continue
        around = _radius(e)
        in_view = w.in_ks(e.id, look)
        # another sighted entity shares its description, or the robot's log tells of it: it is never an item
        twin = delivered[desc_key(e.kind, e.attributes)] > 1 or e.id in told
        if twin:  # identity is not observable: a track may hold any place it was sighted at
            care += [(p, around) for p in _sighting_places(w, e, b0)]
        elif e.mobility == MOVER or in_view:  # a track of it from before the look window may be held at its old place
            care += [(p, around) for p in _held_places(w, e, b0)]
        if in_view:  # in view now, at any pixel count: its place now is don't-care
            if (xy := w.xy_at(e.id, k)) is not None:
                care.append((xy, around))
        elif not twin and e.mobility != MOVER:
            dont, item = _verdict(w, e, b0, here, height, radius, before)
            care += [(p, around) for p in dont]
            if item is not None:
                items.append(Item(item[0], e.id, item[1], around))
    care = [(p, r) for p, r in care if math.dist(here, p) <= radius + EDGE_M + grid.EPS]
    items, care = _swallow(items, care)
    places = _merged([(it.xy, it.radius) for it in items])
    gold: dict = {"type": "positions", "items": [{"xy": list(p)} for p, _ in places]}
    if care:
        gold["dont_care"] = [{"xy": list(p), "radius": r} for p, r in _merged(care)]
    evidence = sorted({it.k for it in items} | {k})
    match_m = max((it.radius for it in items), default=MATCH_M)
    return Cand("out_of_view", {"height": height, "radius": radius}, gold, "answerable", tuple(evidence), (),
                {"match_m": match_m}, decision_k=k,
                subjects=(f"w:{k}", *(f"e:{eid}" for eid in sorted({it.entity for it in items}))), critical=True)


SPECS = (FamilySpec("out_of_view", "during", "positions", "f1", ("see", "pose"), "memory", None, False,
                    ("height", "radius"), _render),)
GOLDS = {"out_of_view": out_of_view}
SLICED: frozenset[str] = frozenset()
