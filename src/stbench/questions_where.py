"""`object_where` (end, `memory`): "Where did you last see the {thing}? Answer in map coordinates." The place the
robot last saw a thing.

The text and its slots (`kind`, `attributes`) name no moment and no task: a question is the thing and nothing else. One
question asks about one thing over the window it is asked at, and its gold is where the thing stood at its last
recognisable sighting in that window. The things are carried things of the relevance kinds of the window's tasks
(`JobSpec.relevance` of each; a thing counts when it is critical under any of them), whose description is unique in the
world (`World.things`), which the robot's own log does not tell about (`World.told`: the thing a task carries, the
blockers that stood in a decision's zone) and which no look-alike hides (`askable`, below). A thing is one of:
- **answerable**: it has a recognisable sighting in the window (`recognise.recognisable`: in one frame enough pixels
  and light, a bounding box clear of the image's edge, a depth), no row at all in the end look window (the current
  view would answer: VISION rule 4), has not moved in the window, and is **critical** (below). It has not moved when
  every sighting of it after its last recognisable one, recognisable or not, lies within `rules.MOVED_NO_M` of that
  place (`World.steady`: so the memory table, which reads the latest sighting, holds it where the gold reads it; a
  glimpse after a gap puts a moved thing elsewhere) and no earlier recognisable sighting lies a metre or more from it
  (`World.arrived`); a thing that moved is `object_was`'s. The gold is its placement's (x, y) at the last recognisable
  sighting, which is its evidence; one moment is asked for each answerable thing, keyed by its description (`moments`),
  so a case's questions are about the things it can ask, not about the job's events;
- **unknown**: it showed no pixel at all in the window and exists at the window's last keyframe (nothing the sensors
  recorded shows where it is), or it is a description that matches nothing in this world, drawn from other seeds'
  worlds of the map by the generator's own sampler, and in either case no look-alike was glimpsed in its colour (S4,
  below);
- **skipped**: anything else (only a glimpse, only dark, clipped by the frame or depth-less, in the end look window,
  moved, not critical, told, or hidden by a look-alike).
Both pools hold the same kinds, so no kind predicts the slice, and every moment offers the one unknown pool beside its
answerable thing: the questionnaire draws the unknown slice per moment (`UNKNOWN_PERCENT`, 20%). A gold is what was last
seen; nothing unobserved changes it (truth rule 2).

A thing is **critical** when the job's relevance makes it so (`critical`): its kind is relevant; its last place lies in
a region of a kind the task works in; where the job sets `Relevance.floor` the thing stood on the floor the robot
drives on; and where it sets `Relevance.near` and `near_m` (the drone's: the ground its camera sees from the drop point)
the place lies within `near_m` of one of the points. An unknown row is critical by kind: no place is known. Only
critical things are answerable. `object_was` and `same_place` ask a task, `person_where` a moment of one
(`questions_change`), and judge it under that task's relevance.

An **absent description** (`never_seen`) is a decoy, offered only when the world gives no clue that it is absent: a
thing of its kind is sighted in the window (so the kind's presence says nothing), and no sighted thing of its own kind
or of a kind it can be mistaken for (`descriptions.CONFUSABLE`) could be taken for its colour (so no colour is a near
miss). "Could be taken for" reads the render, not the catalog's word (`World.may_be_taken_for`, `colour_seen`, review
defect 1): a thing whose recognisable frames show it in that colour, or next to a boundary of it, however it is
labelled. Black, grey and white are one group here: a person can not reliably tell them apart on a thing of several
materials under varied light, so a sighted thing that may be taken for any neutral hides every neutral decoy (a "grey"
laptop in view hides a "white laptop" decoy). The questionnaire asks at most one a case, and the world's own unseen
things before it.

A **real thing** is screened by two rules (`nameable`; `lookalike_sighted` and `askable` are its parts, and `object_was`
and `distance_between` use them as well):
- **Its picture bears out its word** (`World.colour_holds`): the colour most of its recognisable frames confidently
  show it in (`colour_seen.held`; a frame near a boundary votes for none) is its label, so a "black" remote that
  renders silver is not asked. A thing no frame of the window shows recognisably (an unseen one) has nothing to
  contradict its label.
- **No look-alike hides it**: it is not asked when another thing of a kind it can be mistaken for
  (`descriptions.lookalikes`) could be taken for its colour in the window (a blue skip beside a blue dumpster, a bag
  beside a box of its colour, a "grey" bag and a "black" one, or a "white" laptop beside a "grey" one: the neutrals
  are one group), since a system that holds the look-alike can not tell which one the question names. An older root's
  visibility log has no render to read: the label stands, a neutral label for all three neutrals. A thing no frame of
  the window shows, and an absent description, are also hidden by the `CONFUSABLE_UNSEEN` kinds (a bin by a drum of its
  colour, a sack by a pallet of its colour, and the reverse): the model stretches a name to them when nothing seen says
  which is meant, but a seen drum or pallet is told from a bin or a sack, so those groups never hide a seen thing.

The screens read kinds, colours and whether a sighting exists, never a place, a time or a gold, and drop the thing from
both pools (answerable and unknown).

**The glimpse screen (S4, 0.11.0)** drops an `unknown` candidate, and only that, when the window glimpsed a look-alike
in the asked colour (`glimpse_hides`): a thing of a kind its name can be mistaken for that was sighted (a glimpse) and
never recognisably, in a frame whose render shows the asked colour. The screens above read a glimpse-only thing by its
label, which is all they have of it; a render can say otherwise (a skip the catalog calls grey that the glimpse shows
blue), and a person asked for the blue skip could not tell the glimpse from it. An answerable question is not screened:
the thing it names was seen plainly. It costs 6% of the `unknown` rows on the dev root (FD-0.11-evidence §4.1); a wider
screen (any colour of a look-alike glimpse, 48%; any glimpse in the colour, 57%) would leave the AMR's `ego_mem` and the
helper's `object_where` no `unknown` rows at all, and is not applied.

**The tolerance** of an answerable row is `graded` (0.9.0; `tolerances_of`): the need is the job's re-find
radius R (`JobSpec.reacquire_m`), the radius within which the robot driving back to the place finds the thing in view;
it does not widen with range. The resolution `rho` is the larger of one pixel at the last recognisable sighting (the
camera there, `resolution_at`) and the hidden widening. The place asked for is the middle of the thing, which a camera
below the thing's top never saw: so `rho` is at least `rules.hidden_widening` (the thing's half-length beyond
`rules.WIDEN_FREE_M`) when the thing's top was above the camera at every recognisable sighting of it at the place
(`place_widening`: the least widening over them), and not otherwise (one view from at or above the top saw its top face,
so the middle was in view). The direction quarter is `e <= R` alone (going there, the robot finds the thing in view; the
room a thing is in is a fact of the map and the kind, which a prior without the camera knows too, so it earns nothing),
and there is no harmful side. It records the nearest slant of those sightings (`nearest_sighting_at`) as `range_m`,
for the reports; it sets no tolerance. An unknown row has no place and no range and carries the floor resolution. A
row's `answer_class` is its slice.

The rules the change families (`questions_change`) share are functions of their own: `answerable_at` (the raw same-task
rule, for one thing at one event), `unknown_refs` (the unknown pool's referents at one event) and `critical` (whether a
row is task-critical under one task's relevance). `object_where`'s own exclusions (the end look window, a moved thing)
live in `answerable` and `moments`, never in those.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` (`moments`, an end family's) and `SLICED`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import NamedTuple

from . import SEED_KEY, colour_seen
from .catalog import for_map, item_size
from .descriptions import Key, desc_attrs, desc_key, lookalikes, phrase
from .job_registry import Relevance
from .mapspec import load_map
from .placement import sample_items
from .questions_select import Cand, Moment
from .questions_world import World
from .records import FamilySpec
from .resolution_at import UNKNOWN_RHO_M, at_frame, base_of
from .rules import MOVED_NO_M, graded_tolerances, hidden_widening, widening
from .seeding import make_rng
from .timeline import Entity

OTHER_SEEDS = range(1, 41)


@lru_cache(maxsize=8)
def _other_combos(map_name: str, own_seed: int) -> tuple[Key, ...]:
    """Full descriptions that other seeds' worlds of the map contain, in a fixed order. The items are sampled from the
    map's catalog by the generator's own sampler, keyed by `SEED_KEY` (the key every generator since 0.4.0 draws its
    items by), so the world's real things and these descriptions come from one distribution."""
    catalog = for_map(load_map(map_name))
    keys = {desc_key(s.kind, s.attributes) for seed in OTHER_SEEDS if seed != own_seed
            for s in sample_items(make_rng(map_name, seed, SEED_KEY, "items"), catalog=catalog)}
    return tuple(sorted(keys))


def never_seen(w: World, relevance: Relevance | None = None) -> list[Key]:
    """Descriptions of the task's relevance kinds (`relevance`: the last task's when not given) that occur in other
    seeds but match nothing in this world: the robot cannot have seen them. Only the decoys that give nothing away
    (the module docstring): a thing of the kind is sighted in the window, and no sighted thing of the kind or of a kind
    it can be mistaken for (`descriptions.lookalikes`, with the `CONFUSABLE_UNSEEN` groups: nothing was seen of the
    description) could be taken for the description's colour (`World.may_be_taken_for`)."""
    ep = w.tl.episode
    kinds = (w.relevance if relevance is None else relevance).kinds
    sighted = w.sighted_looks
    return [key for key in _other_combos(ep.map, ep.seed)
            if key not in w.groups and key[0] in kinds and key[0] in sighted
            and not lookalike_sighted(w, key[0], dict(key[1]), None)]


def lookalike_sighted(w: World, kind: str, attributes: Mapping[str, object], entity: str | None) -> bool:
    """Whether a thing other than `entity` (None: a description the world lacks) of a kind that `kind` can be mistaken
    for (`descriptions.lookalikes`, `kind` itself included) has a sighting in the window and could be taken for the
    colour in `attributes` (`World.may_be_taken_for`: by its render). A thing that no frame of the window shows
    (`World.in_window`), and a description the world lacks (`entity` None), is also hidden by the `CONFUSABLE_UNSEEN`
    kinds; a seen one is not. A real thing so hidden is not asked, nor a decoy offered (the module docstring); this
    reads no place, time or gold."""
    colour = attributes.get("color")
    unseen = entity is None or not w.in_window(entity)
    return any(w.may_be_taken_for(e, colour) for near in lookalikes(kind, unseen) for e in w.sighted_looks.get(near, ())
               if e != entity)


def glimpse_hides(w: World, kind: str, attributes: Mapping[str, object], entity: str | None,
                  upto: int | None = None) -> bool:
    """The S4 screen of an `unknown` candidate (the module docstring): whether a thing other than `entity` (None: a
    description the world lacks) of a kind that nothing seen says apart from `kind` (`descriptions.lookalikes`, the
    unseen groups too) was sighted in the window at or before the keyframe `upto` (None: the whole window), never in
    a recognisable frame, and in a frame whose `rgb_med` may be taken for the colour in `attributes` (`colour_seen.
    could_be`: the neutrals are one group). A name with no colour is never hidden by it, and a window whose visibility
    log holds no render hides nothing here. It reads kinds, renders and which things were sighted: no place, no gold."""
    colour = attributes.get("color")
    if colour is None:
        return False
    for near in lookalikes(kind, True):
        for e in w.sighted_looks.get(near, ()):
            glimpses = [j for j in w.sighted(e) if upto is None or j <= upto]
            if e == entity or set(glimpses) & set(w.recognised(e)):
                continue
            if any(colour_seen.could_be(colour_seen.classify(rgb), colour)
                   for j in glimpses if (rgb := w.vis[j][e].get("rgb_med")) is not None):
                return True
    return False


def nameable(w: World, kind: str, attributes: Mapping[str, object], entity: str | None) -> bool:
    """Whether a question may name the thing `entity` (None: a description the world lacks, screened by `never_seen`):
    the picture bears out its colour word (`World.colour_holds`) and no sighted look-alike hides it
    (`lookalike_sighted`)."""
    return entity is None or (w.colour_holds(entity) and not lookalike_sighted(w, kind, attributes, entity))


def askable(w: World, relevance: Relevance | None = None) -> list[Entity]:
    """`World.things` for the task's relevance (the last task's when not given; the things the log tells about are
    already left out), less the things a render or a sighted look-alike keeps from naming (`nameable`): what
    `object_where` and `object_was` ask about."""
    return [e for e in w.things(relevance) if nameable(w, e.kind, e.attributes, e.id)]


def askable_in_window(w: World) -> list[Entity]:
    """`askable` under each task of the window's relevance, once each, in world order: the things an end question of the
    window may name."""
    found: dict[str, Entity] = {}
    for i, _, _ in w.tasks:
        for e in askable(w, w.relevance_of(i)):
            found.setdefault(e.id, e)
    return list(found.values())


def widened_ladder(ladder: Sequence[float], *kinds: str) -> list[float]:
    """The distance rungs of a gold that places things of these kinds (`ladder`: the job's, `JobSpec.ladder`), each
    widened by the kinds' `rules.widening` (a large thing's extent: where its surface is seen is not where its centre
    is), to the micrometre."""
    widen = sum(widening(item_size(kind)) for kind in kinds)
    return [round(rung + widen, 6) for rung in ladder]


def nearest_sighting(w: World, e: Entity, ks: Iterable[int]) -> tuple[float, int] | None:
    """(the smallest slant range, its keyframe) over the keyframes `ks` (`World.slant`: camera at the keyframe to the
    middle of the thing there), the range to 4 places; the earliest keyframe of a tie. None when `ks` is empty."""
    return min(((round(r, 4), k) for k in ks if (r := w.slant(e, k)) is not None), default=None)


def nearest_slant(w: World, e: Entity, ks: Iterable[int]) -> float | None:
    """The smallest slant range (`nearest_sighting`) over the keyframes `ks`, None when `ks` is empty: the r a memory
    question's relative rungs scale with and the row's `range_m`."""
    best = nearest_sighting(w, e, ks)
    return None if best is None else best[0]


def _near_place(w: World, e: Entity, xy: tuple[float, float]) -> list[int]:
    """The thing's recognisable sightings in the window at which it stood within `MOVED_NO_M` of `xy`."""
    return [k for k in w.recognised(e.id) if (p := w.xy_at(e.id, k)) is not None and math.dist(p, xy) < MOVED_NO_M]


def nearest_sighting_at(w: World, e: Entity, xy: tuple[float, float]) -> tuple[float, int] | None:
    """`nearest_sighting` over the thing's recognisable sightings in the window at which it stood within `MOVED_NO_M`
    of `xy`: the sighting of the place `xy` itself that was nearest."""
    return nearest_sighting(w, e, _near_place(w, e, xy))


def nearest_slant_at(w: World, e: Entity, xy: tuple[float, float]) -> float | None:
    """`nearest_slant` over the thing's recognisable sightings in the window at which it stood within `MOVED_NO_M` of
    `xy`: the range at which the place `xy` itself was seen, which is the r of a question about that place when the
    thing stood elsewhere at other sightings (the change families)."""
    return nearest_slant(w, e, _near_place(w, e, xy))


def hidden_extent(w: World, e: Entity, k: int) -> float:
    """The widening of the rungs about the middle of `e` as seen from the camera at script keyframe `k`
    (`rules.hidden_widening`): its half-length beyond `WIDEN_FREE_M` when the thing's top, at its base height plus its
    height, was above the camera there, else 0. 0 when the thing does not exist at `k`."""
    if (mid := w.middle(e, k)) is None:
        return 0.0
    return hidden_widening(e.size, w.camera_at(k)[0][2], mid[2] + e.size[2] / 2)


def place_widening(w: World, e: Entity, xy: tuple[float, float]) -> float:
    """The widening of the rungs of a question about the place `xy` of `e`: the least `hidden_extent` over the thing's
    recognisable sightings in the window at that place, so the rungs grow only when no view there showed the thing's
    top (one sighting from above puts the middle in view). 0 when there is no such sighting."""
    return min((hidden_extent(w, e, k) for k in _near_place(w, e, xy)), default=0.0)


def _render(slots: dict) -> str:
    return f"Where did you last see the {phrase(slots['kind'], slots['attributes'])}? Answer in map coordinates."


class LastSeen(NamedTuple):
    """A thing's last recognisable sighting (at or before an event, inside the event's task, for the change families;
    in the window, for `object_where`): where it stood there (x, y) and the keyframe of that sighting, which a gold
    cites as its evidence."""

    xy: tuple[float, float]
    k: int


class Ref(NamedTuple):
    """A thing a question names: its kind, its describable attributes and the world entity's id, None for a description
    that matches nothing in this world."""

    kind: str
    attributes: dict
    entity: str | None


def ref_of(e: Entity) -> Ref:
    """The question's name for a world entity."""
    return Ref(e.kind, desc_attrs(e.attributes), e.id)


def answerable_at(w: World, moment: tuple[int, int], thing: Entity) -> LastSeen | None:
    """The raw same-task rule at an event: `moment` is (case task index, the script keyframe `k_e` the event resolved
    to). The thing's last recognisable sighting at a window keyframe at or before `k_e`, when it lies in the moment's
    own task and the thing exists there; None otherwise (no recognisable sighting by `k_e`, or the last one in another
    task). Nothing else is asked of the thing here: not the end look window, not whether its place is out of date."""
    task, k_e = moment
    recognised = [k for k in w.recognised(thing.id) if k <= k_e]
    if not recognised:
        return None
    xy = w.xy_at(thing.id, recognised[-1])
    if xy is None or w.task_index(recognised[-1]) != task:
        return None
    return LastSeen(xy, recognised[-1])


def _unseen(w: World, things: list[Entity], k: int) -> list[Ref]:
    """The things no frame of the window shows (no visibility row at any pixel count, so none in the end look window and
    no recognisable sighting) that exist at script keyframe `k`, in the order given."""
    t = w.tl.keyframe(k).t
    return [ref_of(e) for e in things if not w.in_window(e.id) and w.tl.placement_of(e.id, t) is not None]


def unknown_refs(w: World, task: int, k_e: int) -> list[Ref]:
    """The referents of the unknown pool for the event at script keyframe `k_e` of case task `task`: first the world's
    own things of the task's relevance kinds that no frame of the window shows and that exist then, then the
    descriptions that match nothing in this world (`never_seen`). A look-alike hides a real thing here as in the
    answerable pool (`askable`)."""
    relevance = w.relevance_of(task)
    return _unseen(w, askable(w, relevance), k_e) + [Ref(kind, dict(attrs), None)
                                                     for kind, attrs in never_seen(w, relevance)]


def critical(w: World, task: int, thing: Ref, place: LastSeen | None) -> bool:
    """Whether a question about `thing` in case task `task` is task-critical against the job's relevance; `place` is its
    answerable gold's last sighting, None on an unknown row. The thing's kind is one of the task's relevance kinds and,
    on an answerable row, its place lies in a region of a kind the task works in (`Relevance.regions`), within
    `Relevance.near_m` of one of the points `Relevance.near` where the job sets that rule, and, where the job sets
    `Relevance.floor`, the thing stood on the floor the robot drives on. An unknown row is critical by kind: no place
    is known."""
    relevance = w.relevance_of(task)
    if thing.kind not in relevance.kinds:
        return False
    if place is None:
        return True
    region = w.map.region_of(place.xy)
    if region is None or w.map.region(region).kind not in relevance.regions:
        return False
    if relevance.near_m is not None and not any(math.dist(place.xy, p) <= relevance.near_m for p in relevance.near):
        return False
    return not relevance.floor or w.on_floor(thing.entity, place.k)


def last_seen(w: World, thing: Entity) -> LastSeen | None:
    """The thing's last recognisable sighting in the window and its placement there; None when no frame of the window
    shows it recognisably."""
    if not (recognised := w.recognised(thing.id)) or (xy := w.xy_at(thing.id, recognised[-1])) is None:
        return None
    return LastSeen(xy, recognised[-1])


def tolerances_of(w: World, thing: Ref, place: LastSeen | None, widen_m: float = 0.0) -> dict:
    """The `tolerances` of a gold about a place to go back to (0.9.0, `graded`): the job's re-find radius R
    (`JobSpec.reacquire_m`) as the need and the resolution the place rests on as `resolution_m`. The resolution is the
    larger of one pixel at the last recognisable sighting (`resolution_at`: the camera there, the place on the plane
    the thing rests on) and `widen_m`, the half-length of a thing whose top the camera never saw (`hidden_extent`:
    where its middle is cannot be read closer than that). An unknown row (`place` None) carries the floor
    resolution."""
    if place is None:
        return graded_tolerances(UNKNOWN_RHO_M, w.job.reacquire_m, None)
    pixel = at_frame(w, place.k, place.xy, base_of(w, thing.entity, place.k))
    return graded_tolerances(max(pixel, widen_m), w.job.reacquire_m, None)


def _cand(w: World, thing: Ref, gold: dict, place: LastSeen | None, range_m: float | None = None,
          widen_m: float = 0.0) -> Cand:
    """One question: slots the text is rendered from (the thing and nothing else), the gold, and the one subject it
    names (the thing). It is critical when any task of the window makes it so. The row is `graded` (`tolerances`, see
    `tolerances_of`); it records `range_m`, the nearest recognisable slant of the thing in the window, for the reports
    (an unknown row has none). Its `answer_class` is its slice."""
    name = (f"e:{thing.entity}" if thing.entity
            else f"d:{thing.kind}/{json.dumps(thing.attributes, sort_keys=True, separators=(',', ':'))}")
    evidence = (place.k,) if place is not None else ()
    slice_ = "answerable" if evidence else "unknown"
    return Cand("object_where", {"kind": thing.kind, "attributes": thing.attributes}, gold, slice_, evidence,
                (thing.entity,) if evidence else (), tolerances_of(w, thing, place, widen_m),
                subjects=(name,), critical=any(critical(w, i, thing, place) for i, _, _ in w.tasks),
                answer_class=slice_, range_m=range_m)


def pools(w: World) -> tuple[list[Cand], list[Cand]]:
    """(answerable, unknown) candidates of the window. The answerable pool holds its critical things whose last
    recognisable sighting lies before the end look window and that did not move (every later sighting within
    `MOVED_NO_M` of it, no earlier recognisable one a metre away), in world order; a thing with any row in the end look
    window is in neither pool. The unknown pool lists the world's own unseen things first, then the absent descriptions
    (`never_seen` under each task's relevance), less those the glimpse screen drops (`glimpse_hides`); every moment
    offers it beside its thing."""
    answerable: list[Cand] = []
    things = askable_in_window(w)
    for e in things:
        if w.in_end_window(e.id) or (last := last_seen(w, e)) is None:
            continue
        if not w.steady(e, last.k, last.xy, w.rec.k1 - 1) or w.arrived(e, last.k, last.xy):  # moved: `object_was`'s
            continue
        if (near := nearest_sighting_at(w, e, last.xy)) is None:   # the sighting of the place itself that was nearest
            continue
        cand = _cand(w, ref_of(e), {"type": "position", "xy": list(last.xy)}, last, near[0],
                     place_widening(w, e, last.xy))
        if cand.critical:
            answerable.append(cand)
    refs = _unseen(w, things, w.rec.k1 - 1)
    refs += [Ref(kind, dict(attrs), None) for kind, attrs in
             dict.fromkeys(key for i, _, _ in w.tasks for key in never_seen(w, w.relevance_of(i)))]
    return answerable, [_cand(w, ref, {"type": "unknown"}, None) for ref in refs
                        if not glimpse_hides(w, ref.kind, ref.attributes, ref.entity)]


def moments(w: World) -> list[Moment]:
    """One moment for each answerable thing, keyed by its description (kind and describable attributes, as JSON): the
    thing in `answerable`, the window's unknown pool beside it."""
    answerable, unknown = pools(w)
    return [Moment(json.dumps([c.slots["kind"], c.slots["attributes"]], sort_keys=True, separators=(",", ":")), [c],
                   unknown) for c in answerable]


SPECS = (FamilySpec("object_where", "end", "position", "graded", ("see", "pose"), "memory", None, False,
                    ("kind", "attributes"), _render),)
GOLDS = {"object_where": moments}
SLICED = frozenset({"object_where"})
