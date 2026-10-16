"""`when_seen`, `order` and `distance_between` (end, `memory`): when the robot last saw a thing, which of two things of
one kind it saw first, and how far apart two were as it last saw them.

They share `object_where`'s things (`World.things`: carried things of the relevance kinds of the window's tasks whose
description is unique and that the log does not tell about) and its kinds of pool: an unknown row asks about something
no sensor recorded. A thing is one of:
- **settled**: it has a recognisable sighting in the window (`World.recognised`); no visibility row in the end look
  window (the current view would answer: VISION rule 4); and every sighting at or above its glimpse bound after its
  last recognisable one lies within `rules.MOVED_NO_M` of that one's place (`World.steady`): the memory table, which
  reads the latest sighting, then holds the thing where the gold read it, give or take a clipped exit frame's few
  centimetres;
- **unseen**: it exists in the window and no visibility row shows it anywhere in it, or it is a description of a
  relevance kind that matches nothing in this world (`questions_where.never_seen`: a decoy that gives nothing away);
- **skipped**: anything else (a glimpse, only dark, clipped or depth-less, a later sighting elsewhere).
A thing is named by a description, `{"kind", "attributes"}` (`descriptions.desc_attrs`), top level for `when_seen`,
nested in `first` and `second` for the two that name a pair (`blind` loosens exactly those two keys). A subject is
`e:<id>`, or `d:<kind>/<attributes>` for a description the world lacks. Every thing is of a relevance kind; an
answerable `distance_between` row is critical when both things are, each at its last recognisable place
(`questions_where.critical`), and any other row is.

- `when_seen`: moments are the relevance kinds (keyed by kind). Gold is the case clock of the thing's last recognisable
  sighting, which is the evidence; the unknown pool is the unseen things and descriptions of the kind. A time is not
  held to `MOVED_NO_M`: the thing must have no sighting at or above its glimpse bound after that one, which the memory
  table would otherwise read as its last. Scored on `TIME_LADDER_S`, the minute scale (design §8.3).
- `order`: **two things of one kind only.** A pair of kinds would name where each lives (shoes on the hall's rack and
  mat, toys on rugs and the bed), which predicts the order; one kind leaves two descriptions that differ in attributes,
  drawn independently of placement. Moments are the relevance kinds. A pair is two settled things of the kind whose
  first recognisable sightings lie at least `ORDER_GAP_S` apart and whose later-seen one has no visibility row at or
  before the earlier one's first recognisable sighting (a detection at that keyframe would leave the table unable to
  order them). Which is named first is a coin per pair (`stable_hash(case id, the two subjects, "named") % 2`),
  independent of the answer, so yes and no balance within each kind. Gold is whether the thing named first was seen
  first. No unknown slice.
- `distance_between`: moments are the unordered pairs of relevance kinds, one kind twice included (keyed "a|b", sorted).
  A pair is two settled things whose positions at their last recognisable sightings lie at least `MIN_APART_M` apart
  (a gold well above 0); gold is that distance in metres, 3 places, from the delivered sightings only: a thing moved
  after its last sighting keeps the place it was last seen. The credit is the absolute ladder (the job's,
  `JobSpec.ladder`) widened by both things' `rules.widening` (a surface seen is not a centre), on the unknown rows too.
  The unknown pool is a settled thing of one kind with an unseen thing of the other, named in the same coin's order.
  The robot's own log tells about some things (`World.told`) and `World.things` leaves them out. **Both things of
  a row are screened** (`questions_where.nameable`, as `object_where`'s are): a real thing, settled or unseen, whose
  picture does not show its colour word, or that another sighted thing of a confusable kind could be taken for the
  colour of (by its render), is in no row, since a system that holds the look-alike can not tell which one is named.
  `when_seen` and `order` (no job asks them) keep their pools.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` and `SLICED`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from itertools import combinations, combinations_with_replacement, product
from typing import NamedTuple

from .descriptions import desc_attrs, phrase
from .questions_select import Cand, Moment
from .questions_where import LastSeen, critical, nameable, never_seen, widened_ladder
from .questions_world import World
from .records import FamilySpec
from .seeding import stable_hash
from .timeline import Entity

TIME_LADDER_S = (15.0, 30.0, 60.0, 120.0)  # seconds: "minute scale" (design §8.3)
ORDER_GAP_S, MIN_APART_M = 30.0, 0.5


class Ref(NamedTuple):
    """A thing a question names: a description, and the entity it matches (None: nothing in this world)."""

    kind: str
    attributes: dict
    entity: str | None

    @property
    def slot(self) -> dict:
        return {"kind": self.kind, "attributes": dict(self.attributes)}

    @property
    def subject(self) -> str:
        if self.entity:
            return f"e:{self.entity}"
        return f"d:{self.kind}/{json.dumps(self.attributes, sort_keys=True, separators=(',', ':'))}"


class Settled(NamedTuple):
    """A settled thing and its first and last recognisable sightings (script keyframes)."""

    ref: Ref
    first: int
    last: int


def kinds(w: World) -> list[str]:
    """The relevance kinds of the window's tasks, sorted."""
    return sorted({k for i, _, _ in w.tasks for k in w.relevance_of(i).kinds})


def _things(w: World) -> list[Entity]:
    """`World.things` under each task's relevance, once each."""
    found: dict[str, Entity] = {}
    for i, _, _ in w.tasks:
        for e in w.things(w.relevance_of(i)):
            found.setdefault(e.id, e)
    return list(found.values())


def _exists(w: World, entity: str) -> bool:
    """The entity exists at some time of the window."""
    t0, t1 = w.tl.keyframe(w.rec.k0).t, w.tl.keyframe(w.rec.k1 - 1).t
    return any(p.t0 <= t1 and (p.t1 is None or p.t1 > t0) for p in w.tl.placements_of(entity))


def settled(w: World) -> dict[str, list[Settled]]:
    """The settled things of each relevance kind (the module docstring), in timeline order."""
    out: dict[str, list[Settled]] = {kind: [] for kind in kinds(w)}
    for e in _things(w):
        seen = w.recognised(e.id)
        if not seen or w.in_end_window(e.id):
            continue
        xy = w.xy_at(e.id, seen[-1])
        if xy is not None and w.steady(e, seen[-1], xy, w.rec.k1 - 1):
            out[e.kind].append(Settled(Ref(e.kind, desc_attrs(e.attributes), e.id), seen[0], seen[-1]))
    return out


def unseen(w: World) -> dict[str, list[Ref]]:
    """The unseen things of each relevance kind: the world's own first, then the descriptions it lacks."""
    out: dict[str, list[Ref]] = {kind: [] for kind in kinds(w)}
    for e in _things(w):
        if not w.in_window(e.id) and _exists(w, e.id):
            out[e.kind].append(Ref(e.kind, desc_attrs(e.attributes), e.id))
    absent = dict.fromkeys(key for i, _, _ in w.tasks for key in never_seen(w, w.relevance_of(i)))
    for kind, attrs in absent:
        out[kind].append(Ref(kind, dict(attrs), None))
    return out


def _named(w: World, x: Ref, y: Ref) -> tuple[Ref, Ref]:
    """The two things in the order a question names them: of the two in subject order, the `stable_hash(case id, both
    subjects, "named") % 2`-th goes first. The coin reads no answer, so it predicts none."""
    a, b = sorted((x, y), key=lambda r: r.subject)
    return (a, b) if stable_hash(w.rec.case, a.subject, b.subject, "named") % 2 == 0 else (b, a)


def _when(w: World, ref: Ref, last: int | None = None) -> Cand:
    """A `when_seen` candidate: the clock of the thing's last recognisable sighting `last`, or unknown (None)."""
    ladder = {"ladder": list(TIME_LADDER_S)}
    if last is None:
        return Cand("when_seen", ref.slot, {"type": "unknown"}, "unknown", (), (), ladder, subjects=(ref.subject,))
    return Cand("when_seen", ref.slot, {"type": "time", "clock": w.clock(last)}, "answerable", (last,), (ref.entity,),
                ladder, subjects=(ref.subject,))


def when_seen(w: World) -> list[Moment]:
    """One moment for each relevance kind: its settled things, and its unseen things and descriptions."""
    found, absent = settled(w), unseen(w)
    return [Moment(kind, [_when(w, s.ref, s.last) for s in found[kind] if w.sighted(s.ref.entity)[-1] == s.last],
                   [_when(w, r) for r in absent[kind]])
            for kind in found]


def _order(w: World, a: Settled, b: Settled) -> Cand | None:
    """The `order` candidate of a pair of settled things of one kind, or None when the table could not order them: the
    first recognisable sightings are under `ORDER_GAP_S` apart, or the later-seen has a row at or before the earlier's
    first recognisable one."""
    early, late = sorted((a, b), key=lambda s: s.first)
    if w.tl.keyframe(late.first).t - w.tl.keyframe(early.first).t < ORDER_GAP_S:
        return None
    if w.in_ks(late.ref.entity, range(w.rec.k0, early.first + 1)):
        return None
    first, second = _named(w, a.ref, b.ref)
    return Cand("order", {"first": first.slot, "second": second.slot},
                {"type": "yesno", "value": first.entity == early.ref.entity}, "answerable",
                (early.first, late.first), (early.ref.entity, late.ref.entity),
                subjects=(first.subject, second.subject))


def order(w: World) -> list[Moment]:
    """One moment for each relevance kind: every orderable pair of its settled things. No unknown slice."""
    return [Moment(kind, [c for a, b in combinations(things, 2) if (c := _order(w, a, b))], [])
            for kind, things in settled(w).items()]


def _distance(w: World, x: Settled, y: Settled) -> Cand | None:
    """The `distance_between` candidate of two settled things, or None when they stood under `MIN_APART_M` apart as
    last seen. It is critical when both things are, each at its last recognisable place in the task it was seen in."""
    xy = [w.xy_at(s.ref.entity, s.last) for s in (x, y)]
    d = math.dist(*xy)
    if d < MIN_APART_M:
        return None
    first, second = _named(w, x.ref, y.ref)
    (k0, e0), (k1, e1) = sorted(((x.last, x.ref.entity), (y.last, y.ref.entity)))
    important = all(critical(w, w.task_index(s.last), s.ref, LastSeen(p, s.last))
                    for s, p in zip((x, y), xy, strict=True))
    return Cand("distance_between", {"first": first.slot, "second": second.slot},
                {"type": "number", "value": round(d, 3)}, "answerable", (k0, k1), (e0, e1),
                widened(w, x.ref, y.ref), subjects=(first.subject, second.subject), critical=important)


def widened(w: World, x: Ref, y: Ref) -> dict:
    """The tolerances of a `distance_between` row: the job's distance rungs plus the two things' widening."""
    return {"ladder": widened_ladder(w.job.ladder, x.kind, y.kind)}


def _far(w: World, x: Ref, y: Ref) -> Cand:
    """A `distance_between` candidate whose answer no sensor recorded: `y` was not seen."""
    first, second = _named(w, x, y)
    return Cand("distance_between", {"first": first.slot, "second": second.slot}, {"type": "unknown"}, "unknown",
                tolerances=widened(w, x, y), subjects=(first.subject, second.subject))


def _pairs(a: list, b: list, same: bool) -> Iterable[tuple]:
    """Every pair of the two kinds' things: the unordered pairs of one list when both kinds are the one."""
    return combinations(a, 2) if same else product(a, b)


def _clear(w: World, ref: Ref) -> bool:
    """Whether `distance_between` may name the thing: a real one has its colour borne out by its picture and is not
    hidden by a sighted look-alike (`questions_where.nameable`); a description the world lacks was screened by
    `never_seen` already."""
    return nameable(w, ref.kind, ref.attributes, ref.entity)


def distance_between(w: World) -> list[Moment]:
    """One moment for each unordered pair of relevance kinds (a kind with itself too), keyed "a|b": the far enough pairs
    of settled things, and each settled thing with an unseen thing of the pair's other kind, the things a look-alike
    hides (`_clear`) left out."""
    found = {kind: [s for s in things if _clear(w, s.ref)] for kind, things in settled(w).items()}
    absent = {kind: [r for r in refs if _clear(w, r)] for kind, refs in unseen(w).items()}
    out = []
    for a, b in combinations_with_replacement(found, 2):
        answerable = [c for x, y in _pairs(found[a], found[b], a == b) if (c := _distance(w, x, y))]
        sides = [(a, b)] if a == b else [(a, b), (b, a)]
        unknown = [_far(w, s.ref, u) for p, q in sides for s, u in product(found[p], absent[q])]
        out.append(Moment(f"{a}|{b}", answerable, unknown))
    return out


def _thing(slots: dict) -> str:
    return phrase(slots["kind"], slots["attributes"])


SPECS = (
    FamilySpec("when_seen", "end", "time", "ladder", ("see",), "memory", None, False, ("kind", "attributes"),
               lambda s: f"When did you last see the {_thing(s)}?"),
    FamilySpec("order", "end", "yesno", "exact", ("see",), "memory", None, False, ("first", "second"),
               lambda s: f"Did you first see the {_thing(s['first'])} before the {_thing(s['second'])}?"),
    FamilySpec("distance_between", "end", "number", "ladder", ("see", "pose"), "memory", None, False,
               ("first", "second"),
               lambda s: f"How far apart were the {_thing(s['first'])} and the {_thing(s['second'])}, as you last saw "
                         f"them? Answer in metres."))
GOLDS = {"when_seen": when_seen, "order": order, "distance_between": distance_between}
SLICED = frozenset({"when_seen", "distance_between"})
