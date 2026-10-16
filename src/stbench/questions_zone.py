"""`zone_clear` (during), `clearance_then` and `slots_empty` (end): is the place the next manoeuvre uses clear right
now, was it clear at a named moment, and which places of a kind were last seen clear.

A zone is a place of the public map (`zones`: a piece's footprint on its ground, an interior opening's way, or the
window of the surface the robot faces; inset, and judged on its own band), chosen from the map and the robot's pose,
never from the world; the job declares which place a decision kind or an event wording names (`JobSpec.zones`) and
which pieces `slots_empty` lists (`JobSpec.slot_zones`). The state of a zone is `zones.zone_state` over the free-space
cells observed with the zone's own ground and band (`World.zone_states`): the one rule the table systems call too. No
family names a robot, a kind, a phase or a wording.
- **`zone_clear`**: the rule of the decision's kind; the zone nearest the base (a piece or a way), or the window the
  base faces (`zones.SPOT`: it needs the delivered heading). It asks about the look window that ends at the decision,
  and nothing else: the gold is `zones.zone_state` over `World.window_ks(k)` alone, so the scope is `now` and there is
  no "as you last saw it" form (a memory row of a zone the window did not cover was a clear bay by construction, and
  `blind` scored +0.67 on it). One blocker in the window certifies *blocked*; at least `ZONE_SHARE` of the zone's cells
  seen free certifies *clear*; else `unknown`, a gold of the unknown slice, never a dead band. The current view answers
  it (VISION rule 4).
- **`clearance_then`**: one moment for each (task, wording) that is both an event the task resolved and a key of
  `JobSpec.zones`; the zone nearest the base at the event, its state over the keyframes of the look window that holds
  the event **up to and including it** (what was seen at or before the moment, never after). A zone with any cell
  observed in the end look window is not asked: the current view would answer it (rule 4). Not sliced: the rule gives
  `unknown`.
- **`slots_empty`**: one moment for each rule of `JobSpec.slot_zones`. Each zone is *passed* when any of its cells was
  observed in the window; a passed zone with a cell observed in the end look window is a don't-care place at its
  centroid (the current view); else clear: an item at its centroid; blocked: nothing; unknown: a don't-care place. A
  zone never passed is nothing. The things standing on a passed zone as last seen are subjects, so no other end question
  names something whose place tells a slot's state.
A zone is a subject as the timeline entity of its piece (`e:`), found by the piece's kind, name and place; a doorway's
is its door's entity, and an arch has none (the question then carries only its time, `w:<k>`).

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` and `SLICED` (none).
"""

from __future__ import annotations

import math
from itertools import combinations

from . import zones
from .geometry import Rect, corners
from .job_registry import JobSpec
from .questions_select import Cand, Moment
from .questions_words import task_phrase
from .questions_world import World
from .records import FamilySpec
from .rules import MATCH_M
from .zones import Zone, ZoneRule

THINGS = ("light", "carried")  # the mobilities a question may name and so a slot's state may be told by
MATCH_SHARE = 0.45  # `slots_empty`'s match radius is at most this share of the least distance between two zones
STATE_GOLD = {"clear": ({"type": "yesno", "value": True}, "answerable"),
            "blocked": ({"type": "yesno", "value": False}, "answerable"),
            "unknown": ({"type": "unknown"}, "unknown")}  # zone_state -> (the yes/no gold, its slice)


def check_rules(job: JobSpec) -> None:
    """ValueError naming the job when its zone rules cannot be found by a table system, which finds a question's rule
    by its words: two different rules (of `zones` and `slot_zones` together) share `words`, or two different slot
    rules share `piece_words`; a slot rule has no `piece_words`; a key of `zones` is neither a decision kind nor an
    event wording; or a decision kind that asks `zone_clear` has no rule."""
    problems = []
    seen: dict[str, ZoneRule] = {}
    for rule in [*job.zones.values(), *job.slot_zones]:
        if seen.setdefault(rule.words, rule) != rule:
            problems.append(f"two different rules share the words {rule.words!r}")
    plural: dict[str, ZoneRule] = {}
    for rule in job.slot_zones:
        if not rule.piece_words:
            problems.append(f"the slot rule {rule.words!r} has no piece_words")
        elif plural.setdefault(rule.piece_words, rule) != rule:
            problems.append(f"two different slot rules share the piece_words {rule.piece_words!r}")
    problems += [f"the zone {key!r} is neither a decision kind nor an event wording of the job"
                 for key in job.zones if key not in job.decisions and key not in job.events]
    problems += [f"the decision kind {kind!r} asks zone_clear and has no zone"
                 for kind, families in job.decisions.items() if "zone_clear" in families and kind not in job.zones]
    if problems:
        raise ValueError(f"the job {job.id}: " + "; ".join(problems))


def _entity(w: World, center, kind: str, mobility: str, what: str, name: str | None = None) -> str:
    """The subject `e:<id>` of the timeline entity of that kind and mobility (and name, when given) that stands at
    `center` at the start; ValueError names the map piece (`what`) when there is none."""
    for e in w.tl.entities:
        if e.kind == kind and e.mobility == mobility and (name is None or e.attributes.get("name") == name):
            p = w.tl.placement_of(e.id, 0.0)
            if p is not None and math.dist(p.pos[:2], center) < 1e-5:
                return f"e:{e.id}"
    raise ValueError(f"the map piece {what} has no entity in the timeline")


def _subject(w: World, rule: ZoneRule, zone: Zone) -> str | None:
    """The subject of the zone: `e:<id>` of the timeline entity of its map piece, found by kind, name and place; a
    doorway's is its door's entity, and an arch (no entity) has none, so None."""
    if rule.piece == zones.DOORWAY:
        o = next(o for o in w.map.openings if o.id == zone.piece)
        return _entity(w, o.center, "door", "fixed", f"{o.id} (door)") if o.type == "door" else None
    f = w.map.furniture_by_id()[zone.piece]
    return _entity(w, f.center, f.kind, f.mobility, f"{f.id} ({f.kind}, {f.name})", f.name)


def _subjects(w: World, k: int, rule: ZoneRule, zone: Zone) -> tuple[str, ...]:
    """A question's subjects: the time `w:<k>` and the zone's piece, when it has an entity."""
    piece = _subject(w, rule, zone)
    return (f"w:{k}",) if piece is None else (f"w:{k}", piece)


def _yesno(state: str, evidence: list[int]) -> tuple[dict, str, tuple[int, ...]]:
    gold, slice_ = STATE_GOLD[state]
    return gold, slice_, tuple(evidence)


# -- zone_clear -------------------------------------------------------------------------------------------------
def _render_clear(slots: dict) -> str:
    return f"Is {slots['zone']} clear of anything taller than {slots['height']:g} m right now?"


def zone_clear(w: World, k: int) -> Cand | None:
    """The question asked at the decision point at script keyframe `k`, with its gold: `zones.zone_state` over the
    decision's look window alone. None when the map has no zone for the rule at that pose (no piece of its kind, or no
    surface faced: a dead band). ValueError when the decision's kind has no zone: a job bug."""
    kind = next(d.kind for d in w.decisions if d.k == k)
    if (rule := w.job.zones.get(kind)) is None:
        raise ValueError(f"the job {w.job.id} asks zone_clear at {kind!r} and declares no zone for it")
    if (zone := zones.zone_of(w.map, rule, w.base(k), w.yaw(k))) is None:
        return None
    gold, slice_, evidence = _yesno(*zones.zone_state(w.zone_states(w.window_ks(k), zone), zone))
    return Cand("zone_clear", {"zone": rule.words, "height": rule.height_m}, gold, slice_, evidence, (), {},
                decision_k=k, subjects=_subjects(w, k, rule, zone), critical=True)


# -- clearance_then ---------------------------------------------------------------------------------------------
def _render_then(slots: dict) -> str:
    return (f"Was {slots['zone']} clear of anything taller than {slots['height']:g} m {slots['event']}"
            f"{task_phrase(slots['task'])}?")


def clearance_then(w: World) -> list[Moment]:
    """One moment for each (task, wording) that the window's events resolved and `JobSpec.zones` names, keyed
    "<task>|<wording>"; a moment whose zone has a cell observed in the end look window, or whose map has no such piece,
    is not asked."""
    out = []
    for (task, wording), k_e in w.events.items():
        if ((rule := w.job.zones.get(wording)) is None
                or (zone := zones.zone_of(w.map, rule, w.base(k_e), w.yaw(k_e))) is None):
            continue
        if zones.coverage(w.zone_states(w.end_window, zone), zone) > 0:
            continue
        seen = w.zone_states([j for j in w.window_of(k_e) if j <= k_e], zone)
        gold, slice_, evidence = _yesno(*zones.zone_state(seen, zone))
        slots = {"zone": rule.words, "height": rule.height_m, "event": wording, "task": task if w.multi else None}
        out.append(Moment(f"{task}|{wording}", [Cand("clearance_then", slots, gold, slice_, evidence, (), {},
                                                     subjects=_subjects(w, k_e, rule, zone), critical=True)], []))
    return out


# -- slots_empty ------------------------------------------------------------------------------------------------
def _render_slots(slots: dict) -> str:
    return (f"Which {slots['places']} that you passed were clear of anything taller than {slots['height']:g} m when "
            f"you last saw them? Give where each one is.")


def _match_m(found: list[Zone]) -> float:
    """The match radius: the least of half the longest side of the rule's zones and `MATCH_SHARE` of the least distance
    between two of their centroids, at least `rules.MATCH_M` (so two zones' items are never one match)."""
    longest = max(math.dist(a, b) for z in found for a, b in zip(z.polygon, z.polygon[1:] + z.polygon[:1],
                                                                 strict=True))
    apart = min((math.dist(zones.centroid(a), zones.centroid(b)) for a, b in combinations(found, 2)),
                default=math.inf)
    return round(max(MATCH_M, min(longest / 2, MATCH_SHARE * apart)), 6)


def _overlaps(rect: Rect, polygon: tuple[tuple[float, float], ...]) -> bool:
    """Whether the rectangle and the convex polygon share area (touching edges do not): no separating edge normal."""
    a, b = corners(rect), list(polygon)
    for poly in (a, b):
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True):
            nx, ny = y1 - y0, x0 - x1
            pa, pb = [nx * x + ny * y for x, y in a], [nx * x + ny * y for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return False
    return True


def _standing(w: World, passed: list[Zone]) -> set[str]:
    """The things (`THINGS`) with a sighting in the window whose placement at the last one overlaps a passed zone."""
    out = set()
    for e in w.tl.entities:
        if e.mobility in THINGS and (ks := w.sighted(e.id)):
            rect = w.tl.footprint(e.id, w.tl.keyframe(ks[-1]).t)
            if rect is not None and any(_overlaps(rect, z.polygon) for z in passed):
                out.add(f"e:{e.id}")
    return out


def slots_empty(w: World) -> list[Moment]:
    """One moment for each rule of `JobSpec.slot_zones`, keyed by its `piece_words`; none when no zone of it was
    passed (observed in the window). The gold is never `unknown`: nothing clear is the empty list."""
    window, latest = sorted(w.rows), {}

    def cells(ks, zone: Zone):  # grid.latest of the keyframes on the zone's band, once a band
        return latest.setdefault((tuple(ks), zone.base_z, zone.band), w.zone_states(ks, zone))

    out = []
    for rule in w.job.slot_zones:
        found = zones.zones_of(w.map, rule)
        passed = [z for z in found if zones.coverage(cells(window, z), z) > 0]
        if not passed:
            continue
        items, care, evidence = [], [], set()
        for z in passed:
            state, seen = zones.zone_state(cells(window, z), z)
            evidence.update(seen)
            if zones.coverage(cells(w.end_window, z), z) > 0 or state == "unknown":
                care.append(zones.centroid(z))
            elif state == "clear":
                items.append(zones.centroid(z))
        radius = _match_m(found)
        gold: dict = {"type": "positions", "items": [{"xy": list(p)} for p in items]}
        if care:
            gold["dont_care"] = [{"xy": list(p), "radius": radius} for p in care]
        subjects = {s for z in passed if (s := _subject(w, rule, z)) is not None} | _standing(w, passed)
        slots = {"places": rule.piece_words, "height": rule.height_m}
        out.append(Moment(rule.piece_words, [Cand("slots_empty", slots, gold, "answerable", tuple(sorted(evidence)), (),
                                                  {"match_m": radius}, subjects=tuple(sorted(subjects)),
                                                  critical=True)], []))
    return out


SPECS = (
    FamilySpec("zone_clear", "during", "yesno", "exact", ("see", "pose"), "now", None, True, ("zone", "height"),
               _render_clear),
    FamilySpec("clearance_then", "end", "yesno", "exact", ("see", "pose", "status"), "memory", None, True,
               ("zone", "height", "event", "task"), _render_then),
    FamilySpec("slots_empty", "end", "positions", "f1", ("see", "pose"), "memory", None, True,
               ("places", "height"), _render_slots))
GOLDS = {"zone_clear": zone_clear, "clearance_then": clearance_then, "slots_empty": slots_empty}
SLICED: frozenset[str] = frozenset()
