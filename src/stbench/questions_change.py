"""The change families (0.6.0): `object_was` and `same_place` (end, `memory`, temporal), and `person_where`.

They ask what only a record of the past can answer: where a thing stood in an earlier or a later task, whether it stood
in one place in two tasks, where a person was when last seen. **Time and order are the tasks'** (FD-0.6.0, §2.4): the
world changes only in the gaps between tasks, so a task is the finest time step that has an answer, and a question
names a task by its ordinal and no event of the job. A gold is what was last seen; nothing unobserved changes it (truth
rule 2).

**A thing's place in a task** is its last recognisable sighting in the task (`questions_where.answerable_at` at the
task's last keyframe: the sighting lies inside the task) when every sighting of it from there to the task's end, a
glimpse or a dark one too, stays within `MOVED_NO_M` of it (`World.steady`: so the memory table, which reads the latest
sighting, holds it where the gold reads it). The thing's **first place** in a task is its first recognisable sighting
when nothing was sighted elsewhere from the task's start to it. The things are `questions_where.askable` under the
relevance of the task a question names (carried, unique, not told, no look-alike hides them).
- `object_was` asks "Where was the {thing} during your {ordinal} task? Answer in map coordinates." (slots `kind`,
  `attributes`, `task`: the case task index) of a thing with places in two tasks that lie **beyond the asked question's
  last rung** of each other. The rungs are the relative ladder (`rules.relative_ladder`) of the nearest recognisable
  slant at which the place was seen in the window (`range_m`), widened for the kind; the gap is read against the rungs
  of the task asked, so an answer at the other task's place earns nothing. A thing is asked at its **earlier** state
  when a later task's place is that far from the place in the task named, at its **later** state when an earlier task's
  is; a thing in a task between two such places has both candidates. A thing in the end look window is not asked
  (the current view would answer it, as for `object_where`). The gold is the place in the named task. The family is a
  `COINS` family: the earlier candidates are the moment's `answerable` pool (class `earlier`), the later ones its
  `unknown` pool (class `later`), and a moment is `draw:<j>`, a draw index over the window (not a task, which a window
  of two tasks could name only once in each state): there are as many draws as the smaller pool has things, each
  offering both pools whole, so a draw is asked only when both pools allow a candidate and the fair coin of the
  questionnaire takes one. Neither the draw nor whether it is asked predicts the state.
- `same_place` asks "Was the {thing} in the same place in your {second} task as in your {first}?" (slots `kind`,
  `attributes`, `tasks`: the two case task indices, earlier first). A thing with a place in task i (its last) and a
  first place in task j > i is **yes** when they are under `SAME_M` (0.15 m) apart and **moved** (a no) from
  max(`APART_M` (0.45 m), twice the job's unit) plus the kind's widening on; between the two it is not asked. A thing
  with a place in task i and no sighting at all in task j is **vacated** (a no) when its task-i place was seen empty in
  task j (`grid.place_free` over the task's free-space cells and the thing's half-width: a place nobody looked at is
  never empty, truth rule 2) and the robot's band can show it gone: the robot drives (a flier reads every ground point
  as floor), the thing stood on the floor and its top is above the band's floor (`_can_vacate`). A thing in the
  end look window is not asked (the current view would answer it). Yes candidates are the moment's `answerable` pool
  (class `yes`), no candidates its `unknown` pool (class `moved` or `vacated`, reported apart). A moment is (task pair,
  kind, draw index), with as many draws as the smaller pool has things. The gold is a yes/no, `answerable` either way
  (its evidence is the two sightings, or the task-i sighting and the keyframe that saw the place empty).
- `person_where` asks where a person was, as the robot last saw them, at a moment of the job's events
  (`World.events`: one wording resolved over one task of the window): the movers (`mobility` "mover") whose
  description is the only one of its kind in the world. The gold is the person's last recognisable sighting at or
  before the moment, in the moment's task, when nothing seen between that sighting and the moment puts the person
  elsewhere (`World.steady`), and the person has no row in the end look window. No slice. It keeps the job's ladder
  and its event wording.
Subjects are `e:<id>` only (a task ordinal is no subject, rule 6 amended); `person_where` keeps its named time `w:<k>`.
A row is critical (`questions_where.critical`, under the task named) as `object_where`'s is; a `same_place` row is
critical when either of its places is.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS`, `SLICED` (none) and `COINS` (`object_was`,
`same_place`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import NamedTuple

from . import grid
from .catalog import item_size
from .descriptions import phrase
from .questions_select import Cand, Moment
from .questions_where import (
    LastSeen,
    answerable_at,
    askable,
    critical,
    nearest_slant_at,
    ref_of,
)
from .questions_words import ORDINALS, task_phrase
from .questions_world import World
from .records import FamilySpec
from .rules import MOVED_NO_M, MOVED_YES_M, relative_ladder, widening
from .timeline import Entity

SAME_M = MOVED_NO_M  # two places closer than this are one place: a `same_place` yes
APART_M = MOVED_YES_M  # two places this far apart, at least (or twice the job's unit) and widened, are two: a no
EARLIER, LATER = "earlier", "later"  # the two states `object_was` asks, its two pools
YES, MOVED, VACATED = "yes", "moved", "vacated"  # the classes of `same_place`: the yes pool and the two no ones


# -- places ------------------------------------------------------------------------------------------------------
def _task_ks(w: World, task: int) -> range:
    """The window's script keyframes of case task `task`."""
    _, k0, k1 = next(t for t in w.tasks if t[0] == task)
    return range(max(k0, w.rec.k0), min(k1, w.rec.k1))


def last_place(w: World, e: Entity, task: int) -> LastSeen | None:
    """The entity's place in case task `task`: its last recognisable sighting there, when it stayed within `MOVED_NO_M`
    of it from then to the task's end; None otherwise."""
    end = _task_ks(w, task)[-1]
    last = answerable_at(w, (task, end), e)
    return last if last is not None and w.steady(e, last.k, last.xy, end) else None


def first_place(w: World, e: Entity, task: int) -> LastSeen | None:
    """The entity's first place in case task `task`: its first recognisable sighting there, when nothing was sighted
    elsewhere (recognisable or not) from the task's start to it; None otherwise."""
    ks = _task_ks(w, task)
    first = next((k for k in w.recognised(e.id) if k in ks), None)
    xy = None if first is None else w.xy_at(e.id, first)
    return LastSeen(xy, first) if xy is not None and w.steady(e, ks[0], xy, first) else None


def _by_task(w: World) -> dict[int, dict[str, Entity]]:
    """The things a question may name in each case task (`questions_where.askable` under its relevance), by id."""
    return {i: {e.id: e for e in askable(w, w.relevance_of(i))} for i, _, _ in w.tasks}


# -- object_was --------------------------------------------------------------------------------------------------
class Spot(NamedTuple):
    """A thing's place in one task and the nearest recognisable slant (m) at which that place was seen."""

    last: LastSeen
    r: float


def spots(w: World, by_task: dict[int, dict[str, Entity]]) -> dict[str, tuple[Entity, dict[int, Spot]]]:
    """For each entity outside the end look window with a place in some task it may be named in, the entity and its
    `Spot` in each such task."""
    out: dict[str, tuple[Entity, dict[int, Spot]]] = {}
    for task, things in by_task.items():
        for eid, e in things.items():
            if w.in_end_window(eid) or (last := last_place(w, e, task)) is None:
                continue
            if (r := nearest_slant_at(w, e, last.xy)) is not None:
                out.setdefault(eid, (e, {}))[1][task] = Spot(last, r)
    return out


def _was(w: World, e: Entity, task: int, spot: Spot, state: str) -> Cand:
    """The `object_was` candidate for one thing in one task at one state: the gold is its place there, the rungs the
    relative ladder of the slant it was seen at."""
    ref = ref_of(e)
    return Cand("object_was", {"kind": ref.kind, "attributes": ref.attributes, "task": task},
                {"type": "position", "xy": list(spot.last.xy)}, "answerable", (spot.last.k,), (e.id,),
                relative_tolerances(spot.r, e.kind), subjects=(f"e:{e.id}",),
                critical=critical(w, task, ref, spot.last), answer_class=state, range_m=spot.r)


def relative_tolerances(r_m: float, kind: str) -> dict:
    """The `tolerances` of a memory position gold about a thing of `kind` seen at its nearest at `r_m` metres:
    `rules.relative_ladder` of `r_m`, widened for the kind (`rules.widening`). The change families' (0.6.0); no family
    a job asks credits so since 0.7.0."""
    return {"ladder": relative_ladder(r_m, widening(item_size(kind)))}


def _draws(first: list[Cand], second: list[Cand]) -> int:
    """How many draws two pools can feed: the smaller number of things either holds."""
    return min(len({c.subjects[0] for c in first}), len({c.subjects[0] for c in second}))


def object_was(w: World) -> list[Moment]:
    """The window's `draw:<j>` moments, each holding every earlier-state candidate in `answerable` and every later-state
    one in `unknown`: a thing is a candidate at a task when a place it has in a later (earlier) task is beyond the last
    rung of this task's relative ladder from its place here."""
    earlier: list[Cand] = []
    later: list[Cand] = []
    for e, per in spots(w, _by_task(w)).values():
        for task, spot in per.items():
            rung = relative_ladder(spot.r, widening(item_size(e.kind)))[-1]
            far = [t for t, other in per.items() if math.dist(other.last.xy, spot.last.xy) > rung + grid.EPS]
            if any(t > task for t in far):
                earlier.append(_was(w, e, task, spot, EARLIER))
            if any(t < task for t in far):
                later.append(_was(w, e, task, spot, LATER))
    return [Moment(f"draw:{j}", earlier, later) for j in range(_draws(earlier, later))]


# -- same_place --------------------------------------------------------------------------------------------------
def empty_since(states: dict, xy: tuple[float, float], radius: float) -> int | None:
    """The latest keyframe that observed a fine cell of the disc of `radius` around `xy`, when the place was seen empty
    (`grid.place_free` over `states`, `grid.latest`'s output); None when it was not."""
    if not grid.place_free(states, xy, radius):
        return None
    us = range(math.floor((xy[0] - radius) / grid.FINE_M) - 1, math.floor((xy[0] + radius) / grid.FINE_M) + 2)
    vs = range(math.floor((xy[1] - radius) / grid.FINE_M) - 1, math.floor((xy[1] + radius) / grid.FINE_M) + 2)
    return max(states[(u, v)][1] for u in us for v in vs if (u, v) in states
               and math.hypot((u + 0.5) * grid.FINE_M - xy[0], (v + 0.5) * grid.FINE_M - xy[1]) <= radius + grid.EPS)


def _can_vacate(w: World, e: Entity, a: LastSeen) -> bool:
    """Whether the robot's free-space cells can show the thing gone from its place `a`: the robot drives (a flier's
    ground band reads every ground point as floor, so a box seen from the air reads empty) and the thing stood on the
    floor (`World.on_floor`) with a top above the band (`grid.band`: a flatter thing leaves no obstacle point)."""
    return not w.robot.flies and w.on_floor(e.id, a.k) and e.size[2] > grid.band(w.robot)[0]


def _same(w: World, i: int, j: int, e: Entity, a: LastSeen, b: LastSeen | None, seen_empty: int | None,
          state: str) -> Cand:
    """The `same_place` candidate for one thing in tasks `i` < `j`, whose place in `i` is `a` and first place in `j`
    `b`; a vacated one has no `b` and the keyframe `seen_empty` that saw the place empty."""
    ref = ref_of(e)
    evidence, entities = ((a.k, b.k), (e.id, e.id)) if b is not None else ((a.k, seen_empty), ())
    crit = critical(w, i, ref, a) or (b is not None and critical(w, j, ref, b))
    slots = {"kind": ref.kind, "attributes": ref.attributes, "tasks": [i, j]}
    return Cand("same_place", slots, {"type": "yesno", "value": state == YES}, "answerable", evidence, entities, {},
                subjects=(f"e:{e.id}",), critical=crit, answer_class=state)


def same_place(w: World) -> list[Moment]:
    """For each pair of the window's tasks (`i` < `j`) and each kind with both a yes and a no candidate, as many
    moments as the smaller pool has things, keyed "<i>|<j>|<kind>|<draw>", each holding the yes candidates in
    `answerable` and the no ones (moved, vacated) in `unknown`. A place pair between `SAME_M` and the no distance is no
    candidate."""
    by_task, out, cells = _by_task(w), [], {}
    for i, j in combinations(sorted(by_task), 2):
        yes: dict[str, list[Cand]] = defaultdict(list)
        no: dict[str, list[Cand]] = defaultdict(list)
        for eid, e in by_task[i].items():
            if eid not in by_task[j] or w.in_end_window(eid) or (a := last_place(w, e, i)) is None:
                continue
            far = max(APART_M, 2 * w.job.unit_m) + widening(item_size(e.kind))
            if (b := first_place(w, e, j)) is not None:
                d = math.dist(a.xy, b.xy)
                if d < SAME_M - grid.EPS:
                    yes[e.kind].append(_same(w, i, j, e, a, b, None, YES))
                elif d >= far - grid.EPS:
                    no[e.kind].append(_same(w, i, j, e, a, b, None, MOVED))
            elif _can_vacate(w, e, a) and not set(w.sighted(eid)) & set(tks := _task_ks(w, j)):
                if j not in cells:
                    cells[j] = w.latest_cells([k for k in tks if k in w.rows])
                if (k := empty_since(cells[j], a.xy, min(e.size[0], e.size[1]) / 2)) is not None:
                    no[e.kind].append(_same(w, i, j, e, a, None, k, VACATED))
        for kind in sorted(yes.keys() & no.keys()):
            out += [Moment(f"{i}|{j}|{kind}|{d}", yes[kind], no[kind]) for d in range(_draws(yes[kind], no[kind]))]
    return out


# -- person_where ------------------------------------------------------------------------------------------------
class At(NamedTuple):
    """One event moment of the window: the case task, the wording of the job's event, and the script keyframe it
    resolved to."""

    task: int
    wording: str
    k: int


def _where(w: World, family: str, at: At, e: Entity, last: LastSeen, size: tuple[float, float, float]) -> Cand:
    """A position question about one entity at one moment, in the slots of `PERSON_SLOTS` and with its ladder (the job's
    rungs widened for a thing of box `size`): the gold is the thing's place at its last recognisable sighting."""
    ref, widen = ref_of(e), widening(size)
    slots = {"kind": ref.kind, "attributes": ref.attributes, "event": at.wording, "task": at.task if w.multi else None}
    return Cand(family, slots, {"type": "position", "xy": list(last.xy)}, "answerable", (last.k,), (e.id,),
                {"ladder": [round(r + widen, 6) for r in w.job.ladder]},
                subjects=(f"e:{e.id}", f"w:{at.k}"), critical=critical(w, at.task, ref, last))


def _moments(w: World) -> list[At]:
    """The window's event moments, in task then wording order."""
    return sorted(At(task, wording, k) for (task, wording), k in w.events.items())


def movers(w: World) -> list[Entity]:
    """The people `person_where` may ask about: movers whose description is the only one of its kind in the world."""
    return [e for e in w.tl.entities if e.mobility == "mover" and w.unique(e)]


def person_where(w: World) -> list[Moment]:
    """One moment for each (task, wording) the window's events resolve, keyed "<task>|<wording>": the unique movers
    whose last recognisable sighting at the moment lies in its task, is steady up to the moment, and who show in no
    row of the end look window. The unknown pool is empty."""
    people = movers(w)
    out = []
    for at in _moments(w):
        found = []
        for e in people:
            last = answerable_at(w, (at.task, at.k), e)
            if last is not None and not w.in_end_window(e.id) and w.steady(e, last.k, last.xy, at.k):
                found.append(_where(w, "person_where", at, e, last, e.size))
        out.append(Moment(f"{at.task}|{at.wording}", found, []))
    return out


# -- the families ------------------------------------------------------------------------------------------------
def _same_text(slots: dict) -> str:
    i, j = slots["tasks"]
    return (f"Was the {phrase(slots['kind'], slots['attributes'])} in the same place in your {ORDINALS[j]} task as in "
            f"your {ORDINALS[i]}?")


def _was_text(slots: dict) -> str:
    return (f"Where was the {phrase(slots['kind'], slots['attributes'])} during your {ORDINALS[slots['task']]} task? "
            f"Answer in map coordinates.")


def _person_text(slots: dict) -> str:
    return (f"Where was the {phrase(slots['kind'], slots['attributes'])}, as you last saw them, "
            f"{slots['event']}{task_phrase(slots['task'])}?")


PERSON_SLOTS = ("kind", "attributes", "event", "task")  # the anchored form: a person, a moment of the job, its task
NEEDS = ("see", "pose", "status")  # what `person_where` reads: sightings, poses, and the phases events resolve on
SPECS = (
    FamilySpec("object_was", "end", "position", "ladder", ("see", "pose"), "memory", None, False,
               ("kind", "attributes", "task"), _was_text, temporal=True),
    FamilySpec("same_place", "end", "yesno", "exact", ("see", "pose"), "memory", None, False,
               ("kind", "attributes", "tasks"), _same_text, temporal=True),
    FamilySpec("person_where", "end", "position", "ladder", NEEDS, "memory", None, False, PERSON_SLOTS, _person_text,
               temporal=True))
GOLDS = {"object_was": object_was, "same_place": same_place, "person_where": person_where}
SLICED: frozenset[str] = frozenset()
COINS = frozenset({"object_was", "same_place"})
