"""`closest_person` and `person_mem`: where people were, and how near one came (0.8.0, docs/families/closest_person.md
and person_mem.md). A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS`, `LOOKS` (`person_mem`) and no
`SLICED`.

**What counts as seeing a person** is `person_seen`'s rule, one function that this gold and the oracle channel's
`floor_hull` of a person both call: at least 1,024 px and `RECOG_LUM`, a depth at every pixel, a box off the image's
bottom edge (the top and the sides may touch), and the contact tie at the job's clearance C. A sighting that passes is
an **evidence sighting**; its **hull** is the floor hull of the person's pixels (`nearest_part.floor_hull`). People are
never named, so no identity is needed: two people are told apart by nothing, and nothing here asks to.

**The credit** (`graded`, 0.9.0; `rules.graded_tolerances`) is on the job's clearance C as the need, with the
resolution `rho` (`resolution_at`: one pixel at the gold's point from the camera of the sighting that gives it). A
`closest_person` answered wider than the gold is harmful (`harm` `over`) and carries `threshold_m`, the body's length
plus C: is there room to move at all. A `person_mem` placed farther from the camera than the person is, is harmful
(`harm` `radial`). An `unknown` row carries `resolution_at.UNKNOWN_RHO_M`.

**`closest_person`** (end, `memory`, `number`, credit `graded` on C): "During your first task, how
close did a person come to you, as you saw them?" The answer is a distance along the floor, in metres, capped at
`CAP_M` = 10 ("no one"). One question a case, about the task that comes first in `stable_hash(case id, "closest_person",
str(task))` order among the window's tasks that pass:
- **a person**: the least, over the task's keyframes and its evidence sightings, of the floor distance from that
  keyframe's camera floor point (the delivered pose row's camera) to the nearest point of the sighting's hull
  (`nearest_part.nearest_on_hull`), each sighting measured from its own frame's camera. The gold is below the cap. It is
  dropped when a row of a person that is not an evidence sighting in the task lies nearer than the gold by more than
  C/2: the truth's footprint (`Timeline.footprint`, from the same camera floor point) decides, because the frames show a
  nearer person that the rule would not count, and the gold would then be farther than what the robot saw;
- **no one** (gold `CAP_M`): only when no evidence sighting is nearer than the cap and no person row of 256 px or more
  lies within the cap (the truth's footprint again); then kept for half the cases, by `stable_hash(case id,
  "closest_person", "empty") % 100 < 50`, so that the answer 10 is not the common one.
The class is `person` or `no_one` (`records.BALANCE_POOLS`). Evidence is the keyframes of the evidence sightings (a "no
one" cites the task's last keyframe), the subject the task's keyframes (`w:<first>-<last>`), so the subject rule keeps
two end questions about one stretch apart. Everything the gold reads lies in the task, before the end ask; the truth
screens only remove a task. A table that keeps the person hulls the channel sends (`table_people`) reproduces it.

**`person_mem`** (look, `memory`, `offset`, credit C's rungs, asked only of a job with `JobSpec.one_person`): "As you
last saw them, where is the point of the person you saw that is nearest to your camera now?" At a look point `k` (any
frame) the person is the world's one (a helper world with two people fails its checks, and this raises for one):
- **answerable**: the window's latest evidence sighting of the person up to `k` lies before the recent view
  (`look_windows.recent`: the last `LOOK_WINDOW` frames of the current task up to `k`, which slides with the point), and
  no row of the person, at any pixel count, falls after it up to `k` (nothing seen puts the person elsewhere, and the
  current view would not answer). The gold is that sighting's hull alone (people shift on the spot, so a union over
  sightings would blur), its point nearest the camera's floor point at `k`, in the camera's frame at `k`
  (`ego_frame.to_offset`); the range is the slant from that camera to the visible point nearest it. "As you last saw
  them", whatever has happened since (truth rule 2);
- **unknown**: the window up to `k` holds no row of any person at all, at any pixel count (the person may never have
  been home). A person who was in the window but never as an evidence sighting asks nothing: the robot saw them, and
  the rule only says it cannot place them to C/2, so `unknown` would be wrong for a reader who answers an offset;
- otherwise nothing is asked at the point.
The family is not sliced: a point holds an answerable candidate or an unknown one, never both, so the unknown ones are
thinned by the job's declared keep share (`JobSpec.look_unknown_keep`), by the class alone. The answerable candidate's
subject is the sighting it reads, `w:<j>` for its keyframe `j` (the one person is implicit), so a case asks a sighting
once however many look points follow it; the unknown one names none.
"""

from __future__ import annotations

import math
from weakref import WeakKeyDictionary

from . import person_seen
from .ego_frame import slant_m, to_offset
from .ego_support import contact_bases
from .geometry import corners, dist_to_polygon
from .nearest_part import nearest_on_hull
from .person_seen import PERSON_KIND
from .questions_select import Cand, Moment
from .questions_words import ORDINALS
from .questions_world import World
from .records import FamilySpec
from .resolution_at import UNKNOWN_RHO_M, at_frame
from .rules import graded_tolerances
from .seeding import stable_hash

CAP_M = 10.0              # the farthest a person's distance is asked: "answer 10 if you saw no one within 10 m"
NO_ONE_PX = 256           # a person row of this many pixels (or more) within the cap keeps a "no one" from being asked
NO_ONE_KEEP_PERCENT = 50  # of the "no one" golds that pass the screens, the share kept
EPS = 1e-9

# The hull of a person in a keyframe, once a World however many look points and tasks read it.
_HULLS: WeakKeyDictionary[World, dict[tuple[int, str], list[list[float]] | None]] = WeakKeyDictionary()


def _render_closest(slots: dict) -> str:
    return (f"During your {ORDINALS[slots['task']]} task, how close did a person come to you, as you saw them? Answer "
            f"the smallest distance in metres along the floor from the point below your camera to the nearest point of "
            f"a person; answer {CAP_M:g} if you saw no one within {CAP_M:g} m.")


def _render_mem(slots: dict) -> str:
    return ("As you last saw them, where is the point of the person you saw that is nearest to your camera now, "
            "measured along the floor? Answer in metres ahead of the camera and to its left.")


# -- evidence ---------------------------------------------------------------------------------------------------
def person_ids(w: World) -> list[str]:
    """The ids of the world's people."""
    return [e.id for e in w.tl.entities if e.kind == PERSON_KIND]


def person_hull(w: World, pid: str, k: int) -> list[list[float]] | None:
    """The hull of the person's sighting in keyframe `k` of the window when it is an evidence sighting
    (`person_seen.hull`), else None. A person with no visibility row there has none, and no picture is read."""
    if pid not in w.vis.get(k, {}):
        return None
    cache = _HULLS.setdefault(w, {})
    if (k, pid) not in cache:
        cache[k, pid] = person_seen.hull(person_seen.frame_of(w, pid, k), pid, w.job.clearance_m)
    return cache[k, pid]


def person_resolution(w: World, pid: str, k: int, xy: tuple[float, float]) -> float:
    """The resolution of the person's point `xy` from the camera of keyframe `k`: the plane is the person's base there
    (`ego_support.contact_bases`), the floor when it names none."""
    return at_frame(w, k, xy, contact_bases(w, pid, [k], xy)[0])


def _tolerances(w: World, rho: float, harm: str, threshold: float | None = None) -> dict:
    """The `tolerances` of a row: the resolution `rho` and the job's clearance C as the need (`graded_tolerances`)."""
    return graded_tolerances(rho, w.job.clearance_m, harm, threshold=threshold)


# -- closest_person ---------------------------------------------------------------------------------------------
def person_true_m(w: World, pid: str, k: int, cam_xy: tuple[float, float]) -> float | None:
    """The floor distance from the camera's floor point to the person's true footprint at keyframe `k`; None when the
    person does not exist then."""
    rect = w.tl.footprint(pid, w.tl.keyframe(k).t)
    return None if rect is None else dist_to_polygon(cam_xy[0], cam_xy[1], corners(rect))


def _task_cand(w: World, people: list[str], task: int, k0: int, k1: int) -> Cand | None:
    """The `closest_person` candidate about task `task` (keyframes `k0` to `k1 - 1`), or None when it is not one (the
    module docstring)."""
    ks = [k for k in range(k0, k1) if k in w.rows]
    cams = {k: w.camera_at(k)[0][:2] for k in ks}
    seen: dict[tuple[int, str], float] = {}
    points: dict[tuple[int, str], tuple[float, float]] = {}   # the hull point nearest each evidence sighting's camera
    for k in ks:
        for pid in people:
            if (hull := person_hull(w, pid, k)) is not None:
                points[k, pid] = nearest_on_hull(hull, cams[k])
                seen[k, pid] = math.dist(points[k, pid], cams[k])
    gold = min(seen.values(), default=CAP_M)
    rows = [(k, pid, d) for k in ks for pid in people if pid in w.vis.get(k, {})
            if (d := person_true_m(w, pid, k, cams[k])) is not None]   # every row of a person, at any pixel count
    if gold < CAP_M:
        if any(gold - d > w.job.clearance_m / 2 + EPS for k, pid, d in rows if (k, pid) not in seen):
            return None   # a nearer person the rule does not count: the gold would be farther than what was seen
        evidence, answer_class = tuple(sorted({k for k, _ in seen})), "person"
        nearest = min(seen, key=seen.__getitem__)   # the sighting that gives the gold, the earliest of equals
        rho = person_resolution(w, nearest[1], nearest[0], points[nearest])
    else:
        if any(w.vis[k][pid]["px"] >= NO_ONE_PX and d <= CAP_M for k, pid, d in rows):
            return None
        if stable_hash(w.rec.case, "closest_person", "empty") % 100 >= NO_ONE_KEEP_PERCENT:
            return None
        gold, evidence, answer_class = CAP_M, (ks[-1],), "no_one"
        far = (cams[ks[-1]][0] + CAP_M, cams[ks[-1]][1])   # no sighting: a floor point at the cap from the last camera
        rho = at_frame(w, ks[-1], far, None)
    tol = _tolerances(w, rho, "over", w.robot.body.length_m + w.job.clearance_m)
    return Cand("closest_person", {"task": task}, {"type": "number", "value": round(gold, 4)}, "answerable", evidence,
                (), tol, subjects=(f"w:{k0}-{ks[-1]}",), answer_class=answer_class)


def closest_person(w: World) -> list[Moment]:
    """At most one moment: the window's tasks in `stable_hash(case id, "closest_person", str(task))` order, the first
    whose candidate passes, keyed by its case index; empty when none does."""
    people = person_ids(w)
    for task, k0, k1 in sorted(w.tasks, key=lambda t: (stable_hash(w.rec.case, "closest_person", str(t[0])), t[0])):
        if any(k in w.rows for k in range(k0, k1)) and (cand := _task_cand(w, people, task, k0, k1)) is not None:
            return [Moment(str(task), [cand], [])]
    return []


# -- person_mem -------------------------------------------------------------------------------------------------
def person_mem(w: World, k: int) -> Moment | None:
    """The look point `k`: the one candidate of the module docstring, in the moment's `answerable` pool or, when
    unknown, its `unknown` one (the family is not sliced: a point holds one candidate, and the draw asks either). None
    for a job that does not declare one person (`JobSpec.one_person`) or when nothing is askable at the point;
    ValueError for a one-person job whose script holds more than one."""
    if not w.job.one_person:
        return None
    people = person_ids(w)
    if len(people) > 1:
        raise ValueError(f"the job {w.job.id} declares one person, but the script of case {w.rec.case} holds "
                         f"{len(people)}: 'the person' names none of them")
    look = w.recent(k)
    window = [j for j in sorted(w.rows) if j <= k]
    if not any(pid in w.vis.get(j, {}) for j in window for pid in people):   # no row of anyone, at any pixel count
        return Moment(str(k), [], [Cand("person_mem", {}, {"type": "unknown"}, "unknown", (), (),
                                        _tolerances(w, UNKNOWN_RHO_M, "radial"), decision_k=k,
                                        answer_class="unknown")])
    seen = [(j, pid) for j in window for pid in people if person_hull(w, pid, j) is not None]
    if not seen:   # the person was seen, but never as evidence: neither a place to ask nor `unknown`
        return None
    j, pid = seen[-1]
    if j >= look.start or any(pid in w.vis.get(m, {}) for m in range(j + 1, k + 1)):
        return None
    cam, yaw = w.camera_at(k)
    part = w.nearest_part(pid, [j], cam[:2])
    ahead, left = to_offset(part.xy, cam[:2], yaw)
    gold = {"type": "offset", "ahead_m": round(ahead, 4), "left_m": round(left, 4)}
    range_m = round(slant_m(cam, (*part.xy, part.z)), 4)
    tol = _tolerances(w, person_resolution(w, pid, j, part.xy), "radial")
    return Moment(str(k), [Cand("person_mem", {}, gold, "answerable", (j,), (), tol, decision_k=k,
                                subjects=(f"w:{j}",), answer_class="answerable", range_m=range_m)], [])


SPECS = (FamilySpec("closest_person", "end", "number", "graded", ("see", "pose"), "memory", None, False, ("task",),
                    _render_closest),
         FamilySpec("person_mem", "during", "offset", "graded", ("see", "pose"), "memory", None, False, (),
                    _render_mem))
GOLDS = {"closest_person": closest_person, "person_mem": person_mem}
LOOKS = frozenset({"person_mem"})
SLICED: frozenset[str] = frozenset()
