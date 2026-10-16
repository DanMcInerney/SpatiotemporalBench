"""`closest_approach` (look, scope by evidence, `number`, credit `graded` on C): how near the person nearest the robot
would come to it in the next few seconds if both kept moving as they are (0.10.0, docs/families/closest_approach.md). A
family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` and `LOOKS`; asked of the AMR and the helper, whose
planners decide to slow or yield from exactly this; not of the drone, for which a floor distance from 15 m up is not the
decision (review B1). Its out-of-view twin, `approach_mem` (0.11.0), is the module's second family (the last section of
this docstring).

**The person** is the one evidence sighting (`person_seen.hull`, `closest_person`'s rule) of frame `k`, "the person you
can see nearest you" in the question's words (the gold is the evidence person in frame `k`; one the frame shows and the
rule does not place is a dead band, below). The sensors give no identity, so the motion of a person, which pairs two
frames' sightings, is estimated only when the pairing is unambiguous (`approach_rule.pair_velocity`): exactly one
evidence person in frame `k` and in the earlier frame used (the latest frame of the task within `VEL_S` of `k` that
holds an evidence sighting of anyone), that person in both, and no
person row of `NO_ONE_PX` pixels or more that is not an evidence sighting in either frame. Otherwise the point is a
dead band: a reader cannot tell a walker from two people standing, and a gold that paired them by the truth's identity
would claim knowledge the sensors do not give (the `knowable` gate caught it, +0.119). The point also asks nothing when
a row of a person who is not an evidence sighting lies nearer than the gold's person by more than C/2 (the truth's
footprint decides, as in `closest_person`): the frame shows someone nearer than the gold would say.

**The gold** is `approach_rule.closest_m`: the robot at its own velocity, the one every delivered step carries
(`case.base_velocity`, the controller's odometry: no frame before `k` is needed to read it), the person at the velocity
of their hull's floor centroid between the earlier frame used and `k` (zero when there is no such frame or under
`V_STILL`), both for `HORIZON_S`, the least floor distance from the camera's floor point to the hull sampled every
`STEP_S`, at most `CAP_M`. Clocks are the delivered ones, to the tenth of a second. A gold at the cap is the class
`no_one` (kept by the balance draw); the others are `person`. With no evidence person in frame `k` the gold is the cap
too, asked only when no person row of `NO_ONE_PX` pixels or more lies within the cap (the truth's footprint) and then
kept by the same draw. A task's first frame is a dead band (`TASK_START`): its delivered velocity is 0 because the
controller has nothing to difference there, not because the robot is known to be still, so the gold would claim a
speed nobody measured. (A look point is at least four frames into its task, so none reaches it.)

**The scope.** A row whose person is still, or has no earlier sighting within `VEL_S` to move against, is `now`: frame
`k` and the step's own velocity give the gold, which a reader holding the latest frame has (an evidence person who
stands where the frame shows them, taken as still, as a reader with no earlier frame must). Its evidence is frame `k`
and its age 0. A row whose person moves (their speed is over `V_STILL`) is `memory`: the gold needs their earlier
sighting, so its evidence is that frame and `k`. A "no one" with no evidence person in frame `k` stays `now`: its gold
is the cap whatever the robot's velocity, which frame `k` alone shows (no one within the cap), and its evidence is
frame `k`. The evidence age (`questions_age`) follows the evidence: `now` and 0 for the `now` rows, `recent` for a
`memory` row at the AMR's pace of 1-5 s a frame. The dev data exercise the `memory` half on 2 rows only (two AMR
walkers; the dev people mostly stand), so a score on this family is mostly about a person who stands.

**The credit** is `graded` on C (`rho`: one pixel at the nearest hull point of frame `k`), the answer longer than the
gold being the harmful side with the threshold the body's length plus C, and the catastrophe tolerance `over`
(`questions_harm.over`). The subject is the recent view (`questions_harm.stretch`): the same person is not asked at two
points within a look window.

**`approach_mem`** (0.11.0, FD-0.11-hard section 3.2, docs/families/approach_mem.md) is `closest_approach` for the
person now out of view, the AMR's blind aisle end: "If you kept moving as you are now, how close would you come in the
next 5 seconds to the person you saw last, where you last saw them?" It needs the memory of a dynamic thing that is in
no layout, composed with the robot's own motion. At look point `k` of task tau, the person is the one of the latest
evidence sighting before `k` (`approach_rule.last_seen`, identity-free), taken to stand where they were seen, and the
gold is `approach_rule.closest_m` from the camera at `k` at the delivered velocity to that hull, at most `CAP_M`. A
point is a dead band (the names below) when the sensors do not say who was last seen or whether they stand: `IN_VIEW`
(frame `k` holds a person row of `NO_ONE_PX` pixels or more: `closest_approach`'s point), `NONE_SEEN` (no evidence
sighting earlier in the task), `TWO_PEOPLE` (the latest holds two) and `UNPLACED` (it holds a row of 256 px or more
that is no evidence) or `UNPLACED_SINCE` (a person shown and not placed after it), `WALKING` (the pairing of the
sighting with its own earlier frame is ambiguous or the person moves: a reader who saw them move cannot say where they
are) and, by the truth, `MOVED_UNSEEN` (the person's footprint centre at `k` is more than C/2 from its centre at the
sighting: the gold is then the world now as well as the last sighting). The scope is `memory` (evidence the sighting
and `k`, the age that of the sighting), the credit `closest_approach`'s, and the subject `a:<j>` for the sighting, so a
case asks one sighting once (not the recent view, which would hold `closest_approach` and this family apart). The
"no one" gold (the person would not come within the cap) is kept by `APPROACH_MEM_NO_ONE_KEEP_PERCENT`.
"""

from __future__ import annotations

import math

from . import approach_rule as rule
from .case import base_velocity
from .nearest_part import nearest_on_hull
from .questions_harm import kept, over, stretch
from .questions_people import NO_ONE_PX, person_hull, person_ids, person_resolution, person_true_m
from .questions_select import Cand, Moment
from .questions_world import World
from .records import FamilySpec
from .resolution_at import at_frame
from .rules import graded_tolerances

EPS = 1e-9
# The share of the "no one" golds (gold at the cap) that the balance draw keeps, set from the measured supply so that no
# job's rows hold more than a third of them (the balance rule): the floor of 100 (1 - p) / (2 p) for the job whose share
# p of "no one" is largest. On the dev root (36 AMR and 36 helper cases) p is 0.903 (keep 5.4) for the AMR and 0.953
# (keep 2.4) for the helper, whose people are in view in 4.5% of the frames: 2 for both jobs, the smaller.
NO_ONE_KEEP_PERCENT = 2
# The same for `approach_mem`, declared 100: the proxy's share of golds at the cap is 22% on the AMR and 0% on the
# helper, both under a third (FD-0.11-hard 3.2). If the compile measures a share p above 1/3 on a job, it becomes the
# floor of 100 (1 - p) / (2 p) for the job with the largest p, as above, and the root is compiled again.
APPROACH_MEM_NO_ONE_KEEP_PERCENT = 100


def _render(slots: dict) -> str:
    return (f"If you kept moving as you are now, how close would the person you can see nearest you come to you in the "
            f"next {rule.HORIZON_S:g} seconds? Answer the smallest distance in metres along the floor from the point "
            f"below your camera to the person; answer {rule.CAP_M:g} if no one would come within {rule.CAP_M:g} m.")


def _render_mem(slots: dict) -> str:
    return (f"If you kept moving as you are now, how close would you come in the next {rule.HORIZON_S:g} seconds to "
            f"the person you saw last, where you last saw them? Answer the smallest distance in metres along the "
            f"floor from the point below your camera to that person; answer {rule.CAP_M:g} if you would not come "
            f"within {rule.CAP_M:g} m.")


def _earlier(w: World, people: list[str], k: int) -> tuple[int, dict[str, list[list[float]]], float] | None:
    """The latest earlier frame of the task within `VEL_S` of frame `k` that holds an evidence sighting of anyone, as
    (its frame, the hull of each evidence person in it by id, its age in seconds) (`approach_rule.latest_within`); None
    when there is none. The frame is chosen without identity, as a reader who is given none would."""
    now, held, ages = w.clock(k), {}, []
    for j in range(k - 1, w.tl.task_of(k).k0 - 1, -1):
        if j not in w.rows:
            continue
        age = rule.elapsed_s(w.clock(j), now)
        if age > rule.VEL_S:
            break
        if hulls := {pid: hull for pid in people if (hull := person_hull(w, pid, j)) is not None}:
            held[j] = hulls
            ages.append((j, age))
    j = rule.latest_within(ages)
    return None if j is None else (j, held[j], dict(ages)[j])


def _unplaced(w: World, people: list[str], k: int) -> bool:
    """Whether frame `k` holds a person row of `NO_ONE_PX` pixels or more that is no evidence sighting: someone the
    frame shows and does not place."""
    return any(w.vis.get(k, {}).get(pid, {}).get("px", 0) >= NO_ONE_PX and person_hull(w, pid, k) is None
               for pid in people)


TASK_START = "no frame before the point in its task"
TWO_PEOPLE = "more than one evidence person in a frame"
UNPLACED = "a person row of 256 px or more that is no evidence, in a frame"
OTHER_EARLIER = "the earlier frame used holds someone else"
NEARER_UNSEEN = "a nearer person who is not evidence"
NO_ONE_RANGE = "a person row in the range of a no one"
NO_ONE_DRAWN = "the draw dropped a no one"
IN_VIEW = "a person row of 256 px or more in the ask's frame"
NONE_SEEN = "no evidence sighting earlier in the task"
UNPLACED_SINCE = "a person row of 256 px or more since the last sighting"
WALKING = "the last sighting is of someone who moves, or cannot be told to stand"
MOVED_UNSEEN = "the person moved out of view"
DEAD_BANDS = (TASK_START, TWO_PEOPLE, UNPLACED, OTHER_EARLIER, NEARER_UNSEEN, NO_ONE_RANGE, NO_ONE_DRAWN, IN_VIEW,
              NONE_SEEN, UNPLACED_SINCE, WALKING, MOVED_UNSEEN)


def _row(w: World, family: str, k: int, gold: float, rho: float, scope: str, evidence: tuple[int, ...],
         subjects: tuple[str, ...], keep_percent: int) -> Cand | str:
    """The candidate of a gold distance `gold` at look point `k`, or `NO_ONE_DRAWN` when the draw drops a "no one": the
    credit both families share (`graded` on C, the harm side `over`, the body's length plus C as the threshold, the
    catastrophe `over`), the class `no_one` at the cap and `person` below it."""
    clearance = w.job.clearance_m
    answer_class = "no_one" if gold >= rule.CAP_M - EPS else "person"
    if answer_class == "no_one" and not kept(w, family, k, keep_percent):
        return NO_ONE_DRAWN
    threshold = w.robot.body.length_m + clearance
    tol = graded_tolerances(rho, clearance, "over", threshold) | {"catastrophe": over(clearance, rho, threshold)}
    return Cand(family, {}, {"type": "number", "value": round(gold, 4)}, "answerable", evidence, (), tol, decision_k=k,
                subjects=subjects, answer_class=answer_class, scope=scope)


def candidate(w: World, k: int) -> Cand | str:
    """The look point `k`'s candidate, or the reason it is a dead band (`DEAD_BANDS`). ValueError for a keyframe outside
    the window."""
    if k not in w.rows:
        raise ValueError(f"keyframe {k} lies outside the window [{w.rec.k0}, {w.rec.k1})")
    if k - 1 not in w.rows or k - 1 < w.tl.task_of(k).k0:
        return TASK_START
    cam, clearance, people = w.camera_at(k)[0][:2], w.job.clearance_m, person_ids(w)
    base_v = tuple(base_velocity(w.tl, w.rows, k))   # the step's own velocity: the delivered one, no frame before it
    rows = {pid: d for pid in people if pid in w.vis.get(k, {}) and (d := person_true_m(w, pid, k, cam)) is not None}
    sighted = {pid: hull for pid in people if (hull := person_hull(w, pid, k)) is not None}
    if not sighted:   # frame k shows no one the rule places: the cap, whatever the robot's velocity (k answers it)
        if any(w.vis[k][pid]["px"] >= NO_ONE_PX and d <= rule.CAP_M for pid, d in rows.items()):
            return NO_ONE_RANGE
        gold, rho, scope, evidence = rule.CAP_M, at_frame(w, k, (cam[0] + rule.CAP_M, cam[1]), None), "now", (k,)
    else:
        earlier = _earlier(w, people, k)
        j, before, age = (None, None, 0.0) if earlier is None else earlier
        walking = rule.pair_velocity(list(sighted.values()), None if before is None else list(before.values()), age)
        if walking is None:
            return TWO_PEOPLE
        if _unplaced(w, people, k) or (j is not None and _unplaced(w, people, j)):
            return UNPLACED
        ((pid, hull),) = sighted.items()
        if before is not None and pid not in before:
            return OTHER_EARLIER   # the lone hull of the earlier frame is another person's: the pairing would be wrong
        point = nearest_on_hull(hull, cam)
        dist = round(math.dist(point, cam), 9)
        if any(dist - d > clearance / 2 + EPS for other, d in rows.items() if person_hull(w, other, k) is None):
            return NEARER_UNSEEN   # the rule does not count them: the gold would be farther than what was seen
        gold, rho = rule.closest_m(cam, base_v, hull, walking), person_resolution(w, pid, k, point)
        # the robot's velocity is delivered with the step; a person who moves needs their earlier sighting (memory of
        # frame j), one who is still is read from frame k alone
        scope, evidence = ("memory", (j, k)) if any(walking) else ("now", (k,))
    return _row(w, "closest_approach", k, gold, rho, scope, evidence, stretch(w, k), NO_ONE_KEEP_PERCENT)


def closest_approach(w: World, k: int) -> Moment | None:
    """The look point `k`'s moment: its one candidate, or None for a dead band (`candidate`)."""
    found = candidate(w, k)
    return None if isinstance(found, str) else Moment(str(k), [found], [])


def _shown(w: World, people: list[str], k: int) -> bool:
    """Whether frame `k` holds a person row of `NO_ONE_PX` pixels or more, evidence or not."""
    return any(w.vis.get(k, {}).get(pid, {}).get("px", 0) >= NO_ONE_PX for pid in people)


def _sightings(w: World, people: list[str], k: int) -> dict[int, dict[str, list[list[float]]]]:
    """The evidence hull of each person, by id, in each frame of `k`'s task before `k` that holds one."""
    found = {}
    for j in range(w.tl.task_of(k).k0, k):
        if j in w.rows and (hulls := {pid: hull for pid in people if (hull := person_hull(w, pid, j)) is not None}):
            found[j] = hulls
    return found


def mem_candidate(w: World, k: int) -> Cand | str:
    """`approach_mem`'s candidate at look point `k`, or the reason it is a dead band (`DEAD_BANDS`): the module
    docstring's rule, step by step in this order. ValueError for a keyframe outside the window."""
    if k not in w.rows:
        raise ValueError(f"keyframe {k} lies outside the window [{w.rec.k0}, {w.rec.k1})")
    if k - 1 not in w.rows or k - 1 < w.tl.task_of(k).k0:
        return TASK_START
    people = person_ids(w)
    if _shown(w, people, k):
        return IN_VIEW   # someone is in view: `closest_approach`'s point
    seen = _sightings(w, people, k)
    if not seen:
        return NONE_SEEN
    found = rule.last_seen({j: list(hulls.values()) for j, hulls in seen.items()}, k, seen)
    if found is None:
        return TWO_PEOPLE   # the latest sighting holds two: the sensors do not say which was seen last
    j, hull = found
    if _unplaced(w, people, j):
        return UNPLACED
    if any(_shown(w, people, m) for m in range(j + 1, k)):
        return UNPLACED_SINCE   # someone shown and not placed since: the last person seen is not the sighting's
    ((pid, _),) = seen[j].items()
    earlier = _earlier(w, people, j)
    before = None if earlier is None else list(earlier[1].values())
    if rule.pair_velocity([hull], before, 0.0 if earlier is None else earlier[2]) != (0.0, 0.0):
        return WALKING   # ambiguous (None) or moving: a reader who saw them move cannot say where they are
    then, now = (w.tl.footprint(pid, w.tl.keyframe(m).t) for m in (j, k))
    if then is None or now is None or math.dist((then.cx, then.cy), (now.cx, now.cy)) > w.job.clearance_m / 2 + EPS:
        return MOVED_UNSEEN   # the truth screen: the gold is then the last sighting and the world now
    cam = w.camera_at(k)[0][:2]
    gold = rule.closest_m(cam, tuple(base_velocity(w.tl, w.rows, k)), hull, (0.0, 0.0))
    rho = person_resolution(w, pid, j, nearest_on_hull(hull, w.camera_at(j)[0][:2]))
    return _row(w, "approach_mem", k, gold, rho, "memory", (j, k), (f"a:{j}",), APPROACH_MEM_NO_ONE_KEEP_PERCENT)


def approach_mem(w: World, k: int) -> Moment | None:
    """The look point `k`'s moment: its one candidate, or None for a dead band (`mem_candidate`)."""
    found = mem_candidate(w, k)
    return None if isinstance(found, str) else Moment(str(k), [found], [])


SPECS = (FamilySpec("closest_approach", "during", "number", "graded", ("see", "pose"), "evidence", None, False, (),
                    _render),
         FamilySpec("approach_mem", "during", "number", "graded", ("see", "pose"), "memory", None, False, (),
                    _render_mem))
GOLDS = {"closest_approach": closest_approach, "approach_mem": approach_mem}
LOOKS = frozenset({"closest_approach", "approach_mem"})
SLICED: frozenset[str] = frozenset()
