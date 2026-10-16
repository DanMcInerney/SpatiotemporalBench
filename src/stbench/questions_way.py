"""`way_to` (0.11, docs/families/way_to.md; FD-0.11-design §3.4.2): which way the robot would set off to fetch a thing
it saw before, by the shortest way its body fits through, of the ways it has seen. The help kind's core decision, "the
way to the goal": a person says "bring me my glasses", and the helper must recall where it saw them and which doorway to
take. It is `ego_mem`'s question turned into an action, and it is scored by what it gives up.

**The subjects are `ego_mem`'s** (`questions_ego.ego_mem`): the named things the robot saw before the recent view and
does not see in it, unmoved since, which no look-alike hides and its own log does not tell about; the `unknown` pool is
`ego_mem`'s too (a real thing with no pixel in the window, or an absent description), drawn per point as an end moment's
is (`SLICED`), so a thing's kind predicts no slice. Both names share the slot 2 of the helper's table
(`job_helper.look_shares`), so a point asks one or the other, and a thing is named once a case (the subject rule).

**The gold** (`route_field.route`): the walkable floor (seen free or driven, eroded by half the body) and its Dijkstra
distance to the floor within the re-find radius R (`JobSpec.reacquire_m`) of the thing's middle at its last recognisable
sighting; for each heading at 1 degree, the metres nearer one R of driving along it brings the robot, null where it has
no room to move (a run under T, the body's length plus C) or the floor it ends on has no route. The best heading is the
answer; the row's `profile_m` is every heading's outcome, `step_m` the R it is told over, `resolution_m` the field's
fine cell and `need_m` the job's C (credit `route`). The gold reads the window's cells and poses up to the point and
nothing after it, and the thing's place at its last sighting, never its place now (truth rule 2).

**Dead bands** (nothing is asked): the phase at the point is not one of `JobSpec.look_phases["way_to"]` (where the robot
drives, as `clear_path`), the robot flies, no route exists through floor seen or driven, the robot is within R of the
goal set already (route length), no heading brings it nearer, or the best heading's outcome is more than a drive can
make (`route_field.max_outcome`: the octile field's excess per metre and the snapped ends, about 1.44 per R on the
helper; there the eroded floor and the body's corridor disagree, final review S5): `route_field.route` returns None for
each. A seen thing with no seen route is never an `unknown` gold, so the `unknown` slice stays unpredictable from the
question. Off by default and turned on by `FamilySpec.own_log_screen`: a point where the robot's own log already points
the best way is a dead band too (`heading_screen`, the fallback declared for the own-log shortcut).

`R` and `T` are slots a table system needs (`reach_m`, `room_m`, as `open_heading`'s `cap`); the words do not state
them. The scope is `memory`: the thing is not in the recent view. The evidence is the thing's sightings at its place and
the keyframes that certified the cells of the shortest route from the base. """

from __future__ import annotations

from dataclasses import replace

from . import clear_profile, grid, heading_screen, route_field
from .descriptions import phrase
from .questions_ego import ego_mem
from .questions_select import Cand, Moment
from .questions_world import World
from .records import FamilySpec

FAMILY = "way_to"


def _render(slots: dict) -> str:
    return (f"If you had to go now to the {phrase(slots['kind'], slots['attributes'])} you saw, which way would you "
            f"set off to get there by the shortest way your body fits through, of the ways you have seen? Answer the "
            f"heading in degrees to the left of the way you face now, from -180 to 180 (to your right is negative).")


def _phases(w: World) -> frozenset[str]:
    """The status phases the job asks the family in: its `JobSpec.look_phases[FAMILY]`. ValueError naming the job when
    it declares none (which phases the robot drives in is the job's to say)."""
    declared = w.job.look_phases.get(FAMILY)
    if not declared:
        raise ValueError(f"the job {w.job.id} asks {FAMILY!r} at look points but declares no `look_phases` for it: the "
                         f"status phases the robot drives in")
    return frozenset(declared)


def _slots(w: World, slots: dict) -> dict:
    """The question's slots: the thing's name and the two numbers a table system needs, the re-find radius R and the
    room T a body needs to move (its length plus the job's clearance C, to the micrometre)."""
    return {**slots, "reach_m": w.job.reacquire_m, "room_m": round(w.robot.body.length_m + w.job.clearance_m, 6)}


def way_to(w: World, k: int) -> Moment | None:
    """The look point `k`'s moment: the `ego_mem` subjects that have a route (the module docstring) as answerable
    candidates, and `ego_mem`'s unknown pool beside them; None (a dead band) when the phase at `k` is not a driving one,
    the robot flies, or no subject has a route. `k` is any frame of the window."""
    if w.robot.flies or w.rows[k]["status"]["phase"] not in _phases(w) or (found := ego_mem(w, k)) is None:
        return None
    states = w.latest_cells([j for j in range(w.rec.k0, k + 1) if j in w.rows])
    runs = route_field.runs_of(w.rows[j] for j in sorted(w.rows) if j <= k)
    base, yaw, reach = w.base(k), w.yaw(k), w.job.reacquire_m
    answerable = []
    for c in found.answerable:
        slots = _slots(w, c.slots)
        thing = w.xy_at(c.entities[0], max(c.evidence_k))   # the middle of the thing at its last recognisable sighting
        way = route_field.route(states, runs, w.robot, base, yaw, thing, slots["reach_m"], slots["reach_m"],
                                slots["room_m"])
        if way is None:
            continue
        gold = {"type": "heading", "deg_left": float(clear_profile.deg_of(way.best, route_field.STEP_DEG))}
        tolerances = {"profile_m": way.profile, "step_deg": route_field.STEP_DEG, "step_m": slots["reach_m"],
                      "resolution_m": grid.FINE_M, "need_m": w.job.clearance_m}
        cand = Cand(FAMILY, slots, gold, "answerable", tuple(sorted({*c.evidence_k, *way.ks})), (), tolerances,
                    decision_k=k, subjects=c.subjects, answer_class="answerable")
        if not (SPECS[0].own_log_screen and heading_screen.told_by_log(w, cand, k, SPECS[0].credit)):
            answerable.append(cand)
    if not answerable:
        return None
    floor = {"resolution_m": grid.FINE_M, "need_m": w.job.clearance_m, "step_deg": route_field.STEP_DEG,
             "step_m": reach}
    unknown = [replace(u, family=FAMILY, slots=_slots(w, u.slots), tolerances=floor) for u in found.unknown]
    return Moment(str(k), answerable, unknown)


SPECS = (FamilySpec(FAMILY, "during", "heading", "route", ("see", "pose"), "memory", None, False,
                    ("kind", "attributes", "reach_m", "room_m"), _render),)
GOLDS = {FAMILY: way_to}
LOOKS = frozenset({FAMILY})
SLICED = frozenset({FAMILY})
