"""What the harm families of 0.10.0 share (`closest_approach`, `drop_zone_people`, `tangle_ahead`; FD-questions-0.10
section 2): the subject a question names, the draw that keeps a share of the empty golds, and the pieces of a row.

**The subject** is the recent view (`look_windows.recent`): the last `LOOK_WINDOW` frames of the task up to the point,
as `w:<first>-<k>`. Two questions of one family at points fewer than `LOOK_WINDOW` frames apart name overlapping
stretches, so the subject rule (`questionnaire.conflicts`) lets only the first be asked: the same person or cable is not
asked about twice within a look window. No person or thing is named, because each family's gold is about whatever the
frame holds.

**The empty golds** (`closest_approach`'s "no one", the lists' empty ones) are kept by the balance rule: a point's empty
gold is asked when `stable_hash(case id, family, point, "empty") % 100` is under the family's keep percent, so the
answer 10 or the empty list is never the common one. The draw reads the class and nothing else, as `closest_person`'s
and `task_things`' do.

**The catastrophe tolerance** is the section 4.4 shape (`records.TOLERANCES`): `over` with the body's length plus C for
a distance, `miss` with the critical places for a list. The weight is `credit_kernel`'s: a person, or a hazard of the
robot's own kind, is `CATASTROPHE_PERSON`. The margin is `max(C, 2 rho)`: an error smaller than the job's clearance, or
than twice the resolution of the point that sets it, is no catastrophe.

**A list's row** (`list_row`): the `positions` gold with its items sorted and its don't-care circles, and the
`graded_set` tolerances (`match_m`, one resolution per item in the gold's order, `credit_kernel.FP_WEIGHT`) with the
`miss` catastrophe.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import look_windows
from .credit_kernel import CATASTROPHE_PERSON, FP_WEIGHT
from .questions_world import World
from .seeding import stable_hash

Xy = tuple[float, float]


def stretch(w: World, k: int) -> tuple[str, ...]:
    """The subject of a question at look point `k`: the recent view (the last `LOOK_WINDOW` frames of the point's task
    up to `k`), `w:<first>-<k>` in script numbering."""
    recent = look_windows.recent(w.tl.task_of(k).k0, k)
    return (f"w:{recent.start}-{recent.stop - 1}",)


def kept(w: World, family: str, k: int, percent: int) -> bool:
    """Whether the empty gold of `family` at point `k` is asked: `stable_hash(case id, family, point, "empty") % 100` is
    under `percent`."""
    return stable_hash(w.rec.case, family, str(k), "empty") % 100 < percent


def margin_m(clearance_m: float, rho: float) -> float:
    """`max(C, 2 rho)`, to the micrometre: the harmful error a catastrophe needs to exceed."""
    return round(max(clearance_m, 2 * rho), 6)


def over(clearance_m: float, rho: float, threshold_m: float) -> dict:
    """The catastrophe tolerance of a distance asked too long: the gold under `threshold_m` (the body's length plus C)
    and the answer at or over it, by more than the margin."""
    return {"weight": CATASTROPHE_PERSON, "rule": "over", "margin_m": margin_m(clearance_m, rho),
            "threshold_m": round(threshold_m, 6)}


def list_row(items: Iterable[tuple[Xy, float]], care: Iterable[tuple[Xy, float]], match_m: float, clearance_m: float,
             critical: Iterable[Xy]) -> tuple[dict, dict]:
    """(gold, tolerances) of a list question. `items` are the gold's places with their resolutions `rho`, `care` the
    don't-care circles as (place, radius), `match_m` the matching radius, `critical` the places whose miss is a
    catastrophe (`miss`). The gold's items are sorted by place, and the tolerances' resolutions follow that order."""
    ordered = sorted(items)
    gold: dict = {"type": "positions", "items": [{"xy": [round(x, 4), round(y, 4)]} for (x, y), _ in ordered]}
    if circles := sorted(care):
        gold["dont_care"] = [{"xy": [round(x, 4), round(y, 4)], "radius": round(r, 6)} for (x, y), r in circles]
    rhos = [rho for _, rho in ordered]
    catastrophe = {"weight": CATASTROPHE_PERSON, "rule": "miss",
                   "margin_m": margin_m(clearance_m, max(rhos, default=0.0)),
                   "critical": [[round(x, 4), round(y, 4)] for x, y in sorted(critical)]}
    tolerances = {"match_m": round(match_m, 6), "resolutions_m": [round(r, 6) for r in rhos], "fp_weight": FP_WEIGHT,
                  "catastrophe": catastrophe}
    return gold, tolerances
