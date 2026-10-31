"""The graded credits (FD-decision-questions §3): `graded`, `graded_set`, `regret` and `route`.

Every credit is the four quarters of `credit_kernel` on an error `e` against the gold row's resolution `rho`
(`tolerances.resolution_m`, what one pixel of the camera resolves at the gold) and the job's need `N`
(`tolerances.need_m`: the clearance C, the re-find radius R or a list's match radius M), widened to a tenth of the
distance the answer is judged over, `g` (0.11.0: `credit_kernel.KAPPA`; an offset's range, a number's gold value, a
heading's best run; a place and a list item have none). This module reads the gold row's tolerances and works out `e`,
`g`, the direction quarter and the harmful side by the gold's type:
- an `offset` (a thing or a person from the camera): `e` the 2-D error; the direction is right when the answered bearing
  is within `ANGLE_DEG` of the gold's (an answer of (0, 0) or a gold on the camera has no bearing, so it is right only
  when `e <= N`) or when `e <= N`; harmful (`harm` is `radial`) when the error points away from the camera along the
  gold's bearing, a thing placed farther than it is;
- a `number` (how far the body could move; how close a person came): `e = |a - g|`; the direction is right when the
  answer and the gold lie on the same side of `threshold_m` (the body's length plus N) and, where the gold is at least
  that, the answer is within a factor of two of it; harmful (`harm` is `over`) when the answer is the larger;
- a `position` (a place in the map): `e` the floor distance; the direction is right when `e <= N` (going there, the
  robot finds the thing in view; the room a thing is in is a fact of the map and the kind, which a sensorless prior
  knows, so it earns no quarter); never harmful;
- a `positions` list (`score_set.graded_set`): the items outside every don't-care circle are matched to the gold's by
  maximum summed credit, then those inside a circle take only the gold items left over;
- a `heading` (`regret`): the answered heading is read off the gold's `profile_m`, the run of the corridor at each
  heading; `e` is the room it gives up against the best heading. The profile has `360 / step_deg` entries, entry `i`
  the heading `i * step_deg` degrees to the left of the way the robot faces (null where the corridor was never seen,
  which runs 0); the right direction is room to move, a run of `threshold_m` (the body's length plus N) or the best
  run less a pixel where no heading has that much, and the harmful side is the rest (0.11.0: before it was half the
  best);
- a `heading` (`route`, the way to a remembered thing): the profile holds how much nearer one `step_m` of driving along
  each heading brings the robot to the thing through space it has seen (any number; null where the heading has no room
  to move or runs into space never seen); `e` is that outcome's shortfall against the best heading's, and a null
  heading has the full regret plus `step_m`, no direction and the harmful side.

The row's tolerances are all required: a missing or malformed key raises ValueError naming it, so no default decides a
credit unseen.
"""

from __future__ import annotations

import math

from stbench.answers import Heading, Number, Offset, Position, Positions
from stbench.credit_kernel import ANGLE_DEG, MIN_RHO_M, graded
from stbench.score_set import graded_set

EPS = 1e-9
HARMS = ("over", "radial", None)


def _tolerances(record: dict) -> dict:
    tolerances = record.get("tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError(f"gold {record.get('qid')!r}: tolerances must be an object, got {tolerances!r}")
    return tolerances


def _positive(record: dict, key: str) -> float:
    value = _tolerances(record).get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not value > 0:
        raise ValueError(f"gold {record.get('qid')!r}: tolerances.{key} must be a positive number, got {value!r}")
    return float(value)


def _harm(record: dict) -> str | None:
    tolerances = _tolerances(record)
    if "harm" not in tolerances or tolerances["harm"] not in HARMS:
        raise ValueError(f"gold {record.get('qid')!r}: tolerances.harm must be one of {list(HARMS)}, got "
                         f"{tolerances.get('harm', 'nothing')!r}")
    return tolerances["harm"]


def radial_error(answer: Offset, gold: Offset) -> float:
    """The component of the answer's error along the gold's bearing, in metres: positive when the answer places the
    point farther from the camera than the gold does (0 for a gold on the camera, which has no bearing). The bearing is
    a unit vector, so no product of two metres can overflow."""
    norm = math.hypot(gold.ahead_m, gold.left_m)
    if norm < EPS:
        return 0.0
    return ((answer.ahead_m - gold.ahead_m) * (gold.ahead_m / norm)
            + (answer.left_m - gold.left_m) * (gold.left_m / norm))


def _scaled(offset: Offset) -> tuple[float, float]:
    larger = max(abs(offset.ahead_m), abs(offset.left_m))
    return offset.ahead_m / larger, offset.left_m / larger


def bearing_deg(answer: Offset, gold: Offset) -> float | None:
    """The angle between the two offsets in degrees, None when either is the zero vector (it has no bearing). Each
    vector is scaled by its larger component first, which leaves the angle alone and keeps the products finite for any
    finite answer."""
    if math.hypot(answer.ahead_m, answer.left_m) < EPS or math.hypot(gold.ahead_m, gold.left_m) < EPS:
        return None
    ax, ay = _scaled(answer)
    gx, gy = _scaled(gold)
    return math.degrees(math.atan2(abs(ax * gy - ay * gx), ax * gx + ay * gy))


def _offset(answer: Offset, gold: Offset, record: dict) -> float:
    rho, need, harm = _positive(record, "resolution_m"), _positive(record, "need_m"), _harm(record)
    e = math.hypot(answer.ahead_m - gold.ahead_m, answer.left_m - gold.left_m)
    bearing = bearing_deg(answer, gold)   # None for a zero vector: no bearing, no direction unless within N
    direction = e <= need + EPS or (bearing is not None and bearing <= ANGLE_DEG + EPS)
    return graded(e, rho, need, direction, harm == "radial" and radial_error(answer, gold) > 0,
                  g=math.hypot(gold.ahead_m, gold.left_m))


def _number(answer: Number, gold: Number, record: dict) -> float:
    rho, need, harm = _positive(record, "resolution_m"), _positive(record, "need_m"), _harm(record)
    threshold = _positive(record, "threshold_m")
    a, g = answer.value, gold.value
    direction = (a >= threshold) == (g >= threshold) and (g < threshold or g / 2 <= a <= 2 * g)
    return graded(abs(a - g), rho, need, direction, harm == "over" and a > g, g=g)


def _position(answer: Position, gold: Position, record: dict) -> float:
    rho, need = _positive(record, "resolution_m"), _positive(record, "need_m")
    e = math.dist(answer.xy, gold.xy)
    return graded(e, rho, need, e <= need + EPS, False)


def graded_credit(answer: Offset | Number | Position, gold: Offset | Number | Position, record: dict) -> float:
    """Credit s of an answer of the gold's own type, by the type (see the module)."""
    match gold:
        case Offset():
            return _offset(answer, gold, record)
        case Number():
            return _number(answer, gold, record)
        case Position():
            return _position(answer, gold, record)
    raise ValueError(f"gold {record.get('qid')!r}: credit graded does not score a {type(gold).__name__}")


def graded_set_credit(answer: Positions, gold: Positions, record: dict) -> float:
    """Credit s of a `positions` answer: `score_set.graded_set` with the row's match radius, one resolution per gold
    item (in the gold's order) and false-item weight."""
    resolutions = _tolerances(record).get("resolutions_m")
    if (not isinstance(resolutions, list) or len(resolutions) != len(gold.items)
            or any(isinstance(r, bool) or not isinstance(r, int | float) or not r > 0 for r in resolutions)):
        raise ValueError(f"gold {record.get('qid')!r}: tolerances.resolutions_m must hold one positive number for each "
                         f"of the {len(gold.items)} gold items, got {resolutions!r}")
    weight = _tolerances(record).get("fp_weight")
    if isinstance(weight, bool) or not isinstance(weight, int | float) or not weight >= 0:
        raise ValueError(f"gold {record.get('qid')!r}: tolerances.fp_weight must be a number of at least 0, got "
                         f"{weight!r}")
    return graded_set(answer, gold, _positive(record, "match_m"), resolutions, float(weight))


def _profile(record: dict, *, signed: bool = False) -> tuple[list[float | None], float]:
    """The row's `profile_m` and `step_deg`; `signed` lets an entry be any finite number (`route`'s outcomes), else it
    is a run, at least 0."""
    tolerances = _tolerances(record)
    step = _positive(record, "step_deg")
    profile = tolerances.get("profile_m")
    size = round(360 / step)
    if abs(size * step - 360) > EPS or not isinstance(profile, list) or len(profile) != size or any(
            v is not None and (isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                               or (v < 0 and not signed)) for v in profile):
        what = "outcomes, each a finite number" if signed else "runs, each a number of at least 0"
        raise ValueError(f"gold {record.get('qid')!r}: tolerances.profile_m must hold {size} {what} or null "
                         "(360 / step_deg entries)")
    if all(v is None for v in profile):
        raise ValueError(f"gold {record.get('qid')!r}: tolerances.profile_m holds no heading that was seen")
    return profile, step


def regret_credit(answer: Heading, gold: Heading, record: dict) -> float:
    """Credit s of a `heading` answer: its heading is rounded to the profile's step (a heading of -180 and one of 180
    are the same entry), `d` is that run (0 for a null) and `r = max(profile) - d` the regret; the credit is the four
    quarters on `r` with the need widened to a tenth of the best run (`credit_kernel.KAPPA`). The right direction is
    room to move: a run of `threshold_m` or the best run less a pixel, whichever is shorter, so a heading with the
    room the body needs earns it however far short of the best; the harmful side is the rest (FD-0.11 §3.3)."""
    profile, step = _profile(record)
    rho, need, threshold = (_positive(record, key) for key in ("resolution_m", "need_m", "threshold_m"))
    run = profile[math.floor(answer.deg_left / step + 0.5) % len(profile)] or 0.0
    best = max(v for v in profile if v is not None)
    clear = run >= min(threshold, best - max(rho, MIN_RHO_M)) - EPS
    return graded(best - run, rho, need, clear, not clear, g=best)


def route_credit(answer: Heading, gold: Heading, record: dict) -> float:
    """Credit s of a `heading` answer to `way_to`, the way to a remembered thing (FD-0.11 §3.4.2): the answered
    heading, rounded to the profile's step, has the outcome `o` the gold's `profile_m` holds (how much nearer one
    `step_m` of driving along it brings the robot to the thing, through space it has seen), and the credit is the four
    quarters on the shortfall `max(profile) - o` against the best heading. A null entry is a heading with no room to
    move or into space never seen: its error is the best outcome plus `step_m`, it has no direction and is harmful,
    which scores it -1; any other heading has the direction, and none is harmful. The need is judged over `step_m`,
    the distance the outcome is measured over, as every distance's is."""
    profile, step = _profile(record, signed=True)
    rho, need, step_m = (_positive(record, key) for key in ("resolution_m", "need_m", "step_m"))
    outcome = profile[math.floor(answer.deg_left / step + 0.5) % len(profile)]
    best = max(v for v in profile if v is not None)
    if outcome is None:
        return graded(best + step_m, rho, need, False, True, g=step_m)
    return graded(best - outcome, rho, need, True, False, g=step_m)
