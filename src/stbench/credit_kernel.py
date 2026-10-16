"""The credit kernel (0.9.0, docs/methodology.md rule 7): four quarters, one per layer of the vision.

An answer with error `e` against the job's need `N` (the clearance C, the re-find radius R or a list's match radius M)
and the gold's sensor resolution `rho` earns `s = (d + t + a + f) / 4`, with `N' = max(N, KAPPA g, 2 rho)`, `g` the
distance the answer is judged over (0 for a place and a list item, which have none):
- `d`, 0 or 1: the direction (the side, the order, blocked or clear), decided by the scorer per answer type;
- `t`: 1 within `2 N'`, falling linearly to 0 at `4 N'`;
- `a`: 1 within `N'`, falling linearly to 0 at `2 N'`;
- `f`: 1 within `rho`, then `1 - ln(e / rho) / ln(N' / rho)`, log-linear to 0 at `N'`.
So an answer within `rho` scores 1, at `N` 3/4, at `2N` 1/2, and at `4N` 1/4 when its direction is right. The curve is
continuous and falls strictly from `rho` to `4 N'`; its one step is the direction quarter. The width `2 rho` the need is
widened to keeps `f` a curve and not a step when one pixel is half the need or more.

On the harmful side (more clearance than there is, a thing or a person placed farther than it is, a heading toward too
little room) the need `max(N, KAPPA g)` is halved (`HARM_FACTOR`) and `rho` is not.

`KAPPA g` (0.11.0, FD-0.11 §3.1) is the one rule for a distance that is far: the margin is the job's own need N within
ten needs (`g <= N / KAPPA`: the near field, where harm happens) and a tenth of the distance beyond, the same for every
robot, where before it was whatever one pixel of the robot's camera resolved. The door 5 m away, answered 4 m, scores 0
on the helper, the vacuum and the AMR.

The scorer, the reports and the compiler import this module; it holds the arithmetic and nothing else (standard library
only), so one place says what a quarter is worth.
"""

from __future__ import annotations

import math

# The far field's relative slack: the need grows to this share of the distance judged (FD-0.11 §3.1; the form of
# numpy.isclose's max(atol, rtol |b|)). The largest value that keeps every constant at the floor and the models apart
# (evidence §1.2, §2.4): 0.2 lifts the AMR's constants. Stated, never tuned to a score.
KAPPA = 0.1
MIN_RHO_M = 0.01    # the least resolution a gold row may carry: an answer's natural precision and the pose's rounding
HARM_FACTOR = 0.5   # the harmful side's need, as a share of the need
FP_WEIGHT = 0.5     # a false item of a list costs this much of a missed one
ANGLE_DEG = 45      # an offset's direction quarter: the answered bearing within this of the gold's
MATCH_REACH = 4     # a list item is matched within this many match radii (`graded_set`)
# The catastrophe weights (0.10.0, FD-questions-0.10 §4.3): what the reported score (never a gate's) charges for an
# answer whose action would end in harm. THING: contact with something the robot saw, which damages or stops the job
# and a person clears. PERSON: contact with a person, or a hazard of the robot's own kind (a tangle), which endangers
# someone or ends the job until a person intervenes. Stated values, never tuned to a score.
CATASTROPHE_THING = 2
CATASTROPHE_PERSON = 3


def tiers(e: float, rho: float, need: float) -> tuple[float, float, float]:
    """The three graded quarters `(t, a, f)` of an error `e` metres against the resolution `rho` and the need `need`
    (module docstring): `rho` is raised to `MIN_RHO_M` and the need to `2 rho`. The direction quarter is the
    scorer's."""
    rho = max(rho, MIN_RHO_M)
    n = max(need, 2 * rho)
    e = abs(e)
    t = min(1.0, max(0.0, (4 * n - e) / (2 * n)))
    a = min(1.0, max(0.0, (2 * n - e) / n))
    f = 1.0 if e <= rho else min(1.0, max(0.0, 1.0 - math.log(e / rho) / math.log(n / rho)))
    return t, a, f


def graded(e: float, rho: float, need: float, direction: bool, harmful: bool, g: float = 0.0) -> float:
    """The credit of an answer `e` metres off: `(direction + t + a + f) / 4` on the need `max(need, KAPPA * g)`, which
    is halved (`HARM_FACTOR`) when the error is on the harmful side. `g` is the distance the answer is judged over (an
    offset's range, a distance's gold value, a heading's best run); 0, the default, is a place or a list item."""
    need = max(need, KAPPA * g)
    t, a, f = tiers(e, rho, need * HARM_FACTOR if harmful else need)
    return (float(direction) + t + a + f) / 4
