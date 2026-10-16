"""How many scripts a pooled mean needs (each column of the headline is one): the variance of a pooled mean over
scripts, the design effect, the Kish ICC and the count rule of the pilot (docs/eval.md, "The pilot report"; design §9).

The rows are `score_stats` result rows (only `script` and `score` are read). A pooled mean (each column of the headline
is one) is R = T / N of the reported score over N rows, and the cases of one script are windows of one world, so the
rows of a script are not independent. The variance of the pooled mean is estimated by the ratio estimator over scripts
(`cluster_variance`), and compared with the variance the same N rows would have were each independent
(`naive_variance`). Their ratio is the design effect DE. With a mean of `m_bar` rows a script, DE = 1 + (m_bar - 1) rho,
so the Kish ICC is `rho = (DE - 1) / (m_bar - 1)`, clipped to [0, 1]. The pooled mean over K scripts of m questions then
has the variance sigma^2 (1 + (m - 1) rho) / (m K), and `scripts_needed` and `half_width` invert one another. `allocate`
cuts the counts back to the render and quota budgets (the rule is FD-M6-M7 §2.4) and `leak_verdict` reads D2's interval.
This module imports numpy and `score_stats` only, so it reads no truth and runs anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from .score_stats import Interval, Row

Z = 1.959964   # the 95% two-sided normal quantile
EPS = 1e-9     # absorbs the float error of a quotient that is a whole number, and of a budget met exactly
VERDICTS = ("pass", "inconclusive", "fail")


class Allocation(NamedTuple):
    """`allocate`'s answer: the scripts a job (`k`), the factor `f` of the count targets that was kept, and whether even
    `k_min` scripts a job overrun a budget."""

    k: dict[str, int]
    f: float
    over_budget: bool


def _by_script(rows: Iterable[Row]) -> tuple[np.ndarray, np.ndarray]:
    """(T_s, n_s): the sum of the scores and the number of rows of each script, in script order."""
    found: dict[str, list[float]] = {}
    for r in rows:
        total = found.setdefault(r["script"], [0.0, 0.0])
        total[0] += r["score"]
        total[1] += 1
    both = np.array([found[s] for s in sorted(found)], dtype=float).reshape(-1, 2)
    return both[:, 0], both[:, 1]


def score_sd(rows: Iterable[Row]) -> float | None:
    """The sample SD (n - 1) of the score over the rows: sigma. None for fewer than two rows."""
    scores = [r["score"] for r in rows]
    return float(np.std(scores, ddof=1)) if len(scores) > 1 else None


def cluster_variance(rows: Iterable[Row]) -> float | None:
    """The variance of the pooled mean of the score with the script as the cluster:
    S / (S - 1) * sum_s (T_s - R n_s)^2 / N^2, for S scripts of n_s rows summing to T_s, N rows in all and R = T / N.
    None for fewer than two scripts."""
    totals, sizes = _by_script(rows)
    if len(totals) < 2:
        return None
    n = sizes.sum()
    r = totals.sum() / n
    return float(len(totals) / (len(totals) - 1) * np.sum((totals - r * sizes) ** 2) / n**2)


def naive_variance(rows: Iterable[Row]) -> float | None:
    """s^2 / N: the variance of the pooled mean were the N rows independent. None for fewer than two rows."""
    rows = list(rows)
    sd = score_sd(rows)
    return None if sd is None else sd**2 / len(rows)


def design_effect(rows: Iterable[Row]) -> float | None:
    """`cluster_variance` over `naive_variance`. None when either is (fewer than two scripts) or the scores do not
    vary."""
    rows = list(rows)
    cluster, naive = cluster_variance(rows), naive_variance(rows)
    if cluster is None or naive is None or naive <= 0:
        return None
    return cluster / naive


def icc(de: float, m_bar: float) -> float:
    """The Kish ICC, (DE - 1) / (m_bar - 1) for a mean of `m_bar` rows a script, clipped to [0, 1]. ValueError for a
    `m_bar` of 1 or less, where nothing can be correlated."""
    if m_bar <= 1:
        raise ValueError(f"the ICC needs more than one row a script, not a mean of {m_bar}")
    return min(1.0, max(0.0, (de - 1.0) / (m_bar - 1.0)))


def _per_script(sigma: float, m: float, rho: float) -> float:
    """The variance of one script's mean of m questions, in units of 1/K: sigma^2 (1 + (m - 1) rho) / m."""
    if m <= 0:
        raise ValueError(f"m must be above 0, got {m}")
    return sigma**2 * (1.0 + (m - 1.0) * rho) / m


def scripts_needed(sigma: float, m: float, rho: float, half_width: float) -> int:
    """The scripts K whose pooled mean has a 95% half-width of at most `half_width`:
    ceil((z sigma / h)^2 (1 + (m - 1) rho) / m), for m questions a script."""
    if half_width <= 0:
        raise ValueError(f"the half-width must be above 0, got {half_width}")
    return max(0, math.ceil(Z**2 * _per_script(sigma, m, rho) / half_width**2 - EPS))


def half_width(sigma: float, m: float, rho: float, k: int) -> float:
    """The 95% half-width of a pooled mean over K scripts of m questions: z sigma sqrt((1 + (m - 1) rho) / (m K))."""
    if k < 1:
        raise ValueError(f"K must be at least 1, got {k}")
    return Z * math.sqrt(_per_script(sigma, m, rho) / k)


def allocate(targets: dict[str, int], seconds: dict[str, float], points: dict[str, float], render_s: float,
             points_cap: float, k_min: int) -> Allocation:
    """K_j = max(k_min, floor(f * targets[j])) for the largest f <= 1 whose scripts fit both budgets: `seconds[j]`
    render seconds and `points[j]` quota points a script of job j, against `render_s` and `points_cap` in all. K only
    changes where f * targets[j] crosses a whole number, so f is searched over those breakpoints (the largest at which
    the allocation fits is the f returned, exact in fractions). When even k_min scripts a job overrun a budget, K is
    k_min everywhere, f is 0 and `over_budget` is set."""
    jobs = sorted(targets)

    def at(f: Fraction) -> dict[str, int]:
        return {j: max(k_min, math.floor(f * targets[j])) for j in jobs}

    def fits(k: dict[str, int]) -> bool:
        return (sum(k[j] * seconds[j] for j in jobs) <= render_s * (1 + EPS)
                and sum(k[j] * points[j] for j in jobs) <= points_cap * (1 + EPS))

    breaks = {Fraction(1)} | {Fraction(n, targets[j]) for j in jobs for n in range(1, targets[j] + 1)}
    for f in sorted(breaks, reverse=True):
        if fits(k := at(f)):
            return Allocation(k, float(f), False)
    return Allocation(at(Fraction(0)), 0.0, True)


def leak_verdict(mean: float, ci: Interval, margin: float = 0.05) -> str:
    """D2's reading of d = on - off over end questions against the margin: `fail` when d is above it, `pass` when the
    interval's upper bound is at or below it (a bound on the margin passes), else `inconclusive`: the check cannot
    exclude a leak of the upper bound. With no interval (fewer than two scripts) nothing passes."""
    if mean > margin:
        return "fail"
    return "pass" if ci is not None and ci[1] <= margin else "inconclusive"
