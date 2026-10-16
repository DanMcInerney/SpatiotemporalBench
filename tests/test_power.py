"""`power.py`: the variance of a pooled mean over scripts, the design effect, the Kish ICC, the count rule and the
budget allocation of the pilot (docs/eval.md, "The pilot report"; FD-M6-M7 §2.4). Every expected value is worked out by
hand in the comment above it, never by the code under test."""

from __future__ import annotations

import pytest

from stbench import power
from stbench.power import (
    allocate,
    cluster_variance,
    design_effect,
    half_width,
    icc,
    leak_verdict,
    naive_variance,
    score_sd,
    scripts_needed,
)

HOUR = 3600.0


def rows(scripts: dict[str, list[float]]) -> list[dict]:
    """Score rows (only the two keys `power` reads) of scripts given as {script: its rows' scores}."""
    return [{"script": s, "score": float(x)} for s, scores in scripts.items() for x in scores]


def ten_scripts() -> list[dict]:
    """Ten scripts of four rows: five score (1, 1, 1, 0), five (1, 0, 0, 0)."""
    return rows({**{f"s{i}": [1, 1, 1, 0] for i in range(5)}, **{f"s{i}": [1, 0, 0, 0] for i in range(5, 10)}})


def test_cluster_variance_design_effect_and_icc_by_hand():
    """Contract (FD-M6-M7 §2.4): the variance of the pooled mean clusters by script, S/(S-1) sum_s (T_s - R n_s)^2 /
    N^2, against the naive s^2/N; their ratio is the design effect and (DE - 1)/(m_bar - 1), clipped to [0, 1], the
    Kish ICC. By hand:
    - scripts {a: 1, 1; b: 0, 0}: N 4, R 0.5, T_s - R n_s = +1, -1, so 2/1 x 2 / 16 = 0.25; s^2 = 4 x 0.25 / 3 = 1/3,
      naive 1/12; DE 3; ICC (3 - 1)/(2 - 1) = 2, clipped to 1;
    - {a: 1, 0; b: 1, 0}: T_s - R n_s = 0 for both, so the cluster variance is 0, DE 0, ICC -1 clipped to 0;
    - ten scripts of four rows (five of 3 hits, five of 1): N 40, R 0.5, T_s - R n_s = +1 or -1, so 10/9 x 10 / 1600 =
      1/144; sum (y - 0.5)^2 = 40 x 0.25 = 10, s^2 = 10/39, naive 1/156; DE 156/144 = 13/12; ICC (1/12)/3 = 1/36.
    Regression caught: clustering by question or case (intervals too narrow), a ratio estimator centred on the wrong
    mean, or an ICC outside [0, 1]."""
    hit = rows({"a": [1, 1], "b": [0, 0]})
    assert cluster_variance(hit) == pytest.approx(0.25)
    assert naive_variance(hit) == pytest.approx(1 / 12)
    assert design_effect(hit) == pytest.approx(3.0)
    assert icc(3.0, 2.0) == 1.0
    even = rows({"a": [1, 0], "b": [1, 0]})
    assert cluster_variance(even) == pytest.approx(0.0, abs=1e-15)
    assert design_effect(even) == pytest.approx(0.0, abs=1e-15)
    assert icc(0.0, 2.0) == 0.0
    ten = ten_scripts()
    assert cluster_variance(ten) == pytest.approx(1 / 144)
    assert naive_variance(ten) == pytest.approx(1 / 156)
    assert design_effect(ten) == pytest.approx(13 / 12)
    assert icc(13 / 12, 4.0) == pytest.approx(1 / 36)


def test_a_variance_that_cannot_be_estimated_is_none():
    """Contract: one script has no cluster variance, one row no SD, and scores that never vary have no design effect
    (0 over 0); the ICC of one row a script is undefined. sigma is the sample SD (n - 1): scores 1, 1, 0, 0 have
    sqrt(1/3) = 0.57735. Regression caught: a division by zero or a made-up number where nothing is estimable (the
    report prints `n/a`), or a population SD where the sample SD is meant."""
    assert cluster_variance(rows({"only": [1, 0]})) is None
    assert naive_variance(rows({"only": [1]})) is None and score_sd(rows({"only": [1]})) is None
    assert design_effect(rows({"a": [0.5, 0.5], "b": [0.5, 0.5]})) is None
    assert score_sd(rows({"a": [1, 1], "b": [0, 0]})) == pytest.approx(0.57735, abs=1e-5)
    with pytest.raises(ValueError, match="more than one row a script"):
        icc(2.0, 1.0)


def test_scripts_needed_and_half_width_by_hand():
    """Contract (FD-M6-M7 §2.4): K = ceil((1.959964 sigma / h)^2 (1 + (m - 1) rho) / m) and its inverse, the half-width
    z sigma sqrt((1 + (m - 1) rho) / (m K)). By hand for sigma 0.8, m 20, rho 0.05, h 0.10: (1.5679712 / 0.1)^2 =
    245.85, (1 + 19 x 0.05)/20 = 0.0975, so 23.97, K 24. The half-width at 24 is 1.5679712 sqrt(0.0975/24) = 0.09994
    (within 0.10) and at 23 it is 1.5679712 sqrt(0.0975/23) = 0.10209 (above). A quotient that is a whole number up to
    float noise (sigma 0.95, h = z sigma / 2, m 1: exactly 4, but 4.000000000000001 in floats) does not round up.
    Regression caught: a count one short or one over, ICC and m entering the wrong way round, or a ceiling that
    overshoots on float error."""
    assert scripts_needed(0.8, 20, 0.05, 0.10) == 24
    assert half_width(0.8, 20, 0.05, 24) == pytest.approx(0.09994, abs=1e-5)
    assert half_width(0.8, 20, 0.05, 23) == pytest.approx(0.10209, abs=1e-5)
    assert half_width(0.8, 20, 0.05, 24) <= 0.10 < half_width(0.8, 20, 0.05, 23)
    assert scripts_needed(0.8, 1, 0.9, 0.10) == 246   # one question a script: rho has no part: 245.85 -> 246
    assert scripts_needed(0.95, 1, 0.5, power.Z * 0.95 / 2) == 4
    assert scripts_needed(0.95, 1, 0.5, power.Z * 0.95 / 3) == 9
    for bad in (lambda: scripts_needed(0.8, 20, 0.05, 0.0), lambda: half_width(0.8, 20, 0.05, 0),
                lambda: scripts_needed(0.8, 0, 0.05, 0.1)):
        with pytest.raises(ValueError):
            bad()


def test_allocate_by_hand():
    """Contract (FD-M6-M7 §2.4): K_j = max(k_min, floor(f K_target)) for the largest f <= 1 whose scripts fit both
    budgets, and `over_budget` when even k_min scripts a job do not. By hand, targets a 20 and b 10, an hour of
    render a script, k_min 6:
    - 15 h: f 0.45 gives a 9 and b max(6, 4) = 6, 15 h; f 0.5 gives 10 + 6 = 16 h; so {9, 6} at f 0.45;
    - 100 h: f 1, K unchanged at {20, 10}; 10 h: 6 + 6 = 12 h overruns whatever f, so {6, 6} over budget (f 0);
    - the points budget binds alike: 2 and 1 points a script against 24 give 18 + 6, {9, 6}, with the hours ample;
    - targets 24 and 36 at 40 h: f 2/3 gives 16 and floor(36 x 2/3) = 24, exactly 40 h, so {16, 24}; f 25/36 gives 16
      and 25, 41 h;
    - one target of 23 at 13 h: f 13/23 gives floor(23 x 13/23) = 13 (in floats 23 x (13/23) is 12.999999999999998,
      which floors to 12), so {13};
    - a target below k_min (3) still gets k_min.
    Regression caught: a floor that loses a script to float error, a budget that is not checked together with the other,
    or K above its target."""
    two = {"a": 20, "b": 10}
    hours = {"a": HOUR, "b": HOUR}
    free = {"a": 0.0, "b": 0.0}
    cheap = allocate(two, hours, free, 15 * HOUR, 1e9, 6)
    assert (cheap.k, cheap.f, cheap.over_budget) == ({"a": 9, "b": 6}, pytest.approx(0.45), False)
    assert allocate(two, hours, free, 100 * HOUR, 1e9, 6) == ({"a": 20, "b": 10}, 1.0, False)
    short = allocate(two, hours, free, 10 * HOUR, 1e9, 6)
    assert (short.k, short.f, short.over_budget) == ({"a": 6, "b": 6}, 0.0, True)
    quota = allocate(two, {"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 1.0}, 1e9, 24, 6)
    assert (quota.k, quota.over_budget) == ({"a": 9, "b": 6}, False)
    steps = allocate({"a": 24, "b": 36}, hours, free, 40 * HOUR, 1e9, 6)
    assert (steps.k, steps.f) == ({"a": 16, "b": 24}, pytest.approx(2 / 3))
    assert allocate({"a": 23}, {"a": HOUR}, {"a": 0.0}, 13 * HOUR, 1e9, 6).k == {"a": 13}
    assert allocate({"a": 3, "b": 10}, hours, free, 100 * HOUR, 1e9, 6).k == {"a": 6, "b": 10}


def test_leak_verdict_at_the_margins_edges():
    """Contract (FD-M6-M7 §2.2, D2): fail when d is above the margin, pass when the interval's upper bound is at or
    below it (a bound on the margin passes), inconclusive otherwise; with no interval nothing passes. By hand at the
    margin 0.05: d 0 with bound 0.05 passes, with bound 0.0501 is inconclusive; d 0.05 (not above) with bound 0.2 is
    inconclusive, d 0.0501 fails; d 0.01 with no interval is inconclusive and 0.06 fails; at margin 0.1, d 0.07 passes
    on bound 0.1. Regression caught: a strict comparison that fails a bound sitting on the margin, d on the margin
    called a failure, or a missing interval called a pass."""
    assert leak_verdict(0.0, (-0.1, 0.05)) == "pass"
    assert leak_verdict(0.0, (-0.1, 0.0501)) == "inconclusive"
    assert leak_verdict(0.05, (-0.1, 0.2)) == "inconclusive"
    assert leak_verdict(0.0501, (-0.1, 0.2)) == "fail"
    assert leak_verdict(0.01, None) == "inconclusive" and leak_verdict(0.06, None) == "fail"
    assert leak_verdict(0.07, (0.0, 0.1), margin=0.1) == "pass"
