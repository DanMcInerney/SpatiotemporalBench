"""The per-kind seen bounds: how many pixels of a kind the robot needs before it can know anything about it (M2 plan 5).

`BOUNDS[kind] = (lo, hi)`, both as fractions of the image (w * h), so another resolution scales them like the global
rule does (`rules.SEEN_FRACTION`, 0.07%):
- `hi`: from this many pixels a thing of the kind counts as seen. The oracle channel detects from it
  (`visibility.status`) and the compiler's sure-seen bound is 1.25 x it (`rules.gold_seen_px`).
- `lo`: under this many pixels a glimpse of the kind counts as unseen, not as the dead band: the probe shows no
  perceiver recognises the kind there, so nothing usable is in the working context. 0 means every pixel counts
  (`rules.glimpse_px` floors it at one pixel).

A kind with no entry (fixed furniture, doors, anything unlisted) uses the global bounds of `rules`. A bound belongs to
a kind, not to a map: a kind that two catalogs share (`box`) is one model at one size, so it has one bound. The lookups
(`seen_px`, `glimpse_px`, `gold_seen_px`, `min_cover_px`) are owned by `rules.py`; this module is data and its checks.

`BOUNDS` is empty, so every kind uses the global bounds (215 px seen, a glimpse from 1 px, at 640x480).
"""

from __future__ import annotations

from collections.abc import Mapping

BOUNDS: dict[str, tuple[float, float]] = {}

# A seen bound above 1% of the image (3,072 px at 640x480) is a pixel count written as a fraction, or a typo: the
# probe's bins stop at 1,024 px.
MAX_HI = 0.01


def problems(bounds: Mapping[str, tuple[float, float]]) -> list[str]:
    """What is wrong with a table: a kind that is not a name, a bound that is not a number, `lo` negative or above
    `hi`, `hi` not above 0 or above MAX_HI. Empty when the table is usable."""
    out = []
    for kind, pair in sorted(bounds.items()):
        if not isinstance(kind, str) or not kind:
            out.append(f"kind {kind!r} is not a name")
            continue
        if not (isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(x, int | float) for x in pair)):
            out.append(f"{kind}: bounds must be a (lo, hi) pair of numbers, not {pair!r}")
            continue
        lo, hi = pair
        if lo < 0:
            out.append(f"{kind}: lo {lo} is negative")
        if lo > hi:
            out.append(f"{kind}: lo {lo} is above hi {hi}")
        if not 0 < hi <= MAX_HI:
            out.append(f"{kind}: hi {hi} is not a fraction of the image in (0, {MAX_HI}]")
    return out
