"""Hand-made per-kind seen bounds for tests of the threshold mechanism (`stbench.seen_table`).

Bounds are written as pixel counts at the standard camera (640x480, 307,200 px) and stored as the fractions of the image
that the table holds, so a test reads in the pixel counts its hand arithmetic uses. `use` swaps the table for the length
of one test (pytest's monkeypatch puts it back; each xdist worker is a process of its own).
"""

from __future__ import annotations

from stbench import seen_table

IMAGE_PX = 640 * 480


def table(**px: tuple[int, int]) -> dict[str, tuple[float, float]]:
    """{kind: (lo, hi)} in pixels at 640x480, as the fractions `seen_table.BOUNDS` holds."""
    return {kind: (lo / IMAGE_PX, hi / IMAGE_PX) for kind, (lo, hi) in px.items()}


def use(monkeypatch, **px: tuple[int, int]) -> None:
    """Make these kinds' (lo, hi) pixel counts at 640x480 the seen table for this test."""
    monkeypatch.setattr(seen_table, "BOUNDS", table(**px))
