"""Decision windows (`stbench/look_windows.py`): the stream protocol's batch rule of 0.9.0, moved to the question
compiler unchanged when every system got one frame per `observe` (DECISIONS 2026-10-14); since 0.10.0 only a decision
point's own families and `stbench view` read it, a look point being any frame. Expected values are hand tables from the
plan (§2.4)."""

from __future__ import annotations

import pytest

from stbench.look_windows import LOOK_WINDOW, windows


def test_windows_restart_after_each_cut():
    """Contract (plan §2.4, map D-20): case-relative [lo, hi) decision windows from step 0; a window ends after 8 steps
    or after a step in `cuts`, whichever is first, and the count restarts after every cut; a cut outside [0, n_steps)
    is ignored. Hand table. Regression caught: a window pattern that depends on the case's offset, or a decision
    window that moves."""
    assert windows(20, {3, 12}) == [(0, 4), (4, 12), (12, 13), (13, 20)]
    assert windows(9, ()) == [(0, 8), (8, 9)]
    assert windows(8, {7}) == [(0, 8)]
    assert windows(0, ()) == []
    assert windows(20, {3, 12, 25}) == [(0, 4), (4, 12), (12, 13), (13, 20)]
    assert windows(20, {25, -1}) == [(0, 8), (8, 16), (16, 20)]


@pytest.mark.parametrize("n_steps, cuts", [(1, ()), (7, {0, 6}), (16, {7, 8}), (30, {1, 2, 3, 20, 29}), (41, ())])
def test_windows_tile_the_steps_and_end_at_every_cut(n_steps, cuts):
    """Contract: the decision windows cover steps 0..n_steps-1 once, in order, none longer than `LOOK_WINDOW`, and every
    cut step is the last of its window (a decision point ends a window). Regression caught: a gap or an overlap between
    windows, or a cut that falls inside one."""
    out = windows(n_steps, cuts)
    assert out[0][0] == 0 and out[-1][1] == n_steps
    assert all(hi == nxt for (_, hi), (nxt, _) in zip(out, out[1:], strict=False))
    assert all(0 < hi - lo <= LOOK_WINDOW for lo, hi in out)
    assert {hi - 1 for _, hi in out} >= set(cuts)
