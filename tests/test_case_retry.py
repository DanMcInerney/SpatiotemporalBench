"""The private index's retry (`case._retry`): a file another process holds open is waited for, with jittered waits, up
to a deadline. Time is a fake clock that only `sleep` advances, so nothing here waits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stbench import case as case_mod

STEPS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.4]   # the waits' bounds: doubling from 5 ms, capped at 0.4 s


def fake_time(monkeypatch) -> list[float]:
    """`case.time` replaced by a clock that `sleep` advances; the waits slept are returned, in order."""
    now, slept = [0.0], []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(case_mod, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    return slept


def locked(times: int):
    """A callable that raises `PermissionError` `times` times, then returns "open"."""
    left = [times]

    def fn() -> str:
        if left[0] > 0:
            left[0] -= 1
            raise PermissionError("held by another process")
        return "open"

    return fn


def test_a_held_file_is_retried_until_it_opens(monkeypatch):
    """Contract: `_retry` calls `fn` again after a `PermissionError` and returns the value of the first call that
    succeeds, sleeping once between the calls; a file that is free is read at once, with no sleep. Hand case: three
    refusals then success is four calls and three sleeps. Regression caught: a read or swap with no retry (a
    `stbench build` worker died in `claim_id` on Windows), or a retry that sleeps before the first try."""
    slept = fake_time(monkeypatch)
    assert case_mod._retry(locked(0)) == "open" and slept == []
    assert case_mod._retry(locked(3)) == "open" and len(slept) == 3


def test_a_file_held_past_the_deadline_raises_and_other_errors_do_not_wait(monkeypatch):
    """Contract: the retry gives up when `RETRY_S` have passed, re-raising the `PermissionError` (never swallowing it
    and never waiting for ever), and any other error is raised at once with no sleep. Hand case: a file held for ever
    is waited for at least 5 s and less than 5.4 s in all (the last wait is at most the 0.4 s cap); a
    `FileNotFoundError` sleeps 0 times. Regression caught: a retry loop with no deadline (a build that hangs on a stuck
    file), or one that retries errors that waiting cannot cure."""
    slept = fake_time(monkeypatch)
    assert case_mod.RETRY_S == 5.0
    with pytest.raises(PermissionError, match="held by another process"):
        case_mod._retry(locked(10**9))
    assert 5.0 <= sum(slept) < 5.4
    slept.clear()

    def gone() -> None:
        raise FileNotFoundError("gone")

    with pytest.raises(FileNotFoundError):
        case_mod._retry(gone)
    assert slept == []


def test_the_waits_are_jittered_so_processes_that_collided_do_not_wake_together(monkeypatch):
    """Contract: each wait is drawn between half and the whole of its step (5 ms doubling to the 0.4 s cap), so two
    retries over the same refusals sleep different times. Hand case: the first eight waits are bounded by 0.005, 0.01,
    0.02, 0.04, 0.08, 0.16, 0.32 and 0.4 s, each at least half its bound, and two runs of them differ. Regression
    caught: a fixed schedule, which put every process of a parallel build on the same clock (1.6 s in all), so the
    ones that collided collided again until the budget ran out (the index race test failed under a loaded suite)."""
    slept = fake_time(monkeypatch)
    runs = []
    for _ in range(2):
        slept.clear()
        assert case_mod._retry(locked(len(STEPS))) == "open"
        runs.append(list(slept))
    for run in runs:
        assert len(run) == len(STEPS)
        assert all(bound / 2 <= wait <= bound for wait, bound in zip(run, STEPS, strict=True)), run
    assert runs[0] != runs[1]
