"""The gap law every job shares (`job_gaps`): the unobserved gaps between a script's tasks, drawn from the map, the seed
and the job alone, so that every case clock ends by 17:00 under the fixed daytime light."""

from __future__ import annotations

from stbench import job_gaps, job_vacuum2_plan

TOL = 1e-6  # seconds: float slack on a sum that is exactly the budget
SPAN_HAND_S = 25200.0  # seven hours, by hand: the span a script may not pass
LEAST_HAND_S, MOST_HAND_S = 600.0, 10800.0  # ten minutes and three hours, by hand


def test_the_gap_law_is_vacuum2s():
    """Contract (M3's extraction of `vacuum/2`'s law): `job_gaps.gaps` with `vacuum/2`'s id and its longest task
    (1,200 s) draws, to the bit, what `job_vacuum2_plan.gaps` returns, over seeds 101-110 and 4 to 6 tasks, and each
    vector holds the law: n - 1 gaps, each from 10 minutes to 3 hours, and its gaps plus n longest tasks fit in the
    seven hours, landing on that span when the draw was shrunk to it (some of these draws are). No draw is recorded.
    Regression caught: the refactor moving a home script (the gaps set every clock), or `vacuum/2`'s constants no
    longer reaching the law."""
    shrunk = 0
    for seed in range(101, 111):
        for n in (4, 5, 6):
            found = job_gaps.gaps("apartment_a", seed, "vacuum/2", n, 1200.0)
            assert found == job_vacuum2_plan.gaps("apartment_a", seed, n), (seed, n)
            assert len(found) == n - 1 and all(LEAST_HAND_S - TOL <= g <= MOST_HAND_S + TOL for g in found), (seed, n)
            total = sum(found) + n * 1200.0
            assert total <= SPAN_HAND_S + TOL, (seed, n)
            shrunk += abs(total - SPAN_HAND_S) < TOL
    assert shrunk > 0  # the shrink path was drawn


def test_gaps_hold_the_span():
    """Contract: for any job and any longest task of 300, 600 or 1,200 s, over 1,000 seeds and one to six tasks, a
    script has n - 1 gaps (none for one task), each from 10 minutes to 3 hours, and its gaps plus n of its longest
    tasks fit in seven hours, so the case clock of its last task ends by 17:00 from a 10:00 start. The draws include
    scripts that needed the shrink and scripts that did not. Regression caught: a job whose case clock runs past 17:00
    under the fixed light, or a shrink that takes a gap under ten minutes."""
    shrunk = loose = 0
    for task_max in (300.0, 600.0, 1200.0):
        for n in range(1, 7):
            for seed in range(1000):
                job = ("helper/1", "amr/1", "drone/1")[seed % 3]
                found = job_gaps.gaps("apartment_a", seed, job, n, task_max)
                assert len(found) == n - 1
                assert all(LEAST_HAND_S - TOL <= g <= MOST_HAND_S + TOL for g in found), (task_max, n, seed)
                total = sum(found) + n * task_max
                assert total <= SPAN_HAND_S + TOL, (task_max, n, seed)
                shrunk += abs(total - SPAN_HAND_S) < TOL
                loose += total < SPAN_HAND_S - 1.0
    assert job_gaps.gaps("apartment_a", 101, "amr/1", 1, 600.0) == []
    assert shrunk > 100 and loose > 100  # both paths of the law were drawn


def test_a_jobs_own_range_and_span_are_honoured():
    """Contract: `gap_s` and `span_s` are the job's to set. With gaps from 60 to 120 s and a span far past their sum,
    six tasks of 300 s draw five gaps inside that range, unshrunk; with the default range and a span of 3,100 s, four
    tasks of 300 s have a budget of 3,100 - 4 x 300 = 1,900 s, so their three gaps sum to it (shrunk toward the
    least) and none is under 600 s. Regression caught: a law that reads the module constants instead of its
    arguments, so a job's range or span is silently the vacuum's."""
    for seed in range(101, 111):
        small = job_gaps.gaps("apartment_a", seed, "amr/1", 6, 300.0, gap_s=(60.0, 120.0), span_s=100000.0)
        assert len(small) == 5 and all(60.0 - TOL <= g <= 120.0 + TOL for g in small), seed
        tight = job_gaps.gaps("apartment_a", seed, "amr/1", 4, 300.0, span_s=3100.0)
        assert len(tight) == 3 and all(g >= LEAST_HAND_S - TOL for g in tight), seed
        assert abs(sum(tight) - 1900.0) < TOL, seed


def test_gaps_depend_on_the_job():
    """Contract: the stream is keyed by the map, the seed and the job id: the same map, seed and number of tasks with
    two job ids draw different gaps (every gap differs), and one id draws the same gaps twice; the map and the seed
    move them too. Regression caught: two jobs drawing one stream (their scripts' clocks would be the same), or a
    draw that depends on the process rather than on its key."""
    first = job_gaps.gaps("apartment_a", 101, "amr/1", 5, 600.0)
    other = job_gaps.gaps("apartment_a", 101, "drone/1", 5, 600.0)
    assert len(first) == len(other) == 4 and set(first).isdisjoint(other)
    assert job_gaps.gaps("apartment_a", 101, "amr/1", 5, 600.0) == first
    assert set(first).isdisjoint(job_gaps.gaps("warehouse_a", 101, "amr/1", 5, 600.0))
    assert set(first).isdisjoint(job_gaps.gaps("apartment_a", 102, "amr/1", 5, 600.0))
