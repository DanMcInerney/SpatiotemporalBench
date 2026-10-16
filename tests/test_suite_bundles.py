"""The pool's tasks (`suite._bundles`): the runs over one case's files go to one process, the longest case first."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stbench import baselines, suite
from stbench.suite import RunOptions, run_suite
from stream_kit import case_on, end_q, hand20
from test_suite_runner import Lengths, Steps, values

CHANNEL = "rgb_pose"


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """Three scripts of the 20-step hand case, a case each with one end question."""
    out = []
    for n in (1, 2, 3):
        synth = hand20()
        synth.seed = n
        out.append(case_on(tmp_path_factory.mktemp("bundles"), synth, [end_q(10 * n, text="q" * n)]))
    return out


def test_runs_over_one_cases_files_are_one_task_and_a_run_fed_another_stream_joins_its_case(cases):
    """Contract: `_bundles` groups the jobs by the case whose files the run reads (its own, or the case whose stream it
    is fed), as lists of indices into the jobs, and puts the group with the most work (jobs times the case's steps)
    first, the others in the order they came. Hand case: two systems over each of three cases of 20 steps, and a
    third system on case 3 that is fed case 1's stream: case 1's group holds its two jobs and that one (60 steps of
    work), the others two jobs each (40). Regression caught: a run in a task of its own meeting its image in a process
    that has not read it, or a heavy case left to the end of the pool's queue."""
    jobs = [(name, Lengths, case, RunOptions({}), True) for case in cases for name in ("a", "b")]
    jobs.append(("fed", Lengths, cases[2], RunOptions({}, stream_from=cases[0], ask_map={}), True))
    assert suite._bundles(jobs) == [[0, 1, 6], [2, 3], [4, 5]]


def test_a_pool_task_of_several_runs_gives_the_serial_runs(cases, monkeypatch):
    """Contract: two systems in one wave over three cases (a task of two runs for each case) give, with two worker
    processes, the answers, statuses and failures of the serial run, systems in wave order, and a log line a run.
    Regression caught: the runs of a task attributed to the wrong system or case when the task's results come back."""
    for name, cls in (("lengths", Lengths), ("steps", Steps)):
        monkeypatch.setitem(baselines.BASELINES, name, SimpleNamespace(cls=cls))
    logs = []
    serial = run_suite(cases, ["lengths", "steps"], CHANNEL)
    pooled = run_suite(cases, ["lengths", "steps"], CHANNEL, workers=2, log=logs.append)
    assert values(pooled) == values(serial) and list(pooled) == ["lengths", "steps"]
    assert sorted(m.split(":")[0] for m in logs) == sorted(f"{n} {c.id}" for n in ("lengths", "steps") for c in cases)
    assert values(serial)["lengths"][0] != values(serial)["steps"][0]   # the two systems answer differently
