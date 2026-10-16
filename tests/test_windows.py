"""`case.windows` with the starts a job allows (no window begins at a `resume`), and `build.write_cases` passing them.
"""

from __future__ import annotations

import pytest

import job_kit
from stbench import build, case
from stbench import episode as ep
from stbench.seeding import make_rng

pytestmark = pytest.mark.xdist_group(name="test_job_vacuum2")  # the scripts of `test_job_vacuum2` are built once


class Low:
    """An rng whose `integers(lo, hi)` returns `lo`, and that records its calls."""

    def __init__(self):
        self.calls: list[tuple] = []

    def integers(self, *args):
        self.calls.append(args)
        return args[0]


def test_windows_without_starts_draw_todays_windows_and_with_every_start_the_same_calls():
    """Contract: with `starts` None `windows` makes exactly today's draws (a one-task script keeps its window), and with
    every task allowed it makes the same `rng.integers` calls, so `write_cases` may always pass the starts. Hand case, 6
    tasks and an rng that always returns the low end: short draws a size in [1, 1] and a start in [0, 6), medium a size
    in [2, 3] (its top is min(3, 6)) and a start in [0, 5), long a size in [4, 6] and a start in [0, 3), so the windows
    are (0, 0, short), (0, 1, medium) and (0, 3, long) and the calls (1, 2), (0, 6), (2, 4), (0, 5), (4, 7), (0, 3). A
    one-task script is [(0, 0, "short")]. Regression caught: windows redrawn for scripts that have no resume, or a start
    draw that differs in form."""
    plain, every = Low(), Low()
    want = [(0, 0, "short"), (0, 1, "medium"), (0, 3, "long")]
    assert case.windows(6, plain) == want and case.windows(6, every, range(6)) == want
    assert plain.calls == every.calls == [(1, 2), (0, 6), (2, 4), (0, 5), (4, 7), (0, 3)]
    assert case.windows(1, make_rng("windows", 1)) == case.windows(1, make_rng("windows", 1), [0]) == [(0, 0, "short")]
    for seed in range(50):  # the real stream agrees too
        assert case.windows(6, make_rng("windows", seed)) == case.windows(6, make_rng("windows", seed), range(6))


def test_no_window_starts_at_a_resume_task():
    """Contract (plan §5.1): over 1,000 draws of a 6-task script whose tasks 1 and 4 are resumes (allowed starts 0, 2, 3
    and 5), `windows` never starts outside the allowed starts and draws every (size, first task) pair that fits: short
    (1, 0), (1, 2), (1, 3), (1, 5); medium size 2 or 3 from 0, 2 or 3; long size 4 from 0 or 2, sizes 5 and 6 from 0
    (a window may run over a resume, only not begin at one). A length no allowed start fits is left out: a script whose
    only allowed start is its last task has a short window and no other, and none has no window. Regression caught: a
    window that reveals the task before it by beginning at the resume."""
    seen: dict[str, set[tuple[int, int]]] = {"short": set(), "medium": set(), "long": set()}
    for i in range(1000):
        for a, b, length in case.windows(6, make_rng("test_windows", i), [0, 2, 3, 5]):
            assert a in (0, 2, 3, 5) and b < 6
            seen[length].add((b - a + 1, a))
    assert seen == {"short": {(1, 0), (1, 2), (1, 3), (1, 5)},
                    "medium": {(2, 0), (2, 2), (2, 3), (3, 0), (3, 2), (3, 3)},
                    "long": {(4, 0), (4, 2), (5, 0), (6, 0)}}
    assert case.windows(6, make_rng("test_windows", 0), [5]) == [(5, 5, "short")]
    assert case.windows(6, make_rng("test_windows", 0), []) == []


def test_write_cases_writes_no_case_that_starts_at_a_resume(tmp_path):
    """Contract: `build.write_cases` passes each script's allowed starts (the tasks whose kind its job does not
    `continue`), so on scripts of seeds 101-104 (a stop and its resume in each of three of them) every case it writes
    begins at a `clean_rooms` task, three cases a script (short, medium, long), and the cases are the same on a second
    run. Regression caught: a case that begins at a resume, which would show the robot that earlier tasks exist."""
    resumes = {}
    for seed in (101, 102, 103, 104):
        _, tl = job_kit.generated2(seed)
        sid = ep.write_script(tmp_path, tl).id
        resumes[sid] = {t.index for t in tl.ego.tasks if t.plan.kind == "resume"}
    assert sum(bool(r) for r in resumes.values()) >= 3
    ids = build.write_cases(tmp_path)
    cases = [e for i, e in case.read_index(tmp_path).items() if e["kind"] == "case"]
    assert len(ids) == len(cases) == 12 and build.write_cases(tmp_path) == ids
    for entry in cases:
        assert entry["a"] not in resumes[entry["script"]]
    assert {e["length"] for e in cases} == {"short", "medium", "long"}
