"""`path_log`'s answer to `open_heading` (`baselines_path_heading`): the heading of the longest straight run over the
robot's own swept path, never the public map or a pool of other scripts' golds (those are `map_reader`'s). Hand cases on
the vacuum (0.34 m disc, front edge 0.17 m from its base) heading +x, cap 3 m, as `test_path_log_clear.py`. The rule is
asked of an `in_pool` prior (`KEEP`), which keeps it unasked: whether it stands on other scripts is
`test_path_log_heading_kept`'s contract."""

from __future__ import annotations

import math

from baseline_kit import EVENTS, question, record, started, step
from stbench import baselines_path_heading, grid, mapspec
from stbench.baselines_path import PathLog
from stbench.baselines_path_heading import own_run
from stbench.blind import Prior

KEEP = Prior((), in_pool=True)
DRIVE = ((0.5, 1.5), (1.5, 1.5), (2.5, 1.5))


def heading(deg: float) -> dict:
    return {"type": "heading", "deg_left": deg}


def ask(tmp_path, positions, prior=KEEP, map_=None):
    """What `path_log` answers to `open_heading` (cap 3) after driving the base positions in the phase 'transit',
    heading +x, on `map_` (the committed apartment by default)."""
    steps = [step(k, x, y, phase="transit") for k, (x, y) in enumerate(positions)]
    system = started(PathLog, tmp_path, map=map_ or mapspec.load_map("apartment_a"), events=EVENTS, prior=prior)
    system.observe(steps)
    q = question("open_heading", {"cap": 3.0}, answer_type="heading")
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, [q])["q1"]


def test_the_heading_is_where_the_robot_has_driven_in_a_line(tmp_path):
    """Contract: the answer is the heading of the longest unbroken run over the cells the body swept, from the turned
    body's edge. Hand case: the base drove (0.5, 1.5), (1.5, 1.5), (2.5, 1.5) and stands at the east end heading +x
    (the swath is the cells with centres within 0.17 m of the segment, so x 0.35 to 2.65 on the row of centre y
    1.55). Ahead the first sample past the front edge (2.67) is at 2.72, an unswept cell: run 0; left and right the
    first samples at 0.05 from the sides, 0.22 from the row's centre line, are unswept; behind, from the rear edge
    (2.33), the samples 2.28, 2.18 ... 0.38 are swept (20 cells) and 0.28 is not: the run is 2.0. The answer is 180
    degrees left. Regression caught: a run measured from the base, the longest run not chosen, or a heading in the
    wrong sign or frame."""
    assert ask(tmp_path, DRIVE) == heading(180.0)


def test_equal_runs_go_to_the_heading_nearest_ahead_and_the_left_one_of_two_as_near(tmp_path, monkeypatch):
    """Contract: of equal longest runs the heading nearest the robot's own wins, the left one when two are as near.
    The runs are set by hand (a run of 1.0 m at the headings named, 0.5 m at every other, the robot heading +x):
    at 90 and 270 degrees left they are as near as each other, so 90 wins; at 270 and 100 the right turn of 90 is nearer
    than a left turn of 100, so -90 wins; at 100 and 180 it is 100. Regression caught: ties broken by index from the
    other end, or toward the farthest-ahead heading."""
    def runs_at(*degrees):
        monkeypatch.setattr(baselines_path_heading, "own_run",
                            lambda swept, start, h, cap: 1.0 if round(math.degrees(h)) % 360 in degrees else 0.5)
        return ask(tmp_path / str(degrees), ((0.5, 1.5), (1.5, 1.5)))

    assert runs_at(90, 270) == heading(90.0)
    assert runs_at(270, 100) == heading(-90.0)
    assert runs_at(100, 180) == heading(100.0)


def test_a_run_is_the_swept_cells_unbroken_from_the_edge():
    """Contract: `own_run` walks the corridor's axis from the edge a fine cell at a time, sampled at 0.05, 0.15 ... m,
    and ends at the first unswept cell: the run is the last swept sample plus 0.05, 0 when the first is unswept, at
    most the cap. Hand case: the fine cells (17, 15), (18, 15), (19, 15) swept, the edge at (1.65, 1.55) heading +x: the
    samples at x 1.70, 1.80, 1.90 are swept and 2.00 (cell 20) is not: the run is 0.3; the same edge turned the other
    way starts in cell (15, 15), unswept: 0; with 4 cells swept and a cap of 0.3 the run is 0.3. Regression caught: a
    run counted past a gap, or from the wrong end of the first cell."""
    swept = {(17, 15), (18, 15), (19, 15)}
    assert own_run(swept, (1.65, 1.55), 0.0, 3.0) == 0.3
    assert own_run(swept, (1.65, 1.55), math.pi, 3.0) == 0.0
    assert own_run(swept | {(20, 15)}, (1.65, 1.55), 0.0, 0.3) == 0.3


def test_path_log_never_reads_the_map_or_a_pool_for_open_heading(tmp_path, monkeypatch):
    """Contract (rule 5): `open_heading` is a structure family: what the public map's raycast or a pose-keyed pool of
    other scripts' golds would say is `map_reader`'s, a reference, not a gated shortcut's. Hand case: the drive of the
    first test, whose own run gives 180, on a map with walls and with a pool of five matching records whose golds are
    the heading 90 (and that carry no `own`): the pool's best response would be 90. Kept unasked (`in_pool`) the rule
    answers 180; asked (a plain prior) it has no evidence and answers `unknown`: never the pool's 90, and the raycast is
    never called. Regression caught: the raycast or the pose pool inside a gated shortcut (it would pass the own-log
    gate with a layout it learned)."""
    calls = []
    monkeypatch.setattr(grid, "raycast", lambda *a: calls.append(a) or 0.0)
    q = question("open_heading", {"cap": 3.0}, answer_type="heading")
    pool = [record(q, heading(90.0), script=f"s{i}", phase="transit", base=(2.5, 1.5), when="during")
            for i in range(5)]
    assert ask(tmp_path / "a", DRIVE) == heading(180.0)
    assert ask(tmp_path / "b", DRIVE, prior=Prior(pool, in_pool=True)) == heading(180.0)
    assert ask(tmp_path / "c", DRIVE, prior=Prior(pool)) == {"type": "unknown"}
    assert calls == []
