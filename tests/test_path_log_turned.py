"""`path_log`'s answer to a turned `clear_path` (0.9.0, `baselines_path_clear`): the detour along the moved heading over
the robot's own swath, never the public map or a pool. Hand cases on the vacuum (a 0.34 m disc: its front edge 0.17 m
from its base and its swath the fine cells whose centres lie within 0.17 m of a segment of its path), heading +x
throughout. The rule is asked of an `in_pool` prior (`KEEP`), which keeps it unasked: whether it stands on other
scripts is `test_path_log_clear`'s contract."""

from __future__ import annotations

from baseline_kit import EVENTS, number, question, record, started, step
from stbench import grid, mapspec
from stbench.baselines_path import PathLog
from stbench.blind import Prior
from stbench.questions_words import clear_slots

UNKNOWN = {"type": "unknown"}
# The robot came from the north (1.0, 2.0), went (1.0, 1.15), swung west to (0.5, 0.8), came back to (1.0, 0.5) and ends
# at (1.0, 0.0), heading +x all the way.
AROUND = [(1.0, 2.0), (1.0, 1.15), (0.5, 0.8), (1.0, 0.5), (1.0, 0.0)]
KEEP = Prior((), in_pool=True)


def mirrored(path):
    return [(x, -y) for x, y in path]


def ask(tmp_path, positions, direction, prior=KEEP):
    steps = [step(k, x, y, phase="transit") for k, (x, y) in enumerate(positions)]
    system = started(PathLog, tmp_path, map=mapspec.load_map("apartment_a"), events=EVENTS, prior=prior)
    system.observe(steps)
    q = question("clear_path", clear_slots("drive", direction, 3.0), answer_type="number")
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, [q])["q1"]


def test_a_left_corridor_runs_north_from_the_turned_front_edge_over_the_robots_own_swath(tmp_path):
    """Contract (brief 4.1): `path_log` reads the moved corridor over its own swath: left of a robot heading +x is
    north, from the front edge on that heading, (1.0, 0.17). Hand case: the base ends at (1.0, 0.0) after coming round a
    thing standing about (1.05, 0.8). The axis samples are the fine cells of y 0.2 to 0.3, 0.3 to 0.4, ... in column x
    1.0 to 1.1 (centres 1.05); the segment (1.0, 0.5) to (1.0, 0.0) sweeps those up to y 0.6 to 0.7 (centre (1.05, 0.65)
    is 0.158 from (1.0, 0.5)), not the one at 0.7 to 0.8 (0.255 from it, 0.24 from the leg west of it), nor those at 0.8
    to 1.0, and the cell at y 1.0 to 1.1 is 0.11 from the swing (1.0, 1.15): the first unswept sample is the fine cell
    at 0.7 to 0.8, 0.55 from the start, so the answer is 0.5. Ahead, right and back start on cells the path never swept:
    `unknown`. Regression caught: a left read on the heading instead of the moved heading (unknown), or a left started
    from the base or the rear edge."""
    assert ask(tmp_path / "a", AROUND, "left") == number(0.5)
    for direction in ("ahead", "right", "back"):
        assert ask(tmp_path / direction, AROUND, direction) == UNKNOWN, direction


def test_a_right_corridor_runs_south_the_same_way(tmp_path):
    """Contract: right is the mirror of left: the path reflected in the x axis (y negated) gives the same 0.5 right and
    `unknown` left. Regression caught: left and right swapped, or one of them read on the wrong side of the heading."""
    assert ask(tmp_path / "b", mirrored(AROUND), "right") == number(0.5)
    assert ask(tmp_path / "c", mirrored(AROUND), "left") == UNKNOWN


def test_path_log_never_reads_the_map_or_a_pool_for_a_turned_corridor(tmp_path, monkeypatch):
    """Contract (rule 5): a turned corridor is a structure question too, so with no detour (a straight drive) `path_log`
    answers `unknown` for left and right on a map with a wall in the way and with a pool of five matching records whose
    golds are numbers: the raycast is never called. Regression caught: the raycast or the pose pool inside a gated
    shortcut for the new directions."""
    straight = [(1.0, 3.0), (1.0, 0.0)]
    calls = []
    monkeypatch.setattr(grid, "raycast", lambda *a: calls.append(a) or 0.0)
    for direction in ("left", "right"):
        q = question("clear_path", clear_slots("drive", direction, 3.0), answer_type="number")
        pool = [record(q, number(2.0), script=f"s{i}", phase="transit", base=(1.0, 0.0), when="during")
                for i in range(5)]
        assert ask(tmp_path / f"{direction}-a", straight, direction) == UNKNOWN
        assert ask(tmp_path / f"{direction}-b", straight, direction, prior=pool) == UNKNOWN
    assert calls == []

