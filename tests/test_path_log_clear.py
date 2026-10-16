"""`path_log`'s answer to `clear_path` (`baselines_path_clear`): the robot's own swept path and its detours, never the
public map or a pool of other scripts' golds (those are `map_reader`'s). Hand cases on the vacuum (0.34 m disc, front
edge 0.17 m
ahead of its base) heading +x, as `test_path_log.py`'s detour."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, number, question, record, started, step
from stbench import grid, mapspec
from stbench.baselines_path import PathLog
from stbench.blind import Prior
from stbench.questions_words import clear_slots

UNKNOWN = {"type": "unknown"}


def ask(tmp_path, positions, direction="ahead", prior=(), map_=None):
    """What `path_log` on `map_` (the committed apartment by default) answers to `clear_path` (cap 3) after driving the
    base positions in the phase 'transit', heading +x."""
    steps = [step(k, x, y, phase="transit") for k, (x, y) in enumerate(positions)]
    system = started(PathLog, tmp_path, map=map_ or mapspec.load_map("apartment_a"), events=EVENTS, prior=prior)
    system.observe(steps)
    q = question("clear_path", clear_slots("drive", direction, 3.0), answer_type="number")
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, [q])["q1"]


def evidence(own: float, gold: float, scripts: int = 4, each: int = 2) -> Prior:
    """A prior of other scripts' `clear_path` records (`each` a script) that all carry `path_log`'s own-log answer `own`
    (a number) and the gold `gold`: the evidence `kept` weighs the rule by."""
    q = question("clear_path", clear_slots("drive", "ahead", 3.0), answer_type="number")
    return Prior({**record(q, number(gold), script=f"s{i}", phase="transit", when="during"), "own": number(own)}
                 for i in range(scripts) for _ in range(each))


RIGHT = evidence(0.6, 0.6)   # the rule's answers are the golds: it beats abstaining on every script
BENT = [(3.5, 1.5), (2.0, 1.5), (1.9, 0.9), (1.2, 0.9), (1.1, 1.5), (0.5, 1.5)]


def test_path_log_answers_clear_path_from_its_detour(tmp_path):
    """Contract (review 2.3): the robot's own path is evidence of what stood in the corridor: the base ends at (0.5,
    1.5) heading +x (front edge 0.67) after coming from the east through (3.5, 1.5), (2.0, 1.5), (1.9, 0.9), (1.2, 0.9),
    (1.1, 1.5) round a box at x 1.5-1.7: the axis cells at 0.72 to 1.22 are swept, those at 1.32 to 1.72 are not and
    1.82 on are again, so the first unswept axis point is at s = 0.65 from the edge and the answer is 0.6, as
    `path_ahead`'s. Regression caught: a family a detour gives away being refused its own-log answer."""
    assert ask(tmp_path, BENT, prior=RIGHT) == number(0.6)


def test_the_detour_stands_only_where_it_beats_abstaining_on_other_scripts(tmp_path):
    """Contract (the rule for every answer a shortcut takes with other scripts' records in hand): the
    own-path rule answers only where the lower end of the 95% interval of its gain over abstaining, over the scripts of
    the prior, is above 0. Hand case: the bent path of the test above, whose detour reads 0.6. A prior whose records
    carry that answer (`own`) and a gold of 0.6 every time (4 scripts, 8 records) gains 1 a record and stands. One whose
    golds are 2.5 loses a point a record (a wrong number scores -1, `unknown` 0): `unknown`, as for a prior of too few
    records (4 records of 2 scripts, under `blind.MIN_POOL_SCRIPTS`) or none; and an `in_pool` prior, in-sample by
    design, keeps it unasked. Regression caught: a detour answered at a loss (the helper's `clear_path` -0.20, below
    `all_unknown`, BROKEN), or a rule judged on a prior too thin to judge by."""
    wrong = evidence(0.6, 2.5)
    for name, prior, want in (("right", RIGHT, number(0.6)), ("wrong", wrong, UNKNOWN),
                              ("thin", evidence(0.6, 0.6, scripts=2), UNKNOWN), ("none", [], UNKNOWN),
                              ("in_pool", Prior(wrong, in_pool=True), number(0.6))):
        assert ask(tmp_path / name, BENT, prior=prior) == want, name


def test_path_log_never_reads_the_map_or_a_pool_for_clear_path(tmp_path, monkeypatch):
    """Contract (review 2.3, rule 5): `clear_path` is a structure family: what the public map's raycast or a pose-keyed
    pool of other scripts' golds would say is `map_reader`'s, a reference, not a gated shortcut's. After a straight
    drive there is no detour, so `path_log` answers `unknown` on a map with a wall in the corridor (where `path_ahead`
    answers a number) and with a pool of five matching records whose golds are a number: the raycast is never called and
    the pool never consulted. Regression caught: the raycast or the pose pool inside a gated shortcut (it would pass the
    own-log gate with a layout it learned)."""
    straight = [(3.5, 1.5), (0.5, 1.5)]
    calls = []
    monkeypatch.setattr(grid, "raycast", lambda *a: calls.append(a) or 0.0)
    q = question("clear_path", clear_slots("drive", "ahead", 3.0), answer_type="number")
    pool = [record(q, number(2.0), script=f"s{i}", phase="transit", base=(0.5, 1.5), when="during") for i in range(5)]
    assert ask(tmp_path / "a", straight) == UNKNOWN
    assert ask(tmp_path / "b", straight, prior=pool) == UNKNOWN
    assert calls == []


@pytest.mark.parametrize("direction, want", [("ahead", UNKNOWN), ("back", number(0.5))])
def test_the_corridor_starts_at_the_edge_the_direction_names(tmp_path, direction, want):
    """Contract: a corridor `back` starts at the rear edge and runs against the heading, `ahead` at the front edge; on
    one swept path they read different cells. Hand case, heading +x throughout (the vacuum's swath is cells with centres
    within 0.17 of a segment): the base drove (0.5, 1.5), (1.5, 1.5), (1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (3.0, 1.5).
    Its rear edge is x 2.83 and the axis samples west of it fall in the fine cells centred 2.75, 2.65, ... (row centre y
    1.55), swept down to 2.35 (0.158 from (2.5, 1.5)), unswept at 2.25, swept again at 1.65 and 1.55 (from the leg at x
    1.5): the first unswept sample is the sixth, s = 0.55, so the answer is 0.5. Ahead of the front edge (3.17) the
    first cell is swept (0.158 from the end) and nothing beyond is: no gap with a swept cell after it, `unknown`.
    Regression caught: a back corridor read from the front edge, or a detour answered without its swept cell beyond."""
    path = [(0.5, 1.5), (1.5, 1.5), (1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (3.0, 1.5)]
    assert ask(tmp_path, path, direction, prior=evidence(0.5, 0.5)) == want
