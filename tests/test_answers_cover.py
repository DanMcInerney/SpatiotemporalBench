"""The answers to `uncovered`: `table_cover` for the table systems and `baselines_path_cover` for `path_log` (the gold
is `questions_cover`'s, pinned by hand in `test_questions_cover.py`). The expected cells are worked out by hand from
`cover_kit`'s plan of a 3 x 2 m living room (fine-cell arithmetic in its docstring) and, for the end-to-end window,
from the geometry of the sweeps and of a camera that looks north from the floor; what the compiler gives is the gold
only where a test says so."""

from __future__ import annotations

import math
from unittest import mock

import pytest

from answers_kit import compiled_case, scored
from baseline_kit import UNKNOWN, question, record, rows, started
from cover_kit import (
    INSIDE,
    box,
    cleaning_run,
    cleaning_views,
    cover_map,
    plan_run,
    strips,
    world,
)
from cover_kit import UNCOVERED_JOB as JOB
from space_kit import EVENTS
from stbench import questions, questions_world
from stbench.answers import parse_answer, parse_gold
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.questions_cover import uncovered
from stbench.score_set import cells_f1
from stbench.stream import run_case
from stbench.table import Store

ASKED = question("uncovered", {}, answer_type="cells")
NORTH = math.pi / 2
# The plan's room at keyframe 18 (cover_kit's docstring and test_questions_cover's hand work): of the 24 cells wholly
# inside the room, (10,3), (11,3), (14,4) and (15,4) were seen free and not swept.
PLAN = {(10, 3), (11, 3), (14, 4), (15, 4)}


def cells_of(items: list) -> set[tuple[int, int]]:
    return {(i, j) for i, j in items}


def answer_of(cells: set[tuple[int, int]]) -> dict:
    return {"type": "cells", "items": [list(c) for c in sorted(cells)]}


def gold_of(views: dict, a: int = 0, b: int | None = None, k: int = 18) -> dict:
    """The compiler's gold row for `uncovered` at keyframe `k` of the plan's script over `views` (`cover_kit.world`)."""
    (_,), (g,) = questions.records([uncovered(world(plan_run(), views, a=a, b=b), k)])
    return g


def f1(answer: dict, gold: dict) -> float:
    return cells_f1(parse_answer(answer, "cells"), parse_gold(gold["gold"]))


class HandMapper:
    """A mapper that has seen `views` ({keyframe: {fine cell: state}}) as the depth would have shown it."""

    def __init__(self, views: dict):
        self.views = views

    def observe(self, step: dict):
        return self.views.get(step["k"])


def fed(cls, tmp_path, delivered: dict, views: dict, windows, **opts):
    """A started table system of class `cls` over the plan's map, its mapper replaced by `HandMapper(views)`, sent the
    delivered pose rows `windows` (lists of keyframes) in turn, one `task` message before the look window that starts
    with keyframe 13 (the second task: the sweeps are before it, the ask after)."""
    system = started(cls, tmp_path, events=EVENTS, map=cover_map(), **opts)
    system.mapper = HandMapper(views)
    system.store = Store(system.mapper)
    system.task(0, delivered[0]["clock"], 0.0, "")
    for window in windows:
        if window[0] == 13:
            system.task(1, delivered[13]["clock"], 7200.0, "")
        system.observe([delivered[k] for k in window])
    return system


def ask_at(system, delivered: dict, k: int = 18) -> dict:
    return system.ask("during", {"k": k, "clock": delivered[k]["clock"]}, [ASKED])["q1"]


# -- end to end, on real depth ----------------------------------------------------------------------------------
# Two tasks of the vacuum in the plan's room R2. Task 0 sweeps the strips y 1.65 and 1.85 along x 6.0 to 6.5 (keyframes
# 7 to 10: the disc, 0.125 m, reaches the fine rows v 15..19 and the columns u 59..65, so the cell (12,3) is swept in
# all 25 fine cells, and (11,3) and (13,3) in five); task 1 sweeps y 2.15 and 2.35 along the same x (keyframes 35 to 38,
# the rows v 20..24: (12,4) swept, (11,4) and (13,4) in five fine cells each). The brush is off elsewhere. The camera
# (0.13 m ahead of the base, 0.08 m up) sees the floor from 0.1 m to about 1.7 m ahead of it: at keyframe 16 the base
# stands at (6.25, 1.0) and at 17 at (7.25, 1.0), both heading north, so the cells (12,3), (12,4) and (14,3), (14,4),
# 0.37 m or more ahead of the camera and well inside the frame's 40-degree half-width (0.31 m at 0.37 m, against the
# cell's 0.25), are seen whole. Decisions: keyframe 18 (task 0) and 46 (task 1), the base at the room's centre.
SWEEP_0 = [(6.0, 1.65), (6.5, 1.65), (6.5, 1.85), (6.0, 1.85)]
SWEEP_1 = [(6.0, 2.15), (6.5, 2.15), (6.5, 2.35), (6.0, 2.35)]
LOOKS = {16: (6.25, 1.0), 17: (7.25, 1.0)}


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    """The two tasks as the case "c0" (the plan's map in the compiler, the vacuum's real depth rendered at keyframes 16
    and 17), with `(case, questions, golds)` compiled."""
    poses = {**strips(7, SWEEP_0)[0], **strips(35, SWEEP_1)[0], **LOOKS}
    script = cleaning_run(2, (7, SWEEP_0), (35, SWEEP_1), poses=poses, yaws=dict.fromkeys(LOOKS, NORTH),
                          depth_ks=set(LOOKS), boxes=[])
    with mock.patch.object(questions_world, "load_map", lambda name: cover_map()):
        case = compiled_case(tmp_path_factory.mktemp("cover"), "c0", script, JOB, 0, 1)
    return (case, *rows(case))


def test_the_table_reproduces_the_gold(window):
    """Contract (E3c §1): through `run_case` on `detections`, `oracle` and `memory_table` give the gold to the cell at
    both decisions of a two-task window, the mapper's states read from the real depth the stream delivers and the swath
    from every pose and brush status logged since the stream began. Hand-worked from the geometry above: at keyframe
    18 the cell (12,3) was seen free but task 0 swept it, so it is not named, and (12,4), (14,3), (14,4) were seen free
    whole and not swept, so they are; at keyframe 46 task 1 has swept (12,4) whole and (11,4), (13,4) in part
    (don't-care or negative), so the gold is the first one less the cells (11,4), (12,4), (13,4), and (12,3) is still
    not named because the window keeps task 0's sweep. Regression caught: the table and the gold reading the room, the
    swath or the states differently, or a table that sweeps only its current task (it would name (12,3) at keyframe
    46)."""
    case, qs, golds = window
    first, second = (cells_of(g["gold"]["items"]) for g in golds)
    assert [q["decision_k"] for q in qs] == [18, 46]
    assert (12, 3) not in first and {(12, 4), (14, 3), (14, 4)} <= first <= INSIDE
    assert second == first - {(11, 4), (12, 4), (13, 4)} and {(14, 3), (14, 4)} <= second
    for cls in (Oracle, MemoryTable):
        run = run_case(lambda state, work, cls=cls: cls(state, work, events=EVENTS, map=cover_map()), case,
                       channel="detections")
        assert run.status == "complete" and run.failures == {}, cls.__name__
        assert scored(golds, run.answers) == dict.fromkeys((q["qid"] for q in qs), 1.0), cls.__name__
        assert [cells_of(run.answers[q["qid"]]["items"]) for q in qs] == [first, second], cls.__name__


def test_path_log_reads_no_depth_and_claims_what_it_did_not_sweep_in_the_window(window):
    """Contract (E3c §2): through `run_case`, `path_log` answers each decision from its pose and status log and the
    public map alone (the depth and detections of every step are taken out before it sees the step, and counted so the
    strip is not vacuous), and its answer is the room's cells wholly inside and clear of the cabinet with no fine cell
    swept so far, whatever the depth showed. Hand-worked: of the 24 inside cells less the cabinet's (10,4), task 0's
    strips touch (11,3), (12,3), (13,3), so at keyframe 18 it names the other 20; task 1's touch (11,4), (12,4), (13,4),
    so at keyframe 46 it names 17. Both score under the gold's 1.0 (it names the unseen and the occupied alike).
    Regression caught: a `path_log` that reads the depth, or one that forgets the window's earlier sweeps."""
    case, qs, golds = window
    stripped = set()

    class Blindfolded(PathLog):
        def observe(self, steps):
            stripped.update(key for s in steps for key in ("rgb", "depth", "detections") if key in s)
            super().observe([{k: v for k, v in s.items() if k not in ("rgb", "depth", "detections")} for s in steps])

    run = run_case(lambda state, work: Blindfolded(state, work, map=cover_map(), events=EVENTS, prior=[]), case,
                   channel="detections")
    assert stripped == {"depth", "detections"} and run.status == "complete" and run.failures == {}
    claim = INSIDE - {(10, 4), (11, 3), (12, 3), (13, 3)}
    assert [cells_of(run.answers[q["qid"]]["items"]) for q in qs] == [claim, claim - {(11, 4), (12, 4), (13, 4)}]
    assert [len(cells_of(run.answers[q["qid"]]["items"])) for q in qs] == [20, 17]
    assert all(score < 1.0 for score in scored(golds, run.answers).values())


# -- by hand, on the plan's room --------------------------------------------------------------------------------
def test_the_table_names_the_cells_it_saw_free_and_did_not_sweep(tmp_path):
    """Contract (E3c §1): `memory_table`, fed the plan's poses and brush statuses and the free-space states of its two
    views (keyframes 9 and 16: the chair leg, clutter, the box and the floor behind it, the light chair's old place),
    names exactly the plan's four cells (10,3), (11,3), (14,4), (15,4) at keyframe 18, each ingredient of the rule
    counting: the rows j 1 and 2 are swept (negative, don't-care), (12,3) holds a chair leg, (13,3) a box front, (11,4),
    (12,4) and (15,3) clutter (occupied), (14,3) has three fine cells behind the box no frame saw (don't-care), (13,4)
    was never seen, (10,4) touches the fixed cabinet. Its answer is the gold's cells. Regression caught: a table that
    names the cells not swept (the swath alone, F1 0.571) or ignores the layout's fixed pieces or the occupied fine
    cells."""
    views = cleaning_views()
    delivered = world(plan_run(), views).rows
    answer = ask_at(fed(MemoryTable, tmp_path, delivered, views, [range(13), range(13, 19)]), delivered)
    assert answer == answer_of(PLAN)
    assert cells_of(gold_of(views)["gold"]["items"]) == PLAN


def test_current_view_forgets_earlier_sweeps(tmp_path):
    """Contract (E3c §1): `current_view` holds the latest look window alone, so its swath and its states are the look
    window's and it names the cells that were swept before the look window. Hand case on the plan's room: the strips
    (keyframes 7 to 12, task 0) sweep the row j 1 whole; the look window of keyframes 13 to 18 (task 1, brush off) sees
    that row free again and the cells (14,4), (15,4) free. `memory_table`, which keeps the whole log, names only (14,4)
    and (15,4), the gold (F1 1); `current_view` names the six cells (10..15, 1) as well, 8 cells against the 2 of the
    gold, F1 = 2 x 2 / (8 + 2) = 0.4. Regression caught: a memoryless baseline that remembers (it would match
    `memory_table`), or a `memory_table` that sweeps only its latest task (it would name the swept row)."""
    views = {17: {**dict.fromkeys(box(50, 79, 5, 9), "free"), **dict.fromkeys(box(70, 79, 20, 24), "free")}}
    delivered = world(plan_run(), views).rows
    gold = gold_of(views)
    assert cells_of(gold["gold"]["items"]) == {(14, 4), (15, 4)}
    windows = [range(13), range(13, 19)]
    memory = ask_at(fed(MemoryTable, tmp_path / "m", delivered, views, windows), delivered)
    current = ask_at(fed(CurrentView, tmp_path / "c", delivered, views, windows), delivered)
    assert memory == answer_of({(14, 4), (15, 4)}) and f1(memory, gold) == 1.0
    assert current == answer_of({(i, 1) for i in range(10, 16)} | {(14, 4), (15, 4)})
    assert f1(current, gold) == pytest.approx(0.4)


def test_path_log_claims_what_it_did_not_sweep(tmp_path):
    """Contract (E3c §2, plan §5.1 N11): from its pose and status log and the public map alone, `path_log` names every
    cell of the room wholly inside and clear of the fixed cabinet with no fine cell swept, so on the plan's room
    (clutter, a chair leg, a box with an area behind it no frame saw) it cannot beat 0.6. Hand-worked: the strips sweep
    the rows j 1 and 2, the cabinet touches (10,4), so it names the rows j 3 and 4 less (10,4): 6 + 5 = 11 cells, among
    which (14,3) is the gold's don't-care and drops from both sides: 10 cells against the gold's 4, all four among
    them, F1 = 2 x 4 / (10 + 4) = 0.571, where `memory_table` scores 1. Regression caught: a `path_log` stronger than
    the agent it models (it would read the clutter) or weaker (it would miss the map's cabinet)."""
    views = cleaning_views()
    delivered = world(plan_run(), views).rows
    gold = gold_of(views)
    log = started(PathLog, tmp_path / "p", map=cover_map(), events=EVENTS, prior=[])
    log.observe([delivered[k] for k in range(19)])
    claim = ask_at(log, delivered)
    memory = ask_at(fed(MemoryTable, tmp_path / "m", delivered, views, [range(13), range(13, 19)]), delivered)
    assert claim == answer_of({(i, j) for i in range(10, 16) for j in (3, 4)} - {(10, 4)})
    assert f1(claim, gold) == pytest.approx(8 / 14) and f1(claim, gold) <= 0.6
    assert f1(memory, gold) == 1.0


def test_path_log_answers_from_its_pool_when_the_pool_knows_better(tmp_path):
    """Contract (E3c §2): `path_log`'s answer is the best response over the pool its prior gives at its first supported
    key, its own claim one more candidate, so a prior that knows the room never leaves it weaker than `blind` (VISION
    rule 5). Hand case on the plan's room at keyframe 18, a prior of five records from three other scripts at the same
    phase and pose: with golds of the plan's four cells the best candidate is those four (F1 1 against the claim's
    0.571); with five empty golds (rooms nothing was left in) it is the empty set (1 against 0). Regression caught: a
    `path_log` that answers its claim whatever the prior says."""
    views = cleaning_views()
    delivered = world(plan_run(), views).rows
    gold = gold_of(views)
    base, yaw = tuple(delivered[18]["pose"]["pos"][:2]), delivered[18]["pose"]["yaw_deg"]
    answers = {}
    for name, items in (("plan", gold["gold"]["items"]), ("empty", [])):
        prior = [{**record(ASKED, gold["gold"], script=f"s{i % 3}", when="during", phase="cleaning", base=base,
                           yaw=yaw), "gold": {**gold, "gold": {**gold["gold"], "items": items, "dont_care": []}}}
                 for i in range(5)]
        log = started(PathLog, tmp_path / name, map=cover_map(), events=EVENTS, prior=prior)
        log.observe([delivered[k] for k in range(19)])
        answers[name] = ask_at(log, delivered)
    assert answers == {"plan": answer_of(PLAN), "empty": answer_of(set())}


def test_a_base_in_no_room_is_unknown_and_a_table_needs_the_map(tmp_path):
    """Contract (E3c §1, §2): a base that stands in no room (the doorway between the kitchen's east edge at x 4.5 and
    the living room's at x 5, which the harness never asks about) is answered `unknown` by the table and by `path_log`
    rather than guessed, and a table system built without the layout's map says so instead of failing on a missing
    key. Hand case: the plan's keyframe 18 with the base moved to (4.75, 1.5). Regression caught: a room picked for a
    robot standing between two, or an answer about a room no map named."""
    views = cleaning_views()
    delivered = world(plan_run(), views).rows
    delivered[18] = {**delivered[18], "pose": {**delivered[18]["pose"], "pos": [4.75, 1.5, 0.0]}}
    table = fed(MemoryTable, tmp_path / "m", delivered, views, [range(13), range(13, 19)])
    log = started(PathLog, tmp_path / "p", map=cover_map(), events=EVENTS, prior=[])
    log.observe([delivered[k] for k in range(19)])
    assert ask_at(table, delivered) == UNKNOWN and ask_at(log, delivered) == UNKNOWN
    bare = started(MemoryTable, tmp_path / "b", events=EVENTS)
    bare.observe([delivered[k] for k in range(19)])
    with pytest.raises(ValueError, match="needs the case layout's public map"):
        ask_at(bare, delivered)
