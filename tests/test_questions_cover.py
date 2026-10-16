"""`uncovered` (questions_cover): the cells of the room the robot is leaving that it saw as open floor and did not
clean. Every expected cell is worked out by hand from `cover_kit`'s plan of the room (fine-cell arithmetic in its
docstring)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cover_kit import (
    INSIDE,
    RING,
    box,
    cleaning_run,
    cleaning_views,
    cover_map,
    plan_run,
    world,
)
from stbench import grid_cover, questionnaire, questions
from stbench.answers import Cells, parse_gold
from stbench.questions_cover import uncovered
from stbench.questions_world import World
from stbench.score_set import cells_f1

# The plan's room at keyframe 18, by hand. Rows of cells: j 1 (y 0.5..1.0) all 25 fine cells swept, negative; j 2 (y 1.0
# ..1.5) fine rows v 10..13 swept, 20 of 25: don't-care; j 3 and j 4 unswept: (10,3), (11,3) free; (12,3) a chair leg
# and (13,3) a box front, occupied; (14,3) three fine cells behind the box unseen: don't-care; (15,3) clutter;
# (10,4) the cabinet: don't-care (a fixed piece touches it); (11,4) and (12,4) clutter; (13,4) never seen; (14,4) and
# (15,4) free (the chair's old place among them).
PLAN_POSITIVE = {(10, 3), (11, 3), (14, 4), (15, 4)}
PLAN_DONT_CARE = RING | {(i, 2) for i in range(10, 16)} | {(14, 3), (10, 4)}


def cells(listed: list[list[int]]) -> set[tuple[int, int]]:
    return {(i, j) for i, j in listed}


def test_the_swath_alone_does_not_answer():
    """Contract (plan §5.1, N11): the gold is the cells seen free and not swept, not the cells not swept: on the
    plan's room (clutter, a chair leg, a box with an area behind it no frame saw) the answer a log of the robot's own
    path gives, every inside, unblocked cell with no swept fine cell, scores F1 at most 0.6 against the gold, and the
    gold's own items score 1. Hand-worked: the swath-only answer is j 3 and j 4 less the cabinet's (10,4): 6 + 5 = 11
    cells, of which (14,3) is the gold's don't-care and drops from both sides: 10 cells against the 4 positives
    (10,3), (11,3), (14,4), (15,4), all four among them: F1 = 2 x 4 / (10 + 4) = 0.571. The row is the compiler's:
    text, `cells`, scope `memory`, credit f1, no tolerance, subjects the decision's time and room, evidence the
    keyframes that showed the cells (9 and 16) with the decision's own (18). Regression caught: a family `path_log`
    answers from its own path (N11), or a gold that names a cell perception could not judge."""
    w = world(plan_run(), cleaning_views())
    c = uncovered(w, 18)
    assert cells(c.gold["items"]) == PLAN_POSITIVE and PLAN_POSITIVE <= INSIDE
    assert cells(c.gold["dont_care"]) == PLAN_DONT_CARE
    assert (c.family, c.slots, c.slice, c.decision_k, c.evidence_k) == ("uncovered", {}, "answerable", 18, (9, 16, 18))
    assert (c.subjects, c.critical, c.tolerances, c.entities, c.region) == (("w:18", "r:R2"), True, {}, (), None)
    swath_only = {(i, j) for i in range(10, 16) for j in (3, 4)} - {(10, 4)}
    swept = grid_cover.brush_swath([(*w.base(k), w.yaw(k), bool(w.rows[k]["status"]["brush"])) for k in range(19)],
                                   grid_cover.brush_of(w.robot))
    open_cells = grid_cover.open_cells(w.map.region("R2").polygon, swept, grid_cover.blocked(w.map, w.robot))
    assert set(open_cells) == swath_only
    gold = parse_gold(c.gold)
    assert cells_f1(Cells(frozenset(swath_only)), gold) == pytest.approx(8 / 14)
    assert cells_f1(Cells(frozenset(map(tuple, c.gold["items"]))), gold) == 1.0
    (q,), (g,) = questions.records([c])
    assert q["text"] == "Which cells of the room you are leaving did you see as open floor but not clean?"
    assert (q["answer_type"], q["needs"], q["slots"], g["scope"], g["credit"], g["metric"]) == (
        "cells", ["see", "pose", "status"], {}, "memory", "f1", None)


def test_a_hidden_move_changes_no_gold():
    """Contract (plan §5.1, truth rule 2): no light piece's true footprint enters the gold, so a piece moved where no
    frame saw it, the depth unchanged, changes nothing. Hand-worked on the plan's room: the map's chair stands at
    (7.25, 2.25) in cell (14,4), where the floor was seen free (it left before the view); in the script it never
    leaves, or moves at t 10 s to (5.75, 1.75), in the positive cell (11,3), with no frame ever showing it: both
    scripts give the plan's gold, (14,4) and (11,3) still named. Regression caught: a gold that reads a light piece's
    true footprint, which would make (14,4) or (11,3) don't-care (and make the answer a function of a move nobody
    saw)."""
    still = plan_run()
    still.item("L01", "chair", {"color": "oak"}, [(0, None, (7.25, 2.25, 0.0))], mobility="light", size=(0.4, 0.4, 0.9))
    moved = plan_run()
    moved.item("L01", "chair", {"color": "oak"}, [(0, 10.0, (7.25, 2.25, 0.0)), (10.0, None, (5.75, 1.75, 0.0))],
               mobility="light", size=(0.4, 0.4, 0.9))
    golds = [uncovered(world(s, cleaning_views()), 18).gold for s in (still, moved)]
    assert golds[0] == golds[1]
    assert {(14, 4), (11, 3)} <= cells(golds[0]["items"]) and cells(golds[0]["items"]) == PLAN_POSITIVE


def test_uncovered_counts_every_run_of_the_window():
    """Contract (plan §5.1, V13): the sweeps counted are those of the whole window, so a room half cleaned in an
    earlier task and finished in the next is covered by both. Hand-worked on a 2-task window, the floor all free at
    keyframe 20: task 0 runs two strips (y 1.05 and 1.35, fine rows v 9..14) along x 5.05 to 6.45 and task 1
    (keyframes 35 on) the same along x 6.45 to 7.95; at task 1's `room_done` (keyframe 46) j 2 is fully swept in all
    six cells (task 0's cells 10..12, task 1's 13..15): negative; j 1 has its row v 9 swept (5 of 25): don't-care; j 3
    and j 4 are unswept and free: the 12 cells of those rows less the cabinet's (10,4), don't-care: 11 named. Task 1's
    strips alone would name (10,1), (11,1), (10,2) and (11,2). Regression caught: a gold that forgets the window's
    earlier coverage."""
    first = [(5.05, 1.05), (6.45, 1.05), (6.45, 1.35), (5.05, 1.35)]
    second = [(6.45, 1.05), (7.95, 1.05), (7.95, 1.35), (6.45, 1.35)]
    s = cleaning_run(2, (7, first), (35, second))
    views = {20: dict.fromkeys(box(50, 79, 5, 24), "free")}
    c = uncovered(world(s, views, a=0, b=1, length="medium"), 46)
    assert cells(c.gold["items"]) == {(i, j) for i in range(10, 16) for j in (3, 4)} - {(10, 4)}
    assert cells(c.gold["dont_care"]) == RING | {(i, 1) for i in range(10, 16)} | {(10, 4)}
    assert c.evidence_k == (20, 46)
    seen_again = {40: views[20]}  # the same floor seen in task 1, for a window that holds task 1 alone
    alone = uncovered(world(s, seen_again, a=1, b=1), 46)  # which knows nothing of task 0's strips
    assert {(10, 1), (11, 1), (10, 2), (11, 2)} <= cells(alone.gold["items"])


def test_a_doorway_is_a_dead_band():
    """Contract: the room is the public map's region that holds the delivered base at the decision, and a base in none
    (a doorway, between the kitchen's east edge at x 4.5 and the living room's at x 5) gives no question: the point is
    drawn and not asked. Hand-worked on the one decision the job asks (keyframe 18, 18 steps into the task): with the
    base at (4.75, 1.5) `uncovered` is None and the draw asks nothing; at (6, 1.5) it asks one question about R2; at
    (2, 1.5) it asks about the kitchen R1. Regression caught: a room picked for a robot standing between two."""
    doorway, home, kitchen = (plan_run(poses={18: xy}) for xy in ((4.75, 1.5), (6.0, 1.5), (2.0, 1.5)))
    w = world(doorway, cleaning_views())
    assert uncovered(w, 18) is None and questionnaire.during(w) == []
    asked = questionnaire.during(world(home, cleaning_views()))
    assert [(c.decision_k, c.subjects) for c in asked] == [(18, ("w:18", "r:R2"))]
    assert uncovered(world(kitchen, cleaning_views()), 18).subjects == ("w:18", "r:R1")


def test_a_room_no_cell_of_which_could_be_named_is_not_asked():
    """Contract (VISION rule 5): `uncovered` is not asked in a room in which no 0.5 m cell lies wholly inside and clear
    of every blocked rectangle, since its gold there is the empty set whatever the robot did (the apartment's hall).
    Hand case: the living room R2 narrowed to the strip x 5.05..5.45, y 0.5..2.5, with the base at (5.25, 1.5) at the
    decision: every cell of the strip's column i = 10 spans x 5.0..5.5, so none is wholly inside, and the question is
    None; the full R2 (x 5..8) asks it. Regression caught: a question whose answer the room's name gives, which `blind`
    would learn."""
    narrow = replace(cover_map(), regions=tuple(
        replace(r, polygon=((5.05, 0.5), (5.45, 0.5), (5.45, 2.5), (5.05, 2.5))) if r.id == "R2" else r
        for r in cover_map().regions))
    w = world(plan_run(poses={18: (5.25, 1.5)}), cleaning_views())
    assert w.map.region_of((5.25, 1.5)) == "R2" and uncovered(w, 18) is not None
    narrowed = World(replace(w.inp, map=narrow), w.rec)
    assert narrowed.map.region_of((5.25, 1.5)) == "R2" and uncovered(narrowed, 18) is None


def test_a_robot_without_a_brush_raises():
    """Contract: the swath is the swath of the robot's tool named `brush`, and a robot with none cannot be asked: the
    family raises rather than guessing a footprint. Regression caught: a family asked of a robot it cannot read, which
    would answer from a made-up footprint."""
    w = world(plan_run(), cleaning_views())
    assert grid_cover.brush_of(w.robot).size_m == (0.25, 0.25)
    bare = world(plan_run(), cleaning_views(), robot=replace(w.robot, tools=()))
    with pytest.raises(ValueError, match="no tool named 'brush'"):
        uncovered(bare, 18)
