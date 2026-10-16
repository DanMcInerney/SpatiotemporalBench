"""The resume (`vacuum/2`'s `resume` task, which finishes the room a stopped task left): where the stop falls, what the
stopped task's brush left unswept, and the strips a resumed room runs over only that, with the optional mask of
`coverage.strips`. Hand-made rooms for the units; seeds 101-140 (the plan's draw) for the scripts."""

from __future__ import annotations

import math

import numpy as np
import pytest

import job_kit
from stbench import coverage, grid, job_vacuum2
from stbench.job_vacuum import _entry, _region_of_kind
from stbench.mapspec import parse_map
from stbench.robots import ROBOTS
from stbench.route import FreeSpace, robot_free_space
from stbench.world_processes import doors_on_ways

pytestmark = pytest.mark.xdist_group(name="test_job_vacuum2")  # the scripts of `test_job_vacuum2` are built once

BRUSH = ROBOTS["vacuum"].tools[0]
RES, CLEAR = 0.05, 0.2
WALL = {"thickness": 0.1, "height": 2.5, "exterior": True}


def room(width: float = 4.0, height: float = 3.0):
    """A `width` x `height` m room, walls 0.1 m thick, no door, no furniture (as `test_coverage`'s)."""
    return parse_map({
        "name": "room", "bounds": [0, 0, width, height], "wall_thickness": 0.1,
        "wall_height": 2.5,
        "regions": [{"id": "R1", "name": "room", "anchor": [width / 2, height / 2], "level": 0, "z0": 0.0, "z1": 2.5,
                     "polygon": [[0.05, 0.05], [width - 0.05, 0.05], [width - 0.05, height - 0.05],
                                 [0.05, height - 0.05]]}],
        "walls": [{"id": "W01", "a": [0, 0], "b": [width, 0], **WALL},
                  {"id": "W02", "a": [width, 0], "b": [width, height], **WALL},
                  {"id": "W03", "a": [0, height], "b": [width, height], **WALL},
                  {"id": "W04", "a": [0, 0], "b": [0, height], **WALL}],
        "openings": [], "windows": [], "furniture": [], "ceiling_lights": [{"region": "R1", "pos": [2, 1.5, 2.45]}],
    })


def test_strips_with_a_mask_keep_only_the_runs_that_cross_it():
    """Contract (`coverage.strips`' mask): with a mask, only the runs that hold a sample on it are driven, each whole;
    without one every run is, as before. Hand case: the 4 x 3 m room's free floor runs x 0.275 to 3.725 and y 0.275 to
    2.725; its strips lie along x at y = 0.05 + 0.2 (i + 0.5) = 0.15 + 0.2 i, so 0.35, 0.55 ... 2.55 are the 12 that
    fit. A mask on the cells from y = 2.1 up keeps the lines at y 2.15, 2.35 and 2.55 and none below, each whole from
    the west end of the free floor (x 0.25 to 0.3) to the east (3.7 and over); the robot starting at (0.3, 2.5) goes
    nowhere else. Regression caught: a resume that re-drives the whole room, or one that drives half-runs."""
    m, start = room(), (0.3, 2.5)
    free = FreeSpace(m, [], CLEAR, RES)
    everything = coverage.strips(free, m.regions[0], (0.4, 0.4), 0.2, start)
    mask = np.zeros_like(free.free)
    mask[free.cell((1.0, 2.1))[1]:, :] = True  # every cell from y = 2.1 up
    kept = coverage.strips(free, m.regions[0], (0.4, 0.4), 0.2, start, mask)
    lines = {round(p[1], 2) for p in everything}
    assert {y for y in lines if y < 2.4} >= {0.35, 0.55, 1.15, 1.95} and max(lines) == 2.55  # the whole room
    assert {round(p[1], 2) for p in kept} <= {2.15, 2.35, 2.5, 2.55}
    assert {2.15, 2.35, 2.55} <= {round(p[1], 2) for p in kept}
    assert min(p[0] for p in kept) <= 0.31 and max(p[0] for p in kept) >= 3.69  # whole runs


def test_the_cut_stops_at_a_share_of_the_floor_swept():
    """Contract (`job_vacuum2.cut`): the polyline is cut where the brush has swept `share` of the floor cells. Hand
    cases (the brush disc is 0.25 m across, so it reaches the centre of a fine cell within 0.125 m of the path: the
    cells of the row it drives along, and in the rows 0.1 m off it only those within sqrt(0.125^2 - 0.1^2) = 0.075 m
    along the path): a straight pass along y = 0.55 from x = 0.05 to 3.95 over a floor of 40 x 3 fine cells (rows y
    0.45, 0.55 and 0.65) is 120 cells; half of them, 60, are the first 20 columns, whose last centre is x = 1.95,
    reached in the off rows when the base is at x = 1.95 - 0.075 = 1.875, so the cut is there; a second leg north
    along x = 1.05 over a column of 11 cells (y 0.55 to 1.55) with a share of half (6 cells, the sixth at y = 1.05) is
    cut at y = 1.05 - 0.125 = 0.925. A share past what the path sweeps keeps the polyline whole. Regression caught: a
    stop cut by path length (the edge pass is a third of a living room's path and most of its floor), or one that
    never cuts."""
    floor = {(u, v) for u in range(40) for v in (4, 5, 6)}
    path = [(0.05, 0.55), (3.95, 0.55)]
    cut = job_vacuum2.cut(path, floor, 0.5, BRUSH)
    assert len(cut) == 2 and cut[0] == path[0] and cut[1][1] == 0.55 and cut[1][0] == pytest.approx(1.875, abs=1e-3)
    assert job_vacuum2.cut(path, floor, 1.1, BRUSH) == path
    leg = job_vacuum2.cut([(0.05, 0.55), (1.05, 0.55), (1.05, 1.55)], {(10, v) for v in range(5, 16)}, 0.5, BRUSH)
    assert leg[:2] == [(0.05, 0.55), (1.05, 0.55)] and len(leg) == 3
    assert leg[2][0] == 1.05 and leg[2][1] == pytest.approx(0.925, abs=1e-3)


def test_the_unswept_mask_is_the_floor_less_the_swath_grown_by_the_reach():
    """Contract (`job_vacuum2.unswept`): the cells of a floor mask whose fine cell is not in the swept set, grown by the
    reach (a square about each). Hand case: a 4 x 3 m free grid of 0.05 m cells, the swept fine cells a 5 x 5 block
    (u, v 10 to 14: x and y 1.0 to 1.5, so 10 x 10 free cells), a reach of 0.1 m (2 cells): only the cells of the block
    at least 2 cells from its edge stay unmarked, 6 x 6 = 36, and the rest of the floor (every other cell) is marked.
    With no reach it is the floor less the 100 block cells. Regression caught: a resume that re-drives swept lines, or
    one that skips the lines beside the unswept floor, which the brush's disc would reach."""
    m = room()
    free = FreeSpace(m, [], CLEAR, RES)
    floor = free.free.copy()
    block = {(u, v) for u in range(10, 15) for v in range(10, 15)}
    plain = job_vacuum2.unswept(free, floor, block)
    grown = job_vacuum2.unswept(free, floor, block, 0.1)
    assert int((floor & ~plain).sum()) == 100 and int((floor & ~grown).sum()) == 36
    assert bool(plain[free.cell((0.6, 0.6))[1], free.cell((0.6, 0.6))[0]])
    assert not bool(grown[free.cell((1.25, 1.25))[1], free.cell((1.25, 1.25))[0]])


def test_resume_covers_what_the_stop_left():
    """Contract (plan §5.1): on scripts with a stop (seeds 101, 102, 104 and 106, the first of the plan's draw with
    stops), the stopped task's brush swept its share of the stopped room's base-reachable fine cells (the stop's,
    30-70%, so 25-75% allowing the cut's resolution), the stopped and resumed tasks together reach what a whole clean
    of the room reaches, within 7 points (a resume has no edge pass, so a stop inside the edge pass leaves the wall
    margin only that pass reaches), and the resumed cleaning is no longer than the room's strips alone (a metre
    allowing for the joins between chains: an edge pass would add 1.8 m or more), a kitchen stopped from half way on
    is resumed in under four fifths of them. Regression caught: a resume that re-cleans the whole room, a stop that
    left no hole, or a resume that leaves the unswept strips."""
    robot, checked, kitchens = ROBOTS["vacuum"], 0, 0
    for seed in (101, 102, 104, 106):
        worlds, tl = job_kit.generated2(seed)
        for task in tl.ego.tasks:
            stop = task.plan.params.get("stop")
            if stop is None:
                continue
            nxt = tl.ego.tasks[task.index + 1]
            m = job_kit.apartment()
            room_ = _region_of_kind(m, task.plan.params["kinds"][-1])
            keep = frozenset({"R1", room_.id})
            shut = sorted(o.id for o in m.toggling_doors() if o.id not in doors_on_ways(m, keep))
            free = robot_free_space(m, worlds[nxt.index], robot, tl.keyframe(nxt.k0).t, shut=shut)
            entry = _entry(m, free, room_, job_kit.OUTSIDE_THE_DOCK)
            floor = job_vacuum2.fine_cells(free, coverage.reachable_floor(free, room_, entry))
            first = job_vacuum2.brush_cells(robot, tl.ego.keyframes[task.k0:task.k1])
            second = job_vacuum2.brush_cells(robot, tl.ego.keyframes[nxt.k0:nxt.k1])
            loop = coverage.perimeter(free, room_, entry)
            whole = loop + coverage.strips(free, room_, entry, 0.2, loop[-1])[1:]
            full = grid.swath([(x, y, 0.0) for x, y in whole], BRUSH.shape, *BRUSH.size_m) & floor
            share = len(first & floor) / len(floor)
            assert 0.25 <= share <= 0.75 and share == pytest.approx(stop, abs=0.05), (seed, task.index)
            assert len((first | second) & floor) >= len(full) - 0.07 * len(floor), (seed, task.index)
            kfs = tl.ego.keyframes[nxt.k0:nxt.k1]
            resumed = sum(math.dist(a.base[:2], b.base[:2]) for a, b in zip(kfs, kfs[1:], strict=False)
                          if a.phase == b.phase == "cleaning")
            strips_only = sum(math.dist(a, b) for a, b in zip(*(lambda p: (p, p[1:]))(
                coverage.strips(free, room_, entry, 0.2, entry)), strict=False))
            assert resumed <= strips_only + 1.0, (seed, task.index)
            if room_.kind == "kitchen" and stop >= 0.5:
                assert resumed <= 0.8 * strips_only, (seed, task.index)
                kitchens += 1
            checked += 1
    assert checked >= 4 and kitchens >= 2


def test_a_stopped_task_ends_its_cleaning_at_the_stop_and_its_resume_finishes_the_room():
    """Contract (plan §5.1): a stopped task marks `cleaning_stopped` on its last cleaning keyframe where a finished one
    marks `room_done`, and its resume marks `room_done` and never stops again. Hand check on seed 101: task 0 cleans the
    living room and stops (`cleaning_stopped` once, no `room_done`), task 1 resumes it (`room_done` once, no
    `cleaning_stopped`), each with a `before_doorway` into the room and out of it. Regression caught: a resume that
    stops again, or a stopped task that finishes its room."""
    _, tl = job_kit.generated2(101)
    kinds = lambda i: [d.kind for d in tl.decisions_in(tl.ego.tasks[i].k0, tl.ego.tasks[i].k1)]  # noqa: E731
    assert (tl.ego.tasks[0].plan.kind, tl.ego.tasks[1].plan.kind) == ("clean_rooms", "resume")
    assert tl.ego.tasks[0].plan.params["kinds"] == tl.ego.tasks[1].plan.params["kinds"] == ["living room"]
    assert kinds(0) == ["before_doorway", "cleaning_stopped", "before_doorway"]
    assert kinds(1) == ["before_doorway", "room_done", "before_doorway"]
