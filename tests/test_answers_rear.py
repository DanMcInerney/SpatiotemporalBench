"""The answerers of `rear_clear`: the table systems' (`table_space`) and `path_log`'s (`baselines_path_space`, whose
corridor `path_ahead` shares). The table cases are `rear_kit`'s tote AMR docked at (7.0, 4.98) facing +x, its rear edge
at x 6.6, the reverse corridor 0.58 m wide (y 4.69 to 5.27) running west, cap 5 m. The `path_log` cases are hand-drawn
halls with the same pose: expected values are worked out in each test (0.1 m fine cells, 0.02 m sub-cells, a corridor
cell's near edge 0.05 m nearer than its centre)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

from answers_kit import compiled_case, scored
from baseline_kit import UNKNOWN, gold, number, question, rows
from rear_kit import AMR, AXIS_Y, DOCK, JOB, NEAR_BOX, lane_run, line_run
from space_kit import use_job
from stbench import baselines_path_space, grid, mapspec, robots, score, validate
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.baselines_path_space import FLOOR_COVERING_M, obstacles
from stbench.geometry import Rect, to_local
from stbench.grid import raycast
from stbench.stream import run_case

CAP = 5.0
RUN_JOB = replace(JOB, assignments={"clean_room": "Its current assignment is to dock at the {room}."})
EVENTS = dict(JOB.events)


# -- the table systems ------------------------------------------------------------------------------------------
def test_the_oracle_answers_rear_clear(tmp_path, monkeypatch):
    """Contract: through `run_case` on `detections` over `rear_kit`'s cases the oracle and `memory_table` give the gold
    of `rear_clear` (`grid.corridor_clear` over the latest cells of the window up to the ask, from the rear edge) and
    `current_view`, which holds the last frame alone, answers `unknown` where the corridor was seen only before it. Hand
    values: (A) `lane_run` with `NEAR_BOX` seen only at frame 3, the view from the lane: the box's east face, 2.1 (its x
    4.49 lies in the sub-cell [4.48, 4.50), whose centre is 6.6 - 4.49 = 2.11 from the rear edge, less 0.01), which the
    oracle and `memory_table` answer; the decision's frame (offset 20) saw nothing, so `current_view` says `unknown`.
    (B) `line_run` with one look at offset 16, four frames before the decision: the empty corridor, 5.0, which the
        oracle and `memory_table` answer; `current_view` holds frame 20 alone, which faces the station, so it says
        `unknown` too (a look window of up to 8 frames would have held frame 16 and answered). Regression caught: the
        table reading other cells than the gold (the base or the camera for the start, a blind band), or a
        `current_view` that remembers an earlier frame."""
    dataset = tmp_path / "ds"
    cases = {"lane": compiled_case(dataset, "c0", lane_run(boxes=[NEAR_BOX], depth_ks={3}, seed=1), JOB, 0, 0),
             "line": compiled_case(dataset, "c1", line_run(depth_ks={16}, seed=2), JOB, 0, 0)}
    use_job(monkeypatch, RUN_JOB)
    wanted = {"lane": number(2.1), "line": number(5.0)}
    systems = {"lane": ((Oracle, number(2.1), 1.0), (MemoryTable, number(2.1), 1.0), (CurrentView, UNKNOWN, 0.0)),
               "line": ((Oracle, number(5.0), 1.0), (MemoryTable, number(5.0), 1.0),
                        (CurrentView, UNKNOWN, 0.0))}   # (system, answer, score)
    for name, case in cases.items():
        qs, golds = rows(case)
        assert [g["gold"] for g in golds] == [wanted[name]] and [q["template"] for q in qs] == ["rear_clear"]
        for cls, answer, points in systems[name]:
            run = run_case(lambda state, work, cls=cls: cls(state, work, events=EVENTS), case, channel="detections")
            assert run.status == "complete" and run.failures == {}, (name, cls.__name__)
            assert run.answers == {qs[0]["qid"]: answer}, (name, cls.__name__)
            assert scored(golds, run.answers) == {qs[0]["qid"]: points}, (name, cls.__name__)


# -- a hall to draw the public map of ---------------------------------------------------------------------------
def piece(fid: str, kind: str, center: tuple, size: tuple, yaw: float = 0.0) -> dict:
    """A fixed piece of furniture that blocks, solid to the floor."""
    return {"id": fid, "name": kind, "kind": kind, "mobility": "fixed", "center": list(center), "size": list(size),
            "yaw": yaw, "color": "grey", "blocks": True, "surfaces": [], "under_m": 0.0}


def wall(wid: str, a: tuple, b: tuple, exterior: bool = False) -> dict:
    return {"id": wid, "a": list(a), "b": list(b), "thickness": 0.1, "height": 3.0, "exterior": exterior}


def hall(walls=(), furniture=(), leaf_at: float | None = None):
    """A 12 x 9 m hall (its own walls 5 m or more from the corridor of a base at (7.0, 4.98) facing +x) with the given
    inner `walls` and fixed `furniture`; `leaf_at` puts a wall across the corridor at that x with a closed exterior door
    0.9 m wide on the axis (y 4.55 to 5.45), whose 0.04 m leaf is the one piece there (the wall's others lie outside
    the corridor, y below 4.55 and above 5.45)."""
    outer = [wall("W1", (0, 0), (12, 0), True), wall("W2", (12, 0), (12, 9), True), wall("W3", (0, 9), (12, 9), True),
             wall("W4", (0, 0), (0, 9), True)]
    door = {"id": "D1", "type": "door", "wall": "W9", "at": 2.0, "width": 0.9, "height": 2.0, "hinge": "a",
            "swing": "left", "interior": False, "initial_open": False, "between": ["R1", "outside"]}
    return mapspec.parse_map({
        "name": "hall", "bounds": [0, 0, 12, 9], "wall_thickness": 0.1, "wall_height": 3.0,
        "regions": [{"id": "R1", "name": "hall", "anchor": [6, 4.5], "level": 0, "z0": 0.0, "z1": 3.0,
                     "polygon": [[0.05, 0.05], [11.95, 0.05], [11.95, 8.95], [0.05, 8.95]]}],
        "walls": [*outer, *walls, *([wall("W9", (leaf_at, 3.0), (leaf_at, 7.0))] if leaf_at else [])],
        "openings": [door] if leaf_at else [], "windows": [], "furniture": list(furniture), "ceiling_lights": []})


def rear_ray(m, base: tuple = DOCK, yaw: float = 0.0) -> float:
    """The raycast of the AMR's reverse corridor at `base` heading `yaw`: from the rear edge, the heading plus pi."""
    heading = yaw + math.pi
    return raycast(obstacles(m, AMR), grid.front_edge(AMR, base, heading), heading, AMR.body, CAP)


def test_the_raycast_is_judged_at_sub_cells():
    """Contract: the corridor stops at the nearest 0.02 m sub-cell whose centre lies in the body's width (half 0.29 m
    for the AMR) and in a piece of the public map, less half a sub-cell, so what stands outside the width never stops it
    however near its fine cell's centre lies, a leaf 0.04 m thick, which no fine cell centre lies in, does, and the
    answer is the obstacle's own 2 cm (D1: the 0.1 m cell's near edge put it up to 8 cm short). Hand cases from the rear
    edge (x 6.6) west along y 4.98, cap 5: (a) a wall x 2.0 to 5.0 (the piece runs 1.95 to 5.05) whose south face is
    0.31 m off the axis (y 5.29): its sub-cells' centres are 0.31 off, outside the width, though the fine cell [5.2,
    5.3), centre 0.27 off, holds them: 5.0; (b) the same wall with its face 0.26 off (y 5.24): the sub-cell centred y
    5.25 is 0.27 off, inside, and the piece's east end at 5.05 holds the centre x 5.05 (on its edge, which is in), 1.55
    from the edge: 1.54; (c) a closed door's leaf across the axis at x 5.6, 1.0 m behind the rear edge: the sub-cell
    centres 5.59 and 5.61 lie in it, the nearer to the edge is 5.61, 0.99 from it: 0.98. Regression caught: fine cells
    catching what stands beside the body (M1's door-frame error), or a test of cell centres, which a 0.04 m leaf slips
    between (the centres 5.55 and 5.65), so that `path_log` would be weaker than an agent that knows the layout."""
    def beside(face_y: float) -> float:
        return rear_ray(hall(walls=[wall("W5", (2.0, face_y + 0.05), (5.0, face_y + 0.05))]))

    assert beside(5.29) == CAP
    assert beside(5.24) == 1.54
    assert rear_ray(hall(leaf_at=5.6)) == 0.98


class Perfect(dict):
    """The cells of a perfect sensor of some rects, as `grid.latest` has them: the occupied fine cells (an `Occupied`
    that names the sub-cells inside a rect), every other cell free."""

    def get(self, cell, default=None):
        return super().get(cell, ("free", 0))


def perfect(rects: list[Rect]) -> Perfect:
    held: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in rects:
        reach = math.hypot(r.sx, r.sy) / 2
        for a in range(math.floor((r.cx - reach) / grid.SUB_M), math.floor((r.cx + reach) / grid.SUB_M) + 1):
            for b in range(math.floor((r.cy - reach) / grid.SUB_M), math.floor((r.cy + reach) / grid.SUB_M) + 1):
                lx, ly = to_local(r, (a + 0.5) * grid.SUB_M, (b + 0.5) * grid.SUB_M)
                if abs(lx) <= r.sx / 2 + 1e-9 and abs(ly) <= r.sy / 2 + 1e-9:
                    held.setdefault((a // grid.SUB_PER_FINE, b // grid.SUB_PER_FINE), []).append((a, b))
    return Perfect({c: (grid.Occupied(subs), 0) for c, subs in held.items()})


def test_the_raycast_is_the_golds_on_a_perfect_sensor():
    """Contract: `raycast` gives what the gold's `grid.corridor_clear` gives over the cells a perfect sensor would make
    of the same pieces (the occupied fine cells with the sub-cells inside a piece, all else free), at any heading: the
    cell, the measure and the width are the gold's, so `path_log` is the shortcut a corridor family must stay under and
    no weaker. Hand scene, the AMR's 0.58 m corridor (half-width 0.29) of length 3 from (4.0, 4.0): a 1.0 x 0.4 m crate
    turned 30 degrees at (6.0, 4.6), a 0.2 m post at (1.5, 4.35), a wall piece 2.0 x 0.1 m turned 100 degrees at (3.8,
    5.5). East: only the crate's lowest corner (5.667, 4.177) lies below y 4.29, and on the corridor's edge row (the
    sub-cells centred y 4.29) its short edge is at x 5.602, so the sub-cell centred 5.61 is the first, 1.61 along: 1.6.
    West: the post's sub-cell centred (1.59, 4.27) is 0.27 off the axis, 2.41 along: 2.4. Along 17 degrees (the crate's
    centre line, the axis 0.9563, 0.2924) the first sub-cell is the one centred (5.49, 4.49) on the crate's short edge,
    0.03 off the axis, 1.49 x 0.9563 + 0.49 x 0.2924 = 1.568 along: 1.558. North, the wall piece's foot (sub-cells from
    y 4.51) is the row centred 4.51, 0.51 along: 0.5. Regression caught: a raycast that drifts
    from the gold off the axes (a fine cell whose centre lies outside the corridor, a rotated piece, a distance taken
    from the sub-cell and not from the cell's centre), which the axis-aligned hand cases cannot show."""
    rects = [Rect(6.0, 4.6, 1.0, 0.4, math.radians(30)), Rect(1.5, 4.35, 0.2, 0.2, 0.0),
             Rect(3.8, 5.5, 2.0, 0.1, math.radians(100))]
    headings = [math.radians(deg) for deg in (0, 180, 17, 90)]
    got = [raycast(rects, (4.0, 4.0), yaw, AMR.body, 3.0) for yaw in headings]
    sensed = perfect(rects)
    golds = [grid.corridor_clear(sensed, AMR, (4.0, 4.0), yaw, 3.0, 0.0)[0] for yaw in headings]
    assert golds == [number(v) for v in got]
    assert got == [1.6, 2.4, 1.558, 0.5]


def test_floor_markings_do_not_stop_the_raycast():
    """Contract: a fixed piece lower than `FLOOR_COVERING_M` (0.05 m: rugs, mats, the painted `staging_lane` and `bay`)
    never stops the raycast, as the gold's band lets the AMR over it, and a piece of 0.05 m or more does; the constant
    is `validate.FLOOR_COVERING_HEIGHT`, which a baseline may not import. Hand-worked from the rear edge (x 6.6) west
    along y 4.98, cap 5, with a staging lane 1.4 m deep across the corridor (x 3.7 to 5.1, the real map's 2.8 x 1.4
    piece turned to face north) and a rack, 2.0 m tall, x 2.9 to 3.4 (its east face in the cell [3.3, 3.4), whose
    centre is 3.25 m from the edge): with the lane 0.005 m high (the map's paint) or 0.01 m (a bay's old stripe) the
    corridor runs to the rack, 3.25 - 0.05 = 3.2; without the rack, to the cap; with the lane 0.05 m high it stops at
    the lane's east edge, the cell [5.0, 5.1): 1.55 - 0.05 = 1.5. Regression caught: `path_log` stopping at paint, a
    shortcut weaker than an agent that knows the map, which would let a corridor family pass its gate vacuously."""
    def hall_with(height: float, rack: bool = True):
        lane = piece("F1", "staging_lane", (4.4, AXIS_Y), (2.8, 1.4, height), math.pi / 2)
        return hall(furniture=[lane, *([piece("F2", "rack", (3.15, AXIS_Y), (0.5, 1.5, 2.0))] if rack else [])])

    assert FLOOR_COVERING_M == validate.FLOOR_COVERING_HEIGHT == 0.05
    assert [rear_ray(hall_with(h)) for h in (0.005, 0.01)] == [3.2, 3.2]
    assert rear_ray(hall_with(0.005, rack=False)) == CAP
    assert rear_ray(hall_with(0.05)) == 1.5


# -- path_log on poses ------------------------------------------------------------------------------------------
def started(tmp_path, m) -> PathLog:
    """A `path_log` of the AMR on the public map `m`, sent `start`."""
    (tmp_path / "public").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "public" / "camera.json").write_text(json.dumps(robots.camera_doc(AMR)), encoding="utf-8")
    system = PathLog(tmp_path / "state", tmp_path, map=m, events=EVENTS, prior=[])
    system.start("0123456789abcdef", robots.robot_doc(AMR), "detections")
    return system


def amr_step(k: int, x: float, y: float, yaw_deg: float = 0.0, **extra) -> dict:
    """A delivered step of the AMR: its base at (x, y) on the floor heading `yaw_deg`, one second a step."""
    yaw = math.radians(yaw_deg)
    cam_pos, cam_rot = robots.camera_pose(AMR, (x, y, 0.0), yaw, math.radians(AMR.camera.pitch_deg))
    return {"k": k, "clock": f"2026-03-02T08:00:{k:02d}",
            "pose": {"pos": [x, y, 0.0], "rot": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], "yaw_deg": yaw_deg},
            "camera": {"pos": list(cam_pos), "rot": list(cam_rot), "pitch_deg": AMR.camera.pitch_deg},
            "status": {"phase": "at_station", "task": 0, "battery_pct": 90, "loaded": False}, **extra}


def ask(system: PathLog, template: str, steps: list[dict]) -> dict:
    """What `system` answers to `template` (cap 5) at the last of `steps`, which it is sent first."""
    system.observe(steps)
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""},
                      [question(template, {"cap": CAP}, answer_type="number")])["q1"]


def drive(*xy: tuple[float, float]) -> list[dict]:
    """Steps along the base positions `xy`, heading +x throughout."""
    return [amr_step(k, x, y) for k, (x, y) in enumerate(xy)]


def test_one_front_edge(tmp_path, monkeypatch):
    """Contract: where a corridor starts is `grid.front_edge`'s, for `path_ahead` and `rear_clear` of `path_log` alike,
    and `baselines_path_space` has none of its own. Hand case: `grid.front_edge` replaced by a recorder that gives the
    real answer; each family asked once at the base (7.0, 4.98) heading +x calls it once with the AMR, the base and its
    own heading (0 for `path_ahead`, pi for `rear_clear`), and the module defines no `front_edge`. Regression caught:
    two owners of the corridor's start, which drift apart (the module's copy was the review's finding 8)."""
    calls = []
    real = grid.front_edge
    monkeypatch.setattr(grid, "front_edge", lambda robot, base, yaw: calls.append((robot.id, tuple(base), yaw))
                        or real(robot, base, yaw))
    system = started(tmp_path, hall())
    system.observe([amr_step(0, 7.0, AXIS_Y)])
    for template in ("path_ahead", "rear_clear"):
        system.ask("during", {"k": 0, "clock": ""}, [question(template, {"cap": CAP}, answer_type="number")])
    assert calls == [("amr", DOCK, 0.0), ("amr", DOCK, math.pi)]
    assert not hasattr(baselines_path_space, "front_edge")


def test_rear_detour_by_hand(tmp_path):
    """Contract (design §8.1, "paths leak"): the cells the body swept count as seen free, and a detour round something
    that stands in what becomes the reverse corridor is evidence: along the corridor's axis from the rear edge, a swept
    first cell, then an unswept one, then a swept one again puts the thing at the first unswept cell. Hand case on an
    empty hall (the raycast says 5.0): the AMR (0.8 x 0.58, heading +x throughout) drove (0.5, 4.98), (3.4, 4.98),
    round a box to (3.4, 3.9), (5.0, 3.9), (5.0, 4.98) and in to the dock (7.0, 4.98). The axis row's cells are swept
    at the centres 6.55 down to 4.65 (the body covers x 4.6 to 7.4 on the last legs), the cell centred 4.55 is not, and
    those at 3.75 and below are (x 3.8 on the first), so the first unswept centre is 6.6 - 4.55 = 2.05 from the rear
    edge: 2.0, against the box's face (x 4.49, the sub-cell centred 4.49) 2.1 by the gold's rule, inside the 0.25 m rung
    where the raycast's 5.0 earns nothing. After a straight drive in along the axis every cell is swept and the raycast
    answers: 5.0. Regression caught: a family a detour gives away passing its gate (a `path_log` blind to its own
    path), or a detour read from a path that merely swept the corridor."""
    round_the_box = drive((0.5, 4.98), (3.4, 4.98), (3.4, 3.9), (5.0, 3.9), (5.0, 4.98), (7.0, 4.98))
    answer = ask(started(tmp_path / "detour", hall()), "rear_clear", round_the_box)
    assert answer == number(2.0)
    assert score.penalised(answer, gold(number(2.1))) == 1.0
    assert score.penalised(number(CAP), gold(number(2.1))) == -1.0
    assert ask(started(tmp_path / "straight", hall()), "rear_clear", drive((0.5, 4.98), (7.0, 4.98))) == number(CAP)


def test_path_log_sees_no_picture(tmp_path):
    """Contract: `path_log` answers `rear_clear` and `path_ahead` from its poses and the public map alone, so its
    answers are the same with every step's `rgb`, `depth` and `detections` removed (here they name files that do not
    exist and detections of things that are not there). Hand case: the AMR came up a lane (x 6.0) to the turn at (6.0,
    4.98) and drove 1 m in to the dock (7.0, 4.98) facing +x, in a hall with a closed door's leaf across the axis at x
    5.4 and a wall at x 8.4 (faces 8.35 and 8.45). Behind: the leaf's sub-cell centres are 5.39 and 5.41, the nearer to
    the rear edge is 5.41, 1.19 from it: 1.18 (the swept cells end at 5.65, nothing swept again beyond). Ahead from the
    front edge 7.4: the sub-cell centred 8.35 (on the wall's face), 0.95 off: 0.94. Regression caught: a shortcut
    that reads depth or a detection (a corridor from the picture is no shortcut)."""
    m = hall(walls=[wall("W6", (8.4, 3.0), (8.4, 7.0))], leaf_at=5.4)
    path = [(6.0, 2.0, 90.0), (6.0, 4.98, 90.0), (6.0, 4.98, 0.0), (7.0, 4.98, 0.0)]
    clean = [amr_step(k, *pose) for k, pose in enumerate(path)]
    seen = [{**s, "rgb": f"inbox/rgb/{s['k']:06d}.jpg", "depth": f"inbox/depth/{s['k']:06d}.png",
             "detections": [{"d": "a", "kind": "pallet", "attributes": {"color": "grey"}, "bbox_px": [0, 0, 40, 40],
                             "center_cam": [0.0, 0.0, 1.0], "center_map": [3.0, AXIS_Y, 0.3], "extent_m": [1.0] * 3}]}
            for s in clean]
    got = {name: {t: ask(started(tmp_path / f"{name}{t}", m), t, steps) for t in ("rear_clear", "path_ahead")}
           for name, steps in (("clean", clean), ("seen", seen))}
    assert got["clean"] == got["seen"] == {"rear_clear": number(1.18), "path_ahead": number(0.94)}
