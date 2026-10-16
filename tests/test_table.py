"""The memory table: how detections become tracks (V0 §10), what it logs of a stream, and its answerers.

Steps are hand-made: the camera sits at (0, 0, 1.5) facing +x with an identity rotation, so a map point is
x_cv = -y, y_cv = -(z - 1.5), z_cv = x. Detections carry the true box centre, as the oracle channel does.
"""

from __future__ import annotations

import json
import math

import pytest

import zone_kit as zk
from baseline_kit import EVENTS as EVENTS_OF_THE_JOB
from baseline_kit import ROBOT, ahead, started
from baseline_kit import step as base_step
from space_kit import box, render_depth
from stbench import grid, robots, table_answers, zones
from stbench.baselines import CurrentView, MemoryTable
from stbench.camera import Camera, map_to_cam
from stbench.events import EventRule
from stbench.render_encode import read_png16, write_png16
from stbench.table import Mapper, Store
from stbench.table_log import Log

POS, ROT = [0.0, 0.0, 1.5], [1.0, 0.0, 0.0, 0.0]


def det(kind, color, centre, key="d"):
    cam = map_to_cam(centre, POS, ROT)
    return {"d": key, "kind": kind, "attributes": {"color": color}, "bbox_px": [0, 0, 9, 9],
            "center_cam": [float(v) for v in cam], "extent_m": [0.1, 0.1, 0.1]}


def step(k, clock, *dets, identity=None, phase="cleaning"):
    s = {"k": k, "clock": clock, "pose": {"pos": POS, "rot": ROT, "yaw_deg": 0.0},
         "camera": {"pos": POS, "rot": ROT, "pitch_deg": 0.0}, "status": {"phase": phase, "task": 0},
         "detections": list(dets)}
    if identity is not None:
        s["identity"] = identity
    return s


def feed(*steps):
    store = Store()
    for s in steps:
        store.add_step(s)
    return store


def places(store):
    return [[(int(s[0]), *(round(v, 3) for v in s[1:])) for s in t.sightings] for t in store.tracks]


def test_detections_in_one_frame_match_tracks_one_to_one_nearest_first():
    """Contract (V0 §10): within a frame a track takes at most one detection, nearest first. Two red mugs 0.2 m apart
    are two tracks, and when they appear in swapped order in the next frame each detection still goes to its nearest
    track. Regression caught: both detections joining one track (an item lost), or an order-dependent assignment."""
    a, b = (2.0, 0.1, 0.95), (2.0, -0.1, 0.95)
    store = feed(step(0, "2026-03-02T08:00:00", det("mug", "red", a, "x"), det("mug", "red", b, "y")),
                 step(1, "2026-03-02T08:00:02", det("mug", "red", b, "x"), det("mug", "red", a, "y")))
    assert places(store) == [[(0, 2.0, 0.1, 0.95), (1, 2.0, 0.1, 0.95)], [(0, 2.0, -0.1, 0.95), (1, 2.0, -0.1, 0.95)]]


def test_a_unique_description_reidentifies_a_moved_item():
    """Contract (V0 §10): a detection that matches no track by position joins the description's only track, which
    was not matched in this frame: a blue mug seen at (2,0) and later at (4,2) is one item that moved.
    Regression caught: a moved item becoming a second track, so a count and a moved question both go wrong."""
    store = feed(step(0, "2026-03-02T08:00:00", det("mug", "blue", (2.0, 0.0, 0.95))),
                 step(1, "2026-03-02T10:00:00", det("mug", "blue", (4.0, 2.0, 0.95))))
    assert len(store.tracks) == 1 and [s[0] for s in store.tracks[0].sightings] == [0, 1]


def test_identical_items_seen_together_stay_separate_and_a_third_is_new():
    """Contract: once two tracks of one description exist, a detection far from both is a new item, not a move. Two
    red mugs seen together, then a third elsewhere: three tracks. Regression caught: re-identifying an item as one of
    two candidates and under-counting identical groups."""
    store = feed(step(0, "2026-03-02T08:00:00", det("mug", "red", (2.0, 0.0, 0.95), "x"),
                      det("mug", "red", (3.0, 1.0, 0.95), "y")),
                 step(1, "2026-03-02T08:00:02", det("mug", "red", (5.0, -2.0, 0.95))))
    assert len(store.tracks) == 3


def test_identical_items_never_seen_together_merge():
    """Contract (a limitation of the table that `knowable` measures): two identical mugs far apart that are never in
    view together look like one item that moved, so the table keeps one track. Regression caught: a table that tells
    identical items apart without evidence, so its answers would depend on more than the sensors."""
    store = feed(step(0, "2026-03-02T08:00:00", det("mug", "red", (2.0, 0.0, 0.95))),
                 step(1, "2026-03-02T08:00:02", det("mug", "red", (5.0, 2.0, 0.95))))
    assert len(store.tracks) == 1


def test_with_perfect_identity_a_track_is_an_entity():
    """Contract (`oracle`): given the stream's identity map, identical mugs never seen together are two tracks.
    Regression caught: the oracle inheriting the table's merging, so the ceiling is not a ceiling."""
    store = feed(step(0, "2026-03-02T08:00:00", det("mug", "red", (2.0, 0.0, 0.95), "x"), identity={"x": "E1"}),
                 step(1, "2026-03-02T08:00:02", det("mug", "red", (5.0, 2.0, 0.95), "y"), identity={"y": "E2"}))
    assert len(store.tracks) == 2 and [t.ident for t in store.tracks] == ["E1", "E2"]


def test_a_detections_center_map_is_used_when_the_channel_gives_it():
    """Contract: with `center_map` in the detection (the detections channel has it) the table does not redo the frame
    transform; a hand-made centre that disagrees with `center_cam` shows which one it used. Regression caught: a
    system that is tested on arithmetic the channel already did for it."""
    d = det("mug", "red", (2.0, 0.0, 0.95))
    d["center_map"] = [7.0, 8.0, 0.95]
    store = feed(step(0, "2026-03-02T08:00:00", d))
    assert places(store) == [[(0, 7.0, 8.0, 0.95)]]


def test_a_detection_is_moved_to_the_map_with_the_camera_not_the_base():
    """Contract (the stream protocol: `pose` is the base, `camera` the camera that saw): a `center_cam` is a point in
    the camera's frame, so it moves to the map with the step's camera. Hand case: the base at (1, 0) heading +x, the
    camera 0.13 m ahead of it and 0.08 m up, pitched 5 degrees; a mug at the map point (2, 0.3, 0.1) is `map_to_cam` of
    that camera, and the table puts it back at (2, 0.3, 0.1) where the base's pose would put it 0.13 m and a pitch away.
    Regression caught: `center_cam` moved by the base (today's `table.py`), which misplaces every detection by the
    camera's mount."""
    s = base_step(0, 1.0, 0.0)
    centre = (2.0, 0.3, 0.1)
    d = det("mug", "red", centre)
    d["center_cam"] = [float(v) for v in map_to_cam(centre, s["camera"]["pos"], s["camera"]["rot"])]
    store = Store()
    store.add_step({**s, "detections": [d]})
    assert places(store) == [[(0, 2.0, 0.3, 0.1)]]


def test_tasks_come_from_task_messages_not_clock_gaps():
    """Contract (map SG-H2): a task begins at its `task` message and holds the steps from the next one delivered; its
    steps are what an event resolves over. Hand case: steps 0-2 then 3-5 with every clock one second after the last (no
    gap at all) and a message before each three: two tasks starting at steps 0 and 3, the last task's steps 3, 4, 5.
    With no message, every step. Regression caught: tasks found from clock gaps, which a job that runs two tasks
    back to back would merge."""
    log = feed(*(step(k, f"2026-03-02T08:00:0{k}") for k in range(3)))
    assert [s["k"] for s in log.task_steps()] == [0, 1, 2]
    store = Store()
    for index, (lo, hi) in enumerate(((0, 3), (3, 6))):
        store.begin_task(index, f"2026-03-02T08:00:0{lo}", 0.0, f"task {index}")
        for k in range(lo, hi):
            store.add_step(step(k, f"2026-03-02T08:00:0{k}"))
    assert [(t.index, t.k0, t.brief) for t in store.tasks] == [(0, 0, "task 0"), (1, 3, "task 1")]
    assert [s["k"] for s in store.task_steps()] == [3, 4, 5] and store.briefs() == ["task 0", "task 1"]


def three_task_log() -> Log:
    """Nine steps, three tasks of three: k 0-2 at 08:00:00, 08:00:02, 08:00:04, k 3-5 at 09:00:00, 09:00:02, 09:00:04
    and k 6-8 at 10:00:00, 10:00:02, 10:00:04, the `task` messages numbering them 0, 1, 2 (a window's case
    numbering)."""
    log = Log()
    for index in range(3):
        hour = f"{8 + index:02d}"
        log.begin_task(index, f"2026-03-02T{hour}:00:00", 0.0, f"task {index}")
        for i in range(3):
            log.add(step(3 * index + i, f"2026-03-02T{hour}:00:0{2 * i}"))
    return log


def test_the_log_by_hand():
    """Contract (E3a §2): the log answers the reads of a question that names a task or a clock. `task_steps(i)` is the
    steps of the task whose message carried the case index i (task 1: k 3, 4, 5), `task_steps()` the last task's (6, 7,
    8), and an index no message carried is `[]`, so an event over it resolves to nothing and the answer is `unknown`,
    never a raise (a partner holding fewer tasks, or `current_view`'s one-frame store with no task message at all).
    `steps_until(4)` is k 0 to 4. `nearest` is the step whose clock is nearest: 09:00:03 lies halfway between k 4
    (09:00:02) and k 5 (09:00:04), so the earlier, k 4; 09:59:59 is nearest k 6 (10:00:00); a clock before the stream is
    k 0; an empty log has none. `task_of(k)` is the case index of the task holding k (k 4: 1, k 8: 2), None before any
    task message (a step logged first). Regression caught: an event resolved in the wrong task, a missing task raising,
    or a clock answered off by a step."""
    log = three_task_log()
    assert [[s["k"] for s in log.task_steps(i)] for i in (0, 1, 2, 4)] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], []]
    assert [s["k"] for s in log.task_steps()] == [6, 7, 8]
    assert [s["k"] for s in log.steps_until(4)] == [0, 1, 2, 3, 4]
    assert [log.nearest(c)["k"] for c in ("2026-03-02T09:00:03", "2026-03-02T09:59:59", "2026-03-01T00:00:00")] == [
        4, 6, 0]
    assert Log().nearest("2026-03-02T09:00:03") is None
    assert [log.task_of(k) for k in (0, 4, 8)] == [0, 1, 2]
    early = Log()
    early.add(step(0, "2026-03-02T07:59:00"))
    early.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    early.add(step(1, "2026-03-02T08:00:00"))
    assert [early.task_of(0), early.task_of(1)] == [None, 0]
    assert early.task_steps(0) == [early.steps[1]] and Log().task_steps(0) == [] and Log().task_steps() == []


def test_the_space_as_last_seen_reads_every_step_up_to_one():
    """Contract (E3a §2, `Store.latest_states`): `grid.latest` over every logged step's cells up to a step, each cell's
    latest state and the step that saw it; `states_of` stays the read of chosen steps. Hand cells: step 1 saw A free
    and B occupied, step 3 saw A occupied, step 5 saw C free. Up to step 2: A free (1), B occupied (1); up to 4: A
    occupied (3), B occupied (1); all: C free (5) too; up to step 0: nothing. Regression caught: the space as last seen
    taken from the newest step alone, or a step after the one asked about leaking into the answer."""
    store = Store()
    store.cells = {1: {(0, 0): "free", (1, 0): "occupied"}, 3: {(0, 0): "occupied"}, 5: {(2, 0): "free"}}
    assert store.latest_states(2) == {(0, 0): ("free", 1), (1, 0): ("occupied", 1)}
    assert store.latest_states(4) == {(0, 0): ("occupied", 3), (1, 0): ("occupied", 1)}
    assert store.latest_states() == {(0, 0): ("occupied", 3), (1, 0): ("occupied", 1), (2, 0): ("free", 5)}
    assert store.latest_states(0) == {} and store.states_of([1, 3]) == store.latest_states(4)


# -- the answerers ------------------------------------------------------------------------------------------------
def mug_store() -> Store:
    """Steps 0-5 in the phases docked, undocking, undocking, cleaning, cleaning, docking; a red mug seen at step 1 at
    (1, 2) and at step 4 at (3, 2.00004), a blue one at step 2 at (5, 5)."""
    phases = ["docked", "undocking", "undocking", "cleaning", "cleaning", "docking"]
    seen = {1: det("mug", "red", (1.0, 2.0, 0.9), "a"), 4: det("mug", "red", (3.0, 2.00004, 0.9), "b"),
            2: det("mug", "blue", (5.0, 5.0, 0.9), "c")}
    store = Store()
    store.begin_task(0, "2026-03-02T08:00:00", 0.0, "clean")
    for k, phase in enumerate(phases):
        store.add_step(step(k, f"2026-03-02T08:00:0{k}", *([seen[k]] if k in seen else []), phase=phase))
    return store


EVENTS = {"when you left": EventRule("last", "undocking"), "when you finished": EventRule("last", "cleaning"),
          "when you paused": EventRule("first", "paused")}


def where(wording, kind="mug", color="red"):
    return {"template": "object_where", "slots": {"kind": kind, "attributes": {"color": color}, "event": wording}}


def test_object_where_answers_the_latest_sighting_at_or_before_the_event():
    """Contract (plan §4.7): with the job's events, the answer is the xy (4 decimals) of the latest sighting at or
    before the step the event resolves to over the last task's steps, `unknown` for a moment that never occurs or a
    description nothing matches. Hand case: 'when you left' is the last undocking step, 2, so the red mug is at (1, 2);
    'when you finished' is step 4, so (3, 2.0), the 2.00004 rounded away; the blue mug, seen at step 2 itself, is at
    (5, 5) ('at or before'); 'when you paused' never occurs. Regression caught: the event ignored (always the latest),
    a sighting after the event used, the event's own step left out, or a missing moment answered with a place."""
    store, ctx = mug_store(), {"now": None, "latest": [], "robot": ROBOT, "events": EVENTS}
    assert table_answers.answer(store, where("when you left"), ctx) == {"type": "position", "xy": [1.0, 2.0]}
    assert table_answers.answer(store, where("when you finished"), ctx) == {"type": "position", "xy": [3.0, 2.0]}
    assert table_answers.answer(store, where("when you paused"), ctx) == {"type": "unknown"}
    assert table_answers.answer(store, where("when you left", color="green"), ctx) == {"type": "unknown"}
    assert table_answers.answer(store, where("when you left", color="blue"), ctx) == {"type": "position",
                                                                                      "xy": [5.0, 5.0]}  # at step 2
    with pytest.raises(ValueError, match="when you vanished"):
        table_answers.answer(store, where("when you vanished"), ctx)


def test_object_where_without_events_is_the_latest_sighting_whatever_time_the_text_names():
    """Contract (`current_view`): with no event table the answer is the latest sighting in the store, whichever moment
    the question names, even one that never occurs in the job. Regression caught: a memoryless baseline that obeys the
    question's time, so the memory-needed gate could pass vacuously (VISION rule 4)."""
    ctx = {"now": None, "latest": [], "robot": ROBOT, "events": None}
    for wording in ("when you left", "when you paused", "when you vanished"):
        assert table_answers.answer(mug_store(), where(wording), ctx) == {"type": "position", "xy": [3.0, 2.0]}


def corridor_states(*occupied, absent=()):
    """Every fine cell of the vacuum's corridor from (0, 0) heading +x to 3 m `free`, but the `occupied` ones, and
    those in `absent` (not observed)."""
    cells = grid.corridor((0.17, 0.0), 0.0, ROBOT.body.width_m, 3.0)
    return {(u, v): "occupied" if (u, v) in occupied else "free" for u, v, _ in cells if (u, v) not in absent}


def test_path_ahead_reads_only_the_latest_frame():
    """Contract (design §8.1: an answer about 'now' from older frames is a guess about space that may have changed): the
    answer is `grid.path_ahead` over the cells of the steps in `ctx["latest"]` alone. Hand case: the vacuum at the
    origin heading +x, its front edge at x = 0.17. Step 1 (an older frame) saw a thing in cell (10, 0) that step 3 (the
    latest) did not observe, and step 3 saw one in cell (22, 0), whose centre is 2.25 - 0.17 = 2.08 m along: 2.03. A
    reader of every step would stop at the older thing (0.88 m along: 0.83). Regression caught: the answer read from
    frames older than the latest one."""
    store = Store()
    store.cells[1] = corridor_states((10, 0))
    store.cells[3] = corridor_states((22, 0), absent={(10, v) for v in range(-2, 2)})
    store.add(step(3, "2026-03-02T08:00:03"))
    ctx = {"now": store.steps[3], "latest": [3], "robot": ROBOT, "events": None}
    ask = {"template": "path_ahead", "slots": {"cap": 3.0}}
    assert table_answers.answer(store, ask, ctx) == {"type": "number", "value": 2.03}
    assert table_answers.answer(store, ask, {**ctx, "latest": [1, 3]}) == {"type": "number", "value": 0.83}


def test_a_system_answers_now_from_the_frame_it_was_just_sent(tmp_path):
    """Contract (plan §4.7): a table system's `now` answer reads the cells of the steps of its last observe, one frame,
    whatever the store holds of earlier frames. Hand case, as above: step 1 saw a thing in cell (10, 0), step 3 a thing
    in cell (22, 0) (the cells are put in the store by hand, as a depth image would have made them), each step sent in
    an `observe` of its own, and asked after step 3, `memory_table` answers 2.03, not the older thing's 0.83.
    Regression caught: a system that hands the answerer every step it has kept, though the gold reads the latest
    frame alone."""
    system = started(MemoryTable, tmp_path, events=EVENTS_OF_THE_JOB)
    system.observe([base_step(0, 0.0, 0.0)])
    system.observe([base_step(1, 0.0, 0.0)])
    system.store.cells[1] = corridor_states((10, 0))
    system.observe([base_step(2, 0.0, 0.0)])
    system.observe([base_step(3, 0.0, 0.0)])
    system.store.cells[3] = corridor_states((22, 0), absent={(10, v) for v in range(-2, 2)})
    assert system.ask("during", {"k": 3, "clock": ""}, [ahead()]) == {"q1": {"type": "number", "value": 2.03}}


def test_window_of_returns_the_observe_call(tmp_path):
    """Contract (E §1, `Log.window_of`): the log remembers which keyframes one `observe` call carried (the harness sends
    one frame a call, so a window of one; a caller that observes a look window in one call gets that window), so a past
    moment is read over the window that held it. Hand case: three observes of 8, 3 and 8 steps (k 0-7, 8-10, 11-18):
    `window_of(9)` is [8, 9, 10], `window_of(0)` the first eight, `window_of(18)` the third's; a step no call carried
    has none ([]). `current_view` keeps a fresh store a call, so only its last window is there. Regression caught: a
    past moment read over the wrong steps (the whole log, or a window cut by the step count)."""
    for cls, remembered in ((MemoryTable, {9: [8, 9, 10], 0: list(range(8)), 18: list(range(11, 19)), 99: []}),
                            (CurrentView, {9: [], 0: [], 18: list(range(11, 19)), 99: []})):
        system = started(cls, tmp_path / cls.__name__, events=EVENTS_OF_THE_JOB)
        for lo, hi in ((0, 8), (8, 11), (11, 19)):
            system.observe([base_step(k, 0.0, 0.0) for k in range(lo, hi)])
        assert {k: system.store.window_of(k) for k in remembered} == remembered, cls.__name__


def test_the_mapper_keeps_only_zone_cells(tmp_path):
    """Contract (E §1, the review's M3): a mapper given the job's zones keeps, for each zone band, only the fine cells
    inside the polygons of the zones of that band, and for a flier no robot-band cell at all: a drone frame at 8 m
    observes ten thousand fine cells and a long window holds hundreds of frames, so whole frames would come to gigabytes
    a table process. Hand case: the drone 8 m up at (3, 8.5), its camera straight down, over the plaza (a slab 0.15 m
    high at x 1..5, y 6.5..10.5) whose drop zone, 24 x 24 = 576 fine cells on the ground 0.15, lies whole in the
    frame; three steps of that frame. Each holds the drop zone's 576 cells, all free, on its band (0.15, (0.3, 3.5)),
    and on the bays' floor band (0, (0.3, 3.5)) only cells of the bays' zones (B2 whole and the edge of B1 that the
    frame reaches), fewer than the two zones' 704, though the frame observes over 10,000 cells; and `Store.cells`, the
    robot band, is empty for each step. Regression caught: gigabytes of cells per table process on a long drone case,
    or a robot band computed for a flier."""
    drone = robots.ROBOTS["drone"]
    (tmp_path / "camera.json").write_text(json.dumps(robots.camera_doc(drone)), encoding="utf-8")
    cam = Camera.load(tmp_path / "camera.json")
    pos, rot = robots.camera_pose(drone, (3.0, 8.5, 8.0), 0.0, -math.pi / 2)
    write_png16(tmp_path / "hover.png", render_depth(cam, pos, rot, [box(1.0, 5.0, 6.5, 10.5, 0.0, 0.15)]))
    frame = {"k": 0, "clock": "2026-03-02T08:00:00", "status": {"phase": "inspect", "task": 0}, "depth": "hover.png",
             "pose": {"pos": [3.0, 8.5, 8.0], "rot": [1.0, 0.0, 0.0, 0.0], "yaw_deg": 0.0},
             "camera": {"pos": list(pos), "rot": list(rot), "pitch_deg": -90.0}}
    mapper = Mapper(drone, cam, tmp_path, zk.MAP, [zk.DROP, zk.BAY], [zk.LANES])
    store, drop, bays = Store(mapper), (0.15, (0.3, 3.5)), (0.0, (0.3, 3.5))
    for k in range(3):
        store.add_step({**frame, "k": k})
    whole = grid.observe(grid.points(read_png16(tmp_path / "hover.png"), cam, pos, rot), *drop)
    assert set(mapper.zone_cells) == {drop, bays} and len(whole) > 10_000
    assert all(set(cells) == set(zones.zone_cells(zk.DROP_ZONE)) for cells in store.zone_cells[drop].values())
    assert all(state == "free" for cells in store.zone_cells[drop].values() for state in cells.values())
    bay_cells = {c for z in (zk.B1, zk.B2) for c in zones.zone_cells(z)}
    kept = [set(cells) for cells in store.zone_cells[bays].values()]
    assert len(kept) == 3 and set(zones.zone_cells(zk.B2)) < kept[0] < bay_cells and kept[0] == kept[1] == kept[2]
    assert store.cells == {k: {} for k in range(3)}


def test_unknown_templates_raise():
    """Contract (map SG-H22): a template no answerer is registered for raises, naming it, instead of answering
    `unknown` silently. Regression caught: a family answered `unknown` by every table baseline for want of an answerer,
    which no gate would show as anything but a weak system."""
    ctx = {"now": None, "latest": [], "robot": ROBOT, "events": None}
    with pytest.raises(KeyError, match="no_such_family"):
        table_answers.answer(Store(), {"template": "no_such_family", "slots": {}}, ctx)
    assert {"path_ahead", "object_where"} <= set(table_answers.ANSWERERS)

