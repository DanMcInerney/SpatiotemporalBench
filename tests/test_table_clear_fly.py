"""A table system's answer to the flier's `clear_path` (0.8.0 lane L2: `table.Mapper`'s wall bins and
`table_clear_fly`), against the gold, on rendered depth.

`fly_kit`'s two walls are boxes the drone's own camera sees: ten keyframes at (0, 0, 12), every heading -x (yaw
pi, so the camera shows `WW`, 14.7 m on) but keyframe 3, which heads +x and shows `WE` (19.7 m on). The case
window is the whole script (0-9), its stream cut after 7 and 9, so the look window of the look point 9 is 8-9. At
keyframe 9 the body heads -x: ahead is `WW`, 14.05 m from the edge (x -0.65), seen by keyframe 9 itself; back is
`WE`, 19.05 m from the edge (x 0.65), seen only by keyframe 3.
"""

from __future__ import annotations

import math
from functools import cache

import pytest

from fly_kit import MAP, drone_camera, face, flight, fly_world, rendered_depth, rendered_points
from stbench import clear_fly, clear_move, episode, table_answers
from stbench.render_encode import write_png16
from stbench.robots import ROBOTS
from stbench.table import Mapper, Store

DRONE = ROBOTS["drone"]
ASK = 9


@cache
def facing_west(saw_east: bool):
    """The ten-keyframe script, every heading -x; keyframe 3 heads +x when `saw_east`."""
    return flight(tasks=1, yaws={k: math.pi for k in range(10)} | ({3: 0.0} if saw_east else {}))


@cache
def _images(saw_east: bool, k: int):
    """Keyframe `k`'s rendered depth and its points, once a process (the scripts of `facing_west` are cached)."""
    script = facing_west(saw_east)
    return rendered_depth(script, k), rendered_points(script, k)


def depth(script, k: int):
    return _images(script is facing_west(True), k)[0]


def points(script, k: int):
    return _images(script is facing_west(True), k)[1]


def gold(script, direction: int):
    """The look point 9's moment, the direction hash patched to `direction` (0 ahead, 1 back), over rendered depth."""
    w = fly_world(script, {k: points(script, k) for k in range(10)})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clear_move, "stable_hash", lambda *parts: direction)
        return clear_fly.moment(w, ASK)


def steps(script, tmp_path, ks):
    """The delivered steps of the keyframes `ks`: the pose rows with each one's rendered depth image."""
    rows = episode.pose_rows(script.timeline())
    out = []
    for k in ks:
        write_png16(tmp_path / f"d{k}.png", depth(script, k))
        out.append({**rows[k], "depth": f"d{k}.png"})
    return out


def store_of(script, tmp_path, ks, robot=DRONE, m=MAP):
    store = Store(Mapper(robot, drone_camera(), tmp_path, m))
    for step in steps(script, tmp_path, ks):
        store.add_step(step)
    return store


def ask(store, slots, m=MAP):
    """The registered answerer's reply to the question at keyframe 9 (the template `clear_path`, as a system is
    asked)."""
    ctx = {"now": store.steps[ASK], "latest": [8, 9], "robot": DRONE, "events": None, "map": m}
    return table_answers.answer(store, {"template": "clear_path", "slots": slots}, ctx)


def test_the_mapper_records_the_wall_bins_each_image_saw(tmp_path):
    """Contract (plan L2): a flier's mapper, given the map, records for each step the (wall id, bin) pairs its depth
    saw (`Store.wall_bins`, the same `wall_seen.bins_of` the gold reads). Hand case: keyframe 3 faces +x with
    the camera 11.9 m up pitched down 35 degrees and sees `WE`'s whole face from y -10.3 to 10.3 (the
    horizontal field of view is 16.5 m each side at 19.7 m), bins 10 and 11 among them; keyframe 0 faces -x and
    sees `WW` (bin 10) and none of `WE`. A ground robot's mapper, or one without the map, keeps no record.
    Regression caught: a table system with no wall memory (its `memory_table` answering `unknown` where the
    gold has a distance), or a ground mapper paying for it."""
    script = facing_west(True)
    store = store_of(script, tmp_path, [0, 3])
    assert {("WE", 10), ("WE", 11)} <= store.wall_bins[3] and not any(w == "WW" for w, _ in store.wall_bins[3])
    assert ("WW", 10) in store.wall_bins[0] and not any(w == "WE" for w, _ in store.wall_bins[0])
    assert store_of(script, tmp_path, [3], robot=ROBOTS["vacuum"]).wall_bins == {}
    assert store_of(script, tmp_path, [3], m=None).wall_bins == {}


def test_the_table_answers_what_the_gold_is_when_the_wall_is_in_its_memory(tmp_path):
    """Contract (plan L2: "so `memory_table` reproduces the gold"): fed every step of the window, the table's answer
    to the asked question equals the gold, an answerable one or an `unknown`. Hand cases at keyframe 9 heading
    -x: back meets `WE` (19.05), seen by keyframe 3 alone, scope `memory`: gold and table say 19.05; ahead
    meets `WW` (14.05), seen by keyframe 9, scope `now`: both say 14.05; with keyframe 3 turned away too,
    back's `WE` was never seen: both say `unknown`. Regression caught: a table whose corridor, bin or neighbour
    rule differs from the gold's (the ceiling gate made false), or one that reads the answer from the map
    without having seen the wall."""
    script = facing_west(True)
    store = store_of(script, tmp_path, range(10))
    back = gold(script, 1)
    (cand,) = back.answerable
    assert (cand.gold, cand.scope, cand.evidence_k) == ({"type": "number", "value": 19.05}, "memory", (3,))
    assert ask(store, cand.slots) == cand.gold
    ahead = gold(script, 0).answerable[0]
    assert (ahead.gold, ahead.scope) == ({"type": "number", "value": 14.05}, "now")
    assert ask(store, ahead.slots) == ahead.gold
    blind = facing_west(False)
    unseen = gold(blind, 1)
    assert unseen.answerable == [] and unseen.unknown[0].gold == {"type": "unknown"}
    assert ask(store_of(blind, tmp_path, range(10)), unseen.unknown[0].slots) == {"type": "unknown"}


def test_the_table_answers_a_wall_seen_along_its_face_as_the_gold_does(tmp_path):
    """Contract (final review S1): a wall the memory holds a bin of, however far along from the stretch the corridor
    meets, is a distance to the table as to the gold, and a memory of another wall only is `unknown` to both. Hand
    case at keyframe 9 heading -x, back meeting `WE` at 19.05 m (bin 10): a store that holds only `WE` bin 15
    (five metres along, from keyframe 3) answers 19.05, as the gold does over a keyframe-3 point on `WE` at y 5;
    one that holds only `WW` bin 3 answers `unknown`. Regression caught: the neighbour rule kept in the table
    alone (the ceiling gate broken by 19.05 against `unknown`)."""
    script = facing_west(True)
    store = store_of(script, tmp_path, [ASK])
    w = fly_world(script, {3: face(19.7, [5.0])})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clear_move, "stable_hash", lambda *parts: 1)
        (cand,) = clear_fly.moment(w, ASK).answerable
    assert cand.gold == {"type": "number", "value": 19.05}
    store.wall_bins = {3: frozenset({("WE", 15)})}
    assert ask(store, cand.slots) == cand.gold
    store.wall_bins = {3: frozenset({("WW", 3)})}
    assert ask(store, cand.slots) == {"type": "unknown"}


def test_a_memoryless_table_answers_only_what_its_latest_frame_showed(tmp_path):
    """Contract: `current_view`'s store holds the latest frame alone, so a wall only an earlier frame showed is
    `unknown` to it while the gold (scope `memory`) is a distance, and a wall the frame shows is answered. Hand
    case at keyframe 9: a store of step 9 alone gets back (`WE`, seen at keyframe 3 only) `unknown` and ahead
    (`WW`, seen at 9) 14.05. Regression caught: a memoryless system given a memory it was not sent (the
    memory-needed gate's floor)."""
    script = facing_west(True)
    latest = store_of(script, tmp_path, [9])
    assert ask(latest, gold(script, 1).answerable[0].slots) == {"type": "unknown"}
    assert ask(latest, gold(script, 0).answerable[0].slots) == {"type": "number", "value": 14.05}


def test_a_system_built_without_the_map_or_a_corridor_into_open_air_is_unknown(tmp_path):
    """Contract (`table_answers`: "None when the system was built without one"): with no map the answer is `unknown`,
    and so is a corridor with no wall in the cap, whatever the store holds. Hand cases: the asked question with
    `ctx["map"]` None; heading +y (yaw pi/2) from a store of keyframes that saw both walls. Regression caught:
    a raise for want of a map, or a distance certified into open air."""
    script = facing_west(True)
    store = store_of(script, tmp_path, range(10))
    slots = gold(script, 1).answerable[0].slots
    assert ask(store, slots, m=None) == {"type": "unknown"}
    store.steps[ASK] = {**store.steps[ASK], "pose": {**store.steps[ASK]["pose"], "yaw_deg": 90.0}}
    assert ask(store, slots) == {"type": "unknown"}
