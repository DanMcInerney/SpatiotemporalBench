"""The timeline: byte-stable JSON, in-run section hashes, rotations, half-open intervals and the query helpers."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import pytest

from builders import CHAIR, DOOR, MUG, MUG2, TABLE, TASK2_T, minimal_timeline
from stbench import timeline as T
from stbench.timeline_ego import DecisionPoint


def full_timeline() -> T.Timeline:
    """`minimal_timeline` with every field of the timeline set to something other than its default: a region kind that
    is not its name, an asset and an open underside on entities, decision points, a float status key."""
    tl = minimal_timeline()
    tl.regions[0] = replace(tl.regions[0], kind="cooking room")
    tl.entities[1] = replace(tl.entities[1], asset="chair_oak", under_m=0.42)
    tl.ego = replace(tl.ego, decisions=[DecisionPoint(1, "before_doorway"), DecisionPoint(4, "room_done")])
    tl.ego.keyframes[2].status["tether_m"] = 1.5
    return tl


def test_timeline_round_trip_is_byte_stable(tmp_path):
    """Contract: save and load of a timeline with every new field (region kind, entity asset and under_m, base, yaw,
    camera pitch, task, phase, status, tasks, decisions) gives identical bytes and the same two hashes, and the loaded
    timeline equals the saved one. Regression caught: a field dropped on load, or written in another order."""
    tl = full_timeline()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    T.save(tl, a)
    loaded = T.load(a)
    T.save(loaded, b)
    assert a.read_bytes() == b.read_bytes()
    assert loaded == tl
    assert (loaded.world_hash, loaded.ego_hash) == (tl.world_hash, tl.ego_hash)


def test_floats_are_written_with_six_decimals():
    """Contract: quantised floats keep the file independent of float noise: 0.1 + 0.2 is written as 0.3. Regression
    caught: float noise reaching the file (0.30000000000000004), which changes its bytes and so its hashes from one
    platform or computation order to the next."""
    tl = minimal_timeline()
    tl.placements[0] = replace(tl.placements[0], pos=(0.1 + 0.2, 1.9, -0.0))
    assert '"pos": [0.3, 1.9, 0.0]' in T.dumps(tl)


def test_sections_have_independent_in_run_content_hashes():
    """Contract: `world_hash` and `ego_hash` each hash their own section, for comparing two timelines in one process
    (nothing writes them to the file): editing a placement changes only the world hash. Regression caught: one hash
    over both sections, where a world edit would also change the ego hash and the reverse."""
    tl = minimal_timeline()
    edited = minimal_timeline()
    edited.placements[0] = replace(edited.placements[0], pos=(2.2, 1.9, 0.0))
    assert edited.ego_hash == tl.ego_hash
    assert edited.world_hash != tl.world_hash


def test_episode_info_is_outside_both_hashes():
    """Contract: robot and job live in `episode`, not in a hash, so changing them cannot change the world or ego hash.
    Regression caught: the script's inputs folded into a hash, so relabelling a script (another robot or job over the
    same world) would change the hash that tells two worlds apart."""
    tl = minimal_timeline()
    other = replace(tl, episode=replace(tl.episode, robot="vacuum", job="vacuum/1", seed=9))
    assert other.world_hash == tl.world_hash and other.ego_hash == tl.ego_hash


def test_status_and_decisions_are_in_the_ego_hash():
    """Contract: golds read a keyframe's status and the decision points, so editing one status value, one phase, one
    decision kind or one task parameter changes `ego_hash`, and none of them touches `world_hash`. Regression caught:
    a golds-relevant ego field left out of the hash that tells two egos apart in a run."""
    d = T.to_dict(full_timeline())
    base = T.from_dict(copy.deepcopy(d))
    for edit in (lambda e: e["keyframes"][1]["status"].update(battery_pct=51),
                 lambda e: e["keyframes"][1].update(phase="waiting"),
                 lambda e: e["decisions"][0].update(kind="room_done"),
                 lambda e: e["tasks"][0]["plan"]["params"].update(room="hall")):
        edited = copy.deepcopy(d)
        edit(edited["ego"])
        loaded = T.from_dict(edited)
        assert loaded.ego_hash != base.ego_hash and loaded.world_hash == base.world_hash


def test_a_timeline_file_of_an_earlier_build_loads_and_its_stamps_are_ignored():
    """Contract: `from_dict` reads only the keys the timeline names. A file written by an earlier build (a `schema`, the
    episode's `map_sha256`, `robot_sha256` and `generator`, a `hash` in each section) loads to the same timeline, and
    whatever those stamps hold is neither checked nor kept. Regression caught: a loader that refuses a file for its
    stamps again, which stops every dataset built before them from loading."""
    tl = full_timeline()
    new = T.to_dict(tl)
    assert list(new) == ["episode", "world", "ego"]
    old = {"schema": "old",
           "episode": {**new["episode"], "map_sha256": "0" * 64, "robot_sha256": "f" * 64, "generator": "old"},
           "world": {"hash": "1" * 64, **new["world"]}, "ego": {"hash": "2" * 64, **new["ego"]}}
    assert T.from_dict(old) == tl
    assert T.loads(json.dumps(old)) == tl


def test_placement_intervals_are_half_open():
    """Contract: a placement covers [t0, t1): the mug is at its first spot at t=3599.9 and its second at exactly t=3600.
    Regression caught: a closed interval, which puts an item in two places at the instant it moves."""
    tl = minimal_timeline()
    assert tl.placement_of(MUG, 3599.9).pos[:2] == (2.1, 1.9)
    assert tl.placement_of(MUG, 3600.0).pos[:2] == (2.5, 1.7)
    assert tl.placement_of(MUG, 0.0) is not None


def test_entities_at_follows_existence():
    """Contract: an entity exists only while a placement covers t; removing one drops it from entities_at. Regression
    caught: an entity that outlives its removal, so a removed item would still be listed (and asked about) after it."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], t1=100.0)  # the second mug is taken away at 100 s
    assert {e.id for e in tl.entities_at(50.0)} == {TABLE, CHAIR, MUG, MUG2, DOOR}
    assert {e.id for e in tl.entities_at(100.0)} == {TABLE, CHAIR, MUG, DOOR}


def test_state_of_returns_the_value_in_force():
    """Contract: states are per key; a closed-then-open door reports each value in its own interval. Regression caught:
    a state read from the wrong interval or the wrong key (an open door reported shut after it opens, or its lock mixed
    with its leaf)."""
    tl = minimal_timeline()
    tl.states = [T.State(DOOR, "open", 0.0, 500.0, True), T.State(DOOR, "open", 500.0, None, False),
                 T.State(DOOR, "locked", 0.0, None, False)]
    assert tl.state_of(DOOR, 499.0) == {"open": True, "locked": False}
    assert tl.state_of(DOOR, 500.0, "open") is False
    assert tl.state_of(CHAIR, 0.0) == {}


def test_values_between_lists_every_value_a_state_has_in_a_span():
    """Contract (V0 §3, the movers' free space): `values_between` answers, through the timeline's own API, which values
    a state key takes at some time in a closed span [t0, t1], so a caller that must be clear of every state a door
    has (a mover's leaf rectangles) need not read the state list. Intervals are half-open: a value that ends at t0 is
    not in the span, one that starts at t1 is. Regression caught: a span that misses the value in force at its start,
    one that includes a value ended before it, a repeat, and a key or entity mixed up with another."""
    tl = minimal_timeline()
    tl.states = [T.State(DOOR, "open", 0.0, 500.0, True), T.State(DOOR, "open", 500.0, None, False),
                 T.State(DOOR, "locked", 0.0, None, False)]
    assert tl.values_between(DOOR, "open", 100.0, 200.0) == [True]
    assert tl.values_between(DOOR, "open", 100.0, 500.0) == [True, False]  # a value starting at t1 is in the span
    assert tl.values_between(DOOR, "open", 500.0, 900.0) == [False]  # a value ending at t0 is not
    assert tl.values_between(DOOR, "locked", 0.0, 9e9) == [False]
    assert tl.values_between(CHAIR, "open", 0.0, 9e9) == []


def test_with_states_replaces_one_entitys_states():
    """Contract (M2, the people overlay): `with_states({e: [s]})` is a copy in which entity `e`'s states are exactly
    `[s]` (read through `state_of`, also when the original's index was already built) and every other entity's states
    are as they were, the original is not changed, and the copy's world hash differs. Regression caught: an overlay that
    cannot add or replace a state without reading the raw list, a copy that keeps the replaced entity's old states or
    serves them from the original's index, or one that drops another entity's states."""
    tl = minimal_timeline()
    tl.states = [T.State(DOOR, "open", 0.0, 500.0, True), T.State(DOOR, "open", 500.0, None, False),
                 T.State(CHAIR, "turned", 0.0, None, False)]
    assert tl.state_of(CHAIR, 100.0) == {"turned": False}  # the original's index is built before the copy is made
    pose = T.State(CHAIR, "pose", 100.0, None, "sitting")
    after = tl.with_states({CHAIR: [pose]})
    assert after.state_of(CHAIR, 50.0) == {} and after.state_of(CHAIR, 100.0) == {"pose": "sitting"}
    assert after.state_of(DOOR, 100.0) == {"open": True} and after.state_of(DOOR, 500.0) == {"open": False}
    assert tl.state_of(CHAIR, 100.0) == {"turned": False}  # the original is untouched
    assert after.world_hash != tl.world_hash
    assert tl.with_states({}).world_hash == tl.world_hash


def test_task_of_uses_half_open_ranges():
    """Contract: tasks are [k0, k1): keyframe 3 starts task 1, and k = 6 is in no task. Regression caught: a closed
    upper bound, where the keyframe that opens a task also belongs to the one before it, or a k past the last task that
    silently takes a task."""
    tl = minimal_timeline()
    assert [tl.task_of(k).index for k in range(6)] == [0, 0, 0, 1, 1, 1]
    with pytest.raises(IndexError):
        tl.task_of(6)


def test_decisions_in_is_half_open_and_in_order():
    """Contract: `decisions_in(k0, k1)` holds the decision points at k0 <= k < k1, in order (a window's decisions).
    Regression caught: a closed upper end, which would hand a case the decision that opens the next task."""
    tl = full_timeline()  # decisions at keyframes 1 and 4
    assert [d.k for d in tl.decisions_in(0, 6)] == [1, 4]
    assert [d.k for d in tl.decisions_in(1, 4)] == [1]
    assert [d.k for d in tl.decisions_in(2, 4)] == []
    assert [d.kind for d in tl.decisions_in(4, 5)] == ["room_done"]


def test_keyframe_lookup_and_clock():
    """Contract: keyframe(k) indexes by k; the clock is start_clock + t rounded half up to the second. Regression
    caught: a clock that truncates (1.5 s would read :01) or rounds up always (3599.4 s would read 09:00:00), or a
    keyframe found by the wrong index."""
    tl = minimal_timeline()
    assert tl.keyframe(3).t == TASK2_T
    assert tl.clock(0.0) == "2026-03-02T08:00:00"
    assert tl.clock(1.5) == "2026-03-02T08:00:02"
    assert tl.clock(3599.4) == "2026-03-02T08:59:59"
    assert tl.clock(86400.0) == "2026-03-03T08:00:00"


def test_region_of_is_none_in_a_wall():
    """Contract: region_of names the room holding a point, and None for a point in a wall or doorway gap. Regression
    caught: a point in a wall given to a neighbouring room, so a robot would count as 'in the kitchen' while standing in
    the wall between it and the hall."""
    tl = minimal_timeline()
    assert tl.region_of((2.0, 1.0)) == "R3"
    assert tl.region_of((4.6, 1.9)) is None  # the wall between kitchen and hall


def test_footprint_uses_base_centre_size_and_yaw():
    """Contract: the table is 0.8 deep along its yaw (north), so its footprint is 0.8 wide in x-local and rotated 90
    degrees; an entity has none at a time it does not exist. Regression caught: a footprint that drops the yaw or swaps
    width and depth (a table facing north blocking the wrong cells), or one returned for an entity before it exists."""
    rect = minimal_timeline().footprint(TABLE, 0.0)
    assert (rect.cx, rect.cy, rect.sx, rect.sy) == (2.1, 1.9, 0.8, 1.4)
    assert rect.yaw == pytest.approx(math.pi / 2, abs=1e-5)
    assert minimal_timeline().footprint(MUG, -1.0) is None


def test_json_has_the_documented_top_level_layout(tmp_path):
    """Contract: the file has episode, world{regions, entities, placements, states} and ego{keyframes, tasks,
    decisions}, with every field of a record always written (the region kind, the entity asset and under_m, the
    keyframe's base, yaw, camera pitch, task, phase and status) in the documented order, and no schema string or
    hash. Regression caught: a field added, dropped or reordered, or a stamp written again, which every reader of
    the file depends on."""
    p = tmp_path / "t.json"
    T.save(minimal_timeline(), p)
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert list(doc) == ["episode", "world", "ego"]
    assert list(doc["episode"]) == ["map", "seed", "robot", "job", "start_clock"]
    assert list(doc["world"]) == ["regions", "entities", "placements", "states"]
    assert list(doc["ego"]) == ["keyframes", "tasks", "decisions"]
    assert list(doc["world"]["regions"][0]) == ["id", "name", "kind", "polygon", "level", "z0", "z1"]
    assert list(doc["world"]["entities"][0]) == ["id", "kind", "attributes", "mobility", "size", "asset", "under_m"]
    assert doc["world"]["entities"][0]["asset"] is None and doc["world"]["entities"][0]["under_m"] == 0.0
    assert set(doc["world"]["states"][0]) == {"entity", "key", "t0", "t1", "value"}
    assert list(doc["ego"]["keyframes"][0]) == ["k", "t", "pos", "rot", "base", "yaw", "cam_pitch", "task", "phase",
                                                "status"]
    assert list(doc["ego"]["tasks"][0]) == ["index", "k0", "k1", "plan", "targets"]


def test_clutter_is_never_describable():
    """Contract (the mobility table): clutter collides as carried, rests on a surface or the floor, obstructs a ground
    robot only on the floor above its climb, and is not describable, so `DESCRIBABLE` is exactly light, carried and
    mover. Regression caught: clutter becoming an askable kind, or a fixed piece that wear paints (map X-H8)."""
    rule = T.MOBILITY["clutter"]
    assert (rule.collides_as, rule.rests_on, rule.obstructs, rule.describable) == (
        "carried", "surface_or_floor", "on_floor_above_climb", False)
    assert T.DESCRIBABLE == ("light", "carried", "mover")
    assert T.MOBILITIES == ("structure", "fixed", "light", "carried", "mover", "clutter")
