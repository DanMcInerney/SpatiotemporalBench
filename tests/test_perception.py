"""The oracle channel: what it carries, what it must not carry, and the true bounding-box centres it reports."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from builders import CHAIR, DOOR, MUG, MUG2, TABLE, keyframe, minimal_timeline
from seen_bounds import use
from stbench import episode as ep
from stbench import perception as P
from stbench.camera import Camera
from stbench.leak import PERCEPTION_KEYS, find_leaks, leak_scan
from stbench.robots import ROBOTS, camera_doc
from stbench.seeding import make_rng
from stbench.timeline import Ego, Task, TaskPlan

CAM = Camera.load(camera_doc(ROBOTS["helper"]))
ALL = [TABLE, CHAIR, MUG, MUG2, DOOR]


def pose(k: int, pos, rot=(1, 0, 0, 0)) -> dict:
    """A pose row of keyframe k with the camera at `pos`: all `detection` reads of it is `k` (the time is the
    timeline's) and the camera block."""
    return {"k": k, "camera": {"pos": list(pos), "rot": list(rot)}}


def long_timeline(n: int):
    """`minimal_timeline`'s world with one task of `n` keyframes 1.5 s apart (the helper in the kitchen)."""
    kfs = [keyframe("helper", k, 1.5 * k, (3.4, 1.0, 0.0), 0.0, task=0, phase="waiting" if k == 0 else "navigating")
           for k in range(n)]
    return replace(minimal_timeline(), ego=Ego(kfs, [Task(0, 0, n, TaskPlan("fetch", {}), ())], []))


def rec(eid: str, px: int = 500, bbox=(100, 100, 140, 140)) -> dict:
    return {"id": eid, "px": px, "bbox": list(bbox), "depth_med_m": 2.0, "lum": 0.3, "centroid_cam": [0, 0, 2.0],
            "centroid_map": [0, 0, 0]}


def vis_rows(tl, entities_per_frame: list[list[dict]]) -> list[dict]:
    return [{"k": k, "entities": e} for k, e in enumerate(entities_per_frame)]


def build(tl, entities_per_frame):
    poses = ep.pose_rows(tl)[: len(entities_per_frame)]
    return P.build(tl, vis_rows(tl, entities_per_frame), poses, CAM)


def test_center_cam_is_the_true_bounding_box_centre_not_a_pixel_centroid():
    """A mug (0.1 m tall) based at z = 0.75 has its box centre at z = 0.80. For a camera at (0.6, 1.9, 0.8) looking
    along +x, that centre is 1.5 m straight ahead of an axis-aligned camera: (x_cv, y_cv, z_cv) = (0, 0, 1.5). Move the
    camera 0.4 m to the right (y = 1.5, so the mug is 0.4 m to its left): x_cv = -0.4. The chair (0.9 m tall, base at
    z = 0) has its centre at z = 0.45: seen from (0.155, 1.9, 0.45) it is 1.0 m ahead. Regression caught: a centre
    taken at the base position or the top, which would shift y_cv by 0.05 and 0.45, or a pixel centroid in its place."""
    tl = minimal_timeline()
    ahead = pose(0, [0.6, 1.9, 0.8])
    assert P.detection(tl, rec(MUG), "00000000", ahead)["center_cam"] == pytest.approx([0, 0, 1.5], abs=1e-6)
    right = pose(0, [0.6, 1.5, 0.8])
    assert P.detection(tl, rec(MUG), "00000000", right)["center_cam"] == pytest.approx([-0.4, 0, 1.5], abs=1e-6)
    low = pose(0, [0.155, 1.9, 0.45])
    assert P.detection(tl, rec(CHAIR), "00000000", low)["center_cam"] == pytest.approx([0, 0, 1.0], abs=1e-6)


def test_center_map_is_the_same_box_centre_in_the_map_frame():
    """Contract: `center_map` is the true bounding-box centre in the map frame (base centre plus half the height), so a
    system is not tested on a frame transform it could do with the pose; it agrees with `center_cam` through the pose.
    A mug at (2.1, 1.9, 0.75), 0.1 m tall, has its centre at (2.1, 1.9, 0.80). Regression caught: a centre taken at
    the base, or a map-frame key that disagrees with the camera-frame one."""
    from stbench.camera import cam_to_map

    tl = minimal_timeline()
    at = pose(0, [0.6, 1.5, 0.8])
    det = P.detection(tl, rec(MUG), "00000000", at)
    assert det["center_map"] == pytest.approx([2.1, 1.9, 0.8], abs=1e-6)
    assert cam_to_map(det["center_cam"], at["camera"]["pos"], at["camera"]["rot"]) == pytest.approx(
        det["center_map"], abs=1e-3)


def test_center_cam_follows_the_placement_at_the_keyframes_time():
    """The mug moves at t = 3600 s from (2.1, 1.9) to (2.5, 1.7). A camera at (0, 1.7, 0.8) sees it 2.5 m ahead and
    level afterwards, 2.1 m ahead and 0.2 m to the right before. The time is the keyframe's own (keyframe 0 is at 0 s
    and keyframe 3 at 7200 s in the fixture). Regression caught: reading the first placement for late frames, which
    reports where the mug used to be."""
    tl = minimal_timeline()
    before = P.detection(tl, rec(MUG), "k", pose(0, [0.0, 1.7, 0.8]))
    after = P.detection(tl, rec(MUG), "k", pose(3, [0.0, 1.7, 0.8]))
    assert before["center_cam"] == pytest.approx([-0.2, 0, 2.1], abs=1e-6)
    assert after["center_cam"] == pytest.approx([0, 0, 2.5], abs=1e-6)


def test_extent_is_the_entity_size_and_attributes_are_visual_only():
    """extent_m is the entity's size; attributes keep colour and size and drop everything that names an instance (a
    furniture name, a door's rooms). Regression caught: an attribute that names an instance, which would identify it
    and defeat the memory test."""
    tl = minimal_timeline()
    at = pose(0, [0.6, 1.9, 0.8])
    mug = P.detection(tl, rec(MUG), "k", at)
    assert mug["extent_m"] == [0.1, 0.1, 0.1] and mug["attributes"] == {"color": "red"} and mug["kind"] == "mug"
    assert P.detection(tl, rec(TABLE), "k", at)["attributes"] == {}  # {"name": "dining table"} is not visual
    assert P.detection(tl, rec(DOOR), "k", at)["attributes"] == {}  # {"between": [...]} names two rooms
    assert P.visual_attributes({"color": "blue", "size": "large", "name": "x", "receptacles": ["a"]}) == \
        {"color": "blue", "size": "large"}


def test_only_doors_carry_state_and_it_is_the_open_flag():
    """A door detection says whether it is open; no other kind has a state key. Regression caught: a state key on every
    detection (the leak table then lists a key nothing but a door writes), or a door that hides its open flag."""
    tl = minimal_timeline()
    at = pose(0, [0.6, 1.9, 0.8])
    assert P.detection(tl, rec(DOOR), "k", at)["state"] == {"open": True}
    assert "state" not in P.detection(tl, rec(MUG), "k", at)


def test_only_seen_entities_are_detected():
    """215 px is detected; 214 px and a single pixel (the dead band of the channel and of the compiler) are not. A
    detection of something the compiler calls unseen (0 px) or below the channel's threshold would let the oracle know
    more than the questions allow; 215..268 px it detects, and the compiler skips any question hinging on it.
    Regression caught: a detection threshold that drifts from the compiler's seen bound."""
    tl = minimal_timeline()
    oracle, ids = build(tl, [[rec(MUG, 215), rec(MUG2, 214), rec(CHAIR, 1)]])
    assert [d["kind"] for d in oracle[0]["detections"]] == ["mug"]
    assert list(ids[0]["ids"].values()) == [MUG]


def test_the_oracle_channel_lists_exactly_the_records_seen_by_their_kind(monkeypatch):
    """Contract (V0 section 6, per kind): a record is detected from its KIND's seen bound. With mug = (20, 200) and
    chair = (40, 600) px, one frame holds a red mug at 210 px (detected: the global 215 would miss it), a blue mug at
    199 (not), the chair at 300 (not: the global rule would detect it), the door at 215 (detected, global bound) and
    the table at 214 (not). The kinds detected are door and mug, and the ids file names the red mug and the door.
    Under the global bounds the same frame detects the chair and the door. Regression caught: the channel judging
    every kind by 215 px, so a table of bounds would change the compiler but not what a system is shown."""
    tl = minimal_timeline()
    frame = [[rec(MUG, 210), rec(MUG2, 199), rec(CHAIR, 300), rec(DOOR, 215), rec(TABLE, 214)]]
    oracle, _ = build(tl, frame)
    assert sorted(d["kind"] for d in oracle[0]["detections"]) == ["chair", "door"]  # the global bounds
    use(monkeypatch, mug=(20, 200), chair=(40, 600))
    oracle, ids = build(tl, frame)
    assert sorted(d["kind"] for d in oracle[0]["detections"]) == ["door", "mug"]
    assert sorted(ids[0]["ids"].values()) == sorted([MUG, DOOR])


def test_detection_keys_are_random_unique_per_frame_fresh_every_frame_and_the_order_is_shuffled():
    """Over 40 identical frames of five entities: keys are 8 lowercase hex characters and unique within a frame; one
    entity gets a different key nearly every frame; and the detection order is not always the sorted entity order. The
    chance that 40 shuffles of five items all come out sorted is (1/120)^40. Regression caught: a persistent key (it
    would be an id), a key repeated within a frame, or an order that reveals ids by position."""
    tl = long_timeline(40)
    poses = ep.pose_rows(tl)
    oracle, ids = P.build(tl, vis_rows(tl, [[rec(e) for e in ALL] for _ in range(40)]), poses, CAM)
    orders, mug_keys = set(), []
    for row, idrow in zip(oracle, ids, strict=True):
        keys = [d["d"] for d in row["detections"]]
        assert len(set(keys)) == 5 and all(len(k) == 8 and set(k) <= set("0123456789abcdef") for k in keys)
        orders.add(tuple(idrow["ids"][k] for k in keys))
        mug_keys.append(next(k for k, v in idrow["ids"].items() if v == MUG))
    assert len(orders) > 10
    assert len(set(mug_keys)) == 40


def test_the_ids_file_maps_every_key_to_its_entity():
    """perception_ids.jsonl joins a detection to its entity: the kind of the mapped entity equals the detection's kind,
    and every key in the oracle row is in the ids row and vice versa. Regression caught: an ids file written from a
    different shuffle than the oracle's, which hands the oracle baseline the wrong entity."""
    tl = minimal_timeline()
    oracle, ids = build(tl, [[rec(e) for e in ALL]])
    assert {d["d"] for d in oracle[0]["detections"]} == set(ids[0]["ids"])
    for d in oracle[0]["detections"]:
        assert tl.entity(ids[0]["ids"][d["d"]]).kind == d["kind"]


def test_keys_are_reproducible_for_a_script_and_differ_between_scripts():
    """The generator is seeded from the script's private inputs (map, seed, robot, job): the same inputs give
    the same file (determinism), and another seed, robot or job gives other keys (so keys from one script say nothing
    about another). Regression caught: a clock- or process-seeded draw (a rerun changes the file) or a seed that leaves
    out an input (two scripts share their keys)."""
    tl = minimal_timeline()
    a, _ = build(tl, [[rec(e) for e in ALL]])
    b, _ = build(tl, [[rec(e) for e in ALL]])
    assert a == b
    for change in ({"seed": 2}, {"robot": "vacuum"}, {"job": "helper/2"}):
        other = replace(tl, episode=replace(tl.episode, **change))
        c, _ = build(other, [[rec(e) for e in ALL]])
        assert c != a, change


def test_the_oracles_draw_is_the_make_rng_of_the_frozen_seed_parts():
    """Contract: each keyframe's keys and shuffle are drawn from `make_rng("perception", map, seed, robot, job,
    "0.11.0", "oracle/4", k)`: the frozen parts `PERCEPTION_SEED_KEY` are written out here and the draw is recomputed
    by hand (n keys from `integers`, then `permutation`), so a rerun of `perceive` over a released script gives its
    `oracle.jsonl` byte for byte; nothing is recorded, the draw is recomputed. Regression caught: a part of the seed
    dropped, reordered or re-valued, which redraws every detection key and shuffle of the released data."""
    assert P.PERCEPTION_SEED_KEY == ("0.11.0", "oracle/4")
    tl = minimal_timeline()
    frames = [[rec(e) for e in ALL], [rec(MUG), rec(DOOR)], [rec(TABLE)]]
    oracle, ids = build(tl, frames)
    e = tl.episode
    for k, entities in enumerate(frames):
        rng = make_rng("perception", e.map, e.seed, e.robot, e.job, "0.11.0", "oracle/4", k)
        keys = [f"{int(x):08x}" for x in rng.integers(0, 1 << 32, size=len(entities))]
        order = rng.permutation(len(entities))
        assert [d["d"] for d in oracle[k]["detections"]] == [keys[i] for i in order]
        assert set(ids[k]["ids"]) == set(keys)
        assert [ids[k]["ids"][d["d"]] for d in oracle[k]["detections"]] == [entities[i]["id"] for i in order]


def test_a_keyframes_keys_do_not_depend_on_the_frames_before_it():
    """Contract (plan §2.7): each keyframe's detection keys and shuffle are drawn from a generator of its own (the
    script's inputs and its script `k`, no tier and no profile), so building keyframe 3 alone gives the row the whole
    script has there, whatever frames 0-2 held (the keys do depend on the keyframe's place in its script, which is why
    the file is never delivered whole and the stream protocol re-keys at delivery). Regression caught: one generator
    for the whole file, whose keys at keyframe 3 depend on how many detections frames 0-2 held."""
    tl = long_timeline(6)
    frames = [[rec(e) for e in ALL], [rec(MUG), rec(DOOR)], [rec(TABLE)], [rec(e) for e in ALL], [rec(CHAIR)], []]
    whole, whole_ids = build(tl, frames)
    row = {"k": 3, "entities": frames[3]}
    alone, alone_ids = P.build(tl, [row], [ep.pose_rows(tl)[3]], CAM)
    assert alone == [whole[3]] and alone_ids == [whole_ids[3]]


def test_the_leak_allowlist_is_exactly_the_keys_the_oracle_writes():
    """Contract (review Q-3, V-13): `leak.PERCEPTION_KEYS` is the key set of `oracle.jsonl` and no other (the names
    under `attributes` are free words, not keys). Hand case: a frame of a table, a chair, two mugs and a door writes a
    row's `k`, `clock` and `detections`, a detection's `d`, `kind`, `attributes`, `bbox_px`, `center_cam`, `center_map`
    and `extent_m`, and the door's `state` with its `open` flag. Regression caught: a key the writer adds that the table
    misses (every perceived script then fails the scan), or one the table keeps after the writer dropped it (the
    never-written `visible_fraction`, which would let a field of that name reach a system unnoticed)."""

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                if k != "attributes":
                    yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    oracle, _ = build(minimal_timeline(), [[rec(e) for e in ALL]])
    assert set(keys(oracle)) == PERCEPTION_KEYS


def test_a_recognisable_detection_carries_its_floor_hull_and_the_others_null():
    """Contract (FD-0.7.0 review, change 8): each detection has a `floor_hull`: the convex hull of the (x,
    y) of the thing's pixels in that frame (`nearest_part.floor_hull`: the map frame, to the millimetre,
    counter-clockwise), when the thing is recognisable there, else null; with no pictures given the key is null
    throughout. By hand: a mug seen at 1,500 px in a box clear of the edge whose points are the corners of a unit square
    at z 0, its centre at z 0.6 and one at (1.5, 1.5, 1.0) has the hull [[1, 1], [2, 1], [2, 2], [1, 2]]; the chair at
    500 px (seen, not recognisable) has null and `points_of` is never asked of it. Regression caught: a hull for a
    glimpse (a polygon of a speck), a hull in another winding or frame than the gold's, or a detection without the key.
    """
    tl = minimal_timeline()
    asked = []

    def points_of(k, eid):
        asked.append((k, eid))
        return np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.5, 1.5, 0.6],
                         [1.5, 1.5, 1.0]])

    frame = [[rec(MUG, 1500), rec(CHAIR, 500)]]
    oracle, ids = P.build(tl, vis_rows(tl, frame), ep.pose_rows(tl)[:1], CAM, points_of)
    by_kind = {d["kind"]: d for d in oracle[0]["detections"]}
    assert by_kind["mug"]["floor_hull"] == [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
    assert by_kind["chair"]["floor_hull"] is None and asked == [(0, MUG)]
    plain, _ = build(tl, frame)
    assert [d["floor_hull"] for d in plain[0]["detections"]] == [None, None]


def test_the_floor_hull_passes_the_leak_scan_and_names_no_one(tmp_path):
    """Contract (`leak.PERCEPTION_KEYS`): a hull is numbers in the map frame, so a script whose oracle carries it passes
    the scan. Regression caught: a perceived script that fails the scan for the new key (every dataset unreleasable),
    or a hull field that carries a name."""
    row = {"k": 0, "clock": "2026-05-14T08:00:00", "detections": [{
        "d": "00000000", "kind": "mug", "attributes": {}, "bbox_px": [1, 2, 3, 4], "center_cam": [0, 0, 1],
        "center_map": [1, 1, 1], "extent_m": [0.1, 0.1, 0.1], "floor_hull": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]}]}
    tl = minimal_timeline()
    paths = ep.write_script(tmp_path, tl)
    ep.write_jsonl(paths.oracle, [row], sort_keys=True)
    assert find_leaks(paths.root, [tl.episode.map]) == []


def test_written_oracle_carries_no_ids_or_names_and_passes_the_leak_scan(tmp_path):
    """End to end through the files: the agent-facing oracle.jsonl has no entity id, no furniture name and no room
    pair, only allowlisted keys, no `t`, and the leak scan of the whole script is clean; the private ids file is where
    the ids are. Regression caught: a writer that puts an id, a name or the script time into the file a system reads."""
    tl = minimal_timeline()
    paths = ep.write_script(tmp_path, tl)
    ep.write_jsonl(paths.visibility, vis_rows(tl, [[rec(e) for e in ALL]] * 6))
    seen, n = P.write(paths)
    assert (seen, n) == (5, 30)
    text = paths.oracle.read_text(encoding="utf-8")
    for needle in ("E000", "dining table", "hall", "kitchen", "apartment_a", "truth"):
        assert needle not in text
    leak_scan(paths.root, [tl.episode.map])
    assert find_leaks(paths.root, [tl.episode.map]) == []
    row = json.loads(text.splitlines()[0])
    assert set(row) == {"k", "clock", "detections"}  # no `t`: the script time never reaches a system
    ids_text = paths.perception_ids.read_text(encoding="utf-8")
    assert "E0003" in ids_text  # the private join file is where ids belong
