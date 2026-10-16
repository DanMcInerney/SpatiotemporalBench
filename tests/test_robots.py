"""The robot spec: where the camera sits, the documents derived from a robot, the status vocabulary and the rules a
robot's numbers decide. Every expected value is hand arithmetic or a literal from the plan."""

from __future__ import annotations

import json
import math

import pytest

from stbench.camera import DEPTH_STEPS, DEPTH_UNITS
from stbench.leak import RULES
from stbench.mapspec import bundled_maps
from stbench.robots import (
    DOC_KEYS,
    ROBOTS,
    blind_m,
    camera_doc,
    camera_pose,
    check_status,
    flies_over,
    from_doc,
    over_m,
    passes_under,
    robot_doc,
)

VACUUM, DRONE = ROBOTS["vacuum"], ROBOTS["drone"]


def test_camera_sits_at_the_mount():
    """Contract (plan §2.1): `camera_pose` puts the camera at the base plus the mount turned by the heading, rotated by
    yaw then pitch. Hand case: the vacuum's mount (0.13, 0, 0.08) at base (1, 2, 0) heading 90 degrees is (1.0, 2.13,
    0.08), and yaw 90 with pitch +5 gives w = cos 45 cos 2.5 = 0.706434, x = sin 45 sin 2.5 = 0.030844, y = -x, z = w.
    Regression caught: a camera at the base origin (today's body-free head), so a rear corridor includes the robot's own
    body (map X-H20), or a mount not turned with the heading."""
    pos, rot = camera_pose(VACUUM, (1.0, 2.0, 0.0), math.radians(90), math.radians(5))
    assert pos == (1.0, 2.13, 0.08)
    assert rot == (0.706434, 0.030844, -0.030844, 0.706434)


def test_camera_json_is_derived_from_the_robot():
    """Contract (plan §2.1): `sensors/camera.json` is `camera_doc(robot)`: 640 x 480 at 80 degrees gives fx = fy =
    320 / tan 40 degrees = 381.36115, the principal point at the centre, and the depth unit and range from the robot's
    depth block. The vacuum's is today's bytes (`tests/test_episode.py`) plus its range. Regression caught: the
    renderer's hfov and the consumers' fx drifting apart (map X-H7)."""
    assert camera_doc(VACUUM) == {"width": 640, "height": 480, "hfov_deg": 80.0, "fx": 381.36115, "fy": 381.36115,
                                  "cx": 320.0, "cy": 240.0, "depth_unit": "mm", "near_m": 0.01, "far_m": 4.0}


def test_every_far_range_fits_its_depth_unit():
    """Contract (plan §2.1): a robot's far range fits 65,535 steps of its depth unit and is beyond its near range, so a
    depth image of the farthest surface is not 0. Regression caught: depth that reads 0 where a gold reads clear (map
    X-H21): a drone's 600 m in millimetres, or a range that ends before it begins."""
    for r in ROBOTS.values():
        assert r.depth.far_m <= DEPTH_STEPS * DEPTH_UNITS[r.depth.unit], r.id
        assert r.depth.near_m < r.depth.far_m, r.id


def test_vacuum_passes_under_tables_not_the_sofa():
    """Contract (plan §2.1): the vacuum (0.09 m tall, 0.01 m margin) passes under a piece whose underside is at least
    0.10 m: a table (0.64) and the coffee table (0.10) yes, the sofa (0.08) and a solid piece (0) no, and 0.099 no (it
    clears the 0.09 m body but not the margin: the 0.09-0.10 band `under_margin_m` decides); a flier never. Regression
    caught: a vacuum planned under the sofa, or never under a table, or a margin dropped from the rule (the vacuum then
    drives under a piece that jams on its top)."""
    assert passes_under(VACUUM, 0.64) and passes_under(VACUUM, 0.10)
    assert not passes_under(VACUUM, 0.08) and not passes_under(VACUUM, 0.0)
    assert not passes_under(VACUUM, 0.099)
    assert not any(passes_under(DRONE, h) for h in (0.0, 0.5, 100.0))


def test_status_vocabulary_is_closed():
    """Contract (plan §2.1, design §4): a status is the closed phase vocabulary plus the robot's declared keys, each of
    its type (`int` is not a bool, `float` takes an int), and nothing else; every speed names a phase of its robot.
    Regression caught: a status carrying world facts (the keys room, region, obstacle), or a phase and a key the robot
    never declared."""
    ok = {"phase": "cleaning", "battery_pct": 50, "brush": True}
    assert check_status(VACUUM, ok, first=False) == []
    for bad in ({**ok, "phase": "flying"}, {k: v for k, v in ok.items() if k != "phase"},
                {k: v for k, v in ok.items() if k != "brush"}, {**ok, "battery_pct": "50"},
                {**ok, "battery_pct": True}, {**ok, "battery_pct": 50.5}, {**ok, "brush": 1}):
        assert check_status(VACUUM, bad, first=False), bad
    for key in ("room", "region", "obstacle", "wheel_slip"):
        problems = check_status(VACUUM, {**ok, key: "kitchen"}, first=False)
        assert len(problems) == 1 and key in problems[0], key
    flier = {"phase": "cruise", "battery_pct": 80, "tether_m": 0}
    assert check_status(DRONE, flier, first=False) == []
    assert check_status(DRONE, {**flier, "tether_m": 1.5}, first=False) == []
    assert check_status(DRONE, {**flier, "tether_m": True}, first=False)
    for r in ROBOTS.values():
        assert set(r.speeds_mps) <= set(r.status.phases), r.id


def test_each_task_starts_in_the_start_set():
    """Contract (plan §2.1, map X-H4): the first step of a task is in the robot's start set: the vacuum starts docked,
    its battery 90-100 (both ends in) and its brush off. Regression caught: status carried over between tasks (a
    battery at 50 or a cleaning phase at the start), which reveals a window's place."""
    docked = {"phase": "docked", "battery_pct": 50, "brush": False}
    assert check_status(VACUUM, docked, first=True) != []
    assert check_status(VACUUM, {**docked, "battery_pct": 95}, first=True) == []
    assert [check_status(VACUUM, {**docked, "battery_pct": b}, first=True) == [] for b in (89, 90, 100, 101)] == [
        False, True, True, False]
    assert check_status(VACUUM, {**docked, "battery_pct": 95, "brush": True}, first=True) != []
    assert check_status(VACUUM, {"phase": "cleaning", "battery_pct": 95, "brush": False}, first=True) != []
    assert check_status(VACUUM, {"phase": "cleaning", "battery_pct": 50, "brush": True}, first=False) == []


def doc_keys(node) -> set[str]:
    """Every key of every dict inside a JSON document."""
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in doc_keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in doc_keys(v)}
    return set()


def test_robot_doc_round_trips():
    """Contract (plan §2.1): `robot_doc` is the public `robot.json` (every field, tuples as lists, no schema string) and
    `from_doc` is its inverse through JSON text; the `robot.json` of an earlier build, which carries a `schema`, parses
    to the same robot and still passes the leak allowlist; every key of every doc is in `DOC_KEYS`, which is how the
    leak allowlist stays closed, and a doc with one more key (at any depth) is caught by that check. Regression caught:
    a system's robot differing from the gold's, a field that reaches `robot.json` unlisted, or a reader that refuses
    the file of an earlier build."""
    for r in ROBOTS.values():
        doc = robot_doc(r)
        assert "schema" not in doc
        assert from_doc(json.loads(json.dumps(doc))) == r
        old = {**doc, "schema": "old"}
        assert from_doc(old) == r and doc_keys(old) <= DOC_KEYS, r.id
        assert doc_keys(doc) <= DOC_KEYS, r.id
        assert doc_keys({**doc, "extra": 1}) - DOC_KEYS == {"extra"}
        assert doc_keys({**doc, "body": {**doc["body"], "secret": 1}}) - DOC_KEYS == {"secret"}


FIELD_KEYS = {  # the field names, written out: a robot's doc may hold these and the names below, nothing else
    "schema", "id", "name", "flies", "body", "shape", "length_m", "width_m", "height_m", "climb_m", "clearance_m",
    "under_margin_m", "speeds_mps", "turn_dps", "camera", "width", "height", "hfov_deg", "mount_m", "pitch_deg",
    "pitch_min_deg", "pitch_max_deg", "isp", "vignette", "noise_a", "noise_b", "meter_sigma", "target", "exposure_ev",
    "gain_ev", "wb_clamp", "sharpen", "jpeg_quality", "contrast", "black_level", "glare", "denoise_luma_px",
    "denoise_chroma_px", "depth", "unit", "near_m", "far_m", "status", "phases", "keys",
    "start", "phase", "frames", "move_m", "turn_deg", "still_s", "min_dt_s", "tools", "size_m", "offset_m", "sources",
}


def test_doc_keys_are_the_hand_written_field_names_plus_the_phases_and_status_keys():
    """Contract (plan §2.1, the M0.0 brief: the leak allowlist is fail-closed): `DOC_KEYS`, the keys a `robot.json` may
    hold, is the schema's field names as written out here, plus every phase name and every status key of every robot
    (they are keys of `speeds_mps`, `status.keys` and `status.start`), and nothing more. Regression caught: `DOC_KEYS`
    computed from the docs or the records (fail-open: a field added to a record is allowed by accident, not refused by
    the leak scan until someone lists it), or a field added to a record and listed without this list changing."""
    phases = {p for r in ROBOTS.values() for p in r.status.phases}
    status_keys = {k for r in ROBOTS.values() for k in r.status.keys}
    assert phases and status_keys
    assert FIELD_KEYS | phases | status_keys == DOC_KEYS


def test_blind_band_matches_the_geometry():
    """Contract (plan §2.1): `blind_m` is the floor distance from the body's front edge to where the lowest image row
    meets the floor, at the default pitch. Hand case: the vacuum's half vertical field is atan(240 / 381.36) = 32.18
    degrees and its camera is pitched up 5, so the lowest row dips 27.18 degrees below the horizontal, meeting the floor
    0.08 / tan 27.18 = 0.1558 m from the camera, 0.13 m ahead of the centre and 0.17 m behind the front edge: 0.116 m. A
    flier has no floor band. Regression caught: a near-field rule with the pitch's sign flipped (0.08 / tan 37.18 =
    0.105 m, less 0.04: 0.065 m)."""
    assert blind_m(VACUUM) == pytest.approx(0.116, abs=0.001)
    with pytest.raises(ValueError, match="flies"):
        blind_m(DRONE)


def test_flies_over_boundary():
    """Contract (plan-city §2, ported from `profiles`): a flier at base height h clears h minus its clearance, and
    passes over a thing exactly when the thing's top is below that; a ground robot passes over nothing. Hand case: the
    drone (clearance 3.0) at 12.0 clears 9.0, so 8.99 and a 0.15 m sidewalk are flown over and 9.0 and 12.0 are not;
    the vacuum clears nothing. Regression caught: the boundary read one way for furniture and the other for movers, or a
    ground robot that flies over a sidewalk."""
    over = over_m(DRONE, 12.0)
    assert over == 9.0
    assert flies_over(over, 8.99) and flies_over(over, 0.15)
    assert not flies_over(over, 9.0) and not flies_over(over, 12.0)
    assert over_m(VACUUM, 0.0) is None and not flies_over(None, 0.0)


def test_robot_docs_trip_no_leak_rule():
    """Contract (plan §2.1, §2.8): what a robot publishes in `robot.json`, its citations included, holds no entity id,
    no region id, not the word "truth" and no bundled map's name, so the leak scan passes it. Regression caught: a
    `sources` string that M0.1's leak test would refuse in a file this lane froze."""
    for r in ROBOTS.values():
        text = json.dumps(robot_doc(r))
        assert [rule for rule, rx in RULES.items() if rx.search(text)] == [], r.id
        assert [name for name in bundled_maps() if name.lower() in text.lower()] == [], r.id


def test_each_robot_is_consistent_with_its_own_schema():
    """Contract (plan §2.1): the four robots obey the schema's own rules: 640 x 480 at 80 degrees (D8); the default
    pitch inside the gimbal's limits; each speed a positive (min, max); the phases distinct; the start set naming the
    phase and every status key, each a value of its type; a body shape of disc (equal sides) or rect; keyframes at
    least 1 s apart (so every task's clock is unique at whole seconds). Regression caught: a typo in a table that no
    other test reads (a start key the status lacks, a pitch outside its own limits)."""
    for r in ROBOTS.values():
        c, st = r.camera, r.status
        assert (c.width, c.height, c.hfov_deg) == (640, 480, 80.0), r.id
        assert c.pitch_min_deg <= c.pitch_deg <= c.pitch_max_deg, r.id
        assert all(0 < lo <= hi for lo, hi in r.speeds_mps.values()), r.id
        assert len(set(st.phases)) == len(st.phases) and st.start["phase"] in st.phases, r.id
        assert set(st.start) == {"phase", *st.keys}, r.id
        start = {k: (v[0] if isinstance(v, tuple) else v) for k, v in st.start.items()}
        assert check_status(r, start, first=True) == [], r.id
        assert r.body.shape in ("disc", "rect") and (r.body.shape == "rect" or r.body.length_m == r.body.width_m), r.id
        assert r.frames.min_dt_s >= 1.0, r.id
