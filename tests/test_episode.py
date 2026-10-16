"""Script layout (`episode.write_script`): agent-facing versus private, random ids, the private index and the written
files. The layout's paths, the index and the cases are tested in `test_case.py`."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from builders import keyframe, minimal_timeline
from conftest import script_on_map
from stbench import episode as ep
from stbench import timeline as T
from stbench.camera import Camera
from stbench.case import read_index
from stbench.geometry import quat_z
from stbench.robots import ROBOTS, camera_doc, from_doc, robot_doc
from stbench.timeline import Ego, Task, TaskPlan

POSES = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, math.pi / 2), (3.2, 3.4, 0.0, math.pi)]  # living room, open floor


@pytest.fixture
def written(tmp_path):
    tl = script_on_map("apartment_a", 1, "vacuum", POSES)
    return tl, ep.write_script(tmp_path, tl), tmp_path


def files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_agent_and_private_folders_hold_the_right_files(written):
    """Truth must be unreachable from agent-facing files: scripts/<sid> holds only the robot, its camera and its pose
    rows (no mission, no floor plan, no protocol: briefs are never files), and the timeline and the index live under
    private/. Regression caught: the timeline, or a mission or floor plan of the old layout, written where a system
    can read it."""
    tl, paths, dataset = written
    assert files(paths.root) == {"robot.json", "sensors/camera.json", "sensors/pose.jsonl"}
    assert files(paths.private) == {"timeline.json"}
    assert (dataset / "private" / "index" / f"{paths.id}.json").exists()
    assert paths.root != paths.private and paths.root.parent.name == "scripts"


def test_script_ids_are_random_16_hex(tmp_path):
    """Ids are 16 hex characters from the OS random source: writing the same inputs into two datasets gives different
    ids, so an id carries no information about map, seed or robot. Regression caught: an id derived from the inputs (a
    hash, a counter), from which a system could read the map or the seed."""
    tl = script_on_map("apartment_a", 1, "vacuum", POSES)
    a = ep.write_script(tmp_path / "a", tl).id
    b = ep.write_script(tmp_path / "b", tl).id
    assert re.fullmatch(r"[0-9a-f]{16}", a) and re.fullmatch(r"[0-9a-f]{16}", b)
    assert a != b


def test_rewriting_the_same_inputs_reuses_the_id(written):
    """Writing a script again overwrites it under the same id instead of piling up duplicates in the index. Regression
    caught: a rebuild that leaves a second folder and a second index entry for the same inputs."""
    tl, paths, dataset = written
    again = ep.write_script(dataset, tl)
    assert again.id == paths.id and len(read_index(dataset)) == 1


def test_distinct_inputs_get_distinct_ids(written):
    """Contract: scripts of different inputs (here another seed) get different ids and both are in the index.
    Regression caught: an id reuse keyed too coarsely, so a second script overwrites the first."""
    tl, paths, dataset = written
    other = ep.write_script(dataset, script_on_map("apartment_a", 2, "vacuum", POSES))
    assert other.id != paths.id and len(read_index(dataset)) == 2


def test_index_maps_the_id_back_to_the_inputs(written):
    """The script's index entry names its inputs and its size, and the split it is in (a public map's scripts are dev:
    a held-out map's split is `test_split.py`'s). Regression caught: an entry that drops an input, so the id can no
    longer be mapped back."""
    tl, paths, dataset = written
    assert read_index(dataset)[paths.id] == {"kind": "script", "map": "apartment_a", "seed": 1, "robot": "vacuum",
                                             "job": "vacuum/1", "tasks": 1, "split": "dev"}


def test_timeline_round_trips_through_the_private_folder(written):
    """Contract: the private `timeline.json` loads back equal to the timeline that was written. Regression caught: a
    writer or loader that drops or reorders a field, so every gold computed from the file differs from the one the
    generator saw."""
    tl, paths, _ = written
    assert T.load(paths.timeline) == tl


def test_robot_json_is_the_robots_spec_and_camera_json_is_derived_from_it(written):
    """Contract (plan §2.1): the files a script carries for its robot are the robot's own documents, `robot_doc` and
    `camera_doc` (whose values `test_robots.py` pins), so a system parses the robot the gold used (`from_doc` of the
    file is the robot) and the camera readers keep working: `Camera.load` of the file reads the vacuum's depth in
    millimetres from 0.01 to 4.0 m, and a file without the range keeps the reader's default. Regression caught: two
    sources for the robot, the spec and the file, drifting apart."""
    tl, paths, _ = written
    robot = json.loads(paths.robot.read_text(encoding="utf-8"))
    assert robot == robot_doc(ROBOTS["vacuum"]) and from_doc(robot) == ROBOTS["vacuum"]
    cam = json.loads(paths.camera.read_text(encoding="utf-8"))
    assert cam == camera_doc(ROBOTS["vacuum"])
    loaded = Camera.load(paths.camera)
    assert (loaded.depth_scale, loaded.near_m, loaded.far_m) == (0.001, 0.01, 4.0)
    bare = Camera.load({k: v for k, v in cam.items() if k not in ("near_m", "far_m")})
    assert (bare.near_m, bare.far_m) == (0.3, 150.0)  # a file without the range keeps today's `rules.Z_RANGE`


def test_the_drones_camera_file_names_centimetres(tmp_path):
    """Contract (plan-city §2): a drone script's `sensors/camera.json` says the depth is in centimetres, from 0.5 m to
    600 m, which every reader scales by (`Camera.depth_scale`). Regression caught: the city's depth read as millimetres
    (every distance ten times too small, the range no longer fitting 16 bits)."""
    tl = script_on_map("downtown_a", 1, "drone", [(40.0, 13.5, 12.0, 0.0)])
    cam = Camera.load(ep.write_script(tmp_path, tl).camera)
    assert (cam.depth_unit, cam.depth_scale, cam.near_m, cam.far_m) == ("cm", 0.01, 0.5, 600.0)


def rendered(paths):
    """Stand in for a render: the script has a render log, and its frames were made from the files on disk."""
    paths.render_log.write_text("{}\n", encoding="utf-8")


def test_a_rendered_script_is_not_rewritten_with_other_content(written, monkeypatch):
    """Contract: once a script has a render log, `write_script` proceeds only if its `timeline.json` and
    `sensors/camera.json` would keep their content (writing the same inputs again is a no-op), and otherwise raises a
    ValueError that names the file and says the script is rendered from it, before anything is written. Hand cases: a
    start clock moved in the timeline (the same index entry, so the same script id), and the camera's field of view
    changed in the robot's spec. Regression caught: `build` regenerating a script in place under its rendered frames,
    so its gold and its pictures describe different worlds with nothing to say so."""
    tl, paths, dataset = written
    rendered(paths)
    files_of = [*(p for p in paths.root.rglob("*") if p.is_file()), paths.timeline, paths.render_log]
    on_disk = {p: p.read_bytes() for p in files_of}
    assert ep.write_script(dataset, tl).id == paths.id
    assert {p: p.read_bytes() for p in on_disk} == on_disk  # the same bytes proceed and change nothing
    moved = replace(tl, episode=replace(tl.episode, start_clock="2026-03-02T09:30:00"))
    with pytest.raises(ValueError, match=r"timeline\.json.*rendered from the file on disk; delete the script's folder"):
        ep.write_script(dataset, moved)
    vacuum = ROBOTS["vacuum"]
    monkeypatch.setitem(ROBOTS, "vacuum", replace(vacuum, camera=replace(vacuum.camera, hfov_deg=70.0)))
    with pytest.raises(ValueError, match=r"camera\.json.*rendered from the file on disk"):
        ep.write_script(dataset, tl)
    assert {p: p.read_bytes() for p in on_disk} == on_disk and len(read_index(dataset)) == 1


def old_format(tl, path):
    """Write `tl` to `path` as an earlier build did: the same content with the keys that build added and nothing reads
    (a schema string, the episode's digests and generator, the section hashes), laid out another way."""
    doc = json.loads(T.dumps(tl))
    doc = {"schema": "stbench.timeline/1", **doc}
    doc["episode"] |= {"map_sha256": "0" * 64, "robot_sha256": "1" * 64, "generator": "0.11.0"}
    doc["world"] = {"hash": "2" * 64, **doc["world"]}
    doc["ego"] = {"hash": "3" * 64, **doc["ego"]}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def test_a_rendered_script_of_an_earlier_build_is_refused_only_for_a_real_change(written):
    """Contract: the guard of a rendered script compares what its `timeline.json` says, not its bytes, so a file an
    earlier build wrote (it holds a schema string, digests, a generator and section hashes, and is laid out another way,
    so no byte matches) proceeds when it holds the same timeline, and is refused, untouched, when it holds another. Hand
    cases: the same timeline in the old format proceeds, as does a rebuild after it; a start clock moved in the
    old-format file is refused naming the file; a file that is no timeline is refused. Regression caught: the byte
    comparison back, which refuses every rebuild over data already rendered and tells the user to delete frames that
    cannot be made again, or a comparison so loose that a changed world proceeds."""
    tl, paths, dataset = written
    rendered(paths)
    old_format(tl, paths.timeline)
    assert paths.timeline.read_bytes() != T.dumps(tl).encode("utf-8")
    assert ep.write_script(dataset, tl).id == paths.id
    assert T.dumps(T.load(paths.timeline)) == T.dumps(tl)
    assert ep.write_script(dataset, tl).id == paths.id
    moved = replace(tl, episode=replace(tl.episode, start_clock="2026-03-02T09:30:00"))
    old_format(moved, paths.timeline)
    before = paths.timeline.read_bytes()
    with pytest.raises(ValueError, match=r"timeline\.json.*rendered from the file on disk; delete the script's folder"):
        ep.write_script(dataset, tl)
    assert paths.timeline.read_bytes() == before
    paths.timeline.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"timeline\.json.*rendered from the file on disk"):
        ep.write_script(dataset, tl)


def test_a_script_with_no_render_log_is_overwritten(written):
    """Contract: with no render log nothing is rendered from the files, so `write_script` overwrites them as it always
    did. Regression caught: the guard refusing every rewrite, which stops `build` regenerating a script it has not
    rendered."""
    tl, paths, dataset = written
    moved = replace(tl, episode=replace(tl.episode, start_clock="2026-03-02T09:30:00"))
    assert ep.write_script(dataset, moved).id == paths.id
    assert T.load(paths.timeline).episode.start_clock == "2026-03-02T09:30:00"


def test_pose_rows_follow_the_timeline(written):
    """pose.jsonl is determined by the timeline: k, the local clock, the camera (position, quaternion, pitch), the base
    (position, quaternion about z, heading) and the status with the phase and the task, with sorted keys. Hand values:
    the vacuum's camera pitch is +5 degrees, the headings 0, 90 and 180 degrees. Regression caught: a row that
    carries a field the timeline does not decide (the script time, a world fact) or loses one a reader needs, or an
    unsorted key order that tells a system something."""
    tl, paths, _ = written
    rows = [json.loads(line) for line in paths.pose.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(tl.ego.keyframes) == 3
    start = datetime.fromisoformat(tl.episode.start_clock)
    for row, kf in zip(rows, tl.ego.keyframes, strict=True):
        assert list(row) == ["camera", "clock", "k", "pose", "status"]
        assert list(row["camera"]) == ["pitch_deg", "pos", "rot"] and list(row["pose"]) == ["pos", "rot", "yaw_deg"]
        assert row["k"] == kf.k and datetime.fromisoformat(row["clock"]) == start + timedelta(seconds=int(kf.t + 0.5))
        assert row["camera"]["pos"] == list(kf.pos) and row["camera"]["rot"] == list(kf.rot)
        assert row["pose"]["pos"] == list(kf.base) and row["pose"]["rot"] == list(quat_z(kf.yaw))
        assert row["camera"]["pitch_deg"] == 5.0
        assert row["status"] == {"battery_pct": 100, "brush": False, "phase": "docked", "task": 0}
    assert [r["pose"]["yaw_deg"] for r in rows] == [0.0, 90.0, 180.0]


def test_pose_rows_carry_no_t(written):
    """Contract (map X-H4): pose rows hold no `t` key, at any depth of a row: the script time never reaches an agent (a
    window's clock is the case's own; `test_perception.py` owns the oracle rows). Regression caught: the script time
    reaching an agent through the pose file."""
    _, paths, _ = written
    rows = [json.loads(line) for line in paths.pose.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3  # the rows are not empty, so the scan below is not vacuous

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert "t" not in set(keys(rows))


def test_heading_survives_a_downward_camera():
    """Contract (map X-H3): a pose row's heading is the base's, not read from the camera: a drone keyframe at yaw 123
    degrees with its camera pitched to nadir (-90) gives `pose.yaw_deg` 123.0 and `camera.pitch_deg` -90.0. Regression
    caught: a heading read from the camera's forward vector, 180.0 at nadir."""
    kf = keyframe("drone", 0, 0.0, (40.0, 13.5, 12.0), math.radians(123.0), task=0, phase="preflight",
                  pitch=math.radians(-90.0))
    tl = replace(minimal_timeline(), ego=Ego([kf], [Task(0, 0, 1, TaskPlan("hand", {}), ())], []))
    (row,) = ep.pose_rows(tl)
    assert row["pose"]["yaw_deg"] == 123.0 and row["camera"]["pitch_deg"] == -90.0
