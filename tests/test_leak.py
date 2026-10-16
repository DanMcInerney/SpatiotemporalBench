"""The leak scan: a clean script passes; each kind of planted truth is caught; the allowlist fails closed."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest
from PIL import Image, PngImagePlugin

from builders import CHAIR, DOOR, MUG, MUG2, TABLE, keyframe, minimal_timeline, start_status
from conftest import script_on_map
from stbench import episode as ep
from stbench import leak
from stbench import perception as P
from stbench.leak import LeakError, find_leaks, leak_scan
from stbench.robots import ROBOTS
from stbench.timeline import Ego, Task, TaskPlan

MAP = "apartment_a"
LIVING_ROOM_FLOOR = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, 0.0), (3.2, 3.4, 0.0, 0.0)]


@pytest.fixture
def script(tmp_path):
    return ep.write_script(tmp_path, script_on_map(MAP, 1, "vacuum", LIVING_ROOM_FLOOR))


def rules(paths) -> set[str]:
    return {x.rule for x in find_leaks(paths.root, [MAP])}


def plant(paths, text: str) -> None:
    """Put `text` where the scan reads free text: the robot's name in `robot.json`."""
    doc = json.loads(paths.robot.read_text(encoding="utf-8"))
    doc["name"] = text
    paths.robot.write_text(json.dumps(doc), encoding="utf-8")


def test_a_written_script_is_clean(script):
    """Negative control, and the real contract: the files `write_script` writes carry no truth. The private timeline
    next to them is full of entity ids and is not scanned. Regression caught: a writer that puts a world fact (an id, a
    room, the map) into an agent-facing file, or an allowlist that refuses what the writers write."""
    assert "E0001" in script.timeline.read_text(encoding="utf-8")
    leak_scan(script.root, [MAP])
    assert find_leaks(script.root, [MAP]) == []


def test_robot_json_passes_the_leak_scan(script):
    """Contract (plan §2.1, §2.8): a written script's `robot.json` (`robots.robot_doc`, every field) holds only the
    allowlisted keys and scans clean; one extra key in it is `unlisted_key`. Regression caught: a new field of the robot
    that leaks, or an allowlist that misses one of the robot's own."""
    assert json.loads(script.robot.read_text(encoding="utf-8"))["id"] == "vacuum"
    assert find_leaks(script.root, [MAP]) == []
    doc = json.loads(script.robot.read_text(encoding="utf-8"))
    doc["seed"] = 3
    script.robot.write_text(json.dumps(doc), encoding="utf-8")
    assert [(x.file, x.rule, x.match) for x in find_leaks(script.root, [MAP])] == [
        ("robot.json", "unlisted_key", "seed")]


@pytest.mark.parametrize("robot", sorted(ROBOTS))
def test_every_robots_pose_rows_pass_the_scan(script, robot):
    """Contract (plan §2.8): `sensors/pose.jsonl` allows the keys of every robot's status (the vacuum's brush, the AMR's
    load, the drone's tether), so a pose row of any robot scans clean. Regression caught: an allowlist written for the
    vacuum alone, which refuses the first drone script."""
    r = ROBOTS[robot]
    tl = replace(minimal_timeline(), ego=Ego(
        [keyframe(robot, 0, 0.0, (3.8, 1.0, 0.0), 0.0, task=0, phase=r.status.start["phase"])],
        [Task(0, 0, 1, TaskPlan("hand", {}), ())], []))
    ep.write_jsonl(script.pose, ep.pose_rows(tl), sort_keys=True)
    assert set(start_status(robot)) <= {k for row in ep.pose_rows(tl) for k in row["status"]}
    assert find_leaks(script.root, [MAP]) == []


@pytest.mark.parametrize("text, rule", [
    ("Look for E0042 in the kitchen.", "entity_id"),
    ("The item is E123456.", "entity_id"),
    ("Go to R3 and wait.", "region_id"),
    ("This is the ground TRUTH.", "truth"),
    ("Welcome to apartment_a.", "map_name"),
])
def test_planted_text_leaks_are_caught(script, text, rule):
    """Contract: entity ids (E + 4 or more digits), region ids, the word truth in any case, and the map name, in a
    file's text, are leaks. Regression caught: a rule dropped or narrowed (ids of 4 digits only, the word truth in
    lower case only), which lets that kind of truth into a script a system is given."""
    plant(script, text)
    assert rule in rules(script)
    with pytest.raises(LeakError):
        leak_scan(script.root, [MAP])


def test_the_callers_names_are_what_is_scanned_for(script):
    """Contract (plan §2.8, map TV-H16): the names a script's files may not hold (its map) come from the caller,
    read from the timeline, never from path arithmetic, and there is no default: a scan that is given none looks for
    none, and a call without them is an error. Regression caught: a lookup that returned [] silently for a folder
    with no twin."""
    plant(script, "Welcome to apartment_a.")
    assert {x.rule for x in find_leaks(script.root, [MAP])} == {"map_name"}
    assert find_leaks(script.root, []) == []
    with pytest.raises(TypeError):
        find_leaks(script.root)
    with pytest.raises(TypeError):
        leak_scan(script.root)


@pytest.mark.parametrize("text", ["R2D2 patrols the room.", "The E12 train and the E-0042 code.", "Rooms R and Rest."])
def test_ordinary_words_are_not_flagged(script, text):
    """Contract: near misses are not leaks: R2D2, short E-numbers, and words that only contain a rule's letters.
    Regression caught: a pattern loosened to match inside words, which refuses clean scripts (a robot named R2D2, a
    kind called Rest)."""
    plant(script, text)
    assert find_leaks(script.root, [MAP]) == []


def test_unlisted_files_fail_closed(script):
    """Contract: anything not on the allowlist is a leak, even if its content looks harmless: new agent-facing files
    must be on the table, and a mission or floor plan of the old layout is not. Truth-looking folders are caught the
    same way. Regression caught: a scan that checks only the files it knows, so a stray file (a mission, a gold
    sketch) next to the allowed ones reaches a system."""
    (script.root / "notes.txt").write_text("hello", encoding="utf-8")
    (script.root / "mission.md").write_text("You are a robot.", encoding="utf-8")
    (script.root / "truth").mkdir()
    (script.root / "truth" / "gold.json").write_text("{}", encoding="utf-8")
    found = {(x.rule, x.file) for x in find_leaks(script.root, [MAP])}
    assert {("unlisted_file", name) for name in ("notes.txt", "truth/gold.json", "mission.md")} <= found
    assert any(x.rule == "truth" for x in find_leaks(script.root, [MAP]))


def test_an_entry_in_the_table_allows_its_file(script, monkeypatch):
    """Contract (plan §2.8): `leak.ALLOWED` is the one allowlist, read at scan time: a lane's new agent-facing file
    passes once the coordinator adds its pattern to the table, and is `unlisted_file` before. Regression caught: a scan
    that caches the table, or a second list that the table does not reach."""
    (script.root / "sensors" / "lidar").mkdir()
    (script.root / "sensors" / "lidar" / "000000.bin").write_bytes(b"\x00\x01")
    assert "unlisted_file" in rules(script)
    monkeypatch.setitem(leak.ALLOWED, "sensors/lidar/*.bin", None)
    assert find_leaks(script.root, [MAP]) == []


def test_unlisted_json_key_in_pose_is_caught(script):
    """Contract: the pose allowlist is k, clock, pose, camera, status and their parts: an extra field (here a stray
    `visit`, or the script time `t` that never reaches a system) leaks. Regression caught: a pose row that gains a
    world fact or the script time, and a key check that looks at the top level only."""
    rows = [json.loads(line) for line in script.pose.read_text(encoding="utf-8").splitlines()]
    rows[1]["visit"] = 1
    rows[2]["t"] = 3.0
    script.pose.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert {(x.rule, x.match) for x in find_leaks(script.root, [MAP])} == {("unlisted_key", "visit"),
                                                                           ("unlisted_key", "t")}


def test_unlisted_key_in_the_camera_file_is_caught(script):
    """Contract: the camera file's keys are fixed (the intrinsics, the depth unit and its range). Regression caught: a
    camera file that carries one more field (a seed, a render setting) and is still accepted."""
    cam = json.loads(script.camera.read_text(encoding="utf-8"))
    cam["seed"] = 3
    script.camera.write_text(json.dumps(cam), encoding="utf-8")
    assert {x.match for x in find_leaks(script.root, [MAP]) if x.rule == "unlisted_key"} == {"seed"}


def test_attribute_values_are_scanned_but_attribute_names_are_free(script):
    """Contract: perception rows describe things with arbitrary attribute names (colour, size, ...), but an id hidden in
    a value is still found. Regression caught: the free `attributes` names turning into a blind spot for the values, or
    an allowlist that refuses a new descriptive word."""
    ok = {"k": 0, "detections": [{"kind": "mug", "attributes": {"color": "red", "handle": "left"},
                                  "bbox_px": [1, 2, 3, 4]}]}
    script.oracle.parent.mkdir(parents=True, exist_ok=True)
    script.oracle.write_text(json.dumps(ok) + "\n", encoding="utf-8")
    assert find_leaks(script.root, [MAP]) == []
    ok["detections"][0]["attributes"]["color"] = "E0042"
    script.oracle.write_text(json.dumps(ok) + "\n", encoding="utf-8")
    assert rules(script) == {"entity_id"}


def test_file_names_are_scanned(script):
    """Contract: a frame named after an entity gives the entity away even if its bytes are clean. Regression caught: a
    scan that reads contents only, so a file name carries an id to a system."""
    Image.new("RGB", (4, 4)).save(script.rgb_dir / "E0007.jpg")
    assert "entity_id" in rules(script)


def test_png_text_chunk_and_jpeg_comment_are_scanned(script):
    """Contract: image metadata is agent-facing too: a PNG text chunk and a JPEG comment segment carrying ids are
    caught, and a metadata-free image passes. Regression caught: a scan that reads text and JSON only, so a renderer
    that stamps its scene into a frame's metadata hands the scene to a system."""
    Image.new("RGB", (4, 4)).save(script.rgb(0))
    assert find_leaks(script.root, [MAP]) == []
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "scene E0042")
    Image.new("L", (4, 4)).save(script.rgb(2), format="PNG", pnginfo=info)   # a PNG is read by its signature
    Image.new("RGB", (4, 4)).save(script.rgb(1), comment=b"ground truth, region R3")
    found = {(x.file, x.rule) for x in find_leaks(script.root, [MAP])}
    assert ("sensors/rgb/000002.jpg", "entity_id") in found
    assert ("sensors/rgb/000001.jpg", "truth") in found and ("sensors/rgb/000001.jpg", "region_id") in found


def test_jpeg_exif_segment_is_scanned(script):
    """Contract: a JPEG's Exif segment (an APP1) is scanned like its comment; the word truth in it is a leak.
    Regression caught: a JPEG reader that stops at the comment segment and skips the APPn ones."""
    Image.new("RGB", (4, 4)).save(script.rgb(2), exif=b"Exif\x00\x00truth")
    assert "truth" in rules(script)


def test_leak_error_lists_what_it_found(script):
    """Contract: `leak_scan` raises `LeakError` carrying every leak found (`leaks`) and naming them in its message, so
    a failed pre-flight says what to remove. Regression caught: an error that stops at the first leak or says only that
    something leaked."""
    plant(script, "E0042 and truth")
    with pytest.raises(LeakError) as e:
        leak_scan(script.root, [MAP])
    assert {x.rule for x in e.value.leaks} == {"entity_id", "truth"} and "E0042" in str(e.value)


def test_a_perceived_script_scans_clean_in_a_process_that_imported_no_writer(tmp_path):
    """Contract (review Q-3, V-13): the static table in `leak.py` is the one owner of the allowlist, so a scan in a
    fresh process that imported neither `stbench.perception` nor `stbench.episode` accepts a real written and perceived
    script, the oracle file with its `d`, `center_map` and a door's `open` included. Regression caught: a key the table
    lacks and a writer's import adds (the scan then refuses every perceived script in the harness pre-flight, `check`
    and the export, which import no writer), or a second list that only an import side effect fills."""
    tl = minimal_timeline()
    paths = ep.write_script(tmp_path, tl)
    seen = [{"id": e, "px": 500, "bbox": [100, 100, 140, 140], "depth_med_m": 2.0, "lum": 0.3,
             "centroid_cam": [0, 0, 2.0], "centroid_map": [0, 0, 0]} for e in (TABLE, CHAIR, MUG, MUG2, DOOR)]
    ep.write_jsonl(paths.visibility, [{"k": k, "entities": seen} for k in range(6)])
    assert P.write(paths) == (5, 30)
    oracle = paths.oracle.read_text(encoding="utf-8")
    assert all(f'"{key}"' in oracle for key in ("d", "center_map", "open"))  # the keys the table once lacked are there
    code = ("import json, sys\nfrom stbench.leak import find_leaks\n"
            "leaks = find_leaks(sys.argv[1], [sys.argv[2]])\n"
            "writers = [m for m in ('stbench.perception', 'stbench.episode') if m in sys.modules]\n"
            "print(json.dumps({'leaks': leaks, 'writers_imported': writers}))\n")
    done = subprocess.run([sys.executable, "-c", code, str(paths.root), tl.episode.map], capture_output=True,
                          text=True, check=True)
    assert json.loads(done.stdout) == {"leaks": [], "writers_imported": []}


def test_depth_under_scripts_is_a_leak_and_private_depth_is_not_scanned(script):
    """Contract (0.6.0, VISION rule 9): the agent is fed the camera, so any file whose path names depth under
    `scripts/<sid>` is a leak of rule `depth`, however clean its bytes, in the old folder or beside the rgb frames, and
    the scorer's `private/<sid>/depth/` is outside the scan. Regression caught: the allowlist entry
    `sensors/depth/*.png` left in place, which let a dataset keep handing depth to a system under `scripts/`."""
    Image.new("L", (4, 4)).save(script.depth(0))   # the scorer's depth: private, the scan never walks it
    assert script.depth(0).parent == script.private / "depth" and find_leaks(script.root, [MAP]) == []
    old = script.sensors / "depth"
    old.mkdir()
    Image.new("L", (4, 4)).save(old / "000000.png")
    Image.new("L", (4, 4)).save(script.sensors / "Depth_000001.png")
    assert {(x.file, x.rule) for x in find_leaks(script.root, [MAP])} == {
        ("sensors/Depth_000001.png", "depth"), ("sensors/depth/000000.png", "depth")}
    with pytest.raises(LeakError, match="depth"):
        leak_scan(script.root, [MAP])
