"""`stbench view`'s stream, frame by frame (`view_frames.py`, `view_depth.py`, docs/eval.md): what the harness delivered
to a system for each step of a case, read into the page's data, and the depth pictures drawn for the eye. What is
expected is what a system on the case was actually sent (a `Recorder` through `run_case`), or worked out by hand from
the synthetic scripts of `synth.py` (steps 1.5 s apart, headings 45 degrees apart, the vacuum's camera range), never
from the reader. The page's own code over this data (the controls, the marks, the text-only rule) is in
test_view_page.py."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pytest
from PIL import Image

from runview_kit import PageParts, window_case, window_run
from stbench.camera import Camera
from stbench.cli import main
from stbench.view import read_run
from stbench.view_depth import depth_view, write_depth_view
from stbench.view_frames import Streams
from stbench.view_page import embed
from stream_kit import QUESTIONS, case_on, hand20, observed, play


def sent_to(log: list) -> list[dict]:
    """The steps a system was handed, in order, from its `Recorder` log."""
    return [step for entry in log if entry[0] == "observe" for step in entry[1]]


def file_of(url: str) -> Path:
    """The path a `file:` address names."""
    return Path(url2pathname(urlparse(url).path))


def stream_of(root: Path, cid: str, channel: str, depth_dir: Path | None = None) -> dict:
    streams, problems = Streams(root, depth_dir), []
    key = streams.get(cid, channel, problems)
    assert problems == [] and key == f"{cid}:{channel}"
    return streams.items[key]


def test_the_frames_are_the_steps_a_system_was_sent_in_the_order_it_got_them(tmp_path):
    """Contract (view_frames): a case's frames are its stream's delivered steps, in order, from the package's own
    stream code: step k from 0 to the last, each with its clock (as seconds from the first step's), the base and the
    camera (position, heading, pitch) and the status as delivered, and the look windows the compiler cuts them into
    (never sent: the system got the steps one at a time). Compared with the steps a `Recorder` was handed through
    `run_case`, and with hand values of `hand20` (steps 1.5 s apart, the clock to the tenth, so the offsets are 0, 1.5,
    3.0, ...; the base at (2, 1.5) turning 45 degrees a step; the vacuum's camera 0.13 m ahead and 0.08 m up at pitch
    5; the status keys sorted; the decisions at 3 and 12 and the task's end at 19 cut it into the windows ending before
    4, 12, 13 and 20). Regression caught: frames the system never got, a frame out of order or off by one in its
    clock, a status of another step, or window ends that are not where the question compiler cuts."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    _, log = play(case, channel="rgb_pose")
    sent, entry = sent_to(log), stream_of(tmp_path, case.id, "rgb_pose")
    assert [s["k"] for s in sent] == list(range(20)) and len(entry["t"]) == 20
    first = datetime.fromisoformat(entry["clock0"])
    for i, step in enumerate(sent):
        assert first + timedelta(seconds=entry["t"][i]) == datetime.fromisoformat(step["clock"])
        pose, camera = step["pose"], step["camera"]
        assert entry["pose"][i] == [*pose["pos"], pose["yaw_deg"], *camera["pos"], camera["pitch_deg"]]
        assert dict(zip(entry["status"]["keys"], entry["status"]["rows"][i], strict=True)) == step["status"]
    assert observed(log) == [[k] for k in range(20)] and entry["windows"] == [4, 12, 13, 20]
    assert entry["t"][:10] == [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
    assert entry["pose"][0] == [2.0, 1.5, 0.0, 0.0, 2.13, 1.5, 0.08, 5.0]
    assert [p[3] for p in entry["pose"][:10]] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 0.0, 45.0]
    assert entry["status"]["keys"] == ["battery_pct", "brush", "phase", "task"]
    assert entry["status"]["rows"][0] == [100, False, "docked", 0]
    assert len(json.dumps(entry)) < len(json.dumps(sent)) / 2   # numbers, not the steps' own words and quaternions


def test_look_windows_tasks_and_questions_fall_on_the_steps_they_are_delivered_at(tmp_path):
    """Contract (view_frames, view_questions): in a window that starts at the script's second task, the stream numbers
    its steps from 0 (so the pictures are named from script keyframe 16), the task changes at step 16, the look
    windows end after each decision point and each task's last step (at most 8 steps each), and the during question is
    placed at the step the harness asked it after, which ends a look window; the end question has no step. By hand
    (`window_case`: tasks of k 16-31 and 32-39, a decision point at k 20): look windows end at 5, 13, 16 and 24, the
    second task begins at step 16, 7,177.5 s after the first task's last step (14,400 - 7,222.5), and the question sits
    at step 4. The system really was asked at step 4. Regression caught: a question marked at the wrong step, the
    script's keyframe numbers shown where the case's belong, or a task boundary that is not where the status changes."""
    case = window_case(tmp_path)
    doc = read_run(window_run(tmp_path), tmp_path)
    (shown,) = doc["arms"][0]["cases"]
    entry = doc["streams"][shown["stream"]]
    assert len(entry["t"]) == 24 and entry["k0"] == 16
    assert entry["windows"] == [5, 13, 16, 24]
    assert [t[:2] for t in entry["tasks"]] == [[0, 0], [1, 16]] and entry["tasks"][0][2] == 0.0
    assert entry["tasks"][1][2] == pytest.approx(7177.5)
    task_at = entry["status"]["keys"].index("task")
    assert [row[task_at] for row in entry["status"]["rows"]] == [0] * 16 + [1] * 8
    assert [(q["when"], q["step"]) for q in shown["questions"]] == [("during", 4), ("end", None)]
    assert shown["questions"][0]["step"] + 1 in entry["windows"]
    _, log = play(case)
    at = next(i for i, e in enumerate(log) if e[0] == "ask" and e[1] == "during")
    assert log[at][2]["k"] == 4 and log[at - 1][0] == "observe" and log[at - 1][1][-1]["k"] == 4


def test_the_camera_stream_lists_its_pictures_and_the_detections_stream_its_detections(tmp_path):
    """Contract (view_frames): on `rgb_pose` the frames name the camera images (by the case's step, from script
    keyframe 16 here), existing files, and no detections; on `detections` there is no camera image, the depth pictures
    drawn for the frames are there (both existing files), and each frame lists its detections (a tag of kind, attributes
    and state, the box, the map and camera centres and the size) in the order and with the numbers the system was
    sent, without the keys, which are fresh for each frame. By hand: the red mug is in view all along, the blue book
    from step 6 (k 22), the book sorting first. Regression caught: a picture named by the script's keyframe where the
    case's belongs, a detection channel that shows the camera image the system never got, or detections that differ
    from those sent."""
    case = window_case(tmp_path)
    pictures = tmp_path / "pictures"
    camera = stream_of(tmp_path, case.id, "rgb_pose", pictures)
    assert camera["sends"] == ["pose", "rgb", "status"] and camera["detections"] is None
    assert camera["missing"] == {"rgb": 0, "depth": 0}
    for i in range(24):
        picture = case.script.rgb(16 + i)
        assert file_of(camera["rgb"]) / f"{16 + i:06d}.jpg" == picture and picture.is_file()
    _, log = play(case, channel="detections")
    sent, dets = sent_to(log), stream_of(tmp_path, case.id, "detections", pictures)
    assert dets["sends"] == ["depth", "detections", "pose", "status"] and dets["rgb"] is None and dets["depth"]
    assert dets["camera"] == {"near_m": 0.01, "far_m": 4.0, "unit": "mm"} and dets["missing"] == {"rgb": 0, "depth": 0}
    assert file_of(dets["depth"]) == (pictures / case.script.id).resolve()
    assert all((file_of(dets["depth"]) / f"{16 + i:06d}.png").is_file() for i in range(24))
    tags = dets["detections"]["tags"]
    assert [len(row) for row in dets["detections"]["rows"]] == [1] * 6 + [2] * 18
    for row, step in zip(dets["detections"]["rows"], sent, strict=True):
        assert [[tags[r[0]][0], tags[r[0]][1], r[5:8], r[8:11], tags[r[0]][3], r[1:5]] for r in row] == [
            [d["kind"], d["attributes"], d["center_map"], d["center_cam"], d["extent_m"], d["bbox_px"]]
            for d in step["detections"]]
        assert not any(d["d"] in embed(dets) for d in step["detections"])
    assert [tags[r[0]] for r in dets["detections"]["rows"][6]] == [["book", {"color": "blue"}, {}, [0.1] * 3],
                                                                  ["mug", {"color": "red"}, {}, [0.1] * 3]]
    mug = dets["detections"]["rows"][0][0]
    assert mug[5:8] == [2.0, 1.0, 0.95] and mug[1:5] == [0, 0, 20, 20] and len(mug) == 11
    assert len(json.dumps(dets)) < len(json.dumps(sent)) / 2


def test_the_camera_alone_stream_names_no_depth_picture_and_draws_none(tmp_path):
    """Contract (view_frames; 2026-10-10: the agent is sent no depth image): the frames of a run on `rgb_pose` name
    the camera images and nothing else to look at: the stream sends pose, rgb and status, its depth folder and range
    are null, no depth image counts as missing, and `stbench view` draws no depth picture for it, though the dataset
    holds depth images (the same case on `detections` draws all 24). The frames' numbers are those the system was sent.
    Regression caught: a viewer that shows the depth image of a run whose agent never got one (a reader would take it
    for something the agent saw), or that draws it anyway."""
    case = window_case(tmp_path)
    pictures = tmp_path / "pictures"
    entry = stream_of(tmp_path, case.id, "rgb_pose", pictures)
    assert entry["sends"] == ["pose", "rgb", "status"] and entry["detections"] is None
    assert entry["depth"] is None and entry["camera"] is None and entry["missing"] == {"rgb": 0, "depth": 0}
    assert all((file_of(entry["rgb"]) / f"{16 + i:06d}.jpg").is_file() for i in range(24))
    assert not pictures.exists()
    gate = stream_of(tmp_path, case.id, "detections", pictures)
    assert file_of(gate["depth"]) == (pictures / case.script.id).resolve()
    assert len(list((pictures / case.script.id).iterdir())) == 24
    _, log = play(case, channel="rgb_pose")
    sent = sent_to(log)
    assert len(sent) == 24 and all("rgb" in s and "depth" not in s for s in sent)
    assert entry["pose"][0] == [*sent[0]["pose"]["pos"], sent[0]["pose"]["yaw_deg"], *sent[0]["camera"]["pos"],
                                sent[0]["camera"]["pitch_deg"]]


def test_a_depth_picture_is_the_values_through_the_cameras_range(tmp_path):
    """Contract (view_depth): the depth picture of a 16-bit image is an 8-bit grey one, black where the sensor read
    nothing (0), white (255) at the camera's near range and black (0) at its far range, linear in metres between
    and clipped at both ends, in the camera's unit; the same values draw the same bytes, and a picture that is not
    older than its source is not drawn again: the agent's own picture. By hand with near 0.2 m and far 10 m:
    1,000 mm is 255 x 9 / 9.8 = 234.2, 234; 2,600 mm is 255 x 7.4 / 9.8 = 192.6, 193; 12,000 mm and 100 mm are the two
    ends. Through the stream of `window_case` (vacuum: near 0.01 m, far 4 m): 2 m is 255 x 2 / 3.99 = 127.8, 128, 6 m
    is past the far range, 0, and the empty rows are 0.
    Regression caught: a depth view that misstates the sensor (an inverted or unclipped range, millimetres read as
    metres), or one that differs between two drawings."""
    camera = {"width": 4, "height": 2, "fx": 1, "fy": 1, "cx": 2, "cy": 1, "near_m": 0.2, "far_m": 10.0}
    mm, cm = Camera.load({**camera, "depth_unit": "mm"}), Camera.load({**camera, "depth_unit": "cm"})
    counts = np.array([[0, 200, 1000, 100], [2600, 10000, 12000, 65535]], dtype=np.uint16)
    assert depth_view(counts, mm).tolist() == [[0, 255, 234, 255], [193, 0, 0, 0]]
    assert depth_view(np.array([[0, 100, 260, 1000]], dtype=np.uint16), cm).tolist() == [[0, 234, 193, 0]]
    source, one, two = tmp_path / "depth.png", tmp_path / "a" / "one.png", tmp_path / "b" / "two.png"
    Image.fromarray(counts).save(source)
    assert write_depth_view(source, one, mm) and write_depth_view(source, two, mm)
    with Image.open(one) as picture:
        assert picture.mode == "L" and np.asarray(picture).tolist() == depth_view(counts, mm).tolist()
    assert one.read_bytes() == two.read_bytes()
    stamp = one.stat().st_mtime_ns
    assert write_depth_view(source, one, mm) is False and one.stat().st_mtime_ns == stamp
    os.utime(source, ns=(stamp + 10**9, stamp + 10**9))
    assert write_depth_view(source, one, mm) is True
    Image.fromarray(np.zeros((2, 4), dtype=np.uint8)).save(tmp_path / "eight.png")
    with pytest.raises(ValueError, match="16-bit"):
        write_depth_view(tmp_path / "eight.png", tmp_path / "c.png", mm)
    case = window_case(tmp_path / "data")
    drawn = tmp_path / "drawn"
    stream_of(tmp_path / "data", case.id, "detections", drawn)
    wall = np.asarray(Image.open(drawn / case.script.id / "000020.png"))
    assert wall.shape == (480, 640) and wall[:10].max() == 0
    assert (wall[10:, :320] == 128).all() and (wall[10:, 320:] == 0).all()
    assert (np.asarray(Image.open(drawn / case.script.id / "000016.png")) == 0).all()


def test_a_case_the_dataset_cannot_stream_is_a_sentence_and_arms_share_one_stream(tmp_path):
    """Contract (view, view_frames): arms that ran the same case on the same channel share one stream in the data (a
    case names its key), and a case on another channel has one of its own. A run that does not record its channel shows
    the case's stream on the default channel and says so; a channel that is not one, or a file of the case that is
    missing, leaves the case with no stream and a sentence in its problems, and everything else about the case still
    shows. Regression caught: the stream copied into every arm (a page several times the size), a guessed channel with
    no word of it, or a missing pose file that stops the page."""
    case = window_case(tmp_path)
    for name, channel in (("pair/a", "rgb_pose"), ("pair/b", "rgb_pose"), ("pair/c", "detections")):
        window_run(tmp_path, name, channel)
    doc = read_run(tmp_path / "runs" / "pair", tmp_path)
    assert [a["name"] for a in doc["arms"]] == ["a", "b", "c"]
    keys = [a["cases"][0]["stream"] for a in doc["arms"]]
    assert keys == [f"{case.id}:rgb_pose", f"{case.id}:rgb_pose", f"{case.id}:detections"]
    assert sorted(doc["streams"]) == sorted(set(keys)) and doc["depth_dir"] is None
    assert doc["streams"][keys[2]]["depth"] is None   # no folder was given, so nothing is drawn
    run = tmp_path / "runs" / "pair" / "a"
    files = (run / "results.json", run / "cases" / case.id / "summary.json")

    def record_channel(channel: str | None) -> None:
        """The run's files say the channel is `channel`, or say nothing of it."""
        for path in files:
            found = json.loads(path.read_text(encoding="utf-8"))
            found.pop("channel", None)
            path.write_text(json.dumps(found if channel is None else {**found, "channel": channel}), encoding="utf-8")

    record_channel(None)
    (shown,) = read_run(run, tmp_path)["arms"][0]["cases"]
    assert shown["stream"] == f"{case.id}:rgb_pose"   # the default channel: the camera alone
    assert any("does not record its channel" in p and "rgb_pose" in p for p in shown["problems"])
    record_channel("rgb")
    (shown,) = read_run(run, tmp_path)["arms"][0]["cases"]
    assert shown["stream"] is None and any("'rgb' is not a channel" in p for p in shown["problems"])
    record_channel("rgb_pose")
    case.script.pose.unlink()
    (shown,) = read_run(run, tmp_path)["arms"][0]["cases"]
    assert shown["stream"] is None and len(shown["questions"]) == 2
    assert any("cannot be shown frame by frame" in p and "FileNotFoundError" in p for p in shown["problems"])


def test_the_command_draws_the_depth_pictures_beside_the_page(tmp_path):
    """Contract (view_cmd, docs/eval.md): `stbench view` writes the page and, beside it, a folder named by the page's
    whole name plus `.frames` holding a grey picture of each depth image of the cases shown (by script id and keyframe,
    24 here), which the page's data names; a second view draws nothing again. Regression caught: depth pictures written
    somewhere the page does not point, under the dataset (where an agent could read them) or anew at every view."""
    case = window_case(tmp_path / "data")
    run = window_run(tmp_path / "data", channel="detections")
    out = tmp_path / "pages" / "v.html"
    assert main(["view", str(run), "--out", str(out)]) == 0
    frames = tmp_path / "pages" / "v.html.frames"
    drawn = sorted(p.name for p in (frames / case.script.id).iterdir())
    assert drawn == [f"{k:06d}.png" for k in range(16, 40)]
    page = PageParts(out.read_text(encoding="utf-8")).embedded()
    assert page["depth_dir"] == frames.resolve().as_posix()
    (entry,) = page["streams"].values()
    assert file_of(entry["depth"]) == (frames / case.script.id).resolve()
    stamps = {p: p.stat().st_mtime_ns for p in (frames / case.script.id).iterdir()}
    assert main(["view", str(run), "--out", str(out)]) == 0
    assert stamps == {p: p.stat().st_mtime_ns for p in (frames / case.script.id).iterdir()}
    assert not [p for p in (tmp_path / "data").rglob("*.frames")]   # nothing of the page is written into the dataset
