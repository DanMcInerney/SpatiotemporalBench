"""Depth is the scorer's (0.6.0, lane A7): the renderer, `perceive`, the free-space rule and the `detections` stream use
`private/<sid>/depth/`, nothing keeps depth under `scripts/<sid>`, and a dataset made before the move is refused with
the move named. The leak rule itself (`depth`) is in test_leak.py and the export layout in test_export.py."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from PIL import Image

import export_kit as kit
from builders import CHAIR, minimal_timeline
from render_kit import vacuum_script, write_passes
from stbench import case as case_mod
from stbench import cli, cmd_compile, export_files, freespace, visibility
from stbench import episode as ep
from stbench import render_cli as R
from stbench.render_encode import read_png16, write_png16
from stream_kit import QUESTIONS, Recorder, case_on, gold_row, hand20, play


def move_to_scripts(script: case_mod.ScriptPaths) -> None:
    """The layout before 0.6.0: the script's depth folder moved under its agent-facing `sensors/`."""
    shutil.move(script.depth_dir, script.sensors / "depth")


def test_the_renderer_writes_depth_under_private_and_nothing_under_scripts(tmp_path):
    """Contract: `encode_frames` writes a keyframe's depth to `private/<sid>/depth/000000.png`, and the agent-facing
    folder holds no file whose name has depth in it. Regression caught: the encoder still writing
    `scripts/<sid>/sensors/depth/`, which the leak scan would then refuse on every rendered script."""
    paths = ep.write_script(tmp_path / "ds", vacuum_script(1))
    write_passes(tmp_path / "raw", 0, np.full((4, 4), 2.0), np.full((4, 4), 3.0), np.full((480, 640, 3), 0.1))
    R.encode_frames(paths, tmp_path / "raw", (0, 1))
    assert paths.depth(0) == tmp_path / "ds" / "private" / paths.id / "depth" / "000000.png"
    assert read_png16(paths.depth(0))[:1, :1].tolist() == [[2000]]   # 2.0 m in the vacuum's millimetres
    assert [p for p in paths.root.rglob("*") if "depth" in p.relative_to(paths.root).as_posix().lower()] == []


def fake_frames(tmp_path):
    """A rendered script as `perceive` reads it: instances, rgb and an instance map, with the depth in `private/`."""
    paths = ep.write_script(tmp_path, minimal_timeline())
    idx = np.zeros((480, 640), dtype=np.uint16)
    idx[100:200, 100:200] = 2   # 10,000 px of the chair, seen at 2 m
    for k in range(6):
        write_png16(paths.instances(k), idx)
        write_png16(paths.depth(k), np.where(idx > 0, 2000, 0).astype(np.uint16))
        Image.fromarray(np.full((480, 640, 3), 120, dtype=np.uint8)).save(paths.rgb(k), quality=90)
    paths.instance_map.write_text(json.dumps({"2": CHAIR}), encoding="utf-8")
    return paths


def test_visibility_and_freespace_read_private_depth_only(tmp_path):
    """Contract: `visibility.build` and `freespace.script_states` read a keyframe's depth from `private/<sid>/depth/`
    and nowhere else: with the depth in the old folder under `scripts/` the first names the missing private file and
    the second raises FileNotFoundError, and with it back in `private/` the chair's median depth is 2.0 m (every pixel
    of its block reads 2000 mm) and the free-space rule returns its keyframe. Regression caught: either reader still
    opening `sensors/depth`, which would work on a dataset that leaks depth and fail on one that does not."""
    paths = fake_frames(tmp_path)
    move_to_scripts(paths)
    with pytest.raises(FileNotFoundError, match=r"private.*depth.*move-depth"):
        visibility.build(paths)
    with pytest.raises(FileNotFoundError):
        freespace.script_states(paths, [0])
    shutil.move(paths.sensors / "depth", paths.depth_dir)
    rows = visibility.build(paths)
    assert len(rows) == 6 and rows[0]["entities"][0]["depth_med_m"] == 2.0 and rows[0]["entities"][0]["id"] == CHAIR
    assert set(freespace.script_states(paths, [0])) == {0}


def test_a_dataset_with_depth_under_scripts_is_refused_with_the_move_named(tmp_path, capsys):
    """Contract: `stbench compile` refuses a case, and `stbench check` a dataset, whose script still holds depth images
    under `scripts/<sid>/sensors/depth/`: exit 2, the script's id and `stbench move-depth` in the message,
    nothing compiled or checked; an empty old folder is no depth, and a dataset with its depth in `private/` is not
    refused for it. Regression caught: a pre-0.6.0 dataset compiled or checked against depth it does not have (or has
    twice, once leaked), which would surface as a missing-file traceback far from the cause."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    ep.write_jsonl(case.gold, [gold_row(q) for q in QUESTIONS])
    script = case.script
    assert "no render log" in cmd_compile.compile_case((str(tmp_path), case.id))[2]   # past the depth rule
    (script.sensors / "depth").mkdir()   # an empty old folder holds no depth
    assert "no render log" in cmd_compile.compile_case((str(tmp_path), case.id))[2]
    (script.sensors / "depth").rmdir()
    move_to_scripts(script)
    cid, found, why = cmd_compile.compile_case((str(tmp_path), case.id))
    assert (cid, found) == (case.id, None)
    assert script.id in why and "private/<id>/depth" in why and "stbench move-depth --data DATASET" in why
    assert cli.main(["compile", str(tmp_path), "--workers", "1"]) == 2
    assert script.id in capsys.readouterr().err
    assert cli.main(["check", "--data", str(tmp_path), "--out", str(tmp_path / "report.json")]) == 2
    err = capsys.readouterr().err
    assert "stbench check:" in err and script.id in err and "private/<id>/depth" in err
    assert not (tmp_path / "report.json").exists()


class Reader(Recorder):
    """A `Recorder` that keeps the bytes of every picture it is sent (class attribute, reset by the test)."""

    got: dict[str, bytes] = {}

    def observe(self, steps):
        super().observe(steps)
        for step in steps:
            for key in ("rgb", "depth"):
                if key in step:
                    type(self).got[step[key]] = (self.workdir / step[key]).read_bytes()


def test_detections_streams_private_depth_and_rgb_pose_streams_none(tmp_path):
    """Contract (plan §2 call 2): on `detections` every step carries `inbox/depth/<k>.png`, a plain copy of the
    scorer's `private/<sid>/depth/<k>.png` (a different file planted at the old `sensors/depth` place is not what a
    system gets), and on `rgb_pose` no step names a depth file and the inbox holds none, while its rgb pictures are the
    script's. Regression caught: the stream copying from the old folder (a missing file once the depth moves), or an
    agent channel handed depth after all."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    script = case.script
    for k in range(20):   # the private depth is told apart per frame, and the decoy at the old place is not it
        Image.fromarray(np.full((480, 640), 1000 + k, dtype=np.uint16)).save(script.depth(k))
    decoy = script.sensors / "depth"
    decoy.mkdir()
    for k in range(20):
        Image.fromarray(np.full((480, 640), 7, dtype=np.uint16)).save(decoy / f"{k:06d}.png")
    Reader.got = {}
    _, log = play(case, Reader, channel="detections")
    steps = [s for entry in log if entry[0] == "observe" for s in entry[1]]
    assert [s["depth"] for s in steps] == [f"inbox/depth/{k:06d}.png" for k in range(20)]
    assert sorted(Reader.got) == sorted(s["depth"] for s in steps)
    assert all(Reader.got[f"inbox/depth/{k:06d}.png"] == script.depth(k).read_bytes() for k in range(20))
    Reader.got = {}
    _, log = play(case, Reader, channel="rgb_pose")
    steps = [s for entry in log if entry[0] == "observe" for s in entry[1]]
    assert all("depth" not in s for s in steps) and sorted(Reader.got) == sorted(s["rgb"] for s in steps)
    assert all("depth" not in name for entry in log for name in entry[-1]["inbox"])


@pytest.fixture(scope="module")
def fixture(tmp_path_factory) -> kit.Fixture:
    return kit.build(tmp_path_factory.mktemp("a7") / "data")


def test_an_export_refuses_depth_left_under_scripts_and_writes_nothing(fixture, tmp_path, monkeypatch, capsys):
    """Contract (the depth is the scorer's): an export of a dataset with a depth image under a script's agent-facing
    folder is refused (exit 2) by the leak scan, rule `depth`, naming the file, and writes nothing; the same dataset
    without it exports. Regression caught: an export that copies `scripts/<sid>/**` whole and so publishes depth to
    every system that is handed the folder."""
    monkeypatch.setattr(export_files, "machine_names", lambda: ["zed-user"])
    data = kit.clone(fixture, tmp_path / "data")
    sid = data.scripts["short"]
    stray = data.script("short").sensors / "depth"
    stray.mkdir()
    shutil.copy(data.script("short").depth(0), stray / "000000.png")
    out = tmp_path / "out"
    assert cli.main(["export", "--data", str(data.data), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert f"script {sid}: sensors/depth/000000.png: depth" in err and not out.exists()
    shutil.rmtree(stray)
    assert cli.main(["export", "--data", str(data.data), "--out", str(out)]) == 0
    shipped = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    assert not [name for name in shipped if name.startswith("scripts/") and "depth" in name]
    assert (out / "private" / sid / "depth" / "000000.png").is_file()
