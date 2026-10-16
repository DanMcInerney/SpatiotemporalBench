"""Leaving the pre-0.6.0 layout (lane A7): `stbench move-depth`, the sentences that point to it, a render that does not
leave stale depth behind, and an export that ships exactly the keyframes' depth."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from PIL import Image

import export_kit as kit
from render_kit import fake_blender, small_vacuum, vacuum_script
from stbench import case as case_mod
from stbench import cli, export_files
from stbench import episode as ep
from stbench import render_cli as R
from stbench.render_encode import read_png16
from stbench.view_frames import Streams
from stream_kit import QUESTIONS, case_on, hand20, play


def old_layout(script: case_mod.ScriptPaths) -> None:
    shutil.move(script.depth_dir, script.sensors / "depth")


def frame(path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4), value, dtype=np.uint16)).save(path)


def test_move_depth_moves_the_old_folder_whole_and_a_second_run_moves_nothing(tmp_path, capsys):
    """Contract: `stbench move-depth --data D` moves each script's `scripts/<sid>/sensors/depth/` to
    `private/<sid>/depth/` with its files unchanged (frame k's bytes are still frame k's), reports the ids it moved,
    and leaves nothing under `scripts/` that names depth; run again it moves nothing and exits 0; a private place that
    exists but holds no file does not stop it. Regression caught: a copy that leaves the leaking frames behind, a move
    that nests the folder (`depth/depth/`), or a second run that fails or moves twice."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    script = case.script
    for k in range(20):
        frame(script.depth(k), 100 + k)
    want = {k: script.depth(k).read_bytes() for k in range(20)}
    old_layout(script)
    script.depth_dir.mkdir()   # an empty private folder in the way
    assert cli.main(["move-depth", "--data", str(tmp_path)]) == 0
    assert f"moved the depth of 1 script(s) to private/<id>/depth/: {script.id}" in capsys.readouterr().out
    assert {k: script.depth(k).read_bytes() for k in range(20)} == want
    assert sorted(p.name for p in script.depth_dir.iterdir()) == [f"{k:06d}.png" for k in range(20)]
    assert not (script.sensors / "depth").exists() and not (script.depth_dir / "depth").exists()
    assert [p for p in script.root.rglob("*") if "depth" in p.relative_to(script.root).as_posix()] == []
    assert cli.main(["move-depth", "--data", str(tmp_path)]) == 0
    assert "moved the depth of 0 script(s)" in capsys.readouterr().out
    assert {k: script.depth(k).read_bytes() for k in range(20)} == want


def test_move_depth_refuses_a_script_with_frames_in_both_places_and_touches_nothing(tmp_path, capsys):
    """Contract: a script whose old folder and private folder both hold frames is refused (exit 2, its id named, the
    cure named) and neither folder changes, since the command cannot say which frames are current; compile then still
    refuses it with the delete-the-old-folder cure, not the move. Regression caught: a move that nests the old frames
    inside the private folder (`depth/depth/`), which would pass the refusal and be exported."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    script = case.script
    frame(script.depth(0), 7)
    frame(script.sensors / "depth" / "000000.png", 9)
    mine, old = script.depth(0).read_bytes(), (script.sensors / "depth" / "000000.png").read_bytes()
    assert cli.main(["move-depth", "--data", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert script.id in err and "delete the old" in err
    assert script.depth(0).read_bytes() == mine and (script.sensors / "depth" / "000000.png").read_bytes() == old
    assert not (script.depth_dir / "depth").exists()
    problem = case_mod.depth_move_problem([script])
    assert "stbench move-depth --data DATASET" in problem and "hold frames in private/<id>/depth/ too" in problem
    assert "delete their old scripts/<id>/sensors/depth/ folder" in problem
    shutil.rmtree(script.sensors / "depth")
    assert case_mod.depth_move_problem([script]) is None


def test_the_sentence_for_a_clean_move_does_not_ask_for_a_delete(tmp_path):
    """Contract: a script with old depth and no private frames gets the move-depth sentence alone, and none for one
    with neither. Regression caught: the delete advice shown to a dataset that only needs the move."""
    script = case_on(tmp_path, hand20(), QUESTIONS).script
    old_layout(script)
    problem = case_mod.depth_move_problem([script])
    assert "stbench move-depth --data DATASET" in problem and "delete" not in problem
    assert case_mod.depth_move_problem([]) is None


def test_perceive_and_the_stream_point_an_old_layout_to_move_depth(tmp_path, capsys):
    """Contract: on a script that still holds its depth under `scripts/`, `stbench perceive` exits 2 and `run_case` on
    `detections` raises FileNotFoundError, both naming `stbench move-depth`, not "render first" or a bare missing
    picture; on `rgb_pose`, which needs no depth, the run is unaffected. Regression caught: an old dataset sent to
    re-render for a file that is only in the wrong folder."""
    case = case_on(tmp_path, hand20(), QUESTIONS)
    old_layout(case.script)
    assert cli.main(["perceive", str(case.script.root)]) == 2
    assert "stbench move-depth" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError, match="stbench move-depth"):
        play(case, channel="detections")
    run, _ = play(case, channel="rgb_pose")
    assert run.status == "complete"


def test_a_whole_render_removes_the_old_depth_folder_and_a_partial_one_keeps_it(tmp_path, monkeypatch):
    """Contract: a render of every keyframe replaces the depth, so it deletes `scripts/<sid>/sensors/depth/` of an
    older layout in the step that drops the `perceive` outputs, and writes the new frames under `private/`; a render of
    some keyframes (`--frames 0:2`) leaves it. Regression caught: stale frames left under `scripts/` after a render,
    which the move would nest into the private folder and the export would ship."""
    small_vacuum(monkeypatch)
    fake_blender(monkeypatch, R, (160, 120))
    paths = ep.write_script(tmp_path, vacuum_script(3))
    stale = paths.sensors / "depth" / "000000.png"
    frame(stale, 5)
    R.render(paths, fast=True, frames="0:2")
    assert stale.is_file()
    R.render(paths, fast=True)
    assert not (paths.sensors / "depth").exists()
    assert [read_png16(paths.depth(k)).shape for k in range(3)] == [(120, 160)] * 3


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> kit.Fixture:
    return kit.build(tmp_path_factory.mktemp("move") / "data")


def test_an_export_ships_exactly_the_keyframes_depth(built, tmp_path, monkeypatch):
    """Contract (the depth is the scorer's): the export holds `private/<sid>/depth/<k>.png` for each keyframe `k` of
    the script and no other file of that folder: a `Thumbs.db`, a frame past the last keyframe and a nested folder stay
    out (6 keyframes of the `short` script, so 000000-000005 only). Regression caught: an export that copies the
    folder whole and so ships stale or stray files under `private/`."""
    monkeypatch.setattr(export_files, "machine_names", lambda: ["zed-user"])
    data = kit.clone(built, tmp_path / "data")
    folder = data.script("short").depth_dir
    (folder / "Thumbs.db").write_bytes(b"junk")
    frame(folder / "000006.png", 1)
    frame(folder / "nested" / "000000.png", 1)
    out = tmp_path / "out"
    assert cli.main(["export", "--data", str(data.data), "--out", str(out)]) == 0
    sid = data.scripts["short"]
    shipped = sorted(p.name for p in (out / "private" / sid / "depth").rglob("*") if p.is_file())
    assert shipped == [f"{k:06d}.png" for k in range(6)]
    assert kit.files(out) == kit.released(data)


def test_the_viewer_notes_depth_still_in_the_old_place(tmp_path):
    """Contract (view_frames): for a `detections` stream whose depth is only in the pre-0.6.0 place, the entry counts
    all 20 frames' depth pictures missing and carries a sentence saying where the viewer reads from; with the depth in
    `private/` it carries none. Regression caught: a viewer showing no depth pictures with no hint why."""
    case = case_on(tmp_path, hand20(), QUESTIONS)

    def entry() -> dict:
        streams, problems = Streams(tmp_path, tmp_path / "drawn"), []
        key = streams.get(case.id, "detections", problems)
        assert problems == []
        return streams.items[key]

    assert entry()["notes"] == [] and entry()["missing"]["depth"] == 0
    shutil.rmtree(tmp_path / "drawn")
    old_layout(case.script)
    found = entry()
    assert found["missing"]["depth"] == 20
    assert any("scripts/<id>/sensors/depth" in note and "private/<id>/depth" in note for note in found["notes"])
