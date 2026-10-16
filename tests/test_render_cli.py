"""`stbench render` and `stbench perceive` without Blender: argument handling, the CPU refusal, and the wiring."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from builders import CHAIR, MUG, minimal_timeline
from render_kit import META, fake_blender, small_vacuum, vacuum_script
from stbench import cli
from stbench import episode as ep
from stbench import render_cli as R
from stbench.leak import leak_scan
from stbench.render_encode import write_png16
from stbench.robots import ROBOTS, robot_doc


@pytest.mark.parametrize("text, expected", [(None, (0, 10)), ("0:6", (0, 6)), (":3", (0, 3)), ("4:", (4, 10)),
                                            ("9:10", (9, 10))])
def test_frames_are_half_open_ranges_with_open_ends(text, expected):
    """--frames A:B means keyframes A <= k < B like every interval in the benchmark; "A:" and ":B" are open ended."""
    assert R.parse_frames(text, 10) == expected


@pytest.mark.parametrize("text", ["6", "6:6", "7:3", "0:11", "-1:3"])
def test_bad_frame_ranges_are_refused_with_the_valid_range(text):
    """Contract: an empty, reversed or out-of-range span is an error that names the script's keyframe count (or the
    A:B syntax). Regression caught: a span accepted that renders no frame, or a frame the script does not have."""
    with pytest.raises(R.RenderError, match="10 keyframes|A:B"):
        R.parse_frames(text, 10)


def test_a_render_on_the_cpu_is_refused():
    """The device in Blender's log must be a GPU with a name: a CPU device, or none, means Cycles fell back and the
    run would take about fifty times longer, so the render is refused rather than recorded (V0 section 5)."""
    gpu = {"device": {"type": "OPTIX", "names": ["NVIDIA GeForce RTX 5090 Laptop GPU"]}}
    R.check_device(gpu)
    for bad in ({"device": {"type": "CPU", "names": ["Intel"]}}, {"device": {"type": "OPTIX", "names": []}}, {}):
        with pytest.raises(R.RenderError, match="GPU"):
            R.check_device(bad)


def test_a_missing_blender_is_a_clear_error_and_leaves_no_temporary_files(tmp_path, monkeypatch):
    """Contract: STBENCH_BLENDER pointing nowhere gives an error that names the variable, and the temporary folder is
    removed. Regression caught: a bare FileNotFoundError that never says which setting to fix, or a half-made render
    folder left in the script's private folder."""
    paths = ep.write_script(tmp_path, minimal_timeline())
    monkeypatch.setenv("STBENCH_BLENDER", str(tmp_path / "no-such-blender.exe"))
    with pytest.raises(R.RenderError, match="STBENCH_BLENDER"):
        R.render(paths)
    assert not (paths.private / R.TMP_NAME).exists()


def test_the_render_command_reports_failure_with_a_nonzero_exit(tmp_path, monkeypatch, capsys):
    """Contract: the CLI turns a RenderError into exit code 2 and a message on stderr, not a traceback. Regression
    caught: the commonest failure (no Blender) reaching the shell as an uncaught exception, which a caller cannot tell
    from a crash."""
    paths = ep.write_script(tmp_path, minimal_timeline())
    monkeypatch.setenv("STBENCH_BLENDER", str(tmp_path / "no-such-blender.exe"))
    assert cli.main(["render", str(paths.root)]) == 2
    assert "STBENCH_BLENDER" in capsys.readouterr().err


def test_perceive_runs_visibility_then_perception_from_the_command_line(tmp_path, capsys):
    """Contract: `stbench perceive` over fake renders (a chair and a mug as blocks) writes visibility.jsonl and the
    agent-facing oracle.jsonl (one row a keyframe, the chair detected and the 25 px mug in the dead band not); the
    agent folder passes the leak scan with the registered oracle keys. Regression caught: `perceive` running only one
    of its two stages, a dead-band entity reaching the oracle, or an agent-facing file the leak scan refuses."""
    tl = minimal_timeline()
    paths = ep.write_script(tmp_path, tl)
    idx = np.zeros((480, 640), dtype=np.uint16)
    idx[100:200, 100:200] = 2  # 10,000 px: seen
    idx[300:305, 300:305] = 3  # 25 px: in the dead band
    depth = np.where(idx > 0, 2000, 0).astype(np.uint16)
    for k in range(6):
        write_png16(paths.instances(k), idx)
        write_png16(paths.depth(k), depth)
        Image.fromarray(np.full((480, 640, 3), 120, dtype=np.uint8)).save(paths.rgb(k), quality=90)
    paths.instance_map.write_text(json.dumps({"2": CHAIR, "3": MUG}), encoding="utf-8")
    R.write_log(paths, _meta(6), (0, 6), 3.0, develop=[])  # the fake frames' render log: a whole render
    assert R.render_binding_problems(paths) == []
    assert cli.main(["perceive", str(paths.root)]) == 0
    assert "6 keyframes" in capsys.readouterr().out
    rows = [json.loads(line) for line in paths.oracle.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and all([d["kind"] for d in r["detections"]] == ["chair"] for r in rows)
    assert {e["id"] for e in json.loads(paths.visibility.read_text(encoding="utf-8").splitlines()[0])["entities"]} == \
        {CHAIR, MUG}
    leak_scan(paths.root, [tl.episode.map])


# -- the appearance in the log, in `build`, and the refusal to build photoreal without a library -------------------
def _meta(n: int = 2, **extra) -> dict:
    """What `render_meta.json` holds for `n` frames of a placeholder render."""
    frames = [{"k": k, "render_s": 1.0, "total_s": 1.5} for k in range(n)]
    return {**META, "samples": 32, "fast": False, "width": 640, "height": 480, "frames": frames, **extra}


def test_the_render_log_records_the_appearance_and_the_samples(tmp_path):
    """Contract (plan section 3.3 item 7): `render_log.json` says whether a render used placeholders or assets, with
    how many samples; a Blender log from before this field is a placeholder render. Regression caught: an asset render
    that cannot be told from a placeholder one afterwards."""
    paths = ep.write_script(tmp_path, minimal_timeline())
    old = R.write_log(paths, {k: v for k, v in _meta().items() if k != "appearance"}, (0, 2), 3.0, develop=[])
    assert old["appearance"] == "placeholder"
    new = R.write_log(paths, _meta(samples=64, appearance="assets"), (0, 2), 3.0, develop=[])
    stored = json.loads(paths.render_log.read_text(encoding="utf-8"))
    assert stored == new and (stored["appearance"], stored["samples"]) == ("assets", 64)


def test_the_render_log_records_how_frames_were_developed(tmp_path):
    """Contract (`write_log`): the log keeps every key it had and gains the unit, the linear clip and the near clip the
    renderer reported, the values the frames were developed with (`isp`, and `isp_override`: whether they are not the
    robot's own), where the linear passes were kept and the frames' logs, and no stamp of a version, a commit or a
    digest. Regression caught: a render whose look cannot be traced to values, an override that looks like a finished
    render, or a stamp written again."""
    paths = ep.write_script(tmp_path, minimal_timeline())
    frames = [{"k": 0, "ev": 0.5}, {"k": 1, "ev": -1.0}]
    isp = replace(ROBOTS["helper"].camera.isp, sharpen=0.5, gain_ev=3.0)
    log = R.write_log(paths, _meta(), (0, 2), 3.0, develop=frames, isp=isp, linear_kept=tmp_path / "linear")
    assert json.loads(paths.render_log.read_text(encoding="utf-8")) == log
    assert {"blender", "device", "samples", "fast", "resolution", "frames", "n_frames", "seconds_per_frame",
            "first_frame_s", "mean_steady_s", "median_steady_s", "wall_s", "use_gpu_subdivision", "seed", "appearance",
            "cycles", "lighting"} <= log.keys()
    assert (log["unit_lux"], log["linear_clip"], log["clip_start_m"]) == (240.0, 60000.0, 0.005)
    assert not {"render", "git_commit", "git_dirty", "manifest_sha256", "library_sha256", "world_hash", "ego_hash",
                "robot_sha256", "camera_sha256"} & log.keys()
    assert log["isp"] == {"vignette": 0.3, "noise_a": 6e-4, "noise_b": 1e-5, "meter_sigma": 0.35, "target": 0.18,
                          "exposure_ev": [-8.0, 2.0], "gain_ev": 3.0, "wb_clamp": 0.1, "sharpen": 0.5,
                          "jpeg_quality": 92, "contrast": 1.0, "black_level": 0.0, "glare": 0.0,
                          "denoise_luma_px": 0.0, "denoise_chroma_px": 0.0}
    assert (log["isp_override"], log["linear_kept"], log["develop"]) == (True, str(tmp_path / "linear"), frames)
    own = R.write_log(paths, _meta(), (0, 2), 3.0, develop=frames)
    assert (own["isp_override"], own["linear_kept"]) == (False, None)
    assert own["isp"] == robot_doc(ROBOTS["helper"])["camera"]["isp"]


def test_the_render_check_names_a_missing_or_partial_render(tmp_path, monkeypatch):
    """Contract (`render_binding_problems`): a script with no log has no render; a log that covers every keyframe gives
    []; a log of keyframes 0-3 of 6 is a partial render; and a re-render deletes the `perceive` outputs of the frames it
    replaces. (What ties the frames to their timeline is `write_script`'s refusal to rewrite a rendered script's files:
    `test_episode.py`.) Regression caught: a gold or a visibility log computed over a script that was never rendered or
    only in part, and `perceive` outputs left beside the frames they no longer describe."""
    small_vacuum(monkeypatch)
    fake_blender(monkeypatch, R, (160, 120))
    tl = vacuum_script(6)
    paths = ep.write_script(tmp_path, tl)
    assert R.render_binding_problems(paths) == ["no render log: the script has not been rendered (stbench render)"]
    for stale in (paths.visibility, paths.perception_ids, paths.oracle):
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("{}\n", encoding="utf-8")
    R.render(paths, fast=True)
    assert R.render_binding_problems(paths) == []
    assert not any(p.exists() for p in (paths.visibility, paths.perception_ids, paths.oracle))
    R.render(paths, fast=True, frames="0:3")
    (problem,) = R.render_binding_problems(paths)
    assert "keyframes [0, 3] of 6" in problem and "a partial render" in problem


def test_kept_linear_passes_redevelop_to_the_same_bytes(tmp_path, monkeypatch):
    """Contract (`render --keep-linear`, `develop`), with a fake Blender that writes synthetic passes: the kept passes
    land in DIR and the temporary folder is gone; `develop` with no override gives the same JPEG bytes; an override
    gives other bytes and `isp_override` true; when the robot's camera model alone changes, `develop` develops the kept
    passes with the new values (other bytes, the log's `isp` the robot's) and no render. Regression caught: a camera
    revision forcing a re-render, or a re-develop that silently drifts."""
    small_vacuum(monkeypatch)
    fake_blender(monkeypatch, R, (160, 120))
    tl = vacuum_script(3)
    paths = ep.write_script(tmp_path / "ds", tl)
    keep = tmp_path / "linear"
    R.render(paths, fast=True, keep_linear=keep)
    assert sorted(p.name for p in keep.iterdir()) == [f"{k:06d}.linear.npy" for k in range(3)]
    assert not (paths.private / R.TMP_NAME).exists()
    assert np.load(keep / "000000.linear.npy").dtype == np.float16
    developed = [paths.rgb(k).read_bytes() for k in range(3)]
    log = R.develop(paths, keep)
    assert [paths.rgb(k).read_bytes() for k in range(3)] == developed and log["isp_override"] is False
    assert log["linear_kept"] == str(keep) and [f["k"] for f in log["develop"]] == [0, 1, 2]
    override = replace(ROBOTS["vacuum"].camera.isp, sharpen=0.0, noise_a=0.0, noise_b=0.0)
    log = R.develop(paths, keep, override)
    assert [paths.rgb(k).read_bytes() for k in range(3)] != developed and log["isp_override"] is True
    R.develop(paths, keep)
    assert [paths.rgb(k).read_bytes() for k in range(3)] == developed
    small_vacuum(monkeypatch, sharpen=0.6)  # the vacuum's camera model alone changes
    log = R.develop(paths, keep)
    assert log["isp"]["sharpen"] == 0.6 and log["isp_override"] is False
    assert R.render_binding_problems(paths) == [] and paths.rgb(0).read_bytes() != developed[0]


def test_develop_refuses_a_script_with_no_render_log_and_a_folder_without_its_passes(tmp_path, monkeypatch):
    """Contract (`develop`): a script that was never rendered has no frames to develop again, and a folder that lacks a
    linear pass for a rendered keyframe is refused naming `--keep-linear`; both are `RenderError`s, nothing is written.
    Regression caught: a develop that crashes on a missing file, or one that rewrites JPEGs from half the passes."""
    small_vacuum(monkeypatch)
    fake_blender(monkeypatch, R, (160, 120))
    paths = ep.write_script(tmp_path / "ds", vacuum_script(3))
    with pytest.raises(R.RenderError, match="no render log"):
        R.develop(paths, tmp_path / "linear")
    R.render(paths, fast=True)
    before = [paths.rgb(k).read_bytes() for k in range(3)]
    (tmp_path / "linear").mkdir()
    with pytest.raises(R.RenderError, match="--keep-linear"):
        R.develop(paths, tmp_path / "linear")
    assert [paths.rgb(k).read_bytes() for k in range(3)] == before


def test_an_isp_override_file_replaces_only_the_fields_it_names(tmp_path):
    """Contract (`load_isp`, the `--isp` look-dev file): any `IspSpec` field may be named, the stages added in the fix
    round (`contrast`, `black_level`, `glare`, `denoise_luma_px`, `denoise_chroma_px`) and `exposure_ev` (a list in
    JSON, a tuple in the spec) among them; the fields it does not name keep the base's values; a name that is not a
    field is refused with the list of the fields. Regression caught: a candidate that silently ignores a new field, so
    a look-dev round measures the robot's own values."""
    base = ROBOTS["vacuum"].camera.isp
    path = tmp_path / "isp.json"
    named = {"contrast": 1.6, "black_level": 0.0025, "glare": 0.01, "denoise_luma_px": 0.8, "denoise_chroma_px": 1.0,
             "exposure_ev": [-6.0, 1.0], "target": 0.09}
    path.write_text(json.dumps(named), encoding="utf-8")
    got = R.load_isp(path, base)
    assert {k: getattr(got, k) for k in named} == {**named, "exposure_ev": (-6.0, 1.0)}
    assert replace(got, **{k: getattr(base, k) for k in named}) == base
    path.write_text(json.dumps({"contrats": 1.6}), encoding="utf-8")
    with pytest.raises(R.RenderError, match="contrats.*not IspSpec fields.*contrast"):
        R.load_isp(path, base)
