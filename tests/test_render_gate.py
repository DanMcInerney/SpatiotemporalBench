"""`render_gate` and its use in `stbench render`, without Blender: a surface that goes dark part-way through."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from render_kit import META, small_vacuum, synthetic_passes, vacuum_script, write_passes
from stbench import episode as ep
from stbench import render_cli as R
from stbench import render_gate as G

BRIGHT = 1.0  # an entity's reference; the figures below are ratios to it, measured on real renders (see render_gate)
HEALTHY_WORST = 0.0148  # the darkest healthy entity's third-darkest frame on linear passes: a table seen from under
DEFECT_WORST = 0.0011  # the defective drone script whose third-darkest frame is the higher of the two (on JPEGs)


def _rows(*runs: tuple[int, float]) -> list[tuple[int, float]]:
    """`(frames, ratio)` runs, laid end to end, as `(keyframe, mean luminance)` rows."""
    values = [BRIGHT * ratio for n, ratio in runs for _ in range(n)]
    return list(enumerate(values))


def test_the_threshold_sits_between_the_healthy_worst_and_the_defects():
    """Contract (`dark_surfaces`, the measured boundaries): three frames at 0.0148 of the reference (a dining table
    seen from underneath, the worst healthy entity on linear passes) are not dark, and three at 0.0011 (the defective
    drone script nearer to the line) are; one such frame alone is not; the entity's frames are named. Regression caught:
    a threshold moved to 1/128, where a healthy table is two-fold from failing the build, or to 1/16, where the
    defect is not the only thing named."""
    assert G.dark_surfaces({"table": _rows((20, 1.0), (3, HEALTHY_WORST))}) == {}
    assert G.dark_surfaces({"paving": _rows((20, 1.0), (3, DEFECT_WORST))}) == {"paving": [20, 21, 22]}
    assert G.dark_surfaces({"paving": _rows((20, 1.0), (1, DEFECT_WORST), (2, 1.0))}) == {}


def test_a_surface_is_judged_against_its_own_bright_frames():
    """Contract (`dark_surfaces`): the reference is the entity's own 90th percentile. Not named: one only dim (a sofa at
    a fifth of its brightest view), one that dips for two frames, one with fewer than MIN_FRAMES frames. Named: one
    dark for most of the frames it is in view (a median reference would read it as normal). The blind spot: an entity
    dark in all its frames has a dark reference and is not named (the module says so). Regression caught: the paving's
    fall, in every frame after its onset, passing because it is most of the frames in view, or a dark piece of
    furniture failing a render."""
    assert G.dark_surfaces({"sofa": _rows((15, 0.2), (15, 1.0))}) == {}
    assert G.dark_surfaces({"flicker": _rows((14, 1.0), (2, 1e-4), (14, 1.0))}) == {}
    assert G.dark_surfaces({"short": _rows((3, 1.0), (4, 1e-4))}) == {}
    assert G.dark_surfaces({"paving": _rows((5, 1.0), (25, 1e-4))}) == {"paving": list(range(5, 30))}
    assert G.dark_surfaces({"all dark": _rows((30, 1e-4))}) == {}
    assert G.describe({"E21": list(range(12, 30))}) == "E21 in keyframes 12-29 (18 frames)"
    assert G.onsets({"E21": [12, 13], "E22": [14]}) == {"E21": 12, "E22": 14}


def test_the_statistic_is_the_worst_entitys_third_lowest_ratio():
    """Contract (`statistic`, what `render_log.json` records): the entity with the lowest third-darkest frame over its
    reference, that ratio, its frames under the threshold and the threshold; None and 0 when no entity was in view long
    enough. Regression caught: a log that cannot show how near a healthy render came to the line."""
    history = {"table": _rows((20, 1.0), (3, HEALTHY_WORST)), "paving": _rows((20, 1.0), (3, DEFECT_WORST)),
               "rug": _rows((5, 1.0))}
    stat = G.statistic(history)
    assert (stat["entity"], stat["frames_under"], stat["threshold"]) == ("paving", 3, 1 / 256)
    assert stat["third_lowest_ratio"] == pytest.approx(DEFECT_WORST)
    assert G.statistic({"table": history["table"]})["frames_under"] == 0
    assert G.statistic({"rug": history["rug"]}) == {"entity": None, "third_lowest_ratio": None, "frames_under": 0,
                                                    "threshold": 1 / 256}


def test_only_entities_that_fill_enough_of_the_frame_are_measured():
    """Contract (`entity_means`): each entity with at least MIN_FRACTION of the pixels gets its mean luminance (the
    Rec. 709 weights over the linear pass); index 0 and an index the instance map lacks are not entities, and a speck
    is not measured. Regression caught: a 30-pixel mug reading as 'dark' and failing a render, or the sky counted as an
    entity."""
    index = np.zeros((40, 50))
    index[:10, :10] = 1  # 5 % of the frame
    index[20:24, 20:24] = 2  # 0.8 %: a speck
    index[30:40, 30:50] = 3  # 10 %, not in the instance map
    linear = np.zeros((40, 50, 3), dtype=np.float16)
    linear[:10, :10] = (0.5, 0.25, 0.125)
    linear[20:24, 20:24] = 1.0
    means = G.entity_means(index, linear, {"1": "E0001", "2": "E0002"})
    assert means.keys() == {"E0001"}
    assert means["E0001"] == pytest.approx(0.2126 * 0.5 + 0.7152 * 0.25 + 0.0722 * 0.125, rel=1e-3)


def _blender_that_darkens(monkeypatch, dark_from: list[int | None], n: int = 12):
    """A stand-in for Blender: run number i darkens entity 2 (index 2) from keyframe `dark_from[i]` on, or not at all
    (None). Returns the list of its runs."""
    runs = []

    def run(cmd: list[str], script_id: str) -> None:
        tmp = Path(cmd[cmd.index("--tmp") + 1])
        runs.append(tmp)
        start = dark_from[len(runs) - 1]
        for k in range(n):
            depth, index, linear = synthetic_passes(160, 120, k)
            if start is not None and k >= start:
                linear[index == 2] *= 1e-4
            write_passes(tmp, k, depth, index, linear)
        (tmp / "instance_map.json").write_text(json.dumps({"1": "E0001", "2": "E0002"}), encoding="utf-8")
        frames = [{"k": k, "render_s": 1.0, "total_s": 1.5} for k in range(n)]
        (tmp / "render_meta.json").write_text(json.dumps({**META, "width": 160, "height": 120, "frames": frames}),
                                              encoding="utf-8")

    monkeypatch.setattr(R, "run_blender", run)
    return runs


def _render(tmp_path, monkeypatch, dark_from: list[int | None]):
    small_vacuum(monkeypatch)
    paths = ep.write_script(tmp_path, vacuum_script(12))
    return paths, _blender_that_darkens(monkeypatch, dark_from)


def test_a_clean_render_logs_the_statistic_and_one_attempt(tmp_path, monkeypatch):
    """Contract (`render`): a render in which nothing goes dark is one run, and its log carries `render_attempts` 1, no
    first trip, and `dark_gate` (the worst entity and its ratio, none under the threshold). Regression caught: a gate
    that records nothing for the healthy renders, which are the only evidence of its margin."""
    paths, runs = _render(tmp_path, monkeypatch, [None])
    log = R.render(paths, fast=True)
    assert len(runs) == 1 and log["render_attempts"] == 1 and log["dark_first_trip"] is None
    assert log["dark_gate"]["entity"] == "E0002" and log["dark_gate"]["frames_under"] == 0
    assert 0 < log["dark_gate"]["third_lowest_ratio"] <= 1 and log["dark_gate"]["threshold"] == 1 / 256
    assert json.loads(paths.render_log.read_text(encoding="utf-8")) == log


def test_the_transient_fault_is_rendered_again_and_recorded(tmp_path, monkeypatch, capsys):
    """Contract (`render`): a first run that darkens a surface from one keyframe and a second that does not give the
    second's frames, `render_attempts` 2 and the first trip's entity and first dark keyframe in the log, with stderr
    saying so; a second trip at another keyframe is the transient fault again and gets a third run. Regression caught:
    a build that ships frames whose paving is black, or that gives up at the first transient fault, or whose log does
    not say a render was repeated."""
    paths, runs = _render(tmp_path, monkeypatch, [6, None])
    log = R.render(paths, fast=True)
    assert len(runs) == 2 and (log["render_attempts"], log["dark_first_trip"]) == (2, {"E0002": 6})
    assert "E0002 in keyframes 6-11" in capsys.readouterr().err
    assert R.render_binding_problems(paths) == [] and not (paths.private / R.TMP_NAME).exists()
    paths.render_log.unlink()
    runs = _blender_that_darkens(monkeypatch, [6, 7, None])
    log = R.render(paths, fast=True)
    assert len(runs) == 3 and (log["render_attempts"], log["dark_first_trip"]) == (3, {"E0002": 6})


def test_a_surface_that_goes_dark_from_the_same_keyframe_twice_is_the_scene_and_fails_at_once(tmp_path, monkeypatch):
    """Contract (`render`): two runs that darken the same entity from the same keyframe are an error that says
    deterministic and names the frames, after two runs, with no render log and no temporary folder; three runs that go
    dark at different keyframes are an error after the third. Regression caught: a legitimate darkening (a surface the
    scene really shades) retried to the limit, or a render that loops."""
    paths, runs = _render(tmp_path, monkeypatch, [6, 6, None])
    with pytest.raises(R.RenderError, match=r"E0002 in keyframes 6-11.*deterministic"):
        R.render(paths, fast=True)
    assert len(runs) == 2 and not paths.render_log.exists() and not (paths.private / R.TMP_NAME).exists()
    runs = _blender_that_darkens(monkeypatch, [6, 7, 5, None])
    with pytest.raises(R.RenderError, match=r"3 renders went dark, at different keyframes"):
        R.render(paths, fast=True)
    assert len(runs) == 3 and not paths.render_log.exists() and not (paths.private / R.TMP_NAME).exists()
