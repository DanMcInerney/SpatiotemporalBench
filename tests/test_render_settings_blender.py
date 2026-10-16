"""The Cycles settings of the renders (realism Phase 1): the photoreal appearance is
rendered with 128 adaptive samples, OpenImageDenoise with its guides and 12 bounces, and every setting the scene ended
with is written to the private render log; and the data passes do not depend on any of it. Needs Blender and a GPU;
marked `blender`, skipped when Blender is not installed. Nothing is downloaded (the library is the packaged manifest's
procedural models). About a minute for the module (one library build, two one-frame renders, one driver run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stbench import blender_run
from stbench import episode as ep
from stbench.render_cli import blender_path, render
from test_render_city_blender import DOWNTOWN_KINDS, apartment_timeline, procedural_library

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return procedural_library(tmp_path_factory.mktemp("settings"), lambda kind: kind not in DOWNTOWN_KINDS,
                              "warehouse_a")


@pytest.fixture(scope="module")
def logs(library, tmp_path_factory):
    """The apartment episode's first frame rendered as placeholders and with assets, at the default samples of each:
    what `render_log.json` holds."""
    manifest, blend = library
    out = {}
    for appearance in ("placeholder", "assets"):
        paths = ep.write_script(tmp_path_factory.mktemp(appearance), apartment_timeline(assets=appearance == "assets"))
        render(paths, frames="0:1", appearance=appearance, library=blend, manifest=manifest)
        out[appearance] = json.loads(paths.render_log.read_text(encoding="utf-8"))
    return out


def test_the_assets_render_log_records_every_setting_the_plan_asks_for(logs):
    """Contract (plan section 5: 128 samples adaptive at threshold 0.01 from 32, OIDN with albedo and normal guides,
    12 bounces with diffuse 4, glossy 4 and transmission 8, light tree, Khronos PBR Neutral, no motion blur, depth of
    field or distortion; and R.1's measured change, no indirect clamp: with the physical sun a clamp of 10 leaves 40%
    of a shaded wall's light): the settings of the scene a render ended with are in `render_log.json` under `cycles`
    and the lamps, sky and unit under `lighting`, the unit, the linear clip and the near clip beside them. Regression
    caught: a setting that is applied and not recorded (a render nobody can reproduce), or recorded and not applied
    (a log that says 128 samples over a scene rendered with 64), or the clamp back on (a sunlit room's bounce light
    cut)."""
    log = logs["assets"]
    c = log["cycles"]
    assert (log["appearance"], log["samples"], c["samples"], c["engine"]) == ("assets", 128, 128, "CYCLES")
    assert (c["adaptive_sampling"], c["adaptive_threshold"], c["adaptive_min_samples"]) == (True, 0.01, 32)
    assert (c["denoiser"], c["denoising_input_passes"]) == ("OPENIMAGEDENOISE", "RGB_ALBEDO_NORMAL")
    assert c["bounces"] == {"max": 12, "diffuse": 4, "glossy": 4, "transmission": 8, "volume": 0, "transparent": 8}
    assert (c["sample_clamp_indirect"], c["light_tree"], c["seed"], c["animated_seed"]) == (0.0, True, 0, False)
    assert c["motion_blur"] is False and c["camera"]["depth_of_field"] is False and c["camera"]["type"] == "PERSP"
    assert (c["view_transform"], c["look"], c["exposure"], c["white_balance"]) == ("Khronos PBR Neutral", "None", 0.0,
                                                                                    None)
    light = log["lighting"]
    assert (light["rig"], light["unit_lux"], light["balance_k"], light["world"]) == ("home", 240.0, 3250.0,
                                                                                     "procedural")
    assert len(light["lamps"]) == 6 and light["portals"] == 5 and len(light["under_cabinet"]) >= 5
    assert (log["unit_lux"], log["linear_clip"], log["clip_start_m"]) == (240.0, 60000.0, 0.005)


def test_the_placeholder_render_keeps_its_settings(logs):
    """Contract (plan section 3: the placeholder appearance, M1, "stays as it is"): 32 samples with no adaptive
    sampling, 5 bounces and an indirect clamp of 8, no lighting record. Regression caught: the photoreal settings
    leaking into the placeholder render (every M1 dataset re-rendered differently, the probe's calibration moved)."""
    log = logs["placeholder"]
    c = log["cycles"]
    assert (log["samples"], c["adaptive_sampling"], c["bounces"]["max"], c["sample_clamp_indirect"]) == (32, False, 5,
                                                                                                         8.0)
    assert c["view_transform"] == "Khronos PBR Neutral" and log["lighting"] is None


def test_depth_and_object_index_are_identical_under_the_old_and_the_new_settings(tmp_path):
    """Contract (plan section 5: measured settings change no truth; V0 section 3): on a room with two indexed boxes and
    a glossy indexed ball, seen from two cameras, the depth and object-index passes are the same bit for bit with the
    assets appearance's settings as `cycles_setup.configure` gives them (128 adaptive samples, 12 bounces, no clamp) and
    with the ones before Phase 1 (64 samples, 8 bounces, clamp 8), while the picture itself differs (the test is not
    vacuous), and the three indices are in the frames. Regression caught: a sampling, filtering or denoising setting
    that blends a pass at an edge (adaptive sampling stopping early on the index pass, a pixel filter on depth), which
    would move every gold answer the passes decide."""
    out = tmp_path / "passes.json"
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_lighting_driver.py"), "--", "--task", "passes", "--render",
           str(REPO / "render"), "--src", str(REPO / "src"), "--out", str(out)]
    blender_run.run(cmd, "rendering the settings pair")
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["identical"] == {"depth": True, "index": True}, result
    assert result["settings_now"] == {"samples": 128, "use_adaptive_sampling": True, "max_bounces": 12,
                                      "sample_clamp_indirect": 0.0}
    assert {1, 2, 3} <= set(result["indices_seen"]) and all(d > 1e-4 for d in result["rgb_mean_abs_diff"])
