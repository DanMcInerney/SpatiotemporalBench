"""Rendering in Blender: a tiny episode (6 keyframes) is rendered end to end and checked against the timeline, and a
vacuum under the coffee table is rendered once to check the near clip and the kept linear pass.

These tests need Blender and a GPU; they carry the `blender` marker and skip when Blender is not installed. Four
renders are shared by all tests here (fast twice and default once of the helper's tour, fast once of the vacuum's
frame), so the module costs about three quarters of a minute."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import apartment, script_on_map
from stbench import camera_model
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera
from stbench.geometry import Rect, dist_to_rect
from stbench.leak import find_leaks
from stbench.render_cli import blender_path, render, render_binding_problems
from stbench.render_encode import read_png16
from stbench.robots import ROBOTS
from stbench.timeline import load

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
FRAMES = 6
# A look-around at the doorway of the bedroom, the poses of the helper whose camera (1.30 m above its base, 15 degrees
# down) is 1.5 m up at (6.25, 3.45), turning from 131.664 degrees in 45 degree steps, as the first six keyframes of
# the retired tour of seed 1 did (its camera was the body-free 1.5 m one).
LOOK_AROUND = [(6.25, 3.45, 0.2, math.radians(131.664 + 45.0 * k)) for k in range(FRAMES)]


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    """Seed 1's world with the look-around above, rendered three times into three datasets: fast, fast again, default
    samples."""
    tl = script_on_map("apartment_a", 1, "helper", LOOK_AROUND)
    out = []
    for i, fast in enumerate((True, True, False)):
        paths = ep.write_script(tmp_path_factory.mktemp(f"dataset{i}"), tl)
        out.append((paths, render(paths, fast=fast, frames=f"0:{FRAMES}")))
    return out


def test_depth_and_instance_images_are_bit_identical_across_renders_and_sample_counts(renders):
    """Depth and instance PNGs are identical byte for byte between two runs (a new Blender process each) and between 8
    and 32 samples per pixel. Regressions: a Cycles seed that moves, GPU subdivision, a pass that is blended at edges
    or accumulates per sample. RGB is allowed to differ and is not compared."""
    (a, _), (b, _), (c, _) = renders
    for k in range(FRAMES):
        for name in ("depth", "instances"):
            ref = getattr(a, name)(k).read_bytes()
            assert getattr(b, name)(k).read_bytes() == ref, f"{name} {k} differs between two renders"
            assert getattr(c, name)(k).read_bytes() == ref, f"{name} {k} differs between 8 and 32 samples"


def _distance_to_leaf(m, point, opening):
    hinge, tip = m.door_leaf(opening, True)
    p, a, b = np.array(point), np.array(hinge), np.array(tip)
    t = float(np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * (b - a))))


def test_every_seen_entity_is_where_the_timeline_puts_it(renders):
    """Contract: for each entity seen in a rendered frame (215 px or more), the ground-plane position of its visible
    pixels, unprojected through the planar depth and the keyframe pose, lies within 0.1 m of the entity's footprint in
    the timeline (for an open door: of its open leaf). (V0 says max(size)/2 + 0.1 around the centre; that is not a
    bound for the corner of a wide piece, so the footprint is used.) Regression caught: a flipped camera axis, a wrong
    quaternion convention, a ray-length depth, a mirrored image or a misplaced object, each of which pushes this by
    metres."""
    paths, _ = renders[0]
    tl, m, cam = load(paths.timeline), apartment(), Camera.load(paths.camera)
    imap = V.load_instance_map(paths.instance_map)
    poses = V.load_pose(paths.pose)
    opening_at = {(round(o.center[0], 3), round(o.center[1], 3)): o for o in m.doors()}
    n_seen, kinds = 0, set()
    for k in range(FRAMES):
        with Image.open(paths.rgb(k)) as im:
            rgb = np.asarray(im.convert("RGB"))
        pose = poses[k]
        for r in V.frame_entities(read_png16(paths.instances(k)), read_png16(paths.depth(k)), rgb, cam,
                                  pose["camera"]["pos"], pose["camera"]["rot"], imap):
            if V.status(r["px"], cam) != "seen":
                continue
            n_seen += 1
            e, p = tl.entity(r["id"]), tl.placement_of(r["id"], tl.keyframe(k).t)
            kinds.add(e.mobility)
            x, y = r["centroid_map"][:2]
            if e.kind == "door":
                d = min(_distance_to_leaf(m, (x, y), opening_at[(round(p.pos[0], 3), round(p.pos[1], 3))]),
                        float(dist_to_rect(np.array([x]), np.array([y]), Rect(p.pos[0], p.pos[1], e.size[0], e.size[1],
                                                                              p.yaw))[0]))
            else:
                d = float(dist_to_rect(np.array([x]), np.array([y]), Rect(p.pos[0], p.pos[1], e.size[0], e.size[1],
                                                                          p.yaw))[0])
            assert d <= 0.1, f"k={k} {e.kind} {r['id']}: pixel centroid {d:.2f} m from where the timeline places it"
    assert n_seen >= 10 and {"fixed", "carried"} <= kinds


def test_rendered_files_have_the_documented_shape_and_carry_nothing_extra(renders):
    """Contract: depth and instance PNGs are 16-bit 640x480; the RGB JPEGs are 640x480 and not black, and the leak
    scan of the agent folder is clean (no render times, no application tags); the log names a GPU, the sample counts
    and a time per frame; the instance map covers every entity; no temporary folder is left. Regression caught: a
    render in the wrong size or format, a black frame, a file that carries the renderer's tags or timings to the
    agent, or a log that cannot show the render ran on a GPU."""
    paths, log = renders[0]
    for k in range(FRAMES):
        assert read_png16(paths.depth(k)).shape == (480, 640) and read_png16(paths.instances(k)).shape == (480, 640)
        with Image.open(paths.rgb(k)) as im:
            assert im.size == (640, 480) and im.format == "JPEG"
        assert np.asarray(Image.open(paths.rgb(k))).mean() > 20  # not a black frame
    assert find_leaks(paths.root, ["apartment_a"]) == []
    stored = json.loads(paths.private.joinpath("render_log.json").read_text(encoding="utf-8"))
    assert stored["device"]["type"] in ("OPTIX", "CUDA") and stored["device"]["names"]
    assert (stored["samples"], renders[2][1]["samples"]) == (8, 32) and len(stored["seconds_per_frame"]) == FRAMES
    assert (stored["unit_lux"], stored["clip_start_m"]) == (240.0, 0.005)
    assert [f["k"] for f in stored["develop"]] == list(range(FRAMES)) and render_binding_problems(paths) == []
    imap = json.loads(paths.instance_map.read_text(encoding="utf-8"))
    assert imap["1"] == load(paths.timeline).entities[0].id and len(imap) == len(load(paths.timeline).entities)
    assert not paths.private.joinpath("render_tmp").exists()


@pytest.fixture(scope="module")
def under_the_coffee_table(tmp_path_factory):
    """The vacuum's frame under the coffee table F11, base (2.40, 4.40) facing west, rendered once with `fast` (the
    placeholder appearance), with its linear pass kept."""
    tl = script_on_map("apartment_a", 101, "vacuum", [(2.40, 4.40, 0.0, math.pi)])
    paths = ep.write_script(tmp_path_factory.mktemp("vacuum"), tl)
    keep = tmp_path_factory.mktemp("linear")
    render(paths, fast=True, keep_linear=keep)
    return paths, tl, keep


def test_the_near_clip_keeps_close_surfaces_whole(under_the_coffee_table):
    """Contract (the review's round 3, `rgb_clip_m`): the coffee table's shelf board has its underside 2 cm above the
    vacuum's lens (the board is 0.10 to 0.13 m up, the lens 0.08), and the camera pitches 5 degrees up. By hand, the
    planar depth of the underside at the top row is 0.02 cos(a) / sin(5 + a) with a = atan(239.5 / 381.36) = 32.1
    degrees: 28 mm, and 34 mm at row 47; the board's top face seen from inside, which today's 0.05 m clip would show
    instead, is at 70 to 85 mm. So the top 10% of the rows read a median depth of 25 to 40 mm; no depth reads under the
    robot's near_m (10 mm) but 0. Regression caught: the near clip cutting holes in close surfaces, as the 5 cm clip
    did."""
    paths, _, _ = under_the_coffee_table
    depth = read_png16(paths.depth(0)).astype(float)
    top = depth[:48]
    assert 25 <= np.median(top[top > 0]) <= 40, np.percentile(top[top > 0], [5, 50, 95])
    assert (top > 0).mean() > 0.9
    near_mm = ROBOTS["vacuum"].depth.near_m * 1000
    assert not ((depth > 0) & (depth < near_mm)).any()
    stored = json.loads(paths.render_log.read_text(encoding="utf-8"))
    assert stored["clip_start_m"] == 0.005 and render_binding_problems(paths) == []


def test_the_kept_linear_pass_is_what_was_developed(under_the_coffee_table):
    """Contract (`--keep-linear`, `render_episode.linear_pass`): the kept pass is float16, 480 x 640 x 3, finite and at
    most 60,000, and the camera model develops it, with the key `(map, seed, robot, job, "camera", k)`, to the stored
    JPEG byte for byte. Regression caught: a kept pass that is not what was developed (a re-develop that drifts), or a
    non-finite or out-of-range value reaching the meter."""
    paths, tl, keep = under_the_coffee_table
    linear = np.load(keep / "000000.linear.npy")
    assert linear.dtype == np.float16 and linear.shape == (480, 640, 3)
    assert np.isfinite(linear).all() and linear.max() <= 60000
    ep_ = tl.episode
    jpeg, _ = camera_model.develop(linear.astype(np.float32), key=(ep_.map, ep_.seed, ep_.robot, ep_.job, "camera", 0),
                                   camera=ROBOTS["vacuum"].camera)
    assert jpeg == paths.rgb(0).read_bytes()
