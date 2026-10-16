"""The clutter in Blender: a world with its clutter is rendered from a vacuum's pose and a standing pose, and what is
seen is where the timeline puts it. Needs Blender and a GPU; marked `blender`, skipped when Blender is not
installed. Two `fast` placeholder renders of one frame each: about half a minute for the module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from realism_kit import apartment, script
from stbench import asset_render as AR
from stbench import episode as ep
from stbench import visibility as V
from stbench.asset_manifest import AssetError
from stbench.camera import Camera
from stbench.geometry import Rect, dist_to_rect
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16
from stbench.timeline import load

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
SEED = 101  # a look-dev seed: no gate script is one (M1-look.md §1)
# One pose of each camera, found by rendering the seed's world: the vacuum in the bedroom's east end facing north (a
# floor lamp, a power strip and a cable bundle in view) and the helper in the living room facing the north-west corner
# (the bookshelf, the TV console and the coffee table, 21 pieces).
POSES = {"vacuum": [(7.5, 3.0, 90.0)], "helper": [(2.3, 3.4, 135.0)]}


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    """Seed 101's cluttered world under a one-frame script of each camera, rendered fast in the placeholder
    appearance."""
    out = {}
    for robot, poses in POSES.items():
        paths = ep.write_script(tmp_path_factory.mktemp(robot), script(SEED, robot, poses))
        render(paths, fast=True)
        out[robot] = paths
    return out


def test_clutter_renders_where_the_timeline_puts_it(renders):
    """Contract (plan-redesign §4.10 item 1: clutter is world content, so the render draws it where the timeline puts
    it): every clutter entity seen in a rendered frame from the vacuum's pose or the standing pose (at least the plan's
    215 px) has its visible pixels' ground-plane position, unprojected through the planar depth and the keyframe pose,
    within 0.1 m of its footprint in the timeline; both floor clutter and surface clutter are among those seen, and the
    frames hold at least ten pieces. Regression caught: a renderer that places clutter on its own (a box of another
    size, a pose that is not the placement's, a yaw lost), which `build_scene` would do if it drew clutter by kind."""
    m, seen = apartment(), []
    for robot, paths in renders.items():
        tl, cam = load(paths.timeline), Camera.load(paths.camera)
        imap, pose = V.load_instance_map(paths.instance_map), V.load_pose(paths.pose)[0]
        with Image.open(paths.rgb(0)) as im:
            rgb = np.asarray(im.convert("RGB"))
        for r in V.frame_entities(read_png16(paths.instances(0)), read_png16(paths.depth(0)), rgb, cam,
                                  pose["camera"]["pos"], pose["camera"]["rot"], imap):
            e = tl.entity(r["id"])
            if e.mobility != "clutter" or V.status(r["px"], cam) != "seen":
                continue
            p = tl.placement_of(r["id"], 0.0)
            x, y = r["centroid_map"][:2]
            d = float(dist_to_rect(np.array([x]), np.array([y]), Rect(p.pos[0], p.pos[1], e.size[0], e.size[1],
                                                                      p.yaw))[0])
            assert d <= 0.1, f"{robot} {e.kind} {r['id']}: pixel centroid {d:.2f} m from where the timeline places it"
            seen.append((e.kind, p.on is None))
    assert len(seen) >= 10 and {True, False} <= {on_floor for _, on_floor in seen}
    assert m.name == "apartment_a"


def _library_or_skip():
    try:
        return AR.require_library()
    except AssetError as e:
        pytest.skip(f"no verified asset library for the packaged manifest: {e}")


def test_every_clutter_model_the_world_holds_builds_into_the_photoreal_scene(tmp_path):
    """Contract (plan-redesign §4.10 item 1, `build_scene` in assets mode): the standing pose's frame of seed 101's
    world renders from the real asset library (the cache `stbench assets download` fills, a library built for this
    manifest): every clutter entity is made from its own model at its own box (the fit is exact, so a box that is
    not its model's raises), tinted models take their colour and the rest keep theirs, and what is seen of the
    clutter lies where the timeline puts it. Skipped without a verified library. Regression caught: a model whose
    slot names, tint slots or box the manifest records wrongly (the library measures it, the scene refuses it), or
    a clutter kind with no mesh."""
    _library_or_skip()
    tl = script(SEED, "helper", POSES["helper"])
    paths = ep.write_script(tmp_path, tl)
    log = render(paths, fast=True, appearance="assets")
    assert log["appearance"] == "assets"
    loaded, cam = load(paths.timeline), Camera.load(paths.camera)
    imap, pose = V.load_instance_map(paths.instance_map), V.load_pose(paths.pose)[0]
    with Image.open(paths.rgb(0)) as im:
        rgb = np.asarray(im.convert("RGB"))
    kinds = set()
    for r in V.frame_entities(read_png16(paths.instances(0)), read_png16(paths.depth(0)), rgb, cam,
                              pose["camera"]["pos"], pose["camera"]["rot"], imap):
        e = loaded.entity(r["id"])
        if e.mobility == "clutter" and V.status(r["px"], cam) == "seen":
            p = loaded.placement_of(r["id"], 0.0)
            x, y = r["centroid_map"][:2]
            box = Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)
            assert float(dist_to_rect(np.array([x]), np.array([y]), box)[0]) <= 0.1, (e.kind, r["id"])
            kinds.add(e.kind)
    assert len(kinds) >= 4
