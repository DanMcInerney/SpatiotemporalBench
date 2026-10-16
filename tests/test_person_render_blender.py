"""The resident in Blender (lane M2-P2): a hand-made vacuum script on seed 101's world with one `person` entity,
rendered with the assets appearance, and the model as the library holds it. Needs Blender and a GPU, the cached
`resident.glb` and the asset library built for the packaged manifest (`stbench assets library`); marked `blender`, and
skipped when any of those is missing. About twenty seconds (one three-frame render, one Blender script)."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import script_on_map
from stbench import asset_manifest as A
from stbench import asset_render as AR
from stbench import blender_run
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera, cam_to_map, unproject
from stbench.geometry import Rect, dist_to_rect, quat_z
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16
from stbench.timeline import Entity, Placement, load

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
PERSON = "P0001"
FOOTWEAR = re.compile(r"shoe|sock|slipper|sandal|boot|sneaker|trainer|flip.?flop", re.IGNORECASE)
# The vacuum's base stands in the hall at (5.4, 1.6) facing north (+y) for all three frames; the resident stands 0.6,
# 1.4 and 2.4 m ahead of it on the hall's free floor: facing the camera, facing east, turned away to the north-west.
BASE = (5.4, 1.6, 0.0, math.pi / 2)
STANDS = [((5.4, 2.2), -math.pi / 2), ((5.4, 3.0), 0.0), ((5.4, 4.0), 3 * math.pi / 4)]
FRAMES = len(STANDS)
FOOT_TOLERANCE_M = 0.1


@pytest.fixture(scope="module")
def library():
    """The cached resident and the library built for the packaged manifest, or a skip that says what to do."""
    m = A.packaged()
    resident = m.asset("resident")
    for f in resident.files:
        if not A.file_path(resident, f).exists():
            pytest.skip("the cache lacks resident.glb: run render/people_make.py (its docstring has the command)")
    try:
        return AR.require_library()
    except AR.LibraryError as e:
        pytest.skip(f"no usable asset library: {str(e).splitlines()[0]}; run `stbench assets library`")


def person_script(box):
    """Seed 101's vacuum script of three frames at BASE, with a resident of the model's box at STANDS."""
    tl = script_on_map("apartment_a", 101, "vacuum", [BASE] * FRAMES)
    placements = [Placement(PERSON, float(k), None if k == FRAMES - 1 else float(k + 1), (*xy, 0.0), quat_z(yaw), None)
                  for k, (xy, yaw) in enumerate(STANDS)]
    entity = Entity(PERSON, "person", {}, "mover", tuple(box), "resident")
    return dataclasses.replace(tl, entities=[*tl.entities, entity], placements=[*tl.placements, *placements])


@pytest.fixture(scope="module")
def rendered(library, tmp_path_factory):
    box = A.packaged().asset("resident").native_bbox_m
    paths = ep.write_script(tmp_path_factory.mktemp("dataset"), person_script(box))
    render(paths, fast=True, frames=f"0:{FRAMES}", appearance="assets")
    return paths


def test_a_seen_person_is_where_the_timeline_puts_it(rendered):
    """Contract (plan section 5.1 P): every pixel of the resident's instance id, unprojected through the planar depth
    and the keyframe's pose, lies within 0.1 m of its footprint in the timeline (the entity's box at its pose), and no
    seen point is above the box or below the floor, in each of three frames with it 0.6 to 2.4 m ahead of the vacuum
    and turned three ways. Regression caught: a mesh that is off its origin, turned the wrong way, scaled away from its
    box or sunk into the floor, or a resident missing from the instance image (a mover path that skips the model)."""
    tl, cam, imap = load(rendered.timeline), Camera.load(rendered.camera), V.load_instance_map(rendered.instance_map)
    poses, box = V.load_pose(rendered.pose), tl.entity(PERSON).size
    index = next(i for i, eid in imap.items() if eid == PERSON)
    for k in range(FRAMES):
        instances, depth = read_png16(rendered.instances(k)), read_png16(rendered.depth(k))
        v, u = np.nonzero(instances == index)
        d = depth[v, u].astype(np.float64) * cam.depth_scale
        v, u, d = v[d > 0], u[d > 0], d[d > 0]
        assert len(d) >= 200, f"frame {k}: only {len(d)} pixels of the resident"
        points = cam_to_map(unproject(u + 0.5, v + 0.5, d, cam).T, poses[k]["camera"]["pos"], poses[k]["camera"]["rot"])
        p = tl.placement_of(PERSON, tl.keyframe(k).t)
        off = dist_to_rect(points[:, 0], points[:, 1], Rect(p.pos[0], p.pos[1], box[0], box[1], p.yaw))
        assert off.max() <= FOOT_TOLERANCE_M, f"frame {k}: a pixel {off.max():.2f} m from the resident's footprint"
        assert points[:, 2].max() <= box[2] + 0.05 and points[:, 2].min() >= -0.05, (k, points[:, 2].min(),
                                                                                       points[:, 2].max())


def test_the_person_is_one_mesh_standing_on_the_floor_facing_plus_x(library, tmp_path):
    """Contract (plan section 5.1 P, the manifest's `resident`): drawn through the mover path with the entity's box the
    model's own, the person is one object at scale 1 with its own materials (none a shoe, sock or slipper), its extent
    is the manifest's box with its feet on z = 0 and centred on the origin, and its eyes are on the +x side of the head
    (it faces +x, which a footprint of 0.3 by 0.5 m cannot show: it is symmetric under a half turn). Regression
    caught: a model exported facing another way, a pivot that floats or sinks it, a scale that is not 1, or a slot
    named for footwear."""
    out = tmp_path / "driver.json"
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_person_driver.py"), "--", "--render", str(REPO / "render"), "--src",
           str(REPO / "src"), "--manifest", str(library.manifest_path), "--library", str(library.blend),
           "--out", str(out)]
    blender_run.run(cmd, "checking the resident's entity")
    got = json.loads(out.read_text(encoding="utf-8"))
    box = A.load(library.manifest_path).asset("resident").native_bbox_m
    assert got["objects"] == 1 and got["pass_index"] == 1 and np.allclose(got["scale"], 1.0, atol=1e-6)
    assert np.allclose(got["min"], (-box[0] / 2, -box[1] / 2, 0.0), atol=1e-3)
    assert np.allclose(got["max"], (box[0] / 2, box[1] / 2, box[2]), atol=1e-3)
    assert got["eyes_centroid"][0] > 0.03 and abs(got["eyes_centroid"][1]) < 0.05 and got["eyes_centroid"][2] > 1.5
    assert [s for s in got["slots"] if FOOTWEAR.search(s)] == [] and all(n > 0 for n in got["slot_polygons"])
