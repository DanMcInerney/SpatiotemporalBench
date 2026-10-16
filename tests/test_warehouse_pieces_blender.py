"""The warehouse's new pieces in Blender (M3-RW, `render/furniture_warehouse.py`): the AMR's charger and the bays in
front of its stations are drawn where the map puts them, and the markings are drawn no higher than their pieces. Needs
Blender and a GPU; marked `blender`, skipped when Blender is not installed. About fifteen seconds: one three-frame
render at fast settings from the AMR's own camera (0.5 m up, 10 degrees down) with only the map's fixed pieces in the
world, and one driver run that builds the three pieces' meshes."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import script_on_map
from stbench import blender_run
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera, cam_to_map, unproject
from stbench.geometry import Rect, dist_to_rect
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16
from stbench.timeline import load

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
# The AMR's base poses (x, y, z, yaw): its camera is 0.38 m ahead of the base. Facing east 1.4 m off the charger's face
# (x 17.38); at T of the packing table F09 (2.2 m off its face, x 16.7) looking over the bay F20 (x 15.1 to 16.7, y 5.4
# to 7.6); at T of the shelving F11 (x 0.7) facing west over the bay F22 (x 0.7 to 2.3, y 5.3 to 7.5).
POSES = [(15.6, 1.4, 0.0, 0.0), (14.5, 6.5, 0.0, 0.0), (2.9, 6.4, 0.0, math.pi)]
SEEN = [("charger", (17.44, 1.4)), ("bay", (15.9, 6.5)), ("bay", (1.5, 6.4))]  # kind and centre, one per pose
MIN_PIXELS = 300  # the oracle's "seen" bound is 215: each piece shows well above it


def pieces_timeline():
    """Seed 101's script of the AMR on `warehouse_a` with the three poses above and nothing but the map's fixed pieces
    in the world (no pallet, jack or cage to stand between the camera and a piece)."""
    tl = script_on_map("warehouse_a", 101, "amr", POSES)
    fixed = {e.id for e in tl.entities if e.mobility == "fixed"}
    return dataclasses.replace(tl, entities=[e for e in tl.entities if e.id in fixed],
                               placements=[p for p in tl.placements if p.entity in fixed],
                               states=[s for s in tl.states if s.entity in fixed])


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """The script rendered once, fast, with the placeholder drawings (the pieces are builders in both appearances)."""
    paths = ep.write_script(tmp_path_factory.mktemp("pieces"), pieces_timeline())
    render(paths, fast=True)
    return paths


@pytest.fixture(scope="module")
def extents(tmp_path_factory):
    """The mesh of the charger, a bay and a staging lane at the map's sizes, built by the driver: the size and the box
    [x0, x1, y0, y1, z0, z1] each fills in its own frame."""
    out = tmp_path_factory.mktemp("meshes") / "pieces.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_warehouse_pieces_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--out", str(out)], "building the warehouse's pieces")
    return json.loads(out.read_text(encoding="utf-8"))


def test_new_pieces_render_where_they_stand(rendered):
    """Contract (the pattern of `test_render_blender`'s placement check, per pixel and to 5 cm): every pixel of the
    charger's and of a bay's instance, unprojected through its planar depth and the keyframe's pose, lies within 0.05 m
    of the piece's footprint in the timeline, from the AMR's camera at the charger (a 0.12 x 0.6 m housing on the east
    wall at x 17.38 to 17.5, y 1.1 to 1.7) and at the bays F20 (centre 15.9, 6.5) and F22 (1.5, 6.4), each 1.6 x 2.2 m;
    a bay's pixels lie on the floor, not over it (95% of them between -2 and 2 cm: the unprojection of a pixel off its
    centre by half a pixel is a few mm off at these ranges, so 5 mm of paint is not what this reads: the driver test
    below measures its height). Regression caught: a piece built off its origin or on the wrong side of it (a charger
    whose face looks at the wall), a mirrored or rotated bay, or paint standing over the floor."""
    paths = rendered
    tl, cam = load(paths.timeline), Camera.load(paths.camera)
    imap, poses = V.load_instance_map(paths.instance_map), V.load_pose(paths.pose)
    wanted = {}
    for kind, (x, y) in SEEN:
        e = next(e for e in tl.entities if e.kind == kind
                 and math.dist(tl.placement_of(e.id, 0.0).pos[:2], (x, y)) < 0.01)
        wanted[e.id] = e
    for k, eid in enumerate(wanted):
        e, p = wanted[eid], tl.placement_of(eid, tl.keyframe(k).t)
        index, depth = read_png16(paths.instances(k)), read_png16(paths.depth(k))
        pass_index = next(i for i, name in imap.items() if name == eid)
        v, u = np.nonzero((index == pass_index) & (depth > 0))
        assert len(u) >= MIN_PIXELS, f"k={k} {e.kind} {eid}: {len(u)} pixels"
        d = depth[v, u].astype(np.float64) * cam.depth_scale
        pts = cam_to_map(unproject(u + 0.5, v + 0.5, d, cam).T, poses[k]["camera"]["pos"], poses[k]["camera"]["rot"])
        off = dist_to_rect(pts[:, 0], pts[:, 1], Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw))
        assert off.max() <= 0.05, f"k={k} {e.kind} {eid}: a pixel {off.max():.3f} m from the footprint"
        if e.kind == "bay":
            assert np.percentile(pts[:, 2], 5) >= -0.02 and np.percentile(pts[:, 2], 95) <= 0.02, (k, eid)


def test_the_markings_are_drawn_no_higher_than_their_pieces_and_the_charger_stands_on_the_floor(extents):
    """Contract (the review's M2: a 1 cm marking reads as an obstacle to the AMR, which climbs 1 cm, so no marking
    builder draws higher than its piece, and M3-maps made the markings 5 mm): by hand from `warehouse_a`, a bay is 1.6 x
    2.2 x 0.005 m and a staging lane 2.8 x 1.4 x 0.005 m, each an outline painted from 1 mm to 5 mm over the floor
    that reaches both ends of its footprint and none beyond it; the charger is 0.12 x 0.6 x 0.3 m, solid from the floor
    to 0.3 m, its two steel contact strips reaching its front face (x +0.06) and none of it behind its back (x -0.06) or
    beyond its sides. Regression caught: paint drawn at the old 1 cm or higher than its piece (an obstacle in every
    frame, never explained by an entity), a bay whose outline stops short of its footprint, a charger floating off the
    floor or standing out of its footprint (it would block a zone the map keeps clear)."""
    for kind in ("bay", "staging_lane"):
        (sx, sy, sz), (x0, x1, y0, y1, z0, z1) = extents[kind]["size"], extents[kind]["box"]
        assert sz == 0.005, kind
        assert (x0, x1, y0, y1) == pytest.approx((-sx / 2, sx / 2, -sy / 2, sy / 2)), kind
        assert (z0, z1) == pytest.approx((0.001, 0.005)), kind
    (sx, sy, sz), (x0, x1, y0, y1, z0, z1) = extents["charger"]["size"], extents["charger"]["box"]
    assert (sx, sy, sz) == (0.12, 0.6, 0.3)
    assert (x0, x1, y0, y1, z0, z1) == pytest.approx((-0.06, 0.06, -0.3, 0.3, 0.0, 0.3))
    assert {"black", "steel"} <= set(extents["charger"]["colours"])
