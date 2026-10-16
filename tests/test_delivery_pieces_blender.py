"""The drone's delivery pieces in Blender (lane M3-RY; `render/furniture_city.py`'s `nest` and `drop_zone`,
`render/materials_street.py`'s paving): the nest and the five drop zones of `downtown_a` are drawn where the map puts
them, at the height it gives them, and are seen straight down from the height the drone hovers at over a zone. One
hand-made script on the map's own pieces (no things on them and no dressing: nothing stands on a zone), a drone camera
pitched straight down 7 m over each piece, rendered once (fast: the placeholder drawings, which are the geometry the
assets scene draws too), and its scene built once more by `blender_delivery_driver.py` for the meshes' boxes and the
paving's nodes. Needs Blender and a GPU; marked `blender`, skipped without it. Nothing is downloaded. About thirty
seconds (one six-frame render, one Blender script). Where the pieces are on the map is `test_downtown_map.py`'s, what
the dressing keeps off them is `test_dressing_downtown_zones.py`'s, and the rest of the city's scene is
`test_render_city_blender.py`'s."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from builders import camera_keyframe, minimal_timeline
from stbench import blender_run
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera, cam_to_map, unproject
from stbench.geometry import Rect, dist_to_rect, quat_z
from stbench.mapspec import load_map
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16
from stbench.timeline import Ego, Entity, Placement, Region, Task, TaskPlan

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
# The pieces, by hand from `downtown_a.json` (M3-maps): (kind, centre, size) with the size's height the piece's top. The
# nest is a pad of 0.02 m on bare ground; a zone is paint 0.005 m thick on its ground (the square's three on the
# paving's 0.15 m, the two street zones on the loading zones' 0.01 m).
PIECES = {"F25": ("nest", (40.0, -18.5), (1.6, 1.6, 0.02)),
          "F26": ("drop_zone", (40.0, -6.0), (3.0, 3.0, 0.155)), "F27": ("drop_zone", (60.0, -6.0), (3.0, 3.0, 0.155)),
          "F28": ("drop_zone", (60.0, -18.0), (3.0, 3.0, 0.155)),
          "F29": ("drop_zone", (20.3, 44.1), (2.6, 3.0, 0.015)), "F30": ("drop_zone", (79.7, 44.1), (2.6, 3.0, 0.015))}
HEIGHTS_M = (7.0,)  # the camera above the piece's top: the drone's lowering height (its transit height shows the same)
SEEN_PX = {7.0: 1500}  # the fewest pixels of a piece (a zone's paint is about 2.5 m2: 7,500 at 7 m)
XY_TOL_M, Z_TOL_M, SHARE = 0.05, 0.02, 0.99
PAINT_M = 0.005


def delivery_timeline():
    """The map's pieces as entities, and the drone's camera pitched straight down at each piece's centre, 7 m above its
    top: keyframe k looks at `views[k]`. Returns the timeline, the views and the entity of each piece."""
    m, base = load_map("downtown_a"), minimal_timeline()
    entities = [Entity(f"E{i + 1:04d}", f.kind, {"name": f.name, "color": f.color}, f.mobility, f.size)
                for i, f in enumerate(m.furniture)]
    placements = [Placement(e.id, 0.0, None, (*f.center, 0.0), quat_z(f.yaw), None)
                  for e, f in zip(entities, m.furniture, strict=True)]
    views = [(pid, h) for h in HEIGHTS_M for pid in PIECES]
    keyframes = [camera_keyframe(k, 1.5 * k, PIECES[pid][1], 0.0, robot="drone", z=PIECES[pid][2][2] + h,
                                 pitch=-math.pi / 2) for k, (pid, h) in enumerate(views)]
    regions = [Region(r.id, r.name, r.kind, tuple(map(tuple, r.polygon)), r.level, r.z0, r.z1) for r in m.regions]
    episode = dataclasses.replace(base.episode, map=m.name, robot="drone", job="drone/1")
    ego = Ego(keyframes, [Task(0, 0, len(keyframes), TaskPlan("hand", {}), ())], [])
    tl = dataclasses.replace(base, episode=episode, regions=regions, entities=entities, placements=placements,
                             states=[], ego=ego)
    return tl, views, {f.id: e.id for e, f in zip(entities, m.furniture, strict=True)}


Rendered = namedtuple("Rendered", "tl paths views entity")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> Rendered:
    tl, views, entity = delivery_timeline()
    paths = ep.write_script(tmp_path_factory.mktemp("delivery"), tl)
    render(paths, fast=True)
    return Rendered(tl, paths, views, entity)


@pytest.fixture(scope="module")
def drawn(rendered) -> dict:
    """The scene of the script built without a render: per delivery piece and plaza its box and materials, and the node
    types of each paving material."""
    out = rendered.paths.root / "drawn.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_delivery_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--private", str(rendered.paths.private), "--out", str(out)],
                    "building the downtown's delivery pieces")
    return json.loads(out.read_text(encoding="utf-8"))


def frame_points(r: Rendered, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Every pixel of keyframe `k` that has a depth, unprojected through it and the keyframe's pose to the map frame:
    the points (x, y, z in rows) and the entity id of the instance each belongs to ("" for none)."""
    cam, imap = Camera.load(r.paths.camera), V.load_instance_map(r.paths.instance_map)
    pose = V.load_pose(r.paths.pose)[k]["camera"]
    index, depth = read_png16(r.paths.instances(k)), read_png16(r.paths.depth(k)).astype(float) * cam.depth_scale
    v, u = np.nonzero(depth > 0)
    pts = cam_to_map(unproject(u + 0.5, v + 0.5, depth[v, u], cam).T, pose["pos"], pose["rot"])
    return pts, np.array([imap.get(int(i), "") for i in index[v, u]])


def test_delivery_pieces_render_where_they_stand(rendered):
    """Contract (`furniture_city.nest` and `drop_zone`, `build_scene`: an entity is drawn where its placement puts it,
    and the drone looks straight down at its zone from 7 m): in each of six frames, the nest or drop zone the camera is
    over shows at least 1,500 pixels of its own instance, 99% of them unprojected through the planar depth and the
    keyframe's pose lie within 5 cm of the piece's footprint and 2 cm of its top (the nest's 0.02 m; a zone's ground
    plus 0.005 m: 0.155 m on the paving, 0.015 m on a loading zone), and they span at least 90% of the footprint along
    each axis (a piece is not drawn smaller than it is, nor shifted). Regression caught: paint drawn inside the paving
    (its instance would be hidden, or sunk below its top), a pad drawn off its place or at the wrong size, or a nest
    that is not solid to its height."""
    cam = Camera.load(rendered.paths.camera)
    for k, (pid, h) in enumerate(rendered.views):
        kind, (cx, cy), (sx, sy, top) = PIECES[pid]
        pts, owner = frame_points(rendered, k)
        mine = pts[owner == rendered.entity[pid]]
        assert V.status(len(mine), cam) == "seen" and len(mine) >= SEEN_PX[h], f"{pid} at {h} m: {len(mine)} pixels"
        off = dist_to_rect(mine[:, 0], mine[:, 1], Rect(cx, cy, sx, sy, 0.0))
        assert (off <= XY_TOL_M).mean() >= SHARE, f"{pid} at {h} m: {(off > XY_TOL_M).sum()} of {len(mine)} pixels off"
        assert (np.abs(mine[:, 2] - top) <= Z_TOL_M).mean() >= SHARE, (pid, h, mine[:, 2].min(), mine[:, 2].max())
        for axis, size in ((0, sx), (1, sy)):
            assert np.ptp(mine[off <= XY_TOL_M, axis]) >= 0.9 * size, (pid, h, axis, np.ptp(mine[:, axis]))


def test_the_pieces_are_drawn_to_their_boxes_and_the_paving_reads_the_world(drawn, rendered):
    """Contract (`furniture_city`: the nest's mat is solid from the ground to its 0.02 m with its paint on top; a drop
    zone is a marking from its ground to 0.005 m above it, as wide as the piece and nothing more;
    `materials_street.paving` reads the position in the world, so the square's two pieces of paving, which lie a
    centimetre apart with different origins, show one pattern across the seam): in the built scene the nest's mesh
    spans x and y 1.6 m about its centre and z 0 to 0.02, each zone's spans its footprint and z (top less 0.005) to its
    top, to a millimetre; both pieces of paving (F21 and F24) are drawn with a paving material, every paving material
    (these and the square's floor) holds a Geometry node and no Texture Coordinate node, and there are at least two.
    Regression caught: paint drawn inside the paving or a thick slab where a marking should be (a robot's climb limit
    would read it as an obstacle), a nest that is a hollow shell or taller than its piece, or a paving whose pattern
    breaks at the seam because each piece reads its own origin."""
    for pid, (kind, (cx, cy), (sx, sy, top)) in PIECES.items():
        row = drawn["pieces"][rendered.entity[pid]]
        low = 0.0 if kind == "nest" else top - PAINT_M
        assert row["kind"] == kind
        want = [cx - sx / 2, cy - sy / 2, low, cx + sx / 2, cy + sy / 2, top]
        assert row["box"] == pytest.approx(want, abs=1e-3), pid
    plazas = [r for r in drawn["pieces"].values() if r["kind"] == "plaza"]
    assert len(plazas) == 2 and all(any(n.startswith("shader:paving") for n in r["materials"]) for r in plazas)
    assert len(drawn["paving"]) >= 2
    for name, nodes in drawn["paving"].items():
        assert "ShaderNodeNewGeometry" in nodes and "ShaderNodeTexCoord" not in nodes, (name, nodes)
