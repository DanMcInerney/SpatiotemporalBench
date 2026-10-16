"""The residential kit in Blender (M4-suburb part a, §3-§4): the pieces' meshes, the materials, the kit map's dressing
and its far scenery, built by `tests/blender_residential_driver.py` with no render and no GPU (about twenty seconds,
most of it Blender starting and the street's asphalt being painted). What the layouts and builders hand over as plain
data is `test_residential_catalog.py`'s and `test_dressing_layout_residential.py`'s. Needs Blender; marked `blender`,
skipped when it is not installed. The city's and the indoor scenes' digests (`test_render_city_blender.py`) are the
other half of this part's exit: they must hold, unedited, after the core hunk of `build_scene`."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from dressing_kit import render_module
from residential_kit import kit
from stbench import blender_run
from stbench.mapspec import parse_map
from stbench.render_cli import blender_path

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
DOWN, UP = (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)
TOL_M = 0.01  # a roof plane's height, worked out by hand to the centimetre
# (kind, size, colour, [ray origin and direction]): the kit's house (9.5 deep, 9.5 wide, 8.4 high), a block of five
# storeys (13 by 30, 15.9 high), and the ground and boundary pieces at the sizes the kit gives them
REQUESTS = [
    {"kind": "house", "size": [9.5, 9.5, 8.4], "color": "beige", "rays": [
        [[0, 0, 15], DOWN], [[3.0, -2.0, 15], DOWN], [[-3.0, -2.0, 15], DOWN], [[-2.1375, 1.995, 15], DOWN],
        [[4.6, 0, 15], DOWN], [[5.2, 0, 15], DOWN], [[0, 5.0, 15], DOWN], [[0, 10, 2.0], [0, -1, 0]],
        [[10, -3.4, 2.0], [-1, 0, 0]], [[0, 0, -1], UP]]},
    {"kind": "block", "size": [13.0, 30.0, 15.9], "color": "red", "rays": [
        [[3.0, 5.0, 25], DOWN], [[5.95, 0, 25], DOWN], [[0, 0, 25], DOWN], [[0, 40, 2.0], [0, -1, 0]],
        [[0, 0, -1], UP]]},
    {"kind": "lawn", "size": [35.0, 7.0, 0.03], "color": "green", "rays": [[[3, 2, 5], DOWN]]},
    {"kind": "driveway", "size": [7.0, 3.2, 0.05], "color": "grey", "rays": [[[1, 1, 5], DOWN]]},
    {"kind": "path", "size": [1.2, 7.0, 0.03], "color": "grey", "rays": [[[0, 1, 5], DOWN]]},
    {"kind": "verge", "size": [1.98, 100.0, 0.15], "color": "green", "rays": [[[0, 10, 5], DOWN]]},
    {"kind": "hedge", "size": [0.6, 9.0, 1.2], "color": "green", "rays": [[[0, 0, 5], DOWN]]},
    {"kind": "fence", "size": [0.1, 16.0, 1.8], "color": "brown", "rays": [[[0, 0.7, 5], DOWN]]},
    {"kind": "garden_bed", "size": [4.0, 4.0, 0.3], "color": "grey", "rays": [[[0, 0, 5], DOWN], [[1.9, 0, 5], DOWN]]},
]
MATERIALS = ["lawn", "hedge", "fence", "driveway", "mulch", "shingle:charcoal", "shingle:brown", "shingle:slate",
             "shingle:terracotta", "siding:beige", "siding:blue", "render:white", "brick:red", "brick:brown"]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """What the driver made of the requests and the kit (see it): per piece the first hits of its rays, the materials,
    the kit's dressing objects and its scenery objects."""
    folder = tmp_path_factory.mktemp("residential")
    (folder / "spec.json").write_text(json.dumps(kit()), encoding="utf-8")
    (folder / "rays.json").write_text(json.dumps(REQUESTS), encoding="utf-8")
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_residential_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--spec", str(folder / "spec.json"), "--rays",
                     str(folder / "rays.json"), "--out", str(folder / "out.json")],
                    "building the residential kit")
    return json.loads((folder / "out.json").read_text(encoding="utf-8"))


def hits(built: dict, kind: str) -> list:
    return next(p["hits"] for p in built["pieces"] if p["kind"] == kind)


def test_houses_and_blocks_are_solid_with_their_roofs(built):
    """Contract (M4-suburb §3: a drone at 15 m over 8 m houses sees roofs, and a building is a solid the planner never
    flies through): rays cast straight down from 15 m onto the kit's house (9.5 by 9.5 by 8.4) hit its ridge cap at
    8.4 m, its chimney at 8.4 m and the roof plane at the heights worked out by hand (the plane's top falls 0.6135 m a
    metre from the ridge line at 8.34 m: 6.50 m at 3 m from it on either side, 5.52 m at 4.6 m, out on the eaves), and
    miss the air a metre beyond the box; a ray at 2 m into the gable wall hits it at the wall (4.5 m from the middle),
    and into the front wall at 4.4 m; a ray from under the floor hits it at 0. The block (13 by 30 by 15.9) is hit on
    its roof deck at 15.03 m, its parapet at 15.9 m, its lift housing at 15.8 m and, at 2 m, on the wall of its gable at
    14.6 m. Regression caught: a house a drone sees into (no roof), a roof at the wrong height in depth (it would
    certify free air that is a roof), planes that leave a gap at the ridge, a wall drawn as a facade with no solid
    behind it, or a block with no parapet."""
    house = hits(built, "house")
    assert [house[i][2] for i in (0, 3)] == pytest.approx([8.4, 8.4], abs=TOL_M)
    assert [house[i][2] for i in (1, 2)] == pytest.approx([6.50, 6.50], abs=TOL_M)
    assert house[4][2] == pytest.approx(5.52, abs=2 * TOL_M)
    assert house[5] is None and house[6] is None
    assert house[7][1] == pytest.approx(4.5, abs=0.002) and house[8][0] == pytest.approx(4.4, abs=0.002)
    assert house[9][2] == pytest.approx(0.0, abs=1e-4)
    block = hits(built, "block")
    assert [block[i][2] for i in (0, 1, 2)] == pytest.approx([15.03, 15.9, 15.8], abs=TOL_M)
    assert block[3][1] == pytest.approx(14.6, abs=0.002) and block[4][2] == pytest.approx(0.0, abs=1e-4)


def test_the_ground_and_boundary_pieces_stand_at_the_heights_the_map_gives(built):
    """Contract (M4-suburb §3: a lawn 0.03 m, a drive 0.05 m, a path 0.03 m, a verge 0.15 m, a hedge and a fence to the
    height the map gives, a garden bed 0.3 m round soil 5 cm under its top): a ray from above hits each at that
    height: 0.03, 0.05, 0.03, 0.15, 1.2 (the hedge's flat top), 1.8 (the fence's capping rail), the bed's soil at 0.25
    and its edging at 0.3. Receptacle surfaces and drop zones are heights the map states (a drive's `parking` surface
    at 0.05, a zone 0.005 over it), so the drawing must stand where the map says. Regression caught: a drive drawn
    thicker than its surface (a car sunk in it), a lawn that is a metre of grass, or a fence below the height the
    planner thinks it blocks at."""
    expected = {"lawn": [0.03], "driveway": [0.05], "path": [0.03], "verge": [0.15], "hedge": [1.2], "fence": [1.8],
                "garden_bed": [0.25, 0.3]}
    for kind, heights in expected.items():
        assert [h[2] for h in hits(built, kind)] == pytest.approx(heights, abs=1e-4), kind


def test_every_residential_material_builds_and_the_kits_dressing_and_scenery_build_inside_their_boxes(built):
    """Contract (M4-suburb §3-§4; the scene must build with no download): every material the pieces name is made (14:
    lawn, hedge, fence, drive concrete, mulch, four roofs, two sidings, a render, two bricks), each with a node tree;
    the kit's dressing comes out of `dressing.build` as one object a `Dressing` in layout order, named
    `dressing_<class>_<n>`, with pass index 0, in the "dressing" collection and no other, standing at its place and
    spanning its box exactly in its own frame (fitted); the far scenery is one ground mesh, the houses, blocks and
    trees of the plan (counted from the plan, in process), all with pass index 0 and each lying outside the kit's
    bounds (0 to 100 by 0 to 60). Regression caught: a material that raises only inside Blender, dressing that is an
    entity (a tree with a pass index would be a thing to track), a piece that spills out of the box the guard
    checked, or scenery inside the map."""
    assert sorted(built["materials"]) == sorted(MATERIALS) and min(built["materials"].values()) >= 4
    layout = render_module("dressing_layout_residential").layout(kit())
    names, seen = [], {}
    for d in layout:
        names.append(f"dressing_{d.cls}_{seen.get(d.cls, 0)}")
        seen[d.cls] = seen.get(d.cls, 0) + 1
    objects = built["dressing"]
    assert [o["name"] for o in objects] == names and {o["pass_index"] for o in objects} == {0}
    assert {tuple(o["collections"]) for o in objects} == {("dressing",)}
    for o, d in zip(objects, layout, strict=True):
        sx, sy, sz = d.size
        assert o["local"] == pytest.approx([-sx / 2, -sy / 2, 0.0, sx / 2, sy / 2, sz], abs=1e-3), o["name"]
        assert o["loc"] == pytest.approx(d.pos, abs=1e-4) and o["verts"] > 0 and None not in o["materials"]
    plan = render_module("scenery_residential").plan(parse_map(kit()))
    scenery = built["scenery"]
    counts = Counter(o["name"].rsplit("_", 1)[0] for o in scenery)
    assert counts == {"scenery": 1, "scenery_house": sum(s.kind == "house" for s in plan.sites),
                      "scenery_block": sum(s.kind == "block" for s in plan.sites), "scenery_tree": len(plan.trees)}
    assert {o["pass_index"] for o in scenery} == {0}
    x0, y0, x1, y1 = kit()["bounds"]
    for o in (o for o in scenery if o["name"] != "scenery_ground"):  # the ground is one mesh of both street ends
        a0, b0, _, a1, b1, _ = o["box"]
        assert a0 >= x1 - 1e-3 or a1 <= x0 + 1e-3 or b0 >= y1 - 1e-3 or b1 <= y0 + 1e-3, o["name"]
