"""The downtown's dressing in Blender (`render/dressing_build_downtown.py`): the
real `downtown_a.json` is built by the dispatcher twice through the driver the household's test uses
(`tests/blender_dressing_driver.py`; no render and no GPU, no asset library: nothing here is an asset), and what the
scene holds is checked against the layout. Needs Blender; marked `blender`, skipped when it is not installed. About
fifteen seconds. Which pieces the layout emits and where is `test_dressing.py`'s and
`test_dressing_layout_downtown.py`'s; how the dispatcher itself behaves (assets, a catalog with no layout) is
`test_dressing_build_blender.py`'s."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from dressing_kit import layout_of, raw_map, render_module
from stbench import blender_run
from stbench.render_cli import blender_path

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
BOX_EPS_M = 0.005  # a built piece stays inside its Dressing's box by this much more at most
FILL = 0.85  # a piece spans at least this much of its box across and up
S = render_module("street_marks")


@pytest.fixture(scope="module")
def layout() -> list:
    return layout_of("downtown", raw_map("downtown_a"))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """What the dispatcher made of downtown_a's dressing, built twice (see the driver): per object a row, the digest of
    each build and the builders called on their own."""
    out = tmp_path_factory.mktemp("downtown") / "built.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_dressing_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--out", str(out), "--map", "downtown_a", "--catalog", "downtown"],
                    "building the downtown's dressing")
    return json.loads(out.read_text(encoding="utf-8"))


def test_every_dressing_is_one_object_named_by_class_and_count_in_the_dressing_collection_at_index_0(built, layout):
    """Contract (the validity guard: dressing is never an entity): every Dressing of the
    layout comes out as one object, in layout order, named `dressing_<class>_<n>` with n counting the pieces of that
    class from 0, with `pass_index` 0 (an entity's id starts at 1, so none is in an `instance_map`), in the one
    collection "dressing" and no other, and nothing else is left in the file. Regression caught: a piece with an
    entity's pass index (the oracle channel would show it, and a street tree would be a thing to track), a loose object,
    or a class that renders as nothing."""
    seen: Counter = Counter()
    names = []
    for d in layout:
        names.append(f"dressing_{d.cls}_{seen[d.cls]}")
        seen[d.cls] += 1
    assert len(layout) > 75 and [o["name"] for o in built["objects"]] == names  # 82: M3-RY keeps the drone's reach bare
    assert {o["pass_index"] for o in built["objects"]} == {0}
    assert {tuple(o["collections"]) for o in built["objects"]} == {("dressing",)}
    assert built["collections"] == ["dressing"] and built["other_objects"] == []
    assert all(o["verts"] > 0 and o["polys"] > 0 for o in built["objects"])


def test_each_object_stands_at_its_dressings_place_inside_its_box_and_paint_lies_at_the_top_of_it(built, layout):
    """Contract (`dressing_spec`: `pos` is the centre of the piece's base, `box` the axis-aligned box it occupies, which
    `dressing_zones` holds clear of every volume a question or the robot depends on; the guard's flush floor: a plate
    under 2 cm needs no clearance): each object is at its Dressing's `pos` turned `yaw_deg`, and its mesh in the map
    frame lies inside the Dressing's box (+ 5 mm). Each piece spans at least 85% of its box across and along (a run of
    lane dashes starts a metre in and ends before its box does, so its 15 m of 18 are exempt) and, but for paint, which
    is flat quads lying exactly at the top of its box (z = 0.012), 85% of its height. Regression caught: a builder that
    draws in the map frame (a bench across the sidewalk, a crown over the road), geometry that spills past its box (a
    crown past the kerb, where the guard has not looked), or a piece drawn token-sized."""
    assert len(built["objects"]) == len(layout)
    for o, d in zip(built["objects"], layout, strict=True):
        assert o["loc"] == pytest.approx(d.pos, abs=1e-4), o["name"]  # float32 at 100 m
        assert math.cos(o["rot_z"]) == pytest.approx(math.cos(math.radians(d.yaw_deg)), abs=1e-5), o["name"]
        assert math.sin(o["rot_z"]) == pytest.approx(math.sin(math.radians(d.yaw_deg)), abs=1e-5), o["name"]
        lo, hi = d.box[:3], d.box[3:]
        w = o["box"]
        for i in range(3):
            assert w[i] >= lo[i] - BOX_EPS_M and w[i + 3] <= hi[i] + BOX_EPS_M, (o["name"], i, w, d.box)
        if d.cls in ("lane_marking", "crosswalk", "stop_bar"):
            assert w[2] == pytest.approx(0.012, abs=1e-6) and w[5] == pytest.approx(0.012, abs=1e-6), o["name"]
        else:
            across = 1 if abs(math.cos(math.radians(d.yaw_deg))) > 0.5 else 0
            assert w[across + 3] - w[across] >= FILL * d.size[1], (o["name"], w, d.box)
            assert w[5] - w[2] >= FILL * (hi[2] - lo[2]), (o["name"], w, d.box)
        if d.cls in ("crosswalk", "stop_bar"):  # the whole rectangle is painted, or its stripes span it
            assert w[3] - w[0] >= FILL * (hi[0] - lo[0]) and w[4] - w[1] >= FILL * (hi[1] - lo[1]), o["name"]


def test_paint_is_the_marks_pattern_in_white_or_yellow_one_quad_a_stripe(built, layout):
    """Contract (`street_marks.paint_rects` is what is drawn: the dressing builder turns each rectangle of a mark into a
    quad at the top of the box; yellow paint for the double centre line, white for everything else): a piece of paint
    has four vertices a rectangle of its pattern (the crossing over the side street's mouth is eleven stripes: 44
    vertices; a stop bar one: 4; a double line two: 8; a run of dashes one a dash), one material, `paint_yellow` for a
    `double` line and `paint_white` for the rest. Regression caught: a pattern drawn with the wrong stripe count (a
    crossing with a gap), yellow dashes or a white centre line, or paint with no material."""
    rows = {o["name"]: o for o in built["objects"]}
    seen: Counter = Counter()
    for d in layout:
        name = f"dressing_{d.cls}_{seen[d.cls]}"
        seen[d.cls] += 1
        if not d.recipe.startswith("proc:paint"):
            continue
        style, axis = d.recipe.split(":")[2:]
        x0, x1, y0, y1 = d.box[0], d.box[3], d.box[1], d.box[4]
        mark = S.Mark(d.cls, style, (-(x1 - x0) / 2, (x1 - x0) / 2, -(y1 - y0) / 2, (y1 - y0) / 2), axis)
        assert rows[name]["verts"] == 4 * len(S.paint_rects(mark)), name
        assert rows[name]["materials"] == ["paint_yellow" if style == "double" else "paint_white"], name
    crossing = next(o for d, o in zip(layout, built["objects"], strict=True) if d.cls == "crosswalk")
    assert crossing["verts"] == 44


def test_a_builder_on_its_own_returns_unlinked_objects_with_materials_and_nothing_the_dispatcher_does(built, layout):
    """Contract (the builders' interface, `dressing.draw`): each builder returns its piece as one unlinked object with
    an identity transform and `pass_index` 0 and a material on every slot, its mesh built in the piece's own frame
    (origin the centre of its base, so its base is at z 0 and its extents are the box's centred on x and y); placing,
    naming and collecting are the dispatcher's. Regression caught: a builder that links its object (it would be in the
    scene twice), places it itself (the yaw applied twice), draws away from the piece's base centre, or leaves a slot
    with no material."""
    rows = built["builders"]
    assert [r["cls"] for r in rows] == [d.cls for d in layout]
    for r, d in zip(rows, layout, strict=True):
        assert r["n"] == 1 and r["linked"] == [] and r["pass_index"] == [0] and r["identity"] == [True], r["cls"]
        assert all(m for slots in r["materials"] for m in slots) and all(r["materials"]), r["cls"]
        x0, y0, z0, x1, y1, z1 = r["local"][0]
        top = d.size[2] if d.cls in ("lane_marking", "crosswalk", "stop_bar") else None
        assert z1 <= d.size[2] + BOX_EPS_M and (z0 == pytest.approx(top) if top else z0 == pytest.approx(0.0, abs=1e-6))
        assert abs(x0 + x1) / 2 <= BOX_EPS_M + 0.5 * d.size[0] and x1 - x0 <= d.size[0] + 2 * BOX_EPS_M
        assert y1 - y0 <= d.size[1] + 2 * BOX_EPS_M


def test_building_the_same_map_twice_gives_the_same_scene_to_the_last_vertex(built):
    """Contract (determinism: the dressing depends on the map alone): two builds in one
    process, each from an empty file, hold the same objects in the same order with the same transforms, geometry (to a
    tenth of a millimetre), material slots and material names: the crowns' lumps are seeded by the piece's place and
    shape, nothing by `random`'s state or a clock. Regression caught: a builder that draws on `random`'s global state or
    the order a dict happens to iterate (a different tree each render), or a material made twice."""
    first, second = built["digest"]
    assert first == second and built["names"][0] == built["names"][1]


def test_identical_pieces_share_one_mesh_and_trees_are_not_alike(built, layout):
    """Contract (`dressing.build`: pieces with the same recipe, size and class share their mesh): the number of distinct
    meshes among the objects is the number of distinct (recipe, size) pairs in the layout, worked out here from the
    layout (the manholes are one mesh, as are the benches and bike racks; the crowns, each a size and a shape of its
    own, are as many meshes as crowns, nearly). Regression caught: a copy of a mesh for each piece (a larger scene), or
    one mesh shared by pieces of different sizes (a crown stretched over the road), or all trees one mesh (a street of
    clones)."""
    pairs = {(d.cls, d.recipe, d.size) for d in layout}
    assert len({o["mesh"] for o in built["objects"]}) == len(pairs)
    crowns = [o["mesh"] for o, d in zip(built["objects"], layout, strict=True) if d.cls == "tree_crown"]
    assert len(set(crowns)) >= 0.8 * len(crowns) >= 10
    assert len({o["mesh"] for o, d in zip(built["objects"], layout, strict=True) if d.cls == "manhole_cover"}) == 1
