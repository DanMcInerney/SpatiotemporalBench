"""The dressing dispatcher and the household's builders in Blender (`render/dressing.py`,
`render/dressing_build_home.py` and the modules under it): the real
`apartment_a.json` is built twice (`tests/blender_dressing_driver.py`; no render and no GPU, no asset library, so a
Poly Haven piece is its procedural stand-in) and a hand-made asset is fitted into boxes. Needs Blender; marked
`blender`, skipped when it is not installed. About ten seconds. How the warehouse's builders draw is
`test_dressing_build_warehouse_blender.py`'s, which pieces the layout emits and where is the layout's and
`test_dressing.py`'s, and which builders exist is `test_dressing_build_home.py`'s."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from dressing_kit import layout_of, raw_map
from stbench import blender_run
from stbench.render_cli import blender_path

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
BOX_EPS_M = 0.005  # a built piece stays inside its Dressing's box by this much more at most
FILL = 0.85  # a piece spans at least this much of its box across and up (a five-bladed fan: 0.89 of its circle's box)


@pytest.fixture(scope="module")
def layout() -> list:
    return layout_of("home", raw_map("apartment_a"))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """What the dispatcher made of apartment_a's dressing, built twice (see the driver): per object a row, the digest of
    each build, the builders called on their own, the fitted test asset and the result for a catalog with no layout."""
    out = tmp_path_factory.mktemp("home") / "built.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_dressing_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--out", str(out), "--map", "apartment_a", "--catalog", "home"],
                    "building the apartment's dressing")
    return json.loads(out.read_text(encoding="utf-8"))


def test_every_dressing_is_one_object_named_by_class_and_count_in_the_dressing_collection_at_index_0(built, layout):
    """Contract (the validity guard: dressing is never an entity: every dressing object
    has `pass_index` 0 and never becomes an entity): the 245 Dressings of the layout (178 of trim, fittings and fixtures
    and the 67 pieces of grime on the floor, counted by hand) come out as 245 objects in layout order, named
    `dressing_<class>_<n>` with n counting the pieces of that class from 0, each with `pass_index` 0 (an entity's id
    starts at 1, so none is in an `instance_map`), each in the one collection "dressing" and no other, and
    nothing else is left in the file (no template object of an asset). The 20 pieces with an `asset:` recipe (16
    pictures, 2 clocks, the pendant and the fan) are there too, as their stand-ins. Regression caught: a piece with
    an entity's pass index (the oracle channel would show it),
    a name that depends on the order across classes, a loose object, or an asset piece dropped for want of its model."""
    seen: Counter = Counter()
    names = []
    for d in layout:
        names.append(f"dressing_{d.cls}_{seen[d.cls]}")
        seen[d.cls] += 1
    assert len(layout) == 245 and sum(d.recipe.startswith("asset:") for d in layout) == 20
    assert [o["name"] for o in built["objects"]] == names
    assert {o["pass_index"] for o in built["objects"]} == {0}
    assert {tuple(o["collections"]) for o in built["objects"]} == {("dressing",)}
    assert built["collections"] == ["dressing"] and built["other_objects"] == []
    assert all(o["verts"] > 0 and o["polys"] > 0 for o in built["objects"])


def test_each_object_stands_at_its_dressings_place_and_yaw_inside_its_box(built, layout):
    """Contract (`dressing_spec`: `pos` is the centre of the piece's base, `yaw_deg` the way it faces, `box` the
    axis-aligned box it occupies after the yaw, which `dressing_zones` holds clear of every volume a question or the
    robot depends on): each object is at its Dressing's `pos` and turned `yaw_deg` about z, and its mesh in the map
    frame lies inside the Dressing's `box` (+ 5 mm), which checks the frame the builders draw in (x out of the wall, y
    across, z up) against the layout's yaw. Each piece also spans at least 85% of the box across and up (so no builder
    draws a token or a fixed size): the plate fills its plate, the radiator its 0.8 m. Regression caught: a builder that
    draws in the map frame or the wrong way round (a baseboard across the room), geometry that spills past its box (a
    cabinet handle into the room), a size the builder ignores, or a dispatcher that drops the yaw."""
    assert len(built["objects"]) == len(layout)
    for o, d in zip(built["objects"], layout, strict=True):
        assert o["loc"] == pytest.approx(d.pos, abs=1e-6), o["name"]
        assert math.cos(o["rot_z"]) == pytest.approx(math.cos(math.radians(d.yaw_deg)), abs=1e-5), o["name"]
        assert math.sin(o["rot_z"]) == pytest.approx(math.sin(math.radians(d.yaw_deg)), abs=1e-5), o["name"]
        lo, hi = d.box[:3], d.box[3:]
        w = o["box"]
        for i in range(3):
            assert w[i] >= lo[i] - BOX_EPS_M and w[i + 3] <= hi[i] + BOX_EPS_M, (o["name"], i, w, d.box)
        local = o["local"]  # the mesh in the piece's own frame, whatever its yaw (a hair lies at any angle)
        assert local[4] - local[1] >= FILL * d.size[1], (o["name"], local, d.size)
        assert local[5] - local[2] >= FILL * d.size[2], (o["name"], local, d.size)


def test_a_builder_on_its_own_returns_objects_with_materials_and_nothing_the_dispatcher_does(built, layout):
    """Contract (the builders' interface, `dressing.draw`; lanes of the warehouse and the city rely on it): a builder
    returns its piece as unlinked objects with an identity transform and `pass_index` 0 and a material on every slot;
    placing, naming and collecting are the dispatcher's. One object a piece for the household's. Regression caught: a
    builder that links its object (it would be in the scene twice), places it itself (the yaw applied twice), or leaves
    a slot with no material (an object that renders magenta or not at all)."""
    rows = built["builders"]
    assert [r["cls"] for r in rows] == [d.cls for d in layout]
    for r in rows:
        assert r["n"] == 1, r["cls"]
        assert r["linked"] == [] and r["pass_index"] == [0] and r["identity"] == [True], r["cls"]
        assert all(m for slots in r["materials"] for m in slots) and all(r["materials"]), r["cls"]


def test_building_the_same_map_twice_gives_the_same_scene_to_the_last_vertex(built):
    """Contract (determinism: the dressing depends on the map alone): two builds in one
    process, each from an empty file, hold the same objects in the same order with the same transforms, geometry (to a
    tenth of a millimetre), material slots and material names. Regression caught: a builder or material that draws on
    `random`'s global state, a clock or the order a dict happens to iterate."""
    first, second = built["digest"]
    assert first == second
    assert built["names"][0] == built["names"][1]


def test_identical_pieces_share_one_mesh_and_different_ones_do_not(built, layout):
    """Contract (`dressing.build`: pieces with the same recipe, size and class share their mesh, so identical items are
    identical in every pixel statistic and a scene stays small): the number of distinct meshes among the objects is the
    number of distinct (recipe, size) pairs in the layout, worked out here from the layout (the 26 sockets are as many
    meshes as they have sizes, the 41 baseboards as many as there are lengths, and the hairs and crumbs as many as their
    four variants and sizes make). Regression caught: a copy of a mesh for
    each piece (a larger, slower scene), or one mesh shared by pieces of different sizes (a baseboard stretched to a
    wall it does not fit)."""
    pairs = {(d.cls, d.recipe, d.size) for d in layout}
    assert len({o["mesh"] for o in built["objects"]}) == len(pairs)
    by_pair: dict = {}
    for o, d in zip(built["objects"], layout, strict=True):
        by_pair.setdefault((d.cls, d.recipe, d.size), set()).add(o["mesh"])
    assert all(len(meshes) == 1 for meshes in by_pair.values())
    sizes = {d.size for d in layout if d.cls == "outlet"}
    assert len({o["mesh"] for o in built["objects"] if o["name"].startswith("dressing_outlet")}) == len(sizes)


def test_an_asset_is_fitted_to_its_box_facing_out_with_its_width_across(built):
    """Contract (the dispatcher's asset recipe: size is (depth, width, height) and a Poly Haven model is width x depth x
    height facing -y, so it must end up filling the box with its front out of the wall): a plank 0.6 wide, 0.05 deep
    (0.04 and a 0.01 bump on its front face) and 0.8 high, its bump at x 0.1 to 0.2 on the front face at heights 0.5
    to 0.6, is fitted into four boxes at yaws 0, 90, 180 and -90. By hand: the fitted box is the piece's box (exactly)
    and the bump's front lies on the facing side (depth / 2 out of the base centre), 0.25 of the width along the piece's
    y (0.15 of 0.6 from the middle) and at 0.6875 of the height (0.55 of 0.8). Regression caught: a model drawn
    sideways (its width out of the wall), turned the wrong way (the front against the wall, which a frame's picture
    would face), mirrored (the bump on the other side) or stretched out of its box."""
    for r in built["fit"]:
        yaw = math.radians(r["yaw"])
        depth, width, height = r["size"]
        assert r["world"] == pytest.approx(r["box"], abs=1e-5), r["yaw"]
        facing, across = (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))
        want = [r["pos"][0] + facing[0] * depth / 2 + across[0] * 0.25 * width,
                r["pos"][1] + facing[1] * depth / 2 + across[1] * 0.25 * width, r["pos"][2] + 0.6875 * height]
        assert r["bump"] == pytest.approx(want, abs=1e-4), r["yaw"]


def test_a_library_model_is_found_by_the_id_a_layout_uses_with_its_material_and_fits_its_box(built):
    """Contract (`Ctx.asset` over lane A's `asset_dressing`: a layout names a model by its manifest id, the library
    holds it as `M:<id>` with its materials, and a model the library lacks, or the manifest lists but the library does
    not hold, is None so that the stand-in is drawn): a one-model library written here is read back through `Ctx(app)`
    by the id, the model arrives with its material, both kinds of missing id are None, the model fits a 0.03 x 0.8 x 0.9
    box at yaw 90 exactly, and releasing the context leaves no loose template. Regression caught: a lookup that cannot
    find the library's names (the scene would silently be all stand-ins), a model without its material (pink in the
    render), a missing id that raises instead of falling back, or a template object left in the scene."""
    lib = built["library"]
    assert lib["mesh"] == "M:test_slab" and lib["materials"] == ["test_slab:wood"]
    assert lib["missing"] == [True, True] and lib["loose"] == []
    assert lib["world"] == pytest.approx(lib["box"], abs=1e-5)


def test_a_catalog_with_no_look_is_an_error_and_builds_nothing(built):
    """Contract (the dispatcher's: a catalog's layout is the `DRESSING` of its `look_<catalog>`): building a catalog
    that has no look module raises `LookError`, makes no collection and leaves no object in the file. Regression caught:
    a catalog nothing knows quietly built with no dressing (an empty room), or a collection or object left behind."""
    assert built["no_layout"] == ["LookError", [], 0]


def test_the_probe_names_each_dressing_piece(built, layout):
    """Contract (`dressing.DEBUG_INDEX_FROM`, `realism_probe.py`; the realism probe's index frames): with the debug base
    set to 1000 every piece of dressing has an object index of its own, 1000 up in layout order (one object a piece in
    the household), `dressing.INDEX` maps each to its class, and entity indices are unchanged (a world's 160-250
    entities are 1 to 250, under the base); with it unset, as in every dataset render, all of it is 0 again.
    Regression caught: a probe whose pixel counts cannot tell a radiator from a baseboard, ids that collide with an
    entity's, or a debug index that stays on for the renders that make a dataset."""
    debug = built["debug"]
    assert debug["indices"] == list(range(1000, 1000 + len(layout)))
    assert {int(k): v for k, v in debug["index"].items()} == {1000 + i: d.cls for i, d in enumerate(layout)}
    assert 1000 + len(layout) < 65535 and debug["after"] == [0]
