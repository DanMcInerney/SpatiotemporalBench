"""The photoreal appearance in Blender, on a tiny asset world written by the test (a few boxes, a 4 x 4 texture, a
procedural glass): the library build, the entities made from it, and a render of an episode that uses them. Needs
Blender (and a GPU for the render and the preview); marked `blender`, skipped when Blender is not installed. Nothing is
downloaded and no real asset is used. About half a minute for the module (one library build, one driver run in Blender,
three two-frame renders, one preview)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from PIL import Image

import asset_world as W
from builders import CHAIR, MUG, MUG2, camera_keyframe, minimal_timeline
from stbench import asset_manifest as A
from stbench import asset_preview as P
from stbench import asset_render as AR
from stbench import blender_run
from stbench import episode as ep
from stbench.catalog import CATALOGS, KINDS
from stbench.leak import find_leaks
from stbench.mapspec import load_map
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
FRAMES = "0:2"


class Library(NamedTuple):
    world: W.World
    blend: Path
    doc: dict
    root: Path


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """The tiny world's library, built once by the real build; the asset cache is the world's, via the environment."""
    root = tmp_path_factory.mktemp("assets")
    world = W.build(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STBENCH_ASSET_CACHE", str(world.cache))
        blend = root / "library" / "test.blend"
        yield Library(world, blend, AR.build_library(world.manifest_path, blend, world.cache), root)


@pytest.fixture(scope="module")
def driver(library):
    """Everything the driver script saw inside Blender: the library's meshes, entities, a pass render, a swatch."""
    out = library.root / "driver.json"
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_asset_driver.py"), "--", "--render", str(REPO / "render"), "--src",
           str(REPO / "src"), "--manifest", str(library.world.manifest_path), "--library", str(library.blend),
           "--out", str(out)]
    blender_run.run(cmd, "checking the asset entities")
    return json.loads(out.read_text(encoding="utf-8"))


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# -- the library ---------------------------------------------------------------------------------------------------
def test_library_build_recentres_joins_and_measures_an_asset(library, driver):
    """Contract (plan section 3.3): every model in the library is one mesh (so one entity is one object with one
    pass_index), standing on z = 0 and centred on x and y (pivot = base centre, V0 section 2), its measured box equal to
    the box the test's own files describe. The OBJ cube is two objects with two materials, off the origin; the slab is a
    Y-up file turned a quarter turn and mirrored into a pair; the glTF is two nodes under a moved parent, Y up.
    Regression caught: a skipped join, a pivot or axis error that floats or sinks items, a pair that overlaps or is
    not centred."""
    meshes = driver["library_meshes"]
    expected = {"M:cube_a": W.CUBE, "M:slab_b": W.SLAB, "M:chair_a": W.CHAIR, "M:gltf_d": W.GLTF,
                "M:proc_glass": (0.07, 0.07, 0.12)}
    assert set(meshes) == set(expected)  # the importers' leftovers are not kept
    for name, (sx, sy, sz) in expected.items():
        lo, hi = meshes[name]["min"], meshes[name]["max"]
        assert np.allclose(lo, (-sx / 2, -sy / 2, 0), atol=2e-5), name
        assert np.allclose(hi, (sx / 2, sy / 2, sz), atol=2e-5), name
    assert meshes["M:cube_a"]["polygons"] == 6 and meshes["M:cube_a"]["materials"] == ["cube_a:top", "cube_a:side"]
    assert meshes["M:slab_b"]["polygons"] == 12  # a slab and its mirror image
    assert sorted(meshes["M:gltf_d"]["materials"]) == ["gltf_d:mat_one", "gltf_d:mat_two"]  # two nodes, one mesh
    assert meshes["M:gltf_d"]["polygons"] == 24 and library.doc["assets"]["gltf_d"]["bbox_m"] == list(W.GLTF)
    assert library.doc["assets"]["cube_a"]["bbox_m"] == list(W.CUBE)
    assert library.doc["assets"]["slab_b"]["bbox_m"] == list(W.SLAB)


def test_every_polygon_keeps_its_material_slot_through_the_library_build(driver):
    """Contract (plan section 3.3: one mesh, a material per slot): a polygon uses the slot its source gave it. The OBJ
    cube has two faces on its textured material and four on its plain one, the glTF two boxes of twelve triangles each
    on two materials. Regression caught: a build that reset every polygon to slot 0 (removing a mesh's material slots
    does), so a lens, a book's pages, a bottle's label or a plant's leaves took the first material's look."""
    meshes = driver["library_meshes"]
    assert meshes["M:cube_a"]["slot_polygons"] == [2, 4]
    assert sorted(meshes["M:gltf_d"]["slot_polygons"]) == [12, 12]


def test_every_slot_of_every_procedural_builder_holds_polygons(driver):
    """Contract (plan section 3.3): a procedural builder returns one polygon group per slot name it declares, so the
    material the manifest names for the slot is what shows on that part. Regression caught: a builder that leaves a
    declared slot empty (its material never shows) or numbers a part past its last slot."""
    assert {"laptop", "plant", "remote", "book", "bottle", "glass", "laundry_basket", "pallet", "tote", "drum", "sack",
            "bin", "pallet_jack", "roll_cage", "forklift", "tote_robot"} <= set(driver["procedural"])
    for name, found in driver["procedural"].items():
        assert len(found["polygons"]) == len(found["slots"]) and all(n > 0 for n in found["polygons"]), (name, found)


def test_a_box_a_millimetre_off_the_manifest_fails_the_build(library, tmp_path):
    """Contract (plan section 3.1): the library build fails when it measures an asset more than 1 mm off the
    `native_bbox_m` the manifest recorded, naming the asset. Regression caught: a changed or re-exported mesh that
    still builds, so catalog boxes and footprints stop describing what renders."""
    def off(raw):
        next(a for a in raw["assets"] if a["id"] == "cube_a")["native_bbox_m"] = [0.2, 0.1, 0.103]
    world = W.build(tmp_path, off)
    with pytest.raises(AR.LibraryError, match="cube_a.*1 mm"):
        AR.build_library(world.manifest_path, tmp_path / "lib.blend", world.cache)


def test_the_gate_accepts_the_library_blender_built(library):
    """Contract: the JSON a real build writes is what `check_library` expects (boxes, slots), so a render can start
    from it. Regression caught: the stand-in library of the free tests drifting from what the build really writes."""
    m = A.load(library.world.manifest_path)
    assert AR.check_library(m, library.blend, library.world.cache) == []


# -- the warehouse's structure -------------------------------------------------------------------------------------
def test_ceiling_lights_are_as_bright_as_their_map_says_and_the_apartment_is_unchanged(driver):
    """Contract (plan-warehouse section 1): `build_scene` gives each ceiling light, in the map's order, the map's
    `watts`. The apartment's fixtures stay at 45 W (they are the apartment's calibrated light: its renders must not
    change) and the warehouse's high-bay fixtures at 5.8 m get the brighter power their map file says, so the check is
    not vacuous. Regression caught: the old constant back in (a warehouse lit by 45 W lamps at 5.8 m, near black), or
    the apartment's lamps changed by the refactor."""
    found = driver["structure"]
    apartment, warehouse = load_map("apartment_a"), load_map("warehouse_a")
    assert found["apartment_a"]["watts"] == [45.0] * len(apartment.ceiling_lights)
    assert found["warehouse_a"]["watts"] == [c.watts for c in warehouse.ceiling_lights]
    assert min(found["warehouse_a"]["watts"]) > 45.0 and len(found["warehouse_a"]["watts"]) == 9


def test_the_warehouse_floor_is_one_concrete_material_and_the_apartments_floors_are_unchanged(driver):
    """Contract (plan-warehouse section 2): in the placeholder appearance all four warehouse regions are floored with
    the one procedural concrete material, and the apartment's five regions keep the floors they had (tile, tile, blue
    tile, wood, carpet: four materials, the hall and the kitchen sharing one). Regression caught: a warehouse region
    on checker tiles, four copies of the concrete (four noise setups), or an apartment floor changed."""
    found = driver["structure"]
    regions = ("dock", "west aisle", "middle aisle", "east aisle")
    assert found["warehouse_a"]["floors"] == {r: "shader:concrete" for r in regions}
    assert found["warehouse_a"]["floor_materials"] == 1
    assert found["apartment_a"]["floors"] == {
        "hall": "floor_tile", "kitchen": "floor_tile", "bathroom": "floor_tile_blue", "living room": "floor_wood",
        "bedroom": "floor_carpet"}
    assert found["apartment_a"]["floor_materials"] == 4


def test_the_assets_appearance_resolves_a_shader_floor_and_a_texture_floor(driver):
    """Contract (plan-warehouse section 3.3): `Appearance.floor_material` returns the library material a floor names,
    builds a `shader:` floor once and shares it between the regions that name it, and returns None for a region the
    manifest does not name. Regression caught: a `shader:concrete` floor looked up as a material asset (KeyError at the
    first warehouse render in assets mode), a concrete rebuilt per region, or the textured floors changed."""
    f = driver["floors"]
    assert f["dock"].split(".")[0] == "shader:concrete"  # (Blender numbers a name a session has used before)
    assert (f["shared"], f["kitchen"], f["other"]) == (True, "mat_a", None)


def test_the_concrete_slab_is_mid_grey_with_a_saw_cut_joint_every_six_metres(driver):
    """Contract (plan-warehouse section 3.3): a flat-lit, diffuse view straight down at the slab shows a mean albedo
    of a pale grey (between 0.15 and 0.35 linear, the greys it is built from being 0.19 to 0.30), uniform enough that no
    column is under half the mean, and across the joint at x = 0 a dark band of 12 mm (6 pixels at 2 mm a pixel, 4 to
    8 counted) centred on it, darker than half the slab. Regression caught: a slab as dark as the ground or as white as
    the ceiling (a warehouse that cannot be told from its walls), a joint missing or in the wrong place, or one wide
    enough to read as a lane."""
    joint, slab = np.array(driver["concrete"]["joint"]), np.array(driver["concrete"]["slab"])
    assert 0.15 < slab.mean() < 0.35 and (slab > 0.5 * slab.mean()).all()
    dark = np.flatnonzero(joint < 0.5 * np.median(joint))
    assert 4 <= len(dark) <= 8 and abs(dark.mean() - (len(joint) - 1) / 2) < 2, dark


def test_a_rack_draws_stock_in_exactly_the_bay_levels_the_map_gives_no_surface(driver):
    """Contract (plan-warehouse section 1, 7.2): `pallet_rack` draws its static stock (cartons on pallets, never an
    entity) in every bay-level of the rack that has no surface and in no other, and a rack with no surface at all is
    stocked in all 12 (4 bays x 3 levels). The expected cells are worked out here from the map's surface heights and
    centres: bay = 0..3 for a surface centre y of -4.05, -1.35, 1.35, 4.05, level = 0..2 for z of 0.0, 1.3, 2.6.
    Regression caught: stock on a surface (an item sinking into cartons), a free bay-level that is not free, or a rack
    that ignores the map and stocks every bay-level the same way."""
    bay, level = {-4.05: 0, -1.35: 1, 1.35: 2, 4.05: 3}, {0.0: 0, 1.3: 1, 2.6: 2}
    racks = driver["rack_stock"]
    assert len(racks) == 6
    everything = [[b, v] for b in range(4) for v in range(3)]
    for rack_id, r in racks.items():
        free = [[bay[cy], level[z]] for cy, z in r["surfaces"]]
        assert len(free) == 6 and r["bare"] == everything, rack_id
        assert r["stock"] == [c for c in everything if c not in free], rack_id


def test_a_rack_draws_a_wire_deck_in_every_free_bay_level_above_the_floor(driver):
    """Contract (review F3, plan-warehouse section 1): an item that rests at level 1 or 2 of a pallet rack rests on a
    deck, a thin plate between the beams, and not on nothing: every free bay-level (a surface in the map) above the
    floor has one deck whose top is the surface's height (1.3 or 2.6 m), 2 cm thick, between the beams (the rack is 1.1
    m deep with 0.14 m beams, so within 0.39 m of its middle) and within its own bay (2.7 m wide, so most of it); level
    0 (the floor) and every stock bay-level have none, and a rack with no surface has none. Expected cells are worked
    out here from the map's surface heights and centres, as in the stock test above. Regression caught: a tote, bin, box
    or drum on level 1 or 2 hanging between two beams (846 of 1,140 elevated placements of the dev set), a deck under
    the stock (a pallet of cartons sunk into it), a deck on the floor level, or a deck that overlaps a beam's top face
    (it z-fights with it)."""
    bay, level = {-4.05: 0, -1.35: 1, 1.35: 2, 4.05: 3}, {0.0: 0, 1.3: 1, 2.6: 2}
    for rack_id, r in driver["rack_stock"].items():
        expected = sorted((bay[cy], level[z]) for cy, z in r["surfaces"] if level[z] >= 1)
        assert len(expected) == 4 and r["bare_decks"] == [], rack_id  # 6 free bay-levels, 2 on the floor
        assert [(b, v) for b, v, *_ in r["decks"]] == expected, rack_id
        for b, v, top, thick, x0, x1, y0, y1 in r["decks"]:
            assert top == pytest.approx(1.3 * v) and thick == pytest.approx(0.02), (rack_id, b, v)
            assert -0.39 <= x0 < x1 <= 0.39 and x1 - x0 > 0.75, (rack_id, b, v)
            assert -5.4 + 2.7 * b < y0 < y1 < -5.4 + 2.7 * (b + 1) and y1 - y0 > 2.4, (rack_id, b, v)


# -- entities ------------------------------------------------------------------------------------------------------
def test_identical_entities_share_mesh_and_material(driver):
    """Contract (plan section 3.3: identical means identical in every pixel statistic): two entities of one asset and
    colour share the mesh and the material objects, a third colour shares the mesh but not the materials, each material
    slot is per object, and the tint targets are the colours' linear values. Regression caught: per-entity material
    copies that differ, or a shared material that is retinted by the next entity."""
    s = driver["sharing"]
    assert s["mesh_shared"] and s["materials_shared_when_identical"] and s["materials_differ_when_colour_differs"]
    assert s["materials_per_object_slot"] and s["pass_indices"] == [1, 2, 3, 4, 5]
    brown, blue = _srgb_to_linear((0.36, 0.20, 0.09)), _srgb_to_linear((0.08, 0.25, 0.80))
    assert all(np.allclose(t, brown, atol=2e-4) for t in s["targets_brown"])
    assert all(np.allclose(t, blue, atol=2e-4) for t in s["targets_blue"])
    assert not s["mesh_shared_across_assets"]


def test_a_swapped_colour_uses_the_palette_texture_and_the_others_are_tinted(driver):
    """Contract (manifest `tint.mode` swap): the chair's "oak" is the manifest's oak texture itself, its "black" a
    tinted copy of its own material, and the uniform fit leaves a chair of the asset's own box at scale 1. Regression
    caught: oak painted as a flat tint, or a swap that also retextures the other colours."""
    s = driver["sharing"]
    assert s["chair_oak_material"] == ["mat_a"]
    assert np.allclose(s["chair_black_targets"][0], _srgb_to_linear((0.20, 0.20, 0.21)), atol=2e-4)
    assert s["chair_scale"] == [1.0, 1.0, 1.0]


def test_every_pixel_of_an_asset_entity_carries_its_pass_index(driver):
    """Contract (V0 section 5): an entity is one object, so a view that holds only the asset (here the top and front
    faces of the two-part cube, filling the frame) has its pass_index in every pixel, with a depth that is the cube's.
    Regression caught: a multi-part asset with a part left at pass 0, so visibility under-counts it."""
    r = driver["render"]
    assert r["index_values"] == [7.0] and r["index_all_seven"]
    assert 0.05 < r["depth_max"] < 0.25


def test_tint_sets_the_mean_albedo_to_the_palette_colour(driver):
    """Contract (VISION: the colour attribute is what the pixels show): a textured surface tinted "brown" has the
    palette brown's mean albedo, whatever the texture's own colour. The test texture is a grey ramp in orange; the tint
    group's Mean is the texture's mean luminance (recomputed here from its pixels; the cube's UVs cover the whole
    texture, so the surface's mean is the image's, to the sampling error of the draw), its Target is brown's linear
    value, and a flat-lit view of the surface averages to brown, in brown's own hue and not the texture's. Regression
    caught: a tint that keeps the texture's hue, a wrong mean, or an sRGB value used as linear."""
    r = driver["render"]
    texels = _srgb_to_linear(W.texture_pixels() / 255.0)
    mean_lum = float((texels @ np.array([0.2126, 0.7152, 0.0722])).mean())
    brown = _srgb_to_linear((0.36, 0.20, 0.09))
    assert r["top_slot"].startswith("cube_a:top") and abs(r["top_mean_input"] - mean_lum) < 1e-3
    assert np.allclose(r["top_target"], brown, atol=2e-4)
    ratio = np.array(r["swatch_mean_rgb"]) / brown
    assert ratio.min() > 0.9 and ratio.max() < 1.05  # the mean albedo is brown's (the texture's edges cost a little)
    assert ratio.max() - ratio.min() < 0.02  # in brown's hue: the orange ramp would skew the channels by far more


def test_tint_on_a_half_used_atlas_sets_the_surface_albedo_to_the_palette_colour(driver):
    """Contract (VISION: the colour attribute is what the pixels show), end to end through the library's
    `finish_slots`: a slot whose triangles use a quarter of an atlas (the rest black, as a scanned object's atlas is)
    gets, as the tint's Mean, the mean luminance of the texels its own triangles map to, and a flat-lit view of the
    surface averages to the palette colour. The top face maps to the lit bottom-left quarter, the other slot's faces to
    the black top-right quarter; the whole image's mean is a quarter of the surface's and is only recorded. Regression
    caught: the bug through `finish_slots` (the whole-image mean, which shows the surface 3 times too bright), a mean
    over every polygon instead of the slot's (a sixth of the lit mean, whose boost is capped at 4), or a top-down read
    of Blender's bottom-up pixels (the black quarter, a mean of nothing)."""
    h = driver["half_atlas"]
    lum = _srgb_to_linear(np.array(h["atlas"]) / 255.0) @ np.array([0.2126, 0.7152, 0.0722])  # rows from the bottom
    lit, whole = float(lum[:2, :2].mean()), float(lum.mean())
    top, side = h["report"]
    assert (top["slot"], top["tinted"], side["tinted"]) == ("top", True, False)
    assert side["mean_lum"] is None and side["mean_lum_image"] is None
    assert top["mean_lum"] == pytest.approx(lit, rel=0.02) and top["mean_lum_image"] == pytest.approx(whole, rel=1e-4)
    assert lit > 3.9 * whole and h["tint_mean_input"] == pytest.approx(top["mean_lum"], rel=1e-4)
    ratio = np.array(h["swatch_mean_rgb"]) / _srgb_to_linear((0.36, 0.20, 0.09))
    assert ratio.min() > 0.95 and ratio.max() < 1.05  # the surface's mean albedo is brown's (the old mean gives 3)
    assert ratio.max() - ratio.min() < 0.02  # in brown's hue, not the orange ramp's


def test_a_tinted_frosted_glass_shows_its_colour(driver):
    """Contract (VISION: the colour attribute is what the pixels show; the data-cal contact sheet showed blue glass as
    a dark navy): a blue and a green frosted glass in a lit scene keep at least half the luminance of the same glass
    made opaque, and the blue one keeps its hue (blue far above green, green not below red). Regression caught: a
    fully transmissive glass whose tint compounds over the four surfaces a ray crosses (palette blue to the fourth
    power: about a quarter of the luminance and red above green)."""
    luma = np.array([0.2126, 0.7152, 0.0722])
    for colour, glass in driver["frosted_glass"].items():
        assert glass["pixels"] > 500, colour
        assert np.dot(glass["glass"], luma) >= 0.5 * np.dot(glass["opaque"], luma), (colour, glass)
    blue = driver["frosted_glass"]["blue"]["glass"]
    assert blue[2] > 3 * blue[1] and blue[1] >= blue[0], blue


def test_the_palettes_black_is_a_real_black(driver):
    """Contract (VISION: the colour attribute is what the pixels show): "black" is an albedo of 0.02 to 0.06 in linear
    light, as real black plastic and ceramic are, so a black item shows its shading and its shape. Regression caught:
    a black near albedo 0.004 (sRGB 0.045), which rendered every black scanned item as a pure-black silhouette in the
    data-cal contact sheet."""
    assert all(0.02 <= c <= 0.06 for c in driver["palette_black"]), driver["palette_black"]


def test_each_mover_builder_is_built_to_its_design_box(driver):
    """Contract (V0 section 5, plan-t3 section 5): a mover is fitted exactly to its catalog box (the footprint the
    generator and the visibility rules use), and the fit stretches the mesh to make it so, so each builder must make
    geometry whose own extents are `DESIGN_BOX` to a millimetre, and the box is the catalog's. Regression caught: a
    builder edited (a bigger turret, a wider wheel, a lens that sticks out) without its box, which the exact fit then
    hides by squashing the whole robot, or a design box that drifted from the catalog's mover size."""
    found = driver["mover_extents"]
    movers = {name: kind for c in CATALOGS.values() for name, kind in c.mover_kinds.items()}  # every catalog's
    assert set(found) == set(movers)
    for kind, s in found.items():
        assert np.allclose(s["natural"], s["design"], atol=0.001), (kind, s)
        assert tuple(s["design"]) == movers[kind].size, kind


def test_a_pallet_is_a_wooden_pallet_under_a_wrapped_load_and_a_drum_has_steel_rings(driver):
    """Contract (plan-warehouse section 3.1, VISION rule 10): the colour attribute shows on the part a person would
    call the colour. The pallet's wrap slot is the load (most of its polygons are the bevelled block and its ridges, in
    the colour) and its wood slot the deck, blocks and bottom boards (19 slabs of 6 polygons: 114); the drum's body slot
    is the colour and its rings slot the lock ring, the foot and the bungs. Regression caught: a builder whose slots are
    numbered the wrong way round, which paints the pallet's wood in the wrap colour and the load brown."""
    pallet, drum = driver["procedural"]["pallet"], driver["procedural"]["drum"]
    assert pallet["slots"] == ["wood", "wrap"] and pallet["polygons"][0] == 114 and pallet["polygons"][1] > 200
    assert drum["slots"] == ["body", "rings"] and drum["polygons"][0] > drum["polygons"][1] > 0


def test_every_slab_of_a_warehouse_builder_carries_its_slot_through_the_bevel(driver):
    """Contract (plan-warehouse section 3.1): `meshkit.solid` puts a bevelled slab's rounded edges on its slot (`slab`
    leaves them on slot 0). A pallet's load is a bevelled block in the wrap slot, so slot 0 (the wood) holds exactly the
    114 polygons of the deck, the blocks and the bottom boards, no bevel faces. Regression caught: the pallet's corners
    painted in the wood's colour, a brown frame round every coloured load (visible in the first contact sheet)."""
    assert driver["procedural"]["pallet"]["polygons"][0] == 114


def test_the_laptops_design_box_is_its_geometrys_own_extents(driver):
    """Contract (V0 section 5, VISION rule 10): an exact-fit builder is scaled to `DESIGN_BOX`, which the manifest and
    the catalog follow, so the box must be the extents the builder's geometry has (to a millimetre), else the fit
    quietly stretches the model. Regression caught: the lid's angle or size changed without the box, which squashes the
    open laptop's lid and screen."""
    s = driver["laptop"]
    assert np.allclose(s["natural"], s["design"], atol=0.001), s


def test_the_laptop_is_open_with_its_screen_towards_the_user(driver):
    """Contract (plan section 4, lane B2): an open laptop is recognisable where a closed slab is not (6 of 42 white
    ones were called a laptop, the rest a book). The lid stands 20 degrees past vertical, so the screen's large face
    looks towards the user (+x) and 20 degrees up, and the whole is taller than the base (0.016 m) by a lid's height.
    Regression caught: a laptop closed again (normals straight up or down), a lid opened to another angle or in the
    wrong direction (the screen's normal off by more than 2 degrees)."""
    s = driver["laptop"]
    want = np.array([math.cos(math.radians(20)), 0.0, math.sin(math.radians(20))])
    faces = [f for f in s["screen"] if f["area"] > 0.05]
    assert faces and any(np.dot(f["normal"], want) > math.cos(math.radians(2)) for f in faces), faces
    assert s["natural"][2] > 0.2


# -- an episode ----------------------------------------------------------------------------------------------------
def asset_episode(root: Path):
    """The hand-built kitchen episode on the tiny assets: a mug (the OBJ cube), a glass (the frosted procedural
    one), an oak chair and the textured kitchen floor, seen from across the room at 320 x 240."""
    tl = minimal_timeline()
    glass = dict(kind="glass", asset="proc_glass", size=(0.07, 0.07, 0.12))
    swap = {MUG: dict(asset="cube_a", size=W.CUBE), MUG2: glass, CHAIR: dict(asset="chair_a", size=W.CHAIR)}
    entities = [dataclasses.replace(e, **swap.get(e.id, {})) for e in tl.entities]
    # the camera of every keyframe stands at (3.8, 1.2) or (3.4, 1.2), 1.5 m up, facing west, 15 degrees down
    cameras = [(3.8, 1.2), (3.8, 1.2), (3.4, 1.2), (3.8, 1.2), (3.8, 1.2), (3.4, 1.2)]
    keyframes = [camera_keyframe(k.k, k.t, xy, math.pi, task=k.task) for k, xy in zip(tl.ego.keyframes, cameras,
                                                                                    strict=True)]
    ego = dataclasses.replace(tl.ego, keyframes=keyframes)
    tl = dataclasses.replace(tl, entities=entities, ego=ego)
    paths = ep.write_script(root, tl)
    cam = json.loads(paths.camera.read_text(encoding="utf-8"))
    scale = 320 / cam["width"]
    cam.update(width=320, height=240, fx=cam["fx"] * scale, fy=cam["fy"] * scale, cx=cam["cx"] * scale,
               cy=cam["cy"] * scale)
    paths.camera.write_text(json.dumps(cam), encoding="utf-8")
    return paths


@pytest.fixture(scope="module")
def renders(library, tmp_path_factory):
    """The episode rendered with assets three times: 4 samples, 4 samples again (a new process), 8 samples."""
    out = []
    for i, samples in enumerate((4, 4, 8)):
        paths = asset_episode(tmp_path_factory.mktemp(f"dataset{i}"))
        log = render(paths, frames=FRAMES, appearance="assets", library=library.blend,
                     manifest=library.world.manifest_path, samples=samples)
        out.append((paths, log))
    return out


def test_depth_and_ids_are_identical_across_runs_and_sample_counts_with_assets(renders):
    """Contract (V0 section 3): depth and instance PNGs stay bit-identical between two runs (a new Blender process each)
    and between 4 and 8 samples with the asset appearance, whose scene holds a tinted textured surface, a frosted
    transmissive glass and a texture-swapped chair; and the frames are not empty (the three entities are in the
    instance image). Regression caught: a change to the asset path (a material, the scene's lights or bounces) that
    makes the ID or depth passes depend on the sample count or the run, as it would if a pass were blended."""
    (a, _), (b, _), (c, _) = renders
    for k in range(2):
        for name in ("depth", "instances"):
            ref = getattr(a, name)(k).read_bytes()
            assert getattr(b, name)(k).read_bytes() == ref, f"{name} {k} differs between two renders"
            assert getattr(c, name)(k).read_bytes() == ref, f"{name} {k} differs between 4 and 8 samples"
    seen = set(np.unique(read_png16(a.instances(0)))) | set(np.unique(read_png16(a.instances(1))))
    assert {3, 4} <= seen  # the mug (E0003 is entity 3) and the glass (E0004) are on screen


def test_the_render_log_names_the_appearance_and_the_samples(renders, library):
    """Contract (plan section 3.3 item 7): the private render log and the files record that this was an asset render
    and its samples, and the agent folder carries nothing extra. Regression caught: an asset render indistinguishable
    from a placeholder one in the record."""
    paths, _ = renders[0]
    log = json.loads(paths.render_log.read_text(encoding="utf-8"))
    assert (log["appearance"], log["samples"]) == ("assets", 4) and renders[2][1]["samples"] == 8
    assert find_leaks(paths.root, ["apartment_a"]) == [] and not paths.private.joinpath("render_tmp").exists()
    assert np.asarray(Image.open(paths.rgb(0))).mean() > 20


def test_preview_writes_a_sheet_per_kind_and_one_of_the_surfaces(library, tmp_path):
    """Contract (plan section 2.4 step 4): `assets preview` renders every colour of a kind from four sides and the
    textured surfaces, and assembles one labelled sheet per kind with a column per colour. Regression caught: a preview
    that cannot build the entities the way an episode does, which would hide a squashed or mis-tinted asset."""
    out = tmp_path / "sheets"
    info = AR.require_library(library.world.manifest_path, library.blend, library.world.cache)
    written = P.preview(info.blend, info.manifest_path, out, ["glass", "chair", "mug"], samples=4, tile=(48, 36))
    assert {p.name for p in written} == {"glass.png", "chair.png", "mug.png", "surfaces.png", "index.json"}
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    for kind in ("glass", "chair"):
        with Image.open(out / f"{kind}.png") as sheet:
            assert sheet.size == (48 * len(index["kinds"][kind]["colours"]), 2 * P.LABEL_H + 4 * 36)
    assert index["kinds"]["chair"]["colours"] == ["beige", "black"] and not (out / "tiles").exists()
    # the catalog's mug box is not the test cube's: an episode would refuse it, the sheet shows it and says by how much
    want, have = KINDS["mug"].size, W.CUBE
    ratios = [w / h for w, h in zip(want, have, strict=True)]
    assert index["kinds"]["mug"]["squash"] == pytest.approx(max(ratios) / min(ratios) - 1, abs=1e-3)
    assert index["kinds"]["glass"]["squash"] == 0.0 and index["kinds"]["chair"]["squash"] == 0.0


def test_the_library_command_builds_and_prints_what_to_record_in_the_manifest(library, tmp_path, capsys):
    """Contract (plan section 2.4 step 5): `stbench assets library` builds the library and prints each model's measured
    box and material slots, the slots that take the colour attribute marked, so a person can fill `native_bbox_m` and
    `tint.materials` into the manifest. Regression caught: a command that builds but leaves no way to see what was
    measured, which is how an unmeasured asset reaches a render."""
    from stbench import cli

    out = tmp_path / "again.blend"
    assert cli.main(["assets", "library", "--manifest", str(library.world.manifest_path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert out.is_file() and AR.json_path(out).is_file()
    assert "cube_a" in text and "[0.2, 0.1, 0.1]" in text and "top*" in text and "library built: 5 models" in text
