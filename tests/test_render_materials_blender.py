"""The materials, bevels, floor markings and wear of the dressing in Blender: bevelled
box parts, the named PBR materials, the warehouse floor's painted lines and tyre marks, and the finish a built apartment
gets from `wear.apply` while every tracked thing keeps its materials. One Blender script (`blender_materials_driver.py`)
does all of it and the tests read what it printed. Needs Blender; marked `blender`, skipped when Blender is not
installed. Nothing is downloaded and no library is built: the scene is the placeholder drawings of a hand-made
apartment episode, and the floor is looked at straight down on the CPU under a white sky, where a pixel is the
albedo. About half a minute for the module."""

from __future__ import annotations

import dataclasses
import json
import math
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from builders import DOOR, minimal_timeline
from conftest import apartment
from stbench import blender_run
from stbench import episode as ep
from stbench.catalog import item_size
from stbench.geometry import quat_z
from stbench.render_cli import blender_path
from stbench.timeline import Entity, Placement

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
def part(loc, dims, mat, bevel, rot=(0, 0, 0), segments=2) -> dict:
    """A box `Part` as the driver script takes it (shape, loc, dims, rot, mat, bevel, segments)."""
    return dict(shape="box", loc=list(loc), dims=list(dims), rot=list(rot), mat=mat, bevel=bevel, segments=segments)


# a plain bevelled box, a turned one, four unit boxes of three materials side by side (two bevelled, two plain), and a
# plate thinner than its bevel is wide
BEVEL_CASES = {
    "plain": [part((0.3, -0.2, 0.5), (0.8, 0.5, 0.4), "grey", 0.008)],
    "turned": [part((1.0, 2.0, 0.5), (0.6, 0.3, 0.2), "grey", 0.005, rot=(0.3, -0.4, 0.9))],
    "mixed": [part((0.5, 0.5, 0.5), (1, 1, 1), "red", 0.01), part((2.5, 0.5, 0.5), (1, 1, 1), "blue", 0.0),
              part((0.5, 2.5, 0.5), (1, 1, 1), "green", 0.02, segments=3),
              part((2.5, 2.5, 0.5), (1, 1, 1), "blue", 0.0)],
    "thin": [part((0, 0, 0.01), (1.0, 1.0, 0.02), "grey", 0.05)],
}
CONTRACT_NAMES = ("plaster_wall", "painted_wood_trim", "white_plastic", "brushed_steel", "painted_metal_red",
                  "painted_metal_grey", "galvanised_steel", "curtain_fabric", "glass_panel", "ceramic_tile",
                  "concrete_floor", "cardboard", "label_paper", "rubber_black", "led_emitter", "led_emitter_warm",
                  "led_emitter_cool")
EMITTERS = ("led_emitter", "led_emitter_warm", "led_emitter_cool", "skylight_panel")
# (kind, colour) of the tracked things in the apartment scene, and the palette colours (sRGB) their body is made of
ITEMS = [("mug", "red"), ("book", "yellow"), ("towel", "blue"), ("laptop", "grey")]
PALETTE_SRGB = {"red": (0.80, 0.07, 0.07), "yellow": (0.95, 0.80, 0.08), "blue": (0.08, 0.25, 0.80),
                "grey": (0.45, 0.45, 0.47)}


def linear(c: float) -> float:
    """sRGB to linear light (the IEC 61966-2-1 curve), worked here and not by `shapes`."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def apartment_timeline():
    """The map's fixed and light furniture as entities, one of each of four carried items and a mover, and the base
    timeline's door; the placeholder drawings of them are what `wear.apply` is given."""
    m, base = apartment(), minimal_timeline()
    entities = [e for e in base.entities if e.id == DOOR]
    placements = [p for p in base.placements if p.entity == DOOR]

    def add(kind, attributes, mobility, pos, yaw, size):
        eid = f"E{100 + len(entities):04d}"  # clear of the door's id
        entities.append(Entity(eid, kind, attributes, mobility, size))
        placements.append(Placement(eid, 0.0, None, pos, quat_z(yaw), None))

    for f in m.furniture:
        add(f.kind, {"name": f.name, "color": f.color}, f.mobility, (*f.center, 0.0), f.yaw, f.size)
    for i, (kind, colour) in enumerate(ITEMS):
        add(kind, {"color": colour}, "carried", (1.0 + 0.3 * i, 1.0, 0.9), 0.0, item_size(kind))
    add("robot_vacuum", {"color": "white"}, "mover", (2.0, 1.0, 0.0), 0.0, item_size("robot_vacuum"))
    return dataclasses.replace(base, entities=entities, placements=placements, states=[])


@pytest.fixture(scope="module")
def found(tmp_path_factory):
    """Everything the driver script found, from one Blender run."""
    root = tmp_path_factory.mktemp("materials")
    paths = ep.write_script(root / "episode", apartment_timeline())
    spec = json.loads(resources.files("stbench.maps").joinpath("warehouse_a.json").read_text(encoding="utf-8"))
    masks = root / "masks"  # three of the manifest's four mask sets, as 4 x 4 greys (Scratches004 is left out)
    for set_id in ("acg_SurfaceImperfections003", "acg_Smear005", "acg_Fingerprints002"):
        (masks / set_id / "x").mkdir(parents=True)
        grey = Image.fromarray(np.full((4, 4), 90, dtype=np.uint8))
        grey.save(masks / set_id / "x" / f"{set_id[4:]}_1K-JPG_Opacity.png")
    cases = root / "cases.json"
    cases.write_text(json.dumps({"bevel": BEVEL_CASES, "warehouse_spec": spec, "masks_root": str(masks)}),
                     encoding="utf-8")
    out = root / "result.json"
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_materials_driver.py"), "--", "--render", str(REPO / "render"), "--src",
           str(REPO / "src"), "--private", str(paths.private), "--cases", str(cases), "--out", str(out)]
    blender_run.run(cmd, "checking the dressing's materials")
    return json.loads(out.read_text(encoding="utf-8"))


# -- bevels --------------------------------------------------------------------------------------------------------
def rotation(part: dict) -> np.ndarray:
    """The part's rotation matrix as Blender's XYZ Euler gives it (Rz Ry Rx), worked out here with numpy."""
    rx, ry, rz = part["rot"]
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def corners(part: dict) -> np.ndarray:
    """The 8 sharp corners of a box part in the entity frame."""
    half = np.array(part["dims"]) / 2
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    return np.array(part["loc"]) + (signs * half) @ rotation(part).T


@pytest.mark.parametrize("case", sorted(BEVEL_CASES))
def test_a_bevelled_box_keeps_its_bounding_box_and_loses_its_sharp_corners(case, found):
    """Contract (`shapes.Part.bevel`): rounding a box's edges takes material off the corners only, so the vertices of a
    part, taken back into its own frame (a turned box's rotation undone with numpy), span exactly its dims, face to
    face, while no vertex is left on a sharp corner (the nearest is at least 0.3 of the bevel width off, where the
    rounding has taken it; a bevel wider than half the thinnest side is cut back to 0.45 of it, so the thin plate keeps
    its 2 cm). The flat tops of the furniture the map rests items on (a counter's worktop at 0.9 m, a sofa's seat) thus
    stay at the height the map gives. Regression caught: a bevel that grows the part (an offset taken outside), one that
    shrinks its flat top, an unrounded corner, or a bevel wider than the plate that folds the mesh inside out."""
    parts, mesh = BEVEL_CASES[case], found["bevel"][case]
    verts = np.array(mesh["verts"])
    for p in parts:
        local = (verts - np.array(p["loc"])) @ rotation(p)  # every vertex in this part's frame
        mine = local[np.all(np.abs(local) <= np.array(p["dims"]) / 2 + 1e-4, axis=1)]  # (the others are other parts')
        if len(parts) == 1:
            assert np.abs(mine).max(axis=0) == pytest.approx(np.array(p["dims"]) / 2, abs=2e-5), case
        if p["bevel"] > 0:
            width = min(p["bevel"], 0.45 * min(p["dims"]))
            nearest = [np.linalg.norm(verts - c, axis=1).min() for c in corners(p)]
            assert min(nearest) > 0.3 * width, (case, nearest)
            assert len(mesh["slots"]) > 6 * len(parts)  # the rounding has faces of its own


def test_the_faces_of_a_bevelled_box_stay_on_its_own_material_slot(found):
    """Contract (`shapes.entity_mesh`): every face of a part, the strips of its rounding included, is on the slot of the
    part's material, with bevelled and plain boxes mixed in one mesh (the bevelled ones are added last, and a bevel
    frees faces that the next part's faces would otherwise fill). Three unit boxes (red bevelled, blue plain, green
    bevelled) and a second blue one stand apart; each polygon's centre is in exactly one of them. Regression caught:
    faces of a bevelled box on slot 0 (the first material, which turned a white fridge and a marble worktop black when
    the bevel replaced their flat faces) or the faces of a plain box taken for the wrong part's after a bevel."""
    mesh, parts = found["bevel"]["mixed"], BEVEL_CASES["mixed"]
    assert mesh["materials"] == ["red", "blue", "green"]
    expected = {"red": 0, "blue": 1, "green": 2}
    for centre, slot in zip(mesh["centres"], mesh["slots"], strict=True):
        owners = [p for p in parts if all(abs(centre[i] - p["loc"][i]) <= p["dims"][i] / 2 + 1e-4 for i in range(3))]
        assert len(owners) == 1, centre
        assert slot == expected[owners[0]["mat"]], (centre, slot)


# -- the named materials -------------------------------------------------------------------------------------------
def test_every_named_material_builds_and_the_contract_names_exist(found):
    """Contract (the dressing lanes' shared contract): `materials.named` builds each of the 15 names the home and
    warehouse builders ask for (and the warm and cool lamp variants) as a Principled material with a node graph, an
    opaque surface, the same material for the same name in one scene, and a clear KeyError for a name it does not have;
    `dressing_material` answers None for it so a caller can fall back; `named` also answers an existing shader name;
    and after a scene rebuild frees every material it builds a fresh one rather than handing out a freed one.
    Regression caught: a builder that fails on a Blender version's socket names, a name dropped or renamed (the builders
    would fail at the first render), an unknown name that returns a plain material silently, or a stale cache."""
    named = found["named"]
    assert set(CONTRACT_NAMES) <= set(named)
    for name, info in named.items():
        if name.startswith("_"):
            continue
        assert info["nodes"] >= 3 and info["same"] and info["alpha"] == 1.0, name
        assert info["name"].endswith(name), name
    assert "no_such_material" in named["_unknown_error"] and "plaster_wall" in named["_unknown_error"]
    assert named["_dressing_none"] is True
    assert named["_shader_name"] == "shader:concrete"
    assert named["_after_reset"].endswith("plaster_wall")


def test_a_lamp_lens_glows_for_the_camera_only(found):
    """Contract (lane R owns every real light): the lamp and skylight materials multiply their emission strength by the
    Light Path node's `Is Camera Ray`, so the lens shows lit and casts no light of its own (the same light counted
    twice would double the room's exposure); and they do emit (a strength above 0 for the camera). Regression caught:
    an emitter left ungated (every high-bay lens a second light source), or one with no emission at all (a black
    lens)."""
    for name in EMITTERS:
        info = found["named"][name]
        assert "ShaderNodeLightPath" in info["emission_upstream"], name
    assert all(found["named"][n]["emission_strength"] == 0.0 for n in EMITTERS)  # the strength is the linked product


def test_the_stretch_film_shader_is_a_glossy_opaque_wrap_that_a_tint_can_colour(found):
    """Contract (lane G, the pallet's `wrap` slot): `shader:stretch_film` is built like `plastic`, a constant 0.9 grey
    base colour (so the manifest's `stbench_tint` group can set the catalog colour on it and the colour stays true),
    but glossy (roughness 0.12 to 0.18), with no transmission and no alpha (opaque: nothing behind it, no wash-out)
    and a bump of wrinkles. Regression caught: a film with real transmission (a pallet whose colour washes out or goes
    dark), a base colour driven by a texture (the tint would be applied to a pattern), or a matte film."""
    film = found["named"]["_stretch_film"]
    assert film["base"] == pytest.approx([0.9, 0.9, 0.9, 1.0]) and not film["base_linked"]
    assert 0.12 <= film["roughness"] <= 0.18 and film["transmission"] == 0.0 and film["alpha"] == 1.0
    assert film["bump"] is True


# -- the warehouse floor -------------------------------------------------------------------------------------------
def test_the_floor_has_yellow_lines_at_the_rack_fronts_and_hatching_at_the_doors(found):
    """Contract (plan-realism section 4): looked at straight down under a white sky, the slab at the first rack line
    (x = 1.45 to 1.55) has yellow in its 10 central columns of a 100-pixel window of 1 cm pixels, and none in the 40
    columns either side; the open slab between the aisle's tyre streaks is grey (red, green and blue within 8% of one
    another) with no yellow; the first door's apron has yellow in part of it (diagonal stripes: 12 to 60% of the
    pixels); and a slab built with no map has no yellow anywhere. Regression caught: markings on the wrong place or
    colour (grey or white paint, a line a metre off), a slab tinted yellow all over, or markings in a scene built
    without a map."""
    f = found["floor"]
    cols = np.array(f["line_yellow_cols"])
    assert cols[46:54].min() > 0.5 and cols[:40].max() < 0.05 and cols[60:].max() < 0.05, cols
    bare = np.array(f["bare_rgb_mean"])
    assert (bare.max() - bare.min()) / bare.mean() < 0.08 and f["bare_yellow"] < 0.01, bare
    assert 0.12 < f["hatch_yellow"] < 0.6, f["hatch_yellow"]
    assert f["plain_yellow"] < 0.01


def test_tyre_marks_darken_the_slab_along_the_aisles(found):
    """Contract (plan-realism section 4, tyre marks): across the west aisle's west streak (x = 2.38 to 2.52) the slab is
    at least 8% darker than the open slab 20 to 50 cm from it. Regression caught: tyre marks missing from the material
    (a floor that is clean everywhere) or a streak so faint that no one could see it."""
    cols = np.array(found["floor"]["track_cols"])  # a window from x = 1.95 to 2.95, 1 cm a column
    assert cols[43:57].min() < 0.92 * np.median(cols[70:]), cols


def test_the_fixed_furniture_has_its_surfaces_where_the_map_puts_them_and_nothing_above_them(found):
    """Contract (`furniture.py`, the map's surfaces): items rest at the height a map gives a receptacle
    surface, so on each of the apartment's 24 surfaces (counters, tables, the sofa's seat, the bed, shelves,
    consoles, rugs) a ray dropped from 3 cm above at 35 points inside the surface's rect (shrunk by 4 cm)
    never lands more than 7 mm above that height (the sink's and the vanity's basins are 4 and 6 mm plates
    on the worktop), lands on it (within 1 mm) at 25% of the points at least (the rest are the rounded edge
    of a cushion, the seam between the sofa's two cushions or the gap between the door mat's ribs), and a
    ray cast up from 8 mm above it meets nothing within 5 cm: no cushion, pillow or handle stands in the
    volume an item sits in. Regression caught: a pillow or duvet that rises above the bed's surface (a towel
    on it sinks), a sofa cushion that sinks below its seat height, a back cushion that leans into the seat's
    rect, or a bevel that rounds a worktop's top away."""
    surfaces = found["surfaces"]
    assert len(surfaces) == 24
    for name, s in surfaces.items():
        assert all(h is not None and h <= s["z"] + 0.007 for h in s["down"]), (name, s["kind"], s["down"])
        flat = sum(1 for h in s["down"] if s["z"] - 0.001 <= h)
        assert flat >= 0.25 * len(s["down"]), (name, s["kind"], s["down"])
        assert all(h is None for h in s["up"]), (name, s["kind"], s["up"])


def test_the_wear_reads_the_pinned_mask_images_where_they_are_and_noise_where_they_are_not(found):
    """Contract (plan-realism section 4, the SurfaceImperfections, Smear, Scratches and Fingerprints masks with a
    procedural fallback): with no mask loaded no worn finish has an image node (the noise stands in); `load_masks`
    loads each manifest set whose Opacity map is on disk as non-colour data and leaves a missing set (Scratches004) out;
    with them a worn wall has the imperfections image, a tile floor the smear, furniture the fingerprints, and a rack's
    paint, whose scratches are the missing set, none; and with none loaded again there is no image once more.
    Regression caught: a build that needs a download (an image node with no file), a mask read as colour (gamma applied
    to a data map), a missing set that raises, or a stale mask that stays in the finishes."""
    m = found["masks"]
    none = {"wall": 0, "tile": 0, "furniture": 0, "paint": 0}
    assert m["none"] == none and m["again"] == none
    assert m["loaded"] == ["fingerprints", "imperfections", "smear"] and m["colourspace"] == ["Non-Color"]
    assert m["with"] == {"wall": 1, "tile": 1, "furniture": 1, "paint": 0}


# -- a built scene -------------------------------------------------------------------------------------------------
def test_wear_goes_on_the_structure_and_the_fixed_furniture_and_never_on_a_tracked_thing(found):
    """Contract (plan-realism sections 1 and 4, the catalog's colours): after `wear.apply` every wall, ceiling,
    apartment floor and fixed piece of furniture (the door and rugs included) has worn copies of its materials in
    its own mesh, and every carried item, piece of light furniture and mover has exactly the materials it had: the
    same names, the same node graphs and, for the items, the palette colour as base colour (sRGB 0.80 0.07 0.07
    for the red mug, worked out here). No material that existed before is modified, so a shared palette material
    (a red mug and a red fridge) cannot move. Regression caught: wear on a tracked item (the catalog's colour no
    longer what the attribute says), a shared material edited in place, or a surface left unworn."""
    worn = found["worn"]
    kinds = {"fixed": 0, "tracked": 0}
    for eid, e in worn["entities"].items():
        if e["mobility"] == "fixed" and e["kind"] != "staging_lane":
            kinds["fixed"] += 1
            assert all("+" in s["name"] for s in e["after"]), (eid, e["kind"])
            assert [s["name"] for s in e["after"]] != [s["name"] for s in e["before"]]
        elif e["mobility"] != "fixed":
            kinds["tracked"] += 1
            assert e["after"] == e["before"], (eid, e["kind"])
    assert kinds["fixed"] >= 20 and kinds["tracked"] >= 6  # the scene holds the apartment's furniture and the items
    for e in worn["entities"].values():
        if e["mobility"] == "carried" and (e["kind"], e["colour"]) == ("mug", "red"):
            want = [linear(c) for c in PALETTE_SRGB["red"]]
            assert e["after"][0]["base"][:3] == pytest.approx(want, abs=1e-4)
    assert set(worn["unchanged_materials"]) == set(worn["materials_before"])


def test_walls_ceilings_floors_and_rugs_get_the_finish_of_their_kind(found):
    """Contract (plan-realism section 4): the walls are plaster, the ceilings plaster too, the kitchen's and hall's
    floors tile, the living room's wood, the bedroom's carpet, and the rugs have their pattern. Regression caught: a
    floor given the wrong kind's finish or left plain, or a rug whose pattern never reaches its mesh."""
    worn = found["worn"]
    assert all(n.endswith("+wall") for n in worn["structure_after"]["walls"])
    ceilings = {k: v for k, v in worn["structure_after"].items() if k.startswith("ceiling_")}
    assert ceilings and all(n.endswith("+ceiling") for v in ceilings.values() for n in v)
    floors = {k: v[0] for k, v in worn["structure_after"].items() if k.startswith("floor_")}
    suffix = {name.rsplit("+", 1)[1] for name in floors.values()}
    assert suffix == {"tile", "wood", "carpet"}, floors
    rugs = [e for e in worn["entities"].values() if e["kind"] == "rug"]
    assert len(rugs) == 2 and all(s["name"].endswith("+rug") for e in rugs for s in e["after"])
