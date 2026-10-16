"""The asset library's gate and the command lines around it, without Blender: what a render is told, and when `assets`
mode refuses (no fallback to placeholders, VISION rule 10). The library is a stand-in file with the JSON a real build
writes (`test_render_assets_blender.py` builds a real one and runs it through the same gate)."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import asset_world as W
from builders import minimal_timeline
from stbench import asset_manifest as A
from stbench import asset_preview as P
from stbench import asset_render as AR
from stbench import blender_run, cli
from stbench import episode as ep
from stbench import render_cli as R
from stbench.catalog import CATALOGS
from stbench.mapspec import bundled_maps, load_map

BLENDER = "5.2.1"
RENDER = Path(__file__).resolve().parents[1] / "render"
RENDER_EPISODE = RENDER / "render_episode.py"


def fake_library(world: W.World, blend: Path, edit=None) -> Path:
    """A library file and the JSON a real build writes for `world`'s manifest, as the gate wants it."""
    m = A.load(world.manifest_path)
    blend.parent.mkdir(parents=True, exist_ok=True)
    blend.write_bytes(b"a blend file")
    doc = {"assets": {a.id: {"kind": a.kind, "bbox_m": list(a.native_bbox_m or (0.1, 0.1, 0.1)),
                             "slots": [{"slot": "body", "tinted": True, "mean_lum": 0.5}]}
                      for a in m.assets if a.role == "model" and not a.alternate_for},
           "materials": {a.id: {"mean_lum": 0.2} for a in m.assets if a.role == "material"}}
    if edit:
        edit(doc)
    AR.json_path(blend).write_text(json.dumps(doc), encoding="utf-8")
    return blend


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STBENCH_ASSET_CACHE", str(tmp_path / "root" / "cache"))
    monkeypatch.setattr(blender_run, "blender_version", lambda: BLENDER)
    return tmp_path / "root"


def gate(world: W.World, blend: Path) -> AR.LibraryInfo:
    return AR.require_library(world.manifest_path, blend, world.cache, BLENDER)


# -- what a render is told -----------------------------------------------------------------------------------------
def test_render_argv_carries_appearance_library_and_samples(tmp_path):
    """Contract: everything the render log records (appearance, samples, the library and the manifest it came from)
    reaches Blender through the command line, and every flag the CLI sends is one `render_episode.py` declares, and
    an exception in a Blender script makes Blender exit non-zero. Regression caught: a flag dropped or renamed between
    the CLI and Blender, so an `assets` render silently uses its defaults."""
    paths = ep.write_script(tmp_path / "ds", minimal_timeline())
    lib, man = tmp_path / "l.blend", tmp_path / "m.json"
    cmd = R.blender_command(paths, tmp_path / "t", (2, 5), fast=True, samples=48, appearance="assets", library=lib,
                            manifest=man)
    args = cmd[cmd.index("--") + 1:]
    for flag, value in (("--appearance", "assets"), ("--samples", "48"), ("--library", str(lib)),
                        ("--manifest", str(man)), ("--frames", "2:5")):
        assert args[args.index(flag) + 1] == value
    assert "--fast" in args and cmd[cmd.index("--python-exit-code") + 1] == "1"
    declared = {a.args[0].value for a in ast.walk(ast.parse(RENDER_EPISODE.read_text(encoding="utf-8")))
                if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "add_argument"}
    assert {a for a in args if a.startswith("--")} <= declared
    plain = R.blender_command(paths, tmp_path / "t", (0, 3))
    plain_args = plain[plain.index("--") + 1:]
    assert plain_args[plain_args.index("--appearance") + 1] == "placeholder"
    assert not {"--library", "--manifest", "--samples", "--fast"} & set(plain_args)


def test_library_path_is_named_by_the_manifest_and_the_blender_version(tmp_path):
    """Contract (plan section 3.3): the library file is `<cache>/library/<manifest sha12>-<blender version>.blend`, so
    a new manifest or a new Blender never reuses an old library. Regression caught: a fixed name that a stale library
    keeps."""
    m = W.build(tmp_path)
    manifest = A.load(m.manifest_path)
    path = AR.library_path(manifest, "5.2.1", tmp_path / "c")
    assert path == tmp_path / "c" / "library" / f"{manifest.sha256[:12]}-5.2.1.blend"
    assert AR.library_path(manifest, "5.3.0", tmp_path / "c") != path


# -- the gate ------------------------------------------------------------------------------------------------------
def test_the_gate_passes_a_cached_manifest_and_a_library_that_covers_it(tmp_path):
    """The positive case behind the refusals below: a manifest whose files are all cached and a library that holds every
    model and material at the recorded boxes is accepted and names the library and the manifest the render is told."""
    world = W.build(tmp_path)
    blend = fake_library(world, tmp_path / "lib" / "a.blend")
    info = gate(world, blend)
    assert info.blend == blend.resolve() and info.manifest_path == world.manifest_path.resolve()


def _uncache(path: str):
    """A good library over a cache that lacks one file of the manifest."""
    def make(world, blend):
        fake_library(world, blend)
        (world.cache / path).unlink()
        return blend
    return make


def _unpack_gone(world, blend):
    """A good library over a cache whose zip is there but whose unpacked folder is not."""
    fake_library(world, blend)
    shutil.rmtree(world.cache / "mat_a" / A.EXTRACT_DIR)
    return blend


def _no_box(raw):
    next(a for a in raw["assets"] if a["id"] == "cube_a")["native_bbox_m"] = None


def _no_tint_slots(raw):
    next(a for a in raw["assets"] if a["id"] == "chair_a")["tint"]["materials"] = None


REFUSALS = [
    ("no library file", lambda w, b: None, None, "no asset library"),
    ("a file missing from the cache", _uncache("cube_a/meshes/model.obj"), None, "cube_a meshes/model.obj: missing"),
    ("a zip not unpacked", _unpack_gone, None, "mat_a mat_a_1K-JPG.zip: not fully unpacked"),
    ("a model missing from the library", lambda w, b: fake_library(w, b, lambda d: d["assets"].pop("chair_a")), None,
     "chair_a: not in the library"),
    ("a box off by 2 mm", lambda w, b: fake_library(
        w, b, lambda d: d["assets"]["cube_a"].update(bbox_m=[0.202, 0.1, 0.1])), None, "more than 1 mm"),
    ("an unrecorded box", lambda w, b: fake_library(w, b), _no_box, "native_bbox_m is not recorded"),
    ("unnamed tint slots", lambda w, b: fake_library(w, b), _no_tint_slots, "tint.materials is null"),
]


@pytest.mark.parametrize("why, make, edit, message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_assets_mode_refuses_without_a_complete_cache_and_a_library_that_covers_the_manifest(
        why, make, edit, message, tmp_path, cache_env, monkeypatch):
    """Contract (VISION rule 10, plan section 3.3): `render --appearance assets` runs only when every file the manifest
    names is in the cache (a zip unpacked) and the library holds every model at the recorded box, with every box and
    tint slot recorded. Anything else is a refusal that names the problem and says there is no fallback; Blender is
    never started. Regression caught: a placeholder render labelled photoreal."""
    world = W.build(cache_env, edit)
    blend = make(world, cache_env / "lib" / "a.blend")
    paths = ep.write_script(tmp_path / "ds", minimal_timeline())
    monkeypatch.setenv("STBENCH_BLENDER", str(tmp_path / "no-such-blender.exe"))
    with pytest.raises(R.RenderError, match=message) as e:
        R.render(paths, appearance="assets", library=blend, manifest=world.manifest_path)
    assert "no fallback to placeholders" in str(e.value) and "STBENCH_BLENDER" not in str(e.value)
    assert not (paths.private / R.TMP_NAME).exists()


def test_a_good_library_lets_the_render_reach_blender(tmp_path, cache_env, monkeypatch):
    """The positive case of the refusal test: the same call with a good library gets past the gate and fails only
    because Blender is missing, naming STBENCH_BLENDER. Regression caught: a gate that refuses every library, which
    the refusals above would not notice."""
    world = W.build(cache_env)
    blend = fake_library(world, cache_env / "lib" / "a.blend")
    paths = ep.write_script(tmp_path / "ds", minimal_timeline())
    monkeypatch.setenv("STBENCH_BLENDER", str(tmp_path / "no-such-blender.exe"))
    with pytest.raises(R.RenderError, match="STBENCH_BLENDER"):
        R.render(paths, appearance="assets", library=blend, manifest=world.manifest_path)


def test_the_preview_gate_waits_for_no_recorded_box_or_tint_slot_but_a_wrong_box_still_stops_it(tmp_path):
    """Contract (plan section 2.4): the contact sheet is what a person looks at before recording each asset's box and
    tint slots in the manifest, so the preview's gate (`complete=False`) accepts an unrecorded box and unnamed tint
    slots, which a render refuses; a recorded box that disagrees with the library still stops both. Regression caught:
    a preview that cannot run until the manifest it is meant to help fill in is complete."""
    def unfinished(raw):
        _no_box(raw)
        _no_tint_slots(raw)
    world = W.build(tmp_path, unfinished)
    blend = fake_library(world, tmp_path / "lib" / "a.blend")
    with pytest.raises(AR.LibraryError, match="native_bbox_m is not recorded"):
        gate(world, blend)
    assert AR.require_library(world.manifest_path, blend, world.cache, BLENDER, complete=False).blend == blend.resolve()
    off = fake_library(world, tmp_path / "lib" / "b.blend", lambda d: d["assets"]["slab_b"].update(bbox_m=[1, 1, 1]))
    with pytest.raises(AR.LibraryError, match="slab_b.*1 mm"):
        AR.require_library(world.manifest_path, off, world.cache, BLENDER, complete=False)


def test_every_packaged_model_records_its_box_and_names_its_tint_slots():
    """Contract (plan section 2.4 step 5, VISION rule 10): the complete gate in front of every `--appearance assets`
    render needs each model's measured `native_bbox_m` and, for a tinted model, the slots that take the colour
    attribute. The packaged manifest is what a render reads, so it must hold both (an alternate is not built unless
    asked for). Regression caught: a model added or changed with its box or tint slots left null, which only the first
    render's gate would notice."""
    m = A.packaged()
    models = [a for a in m.assets if a.role == "model" and not a.alternate_for]
    assert models
    for a in models:
        assert a.native_bbox_m is not None, a.id
        if a.tint["mode"] in AR.TINTED:
            assert a.tint["materials"], a.id  # a non-empty list of slot names


def _module_literal(path: Path, name: str):
    """The literal assigned to a top-level `name` in a source file that cannot be imported here (it needs bpy)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no top-level {name} in {path}")


def test_procedural_builders_agree_with_the_manifest():
    """Contract (plan section 3.3): the manifest's procedural entries are built by the builder they name, with exactly
    the material slots that builder makes, and the box it is built to is the box the manifest records (the library build
    refuses a model whose mesh disagrees with its box, and the catalog test refuses a size that disagrees with it), and
    no builder is left that no manifest entry names (keys and phone were removed with their kinds). Regression caught:
    a builder renamed or its slots changed without the manifest (only a Blender run would fail), a design box that
    drifted from the recorded one, or dead builder code kept after its kind was dropped. The movers' builders (a robot
    vacuum, a home robot, a ball robot) are procedural entries like the rest, so they are held to the same."""
    slots, boxes = {}, {}
    for name in ("procedural.py", "procedural_carried.py", "procedural_toys.py"):  # `procedural.py` merges the others
        slots |= _module_literal(RENDER / name, "SLOTS")
        boxes |= _module_literal(RENDER / name, "DESIGN_BOX")
    procedural = [a for a in A.packaged().assets if a.role == "model" and a.source == A.PROCEDURAL]
    assert {a.builder for a in procedural} == set(slots) == set(boxes)
    for a in procedural:
        assert set(a.materials) == set(slots[a.builder]), a.id
        assert a.native_bbox_m is not None and all(
            abs(have - want) <= 0.001 for have, want in zip(a.native_bbox_m, boxes[a.builder], strict=True)), a.id


def _drawn(*files: str) -> set[str]:
    """The kinds a render/ module has a drawing for: its functions' names (a trailing `_` stripped, as `box_` is `box`),
    read from the source because the module imports bpy."""
    return {n.name.rstrip("_") for f in files for n in ast.parse((RENDER / f).read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef)}


def test_the_placeholder_appearance_draws_every_catalog_and_mover_kind():
    """Contract (plan-t3 section 5, plan-warehouse section 7.3): `--appearance placeholder` renders every kind a
    timeline can hold, so `render/items.py` has a drawing per carried and mover kind of every catalog (a function named
    after the kind, `box_` for the box). Regression caught: a kind added to a catalog with no placeholder drawing, which
    `build_scene` meets only as a ValueError in the middle of the first placeholder render (the warehouse's pallets,
    totes and forklifts among them)."""
    items = _drawn("items.py")
    for name, catalog in CATALOGS.items():
        wanted = {*catalog.kinds, *catalog.mover_kinds}
        assert wanted <= items, (name, sorted(wanted - items))
    assert {"pallet", "tote", "drum", "sack", "bin", "forklift", "tote_robot"} <= items


def test_the_placeholder_appearance_draws_every_piece_of_furniture_of_every_map():
    """Contract (plan-warehouse section 7.3, plan-city section 7.3): a `render/furniture*.py` (read as a glob, so a
    lane's new file is read when it lands) has a drawing for each fixed and light kind of each bundled map (`door` is
    built by `build_scene` itself). Regression caught: a kind on a map with no placeholder drawing (the warehouse's
    racks, lanes, tables, shelving, dock doors, pallet jack and roll cage; the downtown's sidewalks, parking lanes,
    loading zones and paving), met only as a ValueError at the first placeholder render of that map; or a furniture
    file named here by hand and forgotten."""
    furniture = _drawn(*sorted(p.name for p in RENDER.glob("furniture*.py")))
    for name in bundled_maps():
        wanted = {f.kind for f in load_map(name).furniture}
        assert wanted <= furniture, (name, sorted(wanted - furniture))
    assert {"pallet_rack", "staging_lane", "packing_table", "shelving", "dock_door", "pallet_jack",
            "roll_cage", "sidewalk", "parking_lane", "loading_zone", "plaza"} <= furniture


def _builder_names(*files: str) -> set[str]:
    """The keys of the dict literal each file assigns to a top-level `BUILDERS`."""
    out: set[str] = set()
    for f in files:
        for node in ast.parse((RENDER / f).read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "BUILDERS" for t in node.targets) \
                    and isinstance(node.value, ast.Dict):
                out |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return out


def test_every_procedural_manifest_entry_has_a_builder_function_and_no_builder_is_left_over():
    """Contract (plan-warehouse section 3.3, plan-city section 7.3): a procedural model is built by the function its
    `builder` names, in `procedural_home.py` and `procedural.py` (the household kinds), in `procedural_goods.py` /
    `procedural_equipment.py` (the warehouse's) or in `procedural_vehicles.py` / `procedural_city.py` (the downtown's),
    or in `procedural_carried.py` / `procedural_toys.py` (the household's rebuilt bag, box, shoes, stool and toys),
    which `procedural.py` registers, so the builders the render/ modules define are exactly the builders the packaged
    manifest names. Read from the source because the modules import bpy. Regression caught: a manifest entry whose
    builder was never written (the library build fails only when Blender runs, KeyError; the downtown's ten were stubs
    until they were), or a builder written for a kind that has no entry (dead code, as keys and phone once were)."""
    named = {a.builder for a in A.packaged().assets if a.role == "model" and a.source == A.PROCEDURAL}
    defined = _builder_names("procedural_home.py", "procedural.py", "procedural_goods.py", "procedural_equipment.py",
                             "procedural_vehicles.py", "procedural_city.py", "procedural_carried.py",
                             "procedural_toys.py")
    assert named == defined, (sorted(named - defined), sorted(defined - named))


def test_every_shader_the_manifest_names_is_one_materials_builds():
    """Contract (plan-warehouse section 3.3): a `shader:<name>` in the manifest, as a model's slot material or as a
    floor (the warehouse's `shader:concrete`), is one of the procedural shaders `materials.SHADERS` lists, which
    `shader_material` builds. Regression caught: a typo'd or removed shader that fails only inside Blender, at the first
    render or library build that meets it."""
    shaders = set(_module_literal(RENDER / "materials.py", "SHADERS"))
    m = A.packaged()
    named = {ref.split(":", 1)[1] for a in m.assets for ref in (a.materials or {}).values()
             if ref.startswith("shader:")}
    named |= {ref.split(":", 1)[1] for ref in m.floors.values() if ref.startswith("shader:")}
    assert "concrete" in named and named <= shaders, sorted(named - shaders)


def test_every_region_of_every_bundled_map_has_a_placeholder_floor_and_the_warehouses_is_concrete():
    """Contract (plan-warehouse section 2; M3-maps §1): the look modules' `PLACEHOLDER_FLOOR`
    (`render/look_<catalog>.py`) names a floor for the kind of each region of each bundled map (a kind it does not name
    raises), and the warehouse's two region kinds, the dock and the aisles (R2-R4 are of kind "aisle"), are `concrete`,
    the same shader the assets appearance builds from the manifest's `shader:concrete`. Regression caught: a warehouse
    region left on tiles in the placeholder render, or the apartment's floors changed."""
    floors = {}
    for catalog in CATALOGS:
        floors |= _module_literal(RENDER / f"look_{catalog}.py", "PLACEHOLDER_FLOOR")
    for name in bundled_maps():
        assert {r.kind for r in load_map(name).regions} <= set(floors), name
    assert {floors[r] for r in ("dock", "aisle")} == {"concrete"}
    assert floors["kitchen"] == "tile" and floors["bedroom"] == "carpet"
    assert {floors[r] for r in ("high street", "west street", "east street")} == {"asphalt"}
    assert floors["square"] == "paving"


def test_the_ceiling_light_constant_is_gone_and_the_maps_say_how_bright_their_lights_are():
    """Contract (plan-warehouse section 1): a ceiling light's power is the map's `watts` (absent in a map file means the
    apartment's 45 W, `mapspec.DEFAULT_LIGHT_WATTS`), so `build_scene.py` keeps no constant of its own to disagree with
    it, and the apartment's lights are the 45 W its renders were calibrated at (the Blender test reads what the scene
    holds). Regression caught: the old constant back in `build_scene` (a warehouse lit like a flat), or the apartment's
    lights given another power."""
    source = (RENDER / "build_scene.py").read_text(encoding="utf-8")
    assert "CEILING_LIGHT_WATTS" not in source and "lt.watts" in source
    assert {c.watts for c in load_map("apartment_a").ceiling_lights} == {45.0}
    assert min(c.watts for c in load_map("warehouse_a").ceiling_lights) > 45.0


def _rack_layout():
    """`render/rack_layout.py`, which has no bpy, loaded from its file (render/ is not a package)."""
    spec = importlib.util.spec_from_file_location("rack_layout", RENDER / "rack_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stock_fills_exactly_the_bay_levels_a_rack_has_no_surface_for():
    """Contract (plan-warehouse section 1): a pallet rack row has 4 bays of 2.7 m and 3 levels (z 0.0, 1.3, 2.6); the
    map gives a surface to half of the 12 bay-levels and none to the rest, and the rack builder draws static stock in
    exactly the rest. By hand: F13's surfaces are S13 (bay 0, level 0), S14 (0, 2), S15 (1, 1), S16 (2, 0), S17 (2, 2)
    and S18 (3, 1), so its stock is bays 0..3 by levels 0..2 minus those six. The six racks together leave 36 of 72
    bay-levels to stock. A surface off the bay grid is an error. Regression caught: stock drawn where an item may stand
    (an item sinking into cartons), a bay-level left empty that the map promises stock in, or a misaligned map
    accepted."""
    layout = _rack_layout()
    wh = load_map("warehouse_a")
    racks = [f for f in wh.furniture if f.kind == "pallet_rack"]
    f13 = next(f for f in racks if f.id == "F13")
    surfaces = [(*s.rect, s.z) for s in f13.surfaces]
    assert layout.layout(f13.size, surfaces) == (4, 3, {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1)})
    assert layout.stock(f13.size, surfaces) == [(0, 1), (1, 0), (1, 2), (2, 1), (3, 0), (3, 2)]
    assert len(layout.stock(f13.size, [])) == 12
    assert sum(len(layout.stock(f.size, [(*s.rect, s.z) for s in f.surfaces])) for f in racks) == 36
    with pytest.raises(ValueError, match="not a bay-level"):
        layout.stock(f13.size, [(0.0, -3.0, 1.08, 2.6, 1.3)])
    with pytest.raises(ValueError, match="not a bay-level"):
        layout.stock(f13.size, [(0.0, -4.05, 1.08, 2.6, 1.0)])


# -- the commands --------------------------------------------------------------------------------------------------
def test_library_and_preview_commands_refuse_an_empty_cache_without_starting_blender(tmp_path, cache_env,
                                                                                    monkeypatch, capsys):
    """Contract (nothing is fetched without `--yes`): `stbench assets library` and `assets preview` are
    registered, and with the packaged manifest and an empty asset cache they exit 2 saying what is missing, rather than
    starting Blender on files that are not there. Regression caught: a build over absent downloads."""
    monkeypatch.setenv("STBENCH_BLENDER", str(tmp_path / "no-such-blender.exe"))
    assert cli.main(["assets", "library"]) == 2
    assert "every file is cached" in capsys.readouterr().err
    assert cli.main(["assets", "preview", "--out", str(tmp_path / "sheets")]) == 2
    assert "cannot be used" in capsys.readouterr().err


# -- the contact sheets --------------------------------------------------------------------------------------------
def test_preview_covers_every_kind_of_every_catalog_in_every_colour_and_the_light_furniture_of_every_map():
    """Contract (plan section 2.4, step 4; plan-t3 section 5; plan-warehouse section 7.3): the contact sheet shows every
    carried and mover kind of every catalog in every colour it is generated in, at the size it is generated at (a kind
    two catalogs share, the box, once, in the colours of both), and the light furniture each bundled map has in the
    colours it has it (the apartment's chairs beige and black, the stool brown, the laundry basket beige; the
    warehouse's pallet jacks red and yellow and its roll cage grey), each with the asset the manifest names, plus every
    textured surface. Regression caught: a kind or colour missing from the sheet that is reviewed before any episode is
    rendered (a warehouse pallet colour or a forklift whose look is judged on no sheet, the rule 10 check)."""
    m = A.packaged()
    jobs = P.preview_jobs(m)
    items = [i for i in jobs["items"] if m.asset(i["asset"]).role == "model"]  # clutter models are sheets of their own
    by_kind: dict[str, dict] = {}
    for i in items:  # a kind with several models (one a colour) has a sheet per model: its colours are theirs together
        have = by_kind.setdefault(i["kind"], {"colours": [], "size": i["size"]})
        have["colours"] += i["colours"]
        assert i["size"] == have["size"], i["kind"]
    assert len(set(by_kind)) == len(by_kind) and len({(i["kind"], c) for i in items for c in i["colours"]}) == sum(
        len(i["colours"]) for i in items)
    for catalog in CATALOGS.values():
        for kind in (*catalog.kinds.values(), *catalog.mover_kinds.values()):
            assert set(kind.colors) <= set(by_kind[kind.name]["colours"]), kind.name
            assert tuple(by_kind[kind.name]["size"]) == kind.size, kind.name
    assert {"pallet", "tote", "drum", "sack", "bin", "forklift", "tote_robot"} <= set(by_kind)
    assert {i["kind"] for i in P.preview_jobs(m, ["ball_robot"])["items"]} == {"ball_robot"}
    assert {k: sorted(by_kind[k]["colours"]) for k in ("chair", "stool", "laundry_basket", "pallet_jack",
                                                       "roll_cage")} == {
        "chair": ["beige", "black"], "stool": ["brown"], "laundry_basket": ["beige"], "pallet_jack": ["red", "yellow"],
        "roll_cage": ["grey"]}
    assert all(i["asset"] == m.model_for(i["kind"], c).id for i in items for c in i["colours"])
    assert {s["id"] for s in jobs["surfaces"]} == {a.id for a in m.assets if a.role == "material"}
    assert [i["kind"] for i in P.preview_jobs(m, ["mug"])["items"]] == ["mug"]
    with pytest.raises(A.AssetError, match="no such kind"):
        P.preview_jobs(m, ["spoon"])


def test_a_sheet_has_a_column_per_colour_and_a_row_per_view(tmp_path):
    """Contract: `compose` lays a kind's tiles out as one column per catalog colour and `VIEWS` rows under a title and
    a row of colour labels, and writes the index that names them. Regression caught: tiles pasted in the wrong
    order or a sheet cut short, so a colour is judged from the wrong picture."""
    tile = (24, 18)
    jobs = {"items": [{"kind": "mug", "asset": "cube_a", "colours": ["red", "blue", "green"], "size": [0.1] * 3}],
            "surfaces": [{"id": "mat_a", "tile_m": None}]}
    for colour in jobs["items"][0]["colours"]:
        for v in range(P.VIEWS):
            (tmp_path / "tiles" / "mug").mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((18, 24, 3), 40 * v + 10, np.uint8)).save(tmp_path / "tiles" / "mug" /
                                                                             f"{colour}__{v}.png")
    (tmp_path / "tiles" / "_surfaces").mkdir()
    Image.fromarray(np.full((18, 24, 3), 99, np.uint8)).save(tmp_path / "tiles" / "_surfaces" / "mat_a.png")
    written = P.compose(jobs, tmp_path / "tiles", tmp_path, tile)
    assert [p.name for p in written] == ["mug.png", "surfaces.png", "index.json"]
    with Image.open(tmp_path / "mug.png") as sheet:
        px = np.asarray(sheet.convert("RGB"))
    assert px.shape[:2] == (2 * P.LABEL_H + P.VIEWS * 18, 3 * 24)
    for v in range(P.VIEWS):  # row v of the sheet is view v, in every column
        assert (px[2 * P.LABEL_H + v * 18 + 9, [12, 36, 60]] == 40 * v + 10).all()
    assert json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["kinds"]["mug"]["colours"] == [
        "red", "blue", "green"]
