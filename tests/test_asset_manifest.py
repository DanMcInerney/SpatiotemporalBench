"""The asset manifest: the packaged one is valid and complete, licences and hosts are policed in code, and the cache
never lands in git or a dataset. Synthetic manifests come from `asset_synth`; nothing here touches the network."""

from __future__ import annotations

import copy
import hashlib

import pytest

from asset_synth import CCBY, asset, manifest, raw_manifest, write_manifest
from stbench import asset_manifest as A
from stbench.catalog import CATALOGS
from stbench.mapspec import load_map

KINDS_UNDER_TEST, LIGHT_UNDER_TEST = ("mug", "glass"), ()


def problems(edit=None, **kinds) -> list[str]:
    return A.validate(manifest(edit=edit), **kinds)


def as_gltf(raw: dict, up: str) -> None:
    """Turn mug_a into a glTF asset whose transform says `up`."""
    mug = asset(raw, "mug_a")
    mug["format"] = "gltf"
    mug["transform"]["up"] = up


def test_the_synthetic_manifest_is_valid():
    """Contract: the synthetic manifest the mutation tests below start from validates with no problem, so each failure
    they see comes from its one change. Regression caught: a base that is already broken (a fixture edit), which would
    let a mutation test pass on a problem it did not cause."""
    assert problems(catalog_kinds=KINDS_UNDER_TEST, light_kinds=LIGHT_UNDER_TEST) == []
    assert A.validate(manifest()) == []


def test_packaged_manifest_validates_and_covers_every_queried_kind_once():
    """Contract: every catalog kind of every catalog, every light-furniture kind of every bundled map and every mover
    kind has a model of its own mobility for each colour a description of it can have (one model for a kind that has
    one, one a colour for a kind that has several, `tests/test_asset_colours.py`), every material reference resolves
    and every licence is allowed. Regression caught: a kind silently left on placeholder geometry, a colour with two
    models (the entity would get either) or none, or a mover kind whose model is missing so a T3 episode cannot be
    rendered with assets."""
    carried, light, movers = A.queried_kinds()
    colours, m = A.queried_colours(), A.load()
    assert A.validate(m, carried, light, movers, colours=colours) == []
    kinds = [*carried, *light, *movers]
    assert {m.model_for(k, c).kind for k in kinds for c in colours[k]} == set(kinds)
    assert {m.model_for(k, c).mobility for k in movers for c in colours[k]} == {"mover"}
    assert {m.model_for(k, c).mobility for k in light for c in colours[k]} == {"light"}
    assert {a.licence for a in m.assets} <= A.LICENCES | A.CODE_LICENCES


def test_the_manifest_covers_every_warehouse_kind_exactly_once_and_the_shared_box_once():
    """Contract (plan-warehouse §3, §7.3): the five carried kinds the warehouse adds (pallet, tote, drum, sack, bin),
    its two pieces of light equipment (pallet jack, roll cage) and its two movers (forklift, tote robot) each have one
    procedural model of the right mobility, built to the catalog's box (fit `exact`) and tinted by the colour
    attribute; the box, which both catalogs hold, has the same two models in both (a carton in brown and a lidded
    storage box in the warehouse's four colours), not a third for the warehouse. Hand count: 5 + 2 + 2 = 9 new
    models. Regression caught: a warehouse kind with no model (its first assets render
    fails), two models for the shared box (the entity would get either), or a model filed under the wrong mobility."""
    m = A.packaged()
    wanted = {"carried": ["pallet", "tote", "drum", "sack", "bin"], "light": ["pallet_jack", "roll_cage"],
              "mover": ["forklift", "tote_robot"]}
    for mobility, kinds in wanted.items():
        for kind in kinds:
            found = [a for a in m.assets if a.role == "model" and a.kind == kind]
            assert [(a.mobility, a.source, a.fit["mode"]) for a in found] == [(mobility, A.PROCEDURAL, "exact")], kind
            assert found[0].tint["mode"] == "luminance" and found[0].tint["materials"], kind
    assert sum(a.source == A.PROCEDURAL and a.kind in sum(wanted.values(), []) for a in m.assets) == 9
    assert [a.id for a in m.models_of("box")] == ["proc_carton", "proc_storage_box"]
    assert {c: m.model_for("box", c).id for c in CATALOGS["warehouse"].kinds["box"].colors} == {
        "blue": "proc_storage_box", "red": "proc_storage_box", "green": "proc_storage_box",
        "white": "proc_storage_box"}
    carried, light, movers = A.queried_kinds()
    assert carried.count("box") == 1 and set(wanted["carried"]) <= set(carried) and set(wanted["mover"]) <= set(movers)
    assert set(wanted["light"]) <= set(light)


DOWNTOWN = {"carried": {"car": ("body",), "van": ("body",), "pickup": ("body",), "box_truck": ("body",),
                        "skip": ("body",), "dumpster": ("body",), "portable_toilet": ("body",),
                        "food_cart": ("umbrella", "umbrella2")}, "mover": {"taxi": ("body",), "bus": ("body",)}}


def test_the_manifest_covers_every_downtown_kind_with_a_procedural_model_tinted_where_the_plan_says():
    """Contract (plan-city sections 3.1, 3.5 and 7.3): the downtown's eight carried kinds and two movers each have one
    procedural model of the right mobility, built to the catalog's box (fit `exact`), and the colour attribute tints the
    slot a person would name: the body (the toilet's walls, the paint of a vehicle, skip and dumpster) and, for the food
    cart, its umbrella, whose second set of gores is a lighter tint of it; every other slot is a fixed material, so that
    colour recognition rests on one part. Hand-listed from the plan. Regression caught: a kind with no model (its first
    assets render fails), a model tinted on its glass or tyres (every colour reads dark), or the cart's umbrella left
    untinted."""
    m = A.packaged()
    for mobility, kinds in DOWNTOWN.items():
        for kind, slots in kinds.items():
            (found,) = [a for a in m.assets if a.role == "model" and a.kind == kind]
            assert (found.mobility, found.source, found.fit["mode"]) == (mobility, A.PROCEDURAL, "exact"), kind
            assert tuple(found.tint["materials"]) == slots and found.tint["mode"] == "luminance", kind
            assert set(found.tint["materials"]) <= set(found.materials), kind
            assert all(ref.startswith("shader:") for ref in found.materials.values()), kind  # nothing to download
    cart = m.model_for("food_cart")
    assert cart.tint["lighten"] == {"umbrella2": 0.5}
    downtown = {k for kinds in DOWNTOWN.values() for k in kinds}
    assert all("lighten" not in a.tint for a in m.assets if a.role == "model" and a.kind in downtown - {"food_cart"})


def test_the_manifest_names_the_downtown_floors():
    """Contract (plan-city section 7.3): the three streets are procedural asphalt and the square procedural paving
    (`shader:` floors: no download), which `validate` accepts. Regression caught: a street left on the fallback tiles in
    an assets render, or the square drawn as asphalt."""
    m = A.packaged()
    assert {m.floors[r] for r in ("high street", "west street", "east street")} == {"shader:asphalt"}
    assert m.floors["square"] == "shader:paving"
    assert {r.name for r in load_map("downtown_a").regions} <= set(m.floors)


@pytest.mark.parametrize("lighten, ok", [
    pytest.param({"cloth": 0.5}, True, id="a tinted slot, half way"),
    pytest.param({"cloth": 1.5}, False, id="more than white"),
    pytest.param({"cloth": -0.1}, False, id="darker"),
    pytest.param({"wicker": 0.5}, False, id="a slot that is not tinted"),
    pytest.param([0.5], False, id="not a mapping"),
])
def test_tint_lighten_names_a_tinted_slot_and_a_fraction_of_the_way_to_white(lighten, ok):
    """Contract: `tint.lighten` (optional) maps slots of `tint.materials` to the fraction of the way to white they are
    shown at (the food cart's second set of umbrella gores); anything else is a manifest problem. Regression caught: a
    lightened slot the colour attribute never reaches, or a fraction outside 0 to 1 that the renderer would clip
    silently."""
    def edit(raw):
        asset(raw, "proc_glass")["tint"] = {"mode": "luminance", "materials": ["cloth"], "lighten": lighten}
    flagged = [p for p in problems(edit) if "lighten" in p]
    assert (flagged == []) is ok, flagged


def test_every_region_of_every_bundled_map_has_a_floor_and_the_warehouse_floor_is_concrete():
    """Contract (plan-warehouse §2, `build_scene` FLOORS; M3-maps §1): the manifest's `floors` names a floor for the
    kind of each region of each bundled map (a kind it does not name falls back to checker tiles in an assets render),
    and the warehouse's two region kinds, the dock and the aisles (R2-R4 are of kind "aisle"), are procedural concrete
    (`shader:concrete`: no download), which `validate` accepts without a material asset. Regression caught: a warehouse
    region left to the fallback tiles, or a floor shader the validator rejects."""
    m = A.packaged()
    for name in ("apartment_a", "warehouse_a"):
        assert {r.kind for r in load_map(name).regions} <= set(m.floors), name
    assert {m.floors[r] for r in ("dock", "aisle")} == {"shader:concrete"}
    assert m.floors["kitchen"] == "acg_Tiles139"  # the apartment's own are unchanged
    assert A.validate(m) == []


def test_a_floor_may_be_a_shader_but_a_missing_material_is_still_reported():
    """Contract: a floor entry is a material asset id or `shader:<name>`; a walls or palette entry is still a material
    asset id. Regression caught: the shader allowance widened to every reference (a typo'd wall material passes), or a
    missing floor material accepted."""
    assert problems(lambda r: r["floors"].update(hall="shader:concrete")) == []
    assert any("floors: hall refers to 'nope'" in p for p in problems(lambda r: r["floors"].update(hall="nope")))
    assert any("walls: walls refers to 'shader:concrete'" in p
               for p in problems(lambda r: r.update(walls="shader:concrete")))


def test_mover_sizes_follow_the_boxes_of_their_assets():
    """Contract (plan-t3 section 1.1, V0 section 4): `MoverKind.size` is the truth that footprints, collisions and the
    in-view cover use, and the renderer fits the mover's mesh to it, so it must describe the mesh. A mover is
    procedural, built to its catalog box (fit `exact`), whose box the manifest records as `native_bbox_m` and the
    library build holds to 1 mm; it takes the colour attribute (`luminance` tint of its body or shell), because its
    description is colour then kind. This holds for the movers of every catalog (the household's three, the warehouse's
    forklift and tote robot). Regression caught: a mover size changed in the catalog without the builder's design
    box and the manifest (the robot would be stretched, or its footprint would lie), or a mover with no tinted slot,
    which renders every colour the same."""
    for catalog in CATALOGS.values():
        for kind in catalog.mover_kinds.values():
            asset = A.packaged().model_for(kind.name)
            assert (asset.source, asset.mobility, asset.fit["mode"]) == (A.PROCEDURAL, "mover", "exact"), kind.name
            assert asset.native_bbox_m is not None and all(
                abs(want - have) <= 0.001 for want, have in zip(kind.size, asset.native_bbox_m, strict=True)), (
                kind.name, kind.size, asset.native_bbox_m)
            assert asset.tint["mode"] == "luminance" and set(asset.tint["materials"]) <= set(asset.materials), kind.name
            assert asset.tint["materials"], kind.name


def add_mover(raw: dict, kind: str = "rover", asset_id: str = "proc_rover", **extra) -> None:
    """A second procedural model in the synthetic manifest, a copy of the glass with another kind and mobility."""
    raw["assets"].append({**copy.deepcopy(asset(raw, "proc_glass")), "id": asset_id, "kind": kind, "mobility": "mover",
                          **extra})


def test_a_mover_kind_needs_exactly_one_mover_model():
    """Contract (plan-t3 section 5): `validate(..., mover_kinds=)` holds a mover kind to exactly one model whose
    mobility is `mover`, as carried and light kinds are held to theirs; `mover` is a mobility a model may have, and a
    mover model nobody asked about is reported. Regression caught: a mover kind left without a model, two models for one
    kind (the entity would get either), a mover model filed as carried, or mover models accepted when no mover kind is
    queried."""
    assert "mover" in A.MOBILITIES
    kinds = {"catalog_kinds": KINDS_UNDER_TEST, "light_kinds": LIGHT_UNDER_TEST}
    assert problems(add_mover, mover_kinds=("rover",), **kinds) == []
    none = problems(mover_kinds=("rover",), **kinds)
    assert any("kind rover: needs exactly one mover model, found []" in p for p in none)
    two = problems(lambda r: (add_mover(r), add_mover(r, asset_id="proc_rover_2")), mover_kinds=("rover",), **kinds)
    assert any("found ['proc_rover', 'proc_rover_2']" in p for p in two)
    assert any("needs exactly one mover model, found ['proc_rover']" in p
               for p in problems(lambda r: add_mover(r, mobility="carried"), mover_kinds=("rover",), **kinds))
    assert any("proc_rover: kind 'rover' is not a queried kind" in p for p in problems(add_mover, **kinds))
    assert manifest(edit=add_mover).model_for("rover").id == "proc_rover"


# -- licences (VISION rule 11) -------------------------------------------------------------------------------------
@pytest.mark.parametrize("edit, expected", [
    pytest.param(lambda r: r["sources"]["ccby"].update(licence="CC-BY-NC-4.0"), "not on the allowlist",
                 id="a non-commercial source"),
    pytest.param(lambda r: asset(r, "mug_a").update(licence="CC-BY-NC-4.0"), "not on the allowlist",
                 id="a non-commercial asset"),
    pytest.param(lambda r: r["sources"]["cc0"].update(licence="GPL-3.0-or-later"), "not on the allowlist",
                 id="GPL outside the code source"),
    pytest.param(lambda r: r["sources"]["procedural"].update(licence="CC-BY-NC-4.0"), "not on the allowlist",
                 id="a non-commercial procedural source"),
    pytest.param(lambda r: asset(r, "mat_a").update(licence="CC-BY-4.0"), "differs from its source's",
                 id="an asset licence that is not its source's"),
    pytest.param(lambda r: r["sources"]["ccby"].pop("attribution"), "attribution", id="CC BY without a credit line"),
    pytest.param(lambda r: asset(r, "mug_a").update(changes=None), "what we changed", id="CC BY without a change note"),
    pytest.param(lambda r: asset(r, "mug_a").update(title=""), "title", id="CC BY without a title"),
    pytest.param(lambda r: r["sources"]["cc0"].update(licence_url=None), "licence_url", id="no licence page"),
])
def test_only_cc0_ccby_bsd_licences_and_ccby_carries_attribution(edit, expected):
    """Contract (VISION rule 11): assets are CC0, CC BY or BSD only (plus GPL for our own code), the asset's licence is
    its source's, and CC BY ships with a credit line, a title and an indication of changes. Regression caught: a
    CC BY-NC or unlicensed asset entering the manifest, or CC BY terms dropped."""
    found = problems(edit)
    assert any(expected in p for p in found), found


# -- hosts, references, fit -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("edit, expected", [
    pytest.param(lambda r: asset(r, "mug_a")["files"][0].update(url="https://evil.example.net/m.obj"), "hosts",
                 id="a host off the source's list"),
    pytest.param(lambda r: asset(r, "mug_a")["files"][0].update(url=f"http://{CCBY}/m.obj"), "https",
                 id="plain http"),
    pytest.param(lambda r: asset(r, "mug_a")["fit"].update(max_anisotropy=0.3), "max_anisotropy",
                 id="a squash over 0.25"),
    pytest.param(lambda r: asset(r, "proc_glass")["fit"].update(mode="uniform"), "exact", id="a procedural fit"),
    pytest.param(lambda r: r["palette_materials"].update(oak="nope"), "not a material", id="an unknown material"),
    pytest.param(lambda r: r["floors"].update(hall="mug_a"), "not a material", id="a model as a floor"),
    pytest.param(lambda r: asset(r, "proc_glass")["materials"].update(base="nope"), "not a material",
                 id="a procedural slot with no material"),
    pytest.param(lambda r: asset(r, "mug_a").update(source="nowhere"), "unknown source", id="an unknown source"),
    pytest.param(lambda r: r["assets"].append({**copy.deepcopy(asset(r, "mat_a")), "id": "MAT_A"}), "duplicate id",
                 id="an id repeated with other letter case"),
    pytest.param(lambda r: asset(r, "mug_a").update(entry="missing.obj"), "entry", id="an entry that is not a file"),
    pytest.param(lambda r: asset(r, "mug_a").update(files=[]), "needs files", id="a download with no files"),
    pytest.param(lambda r: asset(r, "proc_glass").pop("builder"), "builder", id="procedural with no builder"),
    pytest.param(lambda r: asset(r, "mug_b").update(alternate_for="nope"), "alternate_for",
                 id="an alternate of nothing"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="mug a"), "letters, digits and underscores",
                 id="an id with a space"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="mug/a"), "letters, digits and underscores",
                 id="an id that is two folders"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="mug.a"), "letters, digits and underscores",
                 id="an id with a dot"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="m\u00fcg"), "letters, digits and underscores",
                 id="an id with a non-ASCII letter"),
    pytest.param(lambda r: asset(r, "mug_a")["transform"].update(up="X"), "transform.up",
                 id="an OBJ with an up axis that is neither Y nor Z"),
    pytest.param(lambda r: as_gltf(r, "Y"), "transform.up", id="a glTF whose up is the spec's Y, not the result Z"),
    pytest.param(lambda r: asset(r, "mug_a").update(transform=["Z"]), "a transform", id="a transform that is a list"),
])
def test_validate_reports_each_host_reference_and_fit_problem(edit, expected):
    """Contract: `validate` names each integrity problem, so a manifest edit that breaks the host allowlist, the
    material references, the id's character set, the up axis or the fit limit cannot reach a download or a render.
    Regression caught: a check dropped, so a bad host, a dangling material or an id that is not a plain folder name
    passes unseen."""
    found = problems(edit)
    assert any(expected in p for p in found), found


def test_transform_up_is_the_axis_the_importer_is_told_and_is_held_to_the_format():
    """Contract (module docstring): an OBJ names the up axis of its file, Y or Z; a glTF is always Z, the result after
    Blender's importer converts the spec's Y up. The valid combinations pass, so the refusals above are not vacuous.
    Regression caught: a rule that rejects an OBJ authored Y up, or accepts a glTF entry that disagrees with the
    importer the library build uses."""
    assert problems(lambda r: asset(r, "mug_a")["transform"].update(up="Y")) == []
    assert problems(lambda r: as_gltf(r, "Z")) == []


def test_a_kind_needs_exactly_one_model_and_an_alternate_does_not_count():
    """Contract: one asset per kind. The alternate for mug_a shares its kind but is never the model; dropping its
    `alternate_for` is a second model, and a kind with none, or the wrong mobility, is reported. Regression caught:
    two assets for one kind, or a kind left on placeholders."""
    kinds = {"catalog_kinds": KINDS_UNDER_TEST, "light_kinds": LIGHT_UNDER_TEST}
    assert manifest().model_for("mug").id == "mug_a"
    two = problems(lambda r: asset(r, "mug_b").pop("alternate_for"), **kinds)
    assert any("found ['mug_a', 'mug_b']" in p for p in two)
    assert any("found []" in p for p in problems(lambda r: r["assets"].remove(asset(r, "proc_glass")), **kinds))
    assert any("light model" in p for p in A.validate(manifest(), ("mug",), ("glass",)))
    assert any("not a queried kind" in p for p in A.validate(manifest(), ("mug",), ()))
    with pytest.raises(A.AssetError, match="2 models"):
        manifest(edit=lambda r: asset(r, "mug_b").pop("alternate_for")).model_for("mug")


# -- loading ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("edit, expected", [
    pytest.param(lambda r: asset(r, "mug_a")["files"][0].update(sha256="0" * 64), "unknown field",
                 id="a pin field the manifest no longer has"),
    pytest.param(lambda r: asset(r, "mug_a").update(colourway="red"), "unknown field", id="an unknown asset field"),
    pytest.param(lambda r: r.update(extra=1), "unknown top-level", id="an unknown top-level field"),
    pytest.param(lambda r: asset(r, "mug_a").pop("role"), "missing field", id="a missing required field"),
    pytest.param(lambda r: asset(r, "mug_a")["files"][0].update(path="../up/model.obj"), "relative",
                 id="a traversal path"),
    pytest.param(lambda r: asset(r, "mug_a")["files"][0].update(path="C:/model.obj"), "relative", id="a drive path"),
    pytest.param(lambda r: asset(r, "mug_a")["files"][0].update(path="a\\b.obj"), "relative", id="a backslash path"),
    pytest.param(lambda r: asset(r, "mug_a")["files"][0].update(path="/abs.obj"), "relative", id="an absolute path"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="../x"), "relative", id="an id that climbs out of the cache"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="/abs"), "relative", id="an absolute id"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="a\\b"), "relative", id="an id with a backslash"),
    pytest.param(lambda r: asset(r, "mug_a").update(id="C:x"), "relative", id="an id with a drive"),
    pytest.param(lambda r: asset(r, "mug_a").update(id=7), "relative", id="an id that is not text"),
])
def test_load_refuses_unknown_fields_missing_fields_and_unsafe_paths(edit, expected):
    """Contract: the loader is strict, so a typo or a field of an older manifest cannot be silently ignored, and
    neither a file path nor an asset id (it names the cache folder) can leave the cache, even for a caller that loads
    without `validate`. Regression caught: a misspelt field ignored, or `../` in a path or an id writing outside the
    cache."""
    raw = copy.deepcopy(raw_manifest())
    edit(raw)
    with pytest.raises(A.AssetError, match=expected):
        A.parse(raw)


def test_manifest_sha256_is_of_the_file_bytes_and_loading_reads_the_file(tmp_path):
    """Contract: the asset library is named by the sha256 of the manifest's file bytes, so any edit of the file (a new
    asset, a changed box) makes the next render look for another library; an absent file is an AssetError. Regression
    caught: hashing the parsed content, so a whitespace or order edit leaves a stale library in use, or a hash of
    something other than the file."""
    path = write_manifest(tmp_path / "m.json", raw_manifest())
    m = A.load(path)
    assert m.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    write_manifest(path, {**raw_manifest(), "walls": None})
    assert A.load(path).sha256 != m.sha256
    with pytest.raises(A.AssetError, match="cannot read"):
        A.load(tmp_path / "absent.json")


# -- the cache -------------------------------------------------------------------------------------------------
def test_cache_dir_refuses_the_repository_and_a_dataset(tmp_path, monkeypatch):
    """Contract (AGENTS: truth and boundaries): downloaded assets live outside git and outside any dataset (the
    redesign's `scripts/` + `private/` layout, or a release's `episodes/` + `private/`), so they never land in a commit
    or beside agent-facing files. Regression caught: an STBENCH_ASSET_CACHE pointing into the checkout, a git work tree
    or a dataset root being accepted; a dataset recognised by the retired layout alone."""
    (tmp_path / "ds" / "scripts" / "abc").mkdir(parents=True)
    (tmp_path / "ds" / "private" / "abc").mkdir(parents=True)
    (tmp_path / "old" / "episodes").mkdir(parents=True)
    (tmp_path / "old" / "private").mkdir()
    (tmp_path / "work" / ".git").mkdir(parents=True)
    ds, old, work = tmp_path / "ds", tmp_path / "old", tmp_path / "work"
    for bad in (A.REPO / "asset-cache", A.REPO, ds / "assets", ds, old / "assets", old, work / "a" / "b"):
        monkeypatch.setenv("STBENCH_ASSET_CACHE", str(bad))
        with pytest.raises(A.AssetError, match="STBENCH_ASSET_CACHE"):
            A.cache_dir()
    monkeypatch.setenv("STBENCH_ASSET_CACHE", str(tmp_path / "cache"))
    assert A.cache_dir() == (tmp_path / "cache").resolve() and not (tmp_path / "cache").exists()


def test_cache_dir_defaults_to_the_per_user_cache(tmp_path, monkeypatch):
    """The default is ~/.cache/stbench/assets on every platform, never %LOCALAPPDATA% (which Windows redirects for a
    packaged app, so two shells would see two caches). Regression caught: a default inside the working directory (and
    so the repository), or one under LOCALAPPDATA."""
    monkeypatch.delenv("STBENCH_ASSET_CACHE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(tmp_path / "home"))
    assert A.cache_dir() == (tmp_path / "home" / ".cache" / "stbench" / "assets").resolve()


def test_cache_layout_puts_files_under_the_asset_folder_and_unpacked_zips_under_x(tmp_path):
    """One owner for the layout the fetcher writes and the library build reads. Regression caught: the two disagreeing
    about where a zip's members are."""
    m = manifest()
    mug, mat = m.asset("mug_a"), m.asset("mat_a")
    assert A.file_path(mug, mug.files[1], tmp_path) == tmp_path / "mug_a" / "tex" / "texture.png"
    assert A.extract_dir(mat, tmp_path) == tmp_path / "mat_a" / "x"
