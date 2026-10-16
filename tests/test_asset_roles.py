"""The manifest's `dressing`, `hdri` and `texture` roles and its `skies`, on synthetic manifests from `asset_synth`:
what `validate` holds each to, the accessors the renderer reads them through, and the credit line a CC0 source may ask
for (the `clutter` role is `test_asset_clutter.py`'s). Nothing touches the network; the packaged manifest's assets are
`test_asset_phase1.py`'s."""

from __future__ import annotations

import copy

import pytest

from asset_synth import CC0, asset, manifest, raw_manifest
from stbench import asset_fetch as F
from stbench import asset_manifest as A

KINDS = {"catalog_kinds": ("mug", "glass"), "light_kinds": ()}


def file_entry(path: str, **extra) -> dict:
    return {"path": path, "url": f"https://{CC0}/{path}", **extra}


def dressing(asset_id: str = "dress_a", **extra) -> dict:
    return {"id": asset_id, "source": "cc0", "role": "dressing", "title": "Dress A",
            "page": f"https://{CC0}/a/{asset_id}", "licence": "CC0-1.0",
            "files": [file_entry("dress_1k.gltf"), file_entry("dress.bin")], "entry": "dress_1k.gltf", "format": "gltf",
            "transform": {"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 0]}, "pivot": "base_centre",
            "native_bbox_m": [0.6, 0.02, 0.84], "fit": {"mode": "per_axis", "max_anisotropy": 0.05},
            "uses": ["a picture on a wall"], **extra}


def hdri(asset_id: str = "sky_a", **extra) -> dict:
    return {"id": asset_id, "source": "cc0", "role": "hdri", "title": "Sky A", "page": f"https://{CC0}/a/{asset_id}",
            "licence": "CC0-1.0", "files": [file_entry("sky_a_2k.hdr")], "entry": "sky_a_2k.hdr", "format": "hdr",
            "uses": ["sky of the home catalog"], **extra}


def texture(asset_id: str = "tex_a", **extra) -> dict:
    return {"id": asset_id, "source": "cc0", "role": "texture", "title": "Texture A",
            "page": f"https://{CC0}/a/{asset_id}", "licence": "CC0-1.0",
            "files": [file_entry("tex_a_1K.zip", extract=True)],
            "uses": ["a grime layer"], **extra}


def add(*entries, **top):
    def edit(raw):
        raw["assets"] += copy.deepcopy(entries)
        raw.update(top)
    return edit


def problems(edit=None, **kinds) -> list[str]:
    return A.validate(manifest(edit=edit), **kinds)


FULL = add(dressing(), hdri(), texture(), skies={"home": "sky_a"})


def test_a_manifest_with_a_dressing_model_an_hdri_and_a_texture_set_is_valid_and_asks_for_no_kind():
    """Contract: the dressing, hdri and texture roles pass `validate`, and none of them is a model, so the
    coverage of queried kinds (exactly one model per kind) is unchanged by them. Regression caught: a dressing model
    counted as a model of a kind (the entity of that kind would get either), or the new roles rejected outright."""
    assert {"dressing", "hdri", "texture"} <= set(A.ROLES)
    assert problems(FULL, **KINDS) == []
    assert A.packaged().skies and all(a.role != "model" or a.kind for a in A.packaged().assets)


@pytest.mark.parametrize("edit, expected", [
    pytest.param(add(dressing(kind="mug")), "never an entity", id="a dressing model with a kind"),
    pytest.param(add(dressing(mobility="light")), "never an entity", id="a dressing model with a mobility"),
    pytest.param(add(dressing(tint={"mode": "luminance", "materials": ["*"]})), "never an entity",
                 id="a dressing model tinted by a colour attribute"),
    pytest.param(add(dressing(compose={"mode": "pair_mirror", "gap_m": 0.1})), "never an entity",
                 id="a dressing model composed into a pair"),
    pytest.param(add(dressing(alternate_for="mug_a")), "never an entity", id="an alternate dressing model"),
    pytest.param(add(dressing(format="obj")), "is a glTF", id="an OBJ dressing model"),
    pytest.param(add(dressing(entry="elsewhere.gltf")), "an entry that is one of its files", id="no entry file"),
    pytest.param(add(dressing(transform=None)), "a transform", id="no transform"),
    pytest.param(add(dressing(fit={"mode": "per_axis", "max_anisotropy": 0.5})), "max_anisotropy",
                 id="a squash over the cap"),
    pytest.param(add(dressing(native_bbox_m=[0.6, 0.0, 0.84])), "native_bbox_m", id="a flat box"),
    pytest.param(add(dressing(source="procedural", licence="GPL-3.0-or-later", files=[], builder="x",
                              fit={"mode": "exact"})), "fetched model", id="procedural dressing"),
    pytest.param(add(hdri(files=[file_entry("a.hdr"), file_entry("b.hdr")])), "one file", id="an hdri of two files"),
    pytest.param(add(hdri(format="png")), "one file, its entry", id="an hdri that is no radiance or exr file"),
    pytest.param(add(hdri(entry="other.hdr")), "its entry", id="an hdri whose entry is not its file"),
    pytest.param(add(hdri(transform={"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 0]})), "no transform",
                 id="an hdri with a transform"),
    pytest.param(add(hdri(kind="mug")), "no kind", id="an hdri with a kind"),
    pytest.param(add(texture(files=[file_entry("tex_a_1K.zip")])), "unpacked",
                 id="a texture zip not unpacked"),
    pytest.param(add(texture(tile_m=[0.0, 1.0])), "tile_m", id="a texture with no real size"),
    pytest.param(add(hdri(), skies={"nowhere": "sky_a"}), "not a catalog", id="a sky for no catalog"),
    pytest.param(add(texture(), skies={"home": "tex_a"}), "not an hdri", id="a sky that is a texture set"),
    pytest.param(add(skies={"home": "absent"}), "not an hdri", id="a sky that is nothing"),
    pytest.param(lambda r: asset(r, "mug_a").update(drop_slots=["glass"]), "is for a dressing or clutter model",
                 id="drop_slots on an entity's model"),
    pytest.param(lambda r: asset(r, "mat_a").update(rest_pose=["door"]), "is for a dressing model",
                 id="rest_pose on a material"),
    pytest.param(add(dressing(drop_slots=[3])), "must be names", id="a slot that is not a name"),
    pytest.param(add(dressing(rest_pose=[""])), "must be names", id="a node with no name"),
])
def test_validate_holds_each_new_role_to_its_rules(edit, expected):
    """Contract: a dressing model is never an entity (no kind, mobility, tint, pair or alternate: the validity guard),
    is a fetched glTF whose box and fit are well formed; an HDRI is one `.hdr` or `.exr` file; a texture set is an
    unpacked zip; a sky names a catalog and an HDRI; and the two edits a library build makes to a published mesh are a
    dressing model's alone. Regression caught: a dressing piece that is an entity in disguise (it would carry a pass
    index and be asked about), an HDRI the library cannot read, or an edit that slips onto an entity's mesh."""
    found = problems(edit)
    assert any(expected in p for p in found), found


def test_the_licence_rules_hold_for_the_new_roles_too():
    """Contract (VISION rule 11): a dressing model or texture set under a licence that is not on the allowlist, or from
    a host its source does not list, is refused like any other asset. Regression caught: a new role that skips the
    licence and host checks."""
    assert any("not on the allowlist" in p for p in problems(add(dressing(licence="CC-BY-NC-4.0"))))
    bad_host = dressing(files=[{**file_entry("dress_1k.gltf"), "url": "https://evil.example.net/x.gltf"},
                               file_entry("dress.bin")])
    assert any("hosts" in p for p in problems(add(bad_host)))
    assert any("differs from its source" in p for p in problems(add(hdri(licence="CC-BY-4.0"))))


# -- the accessors -------------------------------------------------------------------------------------------------
def test_hdri_path_and_sky_for_name_the_cached_file_of_a_catalogs_sky(tmp_path):
    """Contract (`asset_manifest.hdri_path`): the sky of a catalog is the one file of the HDRI the manifest's `skies`
    names, in its asset's cache folder, and None for a catalog with no sky. Regression caught: a path that is not
    where `stbench assets fetch` puts the file, or a catalog without a sky given another catalog's."""
    m = manifest(edit=FULL)
    assert A.hdri_path(m, "home", tmp_path) == tmp_path / "sky_a" / "sky_a_2k.hdr"
    assert A.hdri_path(m, "warehouse", tmp_path) is None and m.sky_for("warehouse") is None
    assert m.sky_for("home").role == "hdri"
    with pytest.raises(A.AssetError, match="is a texture, not a hdri"):
        manifest(edit=add(texture(), skies={"home": "tex_a"})).sky_for("home")


def test_texture_dir_is_the_folder_a_set_is_unpacked_to_and_refuses_a_model(tmp_path):
    """Contract (`asset_manifest.texture_dir`): a texture or material set's maps are in `<cache>/<id>/x/`, where
    `fetch` unpacks the zip; any other role is an error. Regression caught: a layer reading from a folder `fetch` never
    fills, or a mesh taken for a texture set."""
    m = manifest(edit=FULL)
    assert A.texture_dir(m, "tex_a", tmp_path) == tmp_path / "tex_a" / "x"
    assert A.texture_dir(m, "mat_a", tmp_path) == tmp_path / "mat_a" / "x"
    with pytest.raises(A.AssetError, match="not a texture set"):
        A.texture_dir(m, "dress_a", tmp_path)


def test_a_texture_set_and_an_hdri_are_fetched_and_unpacked_like_a_material(tmp_path):
    """Contract (`asset_fetch`): the plan of a download lists every file of the new roles, so `fetch` brings them as it
    brings a material. Regression caught: a role the fetcher skips, so a render starts without its sky or its masks."""
    m = manifest(edit=FULL)
    wanted = {(t.asset.id, t.file.path) for t in F.plan(m, cache=tmp_path)}
    assert {("sky_a", "sky_a_2k.hdr"), ("tex_a", "tex_a_1K.zip"), ("dress_a", "dress_1k.gltf")} <= wanted


# -- credits -------------------------------------------------------------------------------------------------------
def test_a_cc0_source_may_ask_for_a_credit_line_and_it_leads_that_sources_courtesy_entry():
    """Contract (Poly Haven's API terms): a CC0 source's own credit line, where the manifest holds one, is printed with
    that source in `CREDITS.md`, before its assets; a source without one prints as before. Regression caught: the
    "Powered by Poly Haven" line dropped from the dataset's credits."""
    def credit(raw):
        raw["sources"]["cc0"]["attribution"] = "Powered by Cee Zero (https://cc0.example.org)"
        add(dressing())(raw)

    text = F.credits(manifest(edit=credit))
    line = next(ln for ln in text.splitlines() if ln.startswith("- Cee Zero"))
    assert line.index("Powered by Cee Zero") < line.index("Dress A") < line.index("Material A")
    plain = next(ln for ln in F.credits(manifest(edit=add(dressing()))).splitlines() if ln.startswith("- Cee Zero"))
    assert "Powered by" not in plain and "Dress A" in plain


def test_the_synthetic_base_manifest_is_untouched_by_these_helpers():
    """The helpers copy: the base manifest the other asset tests share keeps its four assets. Regression caught: an
    edit function that appends to the shared fixture and leaks a dressing model into every other test."""
    assert [a["id"] for a in raw_manifest()["assets"]] == ["mug_a", "mug_b", "mat_a", "proc_glass"]
    manifest(edit=FULL)
    assert [a["id"] for a in raw_manifest()["assets"]] == ["mug_a", "mug_b", "mat_a", "proc_glass"]
