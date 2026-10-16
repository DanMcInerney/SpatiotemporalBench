"""The manifest's `clutter` role: what `validate` holds a clutter model
to, on synthetic manifests from `asset_synth`; how a clutter kind's models are checked against the catalog; and the
packaged manifest, whose clutter models are licensed, reviewed for logos, boxed and listed by their kinds. Nothing
touches the network."""

from __future__ import annotations

import copy

import pytest

from asset_synth import CC0, asset, manifest
from stbench import asset_manifest as A
from stbench.catalog import CATALOGS, ClutterKind

BOX = [0.3, 0.2, 0.2]


def file_entry(path: str) -> dict:
    return {"path": path, "url": f"https://{CC0}/{path}"}


def clutter(asset_id: str = "clut_a", **extra) -> dict:
    """A valid untinted clutter model of the kind `toaster`: a glTF from the CC0 source in its own colour."""
    return {"id": asset_id, "source": "cc0", "role": "clutter", "kind": "toaster", "mobility": "clutter",
            "title": "Clutter A", "page": f"https://{CC0}/a/{asset_id}", "licence": "CC0-1.0",
            "files": [file_entry("clut_1k.gltf"), file_entry("clut.bin")], "entry": "clut_1k.gltf", "format": "gltf",
            "transform": {"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 90]}, "pivot": "base_centre",
            "native_bbox_m": BOX, "fit": {"mode": "exact"}, "tint": {"mode": "none", "materials": []},
            "colour": "grey", "logo_review": "none", **extra}


def procedural(asset_id: str = "proc_clut", **extra) -> dict:
    """A tinted procedural clutter model of the kind `toaster`: it takes the entity's colour and declares none."""
    return {"id": asset_id, "source": "procedural", "role": "clutter", "kind": "toaster", "mobility": "clutter",
            "title": "procedural clutter", "licence": "GPL-3.0-or-later", "builder": "glass",
            "materials": {"glass": "shader:frosted_glass"}, "native_bbox_m": BOX, "fit": {"mode": "exact"},
            "tint": {"mode": "luminance", "materials": ["glass"]}, "logo_review": "none", **extra}


def add(*entries):
    def edit(raw):
        raw["assets"] += copy.deepcopy(entries)
    return edit


def problems(edit=None, **kinds) -> list[str]:
    return A.validate(manifest(edit=edit), **kinds)


def kind(**extra) -> ClutterKind:
    fields = {"name": "toaster", "size": (0.31, 0.21, 0.21), "colors": ("grey", "black"),
              "places": ("kitchen counter",), "count": (1, 1), "models": ("clut_a",), **extra}
    return ClutterKind(**fields)


def test_a_clutter_model_of_either_sort_is_valid_and_is_no_model_of_a_kind():
    """Contract: a clutter model, untinted with its own colour or tinted and procedural,
    passes `validate`, `clutter_models` lists it by its kind in manifest order, and it is no `model` of a queried kind,
    so the exactly-one-model-per-kind coverage is unchanged by it. Regression caught: a clutter model counted as a model
    of its kind (the toaster would have two, or a carried kind a second), or the new role rejected outright."""
    edit = add(clutter(), procedural())
    assert "clutter" in A.ROLES and "clutter" in A.IMPORTED
    assert problems(edit, catalog_kinds=("mug", "glass"), light_kinds=()) == []
    m = manifest(edit=edit)
    assert [a.id for a in m.clutter_models("toaster")] == ["clut_a", "proc_clut"] and m.clutter_models("mug") == []
    assert m.model_for("mug").id == "mug_a"


@pytest.mark.parametrize("edit, expected", [
    pytest.param(add(clutter(fit={"mode": "per_axis", "max_anisotropy": 0.1})), "fitted exact", id="a squashable fit"),
    pytest.param(add(clutter(mobility="carried")), "mobility is 'clutter'", id="the mobility of an item"),
    pytest.param(add(clutter(kind=None)), "needs a kind", id="no kind"),
    pytest.param(add(clutter(colour=None)), "never both or neither", id="untinted with no colour"),
    pytest.param(add(clutter(tint={"mode": "luminance", "materials": ["*"]})), "never both or neither",
                 id="tinted and with a colour"),
    pytest.param(add(procedural(colour="grey")), "never both or neither", id="a procedural model with a colour"),
    pytest.param(add(clutter(tint={"mode": "paint", "materials": []})), "tint needs a mode", id="a tint of no mode"),
    pytest.param(add(clutter(logo_review=None)), "logo_review must be one of", id="no logo review"),
    pytest.param(add(clutter(logo_review="heavy")), "logo_review must be one of", id="a logo review of no grade"),
    pytest.param(add(clutter(logo_review="dominant")), "never used", id="a dominant logo"),
    pytest.param(add(clutter(compose={"mode": "pair_mirror", "gap_m": 0.1})), "no compose", id="a pair"),
    pytest.param(add(clutter(alternate_for="mug_a")), "no alternate_for", id="an alternate"),
    pytest.param(add(clutter(native_bbox_m=[0.3, 0.0, 0.2])), "native_bbox_m", id="a flat box"),
    pytest.param(add(clutter(entry="elsewhere.gltf")), "an entry that is one of its files", id="no entry file"),
    pytest.param(add(clutter(source="ccby", licence="CC-BY-4.0")), "needs a title and a note", id="CC BY unmarked"),
    pytest.param(add(clutter(licence="CC-BY-NC-4.0")), "not on the allowlist", id="a non-commercial licence"),
    pytest.param(add(clutter(files=[{**file_entry("clut_1k.gltf"), "url": "https://evil.example.net/c.gltf"},
                                    file_entry("clut.bin")])), "hosts", id="a file from another host"),
    pytest.param(lambda r: asset(r, "mug_a").pop("logo_review"), "needs a logo_review", id="a CC BY model unreviewed"),
])
def test_validate_holds_a_clutter_model_to_its_rules(edit, expected):
    """Contract: a clutter model is built to its own box (fitted exact, a scan rescaled by `unit_scale`), takes either
    its own colour or the entity's (never both or neither), has a logo review that is not "dominant" (CC BY licenses no
    trademark: a branded scan is never used), has no pair and no alternate, and is licensed and hosted like every other
    asset; a CC BY asset of any role has its logo review. Regression caught: a model squashed to a box it was not
    measured at, an object whose colour label is not what the pixels show, or a retail package with its brand across it
    rendered in a dataset."""
    found = problems(edit)
    assert any(expected in p for p in found), found


def test_a_clutter_model_may_drop_the_slots_a_library_would_leave_opaque_but_a_mug_may_not():
    """Contract (`drop_slots`): the one edit to a published mesh, a picture frame's glass, which the glTF exports
    opaque, is allowed on a clutter model as on a dressing model, and on no entity's model. Regression caught: a frame
    with a black rectangle for a picture, or the edit spreading to a model that needs none."""
    assert problems(add(clutter(drop_slots=["glass"]))) == []
    assert any("dressing or clutter" in p for p in problems(lambda r: asset(r, "mug_a").update(drop_slots=["x"])))


def test_a_clutter_kind_lists_exactly_the_models_the_manifest_holds_of_it():
    """Contract (`validate(..., clutter=...)`): each clutter kind of a catalog lists, as its `models`, exactly the
    manifest's clutter models of that kind; an untinted model's colour is one the kind lists; the kind's box holds each
    measured model box to a millimetre; and every clutter model has a kind in some catalog. Regression caught: a model
    in the manifest that no entity can get, a kind whose listed model does not exist (its first render fails), a
    colour label the kind does not draw, or a model larger than the box that `dressing_zones` and the strips assume."""
    edit = add(clutter(), procedural())
    both = kind(models=("clut_a", "proc_clut"))

    def found(*kinds, **extra):
        return A.validate(manifest(edit=add(clutter(**extra), procedural())), clutter=list(kinds))

    assert A.validate(manifest(edit=edit), clutter=[both]) == []
    assert any("lists models" in p for p in A.validate(manifest(edit=edit), clutter=[kind()]))  # one is not listed
    assert any("lists models" in p for p in A.validate(manifest(edit=edit), clutter=[kind(models=("clut_a", "x"))]))
    narrow = kind(colors=("black",), models=("clut_a", "proc_clut"))
    assert any("colour 'grey' is not one of" in p for p in found(narrow))
    assert any("is not inside" in p for p in found(kind(size=(0.29, 0.21, 0.21), models=("clut_a", "proc_clut"))))
    assert found(kind(size=(0.2995, 0.2, 0.2005), models=("clut_a", "proc_clut"))) == []  # within a millimetre
    assert any("not a clutter kind of any catalog" in p for p in found(kind(name="kettle", models=())))


def test_the_packaged_manifest_is_assets_3_and_its_clutter_is_licensed_reviewed_boxed_and_covered():
    """Contract (plan-redesign §4.10 item 1, VISION rule 11): every clutter model of the packaged manifest is CC0, CC BY
    (credited, with its title and a note of what changed) or our own procedural code; every kind of the catalogs lists
    exactly the models the manifest holds of it; every CC BY asset, of any role, has its logo review and none is
    "dominant"; every model has its measured box recorded (`clutter.place` sizes an entity by it), and a model that is
    untinted names its colour. Regression caught: an unlicensed or branded asset in a render, a model nobody recorded a
    box for (an entity of no size), or a kind listing a model that is not there."""
    m = A.packaged()
    carried, light, movers = A.queried_kinds()
    assert A.validate(m, carried, light, movers, clutter=A.queried_clutter(), colours=A.queried_colours()) == []
    models = [a for a in m.assets if a.role == "clutter"]
    assert models and all(a.native_bbox_m is not None for a in models)
    for a in models:
        if a.source == A.PROCEDURAL:
            assert a.licence == "GPL-3.0-or-later" and a.builder and not a.files, a.id
        else:
            assert a.licence in {"CC0-1.0", "CC-BY-4.0", "BSD-3-Clause"} and a.files and a.page, a.id
        if a.licence == "CC-BY-4.0":
            assert m.sources[a.source].attribution and a.title and a.changes, a.id
        assert (a.colour is None) == ((a.tint or {}).get("mode") in ("luminance", "swap")), a.id
    cc_by = [a for a in m.assets if a.licence.startswith("CC-BY")]
    assert cc_by and all(a.logo_review in ("none", "minor") for a in cc_by)
    assert {a.id for a in models if a.source == "gso"} <= {a.id for a in cc_by}
    assert {a.kind for a in models} == {k.name for c in CATALOGS.values() for k in c.clutter}
