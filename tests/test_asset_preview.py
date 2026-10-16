"""The contact sheets and the library gate for the dressing, hdri and texture roles, without Blender: the dressing
sheet, a surface on a plane and on a sphere, the texture-set sheet, and what the gate demands of a dressing model. The
kinds' sheets and the gate's other rules are `test_asset_render.py`'s; the real renders are
`test_asset_dressing_blender.py`'s."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import asset_world as W
from stbench import asset_manifest as A
from stbench import asset_preview as P
from stbench import asset_render as AR

TILE = (24, 18)
BLENDER = "5.2.1"


def solid(path: Path, value, size=TILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), value, np.uint8)).save(path)


def test_the_packaged_manifests_preview_lists_every_dressing_model_with_its_box_and_every_material_surface():
    """Contract (`asset_preview.preview_jobs`): the job list names every dressing model of the manifest with the box
    the sheet's label shows and every `material` as a surface, and a `texture` set is no surface (it has no material).
    Regression caught: a dressing model nobody looks at before the render, or a grime mask rendered as a surface and
    failing the library lookup."""
    jobs = P.preview_jobs(A.packaged())
    ids = [d["id"] for d in jobs["dressing"]]
    assert len(ids) == 10 and {"wall_clock", "power_box_01", "security_camera_01"} <= set(ids)
    assert all(d["box"] and len(d["box"]) == 3 for d in jobs["dressing"])
    assert "acg_Tiles141" in {s["id"] for s in jobs["surfaces"]}
    assert not {s["id"] for s in jobs["surfaces"]} & {"acg_Scratches004", "acg_RoadLines006", "potsdamer_platz"}


def test_the_dressing_sheet_is_a_row_per_model_and_a_column_per_view_and_a_surface_has_a_plane_and_a_sphere(tmp_path):
    """Contract (`asset_preview.compose`): the dressing sheet lays each model's four views in a row, the models under
    one another, under a title and the views' names; each surface is a plane tile then a sphere tile (three surfaces to
    a row). Regression caught: views of one model shuffled with another's, or the sphere tile dropped, so a material's
    look on a curved surface is never seen."""
    tiles = tmp_path / "tiles"
    jobs = {"items": [], "dressing": [{"id": "clock", "box": [0.3, 0.05, 0.3], "uses": []},
                      {"id": "frame", "box": [0.6, 0.02, 0.8], "uses": []}],
            "surfaces": [{"id": "mat_a", "tile_m": None}, {"id": "mat_b", "tile_m": None}]}
    for r, model in enumerate(("clock", "frame")):
        for v in range(len(P.DRESSING_VIEWS)):
            solid(tiles / "_dressing" / f"{model}__{v}.png", 100 * r + 20 * v + 10)
    for k, name in enumerate(("mat_a", "mat_b")):
        solid(tiles / "_surfaces" / f"{name}.png", 50 + 10 * k)
        solid(tiles / "_surfaces" / f"{name}__sphere.png", 150 + 10 * k)
    written = P.compose(jobs, tiles, tmp_path, TILE)
    assert [p.name for p in written] == ["dressing.png", "surfaces.png", "index.json"]
    with Image.open(tmp_path / "dressing.png") as sheet:
        dressing = np.asarray(sheet.convert("RGB"))
    assert dressing.shape[:2] == (2 * P.LABEL_H + 2 * 18, 4 * 24)
    for r in range(2):
        for v in range(4):
            assert (dressing[2 * P.LABEL_H + r * 18 + 17, v * 24 + 12] == 100 * r + 20 * v + 10).all()
    with Image.open(tmp_path / "surfaces.png") as sheet:
        surfaces = np.asarray(sheet.convert("RGB"))
    assert surfaces.shape[:2] == (18 + P.LABEL_H, 6 * 24)  # two surfaces: one row of two (plane, sphere) pairs
    for x, value in ((12, 50), (36, 150), (60, 60), (84, 160)):
        assert (surfaces[P.LABEL_H + 9, x] == value).all(), x
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["dressing"] == ["clock", "frame"] and index["surfaces"] == ["mat_a", "mat_b"]


def texture_world(tmp_path: Path) -> tuple[A.Manifest, Path]:
    """A manifest with one texture set whose unpacked folder holds a red colour map and an opacity map that is white
    on its left half and black on its right."""
    raw = W.raw_manifest()
    raw["assets"].append({"id": "grime", "source": "cc0", "role": "texture", "title": "Grime", "page": f"https://{W.HOST}/g",
                          "licence": "CC0-1.0", "files": [W._entry("grime_1K.zip", extract=True)],
                          "uses": ["wear"]})
    folder = tmp_path / "cache" / "grime" / A.EXTRACT_DIR
    solid(folder / "Grime_1K-JPG_Color.jpg", (200, 0, 0), (16, 16))
    opacity = np.zeros((16, 16, 3), np.uint8)
    opacity[:, :8] = 255
    Image.fromarray(opacity).save(folder / "Grime_1K-JPG_Opacity.jpg")
    return A.parse(raw), tmp_path / "cache"


def test_the_texture_sheet_shows_each_map_and_the_colour_over_grey_through_the_opacity(tmp_path):
    """Contract (`asset_preview.texture_sheet`): per texture set one row of its colour map, its opacity map, its
    roughness map (blank when it has none) and the colour laid over mid-grey through the opacity, what a decal or a
    grime layer adds to a surface; None when the manifest holds no texture set. Hand-derived: a red colour map under an
    opacity that is white on the left half and black on the right is red on the left and grey (128) on the right in
    the last tile. Regression caught: opacity read as colour, the composite inverted, or a sheet for a manifest without
    textures."""
    assert P.texture_sheet(A.parse(W.raw_manifest()), tmp_path) is None
    m, cache = texture_world(tmp_path)
    sheet = np.asarray(P.texture_sheet(m, cache).convert("RGB")).astype(int)
    side, top = P.MAP_TILE, P.LABEL_H
    assert sheet.shape[:2] == (side + P.LABEL_H, 4 * side)
    assert np.abs(sheet[top + 64, 64] - (200, 0, 0)).max() <= 6                     # the colour map (jpeg-tolerant)
    assert sheet[top + 64, side + 16].min() >= 250 and sheet[top + 64, side + 112].max() <= 5  # the opacity map
    assert (sheet[top + 64, 2 * side + 64] == (24, 24, 24)).all()                   # no roughness map: blank
    assert np.abs(sheet[top + 64, 3 * side + 16] - (200, 0, 0)).max() <= 6           # opaque on the left: the colour
    assert np.abs(sheet[top + 64, 3 * side + 112] - 128).max() <= 6              # clear on the right: the grey


# -- the gate ------------------------------------------------------------------------------------------------------
DRESS = b"a glTF"


def dressed(tmp_path: Path, box) -> tuple[W.World, Path]:
    """The tiny asset world plus a dressing model `dress_x` recorded with `box`, and a stand-in library that holds
    `dress_x` measured at `(0.3, 0.2, 0.1)`."""
    def edit(raw):
        raw["assets"].append({"id": "dress_x", "source": "cc0", "role": "dressing", "title": "Dress X",
                              "page": f"https://{W.HOST}/dress_x", "licence": "CC0-1.0",
                              "files": [W._entry("dress.gltf")], "entry": "dress.gltf", "format": "gltf",
                              "transform": {"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 0]},
                              "pivot": "base_centre", "native_bbox_m": box,
                              "fit": {"mode": "per_axis", "max_anisotropy": 0.05}, "uses": ["a test"]})
    world = W.build(tmp_path, edit)
    (world.cache / "dress_x").mkdir()
    (world.cache / "dress_x" / "dress.gltf").write_bytes(DRESS)
    m = A.load(world.manifest_path)
    blend = tmp_path / "lib" / "a.blend"
    blend.parent.mkdir()
    blend.write_bytes(b"a blend file")
    doc = {"materials": {"mat_a": {"mean_lum": 0.2}},
           "assets": {a.id: {"kind": a.kind, "bbox_m": list(a.native_bbox_m or (0.1, 0.1, 0.1)),
                             "slots": [{"slot": "body", "tinted": True, "mean_lum": 0.5}]}
                      for a in m.assets if a.role == "model" and not a.alternate_for}}
    doc["assets"]["dress_x"] = {"role": "dressing", "kind": None, "bbox_m": [0.3, 0.2, 0.1], "slots": []}
    AR.json_path(blend).write_text(json.dumps(doc), encoding="utf-8")
    return world, blend


def problems(world: W.World, blend: Path, complete: bool = True) -> list[str]:
    return AR.check_library(A.load(world.manifest_path), blend, world.cache, complete)


def test_the_gate_holds_a_dressing_model_to_the_library_and_to_its_recorded_box(tmp_path):
    """Contract (`asset_render.check_library`): a dressing model must be in the library with the box the manifest
    records (to 1 mm), as a model must; a texture set and an HDRI need no entry. Regression caught: a render that starts
    without a dressing model the manifest lists (its builders would draw stand-ins in a render called photoreal), or one
    whose recorded box is stale."""
    world, blend = dressed(tmp_path / "ok", [0.3, 0.2, 0.1])
    assert problems(world, blend) == []
    world, blend = dressed(tmp_path / "off", [0.3, 0.2, 0.103])
    assert any("dress_x" in p and "1 mm" in p for p in problems(world, blend))
    world, blend = dressed(tmp_path / "unrecorded", None)
    assert any("dress_x: native_bbox_m is not recorded" in p for p in problems(world, blend))
    assert problems(world, blend, complete=False) == []
    world, blend = dressed(tmp_path / "missing", [0.3, 0.2, 0.1])
    doc = json.loads(AR.json_path(blend).read_text(encoding="utf-8"))
    del doc["assets"]["dress_x"]
    AR.json_path(blend).write_text(json.dumps(doc), encoding="utf-8")
    assert any("dress_x: not in the library" in p for p in problems(world, blend))


def test_the_gate_still_checks_the_cache_so_a_missing_hdri_or_texture_stops_a_render(tmp_path):
    """Contract (`asset_render.check_library` -> `cache_problems`): every file of every role must be in the cache, so a
    render cannot start without its sky or its masks. Regression caught: a role the cache check skips."""
    world, blend = dressed(tmp_path, [0.3, 0.2, 0.1])
    (world.cache / "dress_x" / "dress.gltf").unlink()
    assert any("dress_x dress.gltf: missing" in p for p in problems(world, blend))
    with pytest.raises(AR.LibraryError, match="dress_x dress.gltf"):
        AR.require_library(world.manifest_path, blend, world.cache, BLENDER)


def test_every_clutter_model_has_a_sheet_of_its_own_in_its_colours_at_its_box():
    """Contract (`asset_preview.preview_jobs`): the job list holds one item per clutter model of the manifest,
    named by the model (a kind has several), at the box the library measured, in the model's own colour when it is
    untinted and in every colour its kind lists when it takes the entity's; `--kinds` picks clutter kinds like any
    other, and `compose` writes a sheet per model beside the others. Hand values: the toaster has the one model, 0.168 x
    0.295 x 0.191 m, untinted grey; the shelved books have three, tinted white, grey, beige and brown. Regression
    caught: two models of a kind drawn over one another on one sheet, a tinted model shown in one colour, or a model no
    person sees before an episode renders with it."""
    m = A.packaged()
    jobs = P.preview_jobs(m)
    clutter = {i["name"]: i for i in jobs["items"] if "name" in i and m.asset(i["asset"]).role == "clutter"}
    assert set(clutter) == {a.id for a in m.assets if a.role == "clutter"}
    toaster = clutter["gso_toaster_bd_tr3500sd"]
    assert (toaster["kind"], toaster["colours"], toaster["size"]) == ("toaster", ["grey"], [0.16778, 0.29528, 0.19072])
    books = [i for i in clutter.values() if i["kind"] == "shelved_book"]
    assert len(books) == 3 and all(i["colours"] == ["white", "grey", "beige", "brown"] for i in books)
    assert sorted(i["name"] for i in P.preview_jobs(m, ["toaster", "router"])["items"]) == [
        "gso_toaster_bd_tr3500sd", "proc_router"]
    assert all(i["asset"] == i["name"] for i in clutter.values())


def test_a_clutter_sheet_is_written_under_the_models_name(tmp_path):
    """Contract (`asset_preview.compose`): an item that names itself (a clutter model) has its tiles read from, and its
    sheet written to, that name, so the models of one kind do not share a file. Regression caught: the second toaster
    overwriting the first."""
    tiles = tmp_path / "tiles"
    jobs = {"items": [{"kind": "toaster", "name": "toast_a", "asset": "toast_a", "colours": ["grey"],
                       "size": [0.2, 0.3, 0.2], "note": "", "outdoor": None},
                      {"kind": "toaster", "name": "toast_b", "asset": "toast_b", "colours": ["grey"],
                       "size": [0.2, 0.3, 0.2], "note": "", "outdoor": None}], "surfaces": [], "dressing": []}
    for name, value in (("toast_a", 60), ("toast_b", 160)):
        for v in range(P.VIEWS):
            solid(tiles / name / f"grey__{v}.png", value)
    written = P.compose(jobs, tiles, tmp_path, TILE)
    assert [p.name for p in written] == ["toast_a.png", "toast_b.png", "index.json"]
    for name, value in (("toast_a", 60), ("toast_b", 160)):
        with Image.open(tmp_path / f"{name}.png") as sheet:
            assert (np.asarray(sheet.convert("RGB"))[2 * P.LABEL_H + 9, 12] == value).all(), name
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert set(index["kinds"]) == {"toast_a", "toast_b"}
