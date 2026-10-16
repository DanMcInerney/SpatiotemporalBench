"""The realism Phase 1 assets of the packaged manifest: which assets it lists, that
each is CC0 from its source's host, that a dressing model's recorded box is the one its published size says and the one
a map's layout fits it to, and (skipped when the asset cache is not filled) that the cached sky is a Radiance file and
the cached sets hold the maps a layer reads. Free: no network; the layouts are plain Python."""

from __future__ import annotations

from pathlib import Path

import pytest

from dressing_kit import layout_of, layout_path, raw_map
from stbench import asset_fetch as F
from stbench import asset_manifest as A
from stbench.mapspec import bundled_maps, load_map

# Poly Haven's own published sizes, (width, depth, height) in millimetres, from api.polyhaven.com/info/<id> read on
# 2026-10-05, and the one the glTF itself gives: power_box_01's API size is for the door open (377 x 418), the glTF's
# box node is 345 x 94 and its door, closed, brings the depth to 107.
POLY_HAVEN_MM = {
    "hanging_picture_frame_01": (594, 16, 841), "hanging_picture_frame_02": (752, 33, 500),
    "hanging_picture_frame_03": (385, 32, 502), "fancy_picture_frame_01": (603, 20, 464), "wall_clock": (320, 47, 320),
    "modern_ceiling_lamp_01": (432, 432, 952), "ceiling_fan": (1463, 1463, 516), "fire_alarm": (99, 26, 144),
    "security_camera_01": (170, 554, 286), "power_box_01": (345, 107, 503),
}
HDRIS = {"home": "potsdamer_platz", "downtown": "kloofendal_48d_partly_cloudy_puresky"}
MATERIALS = ["Tiles141", "Fabric061", "Plastic010", "Concrete047A", "Concrete034", "CorrugatedSteel005",
             "PaintedMetal004", "MetalPlates006", "Cardboard004", "Road007", "PavingStones038"]
TEXTURES = ["SurfaceImperfections003", "Fingerprints002", "Smear005", "Scratches004",
            "RoadLines006", "RoadLines007", "RoadLines008", "RoadLines009", "RoadLines010", "RoadLines011"]
MM = 0.002  # a published size is rounded to a millimetre; the glTF's own is what the library measures


def phase1_ids() -> list[str]:
    return [*POLY_HAVEN_MM, *HDRIS.values(), *(f"acg_{n}" for n in MATERIALS + TEXTURES)]


def test_every_phase1_asset_is_cc0_and_on_its_sources_host():
    """Contract (VISION rule 11): each asset of the plan's Phase 1 list is in the manifest with the role it is used
    for, CC0 from Poly Haven (models and HDRIs) or ambientCG (materials, grime masks and decals), every file at a url
    on the source's host, with the page it was taken from. Hand-listed from the plan and the lane's brief. Regression
    caught: an asset dropped, filed under the wrong role (a mask built as a material, an HDRI as a model), or a licence
    that is not CC0."""
    m = A.packaged()
    roles = {**dict.fromkeys(POLY_HAVEN_MM, "dressing"), **dict.fromkeys(HDRIS.values(), "hdri"),
             **dict.fromkeys((f"acg_{n}" for n in MATERIALS), "material"),
             **dict.fromkeys((f"acg_{n}" for n in TEXTURES), "texture")}
    for asset_id, role in roles.items():
        a = m.asset(asset_id)
        host = "dl.polyhaven.org" if role in ("dressing", "hdri") else "ambientcg.com"
        assert (a.role, a.licence) == (role, "CC0-1.0"), asset_id
        assert a.source == ("polyhaven" if host.startswith("dl.") else "ambientcg") and a.page.startswith("https://")
        assert a.files and all(f.url.startswith(f"https://{host}/") for f in a.files), asset_id
    assert A.validate(m) == [] and len(set(phase1_ids())) == len(phase1_ids())


def test_the_skies_name_the_hdri_of_the_home_and_downtown_catalogs_and_the_warehouse_has_none():
    """Contract: the apartment's windows look out on an urban street HDRI and the
    city's sky is a partly cloudy one; the warehouse has no window and no HDRI. Regression caught: a catalog given
    another's sky, or a sky that points at a file the manifest does not list."""
    m = A.packaged()
    skies = {**HDRIS, "residential": HDRIS["downtown"]}  # the suburb is under the city's sky
    assert m.skies == skies and m.sky_for("warehouse") is None
    for catalog, sky in skies.items():
        assert A.hdri_path(m, catalog, Path("/cache")) == Path("/cache") / sky / f"{sky}_2k.hdr"


def test_a_dressing_models_recorded_box_is_its_published_size_and_its_fit_is_tight():
    """Contract (the library's 1 mm gate): `native_bbox_m` is the box the library measured, which for each
    Poly Haven model is its published size to a millimetre (the sizes a layout is written from), and
    `fit.max_anisotropy` is 5%, so a layout box that disagrees with its model is refused, not stretched. Regression
    caught: a recorded box of a mesh other than the one published, a loose cap that lets a layout squash a model."""
    m = A.packaged()
    for asset_id, mm in POLY_HAVEN_MM.items():
        a = m.asset(asset_id)
        assert a.native_bbox_m is not None and a.fit == {"mode": "per_axis", "max_anisotropy": 0.05}, asset_id
        assert all(abs(have - want / 1000) <= MM for have, want in zip(a.native_bbox_m, mm, strict=True)), asset_id
        assert a.kind is None and a.mobility is None and a.tint is None, asset_id


def test_the_two_edits_to_published_meshes_are_the_ones_the_contact_sheet_called_for():
    """Contract (manifest `drop_slots`, `rest_pose`): the first two hanging picture frames' glass is opaque in the
    glTF (a black rectangle, a screen), so its faces are dropped; the power panel's door is exported open 109 degrees,
    so it is closed. No other model is edited. Hand-listed from the contact sheet of 2026-10-05. Regression caught: an
    edit that spreads to a model that did not need it, or one that is lost (a black frame on every wall)."""
    edits = {a.id: (a.drop_slots, a.rest_pose) for a in A.packaged().assets
             if a.role == "dressing" and (a.drop_slots or a.rest_pose)}
    assert edits == {"hanging_picture_frame_01": (("hanging_picture_frame_01_glass",), ()),
                     "hanging_picture_frame_02": (("hanging_picture_frame_02_glass",), ()),
                     "power_box_01": ((), ("power_box_01_door",))}


def test_every_asset_a_layout_places_is_in_the_manifest_and_fits_its_box_and_every_dressing_model_is_placed():
    """Contract (`render/dressing_spec.py`: `asset:<id>` is a manifest asset fitted to the piece's box): in each bundled
    map's dressing layout every `asset:` recipe names a dressing model of the manifest, the layout's size
    (depth, width, height) is the model's recorded box (y, x, z) to within the model's own `max_anisotropy`, and every
    dressing model is placed by some layout (no dead entry). Regression caught: a layout box written from a published
    size the glTF does not have (the open power panel), a recipe naming an asset the manifest lacks, or an entry left
    behind when a layout stops using it."""
    m = A.packaged()
    placed, wrong = set(), []
    for name in bundled_maps():
        catalog = load_map(name).catalog
        if not layout_path(catalog).exists():
            continue
        for d in layout_of(catalog, raw_map(name)):
            if not d.recipe.startswith("asset:"):
                continue
            asset_id = d.recipe.removeprefix("asset:")
            a = m.of_role(asset_id, "dressing")
            placed.add(asset_id)
            ratio = [want / have for want, have in zip(d.size, (a.native_bbox_m[1], a.native_bbox_m[0],
                                                                  a.native_bbox_m[2]), strict=True)]
            if max(ratio) / min(ratio) - 1.0 > a.fit["max_anisotropy"]:
                wrong.append((name, d.cls, asset_id, d.size, a.native_bbox_m))
    assert wrong == [] and placed == {a.id for a in m.assets if a.role == "dressing"}


def test_the_credits_carry_polyhavens_line_and_the_page_of_every_phase1_asset():
    """Contract (Poly Haven's API terms, 2.5; ambientCG's page): CREDITS.md says "Powered by Poly Haven" and links the
    page of every asset taken from either source, as a courtesy for CC0. Regression caught: the credit dropped from the
    dataset, or an asset missing from it."""
    text = F.credits(A.packaged())
    assert "Powered by Poly Haven" in text
    for asset_id in phase1_ids():
        assert A.packaged().asset(asset_id).page in text, asset_id


def test_the_sources_licence_pages_were_read_on_a_recorded_day():
    """Contract (DECISIONS, "M2: assets"): each download source records the date its licence page was read.
    Regression caught: a source added without a date."""
    for key, s in A.packaged().sources.items():
        if key != A.PROCEDURAL:
            assert s.checked and len(s.checked) == 10 and s.checked[4] == s.checked[7] == "-", key
    assert A.packaged().sources["polyhaven"].checked >= "2026-10-05"


# -- the cache (skipped until `stbench assets fetch --yes` has run) -------------------------------------------------
@pytest.fixture
def cache() -> Path:
    try:
        path = A.cache_dir()
    except A.AssetError as e:
        pytest.skip(str(e))
    if not (path / "potsdamer_platz").is_dir():
        pytest.skip("the asset cache is not filled: run `stbench assets fetch --yes`")
    return path


def test_the_cached_hdris_are_radiance_files_and_the_sets_hold_the_maps_a_layer_reads(cache):
    """Contract (`asset_manifest.hdri_path`, `texture_dir`): the sky is a Radiance file (Blender's environment texture
    reads it), every material has a colour map, every decal a colour and an opacity map, and every grime mask an opacity
    map. Regression caught: a download that is an error page, or a set without the map a layer reads."""
    m = A.packaged()
    for catalog in HDRIS:
        assert A.hdri_path(m, catalog, cache).read_bytes()[:10] == b"#?RADIANCE"

    def has(asset_id: str, name: str) -> bool:
        return any(p.stem.endswith(f"_{name}") for p in A.texture_dir(m, asset_id, cache).glob("*.jpg"))

    assert all(has(f"acg_{n}", "Color") for n in MATERIALS)
    assert all(has(f"acg_{n}", "Color") and has(f"acg_{n}", "Opacity") for n in TEXTURES)
