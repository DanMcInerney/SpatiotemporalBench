"""The render's look modules (`render/look.py` and `render/look_<catalog>.py`, plan-redesign section 5.1 "M2.0 first"):
the tables that say how a place looks, keyed by a region's kind in one module a catalog, with no fallback. Free: every
module here is plain Python, loaded by path. That the Blender scenes did not move is the digests' job
(`test_render_city_blender.py`, `test_wear_home_blender.py`, `test_floor_look_blender.py`,
`test_render_lighting_blender.py`)."""

from __future__ import annotations

import ast
import sys

import pytest

from dressing_kit import RENDER, raw_map, render_module
from stbench import asset_manifest as A
from stbench.catalog import CATALOGS
from stbench.mapspec import bundled_maps, load_map

HOME_KINDS = ["bathroom", "bedroom", "hall", "kitchen", "living room"]


@pytest.fixture(scope="module")
def look():
    return render_module("look")


def test_every_bundled_maps_kinds_are_in_its_catalogs_vocabulary(look):
    """Contract: every region of every bundled map has a kind (`Region.kind`) that is in the vocabulary of the map's
    catalog, and every catalog of `catalog.CATALOGS` has a look module. Regression caught: a new map whose rooms nothing
    knows, dressed and lit by a fallback (a checkerboard bedroom), or a catalog added with no look."""
    assert set(look.catalogs()) == set(CATALOGS)
    for name in bundled_maps():
        m = load_map(name)
        assert {r.kind for r in m.regions} <= look.kinds(m.catalog), (name, m.catalog)


def test_every_table_covers_exactly_the_vocabulary(look):
    """Contract: in each look module every per-kind table (`PLACEHOLDER_FLOOR`, and `FLOOR_FINISH`, `WALL_PAINT`,
    `ROLE`, `RIG["lux"]` and `RIG["cct_k"]` where the catalog has one: `None` or absent says it needs none) has exactly
    `KINDS` as its keys, the manifest's `floors` has a floor for every kind of every catalog, and `DRESSING` names a
    layout module that exists (or is `None`). Regression caught: a kind added to the vocabulary and forgotten in one
    table (the silent fallback, moved), a row left behind for a kind that is gone, a floor the assets appearance cannot
    find, or a dressing that names no layout."""
    floors = set(A.packaged().floors)
    for catalog in look.catalogs():
        mod = look.module(catalog)
        tables = {"PLACEHOLDER_FLOOR": mod.PLACEHOLDER_FLOOR, "FLOOR_FINISH": mod.FLOOR_FINISH,
                  "WALL_PAINT": getattr(mod, "WALL_PAINT", None), "ROLE": getattr(mod, "ROLE", None),
                  "RIG lux": mod.RIG["lux"], "RIG cct_k": mod.RIG["cct_k"]}
        assert tables["PLACEHOLDER_FLOOR"] is not None, catalog
        for name, table in tables.items():
            assert table is None or set(table) == mod.KINDS, (catalog, name)
        assert floors >= mod.KINDS, (catalog, sorted(mod.KINDS - floors))
        assert mod.DRESSING is None or (RENDER / f"dressing_layout_{mod.DRESSING}.py").is_file(), catalog


def test_an_unknown_kind_or_catalog_raises(look):
    """Contract: `look.get` and `look.module` raise `LookError` for a kind a table has no row for, a table a catalog
    declares `None` or does not hold, and a catalog with no module; each message names what was asked for and what is
    known. Regression caught: the `.get(..., default)` this lane removed coming back (a ballroom floored as a tile
    hall)."""
    with pytest.raises(look.LookError) as kind:
        look.get("home", "PLACEHOLDER_FLOOR", "ballroom")
    assert "'ballroom'" in str(kind.value) and all(k in str(kind.value) for k in HOME_KINDS)
    with pytest.raises(look.LookError) as catalog:
        look.module("suburb")
    assert "'suburb'" in str(catalog.value) and all(c in str(catalog.value) for c in ("home", "warehouse", "downtown"))
    for table in ("FLOOR_FINISH", "WALL_PAINT"):  # the warehouse declares the first `None` and holds no second
        with pytest.raises(look.LookError, match=table):
            look.get("warehouse", table, "dock")
    assert look.get("home", "PLACEHOLDER_FLOOR", "bathroom") == "tile_blue"  # a known kind still answers


def test_the_kind_of_a_region_is_its_kind_else_its_name_as_a_region_or_a_spec_dict(look):
    """Contract: `look.kind_of` reads a parsed `Region` and a region of the map spec dict alike, giving the region's
    `kind` where the map names one and its name where it does not (the rule `mapspec` parses with). Regression caught:
    the dressing (which reads the dict) and the structure (which reads the `Region`) disagreeing on what a room is."""
    spec = raw_map("apartment_a")
    assert [look.kind_of(r) for r in spec["regions"]] == [look.kind_of(r) for r in load_map("apartment_a").regions]
    assert look.kind_of({"id": "R9", "name": "west aisle"}) == "west aisle"
    assert look.kind_of({"id": "R9", "name": "west aisle", "kind": "aisle"}) == "aisle"


def test_the_home_roles_are_the_ones_the_furniture_gave(look):
    """Contract: `dressing_home_plan.Plan` over `apartment_a` gives its rooms R1 to R5 the roles hall, living, kitchen,
    bedroom and bathroom, which are the roles the fixed furniture each room holds gave them before the role became a
    table (written by hand here). Regression caught: a role table that dresses the bathroom as a hall."""
    plan = render_module("dressing_home_plan").Plan(raw_map("apartment_a"))
    assert [(r.id, r.name, r.role) for r in plan.rooms] == [
        ("R1", "hall", "hall"), ("R2", "living room", "living"), ("R3", "kitchen", "kitchen"),
        ("R4", "bedroom", "bedroom"), ("R5", "bathroom", "bathroom")]


def test_look_modules_import_nothing_from_bpy_or_stbench():
    """Contract: `look.py` and the look modules import the standard library only, so a free test (and the dressing
    plan, which runs in Blender and in a test alike) can load them. Regression caught: a look module that imports `bpy`
    (or a `render/` module that does) and can no longer be loaded without Blender."""
    files = [RENDER / "look.py", *sorted(RENDER.glob("look_*.py"))]
    assert len(files) == 1 + len(CATALOGS)
    for file in files:
        name, imported = file.name, set()
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= set(sys.stdlib_module_names), (name, sorted(imported - set(sys.stdlib_module_names)))


def test_the_downtown_skys_light_desaturation_is_one_number_in_both_files(look):
    """Contract: `look_downtown.RIG["hdri"]["light_desaturation"]` (a `bpy`-free file cannot read it) is
    `lighting_sky.SKY_LIGHT_DESAT`, the number the procedural sky is desaturated by. Regression caught: one edited and
    not the other, so the procedural sky and the photographic one light the street in different colour."""
    source = ast.parse((RENDER / "lighting_sky.py").read_text(encoding="utf-8"))
    (desat,) = [ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "SKY_LIGHT_DESAT" for t in n.targets)]
    assert look.module("downtown").RIG["hdri"]["light_desaturation"] == desat
