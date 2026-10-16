"""The downtown's share of the asset pipeline, without Blender: the shaders its materials module builds are the ones the
manifest may name, and the contact sheets show each of its ten kinds in every colour under the map's sun and sky (the
builders against the manifest, the placeholder drawings and the fixed furniture are in `test_asset_render.py`)."""

from __future__ import annotations

import ast
from pathlib import Path

from stbench import asset_manifest as A
from stbench import asset_preview as P
from stbench.mapspec import load_map

RENDER = Path(__file__).resolve().parents[1] / "render"


def _module_literal(path: Path, name: str):
    """The literal assigned to a top-level `name` in a source file that cannot be imported here (it needs bpy)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no top-level {name} in {path}")


def test_every_shader_materials_city_builds_is_one_the_manifest_may_name():
    """Contract (plan-city section 7.3): `materials.SHADERS` lists every shader `materials.shader_material` can build,
    the downtown's included (`materials_city.NAMES`: asphalt, paving, car paint, glass, tyres, trim, lamps, rubble), and
    the downtown's models and floors use no others than those. Regression caught: a shader added to `materials_city`
    and used by a manifest slot without being listed (the manifest check passes, the library build raises KeyError), or
    one listed with no builder."""
    shaders = set(_module_literal(RENDER / "materials.py", "SHADERS"))
    city = {"asphalt", "paving", "paint", "roof_panel", *_module_literal(RENDER / "materials_city.py", "FIXED"),
            *_module_literal(RENDER / "materials_city.py", "LAMPS")}
    assert city <= shaders
    older = {"frosted_glass", "dark_glass", "rubber", "plastic", "stretch_film", "leaf", "soil", "concrete"}
    assert shaders == city | older | {"lawn"}  # the suburb's ground (`materials_residential`)


DOWNTOWN_COLOURS = {"car": ["white", "red", "blue", "green", "beige"],
                    "van": ["white", "blue", "red", "green", "yellow", "orange"],
                    "pickup": ["red", "blue", "white", "green", "orange", "beige"],
                    "box_truck": ["white", "yellow", "red", "blue", "green", "orange"],
                    "skip": ["yellow", "orange", "green", "blue", "red"],
                    "dumpster": ["green", "blue", "red", "orange", "yellow"],
                    "portable_toilet": ["blue", "green", "orange", "red"],
                    "food_cart": ["red", "blue", "yellow", "green", "white", "orange"],
                    "taxi": ["yellow", "orange"], "bus": ["blue", "red", "white", "green"]}


def test_preview_shows_every_downtown_kind_in_every_colour_under_the_maps_sun_and_sky():
    """Contract (plan-city sections 3 and 8.1, VISION rule 10): the contact sheets show the downtown's eight carried
    kinds and two movers in each of their 43 + 6 catalog colours (hand-listed from the plan's tables), at the catalog's
    size, and light each of them with the downtown's sun and sky (the sheet is where a person judges whether orange
    reads as orange in that light), while a household or warehouse kind keeps the default light. Regression caught: a
    downtown colour missing from the sheet the coordinator reviews before any episode is rendered (red against orange,
    white against beige), or the sheets lit with the household's sun, which would hide the shade problems the sun and
    sky of the map create."""
    jobs = P.preview_jobs(A.packaged())
    by_kind = {i["kind"]: i for i in jobs["items"]}
    assert {k: by_kind[k]["colours"] for k in DOWNTOWN_COLOURS} == DOWNTOWN_COLOURS
    assert sum(len(by_kind[k]["colours"]) for k in DOWNTOWN_COLOURS if k not in ("taxi", "bus")) == 43
    sky = load_map("downtown_a").outdoor
    for kind in DOWNTOWN_COLOURS:
        assert by_kind[kind]["outdoor"] == {"sun_elevation_deg": sky.sun_elevation_deg,
                                            "sun_rotation_deg": sky.sun_rotation_deg,
                                            "sun_strength": sky.sun_strength, "sky_strength": sky.sky_strength}, kind
    assert all(by_kind[k]["outdoor"] is None for k in ("mug", "pallet", "forklift", "chair"))
