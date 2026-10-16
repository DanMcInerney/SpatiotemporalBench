"""The validity guard of render-only dressing: the registry of every non-entity object class
(`scenery.DRESSING`), what a map's layout module (`render/dressing_layout_<catalog>.py`, `layout(spec) -> [Dressing]`)
may emit, and whether what it emits can touch the timeline. The registry and name checks are data; the layout checks
run the layouts, and are skipped for a catalog whose layout is not written yet."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from functools import cache

import pytest

from dressing_kit import RENDER, layout_of, layout_path, raw_map
from stbench.catalog import CATALOGS
from stbench.dressing_zones import DressingGuard
from stbench.mapspec import bundled_maps, load_map
from stbench.scenery import (
    DRESSING,
    SYNONYMS,
    DressingClass,
    class_problems,
    dressing_of,
    layout_problems,
    names_a_kind,
    tracked_colours,
)

# The plan's classes. Two names differ from the brief's: its power_box and
# utility_box end in "box", a tracked kind of the warehouse, so they are power_panel and utility_enclosure.
HOME_PLAN = [
    "baseboard", "door_casing", "window_sill", "outlet", "light_switch", "thermostat", "smoke_detector", "radiator",
    "curtain", "roller_blind", "wall_tv", "upper_cabinet", "range_hood", "backsplash", "bathroom_cabinet",
    "toilet_roll_holder", "shower_curtain", "picture_frame", "wall_clock", "ceiling_lamp", "ceiling_fan",
    "dust_line", "hair", "crumbs", "scuff",  # M1's R.2: the grime on the floor, flush
]
WAREHOUSE_PLAN = [
    "column", "roof_truss", "purlin", "roof_deck", "skylight", "sprinkler_main", "sprinkler_head", "cable_tray",
    "high_bay_light", "wall_cladding", "safety_band", "dock_door_seal", "dock_bumper", "dock_leveler", "dock_light",
    "door_number", "rack_upright_protector", "end_of_aisle_guard", "beam_label", "aisle_sign", "fire_extinguisher",
    "fire_alarm", "security_camera", "power_panel", "utility_enclosure", "air_duct", "cable_run"
]
BOOK_COLOURS = {"red", "blue", "green", "yellow", "black", "orange", "purple"}  # catalog.py's book, by hand
MAPS = tuple(bundled_maps())


def tracked_kinds(catalog: str) -> set[str]:
    """The kinds a class could be mistaken for in a catalog's maps: the carried and mover kinds, the light furniture the
    maps hold (the mission names it) and, where a map has doors, the door."""
    kinds = {*CATALOGS[catalog].kinds, *CATALOGS[catalog].mover_kinds}
    for name in MAPS:
        m = load_map(name)
        if m.catalog == catalog:
            kinds |= {f.kind for f in m.furniture if f.mobility == "light"} | ({"door"} if m.doors() else set())
    return kinds


@cache
def _layout(map_name: str) -> tuple[dict, str, list]:
    raw = raw_map(map_name)
    catalog = raw.get("catalog", "home")
    if not layout_path(catalog).exists():
        pytest.skip(f"render/dressing_layout_{catalog}.py is not written yet")
    return raw, catalog, layout_of(catalog, raw)


# -- the registry ----------------------------------------------------------------------------------------------------
def test_the_registry_holds_the_plans_classes_by_catalog_and_phase():
    """Contract: every non-entity object class a render draws is registered once, for
    the catalogs that draw it. The household's phase 1 classes are the plan's 21 and the grime's four, and the
    warehouse's its 27 (power_box
    and utility_box under names that avoid the tracked kind "box"); the warehouse's rack stock is phase 0 (the furniture
    builders draw it) and the bookshelf's fixed books phase 2 (a map edit). Regression caught: a class dropped from the
    registry (its layout silently unchecked) or one added to a catalog it is not drawn in."""
    assert [c.name for c in dressing_of("home", 1)] == HOME_PLAN
    assert [c.name for c in dressing_of("warehouse", 1)] == WAREHOUSE_PLAN
    assert [c.name for c in dressing_of("warehouse", 0)] == ["rack_stock"]
    assert [c.name for c in dressing_of("home", 2)] == ["shelved_book"]


@pytest.mark.parametrize("name", list(DRESSING))
def test_no_class_is_called_what_a_tracked_kind_is_called_unless_it_declares_the_resemblance(name):
    """Contract: the robot is asked about carried things, light furniture, movers (and,
    in the apartment, doors); a class named for one of them, or for a synonym (crate, barrel, trash can, cushion,
    suitcase, potted plant...), looks like one in the pictures, so it must list that kind in `resembles` (with colours
    no tracked thing of the kind has) or not exist. Regression caught: a class added under a tracked kind's name
    (a "crate" or "drum" among the warehouse's fittings) with nothing declared."""
    c = DRESSING[name]
    for catalog in c.catalogs:
        hits = names_a_kind(c.name, tracked_kinds(catalog))
        assert set(hits) <= {kind for kind, _ in c.resembles}, f"{c.name} is called {hits} in the {catalog} catalog"


@pytest.mark.parametrize(("name", "kinds"), [
    ("crate", {"box", "tote"}), ("carton", {"box"}), ("cardboard_box", {"box"}), ("power_box", {"box"}),
    ("barrel", {"drum"}), ("trash_can", {"bin", "laundry_basket"}), ("waste_basket", {"bin", "laundry_basket"}),
    ("suitcase", {"bag"}), ("backpack", {"bag"}), ("potted_plant", {"plant"}), ("houseplant", {"plant"}),
    ("cushion", {"towel"}), ("throw_pillow", {"towel"}), ("vase", {"bottle"}), ("binder", {"book"}),
    ("shelved_book", {"book"}), ("coffee_cup", {"mug"}), ("boxes", {"box"}), ("shoes", {"shoes"}),
    ("pallet_truck", {"pallet_jack"}), ("shower_curtain", set()), ("door_casing", set()), ("dock_door_seal", set()),
    ("toilet_roll_holder", set()), ("power_panel", set()), ("wall_clock", set()), ("cable_tray", set()),
])
def test_a_name_is_a_kind_when_it_ends_in_the_kinds_name_or_a_synonym(name, kinds):
    """Contract: `names_a_kind` reads a class name's last word as its head noun and compares it, singular, with every
    kind and its synonyms (crate, barrel, trash can, cushion, suitcase, potted plant, vase,
    binder). A modifier does not count (a door casing is no door, a toilet roll holder no roll cage, a power panel no
    box). A cushion or a pillow is a towel on the pictures: both are soft saturated rectangles, and a towel may lie on a
    sofa or a bed. Regression caught: a synonym dropped (a crate in the racks), plurals missed (shoes, boxes), or a
    modifier mistaken for the thing (every door casing rejected)."""
    everything = set().union(*(tracked_kinds(c) for c in CATALOGS))
    assert set(names_a_kind(name, everything)) == kinds


def test_every_synonym_belongs_to_a_kind_that_exists():
    """Contract: a synonym entry guards a kind of some catalog, a light piece of some map or the door. Regression
    caught: a typo in a kind's name (the entry silently guards nothing) or a kind renamed in the catalog without its
    synonyms."""
    everything = set().union(*(tracked_kinds(c) for c in CATALOGS))
    assert set(SYNONYMS) <= everything, set(SYNONYMS) - everything


def test_every_resemblance_is_a_colour_no_tracked_thing_has():
    """Contract: a class that resembles a tracked kind lists the colours the pictures show
    it in, none of which a tracked thing of that kind has (so the robot is never told to ignore a real one:
    `lookalike_problems`). The bookshelf's fixed books (phase 2, not rendered yet) are white, grey, beige and brown,
    none of which a tracked book is (red, blue, green, yellow, black, orange, purple, read from the catalog by hand).
    Regression caught: a tracked colour among a class's (a real mug declared scenery), or a book colour the catalog
    later gives books."""
    assert [p for c in DRESSING.values() for p in class_problems(c)] == []
    assert set(dict(DRESSING["shelved_book"].resembles)["book"]).isdisjoint(BOOK_COLOURS)
    assert tracked_colours("home", "book") == BOOK_COLOURS


def test_the_registry_checks_report_what_they_are_for():
    """Contract: `class_problems` rejects a resemblance to a colour a tracked thing has, to a kind the catalog lacks and
    with no colour; `layout_problems` names an unregistered class, one of another catalog, one in the wrong phase, one a
    phase 1 build never draws and every phase 1 class left unused. By hand on made-up classes. Regression caught: any
    of the checks going quiet, which would let a real class through unchecked."""
    bad = DressingClass("bad", ("home",), (("book", ("red", "grey")), ("pallet", ("grey",)), ("mug", ())), 1, "made up")
    assert class_problems(bad) == ["bad: (book, red) is a tracked thing of the home catalog, not scenery",
                                   "bad: 'pallet' is not a kind of the home catalog",
                                   "bad: 'mug' has no look-alike colour"]
    emitted = [("crate", 1), ("power_panel", 1), ("baseboard", 2), ("shelved_book", 2)]
    problems = layout_problems("home", emitted)
    assert "crate is not a registered dressing class of the home catalog" in problems
    assert "power_panel is not a registered dressing class of the home catalog" in problems
    assert "baseboard is emitted in phase 2 but registered in phase 1" in problems
    assert "shelved_book is phase 2: a phase 1 build never draws it" in problems
    never = [p for p in problems if p.endswith("that the layout never emits")]
    assert len(never) == len(HOME_PLAN) - 1  # baseboard was emitted, in the wrong phase: that is its own problem
    assert layout_problems("home", [(n, 1) for n in HOME_PLAN]) == []
    assert layout_problems("warehouse", [(n, 1) for n in WAREHOUSE_PLAN]) == []


# -- what the layouts emit ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("map_name", MAPS)
def test_every_class_a_layout_emits_is_registered_for_its_catalog_and_every_phase_1_class_is_drawn(map_name):
    """Contract: a map's layout emits only classes registered for the map's catalog, in the
    phase the registry gives them and no later than 1 (phase 2 edits the world), and uses every phase 1 class the
    registry lists for the catalog, so no name is registered that no picture shows. Regression caught: a class invented
    in a layout (unreviewed for look-alikes), a registered class no layout draws, or a phase 2 object drawn by a render
    that must leave the worlds alone."""
    _, catalog, layout = _layout(map_name)
    assert layout, f"the {catalog} layout emits nothing"
    assert layout_problems(catalog, [(d.cls, d.phase) for d in layout]) == []


@pytest.mark.parametrize("map_name", MAPS)
def test_no_phase_1_dressing_box_meets_a_forbidden_volume_or_leaves_the_allowed_places(map_name):
    """Contract: every box a layout emits lies wholly inside the wall bands (at most
    0.12 m proud), the ceiling slab, the structure above 5 m, the columns over blocking furniture or a flush floor
    plate, and meets no receptacle volume, no open floor the camera, light furniture or movers use, no door swing or
    opening and no window glass. Together these prove the timelines, routes, placements and golds are untouched:
    dressing can only ever occlude. Regression caught: trim across a doorway, a sill in the counter's items, a
    cabinet in the camera's path, a fitting in a window, anything that stands where a tracked thing is put or the
    robot goes."""
    raw, catalog, layout = _layout(map_name)
    guard = DressingGuard(raw, catalog)
    bad = [f"{d.cls} at {tuple(round(v, 2) for v in d.pos)} yaw {d.yaw_deg:g}: {guard.problems(d.box, d.cls)}"
           for d in layout if guard.problems(d.box, d.cls)]
    assert bad == []


@pytest.mark.parametrize("map_name", MAPS)
def test_a_layout_depends_on_the_map_alone(map_name, tmp_path):
    """Contract (VISION rule 13): a map's dressing is the same for every episode, so its layout
    is a function of the map spec alone: the same list every call, whatever the global random state, the order of the
    spec's keys or the interpreter's hash seed (a set iterated in another order), and the spec is left as it was.
    Regression caught: dressing seeded from the clock or an episode, an unseeded `random`, a set whose order is the
    hash seed's, or a layout that edits the spec it was given (the next map's dressing changes)."""
    raw, catalog, first = _layout(map_name)
    before = copy.deepcopy(raw)
    random.seed(1)
    again = layout_of(catalog, raw)
    random.seed(2)
    reordered = layout_of(catalog, dict(reversed(list(copy.deepcopy(raw).items()))))
    assert raw == before and first == again == reordered
    spec_file = tmp_path / "map.json"
    spec_file.write_text(json.dumps(raw), encoding="utf-8")
    code = ("import hashlib, importlib, json, sys\nsys.path.insert(0, sys.argv[1])\n"
            "layout = importlib.import_module(sys.argv[3]).layout(json.load(open(sys.argv[2], encoding='utf-8')))\n"
            "print(hashlib.sha256(repr(layout).encode()).hexdigest())\n")
    digests = {seed: subprocess.run([sys.executable, "-c", code, str(RENDER), str(spec_file),
                                     f"dressing_layout_{catalog}"], env={**os.environ, "PYTHONHASHSEED": seed},
                                    capture_output=True, text=True, check=True).stdout.strip()
               for seed in ("0", "12345")}
    assert len(set(digests.values())) == 1, digests
    assert digests["0"] == hashlib.sha256(repr(first).encode()).hexdigest()
