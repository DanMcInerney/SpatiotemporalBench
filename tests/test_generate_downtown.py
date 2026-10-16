"""Generator contracts for the downtown (generator 0.7.0, plan-city §7.2): `build_world` on `downtown_a` makes valid
worlds of the downtown catalog with its squared-up vehicles, which the drone's hand-made script flies over. The
apartment's and the warehouse's contracts are in `test_generate.py` and `test_generate_warehouse.py`; the tiers, the
movers' overlay and the tour planner they once tested are gone (M2's lane P rebuilds the overlay, lane A the drone's
path)."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

import test_generate
from conftest import script_on_map
from stbench import timeline as T
from stbench.catalog import CATALOGS
from stbench.geometry import Rect, corners, to_local
from stbench.mapspec import load_map, surface_world_rect
from stbench.robots import ROBOTS, camera_doc
from stbench.validate import validate
from world_kit import after_gap, initial_carried, initial_world, manifest_model_of

DOWNTOWN = CATALOGS["downtown"]
REGIONS = ["high street", "west street", "east street", "square"]
HIGH_STREET = [(40.0, 13.5, 12.0, 0.0), (44.0, 13.5, 12.0, 0.0), (48.0, 13.5, 12.0, math.pi / 2)]  # 12 m up the roadway


@pytest.fixture(scope="module")
def city():
    return load_map("downtown_a")


def test_downtown_worlds_build_and_validate_with_a_drone_script(city):
    """Contract (plan-city §4, §7.2; seeds 1-30 are proven by the build's generate stage): `build_world` on `downtown_a`
    makes a world that passes `validate` for seed 1, before and after a gap's changes, with a drone's hand-made script
    on the high street (12 m up, over the parked vehicles), the map's name, the robot and the four regions.
    Regression caught: a city that the collision rules reject (a skip standing in a facade), a drone that the body rule
    refuses over the roadway (its 3 m clearance counting the parked cars it passes over), or a script that does not name
    its map and robot."""
    tl = script_on_map("downtown_a", 1, "drone", HIGH_STREET)
    assert validate(tl, city) == []
    assert validate(after_gap("downtown_a", 1), city) == []
    assert (tl.episode.map, tl.episode.robot) == ("downtown_a", "drone")
    assert [r.name for r in tl.regions] == REGIONS


def test_downtown_items_lie_wholly_on_a_surface_of_the_map_squared_up_at_every_time(city):
    """Contract (VISION rule 2, V0 §3: truth is where things are; `Kind.square`): every placement of a carried item of a
    downtown world, at the start and after a gap (moves and arrivals), has its whole footprint inside one surface of its
    support's map furniture (a parking bay, a loading zone, a sidewalk strip, a paving pad), at that surface's height,
    turned a quarter turn to the surface within 0.04 rad (a car parked along its bay). Checked against the map file, not
    the sampler. Hand case: a bay is 2.5 x 6.7 m, so a 5.4 x 2.0 m pickup turned 30 degrees (4.1 m across) would stick
    out of it, and a 4.6 m car turned across the bay would. A piece is found by where it stands and by having
    surfaces, since a drop zone (no surface) stands at the middle of its loading zone, with its yaw. Regression
    caught: a footprint test dropped for a long item (a van's rear in the next bay), a surface chosen at the wrong
    height (a car on the sidewalk strip), or a yaw drawn at any angle."""
    seen = 0
    for seed in (1, 2, 3):
        tl = after_gap("downtown_a", seed)
        for e in (e for e in tl.entities if e.mobility == "carried"):
            for p in tl.placements_of(e.id):
                support = tl.placement_of(p.on, p.t0)
                pieces = [f for f in city.furniture if f.mobility == "fixed" and f.surfaces  # a drop zone shares
                          and math.dist(f.center, support.pos[:2]) < 1e-6 and abs(f.yaw - support.yaw) < 1e-4]
                assert len(pieces) == 1, (seed, e.id, p.on)
                rects = [surface_world_rect(pieces[0].center, pieces[0].yaw, s) for s in pieces[0].surfaces
                         if abs(s.z - p.pos[2]) < 1e-6]
                assert rects, (seed, e.id, p.pos)
                body = corners(Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw))
                inside = [all(abs(x) <= r.sx / 2 + 1e-3 and abs(y) <= r.sy / 2 + 1e-3
                              for x, y in (to_local(r, cx, cy) for cx, cy in body)) for r in rects]
                assert any(inside), (seed, e.id, p.pos)
                turn = (p.yaw - rects[0].yaw) % (math.pi / 2)
                assert min(turn, math.pi / 2 - turn) <= 0.04 + 1e-4, (seed, e.id, p.yaw)
                seen += 1
    assert seen > 34  # more than the 34 starting places a world has at most, so some of what was checked came of a gap


def test_a_downtown_world_holds_the_downtown_catalog():
    """Contract (plan-city §3.1; `Catalog.n_items`): a downtown world starts with 24-34 of the catalog's eight kinds
    (a world may place a little fewer than it draws), each in a colour of its kind with its kind's box and manifest
    model, and holds no light furniture and no mover. Regression caught: the household's mugs or the warehouse's pallets
    on a street, a colour a kind is not drawn in, an entity without its asset, or light furniture in a map that has
    none."""
    for seed in (1, 2):
        tl = initial_world("downtown_a", seed)
        items = initial_carried(tl)
        assert 24 <= len(items) <= 34, seed
        for e in items:
            kind = DOWNTOWN.kinds[e.kind]
            assert set(e.attributes) == {"color"} and e.attributes["color"] in kind.colors, (seed, e)
            assert e.size == kind.size and e.asset == manifest_model_of(e.kind, e.attributes.get("color")), (seed, e)
        assert not [e for e in tl.entities if e.mobility in ("light", "mover")], seed


def test_a_downtown_world_is_the_same_bytes_in_another_process():
    """Contract (as `test_hash_is_stable_across_processes`, for the new catalog): nothing in the downtown's sampler or
    changes depends on Python's salted hash or on the iteration order of a set: a subprocess with another PYTHONHASHSEED
    makes the world of seed 1 after a gap that this process made. Regression caught: a set or hash-ordered draw in the
    city's path (the build's workers would disagree with the tests, and worlds would not reproduce)."""
    out = subprocess.run([sys.executable, test_generate.__file__, "after_gap", "downtown_a", "1"],
                         capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "777"}).stdout.strip()
    assert out == T.digest(after_gap("downtown_a", 1))


@pytest.mark.parametrize("map_name, robot, unit", [("apartment_a", "vacuum", "mm"), ("warehouse_a", "amr", "mm"),
                                                   ("downtown_a", "drone", "cm")])
def test_a_robots_depth_unit_is_its_maps(map_name, robot, unit):
    """Contract (plan-city §2; `check_map`'s range rule): the unit a robot's `camera.json` names is the unit its map
    names, and so the unit the map's 3D diagonal was checked to fit: the city's depth is in centimetres (the drone sees
    600 m, past what millimetres hold in 16 bits), the apartment's and the warehouse's in millimetres. Regression
    caught: a robot whose depth image is in another unit than its map's (the city's depth read as millimetres, ten
    times too near, or a range that no longer fits 16 bits)."""
    assert load_map(map_name).depth_unit == ROBOTS[robot].depth.unit == camera_doc(ROBOTS[robot])["depth_unit"] == unit
