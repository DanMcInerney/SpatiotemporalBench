"""`suburb`, a street of detached houses (M4's public residential layout, `drone/1` on it): the map file, its maker
(`tools/maps/suburb.py`), the contract the drone's job and the dressing read from it, its pedestrians and its
processes. Model-free; the scene is `test_suburb_blender.py`. The generated script is one flight with one pedestrian
(a walk on the 0.6 million cells of the people's grid costs about four seconds): a script of five flights with its
pedestrians takes about a minute.

Hand-worked numbers, from the maker's plan (x east, y north, metres): the street is y 35 to 47 across x 0 to 300 (verge
1.2, sidewalk 1.8, carriageway 6.0, sidewalk 1.8, verge 1.2); 18 plots of 15 x 35 m on each side (the south plots even,
the north plots odd, 36 in all); the south side holds the passage (x 135-138) and the green (138-165) between its ninth
and tenth plots, the north side the car park (135-165) with the nest at (150, 64.5); a third of the plots, 12, carry
a drop zone at the street end of their driveway (3 m square, its centre 1.9 m from the plot's street boundary)."""

from __future__ import annotations

import importlib.util
import json
import math
from functools import cache
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from dressing_kit import layout_of, raw_map, render_module
from residential_kit import tall_near
from stbench import episode, generate, mapspec, people_residential, validate, world_processes
from stbench import world_processes_downtown as downtown
from stbench.catalog import CATALOGS
from stbench.geometry import Rect, dist_to_rect, point_in_rect, rect_inside_rect, rects_overlap
from stbench.leak import find_leaks
from stbench.zones import ground_z

pytestmark = pytest.mark.xdist_group(name=__name__)  # the one-flight script is built once, by one worker
REPO = Path(__file__).resolve().parents[1]
MAKER = REPO / "tools" / "maps" / "suburb.py"
FILE = REPO / "src" / "stbench" / "maps" / "suburb.json"
DRONE_REACH_M = 3.65  # the drone's 0.65 m radius and its 3.0 m clearance: nothing tall within it of a lowering point
PLOTS, ZONES, STREET_M = 36, 12, 300.0


@cache
def suburb() -> mapspec.MapSpec:
    return mapspec.load_map("suburb")


@cache
def flight():
    """One flight on the suburb with one pedestrian present for it (`COUNT` 1, `P_HOME` 1: the draw is the seed's, the
    count is not what a test should depend on, and a walk costs seconds)."""
    with mock.patch.object(people_residential, "COUNT", 1), mock.patch.object(people_residential, "P_HOME", 1.0):
        return generate.generate_script("suburb", 101, "drone", "drone/1", 1)


def maker():
    spec = importlib.util.spec_from_file_location("suburb_maker", MAKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def of_kind(m: mapspec.MapSpec, kind: str) -> list[mapspec.Furniture]:
    return [f for f in m.furniture if f.kind == kind]


def gap(f: mapspec.Furniture, x: float, y: float) -> float:
    """The distance from a point to a piece's footprint (0 on it)."""
    return float(dist_to_rect(np.array([x]), np.array([y]), f.footprint())[0])


def test_suburb_is_a_valid_rectilinear_place():
    """Contract (M4 suburb §5; `dressing_boxes.py:63-83`, `job_drone_plan.nest_of`, M3-Y §1): `check_map` reports
    nothing; the map is residential, outdoor, in centimetre depth, with no wall, opening, window or light; every region
    is an axis-aligned rectangle at least 1 m across of a kind the residential look has, every piece is turned a
    quarter; the street is a region of kind `street` 300 m long, the plots are 36 `garden` regions named by address
    (south even, north odd, 18 a side) and the passage, the green and the car park are one each; the walls' height
    clears the dressing's 8.2 m. The one `nest` is a pad under 0.05 m on bare ground in the car park, with no piece
    within 3.65 m of its centre; the 12 drop zones are markings with no surface, 0.005 m or less over their ground, in
    the plot of their name, each 3.65 m from every house and from the nest (the lowering rule: a house stands taller
    than the altitude less 3 m) with no blocking piece on it (nothing in its band, as a truck's roof was in the
    downtown's street zones). Regression caught: a map the drone's job refuses, a nest the drone takes off beside a
    kerb, a house a lowering would hit, or a plot named like another."""
    m, raw = suburb(), json.loads(FILE.read_text(encoding="utf-8"))
    assert mapspec.check_map(m) == []
    assert (m.catalog, m.depth_unit, m.bounds) == ("residential", "cm", (0.0, 0.0, 300.0, 82.0))
    assert m.outdoor is not None and m.walls == () and m.openings == () and m.windows == () and m.ceiling_lights == ()
    assert m.wall_height >= 8.2
    kinds = [r.kind for r in m.regions]
    assert set(kinds) <= render_module("look").kinds("residential")
    for r in m.regions:
        xs, ys = sorted({p[0] for p in r.polygon}), sorted({p[1] for p in r.polygon})
        assert len(r.polygon) == 4 and len(xs) == 2 and len(ys) == 2 and min(xs[1] - xs[0], ys[1] - ys[0]) >= 1.0, r.id
    assert all(abs(math.sin(2 * f.yaw)) < 1e-9 for f in m.furniture)
    (street,) = [r for r in m.regions if r.kind == "street"]
    assert max(p[0] for p in street.polygon) - min(p[0] for p in street.polygon) >= STREET_M
    gardens = [r for r in m.regions if r.kind == "garden"]
    numbers = {r.id: int(r.name.removesuffix(" Elm Street")) for r in gardens}
    assert len(gardens) == PLOTS and sorted(numbers.values()) == list(range(1, PLOTS + 1))
    assert all((numbers[r.id] % 2 == 0) == (r.anchor[1] < street.anchor[1]) for r in gardens)  # south even, north odd
    assert {r.kind: kinds.count(r.kind) for r in m.regions} == {"street": 1, "garden": PLOTS, "passage": 1, "green": 1,
                                                                "car park": 1}
    assert all("kind" in r for r in raw["regions"])
    (nest,) = of_kind(m, "nest")
    assert nest.mobility == "fixed" and not nest.blocks and nest.size[2] < 0.05 and not nest.surfaces
    assert m.region(m.region_of(nest.center)).kind == "car park" and ground_z(m, nest) == 0.0
    assert [f.id for f in m.furniture if f.id != nest.id and gap(f, *nest.center) < DRONE_REACH_M] == []
    houses, zones = of_kind(m, "house"), of_kind(m, "drop_zone")
    assert len(houses) == PLOTS and len(zones) == ZONES and len({z.name for z in zones}) == ZONES
    for z in zones:
        assert not z.blocks and not z.surfaces and 0 < z.size[2] - ground_z(m, z) <= 0.005 + 1e-9, z.id
        assert m.region(m.region_of(z.center)).name == z.name, z.id  # the zone's address is its plot's
        assert [h.name for h in houses if gap(h, *z.center) < DRONE_REACH_M] == [], z.id
        assert math.dist(z.center, nest.center) >= DRONE_REACH_M
        # nothing that blocks stands on the zone: no roof, wall, hedge or bed in its band (the zone rule's 3.5 m)
        assert [f.id for f in m.furniture if f.blocks and rects_overlap(f.footprint(), z.footprint())] == []
    assert {m.region(m.region_of(h.center)).name for h in houses} == {r.name for r in gardens}


def test_every_drop_zone_lies_on_a_driveway():
    """Contract (M4 suburb §5; M3-Y §4): each `drop_zone`'s footprint lies inside the `parking` surface of a `driveway`
    (a surface the zone drop moves a vehicle onto, a parked one blocks it), the driveway is 0.05 m high (the surface as
    high as the slab), at most one zone to a driveway, and a third of the plots carry one (12 of 36). Regression caught:
    a zone on a lawn, which no item's receptacle is and nobody walks on, so its families would have no balance."""
    m = suburb()
    drives = [(f, mapspec.surface_world_rect(f.center, f.yaw, s)) for f in of_kind(m, "driveway")
              for s in f.surfaces if s.kind == "parking"]
    assert len(drives) == PLOTS and all(f.size[2] == 0.05 and not f.blocks for f, _ in drives)
    used = []
    for z in of_kind(m, "drop_zone"):
        (hit,) = [f.id for f, rect in drives if rect_inside_rect(z.footprint(), rect, 1e-6)]
        used.append(hit)
    assert len(set(used)) == len(used) == ZONES


def test_nothing_tall_stands_by_a_drop_zone_or_the_nest():
    """Contract (M4 suburb §4; the drone's lowering rule, `residential_kit.tall_near`): in the suburb's dressing no tree
    stem, canopy or lamp post taller than 3.9 m comes within 3.65 m of a drop zone's or the nest's centre, and no house
    does. Regression caught: a street tree over a driveway's end that the drone lowers beside, which `validate`'s body
    rule would refuse at the keyframe."""
    raw = raw_map("suburb")
    assert tall_near(raw, layout_of("residential", raw)) == []


def test_the_bundled_map_is_its_makers():
    """Contract (`tools/maps/suburb.py`: a public map's maker is committed): `src/stbench/maps/suburb.json` parses to
    exactly the maker's `spec()` and is the text the maker writes. Regression caught: a hand edit of the JSON that the
    maker would undo (the next run silently changes the map every script names)."""
    module = maker()
    text = FILE.read_text(encoding="utf-8")
    assert json.loads(text) == module.spec()
    assert text == module.dump(module.spec())


def test_a_drone_script_generates_and_validates(tmp_path):
    """Contract (M4 suburb §6; `job_drone.py`): a `drone/1` flight on the suburb, written by `episode.write_script`,
    passes what `stbench validate` runs (the timeline's rules and the job's, the leak scan of the
    agent-facing files); it flies to a drop zone of the map by address, its world holds 18 to 24 vehicles and containers
    (the catalog's `n_items`) on parking and sidewalk surfaces, and its pedestrian walks with its whole body on a
    sidewalk, 0.15 m up, outside every drop zone. Regression caught: a job that cannot run here (no nest, no zone the
    world can set, a way no flier can fly), a script that leaks truth, or a pedestrian on a lawn or in the road."""
    tl, m = flight(), suburb()
    assert validate.validate(tl, m) == []
    paths = episode.write_script(tmp_path, tl)
    assert find_leaks(paths.root, ["suburb"]) == []
    (task,) = tl.ego.tasks
    assert task.plan.params["address"] in {z.name for z in of_kind(m, "drop_zone")}
    items = [e for e in tl.entities if e.mobility == "carried"]
    assert CATALOGS["residential"].n_items[0] <= len(items) <= CATALOGS["residential"].n_items[1]
    assert {e.kind for e in items} <= set(CATALOGS["residential"].kinds)
    (person,) = [e for e in tl.entities if e.mobility == "mover"]
    walks = [f.footprint() for f in of_kind(m, "sidewalk")]
    zones = [f.footprint() for f in of_kind(m, "drop_zone")]
    placed = tl.placements_of(person.id)
    assert placed
    for p in placed:
        body = Rect(p.pos[0], p.pos[1], person.size[0], person.size[1], p.yaw)
        assert abs(p.pos[2] - 0.15) < 1e-6 and any(rect_inside_rect(body, w, 1e-6) for w in walks)
        assert not any(point_in_rect(z, p.pos[0], p.pos[1]) for z in zones)


def test_the_suburb_runs_the_downtowns_processes_between_flights():
    """Contract (`world_processes.gaps`, `world_processes_residential`; M4 suburb §5): a map of the residential catalog
    loads its processes module, whose rates are the downtown's (every kind of the catalog has a half-life there): the
    callback writes nothing at the script's start and, over a gap of hours, moves things at one instant at least 30 s
    from both ends of it and adds or removes none. Regression caught: a catalog whose module is never found (the
    processes silently off, the world the same before every flight), or a catalog kind no half-life covers (a KeyError
    in the middle of a gap)."""
    m = suburb()
    assert set(CATALOGS["residential"].kinds) <= set(downtown.HALF_LIFE_S)
    w = generate.build_world(m, 101)
    items, before = set(w.carried), w.ledger.finish()
    between = world_processes.gaps(w, m, 101)
    between(None, 0.0, frozenset())
    assert w.ledger.finish() == before
    between(100.0, 5 * 3600.0, frozenset())
    changed = {p.t0 for p in w.ledger.finish()[0] if p.t0 > 0.0}
    assert len(changed) == 1 and 130.0 <= changed.pop() <= 5 * 3600.0 - 30.0
    assert set(w.carried) == items
