"""`house_floor`, the ground floor of a large detached two-storey house (M4's public home layout, `vacuum/2` on it): the
map file, its maker (`tools/maps/house_floor.py`), the plan the job runs on and the room graph `route_back` will
need.
Model-free; the staircase's drawing is `test_house_floor_blender.py`. The generated script is one task (about three
seconds): a four-task script takes about twenty."""

from __future__ import annotations

import importlib.util
import json
import math
from functools import cache
from pathlib import Path

import numpy as np

from dressing_kit import render_module
from stbench import coverage, episode, generate, mapspec, route, validate
from stbench.geometry import dist_to_rect, rects_overlap
from stbench.job_vacuum import UNDOCK_M, docked_pose
from stbench.leak import find_leaks
from stbench.robots import ROBOTS
from test_mapspec import apron_problems

REPO = Path(__file__).resolve().parents[1]
MAKER = REPO / "tools" / "maps" / "house_floor.py"
FILE = REPO / "src" / "stbench" / "maps" / "house_floor.json"
HOME_KINDS = {"hall", "living room", "kitchen", "bedroom", "bathroom", "dining room", "utility room", "study",
              "corridor", "cloakroom"}
# The eleven openings the plan gives (hall-living room, hall-study, an arch hall-corridor, corridor-kitchen, -cloakroom,
# -utility room and -bedroom, bedroom-bathroom, kitchen-dining room, kitchen-utility room, an arch living room-dining
# room), written out by hand.
JOINS = {frozenset(p) for p in (("hall", "living room"), ("hall", "study"), ("hall", "corridor"),
                                ("corridor", "kitchen"), ("corridor", "cloakroom"), ("corridor", "utility room"),
                                ("corridor", "bedroom"), ("bedroom", "bathroom"), ("kitchen", "dining room"),
                                ("kitchen", "utility room"), ("living room", "dining room"))}


@cache
def house() -> mapspec.MapSpec:
    return mapspec.load_map("house_floor")


def maker():
    spec = importlib.util.spec_from_file_location("house_floor_maker", MAKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_house_floor_is_a_valid_rectilinear_home():
    """Contract (M4 house_floor §1; `dressing_boxes.py:63-83`, `dressing_home_plan.py:303-325`): `check_map` reports
    nothing; every region is a 4-vertex axis-aligned rectangle at least 1.0 m between opposite faces, every piece is at
    a quarter turn, the walls are axis-aligned, and the regions are the home catalog's ten kinds, each named by its kind
    (the file has no `kind` key: the household's clutter matches room names), each once, each with a ceiling light.
    Regression caught: a map that `clutter.place` or the dressing refuses only at generation or render time (a turned
    piece, a room under a metre across, which has no spot for its smoke detector), a kind missing from the home
    vocabulary (a room dressed by a fallback) or two rooms of one kind (`vacuum/2` names a room by a unique kind)."""
    m, raw = house(), json.loads(FILE.read_text(encoding="utf-8"))
    assert mapspec.check_map(m) == []
    assert m.catalog == "home" and "catalog" not in raw and all("kind" not in r for r in raw["regions"])
    names = [r.name for r in m.regions]
    assert sorted(names) == sorted(HOME_KINDS) and [r.kind for r in m.regions] == names
    assert set(names) == render_module("look").kinds("home")
    for r in m.regions:
        xs, ys = sorted({p[0] for p in r.polygon}), sorted({p[1] for p in r.polygon})
        assert len(r.polygon) == 4 and len(xs) == 2 and len(ys) == 2, r.name
        assert min(xs[1] - xs[0], ys[1] - ys[0]) >= 1.0, r.name
        assert any(light.region == r.id for light in m.ceiling_lights), r.name
    assert all(abs(math.sin(2 * f.yaw)) < 1e-9 for f in m.furniture)
    assert all(w.a[0] == w.b[0] or w.a[1] == w.b[1] for w in m.walls)


def test_the_vacuum_has_its_dock_and_the_stairs_bound_the_floor():
    """Contract (M4 house_floor §1-§2; docs/jobs.md, `vacuum/2`): exactly one `dock`, in the region of kind `hall`, with
    its apron clear (`apron_problems`: no blocking piece, no door swing or band); exactly one `stairs`, a blocking fixed
    piece solid to the floor with no surface; every room has floor the vacuum can reach from the dock's undocking point
    (`robot_free_space`), and none of it lies inside the staircase's footprint. Regression caught: a dock the job cannot
    undock from (a console or a door in its apron), a staircase the vacuum would drive into, or a room walled off by a
    piece or a door leaf."""
    m = house()
    (dock,) = [f for f in m.furniture if f.kind == "dock"]
    assert m.region(m.region_of(dock.center)).kind == "hall" and apron_problems(m) == []
    (stairs,) = [f for f in m.furniture if f.kind == "stairs"]
    assert (stairs.mobility, stairs.blocks, stairs.under_m, stairs.surfaces) == ("fixed", True, 0.0, ())
    assert m.region(m.region_of(stairs.center)).kind == "hall"
    x, y, yaw = docked_pose(m)
    out = (x + UNDOCK_M * math.cos(yaw), y + UNDOCK_M * math.sin(yaw))
    free = route.robot_free_space(m, generate.build_world(m, 101), ROBOTS["vacuum"])
    for r in m.regions:
        assert coverage.reachable_floor(free, r, out).any(), f"the vacuum cannot reach the {r.name}"
    gx, gy = np.meshgrid(free.x0 + (np.arange(free.nx) + 0.5) * free.res,
                         free.y0 + (np.arange(free.ny) + 0.5) * free.res)
    assert not (free.free & (dist_to_rect(gx, gy, stairs.footprint()) <= 0.0)).any()


def test_every_door_swings_into_a_room_and_opens_clear_of_the_furniture():
    """Contract (M4 house_floor §1: doors swing into a region): every interior door's leaf, open, lies in a room that is
    not the corridor (an open leaf across the 1.3 m corridor would leave 0.45 m) and meets no blocking piece (to 5 cm),
    and the front door swings into the hall. Regression caught: a door that cannot open for a console, or a leaf that
    shuts the corridor when a door is open."""
    m = house()
    corridor = next(r.id for r in m.regions if r.kind == "corridor")
    for o in m.doors():
        assert o.swings_into is not None and (not o.interior or o.swings_into != corridor), o.id
        leaf = validate.door_leaf_rect(m, o, True, 0.04)
        assert not [f.id for f in m.furniture if f.blocks and rects_overlap(leaf, f.footprint(), 0.05)], o.id
    assert next(o for o in m.doors() if not o.interior).swings_into == m.region_of(docked_pose(m)[:2])


def test_a_vacuum2_script_generates_and_validates(tmp_path):
    """Contract (M4 house_floor §3; `job_vacuum.py:69-73`): a `vacuum/2` script of the map, written by
    `episode.write_script`, passes what `stbench validate` runs (the timeline's rules and the job's, the
    leak scan of the agent-facing files), and its tasks start in no hall (the dock's room) and name rooms the map has
    once. Regression caught: a job that cannot run here (a room kind shared by two rooms, a room no path reaches) or a
    script that leaks truth."""
    tl = generate.generate_script("house_floor", 101, "vacuum", "vacuum/2", 1)
    m = house()
    assert validate.validate(tl, m) == []
    paths = episode.write_script(tmp_path, tl)
    assert find_leaks(paths.root, ["house_floor"]) == []
    kinds = [str(k) for t in tl.ego.tasks for k in t.plan.params["kinds"]]
    assert kinds and kinds[0] != "hall" and all(sum(r.kind == k for r in m.regions) == 1 for k in kinds)


def test_the_bundled_map_is_its_makers():
    """Contract (`tools/maps/house_floor.py`: a public map's maker is committed): `src/stbench/maps/house_floor.json`
    parses to exactly the maker's `spec()` and is the text the maker writes. Regression caught: a hand edit of the
    JSON that the maker would undo (the next run of the maker silently changes the map's hash, which every script
    pins)."""
    module = maker()
    text = FILE.read_text(encoding="utf-8")
    assert json.loads(text) == module.spec()
    assert text == module.render(module.spec())


def test_every_door_and_arch_joins_two_rooms():
    """Contract (M4-integration call 18): the interior openings are the eleven the plan gives, each joining two
    different rooms; the room graph they make is connected and has two independent cycles (edges - nodes + 1), so
    `route_back` can come later with no map edit; the front door joins the hall to the outside. Regression caught: a
    layout whose rooms are a tree (no way back that is not the way there) or one with a room no opening reaches."""
    m = house()
    name = {r.id: r.name for r in m.regions}
    interior = [o for o in m.openings if o.interior]
    assert {frozenset(name[i] for i in o.between) for o in interior} == JOINS and len(interior) == len(JOINS) == 11
    seen, todo = set(), ["hall"]
    while todo:
        room = todo.pop()
        seen.add(room)
        todo += [n for pair in JOINS if room in pair for n in pair if n not in seen]
    assert seen == HOME_KINDS
    assert len(JOINS) - len(HOME_KINDS) + 1 == 2
    (front,) = [o for o in m.openings if not o.interior]
    assert sorted(front.between) == sorted(["outside", next(r.id for r in m.regions if r.kind == "hall")])
